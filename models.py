"""Model loading and inference: CLIP embeddings + Ollama VLM captions."""
import base64
import io
import numpy as np
import requests
from PIL import Image
from functools import lru_cache
from contextlib import nullcontext

import config


def _get_device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _autocast():
    """Return a torch.autocast context on CUDA, or a no-op context on CPU.

    Prevents ``RuntimeError: autocast can't be used with 'cuda' on CPU-only PyTorch``.
    """
    import torch
    if _get_device() == "cuda":
        return torch.autocast("cuda")
    return nullcontext()


# ---- CLIP Model (for fast vector search) ----

_clip_model = None
_clip_processor = None


def _load_clip():
    global _clip_model, _clip_processor
    if _clip_model is None:
        from transformers import CLIPModel, CLIPProcessor
        device = _get_device()
        print(f"[models] Loading CLIP model to {device}: {config.CLIP_MODEL_NAME}")
        _clip_processor = CLIPProcessor.from_pretrained(config.CLIP_MODEL_NAME)
        _clip_model = CLIPModel.from_pretrained(config.CLIP_MODEL_NAME).to(device)
        if device == "cuda":
            _clip_model = _clip_model.half()
        _clip_model.eval()
        print("[models] CLIP model loaded (FP16).")
    return _clip_model, _clip_processor


def encode_images_clip(images: list) -> np.ndarray:
    """Encode a batch of PIL images into normalized CLIP embeddings."""
    import torch
    model, processor = _load_clip()
    all_embeddings = []
    for i in range(0, len(images), config.CLIP_BATCH_SIZE):
        batch = images[i:i + config.CLIP_BATCH_SIZE]
        inputs = processor(images=batch, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            with _autocast():
                output = model.get_image_features(**inputs)
        # transformers 5.x returns BaseModelOutputWithPooling; older returns tensor
        embeds = output.pooler_output if hasattr(output, 'pooler_output') else output
        embeds = embeds / embeds.norm(dim=-1, keepdim=True)
        all_embeddings.append(embeds.cpu().numpy())
    return np.vstack(all_embeddings).astype("float32")


def encode_text_clip(text: str) -> np.ndarray:
    """Encode a text query into a normalized CLIP embedding."""
    import torch
    model, processor = _load_clip()
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        with _autocast():
            output = model.get_text_features(**inputs)
    # transformers 5.x returns BaseModelOutputWithPooling; older returns tensor
    embeds = output.pooler_output if hasattr(output, 'pooler_output') else output
    embeds = embeds / embeds.norm(dim=-1, keepdim=True)
    return embeds.cpu().numpy().astype("float32")


# ---- Ollama VLM (for rich text captions) ----

def _image_to_base64(image: Image.Image) -> str:
    """Convert PIL image to base64 string."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def caption_image_vlm(image: Image.Image, prompt: str = None) -> str:
    """Generate a detailed caption for an image using Ollama VLM."""
    if prompt is None:
        prompt = (
            "Describe this person/object in detail. Include: gender, approximate age, "
            "clothing colors and types, accessories (bags, hats, glasses), hair style/color, "
            "body posture, and any actions being performed. Be concise but thorough. "
            "Do NOT use markdown formatting. Just plain text description."
        )
    
    img_b64 = _image_to_base64(image)
    
    try:
        resp = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/chat",
            json={
                "model": config.OLLAMA_VLM_MODEL,
                "messages": [{
                    "role": "user",
                    "content": prompt,
                    "images": [img_b64],
                }],
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 150,
                }
            },
            timeout=60
        )
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "").strip()
    except Exception as e:
        print(f"[models] VLM caption error: {e}")
        return ""


def caption_scene_vlm(image: Image.Image) -> str:
    """Generate a comprehensive caption for a full frame — the PRIMARY search data."""
    prompt = (
        "Describe everything visible in this image in detail. Include: "
        "all people (clothing colors, accessories, actions, gender, age), "
        "all animals (type, color, what they are doing), "
        "all objects, vehicles, furniture, and items visible, "
        "the environment/setting, lighting, and any text or signs. "
        "Be thorough but use plain text only, no markdown."
    )
    return caption_image_vlm(image, prompt)


# ---- YOLO Detection ----

_yolo_model = None


def _load_yolo():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        print(f"[models] Loading YOLO model: {config.YOLO_MODEL_PATH}")
        _yolo_model = YOLO(config.YOLO_MODEL_PATH)
        print("[models] YOLO model loaded.")
    return _yolo_model


def detect_objects(frame_bgr: np.ndarray) -> list:
    """
    Run YOLO detection on a frame.
    Returns list of dicts: {class_name, confidence, bbox: (x1,y1,x2,y2)}
    """
    model = _load_yolo()
    results = model(
        frame_bgr, 
        conf=config.YOLO_CONFIDENCE, 
        classes=config.YOLO_CLASSES,
        half=True,
        verbose=False
    )
    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            detections.append({
                "class_name": model.names[cls_id],
                "confidence": float(box.conf[0]),
                "bbox": tuple(box.xyxy[0].cpu().numpy().astype(int).tolist()),
            })
    return detections


def detect_objects_batch(frames_bgr: list) -> list:
    """
    Run YOLO detection on a batch of frames.
    Returns list of lists of detections.
    """
    model = _load_yolo()
    if not frames_bgr: return []
    # stream=True is more memory efficient for large batches
    results = model(frames_bgr, conf=config.YOLO_CONFIDENCE, classes=config.YOLO_CLASSES, half=True, verbose=False, stream=True)
    batch_dets = []
    for r in results:
        dets = []
        for box in r.boxes:
            cls_id = int(box.cls[0])
            dets.append({
                "class_name": model.names[cls_id],
                "confidence": float(box.conf[0]),
                "bbox": tuple(box.xyxy[0].cpu().numpy().astype(int).tolist()),
            })
        batch_dets.append(dets)
    return batch_dets


# ---- YOLO-Face Detection (for face crops) ----

_yolo_face_model = None


def _load_yolo_face():
    global _yolo_face_model
    if _yolo_face_model is None:
        from ultralytics import YOLO
        import os
        if not os.path.exists(config.YOLO_FACE_MODEL_PATH):
            print(f"[models] YOLO-Face model not found: {config.YOLO_FACE_MODEL_PATH}")
            print("[models] Face detection disabled. Download yolo11n-face.pt to enable.")
            return None
        print(f"[models] Loading YOLO-Face model: {config.YOLO_FACE_MODEL_PATH}")
        _yolo_face_model = YOLO(config.YOLO_FACE_MODEL_PATH)
        print("[models] YOLO-Face model loaded.")
    return _yolo_face_model


def detect_faces(frame_bgr: np.ndarray) -> list:
    """
    Run YOLO-Face detection on a frame.
    Returns list of dicts: {confidence, bbox: (x1,y1,x2,y2)}
    """
    model = _load_yolo_face()
    if model is None:
        return []
    try:
        results = model(frame_bgr, conf=config.FACE_DETECTION_CONFIDENCE, half=True, verbose=False)
        faces = []
        for r in results:
            for box in r.boxes:
                faces.append({
                    "confidence": float(box.conf[0]),
                    "bbox": tuple(box.xyxy[0].cpu().numpy().astype(int).tolist()),
                })
        return faces
    except Exception as e:
        print(f"[models] Face detection error: {e}")
        return []


def detect_faces_batch(frames_bgr: list) -> list:
    """
    Run YOLO-Face detection on a batch of frames.
    Returns list of lists of face detections.
    """
    model = _load_yolo_face()
    if model is None or not frames_bgr: return [[] for _ in frames_bgr]
    try:
        results = model(frames_bgr, conf=config.FACE_DETECTION_CONFIDENCE, half=True, verbose=False, stream=True)
        batch_faces = []
        for r in results:
            faces = []
            for box in r.boxes:
                faces.append({
                    "confidence": float(box.conf[0]),
                    "bbox": tuple(box.xyxy[0].cpu().numpy().astype(int).tolist()),
                })
            batch_faces.append(faces)
        return batch_faces
    except Exception as e:
        print(f"[models] Face detection batch error: {e}")
        return [[] for _ in frames_bgr]


# ---- Biometric Re-ID (Body Appearance) ----

_dinov2_model = None


def _load_dinov2():
    global _dinov2_model
    if _dinov2_model is None:
        import torch
        device = _get_device()
        print(f"[models] Loading DINOv2 to {device}: {config.REID_MODEL_NAME}")
        # Use torch.hub to load DINOv2 (pure PyTorch Foundation Model)
        _dinov2_model = torch.hub.load('facebookresearch/dinov2', config.REID_MODEL_NAME)
        _dinov2_model = _dinov2_model.to(device)
        if device == "cuda":
            _dinov2_model = _dinov2_model.half()
        _dinov2_model.eval()
        print("[models] DINOv2 loaded (FP16).")
    return _dinov2_model


def encode_person_reid(images: list) -> np.ndarray:
    """Encode person crops with DINOv2 (True visual Re-ID). Fully batched."""
    import torch
    from torchvision import transforms
    model = _load_dinov2()
    if model is None or not images:
        return np.zeros((len(images), config.REID_DIM), dtype="float32")

    device = _get_device()
    # DINOv2 standard preprocessing requires multiple of 14 for patch size
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Batch preprocess + chunked forward pass for GPU OOM safety
    tensors = [preprocess(img) for img in images]
    out = np.zeros((len(images), config.REID_DIM), dtype="float32")
    batch_size = 32
    for i in range(0, len(tensors), batch_size):
        batch = torch.stack(tensors[i:i + batch_size]).to(device)
        with torch.no_grad():
            with _autocast():
                feats = model(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        out[i:i + batch.shape[0]] = feats.cpu().float().numpy()
    return out


# ---- Face Recognition (Identity) ----

_facenet_model = None


def _load_facenet():
    global _facenet_model
    if _facenet_model is None:
        try:
            from facenet_pytorch import InceptionResnetV1
            device = _get_device()
            print(f"[models] Loading FaceNet (InceptionResnetV1) to {device}...")
            _facenet_model = InceptionResnetV1(pretrained='vggface2').to(device).eval()
            if device == "cuda":
                _facenet_model = _facenet_model.half()
            print("[models] FaceNet ready (FP16).")
        except ImportError:
            print("[models] facenet-pytorch not installed. Face Identity disabled.")
            return None
    return _facenet_model


def encode_face_identity(frame_bgr: np.ndarray, bbox: tuple) -> np.ndarray:
    """Extract identity embedding for a face in a frame (BGR)."""
    model = _load_facenet()
    if model is None: return np.zeros((config.FACE_REC_DIM,), dtype="float32")
    
    import torch
    from torchvision import transforms
    from PIL import Image
    import cv2
    
    # Extract the face crop using the provided bbox
    x1, y1, x2, y2 = bbox
    h, w = frame_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    face_img = frame_bgr[y1:y2, x1:x2]
    
    if face_img.size == 0 or face_img.shape[0] < 10 or face_img.shape[1] < 10: 
        return np.zeros((config.FACE_REC_DIM,), dtype="float32")
    
    # Convert BGR to RGB PIL Image
    face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
    
    device = _get_device()
    # FaceNet requires 160x160 aligned input
    preprocess = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        # Fixed standardization used in FaceNet
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    inp = preprocess(face_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        with _autocast():
            emb = model(inp)
    
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().astype("float32")


def encode_faces_batch(face_crops_bgr: list) -> np.ndarray:
    """Extract identity embeddings for a batch of cropped face images (BGR)."""
    model = _load_facenet()
    if model is None or not face_crops_bgr: 
        return np.zeros((len(face_crops_bgr), config.FACE_REC_DIM), dtype="float32")
    
    import torch
    from torchvision import transforms
    from PIL import Image
    import cv2
    device = _get_device()
    
    preprocess = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    valid_indices = []
    tensors = []
    for i, face_bgr in enumerate(face_crops_bgr):
        if face_bgr is None or face_bgr.size == 0 or face_bgr.shape[0] < 10 or face_bgr.shape[1] < 10:
            continue
        try:
            face_pil = Image.fromarray(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB))
            tensors.append(preprocess(face_pil))
            valid_indices.append(i)
        except Exception:
            pass

    out_embeddings = np.zeros((len(face_crops_bgr), config.FACE_REC_DIM), dtype="float32")
    if not tensors:
        return out_embeddings
        
    # Process in chunks to avoid GPU OOM if there are hundreds of faces
    batch_size = 64
    for i in range(0, len(tensors), batch_size):
        batch_tensors = torch.stack(tensors[i:i+batch_size]).to(device)
        with torch.no_grad():
            with _autocast():
                emb = model(batch_tensors)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        
        for j, valid_idx in enumerate(valid_indices[i:i+batch_size]):
            out_embeddings[valid_idx] = emb[j].cpu().float().numpy()
            
    return out_embeddings

