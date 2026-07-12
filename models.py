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


def _maybe_compile(model, name: str):
    """Wrap ``model`` in ``torch.compile`` when enabled by config, else return as-is.

    * No-op on CPU (compile only helps CUDA kernels).
    * Any exception during compilation is caught and logged — we fall back to the
      eager model so a Dynamo incompatibility in one dependency (e.g. facenet's
      InceptionResnetV1) cannot break the whole pipeline.
    * We also set ``torch._dynamo.config.suppress_errors = True`` so *runtime*
      compilation failures (e.g. triton/torch version skew that only surfaces
      on the first real forward pass) transparently fall back to eager without
      crashing the caller. Without this flag, a bad triton install would kill
      the indexing thread the moment the first mega-batch reaches DINOv2.
    * First forward call after compile pays a 30-60s warmup; every subsequent
      call is faster.
    """
    if not getattr(config, "TORCH_COMPILE_ENABLED", False):
        return model
    if _get_device() != "cuda":
        return model
    try:
        import torch
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
        mode = getattr(config, "TORCH_COMPILE_MODE", "default")
        compiled = torch.compile(model, mode=mode)
        print(f"[models] torch.compile enabled for {name} (mode={mode}).")
        return compiled
    except Exception as e:
        print(f"[models] torch.compile failed for {name}: {type(e).__name__}: {e} — falling back to eager.")
        return model


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
        _clip_model = _maybe_compile(_clip_model, "CLIP")
        print("[models] CLIP model loaded (FP16).")
    return _clip_model, _clip_processor


def encode_images_clip(images: list) -> np.ndarray:
    """Encode a batch of PIL images into normalized CLIP embeddings."""
    import torch
    model, processor = _load_clip()
    device = _get_device()
    all_embeddings = []
    for i in range(0, len(images), config.CLIP_BATCH_SIZE):
        batch = images[i:i + config.CLIP_BATCH_SIZE]
        inputs = processor(images=batch, return_tensors="pt", padding=True).to(device)
        # Cast pixel_values to FP16 on GPU once (matches the FP16 model weights).
        # Skips the FP32->FP16 conversion autocast would do inside every forward
        # pass and avoids one full-size tensor copy per batch. Only apply when
        # the model is actually running in FP16.
        if device == "cuda" and "pixel_values" in inputs:
            try:
                weight_dtype = next(model.parameters()).dtype
            except StopIteration:
                weight_dtype = torch.float16
            if weight_dtype == torch.float16:
                inputs["pixel_values"] = inputs["pixel_values"].half()
        with torch.no_grad():
            with _autocast():
                output = model.get_image_features(**inputs)
        # transformers 5.x returns BaseModelOutputWithPooling; older returns tensor
        embeds = output.pooler_output if hasattr(output, 'pooler_output') else output
        embeds = embeds / embeds.norm(dim=-1, keepdim=True)
        all_embeddings.append(embeds.float().cpu().numpy())
    return np.vstack(all_embeddings).astype("float32")


def encode_text_clip(text: str) -> np.ndarray:
    """Encode a text query into a normalized CLIP embedding."""
    import torch
    model, processor = _load_clip()
    device = _get_device()
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        with _autocast():
            output = model.get_text_features(**inputs)
    # transformers 5.x returns BaseModelOutputWithPooling; older returns tensor
    embeds = output.pooler_output if hasattr(output, 'pooler_output') else output
    embeds = embeds / embeds.norm(dim=-1, keepdim=True)
    return embeds.float().cpu().numpy().astype("float32")


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
        _dinov2_model = _maybe_compile(_dinov2_model, "DINOv2")
        print("[models] DINOv2 loaded (FP16).")
    return _dinov2_model


def encode_person_reid(images: list) -> np.ndarray:
    """Encode person crops with DINOv2 (True visual Re-ID). Fully batched.

    Preprocessing runs on the GPU: PIL resize (fast C code) → single ``np.stack`` →
    one H2D copy of a uint8 ``(N,H,W,3)`` tensor → vectorized ``permute + float +
    div_(255) + normalize`` on device. This eliminates the per-image ``ToTensor +
    Normalize`` CPU loop (~2-5 ms/crop) that used to leave the GPU idle between
    mega-batches.
    """
    import torch
    model = _load_dinov2()
    if model is None or not images:
        return np.zeros((len(images), config.REID_DIM), dtype="float32")

    device = _get_device()
    # DINOv2 patch size is 14 — 224 is a safe multiple.
    target_size = (224, 224)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    out = np.zeros((len(images), config.REID_DIM), dtype="float32")
    # DINOv2 ViT-S/14 is small (~90 MB FP16); 64 saturates the GPU better than 32
    # without threatening the 6 GB VRAM budget shared with CLIP-L.
    batch_size = 64
    for i in range(0, len(images), batch_size):
        chunk = images[i:i + batch_size]
        # Resize on CPU (PIL is fast C code) and stack into one uint8 array.
        resized = np.stack([
            np.asarray(img.convert("RGB").resize(target_size, Image.BILINEAR), dtype=np.uint8)
            for img in chunk
        ], axis=0)  # (B, H, W, 3)
        batch = torch.from_numpy(resized).to(device, non_blocking=True)
        # HWC uint8 -> CHW float, /255, normalize — all on GPU.
        batch = batch.permute(0, 3, 1, 2).contiguous().float().div_(255.0)
        batch = (batch - mean) / std
        if device == "cuda":
            batch = batch.half()
        with torch.no_grad():
            with _autocast():
                feats = model(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        out[i:i + batch.shape[0]] = feats.float().cpu().numpy()
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
            _facenet_model = _maybe_compile(_facenet_model, "FaceNet")
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
    """Extract identity embeddings for a batch of cropped face images (BGR).

    Preprocessing runs on the GPU: cv2 resize per face (fast C code) → single
    ``np.stack`` → one H2D copy → vectorized permute/float/div/normalize on
    device. Replaces the per-image ``ToTensor + Normalize`` CPU loop.
    """
    model = _load_facenet()
    if model is None or not face_crops_bgr:
        return np.zeros((len(face_crops_bgr), config.FACE_REC_DIM), dtype="float32")

    import torch
    import cv2
    device = _get_device()
    target_size = (160, 160)
    # FaceNet's fixed standardization: (x - 0.5) / 0.5
    mean = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)

    # Filter invalid crops, resize, and remember original indices for scatter-back.
    valid_indices = []
    resized_list = []
    for i, face_bgr in enumerate(face_crops_bgr):
        if face_bgr is None or face_bgr.size == 0 or face_bgr.shape[0] < 10 or face_bgr.shape[1] < 10:
            continue
        try:
            rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, target_size, interpolation=cv2.INTER_LINEAR)
            resized_list.append(resized)
            valid_indices.append(i)
        except Exception:
            pass

    out_embeddings = np.zeros((len(face_crops_bgr), config.FACE_REC_DIM), dtype="float32")
    if not resized_list:
        return out_embeddings

    stacked = np.stack(resized_list, axis=0)  # (N, 160, 160, 3) uint8
    # Process in chunks to bound GPU memory when there are hundreds of faces.
    batch_size = 64
    for i in range(0, len(resized_list), batch_size):
        chunk = stacked[i:i + batch_size]
        batch = torch.from_numpy(chunk).to(device, non_blocking=True)
        batch = batch.permute(0, 3, 1, 2).contiguous().float().div_(255.0)
        batch = (batch - mean) / std
        if device == "cuda":
            batch = batch.half()
        with torch.no_grad():
            with _autocast():
                emb = model(batch)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        emb_np = emb.float().cpu().numpy()
        for j, valid_idx in enumerate(valid_indices[i:i + batch_size]):
            out_embeddings[valid_idx] = emb_np[j]

    return out_embeddings

