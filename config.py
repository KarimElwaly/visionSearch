"""Configuration constants for Vision-Search Agent."""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
FRAMES_DIR = os.path.join(DATA_DIR, "frames")
DB_PATH = os.path.join(DATA_DIR, "vision_search.db")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "index.faiss")

# Separate FAISS indexes for security search
PERSON_FAISS_INDEX_PATH = os.path.join(DATA_DIR, "person_index.faiss")
FACE_FAISS_INDEX_PATH = os.path.join(DATA_DIR, "face_index.faiss")

# Ensure directories exist
os.makedirs(FRAMES_DIR, exist_ok=True)

# CCTV / investigation profile
CCTV_MODE = True
INDEX_PROFILE = "investigation"  # fast_preview | investigation | high_sensitivity

INDEX_PROFILES = {
    # Lower-cost first pass for quickly browsing long archives.
    "fast_preview": {"extract_fps": 0.5, "scene_skip_enabled": True},
    # Default CCTV mode: sample consistently so small moving people are not skipped.
    "investigation": {"extract_fps": 1.0, "scene_skip_enabled": False},
    # Use for short critical clips where missing a brief appearance is worse than speed.
    "high_sensitivity": {"extract_fps": 2.0, "scene_skip_enabled": False},
}

_ACTIVE_INDEX_PROFILE = INDEX_PROFILES.get(INDEX_PROFILE, INDEX_PROFILES["investigation"])


def get_index_profile(name: str = None) -> dict:
    """Return a valid indexing profile, falling back to investigation mode."""
    return INDEX_PROFILES.get(name or INDEX_PROFILE, INDEX_PROFILES["investigation"])

# Video frame extraction
EXTRACT_FPS = float(_ACTIVE_INDEX_PROFILE["extract_fps"])  # Frames per second to extract
SCENE_SKIP_ENABLED = bool(_ACTIVE_INDEX_PROFILE["scene_skip_enabled"])

# YOLO detection
YOLO_MODEL_PATH = os.path.join(os.path.dirname(BASE_DIR), "yolo11n.pt")
YOLO_CONFIDENCE = 0.3
YOLO_CLASSES = None  # None = all 80 COCO classes. Use [0] for person-only.
IMAGE_SEARCH_YOLO_CONFIDENCE = 0.2

# YOLO-Face detection (for face cropping during indexing)
YOLO_FACE_MODEL_PATH = os.path.join(os.path.dirname(BASE_DIR), "yolo11n-face.pt")
FACE_DETECTION_CONFIDENCE = 0.4

# YOLO-World (open-vocabulary detection — used at search time for re-ranking)
YOLO_WORLD_MODEL_PATH = os.path.join(os.path.dirname(BASE_DIR), "yolov8m-worldv2.pt")
YOLO_WORLD_CONFIDENCE = 0.15

# CLIP model — ViT-L/14 is much better at colors, clothing, fine-grained attributes
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
CLIP_DIM = 768  # ViT-L/14 uses 768-dim embeddings (ViT-B/32 was 512)
CLIP_BATCH_SIZE = 16  # Smaller batch for larger model
CLIP_MEGA_BATCH = 64  # Increased to 64: forces the GPU to solve 64 frames/faces simultaneously to max out 6GB VRAM

# Dedicated Biometric Models — Pure PyTorch (C++ Free)
REID_MODEL_NAME = "dinov2_vits14" # DINOv2 ViT-Small for body Re-ID
REID_DIM = 384
FACE_REC_DIM = 512 # FaceNet InceptionResnetV1 for identity

# Matching thresholds (Cosine Similarity)
# Dedicated foundation models are highly sensitive; thresholds are tuned for metric learning spaces.
# Multi-orientation body Re-ID handles rotated queries — thresholds stay strict to avoid false positives.
PERSON_MATCH_THRESHOLD = 0.72  # DINOv2 body similarity
FACE_MATCH_THRESHOLD = 0.50    # FaceNet similarity
BODY_ONLY_DISCOUNT = 0.6       # Discount factor for body-only results (no face confirmation)

# Cross-camera person Re-ID robustness
PERSON_MULTI_ORIENTATION = True    # Test multiple rotations/flips of the query image
PERSON_LIGHTING_NORMALIZE = True   # Apply CLAHE normalization for day/night invariance
PERSON_CLIP_FALLBACK = True        # Also search CLIP general index as safety net
CLIP_FALLBACK_SCORE_WEIGHT = 0.45  # Weight for CLIP fallback results vs biometric matches

# Lightweight CCTV tracking (groups frame-level detections into event tracks)
TRACK_IOU_THRESHOLD = 0.25      # Lower values tolerate low-FPS jumps; higher values reduce ID switches
TRACK_MAX_MISSED = 5            # Number of sampled frames to keep a track alive without a match

# CCTV vehicle investigation classes (COCO names emitted by YOLO)
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle", "train", "boat"}


# Ollama VLM (for rich text descriptions)
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_VLM_MODEL = "qwen3-vl:2b"  # qwen3.5:0.8b doesn't process images. qwen3-vl:2b works.

# Search
DEFAULT_TOP_K = 20
CLIP_CANDIDATE_POOL_FACTOR = 8
CLIP_MIN_CANDIDATES = 100
IMAGE_CANDIDATE_POOL_FACTOR = 8
IMAGE_MIN_CANDIDATES = 100
PERSON_CANDIDATE_POOL_FACTOR = 20
PERSON_MIN_CANDIDATES = 500
FACE_CANDIDATE_POOL_FACTOR = 20
FACE_MIN_CANDIDATES = 500

# On-disk frame + crop storage (Phase 5).
# Full frames tend to have detail across the whole image, crops are already small,
# and face crops need to preserve identity cues — hence three settings.
FRAME_JPEG_QUALITY = 85
CROP_JPEG_QUALITY = 85
FACE_JPEG_QUALITY = 90

# Server
HOST = "0.0.0.0"
PORT = 8420

