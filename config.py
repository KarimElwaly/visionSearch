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

# Video frame extraction
EXTRACT_FPS = 1  # Frames per second to extract

# YOLO detection
YOLO_MODEL_PATH = os.path.join(os.path.dirname(BASE_DIR), "yolo11n.pt")
YOLO_CONFIDENCE = 0.3
YOLO_CLASSES = None  # None = all 80 COCO classes. Use [0] for person-only.

# YOLO-Face detection (for face cropping during indexing)
YOLO_FACE_MODEL_PATH = os.path.join(os.path.dirname(BASE_DIR), "yolo11n-face.pt")

# YOLO-World (open-vocabulary detection — used at search time for re-ranking)
YOLO_WORLD_MODEL_PATH = os.path.join(os.path.dirname(BASE_DIR), "yolov8m-worldv2.pt")

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
PERSON_MATCH_THRESHOLD = 0.55  # DINOv2 similarity
FACE_MATCH_THRESHOLD = 0.50    # FaceNet similarity


# Ollama VLM (for rich text descriptions)
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_VLM_MODEL = "qwen3-vl:2b"  # qwen3.5:0.8b doesn't process images. qwen3-vl:2b works.

# Search
DEFAULT_TOP_K = 20

# Server
HOST = "0.0.0.0"
PORT = 8420

