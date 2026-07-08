# 🔍 Vision-Search Agent

**Context-Aware Video Retrieval** — A local, GPU-accelerated video search engine that finds specific people and objects in videos using natural language, image similarity, or person re-identification.

---

## Architecture

```
Video → Frame Extraction (1 FPS) → Scene-Change Filter → YOLO Detection → Crop Each Person/Object
                                                                │
                                          ┌─────────────────────┼──────────────────────┐
                                          ▼                     ▼                      ▼
                                   CLIP ViT-L/14         DINOv2 ViT-S14         FaceNet (VGGFace2)
                                  (Vector Embeddings)    (Person Re-ID)         (Face Identity)
                                          │                     │                      │
                                          ▼                     ▼                      ▼
                                    FAISS Index          Person FAISS Index      Face FAISS Index
                                  (General Search)      (Body Matching)         (Face Matching)

Search Query ──► CLIP Recall (Stage 1) ──► YOLO-World Re-ranking (Stage 2) ──► Temporal Boost (Stage 3)
                     <1ms                        ~1-2 seconds                       instant
```

## Search Modes

| Mode | Description |
|------|-------------|
| **Text Search** | Natural language queries like "person in black jacket with bag" |
| **Image Search** | Upload a photo to find visually similar frames/objects |
| **Person Search** | Upload a person's photo to find them across all indexed videos (face + body) |

## Components

| Component | File | Description |
|---|---|---|
| **Config** | `config.py` | All settings: FPS, model paths, thresholds, server port |
| **Models** | `models.py` | CLIP encoding, YOLO detection, DINOv2 Re-ID, FaceNet identity, Ollama VLM |
| **Database** | `database.py` | SQLite with FTS5 full-text search for metadata & captions |
| **Indexer** | `indexer.py` | Video → frames → YOLO crops → CLIP/DINOv2/FaceNet embeddings (mega-batched) |
| **Searcher** | `searcher.py` | 3-stage smart search: CLIP recall → YOLO-World re-rank → temporal boost |
| **API Server** | `app.py` | FastAPI REST API + static file serving |
| **Frontend** | `static/` | Dark-themed web UI with text, image, and person search tabs |

## Tech Stack

- **Object Detection**: YOLOv11n (ultralytics) — indexing & crop extraction
- **Face Detection**: YOLOv11n-face — face crop extraction during indexing
- **Open-Vocab Detection**: YOLOv8m-WorldV2 — search-time re-ranking with user's exact query
- **Vector Embeddings**: CLIP ViT-L/14 (768-dim, FP16 on GPU)
- **Person Re-ID**: DINOv2 ViT-S/14 (384-dim body appearance matching)
- **Face Recognition**: FaceNet InceptionResnetV1 (512-dim identity matching)
- **VLM Captioning**: Qwen3-VL 2B via Ollama (optional, for rich text descriptions)
- **Vector Search**: FAISS (3 separate indexes: general, person, face)
- **Text Search**: SQLite FTS5
- **Backend**: FastAPI + Uvicorn
- **Frontend**: Vanilla HTML/CSS/JS

---

## Prerequisites

1. **Python 3.10+** with a virtual environment
2. **CUDA GPU** recommended (6GB+ VRAM) — falls back to CPU
3. **Ollama** (optional, for VLM captions):
   ```bash
   ollama pull qwen3-vl:2b
   ```
4. **Model files** in the parent directory (`../`):
   - `yolo11n.pt` — object detection
   - `yolo11n-face.pt` — face detection
   - `yolov8m-worldv2.pt` — open-vocabulary re-ranking

## Installation

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows
# source .venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

## Running

```bash
# (Optional) Start Ollama for VLM captions
ollama serve

# Start the Vision-Search Agent
python app.py
```

Open **http://localhost:8420** in your browser.

> **Note:** On first launch, all AI models are loaded into memory (10-20 seconds). Subsequent searches are instant.

## Usage

### 1. Index a Video
- Click **"+ Index New Video"** to expand the panel
- Paste a video file path (e.g., `D:\videos\sample.mp4`) or a directory
- Options:
  - ✅ **CLIP Embeddings** — vector embeddings + DINOv2 + FaceNet (recommended)
  - ✅ **VLM Captions** — Ollama generates text descriptions (slower, optional)
- Click **Start Indexing** and watch the progress bar

### 2. Text Search
- Type a natural language query:
  - `"person in black shirt with bag"`
  - `"red car"`, `"man walking"`, `"woman with glasses"`
- Supports negative prompts: `"person in jacket --no hat"`
- Search modes: **Hybrid** (default), **Text** only, **CLIP** only

### 3. Image Search
- Switch to the **Image** tab
- Upload or drag-drop a reference image
- Finds visually similar frames and objects across all indexed videos

### 4. Person Search (Re-ID)
- Switch to the **Person** tab
- Upload a photo of a person (face crop or full body)
- Uses dedicated biometric models (DINOv2 + FaceNet) for accurate identity matching
- Finds the same person across different cameras, angles, and lighting

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/search?q=...&top_k=20&mode=hybrid` | Text search indexed videos |
| `POST` | `/api/search-by-image` | Image similarity search (multipart file upload) |
| `POST` | `/api/search-by-person` | Person re-identification (multipart file upload) |
| `POST` | `/api/index` | Start indexing (body: `{path, use_vlm, use_clip}`) |
| `GET` | `/api/status` | Get indexing progress |
| `GET` | `/api/videos` | List all indexed videos |
| `DELETE` | `/api/videos/{id}` | Remove video from index |
| `GET` | `/api/frame?path=...` | Serve frame/crop image |

## Configuration

Edit `config.py` to customize:

```python
EXTRACT_FPS = 1                # Frames per second to extract
YOLO_CONFIDENCE = 0.3          # Detection confidence threshold
YOLO_CLASSES = None            # None = all 80 COCO classes; [0] = person only
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
CLIP_MEGA_BATCH = 64           # Frames processed simultaneously (GPU batch size)
PERSON_MATCH_THRESHOLD = 0.55  # DINOv2 cosine similarity threshold
FACE_MATCH_THRESHOLD = 0.50    # FaceNet cosine similarity threshold
OLLAMA_VLM_MODEL = "qwen3-vl:2b"
PORT = 8420
```

## Performance

- **Indexing**: ~2-5 frames/sec on GPU (depends on video resolution and detection count)
- **Text Search**: <2 seconds (CLIP recall + YOLO-World re-ranking)
- **Image Search**: <1 second
- **Person Search**: <1 second
- Scene-change deduplication skips redundant frames automatically
