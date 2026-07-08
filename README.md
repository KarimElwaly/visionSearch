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
- Click **Cancel** to pause; the job resumes automatically on the next indexing call
- If a video fails, its row is marked `failed` (with error message) and can be retried
  via `POST /api/videos/{id}/reindex`

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
| `POST` | `/api/index/cancel` | Cooperatively stop the current indexing job (state → `paused`) |
| `GET` | `/api/status` | Get current indexing progress |
| `GET` | `/api/health` | Lightweight readiness probe (no model calls) |
| `GET` | `/api/videos` | List all indexed videos |
| `GET` | `/api/videos/{id}` | Per-video details + row-count stats |
| `POST` | `/api/videos/{id}/reindex` | Purge derived data + re-index from source |
| `DELETE` | `/api/videos/{id}` | Cascade delete: DB rows + FAISS vectors + frames dir |
| `GET` | `/api/storage` | Per-video row counts (frames / detections / faces) |
| `GET` | `/api/frame?path=...` | Serve frame or crop image (path-traversal safe) |

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
FRAME_JPEG_QUALITY = 85        # On-disk frame quality (Phase 5)
CROP_JPEG_QUALITY = 85         # On-disk detection-crop quality
FACE_JPEG_QUALITY = 90         # On-disk face-crop quality (higher = preserve identity cues)
OLLAMA_VLM_MODEL = "qwen3-vl:2b"
PORT = 8420
```

## Scalability & Resilience

Designed to handle **multi-hour videos** and large corpora without falling over:

- **FAISS HNSW + IDMap2** — sublinear search (M=32, efSearch=64) with stable ids that
  survive restarts and deletions. Legacy `IndexFlatIP` files auto-migrate on first load.
  A `.faiss.meta.json` sidecar records `dim / count / next_id / saved_at` for integrity checks.
- **Atomic saves** — every FAISS write is `.tmp` → `os.replace()`, so a crash mid-save cannot corrupt the index.
- **Checkpoint / resume** — the indexer persists `videos.last_frame_number` + `frames_processed` after each mega-batch
  and flushes FAISS to disk every 5 mega-batches. Restarting a crashed / paused job resumes near the last checkpoint
  (uses `MAX(frame_number)` as ground truth).
- **Bounded RAM per mega-batch** — heavy per-frame buffers (`frame_bgr_ref`, `crop_images`) are freed immediately
  after each flush, so peak memory does not grow with video length.
- **Failure isolation** — exceptions during indexing set `videos.status='failed'` with the message stored in
  `videos.error_message`; the API stays healthy and the row can be re-indexed via `POST /api/videos/{id}/reindex`.
- **Cancellation** — `POST /api/index/cancel` cooperatively stops the current job after the in-flight batch;
  status becomes `paused` and the next indexing call resumes automatically.
- **Drift-safe seeking** — `cap.set(CAP_PROP_POS_FRAMES, n)` is checked for drift and falls back to sequential
  `cap.grab()` on codecs where seeking is inaccurate (variable-GOP h.264 with B-frames).
- **Cascade delete** — removing a video via the API deletes DB rows (`ON DELETE CASCADE`), FAISS vectors
  (via `remove_ids` or reconstruct-rebuild for HNSW), the on-disk `data/frames/{id}` directory, and runs
  `VACUUM/ANALYZE` to reclaim SQLite pages.
- **SQLite indexes** — covering indexes on `frames(video_id)`, `detections(faiss_id | person_faiss_id | frame_id)`,
  `faces(face_faiss_id | detection_id)`, `frame_embeddings(faiss_id)`, plus a unique constraint on
  `frames(video_id, frame_number)`. `PRAGMA journal_mode=WAL / synchronous=NORMAL / temp_store=MEMORY`.

## Search Quality

- **Prompt-template ensemble** — CLIP text queries are averaged over multiple templates
  (`"a photo of {q}"`, `"a picture of {q}"`, …) for +2–5 pts recall on retrieval-style queries.
- **Attribute-preserving parsing** — color words (`black`, `white`, `red`, …) are **not** treated as stopwords
  when constructing YOLO-World classes, so `"black jacket"` searches for black jackets rather than any jacket.
- **Negative prompts** — `"person with jacket --no hat"` subtracts the negative embedding at 0.5 strength.
- **YOLO-World re-rank** operates on detection crops (when available) instead of full frames, avoiding
  background-object noise.
- **BM25 normalization** — FTS scores are transformed to `1 / (1 + |rank|)` so text and CLIP scores merge
  cleanly instead of having text scores dominate.
- **Person search** — FaceNet is only invoked when YOLO-Face actually detects a face in the query image,
  preventing meaningless embeddings from non-face inputs.
- **Temporal continuity boost** — results in a cluster (±3s in the same video) get +20 %, isolated hits
  get −10 % (real objects persist, glitches don't).
- **Near-duplicate clustering** — result cards from the same video within ±2 frames are collapsed into
  a single card with a `+N` badge showing sibling count.

## Performance

- **Indexing**: ~2-5 frames/sec on GPU (depends on video resolution and detection count)
- **Text Search**: <2 seconds (CLIP recall + YOLO-World re-ranking)
- **Image Search**: <1 second
- **Person Search**: <1 second
- Scene-change deduplication skips redundant frames automatically
