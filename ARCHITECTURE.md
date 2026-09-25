# Vision-Search Agent — Architecture

This document describes the runtime architecture of Vision-Search Agent: what
runs where, what data flows between components, and which files own each
responsibility.

For per-stage math and heuristics see [ALGORITHMS.md](ALGORITHMS.md).

---

## 1. High-level view

```
                          ┌────────────────────────────────────────┐
                          │  Browser  (static/index.html, app.js)  │
                          │  4 tabs · CCTV filters · track browser │
                          └────────────┬───────────────────────────┘
                                       │  HTTP + multipart/JSON
                                       ▼
                          ┌────────────────────────────────────────┐
                          │  FastAPI + Uvicorn  (app.py)           │
                          │  routing · filters · track grouping    │
                          └───┬─────────────┬───────────────┬──────┘
                              │             │               │
              ┌───────────────┘             │               └──────────────┐
              ▼                             ▼                              ▼
     ┌────────────────┐          ┌──────────────────┐          ┌──────────────────┐
     │  Indexer       │          │  Searcher        │          │  Database (SQLite│
     │  (indexer.py)  │          │  (searcher.py)   │──reads──►│   + FTS5)        │
     │  ingest/write  │──writes─►│  recall + rerank │          │  (database.py)   │
     └───┬─────────┬──┘          └────────┬─────────┘          └──────────────────┘
         │         │                      │                              ▲
         │         │                      │                              │
         ▼         ▼                      ▼                              │
   ┌──────────┐  ┌─────────┐        ┌──────────┐                         │
   │ Models   │  │ Tracker │        │ FAISS    │  cached in-memory,      │
   │(models.py│  │(tracker │        │ (3 idx)  │  invalidated by mtime   │
   │ CLIP,    │  │ .py)    │        │ general  │─────────────────────────┘
   │ YOLO x2, │  │ IoU     │        │ person   │
   │ DINOv2,  │  │ tracker │        │ face     │
   │ FaceNet) │  └─────────┘        └──────────┘
   └──────────┘
```

## 2. Components

| File | Layer | Responsibility |
|------|-------|----------------|
| [`app.py`](app.py) | API | FastAPI routes, filter/grouping helpers, path-safe frame serving, index-health report, model warm-up on lifespan startup |
| [`indexer.py`](indexer.py) | Pipeline | Frame extraction, YOLO detection, tracking, DINOv2/FaceNet encoding, FAISS + DB persistence, cascade delete |
| [`searcher.py`](searcher.py) | Retrieval | CLIP recall + FTS5 recall, YOLO-World re-ranking, color re-ranking, temporal boost, near-duplicate clustering, FAISS mtime-cache |
| [`models.py`](models.py) | ML | Lazy singletons for CLIP, YOLO, YOLO-Face, DINOv2, FaceNet, optional Ollama VLM |
| [`tracker.py`](tracker.py) | Pipeline | `SimpleIOUTracker` — per-class greedy IoU tracker used during indexing |
| [`database.py`](database.py) | Storage | Schema, migrations, FTS5, batch inserts, cascade queries, storage/track stats |
| [`config.py`](config.py) | Config | All tunables: profile, thresholds, batch sizes, model paths |
| [`static/index.html`](static/index.html), [`static/app.js`](static/app.js), [`static/style.css`](static/style.css) | UI | Tabbed search UI, CCTV filter panel, track browser, indexing progress |

## 3. Data model

### 3.1 SQLite tables (defined in [database.py](database.py))

```
videos ──┬─< frames ──┬─< detections ──┬─< faces
         │            │                │
         │            │                └─< track_detections >──┐
         │            │                                        │
         │            └──────────────< frame_embeddings        │
         │                                                     │
         └──< tracks ──────────────────────────────────────────┘
             │
             └─< track_attributes  (optional per-track metadata k/v)
```

- **`videos`** — one row per source video. Holds duration/fps, camera_id,
  location, timezone, index_profile, indexing status, resume checkpoint
  (`last_frame_number`, `frames_processed`), and failure message.
- **`frames`** — sampled frames on disk (`FRAMES_DIR/{video_id}/frame_*.jpg`).
  Unique on `(video_id, frame_number)`.
- **`detections`** — one row per YOLO crop with class/confidence/bbox, plus
  three FAISS ids: `faiss_id` (general CLIP), `person_faiss_id` (DINOv2 Re-ID
  for `person` crops), and `track_id` (link to the `tracks` row).
- **`faces`** — one row per face crop within a person detection, with
  `face_faiss_id` (FaceNet identity).
- **`frame_embeddings`** — full-frame CLIP vectors (scene-level recall).
- **`tracks`** — CCTV events (start/end frame + timestamp, best detection,
  detection_count, best_confidence, class). One track per moving object.
- **`track_detections`** — timeline: every detection that belongs to a track,
  ordered by timestamp. Feeds the evidence export.
- **FTS5 shadow tables** — `detections_fts` and `frame_embeddings_fts` are
  auto-populated by triggers on the `caption` column.

Covering indexes exist for all foreign-key joins the searcher performs.

### 3.2 FAISS indexes (all IDMap2)

| Index | Path (`config.py`) | Dim | Backing | Content |
|-------|--------------------|-----|---------|---------|
| general | `FAISS_INDEX_PATH` | 768 | HNSW (M=32, efSearch=64) | CLIP ViT-L/14 for every frame + every crop |
| person | `PERSON_FAISS_INDEX_PATH` | 384 | Flat IP | DINOv2 ViT-S/14 for `person` crops only |
| face | `FACE_FAISS_INDEX_PATH` | 512 | Flat IP | FaceNet InceptionResnetV1 for face crops |

Every index has a sidecar `<path>.meta.json` with `{dim, count, next_id, saved_at}`.
The sidecar is what powers the `/api/index/health` integrity check without
loading the vectors.

### 3.3 On-disk layout

```
data/
  vision_search.db            SQLite (WAL mode)
  vision_search.db-wal        WAL file
  index.faiss                 general index
  index.faiss.meta.json       sidecar
  person_index.faiss
  person_index.faiss.meta.json
  face_index.faiss
  face_index.faiss.meta.json
  frames/
    <video_id>/
      frame_000030.jpg           extracted frame
      frame_000030_det_0.jpg     detection crop
      frame_000030_det_0_face_0.jpg  face crop
      ...
```

## 4. Runtime concerns

### 4.1 Concurrency

- One indexing job at a time, guarded by `indexing_status["active"]`. The
  cooperative cancel flag is `indexing_status["cancel_requested"]`.
- The API runs on Uvicorn's async loop; blocking work (indexing, some searches)
  runs on background threads.
- SQLite is opened per-thread via `threading.local()` in
  [`database.get_connection`](database.py), with `PRAGMA journal_mode=WAL /
  synchronous=NORMAL / temp_store=MEMORY / foreign_keys=ON`.
- FAISS cache in `searcher._FAISS_CACHE` is guarded by
  `_FAISS_CACHE_LOCK`. Cache key = (mtime_ns, size); invalidation happens
  automatically after every indexing checkpoint since atomic writes bump the
  mtime.

### 4.2 Memory & disk

- Peak RAM is bounded by one **mega-batch** (default 64 frames + their crops).
  After a flush the pipeline drops `frame_bgr_ref` and `crop_images` so the
  next batch starts from a clean baseline.
- All frame and crop writes go through a `ThreadPoolExecutor(max_workers=4)`
  so encoding never stalls on disk I/O.
- FAISS is checkpointed every `FAISS_CHECKPOINT_EVERY = 5` mega-batches and
  once more at the end. Every write is `.tmp → os.replace()` (atomic).

### 4.3 Crash safety

- Video row status transitions: `pending → indexing → done`,
  or `indexing → failed` (with `error_message`), or `indexing → paused`
  (cooperative cancel).
- On restart the indexer resumes from `max(MAX(frame_number in frames),
  videos.last_frame_number)` — the DB is ground truth, so a mid-batch crash
  never loses more than one mega-batch.
- Cascade delete ([`indexer.delete_video_fully`](indexer.py)) removes FAISS
  vectors, DB rows (via `ON DELETE CASCADE`), the `data/frames/{id}` directory,
  then runs `VACUUM/ANALYZE` to reclaim SQLite pages.

### 4.4 Security

- Frame serving in [`app.py`](app.py) validates paths with
  `os.path.realpath + commonpath` against `FRAMES_DIR` so `..` and symlink
  escapes cannot exfiltrate arbitrary files.
- FTS5 queries are sanitized (see [`searcher._build_fts_query`](searcher.py))
  before hitting SQLite, so no user text can trigger `OperationalError`.
- The server binds to `0.0.0.0:8420` and does not authenticate; treat as
  local-network only.

## 5. Request lifecycles

### 5.1 Indexing (`POST /api/index`)

```
Client POST {path, camera_id, location, index_profile}
   │
   ▼
app.py._run() launches a daemon Thread that calls
   indexer.index_video(...)
     │
     ▼
   _index_video_impl:
     1. Resolve/insert `videos` row, set status=indexing
     2. Compute frame_interval from profile FPS
     3. Resume checkpoint from MAX(frame_number)+interval
     4. Spawn extractor thread → frame_queue (bounded 128)
     5. Loop:
          - dequeue frame
          - optional scene-skip (fast_preview only)
          - save frame async
          - append to mega_batch
          - when full → _detect_track_crop_batch → _flush_mega_batch
              (YOLO → tracker → crop persist → CLIP+DINOv2+face)
          - checkpoint DB every batch, FAISS every 5 batches
     6. Flush remainder
     7. Save FAISS, set status=done|paused|failed
```

### 5.2 Text search (`GET /api/search`)

```
Client GET /api/search?q=...&video_id=...&group_tracks=true
   │
   ▼
app.py._expanded_top_k() bumps recall when filters/grouping are on
   │
   ▼
searcher.search(query, top_k, mode="hybrid"):
   Stage 1: CLIP recall via _get_cached_faiss(FAISS_INDEX_PATH)
              → detections + frame-level scene results
             FTS5 recall via db.search_by_text / search_frames_by_text
              (query passed through _build_fts_query)
   Stage 2: YOLO-World re-rank on full frames of top candidates
   Stage 3: Temporal continuity boost (±3 s neighbours)
   Stage 4: Near-duplicate clustering (±2 frames per video)
   │
   ▼
app.py._filter_results() applies CCTV filters
app.py._group_results_by_track() collapses hits by track_id when asked
   │
   ▼
JSON response with score, source, track_id, camera_id, cluster count …
```

### 5.3 Image / vehicle / person search

- Image: encode with CLIP → search general index → color re-rank → temporal
  boost → cluster.
- Vehicle: same as image, then keep only classes in `VEHICLE_CLASSES` and
  group by track.
- Person: YOLO-Face on the query → run DINOv2 on the body (skipped if the
  query is a tight face crop) → hit **person** index → run FaceNet only when
  a face was detected → hit **face** index → merge (body-only discounted,
  body+face boosted).

### 5.4 Delete (`DELETE /api/videos/{id}`)

```
delete_video_fully:
  1. Read all FAISS ids for the video from DB (grouped by index)
  2. remove_ids on each FAISS index (HNSW falls back to rebuild)
  3. atomic save FAISS + sidecar
  4. db.delete_video → cascade removes frames/detections/faces/tracks/…
  5. shutil.rmtree(FRAMES_DIR/<id>)
  6. VACUUM + ANALYZE
```

## 6. Configuration surface

Everything tunable lives in [`config.py`](config.py):

- **Profiles** — `INDEX_PROFILES` maps `fast_preview | investigation |
  high_sensitivity` to `(extract_fps, scene_skip_enabled)`.
- **Thresholds** — `YOLO_CONFIDENCE`, `FACE_DETECTION_CONFIDENCE`,
  `YOLO_WORLD_CONFIDENCE`, `PERSON_MATCH_THRESHOLD`, `FACE_MATCH_THRESHOLD`,
  `TRACK_IOU_THRESHOLD`, `TRACK_MAX_MISSED`.
- **Recall pools** — `*_CANDIDATE_POOL_FACTOR` and `*_MIN_CANDIDATES` control
  how many vectors each search stage pulls before re-ranking.
- **Storage quality** — `FRAME_JPEG_QUALITY`, `CROP_JPEG_QUALITY`,
  `FACE_JPEG_QUALITY`.

## 7. Extension points

- **New search stage** — implement a `_stage(results) → results` function in
  [`searcher.py`](searcher.py) and slot it into the pipeline in
  `search()` / `search_by_image()`.
- **Better tracker** — swap [`tracker.SimpleIOUTracker`](tracker.py) for a
  ByteTrack/BoT-SORT wrapper that exposes the same `update()` signature.
- **New embedding** — add a loader in [`models.py`](models.py), an FAISS index
  path in [`config.py`](config.py), and populate it inside `_flush_mega_batch`.
- **Track attributes** — write into `track_attributes` (color, direction,
  plate, gender…) during indexing and filter/search on them in `app.py`.
