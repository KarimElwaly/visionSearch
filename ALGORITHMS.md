# Vision-Search Agent — Algorithms & Full Cycle

This document walks through **every stage** of the ingest → index → search →
serve cycle and the algorithms that power each stage. Use it alongside
[ARCHITECTURE.md](ARCHITECTURE.md), which covers the component wiring.

Legend: file references use workspace-relative paths.

---

## 1. Ingest — from a video path to a resumable job

Owner: [`indexer.index_video`](indexer.py) → `_index_video_impl`.

1. **Path resolution + row upsert.** The video is looked up by absolute path.
   If new, `videos` gets a row with the CCTV metadata; if it exists, its
   status is flipped to `indexing` and metadata is filled in (never
   overwritten with blanks).
2. **Profile selection.** `index_profile` chooses `extract_fps` and
   `scene_skip_enabled`. `frame_interval = max(1, video_fps / extract_fps)`.
3. **Resume checkpoint.** `resume_from = max(MAX(frame_number in frames),
   videos.last_frame_number) + frame_interval`. Using SQL as the ground truth
   means a crash in the middle of a mega-batch never loses more than one
   batch of work.
4. **Tracker id continuation.** `db.get_next_track_local_id(video_id)` returns
   `COALESCE(MAX(local_track_id),0)+1` so resumed jobs don't collide with
   already-persisted tracks.
5. **Background producer/consumer.** A daemon thread produces frames into a
   bounded `queue.Queue(maxsize=128)` while the main thread consumes them.
   The producer always enqueues a `None` sentinel in `finally` so the
   consumer can't deadlock on an extractor crash. Any producer exception is
   re-raised on the consumer side and surfaces as `videos.status='failed'`.

## 2. Frame extraction — accurate at low FPS

Owner: [`indexer.extract_frames`](indexer.py).

- Uses `cap.set(CAP_PROP_POS_FRAMES, n)` to jump directly to the target frame.
- After each read, checks the delta between requested and actual position; if
  drift exceeds 5 frames (common on variable-GOP h.264 with B-frames), the
  loop **falls back to sequential `cap.grab()`** and re-opens the capture to
  re-establish alignment. This keeps 1 FPS sampling honest on long CCTV files.
- Yields `(frame_number, timestamp, frame_bgr)` at the profile FPS.

## 3. Scene-skip — profile-aware, off by default

Owner: `_frame_hash` and `_frames_similar` in [`indexer.py`](indexer.py).

- Only enabled for `fast_preview`. Computes a 16×16 grayscale L2-normalised
  vector and cosine-compares against the previous kept frame. Similarity >
  0.92 → drop the frame.
- **Deliberately off** for `investigation` and `high_sensitivity`: on
  static-background CCTV, a small moving person doesn't perturb a 16×16 hash
  enough to be considered "new", so scene skip silently drops the very
  evidence you're trying to find.

## 4. Mega-batch flush — the hot path

Owner: [`indexer._flush_mega_batch`](indexer.py) with the shared
[`indexer._detect_track_crop_batch`](indexer.py) helper.

```
mega_batch = [ {frame_no, timestamp, frame_bgr_ref, frame_path,
                detections, crop_images, crop_records}, … ]  (size = 64)
```

### 4.1 Detection + tracking + crop persistence (`_detect_track_crop_batch`)

- `models.detect_objects_batch(frames)` runs YOLO11n once on the full batch
  with `stream=True` and FP16.
- Per frame, detections are fed to `SimpleIOUTracker.update(frame_no, ts,
  dets)` (see §5). The tracker returns one assignment per detection.
- For each detection: crop with 10 % padding, save the JPEG asynchronously,
  append the record `{class, conf, bbox, crop_path, track_local_id,
  track_iou, track_is_new}` to `crop_records`.

### 4.2 Bulk CLIP encoding

- Collect all PIL images: for each frame `{frame_pil, crop_0, crop_1, …}`.
- Convert BGR → RGB lazily so we don't hold two full-frame copies in RAM.
- Encode in one call `models.encode_images_clip` (batch size = 16 inside CLIP)
  with autocast; embeddings are L2-normalised, so FAISS's inner-product
  metric equals cosine similarity.
- Reserve monotonic ids from the general FAISS via `_reserve_ids`, then
  `add_with_ids(embeddings, ids)`. Frame vs. crop mapping is preserved so
  each row knows its own FAISS id before hitting the DB.

### 4.3 DINOv2 person Re-ID (body appearance)

- Filter `crop_records` to `class_name == "person"`.
- `models.encode_person_reid` resizes to 224×224 (multiple of DINOv2 patch
  size 14), applies ImageNet normalisation, forwards through DINOv2 ViT-S/14,
  L2-normalises the output.
- Vectors go into the **person** FAISS index (FlatIP+IDMap2 — flat because
  the counts stay small enough that HNSW's incremental-insert graph
  corruption on tiny batches is a net loss).

### 4.4 DB persistence

Order matters — every write is one transaction:

1. `batch_add_frames` → returns frame ids.
2. `batch_add_frame_embeddings` → one row per frame with the general FAISS id.
3. `batch_add_detections` → returns detection ids; each row already has the
   general FAISS id and, for persons, `person_faiss_id`.
4. `batch_update_person_faiss_ids` → updates the person_faiss_id column
   using the detection ids returned above.
5. `batch_link_track_detections` → for every detection with a
   `track_local_id`, upserts the `tracks` row (updating end frame, count,
   best detection) and inserts a `track_detections` timeline row. Also
   updates `detections.track_id` so downstream searches can filter by track.

### 4.5 Face detection + FaceNet identity

- For each person crop, run YOLO-Face on the region.
- Face bbox is converted to frame-absolute coordinates and re-cropped with
  15 % padding.
- `models.encode_faces_batch` resizes to 160×160, applies FaceNet's
  standardisation (mean=0.5, std=0.5), forwards through
  InceptionResnetV1(vggface2), L2-normalises.
- Face vectors are added to the **face** FAISS index; `faces` rows link back
  to the parent detection.

### 4.6 Checkpointing

- After each mega-batch: `db.update_video_checkpoint(last_frame_no,
  processed)`.
- After every 5 mega-batches: atomic FAISS save on all three indexes.
- Because writes are `.tmp → os.replace()`, an OS-level crash can leave the
  old file or the new file, never a half-written one.

## 5. Tracking — `SimpleIOUTracker`

Owner: [`tracker.SimpleIOUTracker`](tracker.py).

Per frame, given the current YOLO detections:

1. Split detections and active tracks by `class_name`.
2. For each class, compute IoU between every active track's last bbox and
   every detection bbox; keep pairs with IoU ≥ `TRACK_IOU_THRESHOLD` (default
   0.25 — low enough for 1 FPS sampling).
3. Greedy assignment sorted by IoU descending: highest-IoU pair first, remove
   both from the unmatched sets, repeat.
4. Unmatched detections become **new tracks** with a fresh monotonic
   `local_track_id`.
5. Unmatched active tracks age (`missed += 1`); tracks with
   `missed > TRACK_MAX_MISSED` (default 5) are retired.
6. Track state stored per active id: `bbox, first/last frame_no, first/last
   timestamp, hits, missed, best_confidence`.

The output of `update()` is one assignment per input detection
(`track_local_id`, `track_iou`, `track_is_new`, `track_hits`,
`track_class_name`) that the indexer merges into `crop_records`.

## 6. Text search — pipeline in [`searcher.search`](searcher.py)

```
query text
  │
  ▼
Stage 1  CLIP recall
   • _get_query_embedding: negative-prompt split → prompt-template ensemble
     → attribute blending
   • faiss.search on general index (cached), pool =
     max(top_k × 8, 100), floor score 0.20
   • Also: FTS5 recall on detections_fts and frame_embeddings_fts
     (_build_fts_query sanitizes user text)
  │
  ▼
Stage 2  YOLO-World re-rank
   • _rerank_with_yolo_world sets classes = [full query, core noun]
   • Runs on the full frame (not the crop), keeps small scene objects
   • Detected → score += conf × 0.8 (exact) or 0.3 (fallback noun)
   • Missed  → score ×= 0.35 (soft demote, not hard filter)
  │
  ▼
Stage 3  Temporal continuity boost
   • Group by video, sort by timestamp
   • Within ±3 s of another hit → score × 1.20
   • Isolated                    → score × 0.90
  │
  ▼
Stage 4  Near-duplicate clustering
   • Same video, |Δframe| ≤ 2 → collapse to best-scoring representative
   • Losers stored in _cluster; UI renders a `+N` badge
```

### 6.1 Query encoding details

- **Negative prompts.** `"person with jacket --no hat"` splits on `--no`.
  Both sides go through the prompt ensemble, then final =
  `pos - 0.5 · neg`, re-normalised.
- **Prompt ensemble.** Same query is templated into
  `"{q}" / "a photo of {q}" / "a picture of {q}" / "an image of {q}" / "a
  photo of {q} in a scene"`; embeddings are averaged and re-normalised.
  Empirically +2–5 pts recall.
- **Attribute blending.** `_parse_query_attributes` splits into
  `(attributes, core_noun)`. Final embedding =
  `0.6 · encode(full_query) + 0.4 · encode(attributes)`, re-normalised. This
  is what keeps `"black jacket"` from drifting into "any jacket".
- **BM25 normalisation.** FTS5 rank is negative; convert to
  `1 / (1 + |rank|)` so text scores and cosine similarities live in the same
  numeric range.

### 6.2 FTS5 sanitization

`_build_fts_query`:

1. Reject NUL and other C0 control bytes (SQLite raises
   `unterminated string` when it sees `\x00` in a bound parameter).
2. Strip FTS5 syntax characters `" ' ( ) * : ^ - + ~ [ ] { } \`.
3. Discard reserved operator words (`AND / OR / NOT / NEAR`, case-insensitive).
4. Double-quote each surviving token — any remaining character inside the
   quotes is a **literal** to FTS5, so SQL injection / operator abuse is
   physically impossible from user text.
5. Fallback: `""` when nothing survives, so callers never crash on empty
   input.

### 6.3 FAISS cache

`_get_cached_faiss(path)`:

- Cache key = `(mtime_ns, size)`.
- If the file was modified (indexer just checkpointed), the cache key changes
  and the vectors are re-read.
- If the file was deleted (cascade delete), returns `None` and drops the
  cache entry.
- Thread-safe via `_FAISS_CACHE_LOCK`; the actual `faiss.read_index` call is
  deliberately outside the lock so long reads on a big HNSW don't stall
  every request.
- Measured impact on the production 57 739-vector index: cold 0.27 s,
  warm ~0 s.

## 7. Image search — [`searcher.search_by_image`](searcher.py)

1. **Object-centric crop.** A CPU-resident YOLO detects the largest object in
   the uploaded image and crops the background off. If no object is found,
   the full image is used. This keeps a messy background from polluting the
   query embedding.
2. Encode with CLIP → search cached general index, pool = `max(top_k × 8,
   100)`, floor 0.15.
3. **Color re-rank.** `_rerank_by_color` computes an HSV histogram (Hue+Sat,
   16 bins each) of the query and each candidate, blends with
   `final = 0.6·clip_score + 0.4·color_sim·max_clip_score`. Fixes CLIP's
   colour blindness on "red bag" vs. "black bag" style queries.
4. Temporal boost + near-duplicate clustering (same as text search).

## 8. Person search — [`searcher.search_by_person`](searcher.py)

Purpose: identity, not clothing.

1. **Face gate.** YOLO-Face on the uploaded image.
   - If a face fills ≥ 40 % of the image area → treat as a *tight face crop*
     and **skip DINOv2** (running body Re-ID on a face zoomed in produces
     wrong clothing matches).
   - If any face detected → run FaceNet on that bbox.
2. **Stage 1: body via DINOv2.** Cached person index → pool = `max(top_k ×
   20, 500)`, keep only cosine ≥ `PERSON_MATCH_THRESHOLD` (0.72).
   Body-only scores are multiplied by `BODY_ONLY_DISCOUNT` (0.6) so they
   can never outrank face matches.
3. **Stage 2: face via FaceNet.** Cached face index → pool = `max(top_k ×
   20, 500)`, keep only cosine ≥ `FACE_MATCH_THRESHOLD` (0.50).
4. **Merge.** If a detection appears in both stages:
   - `merged = 0.3 × body_score + 0.7 × face_score`
   - `match_type = "body+face"`.
   Face-only matches get `face_score × 0.9` and `match_type = "face"`.
   Body-only matches keep `match_type = "body"`.
5. Sort → temporal boost → near-duplicate cluster.

The UI shows a coloured badge per `match_type` so investigators can trust
face matches over clothing matches at a glance.

## 9. Vehicle search — `POST /api/search-by-vehicle`

Owner: [`app.api_search_by_vehicle`](app.py).

- Reuses `searcher.search_by_image` with an inflated pool
  (`max(search_top_k, 100)`).
- Post-filters `class_name in VEHICLE_CLASSES` (car, truck, bus, motorcycle,
  bicycle, train, boat).
- `group_tracks=True` by default → all detections of the same vehicle track
  collapse to a single evidence card with a duration/hit count.

## 10. Filtering & grouping — [`app._filter_results`](app.py)

Applied *after* the searcher returns, so all endpoints share one
implementation:

- **Filters**: `video_id`, `camera_id`, `class_name`, `start_time`,
  `end_time`, `min_score`, `match_type`.
- **Track grouping**: `_group_results_by_track` collapses all results with
  the same `track_id` into one card, keeping the best score, aggregating
  timestamps (`track_start_timestamp`, `track_end_timestamp`), counting
  hits (`detection_count`), and stashing losers in `_cluster` for the UI.
- **Recall expansion**: `_expanded_top_k(top_k, *filters)` multiplies the
  search pool by 5× when any filter or grouping flag is truthy — otherwise
  post-filtering could shrink a 20-result response to 3.

## 11. Track browsing & evidence export

- `GET /api/videos/{id}/tracks` — [`db.get_tracks_for_video`](database.py)
  joins `tracks` to its `best_detection_id` + frame path + video row,
  ordered by `(detection_count DESC, best_confidence DESC, start_timestamp
  ASC)`.
- `GET /api/tracks/{id}` — [`db.get_track_timeline`](database.py) returns
  every `track_detections` row joined to its crop and frame, ordered by
  `timestamp`.
- `GET /api/tracks/{id}/export?format=csv|json` — same timeline serialised
  as either JSON or a flat CSV suitable for evidence packages.

## 12. Cascade delete — [`indexer.delete_video_fully`](indexer.py)

Order is deliberate:

1. `db.get_faiss_ids_for_video(video_id)` reads all `faiss_id /
   person_faiss_id / face_faiss_id` referenced by the video (before rows
   disappear).
2. For each index, `_remove_ids_from_index` tries native `remove_ids`. HNSW
   doesn't support this, so on `RuntimeError` we fall back to a full
   reconstruct-and-rebuild of the *survivors*, preserving the monotonic
   next-id counter so future adds don't collide.
3. Save each FAISS index atomically → sidecar refresh.
4. `db.delete_video` cascades through the FK graph
   (`frames → detections → track_detections | faces → face rows`).
5. `shutil.rmtree(FRAMES_DIR/<id>)`.
6. `VACUUM + ANALYZE` reclaims SQLite pages and refreshes planner stats.

`GET /api/index/health` afterwards should return `matches: true` for all
three indexes and `files.missing: 0`.

## 13. UI cycle — [`static/app.js`](static/app.js)

- **Tabs**: text / image / person / vehicle. Each tab has its own drop-zone,
  preview, and clear button.
- **Blob URL hygiene**: `revokePreviewUrl(imgEl)` is called before every
  `URL.createObjectURL(file)` assignment and on every clear button. Exactly
  three call sites for `createObjectURL` (one per file-upload tab) —
  guaranteed by grep so no future tab accidentally leaks blobs.
- **Filter panel**: `appendFilterParams(params, {includeClass?})` centralises
  the query-string construction so text/image/person/vehicle all send the
  same CCTV filters.
- **Load more**: increments `currentTopK` by 30 and re-issues the last search
  with the same params.
- **Track badge**: any card with `track_id` gets a `Track #N` chip; the video
  list has a "Tracks" button that fetches `/api/videos/{id}/tracks` and
  renders them as normal result cards.
- **Timestamp formatting**: `formatTime` returns `H:MM:SS` for videos ≥ 1
  hour, `M:SS` otherwise — long-CCTV friendly.

## 14. Health & integrity

- `/api/health` — no model calls, safe for load balancers.
- `/api/index/health` — compares DB vector references vs. FAISS sidecar
  counts vs. on-disk files:
  - `matches: true` when `db_refs == count` for each of general/person/face.
  - `files.missing` lists frames/crops referenced in DB but not present on
    disk. This is the check to run before trusting a long investigation.

## 15. Failure taxonomy (what happens when…)

| Failure mode | Effect | Recovery |
|--------------|--------|----------|
| Video codec cannot seek | Extractor drift detected → sequential fallback | Automatic |
| Model download fails on first start | Startup lifespan raises; server 500 | Fix network, restart |
| Indexer crashes mid-video | `videos.status='failed'` + `error_message` | `POST /api/videos/{id}/reindex` |
| User cancels | `videos.status='paused'` + checkpoint intact | Next `POST /api/index` on same path resumes |
| FAISS file corrupted | `_load_or_create_index` catches → rebuilds empty | Reindex affected videos |
| FAISS legacy IndexFlatIP file | Auto-migrated to IDMap2 on first read | Automatic |
| FAISS dim mismatch | Discard + rebuild empty | Reindex |
| SQLite WAL grows unbounded | Not automatic; run `vacuum_analyze` manually | `db.vacuum_analyze()` |
| User query breaks FTS5 | Sanitizer replaces syntax with literals | Automatic |
| Concurrent search during indexing | Cache mtime-invalidates after each checkpoint | Automatic |
| Deleted video's vectors leak | `delete_video_fully` removes ids from all 3 indexes | Automatic |

---

**TL;DR of the full cycle**: a video is chunked at profile FPS, each frame
yields YOLO crops that get tracked by IoU, embedded three ways (CLIP for
general, DINOv2 for body, FaceNet for face), and persisted alongside track
timelines to SQLite + three FAISS indexes with mtime-cached lookups. Search
is CLIP + FTS5 recall → YOLO-World rerank → temporal + duplicate → CCTV
filters → optional track grouping, all served over FastAPI with a static
JS/HTML frontend that treats every hit as an evidence card.
