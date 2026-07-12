"""Video indexing — FAST MODE (optimized).

Approach: "Index Fast, Search Smart"
- Index time: ONLY CLIP + YOLO crops. No VLM.
- Smart search happens at query time (see searcher.py).

Optimizations applied:
1. Frame seeking — jump to target frames instead of reading every frame
2. Stream processing — process frames on-the-fly, no RAM accumulation
3. Mega-batch CLIP — accumulate images across multiple frames for bulk encoding
4. Threaded disk I/O — fire-and-forget image writes
5. Batch SQLite inserts — single transaction per mega-batch
"""
import os
import json
import time
import cv2
import numpy as np
import faiss
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

import config
import database as db
import models
import tracker as tracking


# Global indexing state
indexing_status = {
    "active": False,
    "video": "",
    "progress": 0.0,
    "message": "",
    "frames_processed": 0,
    "total_frames": 0,
    "cancel_requested": False,
}

# Scene change detection threshold
SCENE_CHANGE_THRESHOLD = 0.92

# Persist FAISS to disk every N mega-batches so a crash doesn't lose everything.
FAISS_CHECKPOINT_EVERY = 5


class IndexingCancelled(Exception):
    """Raised when the user cancels an in-progress indexing job."""


def _fmt_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s}s"
    else:
        h, rem = divmod(int(seconds), 3600)
        m, s = divmod(rem, 60)
        return f"{h}h {m}m {s}s"


def _index_meta_path(index_path: str) -> str:
    return index_path + ".meta.json"


# FAISS wrappers disallow arbitrary attribute assignment, so we keep the per-index
# monotonic id counter in this module-level dict keyed by python id(index).
_INDEX_NEXT_IDS: dict[int, int] = {}


def _get_next_id(index: faiss.Index) -> int:
    return _INDEX_NEXT_IDS.get(id(index), index.ntotal)


def _set_next_id(index: faiss.Index, value: int) -> None:
    _INDEX_NEXT_IDS[id(index)] = int(value)


def _make_index(dim: int, hnsw_m: int = 32) -> faiss.Index:
    """Build a fresh IDMap2(HNSW) index with inner-product metric."""
    hnsw = faiss.IndexHNSWFlat(dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = 200
    hnsw.hnsw.efSearch = 64
    idx = faiss.IndexIDMap2(hnsw)
    _set_next_id(idx, 0)
    return idx


def _make_flat_index(dim: int) -> faiss.Index:
    """Build a FlatIP+IDMap2 index — best for small indexes (< 10K vectors)
    like face/person where HNSW incremental inserts can corrupt the graph."""
    flat = faiss.IndexFlatIP(dim)
    idx = faiss.IndexIDMap2(flat)
    _set_next_id(idx, 0)
    return idx


def _load_or_create_index(path: str, dim: int, use_flat: bool = False) -> faiss.Index:
    """Load ``path`` if present, migrating legacy indexes as needed.

    Args:
        use_flat: If True, use FlatIP+IDMap2 (ideal for small indexes < 10K
                  like face/person where HNSW can corrupt on incremental adds).

    Handles:
    - Missing file → new index (HNSW or Flat depending on use_flat).
    - Dimension mismatch → discard + rebuild empty.
    - Legacy IndexFlatIP → reconstruct all vectors and re-add with monotonic ids.
    - Corrupt file → log + rebuild empty.
    """
    _new = _make_flat_index if use_flat else _make_index

    if not os.path.exists(path):
        return _new(dim)
    try:
        idx = faiss.read_index(path)
    except Exception as e:
        print(f"[indexer] failed to read {path}: {e}; rebuilding empty index.")
        return _new(dim)

    if idx.d != dim:
        print(f"[indexer] dim mismatch in {path} ({idx.d} != {dim}); rebuilding empty index.")
        try:
            os.remove(path)
        except OSError:
            pass
        return _new(dim)

    if not isinstance(idx, faiss.IndexIDMap2):
        # Legacy flat index — migrate.
        print(f"[indexer] migrating legacy flat index {path} → IDMap2 (ntotal={idx.ntotal})")
        new_idx = _new(dim)
        if idx.ntotal > 0:
            try:
                vecs = np.vstack([idx.reconstruct(i) for i in range(idx.ntotal)]).astype("float32")
                ids = np.arange(idx.ntotal, dtype=np.int64)
                new_idx.add_with_ids(vecs, ids)
                _set_next_id(new_idx, int(idx.ntotal))
            except Exception as e:
                print(f"[indexer] migration failed on {path}: {e}; starting empty.")
        return new_idx

    # Existing IDMap2 index — restore monotonic id counter from sidecar.
    meta_path = _index_meta_path(path)
    next_id = 0
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                next_id = int(json.load(f).get("next_id", 0))
        except Exception:
            next_id = 0
    if next_id <= 0 and idx.ntotal > 0:
        try:
            id_map = faiss.vector_to_array(idx.id_map)
            next_id = int(id_map.max()) + 1 if len(id_map) > 0 else 0
        except Exception:
            next_id = int(idx.ntotal)
    _set_next_id(idx, next_id)
    return idx


def _save_index_atomic(index: faiss.Index, path: str) -> None:
    """Atomically write index + sidecar meta. Safe against mid-write crashes."""
    tmp = path + ".tmp"
    faiss.write_index(index, tmp)
    os.replace(tmp, path)
    meta = {
        "dim": int(index.d),
        "count": int(index.ntotal),
        "next_id": _get_next_id(index),
        "saved_at": time.time(),
    }
    meta_tmp = _index_meta_path(path) + ".tmp"
    with open(meta_tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f)
    os.replace(meta_tmp, _index_meta_path(path))


def _reserve_ids(index: faiss.Index, n: int) -> np.ndarray:
    """Reserve `n` monotonically-increasing ids from the index counter.

    Guarantees ids are unique across the index's lifetime, even after remove_ids.
    """
    start = _get_next_id(index)
    ids = np.arange(start, start + n, dtype=np.int64)
    _set_next_id(index, start + n)
    return ids


def _load_faiss_index():
    """Load or create the general CLIP FAISS index."""
    return _load_or_create_index(config.FAISS_INDEX_PATH, config.CLIP_DIM)


def _save_faiss_index(index):
    _save_index_atomic(index, config.FAISS_INDEX_PATH)


def _frame_hash(frame_bgr: np.ndarray, size: int = 16) -> np.ndarray:
    """Fast perceptual hash for scene change detection."""
    small = cv2.resize(frame_bgr, (size, size))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return gray.flatten() / 255.0


def _frames_similar(hash1, hash2) -> bool:
    if hash1 is None or hash2 is None:
        return False
    sim = np.dot(hash1, hash2) / (np.linalg.norm(hash1) * np.linalg.norm(hash2) + 1e-8)
    return sim > SCENE_CHANGE_THRESHOLD


def extract_frames(video_path: str, fps: float = None, start_frame: int = 0):
    """Generator: yields (frame_number, timestamp, frame_bgr) at configured FPS.

    Uses frame seeking (CAP_PROP_POS_FRAMES) to jump directly to target frames.
    If the codec drifts on seek (common with variable-GOP h.264/B-frames on long
    videos), falls back to sequential ``cap.grab`` so we stay aligned.
    Accepts ``start_frame`` to support resume-from-crash.
    """
    if fps is None:
        fps = config.EXTRACT_FPS
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(video_fps / fps))

    frame_no = start_frame
    seek_ok = True

    # If resuming, position the cursor once via seek so the first read is aligned.
    if frame_no > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)

    while frame_no < total_frames:
        if seek_ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()
            if not ret:
                break
            # OpenCV advances POS_FRAMES after read; subtract 1 to get read position.
            actual = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            if abs(actual - frame_no) > 5:
                print(f"[indexer] seek drift {actual - frame_no} at frame {frame_no}; "
                      "falling back to sequential reads.")
                seek_ok = False
                # Re-position by re-opening and grabbing frames sequentially.
                cap.release()
                cap = cv2.VideoCapture(video_path)
                for _ in range(frame_no):
                    if not cap.grab():
                        cap.release()
                        return
                ret, frame = cap.read()
                if not ret:
                    break
            yield frame_no, frame_no / video_fps, frame
            frame_no += frame_interval
        else:
            # Sequential path: skip (frame_interval - 1) frames with grab(), then read one.
            for _ in range(frame_interval - 1):
                if not cap.grab():
                    cap.release()
                    return
            ret, frame = cap.read()
            if not ret:
                break
            yield frame_no, frame_no / video_fps, frame
            frame_no += frame_interval
    cap.release()


def crop_detection(frame_bgr: np.ndarray, bbox: tuple, pad_ratio: float = 0.1) -> Image.Image:
    """Crop a detection with padding, return as PIL Image."""
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    x1, y1 = max(0, x1 - int(bw * pad_ratio)), max(0, y1 - int(bh * pad_ratio))
    x2, y2 = min(w, x2 + int(bw * pad_ratio)), min(h, y2 + int(bh * pad_ratio))
    crop = cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
    return Image.fromarray(crop)


def _detect_track_crop_batch(mega_batch, object_tracker, io_executor, video_frames_dir):
    """Run YOLO + IoU tracker + crop persistence for one mega-batch.

    Mutates each ``mb`` in ``mega_batch`` in place by setting ``detections``,
    ``crop_images`` and ``crop_records``. Returns the total number of YOLO
    detections produced so callers can update aggregate counters.
    Extracted from ``_index_video_impl`` so the main loop and the remainder
    flush share one implementation.
    """
    if not mega_batch:
        return 0

    bgr_frames = [mb["frame_bgr_ref"] for mb in mega_batch]
    batch_dets = models.detect_objects_batch(bgr_frames)

    total = 0
    for mb, dets in zip(mega_batch, batch_dets):
        mb["detections"] = dets
        total += len(dets)
        track_assignments = object_tracker.update(mb["frame_no"], mb["timestamp"], dets)
        for di, det in enumerate(dets):
            track_assignment = track_assignments[di] if di < len(track_assignments) else {}
            crop_img = crop_detection(mb["frame_bgr_ref"], det["bbox"])
            crop_filename = f"frame_{mb['frame_no']:06d}_det_{di}.jpg"
            crop_path = os.path.join(video_frames_dir, crop_filename)
            io_executor.submit(
                lambda img=crop_img, p=crop_path, q=config.CROP_JPEG_QUALITY:
                    img.save(p, format="JPEG", quality=q)
            )
            mb["crop_images"].append(crop_img)
            mb["crop_records"].append({
                "class_name": det["class_name"],
                "confidence": det["confidence"],
                "bbox": det["bbox"],
                "crop_path": crop_path,
                "track_local_id": track_assignment.get("track_local_id"),
                "track_iou": track_assignment.get("track_iou"),
                "track_is_new": track_assignment.get("track_is_new", False),
            })
    return total


def _flush_mega_batch(mega_batch, faiss_index, person_faiss_index, face_faiss_index,
                      use_clip, video_id, io_executor, video_frames_dir):
    """Process a mega-batch: bulk CLIP encode + batch DB inserts.
    
    Now also populates:
    - person_faiss_index: CLIP vectors for person crops only (for Re-ID)
    - face_faiss_index: CLIP vectors for face crops within person bboxes
    """
    if not mega_batch:
        return

    # --- 1. Batch-insert frames into DB (single transaction) ---
    frame_rows = [(video_id, mb["frame_no"], mb["timestamp"], mb["frame_path"]) for mb in mega_batch]
    frame_ids = db.batch_add_frames(frame_rows)
    for i, mb in enumerate(mega_batch):
        mb["frame_id"] = frame_ids[i]

    # --- 2. Bulk CLIP encode all images at once ---
    if use_clip:
        # Collect ALL images: one frame PIL + its crop PILs, for every frame in batch.
        # frame_pil is built lazily from frame_bgr_ref to avoid keeping two full copies
        # per frame in memory during long-video indexing.
        all_images = []
        image_map = []  # tracks (mega_batch_index, "frame"|"crop", crop_index_or_None)

        for bi, mb in enumerate(mega_batch):
            frame_pil = Image.fromarray(cv2.cvtColor(mb["frame_bgr_ref"], cv2.COLOR_BGR2RGB))
            all_images.append(frame_pil)
            image_map.append((bi, "frame", None))
            for ci, crop_img in enumerate(mb["crop_images"]):
                all_images.append(crop_img)
                image_map.append((bi, "crop", ci))

        if all_images:
            all_embeddings = models.encode_images_clip(all_images)

            # Batch-add with explicit monotonic ids (IDMap2 lets us remove later).
            new_ids = _reserve_ids(faiss_index, len(all_embeddings))
            faiss_index.add_with_ids(all_embeddings, new_ids)
            for emb_idx, (bi, kind, ci) in enumerate(image_map):
                fid = int(new_ids[emb_idx])
                if kind == "frame":
                    mega_batch[bi]["frame_faiss_id"] = fid
                else:
                    mega_batch[bi]["crop_records"][ci]["faiss_id"] = fid

        # --- 2b. DINOv2 Biometric Re-ID for Person Crops ---
        person_images = []
        person_map = []  # tracks (mb_idx, ci)
        for bi, mb in enumerate(mega_batch):
            for ci, rec in enumerate(mb["crop_records"]):
                if rec["class_name"] == "person":
                    # Use the PIL crop from the mega-batch
                    person_images.append(mb["crop_images"][ci])
                    person_map.append((bi, ci))
        
        if person_images:
            reid_embeddings = models.encode_person_reid(person_images)
            new_ids = _reserve_ids(person_faiss_index, len(person_images))
            person_faiss_index.add_with_ids(reid_embeddings, new_ids)
            for i, (bi, ci) in enumerate(person_map):
                mega_batch[bi]["crop_records"][ci]["person_faiss_id"] = int(new_ids[i])


    # --- 3. Batch-insert frame_embeddings (single transaction) ---
    emb_rows = [
        (mb["frame_id"], mb.get("frame_faiss_id"), "")
        for mb in mega_batch
    ]
    db.batch_add_frame_embeddings(emb_rows)

    # --- 4. Batch-insert detections (single transaction) ---
    det_rows = []
    for mb in mega_batch:
        for rec in mb["crop_records"]:
            x1, y1, x2, y2 = rec["bbox"]
            det_rows.append((
                mb["frame_id"], rec["class_name"], rec["confidence"],
                x1, y1, x2, y2, rec["crop_path"], rec.get("faiss_id"), rec["class_name"]
            ))
    det_ids = []
    if det_rows:
        det_ids = db.batch_add_detections(det_rows)

    # --- 5. Update person_faiss_id for person detections ---
    person_updates = []
    det_idx = 0
    for mb in mega_batch:
        for rec in mb["crop_records"]:
            if det_ids and det_idx < len(det_ids):
                rec["detection_id"] = det_ids[det_idx]
                if "person_faiss_id" in rec:
                    person_updates.append((rec["person_faiss_id"], det_ids[det_idx]))
            det_idx += 1
    if person_updates:
        db.batch_update_person_faiss_ids(person_updates)

    # --- 5b. Persist CCTV track assignments for each detection ---
    track_observations = []
    for mb in mega_batch:
        for rec in mb["crop_records"]:
            if rec.get("detection_id") is None or rec.get("track_local_id") is None:
                continue
            track_observations.append({
                "local_track_id": rec["track_local_id"],
                "class_name": rec["class_name"],
                "frame_id": mb["frame_id"],
                "detection_id": rec["detection_id"],
                "frame_number": mb["frame_no"],
                "timestamp": mb["timestamp"],
                "confidence": rec["confidence"],
                "bbox": rec["bbox"],
            })
    if track_observations:
        db.batch_link_track_detections(video_id, track_observations)

    # --- 6. Batch Face detection + FaceNet encode for person crops ---
    person_regions = []
    person_meta = []  # (bi, ci, px1, py1, px2, py2, det_id, frame_id, frame_no)

    # A person shorter than ~80 px or narrower than ~40 px cannot produce a face
    # detectable by YOLO-Face after it resizes the crop to 640² (the face itself
    # would be < 6 px). Skipping these person crops removes a large chunk of the
    # per-batch YOLO-Face cost with zero recall loss.
    min_pw = int(getattr(config, "FACE_MIN_PERSON_WIDTH", 0))
    min_ph = int(getattr(config, "FACE_MIN_PERSON_HEIGHT", 0))

    for bi, mb in enumerate(mega_batch):
        for ci, rec in enumerate(mb["crop_records"]):
            if rec["class_name"] != "person": continue
            px1, py1, px2, py2 = rec["bbox"]
            if (px2 - px1) < min_pw or (py2 - py1) < min_ph:
                continue
            if mb.get("frame_bgr_ref") is not None:
                region = mb["frame_bgr_ref"][py1:py2, px1:px2]
                if region.size > 0:
                    person_regions.append(region)
                    person_meta.append((bi, ci, px1, py1, px2, py2, rec.get("detection_id"), mb["frame_id"], mb["frame_no"]))

    if person_regions:
        batch_faces = models.detect_faces_batch(person_regions)
        
        face_crops_bgr = []
        face_meta = []
        for i, faces in enumerate(batch_faces):
            region = person_regions[i]
            bi, ci, px1, py1, px2, py2, det_id, fid, fno = person_meta[i]
            
            for fi, face in enumerate(faces):
                fx1, fy1, fx2, fy2 = face["bbox"]
                abs_fx1, abs_fy1 = px1 + fx1, py1 + fy1
                abs_fx2, abs_fy2 = px1 + fx2, py1 + fy2
                
                # Crop face from region (BGR) for FaceNet
                bw, bh = fx2 - fx1, fy2 - fy1
                pad = 0.15
                cx1, cy1 = max(0, fx1 - int(bw * pad)), max(0, fy1 - int(bh * pad))
                cx2, cy2 = min(region.shape[1], fx2 + int(bw * pad)), min(region.shape[0], fy2 + int(bh * pad))
                face_bgr = region[cy1:cy2, cx1:cx2].copy()
                
                if face_bgr.size == 0 or face_bgr.shape[0] < 20 or face_bgr.shape[1] < 20:
                    continue
                
                # Save face crop asynchronously
                face_filename = f"frame_{fno:06d}_det_{ci}_face_{fi}.jpg"
                face_path = os.path.join(video_frames_dir, face_filename)
                
                face_pil = Image.fromarray(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB))
                io_executor.submit(lambda img=face_pil, p=face_path, q=config.FACE_JPEG_QUALITY: img.save(p, format="JPEG", quality=q))
                
                face_crops_bgr.append(face_bgr)
                face_meta.append({
                    "detection_id": det_id, "frame_id": fid, "confidence": face["confidence"],
                    "bbox": (abs_fx1, abs_fy1, abs_fx2, abs_fy2), "crop_path": face_path
                })
                
        # --- 7. Batch Biometric Face Recognition (Identity) ---
        if face_crops_bgr:
            face_embeddings = models.encode_faces_batch(face_crops_bgr)
            valid_indices = [i for i, emb in enumerate(face_embeddings) if np.any(emb)]
            if valid_indices:
                valid_embs = np.vstack([face_embeddings[i] for i in valid_indices]).astype("float32")
                new_ids = _reserve_ids(face_faiss_index, len(valid_indices))
                face_faiss_index.add_with_ids(valid_embs, new_ids)
                face_rows = []
                for out_i, src_i in enumerate(valid_indices):
                    meta = face_meta[src_i]
                    ffid = int(new_ids[out_i])
                    x1, y1, x2, y2 = meta["bbox"]
                    face_rows.append((
                        meta["detection_id"], meta["frame_id"], ffid,
                        meta["confidence"], x1, y1, x2, y2, meta["crop_path"]
                    ))
                if face_rows:
                    db.batch_add_faces(face_rows)


def index_video(video_path: str, use_vlm: bool = False, use_clip: bool = True,
                camera_id: str = None, location: str = None, recording_start_time: str = None,
                timezone: str = None, index_profile: str = None,
                yolo_classes: list = None, yolo_confidence: float = None):
    """Public entry: index one video with crash-safe error handling.

    On any unhandled exception we mark the video as ``failed`` (with the
    exception message stored in ``videos.error_message``) so it can be
    diagnosed or reindexed later, and we reset ``indexing_status`` so the
    UI recovers.

    Per-job overrides:
        ``yolo_classes`` and ``yolo_confidence`` temporarily mutate the
        corresponding ``config`` module attributes for the lifetime of
        this call, then restore them in a ``finally`` block. This lets
        the caller (POST /api/index) change detection settings for the
        next job after a cancel, without restarting the process.
    """
    global indexing_status
    _pre_video_path = os.path.abspath(video_path)
    _existing = db.get_video_by_path(_pre_video_path)

    # --- per-job config overrides (restored in finally) ---
    _original_classes = config.YOLO_CLASSES
    _original_conf = config.YOLO_CONFIDENCE
    try:
        if yolo_classes is not None:
            config.YOLO_CLASSES = list(yolo_classes)
            print(f"[indexer] Override YOLO_CLASSES = {config.YOLO_CLASSES}")
        if yolo_confidence is not None:
            config.YOLO_CONFIDENCE = float(yolo_confidence)
            print(f"[indexer] Override YOLO_CONFIDENCE = {config.YOLO_CONFIDENCE}")

        try:
            return _index_video_impl(
                video_path,
                use_vlm=use_vlm,
                use_clip=use_clip,
                camera_id=camera_id,
                location=location,
                recording_start_time=recording_start_time,
                timezone=timezone,
                index_profile=index_profile,
            )
        except IndexingCancelled:
            # Not an error path \u2014 request_cancel() sets status via the impl.
            raise
        except Exception as e:
            import traceback
            traceback.print_exc()
            try:
                vid_row = db.get_video_by_path(_pre_video_path) or _existing
                if vid_row is not None:
                    db.set_video_error(int(vid_row["id"]), f"{type(e).__name__}: {e}")
            except Exception as inner:
                print(f"[indexer] failed to persist error state: {inner}")
            indexing_status.update({
                "active": False,
                "cancel_requested": False,
                "message": f"FAILED: {type(e).__name__}: {e}",
            })
            return
    finally:
        config.YOLO_CLASSES = _original_classes
        config.YOLO_CONFIDENCE = _original_conf


def _index_video_impl(video_path: str, use_vlm: bool = False, use_clip: bool = True,
                      camera_id: str = None, location: str = None, recording_start_time: str = None,
                      timezone: str = None, index_profile: str = None):
    """
    Index a video FAST \u2014 CLIP embeddings + YOLO crops only.

    Pipeline (optimized):
    1. Seek directly to target frames (skip intermediate frames)
    2. Stream frames \u2014 process on-the-fly instead of collecting all in RAM
    3. Accumulate MEGA_BATCH frames, then bulk CLIP encode + batch DB insert
    4. Disk I/O (frame/crop saves) offloaded to thread pool
    """
    global indexing_status

    video_path = os.path.abspath(video_path)
    filename = os.path.basename(video_path)

    existing = db.get_video_by_path(video_path)
    if existing and existing["status"] == "done":
        indexing_status["message"] = f"Already indexed: {filename}"
        return

    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames_raw / video_fps if video_fps > 0 else 0
    cap.release()

    active_profile = index_profile or config.INDEX_PROFILE
    if active_profile not in config.INDEX_PROFILES:
        print(f"[indexer] unknown index profile '{active_profile}', using investigation")
        active_profile = "investigation"
    profile_cfg = config.get_index_profile(active_profile)
    extract_fps = float(profile_cfg["extract_fps"])
    scene_skip_enabled = bool(profile_cfg["scene_skip_enabled"])

    # Estimate how many frames we'll actually process (before optional scene-change filtering)
    frame_interval = max(1, int(video_fps / extract_fps))
    estimated_frames = total_frames_raw // frame_interval if frame_interval > 0 else total_frames_raw

    if existing:
        video_id = existing["id"]
        db.update_video_status(video_id, "indexing")
        db.update_video_metadata(
            video_id,
            camera_id=camera_id,
            location=location,
            recording_start_time=recording_start_time,
            timezone=timezone,
            index_profile=active_profile,
        )
    else:
        video_id = db.add_video(
            filename,
            video_path,
            duration,
            video_fps,
            total_frames_raw,
            camera_id=camera_id,
            location=location,
            recording_start_time=recording_start_time,
            timezone=timezone,
            index_profile=active_profile,
        )

    # Resume from last checkpoint if this video was partially indexed. Use the max frame
    # already stored in DB as the ground truth (survives crashes better than the counter).
    resume_from = 0
    resume_processed = 0
    if existing:
        max_in_db = db.get_max_frame_number(video_id)
        checkpoint = existing["last_frame_number"] if "last_frame_number" in existing.keys() else 0
        checkpoint = int(checkpoint or 0)
        resume_from = max(max_in_db, checkpoint)
        if resume_from > 0:
            # Advance past the last-completed frame to the next scheduled one.
            resume_from += frame_interval
            resume_processed = int(existing["frames_processed"] or 0) if "frames_processed" in existing.keys() else 0
            print(f"[indexer] RESUME: skipping to frame {resume_from} (already have {resume_processed} frames)")

    video_frames_dir = os.path.join(config.FRAMES_DIR, str(video_id))
    os.makedirs(video_frames_dir, exist_ok=True)

    faiss_index = _load_faiss_index() if use_clip else None

    # Person + face FAISS indexes: FlatIP+IDMap2 (small enough for brute-force,
    # avoids HNSW graph corruption from incremental small-batch inserts).
    person_faiss_index = _load_or_create_index(config.PERSON_FAISS_INDEX_PATH, config.REID_DIM, use_flat=True) if use_clip else None
    face_faiss_index = _load_or_create_index(config.FACE_FAISS_INDEX_PATH, config.FACE_REC_DIM, use_flat=True) if use_clip else None

    print(f"\n{'='*60}")
    print(f"[indexer] START: {filename}")
    print(f"[indexer]   Video duration : {_fmt_time(duration)}")
    print(f"[indexer]   Video FPS      : {video_fps:.1f}")
    print(f"[indexer]   Total frames   : {total_frames_raw:,}")
    print(f"[indexer]   Extract at     : {extract_fps} FPS (1 frame every {frame_interval} frames)")
    print(f"[indexer]   Index profile  : {active_profile} (scene skip: {'on' if scene_skip_enabled else 'off'})")
    print(f"[indexer]   Est. frames    : ~{estimated_frames:,}")
    print(f"[indexer]   Mega-batch size: {config.CLIP_MEGA_BATCH}")
    print(f"{'='*60}")

    indexing_status.update({
        "active": True, "video": filename, "progress": 0.0,
        "message": f"Indexing {filename} (~{estimated_frames} frames)...",
        "frames_processed": resume_processed, "total_frames": estimated_frames,
        "cancel_requested": False,
    })

    # Thread pool for non-blocking disk writes
    io_executor = ThreadPoolExecutor(max_workers=4)

    # --- Stream + Mega-batch processing ---
    mega_batch = []
    prev_hash = None
    skipped = 0
    processed = resume_processed
    total_detections = 0
    track_start_id = db.get_next_track_local_id(video_id)
    object_tracker = tracking.SimpleIOUTracker(
        iou_threshold=config.TRACK_IOU_THRESHOLD,
        max_missed=config.TRACK_MAX_MISSED,
        start_local_id=track_start_id,
    )
    batches_since_checkpoint = 0
    cancelled = False
    t_start = time.time()
    
    import queue
    import threading

    frame_queue = queue.Queue(maxsize=128)
    extractor_error: list = []

    def _extractor():
        try:
            for fno, ts, fb in extract_frames(video_path, fps=extract_fps, start_frame=resume_from):
                frame_queue.put((fno, ts, fb))
        except Exception as exc:
            extractor_error.append(exc)
        finally:
            # Sentinel MUST be enqueued no matter what \u2014 otherwise the consumer
            # loop blocks forever on frame_queue.get().
            frame_queue.put(None)

    threading.Thread(target=_extractor, daemon=True).start()

    while True:
        # Honour cooperative cancellation from POST /api/index/cancel.
        if indexing_status.get("cancel_requested"):
            cancelled = True
            print("[indexer] cancel requested \u2014 stopping after current batch")
            break

        data = frame_queue.get()
        if data is None: break

        frame_no, timestamp, frame_bgr = data

        # Scene change detection. Disabled by default in CCTV investigation mode:
        # a static background can hide a small moving person/object from the
        # global 16x16 hash and cause missed evidence.
        if scene_skip_enabled:
            curr_hash = _frame_hash(frame_bgr)
            if _frames_similar(prev_hash, curr_hash):
                skipped += 1
                continue
            prev_hash = curr_hash

        # Save frame to disk (non-blocking)
        frame_path = os.path.join(video_frames_dir, f"frame_{frame_no:06d}.jpg")
        frame_copy = frame_bgr.copy()  # copy so original can be GC'd
        io_executor.submit(cv2.imwrite, frame_path, frame_copy, [cv2.IMWRITE_JPEG_QUALITY, config.FRAME_JPEG_QUALITY])

        mega_batch.append({
            "frame_no": frame_no,
            "timestamp": timestamp,
            "frame_bgr_ref": frame_bgr,
            "frame_path": frame_path,
            "detections": [],
            "crop_images": [],
            "crop_records": [],
            "frame_faiss_id": None,
        })

        # Process and flush mega-batch when full
        if len(mega_batch) >= config.CLIP_MEGA_BATCH:
            total_detections += _detect_track_crop_batch(
                mega_batch, object_tracker, io_executor, video_frames_dir
            )

            _flush_mega_batch(mega_batch, faiss_index, person_faiss_index, face_faiss_index,
                             use_clip, video_id, io_executor, video_frames_dir)

            processed += len(mega_batch)
            last_frame_no = mega_batch[-1]["frame_no"]

            # Free heavy per-frame references immediately so peak RAM is bounded by
            # a single mega-batch, not the whole video.
            for mb in mega_batch:
                mb["frame_bgr_ref"] = None
                mb["crop_images"] = []
            mega_batch = []

            # Persist checkpoint (DB + FAISS) so a crash won't waste this work.
            db.update_video_checkpoint(video_id, last_frame_no, processed)
            batches_since_checkpoint += 1
            if use_clip and batches_since_checkpoint >= FAISS_CHECKPOINT_EVERY:
                if faiss_index is not None:
                    _save_faiss_index(faiss_index)
                if person_faiss_index is not None:
                    _save_index_atomic(person_faiss_index, config.PERSON_FAISS_INDEX_PATH)
                if face_faiss_index is not None:
                    _save_index_atomic(face_faiss_index, config.FACE_FAISS_INDEX_PATH)
                batches_since_checkpoint = 0

            elapsed = time.time() - t_start
            fps_rate = processed / elapsed if elapsed > 0 else 0
            pct = processed / estimated_frames * 100 if estimated_frames > 0 else 0
            remaining = (estimated_frames - processed) / fps_rate if fps_rate > 0 else 0
            status_msg = (f"Frame {processed}/~{estimated_frames} ({pct:.0f}%) \u2014 "
                         f"{fps_rate:.2f} frames/s \u2014 ETA {_fmt_time(remaining)} \u2014 "
                         f"{skipped} skipped")
            print(f"[indexer] {status_msg}")
            indexing_status.update({
                "frames_processed": processed,
                "progress": pct,
                "message": status_msg,
            })

    # Flush remaining frames
    if mega_batch and not cancelled:
        total_detections += _detect_track_crop_batch(
            mega_batch, object_tracker, io_executor, video_frames_dir
        )

        _flush_mega_batch(mega_batch, faiss_index, person_faiss_index, face_faiss_index,
                         use_clip, video_id, io_executor, video_frames_dir)
        processed += len(mega_batch)
        last_frame_no = mega_batch[-1]["frame_no"]
        for mb in mega_batch:
            mb["frame_bgr_ref"] = None
            mb["crop_images"] = []
        mega_batch = []
        db.update_video_checkpoint(video_id, last_frame_no, processed)

    # Wait for all disk writes to finish
    io_executor.shutdown(wait=True)

    # Surface any extractor thread error \u2014 the outer index_video() wrapper turns
    # this into videos.status='failed' + error_message.
    if extractor_error and not cancelled:
        raise extractor_error[0]

    if use_clip and faiss_index is not None:
        _save_faiss_index(faiss_index)
    if use_clip and person_faiss_index is not None:
        _save_index_atomic(person_faiss_index, config.PERSON_FAISS_INDEX_PATH)
    if use_clip and face_faiss_index is not None:
        _save_index_atomic(face_faiss_index, config.FACE_FAISS_INDEX_PATH)

    elapsed_total = time.time() - t_start
    final_status = "paused" if cancelled else "done"
    db.update_video_status(video_id, final_status)
    if cancelled:
        # Leave last_frame_number/frames_processed in DB so next run resumes here.
        indexing_status.update({
            "active": False,
            "cancel_requested": False,
            "message": f"Cancelled at frame {processed} of {estimated_frames} ({filename}). Resumable.",
            "frames_processed": processed,
        })
        print(f"[indexer] CANCELLED: {filename} at frame {processed}")
        return

    # --- Final summary ---
    speed_ratio = duration / elapsed_total if elapsed_total > 0 else 0
    fps_final = processed / elapsed_total if elapsed_total > 0 else 0

    person_count = person_faiss_index.ntotal if person_faiss_index else 0
    face_count = face_faiss_index.ntotal if face_faiss_index else 0
    track_stats = db.get_track_stats(video_id)

    print(f"\n{'='*60}")
    print(f"[indexer] DONE: {filename}")
    print(f"[indexer]   Video duration    : {_fmt_time(duration)}")
    print(f"[indexer]   Indexing time     : {_fmt_time(elapsed_total)}")
    print(f"[indexer]   Speed ratio       : {speed_ratio:.1f}x (1 min video ≈ {60/speed_ratio:.0f}s indexing)" if speed_ratio > 0 else "")
    print(f"[indexer]   Frames indexed    : {processed:,}")
    print(f"[indexer]   Frames skipped    : {skipped:,} (scene duplicates)")
    print(f"[indexer]   Objects detected  : {total_detections:,}")
    print(f"[indexer]   Tracks created    : {track_stats['tracks']:,}")
    print(f"[indexer]   Person vectors    : {person_count:,} (Re-ID index)")
    print(f"[indexer]   Face vectors      : {face_count:,} (Face index)")
    print(f"[indexer]   Avg speed         : {fps_final:.2f} frames/s")
    print(f"[indexer]   ---")
    print(f"[indexer]   Projection: 10 min video ≈ {_fmt_time(600/speed_ratio)}" if speed_ratio > 0 else "")
    print(f"[indexer]   Projection: 1 hr video  ≈ {_fmt_time(3600/speed_ratio)}" if speed_ratio > 0 else "")
    print(f"[indexer]   Projection: 2 hr video  ≈ {_fmt_time(7200/speed_ratio)}" if speed_ratio > 0 else "")
    print(f"{'='*60}\n")

    done_msg = (f"Done! {processed} frames indexed ({skipped} skipped) "
                f"from {filename} in {_fmt_time(elapsed_total)} "
                f"({speed_ratio:.1f}x realtime)")
    indexing_status.update({
        "active": False, "progress": 100.0,
        "message": done_msg,
        "frames_processed": processed,
    })


def index_directory(dir_path: str, use_vlm: bool = False, use_clip: bool = True,
                    camera_id: str = None, location: str = None, recording_start_time: str = None,
                    timezone: str = None, index_profile: str = None,
                    yolo_classes: list = None, yolo_confidence: float = None):
    """Recursively index all video files in a directory."""
    exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}
    videos = [os.path.join(r, f) for r, _, fs in os.walk(dir_path)
              for f in fs if os.path.splitext(f)[1].lower() in exts]
    for i, vp in enumerate(videos):
        indexing_status["message"] = f"Video {i + 1}/{len(videos)}: {os.path.basename(vp)}"
        index_video(
            vp,
            use_vlm=use_vlm,
            use_clip=use_clip,
            camera_id=camera_id,
            location=location,
            recording_start_time=recording_start_time,
            timezone=timezone,
            index_profile=index_profile,
            yolo_classes=yolo_classes,
            yolo_confidence=yolo_confidence,
        )


def _remove_ids_from_index(path: str, ids: list, dim: int, use_flat: bool = False) -> int:
    """Remove FAISS ids from the index at ``path``. Returns removed count.

    HNSW does not implement ``remove_ids`` natively; when that raises RuntimeError
    we fall back to a full reconstruct + rebuild of the survivors. This is O(N) but
    only runs on video deletion (rare), and preserves the outer id mapping.
    """
    if not ids or not os.path.exists(path):
        return 0
    index = _load_or_create_index(path, dim, use_flat=use_flat)
    id_set = set(int(i) for i in ids)
    try:
        arr = np.asarray(list(id_set), dtype=np.int64)
        removed = int(index.remove_ids(faiss.IDSelectorArray(arr)))
        _save_index_atomic(index, path)
        return removed
    except RuntimeError as e:
        print(f"[indexer] native remove_ids unsupported ({e}); rebuilding index for removal.")

    # Rebuild path: reconstruct all survivors, add to a fresh index.
    try:
        id_map = faiss.vector_to_array(index.id_map)  # type: ignore[attr-defined]
    except Exception as e:
        print(f"[indexer] cannot read id_map from {path}: {e}")
        return 0

    survivors_ids = [int(i) for i in id_map if int(i) not in id_set]
    total_before = int(index.ntotal)
    if len(survivors_ids) == total_before:
        return 0

    new_index = _make_flat_index(dim) if use_flat else _make_index(dim)
    if survivors_ids:
        vecs = np.vstack([index.reconstruct(int(i)) for i in survivors_ids]).astype("float32")
        new_index.add_with_ids(vecs, np.asarray(survivors_ids, dtype=np.int64))
    # Preserve monotonic counter so future adds keep unique ids.
    _set_next_id(new_index, max(_get_next_id(index), (max(survivors_ids) + 1) if survivors_ids else 0))
    _save_index_atomic(new_index, path)
    return total_before - len(survivors_ids)


def delete_video_fully(video_id: int) -> dict:
    """Cascade-delete: remove FAISS vectors, DB rows, and on-disk frames for ``video_id``.

    Returns a small stats dict describing what was removed. Safe to call on
    partially-indexed videos (missing ids are ignored).
    """
    import shutil
    ids = db.get_faiss_ids_for_video(video_id)

    removed_general = _remove_ids_from_index(config.FAISS_INDEX_PATH, ids["general"], config.CLIP_DIM)
    removed_person = _remove_ids_from_index(config.PERSON_FAISS_INDEX_PATH, ids["person"], config.REID_DIM, use_flat=True)
    removed_face = _remove_ids_from_index(config.FACE_FAISS_INDEX_PATH, ids["face"], config.FACE_REC_DIM, use_flat=True)

    # ON DELETE CASCADE in schema removes frames \u2192 detections \u2192 faces \u2192 frame_embeddings.
    db.delete_video(video_id)

    frames_dir = os.path.join(config.FRAMES_DIR, str(video_id))
    frames_removed = False
    if os.path.isdir(frames_dir):
        try:
            shutil.rmtree(frames_dir)
            frames_removed = True
        except OSError as e:
            print(f"[indexer] failed to remove {frames_dir}: {e}")

    # Reclaim SQLite pages so the file shrinks after large videos are deleted.
    db.vacuum_analyze()

    return {
        "video_id": video_id,
        "faiss_removed": {
            "general": removed_general,
            "person": removed_person,
            "face": removed_face,
        },
        "frames_dir_removed": frames_removed,
    }


def request_cancel() -> bool:
    """Signal the running indexing loop to stop after the current mega-batch."""
    if not indexing_status.get("active"):
        return False
    indexing_status["cancel_requested"] = True
    return True
