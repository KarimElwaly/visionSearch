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
import time
import cv2
import numpy as np
import faiss
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

import config
import database as db
import models


# Global indexing state
indexing_status = {
    "active": False,
    "video": "",
    "progress": 0.0,
    "message": "",
    "frames_processed": 0,
    "total_frames": 0,
}

# Scene change detection threshold
SCENE_CHANGE_THRESHOLD = 0.92


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


def _load_faiss_index():
    """Load or create FAISS index for CLIP embeddings."""
    dim = config.CLIP_DIM
    if os.path.exists(config.FAISS_INDEX_PATH):
        return faiss.read_index(config.FAISS_INDEX_PATH)
    return faiss.IndexFlatIP(dim)


def _save_faiss_index(index):
    faiss.write_index(index, config.FAISS_INDEX_PATH)


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


def extract_frames(video_path: str, fps: float = None):
    """Generator: yields (frame_number, timestamp, frame_bgr) at configured FPS.
    
    Uses frame seeking (CAP_PROP_POS_FRAMES) to jump directly to target frames
    instead of reading and discarding every intermediate frame.
    """
    if fps is None:
        fps = config.EXTRACT_FPS
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(video_fps / fps))

    frame_no = 0
    while frame_no < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
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
        # Collect ALL images: one frame PIL + its crop PILs, for every frame in batch
        all_images = []
        image_map = []  # tracks (mega_batch_index, "frame"|"crop", crop_index_or_None)

        for bi, mb in enumerate(mega_batch):
            all_images.append(mb["frame_pil"])
            image_map.append((bi, "frame", None))
            for ci, crop_img in enumerate(mb["crop_images"]):
                all_images.append(crop_img)
                image_map.append((bi, "crop", ci))

        if all_images:
            all_embeddings = models.encode_images_clip(all_images)

            # Assign FAISS IDs and add to index (General CLIP index)
            for emb_idx, (bi, kind, ci) in enumerate(image_map):
                fid = faiss_index.ntotal
                faiss_index.add(all_embeddings[emb_idx:emb_idx + 1])
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
            for i, (bi, ci) in enumerate(person_map):
                pfid = person_faiss_index.ntotal
                person_faiss_index.add(reid_embeddings[i:i + 1])
                mega_batch[bi]["crop_records"][ci]["person_faiss_id"] = pfid


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

    # --- 6. Batch Face detection + FaceNet encode for person crops ---
    person_regions = []
    person_meta = []  # (bi, ci, px1, py1, px2, py2, det_id, frame_id, frame_no)
    
    for bi, mb in enumerate(mega_batch):
        for ci, rec in enumerate(mb["crop_records"]):
            if rec["class_name"] != "person": continue
            px1, py1, px2, py2 = rec["bbox"]
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
                io_executor.submit(lambda img=face_pil, p=face_path: img.save(p, format="JPEG", quality=90))
                
                face_crops_bgr.append(face_bgr)
                face_meta.append({
                    "detection_id": det_id, "frame_id": fid, "confidence": face["confidence"],
                    "bbox": (abs_fx1, abs_fy1, abs_fx2, abs_fy2), "crop_path": face_path
                })
                
        # --- 7. Batch Biometric Face Recognition (Identity) ---
        if face_crops_bgr:
            face_embeddings = models.encode_faces_batch(face_crops_bgr)
            face_rows = []
            for i, meta in enumerate(face_meta):
                emb = face_embeddings[i]
                if np.sum(emb) == 0: continue # Invalid face
                
                ffid = face_faiss_index.ntotal
                face_faiss_index.add(emb.reshape(1, -1))
                
                x1, y1, x2, y2 = meta["bbox"]
                face_rows.append((
                    meta["detection_id"], meta["frame_id"], ffid,
                    meta["confidence"], x1, y1, x2, y2, meta["crop_path"]
                ))
            if face_rows:
                db.batch_add_faces(face_rows)


def index_video(video_path: str, use_vlm: bool = False, use_clip: bool = True):
    """
    Index a video FAST — CLIP embeddings + YOLO crops only.
    
    Pipeline (optimized):
    1. Seek directly to target frames (skip intermediate frames)
    2. Stream frames — process on-the-fly instead of collecting all in RAM
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

    # Estimate how many frames we'll actually process (before scene-change filtering)
    frame_interval = max(1, int(video_fps / config.EXTRACT_FPS))
    estimated_frames = total_frames_raw // frame_interval if frame_interval > 0 else total_frames_raw

    if existing:
        video_id = existing["id"]
        db.update_video_status(video_id, "indexing")
    else:
        video_id = db.add_video(filename, video_path, duration, video_fps, total_frames_raw)

    video_frames_dir = os.path.join(config.FRAMES_DIR, str(video_id))
    os.makedirs(video_frames_dir, exist_ok=True)

    faiss_index = _load_faiss_index() if use_clip else None

    # Load person and face FAISS indexes with specific biometric dimensions
    def _load_or_create_faiss(path, dim):
        if os.path.exists(path):
            idx = faiss.read_index(path)
            if idx.d == dim:
                return idx
            print(f"[indexer] Dimension mismatch in {path} ({idx.d} != {dim}). Recreating.")
            # Clear old index to avoid crash later
            if os.path.exists(path): os.remove(path)
            
        return faiss.IndexFlatIP(dim)
    
    person_faiss_index = _load_or_create_faiss(config.PERSON_FAISS_INDEX_PATH, config.REID_DIM) if use_clip else None
    face_faiss_index = _load_or_create_faiss(config.FACE_FAISS_INDEX_PATH, config.FACE_REC_DIM) if use_clip else None

    print(f"\n{'='*60}")
    print(f"[indexer] START: {filename}")
    print(f"[indexer]   Video duration : {_fmt_time(duration)}")
    print(f"[indexer]   Video FPS      : {video_fps:.1f}")
    print(f"[indexer]   Total frames   : {total_frames_raw:,}")
    print(f"[indexer]   Extract at     : {config.EXTRACT_FPS} FPS (1 frame every {frame_interval} frames)")
    print(f"[indexer]   Est. frames    : ~{estimated_frames:,}")
    print(f"[indexer]   Mega-batch size: {config.CLIP_MEGA_BATCH}")
    print(f"{'='*60}")

    indexing_status.update({
        "active": True, "video": filename, "progress": 0.0,
        "message": f"Indexing {filename} (~{estimated_frames} frames)...",
        "frames_processed": 0, "total_frames": estimated_frames,
    })

    # Thread pool for non-blocking disk writes
    io_executor = ThreadPoolExecutor(max_workers=4)

    # --- Stream + Mega-batch processing ---
    mega_batch = []
    prev_hash = None
    skipped = 0
    processed = 0
    total_detections = 0
    t_start = time.time()
    
    import queue
    import threading

    frame_queue = queue.Queue(maxsize=128)
    def _extractor():
        for fno, ts, fb in extract_frames(video_path):
            frame_queue.put((fno, ts, fb))
        frame_queue.put(None)
        
    threading.Thread(target=_extractor, daemon=True).start()

    while True:
        data = frame_queue.get()
        if data is None: break
        
        frame_no, timestamp, frame_bgr = data
        
        # Scene change detection
        curr_hash = _frame_hash(frame_bgr)
        if _frames_similar(prev_hash, curr_hash):
            skipped += 1
            continue
        prev_hash = curr_hash

        # Save frame to disk (non-blocking)
        frame_path = os.path.join(video_frames_dir, f"frame_{frame_no:06d}.jpg")
        frame_copy = frame_bgr.copy()  # copy so original can be GC'd
        io_executor.submit(cv2.imwrite, frame_path, frame_copy, [cv2.IMWRITE_JPEG_QUALITY, 85])

        frame_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

        mega_batch.append({
            "frame_no": frame_no,
            "timestamp": timestamp,
            "frame_bgr_ref": frame_bgr,
            "frame_path": frame_path,
            "frame_pil": frame_pil,
            "detections": [],
            "crop_images": [],
            "crop_records": [],
            "frame_faiss_id": None,
        })

        # Process and flush mega-batch when full
        if len(mega_batch) >= config.CLIP_MEGA_BATCH:
            # 1. Batch YOLO Object Detection
            bgr_frames = [mb["frame_bgr_ref"] for mb in mega_batch]
            batch_dets = models.detect_objects_batch(bgr_frames)
            
            for mb, dets in zip(mega_batch, batch_dets):
                mb["detections"] = dets
                total_detections += len(dets)
                for di, det in enumerate(dets):
                    crop_img = crop_detection(mb["frame_bgr_ref"], det["bbox"])
                    crop_filename = f"frame_{mb['frame_no']:06d}_det_{di}.jpg"
                    crop_path = os.path.join(video_frames_dir, crop_filename)
                    io_executor.submit(lambda img=crop_img, p=crop_path: img.save(p, format="JPEG", quality=85))
                    mb["crop_images"].append(crop_img)
                    mb["crop_records"].append({
                        "class_name": det["class_name"],
                        "confidence": det["confidence"],
                        "bbox": det["bbox"],
                        "crop_path": crop_path,
                    })

            _flush_mega_batch(mega_batch, faiss_index, person_faiss_index, face_faiss_index,
                             use_clip, video_id, io_executor, video_frames_dir)
            
            processed += len(mega_batch)
            elapsed = time.time() - t_start
            fps_rate = processed / elapsed if elapsed > 0 else 0
            pct = processed / estimated_frames * 100 if estimated_frames > 0 else 0
            remaining = (estimated_frames - processed) / fps_rate if fps_rate > 0 else 0
            status_msg = (f"Frame {processed}/~{estimated_frames} ({pct:.0f}%) — "
                         f"{fps_rate:.2f} frames/s — ETA {_fmt_time(remaining)} — "
                         f"{skipped} skipped")
            print(f"[indexer] {status_msg}")
            indexing_status.update({
                "frames_processed": processed,
                "progress": pct,
                "message": status_msg,
            })
            mega_batch = []

    # Flush remaining frames
    if mega_batch:
        bgr_frames = [mb["frame_bgr_ref"] for mb in mega_batch]
        batch_dets = models.detect_objects_batch(bgr_frames)
        for mb, dets in zip(mega_batch, batch_dets):
            mb["detections"] = dets
            total_detections += len(dets)
            for di, det in enumerate(dets):
                crop_img = crop_detection(mb["frame_bgr_ref"], det["bbox"])
                crop_filename = f"frame_{mb['frame_no']:06d}_det_{di}.jpg"
                crop_path = os.path.join(video_frames_dir, crop_filename)
                io_executor.submit(lambda img=crop_img, p=crop_path: img.save(p, format="JPEG", quality=85))
                mb["crop_images"].append(crop_img)
                mb["crop_records"].append({
                    "class_name": det["class_name"],
                    "confidence": det["confidence"],
                    "bbox": det["bbox"],
                    "crop_path": crop_path,
                })
                
        _flush_mega_batch(mega_batch, faiss_index, person_faiss_index, face_faiss_index,
                         use_clip, video_id, io_executor, video_frames_dir)
        processed += len(mega_batch)

    # Wait for all disk writes to finish
    io_executor.shutdown(wait=True)

    if use_clip and faiss_index is not None:
        _save_faiss_index(faiss_index)
    if use_clip and person_faiss_index is not None:
        faiss.write_index(person_faiss_index, config.PERSON_FAISS_INDEX_PATH)
    if use_clip and face_faiss_index is not None:
        faiss.write_index(face_faiss_index, config.FACE_FAISS_INDEX_PATH)

    elapsed_total = time.time() - t_start
    db.update_video_status(video_id, "done")

    # --- Final summary ---
    speed_ratio = duration / elapsed_total if elapsed_total > 0 else 0
    fps_final = processed / elapsed_total if elapsed_total > 0 else 0

    person_count = person_faiss_index.ntotal if person_faiss_index else 0
    face_count = face_faiss_index.ntotal if face_faiss_index else 0

    print(f"\n{'='*60}")
    print(f"[indexer] DONE: {filename}")
    print(f"[indexer]   Video duration    : {_fmt_time(duration)}")
    print(f"[indexer]   Indexing time     : {_fmt_time(elapsed_total)}")
    print(f"[indexer]   Speed ratio       : {speed_ratio:.1f}x (1 min video ≈ {60/speed_ratio:.0f}s indexing)" if speed_ratio > 0 else "")
    print(f"[indexer]   Frames indexed    : {processed:,}")
    print(f"[indexer]   Frames skipped    : {skipped:,} (scene duplicates)")
    print(f"[indexer]   Objects detected  : {total_detections:,}")
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


def index_directory(dir_path: str, use_vlm: bool = False, use_clip: bool = True):
    """Recursively index all video files in a directory."""
    exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}
    videos = [os.path.join(r, f) for r, _, fs in os.walk(dir_path)
              for f in fs if os.path.splitext(f)[1].lower() in exts]
    for i, vp in enumerate(videos):
        indexing_status["message"] = f"Video {i + 1}/{len(videos)}: {os.path.basename(vp)}"
        index_video(vp, use_vlm=use_vlm, use_clip=use_clip)
