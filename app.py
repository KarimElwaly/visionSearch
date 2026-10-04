"""FastAPI server for Vision-Search Agent."""
import os
import asyncio
import csv
import json
from io import StringIO
from threading import Lock, Thread
from fastapi import FastAPI, BackgroundTasks, Query, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel
from PIL import Image
import io

import config
import database as db
import indexer
import searcher
import models
from contextlib import asynccontextmanager

# Initialize database
db.init_db()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n[startup] Eagerly loading AI models into memory. This may take 10-20 seconds...")

    def load_all():
        models._load_clip()
        models._load_yolo()
        models._load_yolo_face()
        models._load_osnet()
        models._load_facenet()
        searcher._load_yolo_world()

    # asyncio.to_thread yields the event loop while the models load in a worker thread.
    await asyncio.to_thread(load_all)
    print("[startup] AI models loaded successfully. Ready for instant search!")
    yield

app = FastAPI(title="Vision-Search Agent", version="1.0", lifespan=lifespan)
_index_job_lock = Lock()
_index_job_thread: Thread | None = None


# ---- API Models ----

class IndexRequest(BaseModel):
    path: str
    use_vlm: bool = True
    use_clip: bool = True
    camera_id: str | None = None
    location: str | None = None
    recording_start_time: str | None = None
    timezone: str | None = None
    index_profile: str | None = None
    # Optional per-job overrides are passed directly into detector inference;
    # they never mutate process-global model configuration.
    yolo_classes: list[int] | None = None
    yolo_confidence: float | None = None


def _expanded_top_k(top_k: int, *filters) -> int:
    """Fetch more candidates when applying post-search CCTV filters/grouping."""
    return max(top_k * 5, 100) if any(v is not None and v is not False for v in filters) else top_k


def _filter_results(results: list, video_id: int = None, camera_id: str = None,
                    class_name: str = None, start_time: float = None, end_time: float = None,
                    min_score: float = None, match_type: str = None) -> list:
    """Apply CCTV filters to already-ranked search results."""
    filtered = []
    camera_norm = camera_id.lower() if camera_id else None
    class_norm = class_name.lower() if class_name else None
    match_norm = match_type.lower() if match_type else None
    for r in results:
        if video_id is not None and int(r.get("video_id") or -1) != int(video_id):
            continue
        if camera_norm and str(r.get("camera_id") or "").lower() != camera_norm:
            continue
        if class_norm and str(r.get("class_name") or "").lower() != class_norm:
            continue
        ts = float(r.get("timestamp") or 0)
        if start_time is not None and ts < float(start_time):
            continue
        if end_time is not None and ts > float(end_time):
            continue
        if min_score is not None and float(r.get("score") or 0) < float(min_score):
            continue
        if match_norm and str(r.get("match_type") or "").lower() != match_norm:
            continue
        filtered.append(r)
    return filtered


def _group_results_by_track(results: list) -> list:
    """Collapse multiple candidate detections from the same track into one event card."""
    grouped = {}
    untracked = []
    for r in sorted(results, key=lambda item: item.get("score", 0), reverse=True):
        track_id = r.get("track_id")
        if not track_id:
            untracked.append(r)
            continue
        key = int(track_id)
        if key not in grouped:
            item = dict(r)
            item["type"] = "track"
            item["source"] = f"{item.get('source', 'search')}+track"
            item["track_result"] = True
            item["detection_count"] = 1
            item["track_start_timestamp"] = float(item.get("timestamp") or 0)
            item["track_end_timestamp"] = float(item.get("timestamp") or 0)
            grouped[key] = item
        else:
            item = grouped[key]
            item["detection_count"] += 1
            ts = float(r.get("timestamp") or 0)
            item["track_start_timestamp"] = min(item["track_start_timestamp"], ts)
            item["track_end_timestamp"] = max(item["track_end_timestamp"], ts)
            item.setdefault("_cluster", []).append(r)

    merged = list(grouped.values()) + untracked
    merged.sort(key=lambda item: item.get("score", 0), reverse=True)
    return merged


def _faiss_count_from_meta(index_path: str) -> dict:
    """Read FAISS count from sidecar when possible; fall back to faiss.read_index."""
    meta_path = index_path + ".meta.json"
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            return {"exists": os.path.exists(index_path), "count": int(meta.get("count", 0)), "source": "meta"}
        except Exception as exc:
            return {"exists": os.path.exists(index_path), "count": None, "source": "meta", "error": str(exc)}
    if not os.path.exists(index_path):
        return {"exists": False, "count": 0, "source": "missing"}
    try:
        import faiss
        idx = faiss.read_index(index_path)
        return {"exists": True, "count": int(idx.ntotal), "source": "faiss"}
    except Exception as exc:
        return {"exists": True, "count": None, "source": "faiss", "error": str(exc)}


def _index_health_report() -> dict:
    """Compare DB vector references, FAISS counts, and stored image files."""
    with db.get_db() as conn:
        general_db = conn.execute(
            """SELECT
                   (SELECT COUNT(*) FROM detections WHERE faiss_id IS NOT NULL) +
                   (SELECT COUNT(*) FROM frame_embeddings WHERE faiss_id IS NOT NULL) AS n"""
        ).fetchone()["n"]
        person_db = conn.execute("SELECT COUNT(*) AS n FROM detections WHERE person_faiss_id IS NOT NULL").fetchone()["n"]
        face_db = conn.execute("SELECT COUNT(*) AS n FROM faces WHERE face_faiss_id IS NOT NULL").fetchone()["n"]
        frame_paths = [r["frame_path"] for r in conn.execute("SELECT frame_path FROM frames")]
        crop_paths = [r["crop_path"] for r in conn.execute("SELECT crop_path FROM detections WHERE crop_path IS NOT NULL")]
        face_paths = [r["crop_path"] for r in conn.execute("SELECT crop_path FROM faces WHERE crop_path IS NOT NULL")]

    indexes = {
        "general": {"db_refs": int(general_db), **_faiss_count_from_meta(config.FAISS_INDEX_PATH)},
        "person": {"db_refs": int(person_db), **_faiss_count_from_meta(config.PERSON_FAISS_INDEX_PATH)},
        "face": {"db_refs": int(face_db), **_faiss_count_from_meta(config.FACE_FAISS_INDEX_PATH)},
    }
    for item in indexes.values():
        item["matches"] = item.get("count") == item.get("db_refs") if item.get("count") is not None else False

    all_paths = frame_paths + crop_paths + face_paths
    missing = [p for p in all_paths if p and not os.path.exists(p)]
    return {
        "ok": all(item["matches"] for item in indexes.values()) and not missing,
        "indexes": indexes,
        "files": {
            "checked": len(all_paths),
            "missing": len(missing),
            "missing_samples": missing[:20],
        },
    }


# ---- API Endpoints ----

@app.get("/api/search")
async def api_search(
    q: str = Query(..., description="Search query"),
    top_k: int = Query(20, description="Number of results"),
    mode: str = Query("hybrid", description="Search mode: text, clip, or hybrid"),
    video_id: int | None = Query(None),
    camera_id: str | None = Query(None),
    class_name: str | None = Query(None),
    start_time: float | None = Query(None, description="Video-relative start time in seconds"),
    end_time: float | None = Query(None, description="Video-relative end time in seconds"),
    min_score: float | None = Query(None),
    match_type: str | None = Query(None),
    group_tracks: bool = Query(False, description="Collapse detections by track_id"),
):
    """Search indexed videos by natural language query."""
    search_top_k = _expanded_top_k(
        top_k, video_id, camera_id, class_name, start_time, end_time, min_score, match_type, group_tracks
    )
    # Run the (potentially slow) CLIP+YOLO-World pipeline on a worker thread so
    # /api/status polls and other requests stay responsive during a search.
    results = await asyncio.to_thread(searcher.search, q, search_top_k, mode)
    results = _filter_results(results, video_id, camera_id, class_name, start_time, end_time, min_score, match_type)
    if group_tracks:
        results = _group_results_by_track(results)
    results = results[:top_k]
    return {"query": q, "mode": mode, "count": len(results), "results": results}


@app.post("/api/search-by-image")
async def api_search_by_image(
    file: UploadFile = File(...),
    top_k: int = Query(20),
    video_id: int | None = Query(None),
    camera_id: str | None = Query(None),
    class_name: str | None = Query(None),
    start_time: float | None = Query(None),
    end_time: float | None = Query(None),
    min_score: float | None = Query(None),
    group_tracks: bool = Query(False),
):
    """Search by uploading an image — finds visually similar frames."""
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    search_top_k = _expanded_top_k(top_k, video_id, camera_id, class_name, start_time, end_time, min_score, group_tracks)
    results = await asyncio.to_thread(searcher.search_by_image, image, search_top_k)
    results = _filter_results(results, video_id, camera_id, class_name, start_time, end_time, min_score)
    if group_tracks:
        results = _group_results_by_track(results)
    results = results[:top_k]
    return {"query": f"[image: {file.filename}]", "mode": "image", "count": len(results), "results": results}


@app.post("/api/search-by-vehicle")
async def api_search_by_vehicle(
    file: UploadFile = File(...),
    top_k: int = Query(20),
    video_id: int | None = Query(None),
    camera_id: str | None = Query(None),
    class_name: str | None = Query(None),
    start_time: float | None = Query(None),
    end_time: float | None = Query(None),
    min_score: float | None = Query(None),
    group_tracks: bool = Query(True),
):
    """Vehicle-focused image search for cars/trucks/bikes in CCTV footage."""
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    search_top_k = _expanded_top_k(top_k, video_id, camera_id, class_name, start_time, end_time, min_score, group_tracks)
    results = await asyncio.to_thread(searcher.search_by_image, image, max(search_top_k, 100))
    vehicle_classes = {c.lower() for c in config.VEHICLE_CLASSES}
    results = [r for r in results if str(r.get("class_name") or "").lower() in vehicle_classes]
    results = _filter_results(results, video_id, camera_id, class_name, start_time, end_time, min_score)
    if group_tracks:
        results = _group_results_by_track(results)
    results = results[:top_k]
    return {"query": f"[vehicle: {file.filename}]", "mode": "vehicle", "count": len(results), "results": results}


@app.post("/api/search-by-person")
async def api_search_by_person(
    file: UploadFile = File(...),
    top_k: int = Query(20),
    video_id: int | None = Query(None),
    camera_id: str | None = Query(None),
    start_time: float | None = Query(None),
    end_time: float | None = Query(None),
    min_score: float | None = Query(None),
    match_type: str | None = Query(None),
    group_tracks: bool = Query(False),
):
    """Search by person photo — find this specific person across all indexed videos.
    Uses dedicated person Re-ID and face recognition FAISS indexes."""
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    search_top_k = _expanded_top_k(top_k, video_id, camera_id, start_time, end_time, min_score, match_type, group_tracks)
    results = await asyncio.to_thread(searcher.search_by_person, image, search_top_k)
    results = _filter_results(results, video_id, camera_id, "person", start_time, end_time, min_score, match_type)
    if group_tracks:
        results = _group_results_by_track(results)
    results = results[:top_k]
    return {"query": f"[person: {file.filename}]", "mode": "person", "count": len(results), "results": results}


@app.post("/api/index")
async def api_index(req: IndexRequest, background_tasks: BackgroundTasks):
    """Start indexing a video file or directory."""
    global _index_job_thread
    path = req.path.strip()
    if not os.path.exists(path):
        return JSONResponse(status_code=400, content={"error": f"Path not found: {path}"})

    def _run():
        global _index_job_thread
        try:
            if os.path.isdir(path):
                indexer.index_directory(
                    path,
                    use_vlm=req.use_vlm,
                    use_clip=req.use_clip,
                    camera_id=req.camera_id,
                    location=req.location,
                    recording_start_time=req.recording_start_time,
                    timezone=req.timezone,
                    index_profile=req.index_profile,
                    yolo_classes=req.yolo_classes,
                    yolo_confidence=req.yolo_confidence,
                )
            else:
                indexer.index_video(
                    path,
                    use_vlm=req.use_vlm,
                    use_clip=req.use_clip,
                    camera_id=req.camera_id,
                    location=req.location,
                    recording_start_time=req.recording_start_time,
                    timezone=req.timezone,
                    index_profile=req.index_profile,
                    yolo_classes=req.yolo_classes,
                    yolo_confidence=req.yolo_confidence,
                )
        finally:
            with _index_job_lock:
                _index_job_thread = None
            indexer.indexing_status["active"] = False

    with _index_job_lock:
        if _index_job_thread is not None and _index_job_thread.is_alive():
            return JSONResponse(status_code=409, content={"error": "Indexing already in progress"})
        indexer.indexing_status.update({
            "active": True,
            "cancel_requested": False,
            "message": f"Starting indexing: {path}",
        })
        _index_job_thread = Thread(target=_run, daemon=True)
        _index_job_thread.start()
    
    return {"message": f"Indexing started: {path}"}


@app.get("/api/status")
async def api_status():
    """Get current indexing status."""
    return indexer.indexing_status


@app.get("/api/videos")
async def api_videos():
    """List all indexed videos."""
    videos = db.get_all_videos()
    return {"videos": [dict(v) for v in videos]}


@app.get("/api/videos/{video_id}")
async def api_video_details(video_id: int):
    """Return per-video details (row + row-count stats)."""
    with db.get_db() as conn:
        row = conn.execute("SELECT * FROM videos WHERE id = ?", (video_id,)).fetchone()
    if row is None:
        return JSONResponse(status_code=404, content={"error": "Video not found"})
    stats = db.get_storage_stats()
    my = next((v for v in stats["videos"] if v["id"] == video_id), None)
    return {"video": dict(row), "stats": my}


@app.post("/api/videos/{video_id}/reindex")
async def api_reindex_video(video_id: int):
    """Delete the video's derived data (frames + FAISS ids) and re-index from source."""
    if indexer.indexing_status["active"]:
        return JSONResponse(status_code=409, content={"error": "Indexing already in progress"})
    with db.get_db() as conn:
        row = conn.execute("SELECT * FROM videos WHERE id = ?", (video_id,)).fetchone()
    if row is None:
        return JSONResponse(status_code=404, content={"error": "Video not found"})
    source_path = row["path"]
    # Full cascade purge, then start indexing again in the background.
    stats = indexer.delete_video_fully(video_id)

    def _run():
        indexer.index_video(
            source_path,
            use_vlm=False,
            use_clip=True,
            camera_id=row["camera_id"] if "camera_id" in row.keys() else None,
            location=row["location"] if "location" in row.keys() else None,
            recording_start_time=row["recording_start_time"] if "recording_start_time" in row.keys() else None,
            timezone=row["timezone"] if "timezone" in row.keys() else None,
            index_profile=row["index_profile"] if "index_profile" in row.keys() else None,
        )

    Thread(target=_run, daemon=True).start()
    return {"message": f"Reindexing started: {source_path}", "purged": stats}


@app.delete("/api/videos/{video_id}")
async def api_delete_video(video_id: int):
    """Cascade delete: DB rows + FAISS vectors + frames dir."""
    stats = indexer.delete_video_fully(video_id)
    return {"message": "Video deleted", "stats": stats}


@app.post("/api/index/cancel")
async def api_cancel_index():
    """Cooperatively stop the current indexing job (state becomes ``paused``)."""
    ok = indexer.request_cancel()
    if not ok:
        return JSONResponse(status_code=409, content={"error": "No indexing job to cancel"})
    return {"message": "Cancellation requested; indexer will stop after the current batch"}


@app.get("/api/health")
async def api_health():
    """Lightweight health/readiness probe (no model calls)."""
    return {
        "status": "ok",
        "indexing_active": bool(indexer.indexing_status.get("active")),
        "clip_dim": config.CLIP_DIM,
        "reid_dim": config.REID_DIM,
        "face_dim": config.FACE_REC_DIM,
    }


@app.get("/api/index/health")
async def api_index_health():
    """Integrity check: DB vector references vs FAISS counts and missing stored images."""
    return _index_health_report()


@app.get("/api/storage")
async def api_storage():
    """Per-video row counts (used by the frontend for storage/stats views)."""
    return db.get_storage_stats()


@app.get("/api/tracks/stats")
async def api_track_stats(video_id: int | None = Query(None)):
    """Aggregate CCTV track counts, optionally for one video."""
    return db.get_track_stats(video_id)


@app.get("/api/videos/{video_id}/tracks")
async def api_video_tracks(
    video_id: int,
    class_name: str | None = Query(None),
    limit: int = Query(100, ge=1, le=500),
):
    """List representative object/person tracks for a video."""
    with db.get_db() as conn:
        video = conn.execute("SELECT * FROM videos WHERE id = ?", (video_id,)).fetchone()
    if video is None:
        return JSONResponse(status_code=404, content={"error": "Video not found"})
    rows = db.get_tracks_for_video(video_id, class_name=class_name, limit=limit)
    tracks = []
    for row in rows:
        item = dict(row)
        item["duration"] = max(0, float(item.get("end_timestamp") or 0) - float(item.get("start_timestamp") or 0))
        tracks.append(item)
    return {"video": dict(video), "count": len(tracks), "tracks": tracks}


@app.get("/api/tracks/{track_id}")
async def api_track_timeline(track_id: int, limit: int = Query(500, ge=1, le=2000)):
    """Return ordered detections for one CCTV track."""
    rows = db.get_track_timeline(track_id, limit=limit)
    if not rows:
        return JSONResponse(status_code=404, content={"error": "Track not found"})
    return {"track_id": track_id, "count": len(rows), "timeline": [dict(r) for r in rows]}


@app.get("/api/tracks/{track_id}/export")
async def api_track_export(track_id: int, format: str = Query("json", pattern="^(json|csv)$")):
    """Export a track timeline as JSON or CSV evidence report."""
    rows = [dict(r) for r in db.get_track_timeline(track_id, limit=5000)]
    if not rows:
        return JSONResponse(status_code=404, content={"error": "Track not found"})
    if format == "csv":
        fields = [
            "track_id", "detection_id", "frame_number", "timestamp", "confidence",
            "class_name", "video_filename", "video_path", "frame_path", "crop_path",
            "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
        ]
        buf = StringIO()
        writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            row["track_id"] = track_id
            writer.writerow(row)
        return Response(
            content=buf.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=track_{track_id}_evidence.csv"},
        )
    return {"track_id": track_id, "count": len(rows), "timeline": rows}


def _is_safe_frame_path(path: str) -> bool:
    """Return True only if ``path`` resolves inside FRAMES_DIR and is not a symlink escape.

    Uses realpath + commonpath so `..` traversal and symlink-based escapes both fail.
    """
    try:
        frames_root = os.path.realpath(config.FRAMES_DIR)
        target = os.path.realpath(path)
        return os.path.commonpath([frames_root, target]) == frames_root
    except (ValueError, OSError):
        # commonpath raises ValueError across drives on Windows
        return False


@app.get("/api/frame")
async def api_get_frame(path: str = Query(...)):
    """Serve a frame or crop image file (restricted to FRAMES_DIR)."""
    if _is_safe_frame_path(path) and os.path.isfile(path):
        return FileResponse(path, media_type="image/jpeg")
    return JSONResponse(status_code=404, content={"error": "Frame not found"})


# ---- Static Files ----

static_dir = os.path.join(config.BASE_DIR, "static")
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")


# ---- Run ----

if __name__ == "__main__":
    import uvicorn
    print(f"\n  Vision-Search Agent running at http://localhost:{config.PORT}\n")
    uvicorn.run(app, host=config.HOST, port=config.PORT, log_level="info")
