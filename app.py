"""FastAPI server for Vision-Search Agent."""
import os
import asyncio
from threading import Thread
from fastapi import FastAPI, BackgroundTasks, Query, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
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
        models._load_dinov2()
        models._load_facenet()
        searcher._load_yolo_world()

    # asyncio.to_thread yields the event loop while the models load in a worker thread.
    await asyncio.to_thread(load_all)
    print("[startup] AI models loaded successfully. Ready for instant search!")
    yield

app = FastAPI(title="Vision-Search Agent", version="1.0", lifespan=lifespan)


# ---- API Models ----

class IndexRequest(BaseModel):
    path: str
    use_vlm: bool = True
    use_clip: bool = True


# ---- API Endpoints ----

@app.get("/api/search")
async def api_search(
    q: str = Query(..., description="Search query"),
    top_k: int = Query(20, description="Number of results"),
    mode: str = Query("hybrid", description="Search mode: text, clip, or hybrid"),
):
    """Search indexed videos by natural language query."""
    results = searcher.search(q, top_k=top_k, mode=mode)
    return {"query": q, "mode": mode, "count": len(results), "results": results}


@app.post("/api/search-by-image")
async def api_search_by_image(
    file: UploadFile = File(...),
    top_k: int = Query(20),
):
    """Search by uploading an image — finds visually similar frames."""
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    results = searcher.search_by_image(image, top_k=top_k)
    return {"query": f"[image: {file.filename}]", "mode": "image", "count": len(results), "results": results}


@app.post("/api/search-by-person")
async def api_search_by_person(
    file: UploadFile = File(...),
    top_k: int = Query(20),
):
    """Search by person photo — find this specific person across all indexed videos.
    Uses dedicated person Re-ID and face recognition FAISS indexes."""
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    results = searcher.search_by_person(image, top_k=top_k)
    return {"query": f"[person: {file.filename}]", "mode": "person", "count": len(results), "results": results}


@app.post("/api/index")
async def api_index(req: IndexRequest, background_tasks: BackgroundTasks):
    """Start indexing a video file or directory."""
    path = req.path.strip()
    if not os.path.exists(path):
        return JSONResponse(status_code=400, content={"error": f"Path not found: {path}"})
    
    if indexer.indexing_status["active"]:
        return JSONResponse(status_code=409, content={"error": "Indexing already in progress"})
    
    def _run():
        if os.path.isdir(path):
            indexer.index_directory(path, use_vlm=req.use_vlm, use_clip=req.use_clip)
        else:
            indexer.index_video(path, use_vlm=req.use_vlm, use_clip=req.use_clip)
    
    thread = Thread(target=_run, daemon=True)
    thread.start()
    
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
        indexer.index_video(source_path, use_vlm=False, use_clip=True)

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


@app.get("/api/storage")
async def api_storage():
    """Per-video row counts (used by the frontend for storage/stats views)."""
    return db.get_storage_stats()


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
