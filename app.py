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
    # Load in thread to prevent blocking event loop
    def load_all():
        models._load_clip()
        models._load_yolo()
        models._load_yolo_face()
        models._load_dinov2()
        models._load_facenet()
        searcher._load_yolo_world()
    
    thread = Thread(target=load_all)
    thread.start()
    thread.join() # Wait for models to load before accepting requests
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


@app.delete("/api/videos/{video_id}")
async def api_delete_video(video_id: int):
    """Delete a video and its data from the index."""
    db.delete_video(video_id)
    return {"message": "Video deleted"}


@app.get("/api/frame")
async def api_get_frame(path: str = Query(...)):
    """Serve a frame or crop image file."""
    if os.path.exists(path) and path.startswith(config.FRAMES_DIR):
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
