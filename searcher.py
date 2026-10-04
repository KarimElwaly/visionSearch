"""3-Stage Smart Search — "Index Fast, Search Smart"

Stage 1: CLIP Recall (instant, <1ms)
  → Find top 50 frames/crops by vector similarity

Stage 2: YOLO-World Re-ranking (fast, ~1 second total)  
  → Run user's EXACT query as open-vocabulary detection prompt
  → On only the top candidates from Stage 1
  → Frames where YOLO-World detects a match → boost score

Stage 3: VLM Verification (optional, ~5-10 seconds)
  → For top 5 results, ask VLM to confirm and describe the match
  → Adds rich text captions for display
"""
import os
import json
import threading
import time
from functools import wraps
import numpy as np
import faiss
from PIL import Image
import cv2

import config
import database as db
import models


# ---- Cached FAISS index loaders ----
#
# Reading a FAISS index from disk on every request is prohibitive for large
# indexes (100k+ vectors → seconds per search). We keep the index resident in
# memory and invalidate the cache when the on-disk file mtime changes (which
# happens after every indexing checkpoint / cascade delete).
_FAISS_CACHE: dict[str, tuple[float, int, "faiss.Index"]] = {}
_FAISS_CACHE_LOCK = threading.Lock()


def _get_cached_faiss(path: str):
    """Return the FAISS index at ``path`` with mtime-based cache invalidation.

    Returns None when the file does not exist yet (no vectors indexed).
    """
    if not os.path.exists(path):
        # Drop any stale cache entry so a deleted index isn't served.
        with _FAISS_CACHE_LOCK:
            _FAISS_CACHE.pop(path, None)
        return None
    try:
        stat = os.stat(path)
        signature = (stat.st_mtime_ns, stat.st_size)
    except OSError:
        return None

    with _FAISS_CACHE_LOCK:
        cached = _FAISS_CACHE.get(path)
        if cached is not None and cached[0] == signature:
            return cached[2]

    # Load outside the lock (can take a while on large indexes) then re-check.
    try:
        idx = faiss.read_index(path)
    except Exception as exc:
        print(f"[search] failed to read FAISS index {path}: {exc}")
        return None

    with _FAISS_CACHE_LOCK:
        # Double-check nothing raced past us with a newer mtime.
        _FAISS_CACHE[path] = (signature, idx.ntotal, idx)
    return idx


def _load_faiss_index():
    """Backwards-compatible accessor for the general (CLIP) FAISS index."""
    return _get_cached_faiss(config.FAISS_INDEX_PATH)


# ---- YOLO-World for open-vocabulary re-ranking ----

_yolo_world_model = None
_yolo_world_device = "cpu"
_image_search_yolo = None  # cached base YOLO for search_by_image object cropping
_YOLO_WORLD_LOCK = threading.Lock()


def _timed_operation(name: str):
    """Log one machine-readable end-to-end latency record per operation."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            started = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                timing = {
                    "operation": name,
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                }
                print(f"[search][timing] {json.dumps(timing, sort_keys=True)}")
        return wrapper
    return decorator


def _load_image_search_yolo():
    """Cached base YOLO detector used by search_by_image to crop the primary object.

    Runs on CPU to keep GPU memory free for CLIP / OSNet / FaceNet.
    """
    global _image_search_yolo
    if _image_search_yolo is None:
        from ultralytics import YOLO
        print(f"[search] Loading base YOLO for image search: {config.YOLO_MODEL_PATH}")
        _image_search_yolo = YOLO(config.YOLO_MODEL_PATH)
        _image_search_yolo.to("cpu")
    return _image_search_yolo


def _load_yolo_world():
    global _yolo_world_model, _yolo_world_device
    if _yolo_world_model is None:
        from ultralytics import YOLO
        import clip as clip_pkg
        import torch
        model_path = config.YOLO_WORLD_MODEL_PATH
        print(f"[search] Loading YOLO-World: {model_path}")
        _yolo_world_model = YOLO(model_path)
        requested_device = str(getattr(config, "YOLO_WORLD_DEVICE", "auto")).lower()
        target_device = "cuda" if requested_device == "auto" and torch.cuda.is_available() else requested_device
        if target_device not in {"cpu", "cuda"}:
            target_device = requested_device

        try:
            _yolo_world_model.to(target_device)
            _yolo_world_device = target_device
        except Exception as exc:
            print(f"[search] YOLO-World could not use {target_device}: {exc}; falling back to CPU")
            _yolo_world_model.to("cpu")
            _yolo_world_device = "cpu"

        # Pre-load YOLO-World's internal CLIP text encoder on CPU
        # so dynamic query classes do not consume additional GPU memory.
        clip_model, _ = clip_pkg.load("ViT-B/32", device="cpu")
        _yolo_world_model.model.clip = clip_model
        print(f"[search] YOLO-World loaded ({_yolo_world_device}).")

        if getattr(config, "YOLO_WORLD_WARMUP", True):
            try:
                _yolo_world_model.set_classes(["person"])
                dummy_frame = np.zeros((config.YOLO_WORLD_IMAGE_SIZE, config.YOLO_WORLD_IMAGE_SIZE, 3), dtype=np.uint8)
                _yolo_world_model(
                    [dummy_frame],
                    conf=config.YOLO_WORLD_CONFIDENCE,
                    imgsz=config.YOLO_WORLD_IMAGE_SIZE,
                    batch=1,
                    device=_yolo_world_device,
                    verbose=False,
                )
                print(f"[search] YOLO-World warmup complete ({_yolo_world_device}).")
            except Exception as exc:
                if _yolo_world_device != "cpu":
                    print(f"[search] YOLO-World GPU warmup failed: {exc}; falling back to CPU")
                    _yolo_world_model.to("cpu")
                    _yolo_world_device = "cpu"
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
    return _yolo_world_model


@_timed_operation("yolo_world_rerank")
def _rerank_with_yolo_world(candidates: list, query: str, top_k: int) -> list:
    """
    Stage 2: Re-rank candidates using YOLO-World open-vocabulary detection.
    Uses the FULL query as the detection class (not individual words).
    "white jacket" → detects white jackets, NOT any jacket.
    """
    global _yolo_world_device
    try:
        model = _load_yolo_world()
    except Exception as e:
        print(f"[search] YOLO-World not available: {e}, skipping re-ranking")
        return candidates

    # Use the full query (e.g., "black bag") and the core object (e.g., "bag")
    # Colors are meaningful attributes and must NOT be stripped.
    words = query.lower().split()
    core_nouns = [w for w in words if len(w) > 3 and w not in
                  {"with", "wearing", "that", "this", "from", "into", "near", "next", "have", "some"}]
    
    query_classes = [query.strip()]
    if core_nouns and core_nouns[-1] != query.strip():
        query_classes.append(core_nouns[-1])

    # Multiple detection candidates often point to the same full frame. Infer
    # each unique frame once, then reuse its open-vocabulary result.
    candidates_by_path = {}
    for candidate in candidates:
        target_path = candidate.get("frame_path") or candidate.get("crop_path", "")
        if target_path and os.path.exists(target_path):
            candidates_by_path.setdefault(target_path, []).append(candidate)

    paths = list(candidates_by_path)
    batch_size = max(1, int(getattr(config, "YOLO_WORLD_BATCH_SIZE", 8)))
    image_size = int(getattr(config, "YOLO_WORLD_IMAGE_SIZE", 640))
    inference_by_path = {}

    def run_batches(device: str) -> dict:
        results_by_path = {}
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start:start + batch_size]
            batch_frames = []
            loaded_paths = []
            for path in batch_paths:
                frame = cv2.imread(path)
                if frame is not None:
                    batch_frames.append(frame)
                    loaded_paths.append(path)

            if not batch_frames:
                continue

            batch_results = model(
                batch_frames,
                conf=config.YOLO_WORLD_CONFIDENCE,
                imgsz=image_size,
                batch=len(batch_frames),
                device=device,
                verbose=False,
            )
            for path, result in zip(loaded_paths, batch_results):
                max_conf = 0.0
                det_count = 0
                best_class_idx = -1
                for box in result.boxes:
                    det_count += 1
                    confidence = float(box.conf[0])
                    if confidence > max_conf:
                        max_conf = confidence
                        best_class_idx = int(box.cls[0])
                results_by_path[path] = (det_count, max_conf, best_class_idx)
        return results_by_path

    # set_classes mutates shared model state, so keep class selection and all
    # corresponding inference together when concurrent searches arrive.
    try:
        with _YOLO_WORLD_LOCK:
            model.set_classes(query_classes)
            print(f"[search] YOLO-World classes: {query_classes}")
            try:
                inference_by_path = run_batches(_yolo_world_device)
            except Exception as gpu_exc:
                if _yolo_world_device == "cpu":
                    raise
                print(f"[search] YOLO-World GPU inference failed: {gpu_exc}; retrying on CPU")
                model.to("cpu")
                _yolo_world_device = "cpu"
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                inference_by_path = run_batches("cpu")
    except Exception as e:
        print(f"[search] YOLO-World batch inference failed: {e}, skipping re-ranking")
        return candidates[:top_k]

    for path, path_candidates in candidates_by_path.items():
        det_count, max_conf, best_class_idx = inference_by_path.get(path, (0, 0.0, -1))
        for candidate in path_candidates:
            if det_count > 0:
                # Strong boost for the exact query; smaller boost for its noun.
                boost = 0.8 if best_class_idx == 0 else 0.3
                candidate["score"] = candidate.get("score", 0) + max_conf * boost
                candidate["yolo_world_match"] = True
                candidate["yolo_world_conf"] = max_conf
                candidate["yolo_world_count"] = det_count
            else:
                # Preserve possible small/rare CLIP matches but demote them.
                candidate["score"] = candidate.get("score", 0) * 0.35
                candidate["yolo_world_match"] = False

    # Re-sort by updated score
    candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
    return candidates[:top_k]




def _parse_query_attributes(query: str) -> tuple[str, str]:
    """
    Extracts modifiers/attributes (like colors) and the core noun.
    Returns (attributes_string, core_noun_string).
    """
    words = query.lower().split()
    if len(words) <= 1:
        return "", query.strip()
        
    stop_words = {"with", "wearing", "that", "this", "from", "into", "near", "next", "have", "some", "the", "a", "an", "and", "or", "in", "on", "at"}
    filtered = [w for w in words if w not in stop_words]
    
    if len(filtered) <= 1:
        return "", query.strip()
        
    core_noun = filtered[-1]
    attributes = " ".join(filtered[:-1])
    return attributes, core_noun


def _blend_attributes(query: str) -> np.ndarray:
    """Helper to apply attribute blending to a single query string."""
    attributes, core_noun = _parse_query_attributes(query)
    if attributes:
        query_emb = models.encode_text_clip(query)
        attr_emb = models.encode_text_clip(attributes)
        blended = (query_emb * 0.6) + (attr_emb * 0.4)
        print(f"[search] Attribute blending: '{attributes}' (40%) + '{query}' (60%)")
        return blended / np.linalg.norm(blended, axis=1, keepdims=True)
    return models.encode_text_clip(query)


# CLIP prompt templates used for ensemble query encoding (proven +2–5 pts recall
# on retrieval benchmarks). Kept neutral — forcing a "person" context hurts
# recall on object queries like "ring", "bag", "car".
_PROMPT_TEMPLATES = (
    "{q}",
    "a photo of {q}",
    "a picture of {q}",
    "an image of {q}",
    "a photo of {q} in a scene",
)


def _prompt_ensemble_embedding(query: str) -> np.ndarray:
    """Average CLIP text embeddings over a small template ensemble."""
    attributes, _ = _parse_query_attributes(query)
    embs = []
    for tmpl in _PROMPT_TEMPLATES:
        try:
            prompted_query = tmpl.format(q=query)
            query_emb = models.encode_text_clip(prompted_query)
            if attributes:
                prompted_attributes = tmpl.format(q=attributes)
                attr_emb = models.encode_text_clip(prompted_attributes)
                query_emb = (query_emb * 0.6) + (attr_emb * 0.4)
                query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
            embs.append(query_emb)
        except Exception as e:
            print(f"[search] prompt ensemble error on '{tmpl}': {e}")
    if not embs:
        return _blend_attributes(query)
    avg = np.mean(np.vstack(embs), axis=0, keepdims=True)
    return avg / np.linalg.norm(avg, axis=1, keepdims=True)


def _get_query_embedding(query: str) -> np.ndarray:
    """
    Parses negative prompts (--no) and applies attribute blending +
    prompt-template ensemble. Returns unit-length embedding.
    """
    query_lower = query.lower()
    if " --no " in query_lower:
        parts = query_lower.split(" --no ", 1)
        pos_query = parts[0].strip()
        neg_query = parts[1].strip()

        pos_emb = _prompt_ensemble_embedding(pos_query)
        neg_emb = _prompt_ensemble_embedding(neg_query)
        
        # Vector Subtraction (Option 2)
        print(f"[search] Negative Prompt applied: '{pos_query}' minus '{neg_query}'")
        blended = pos_emb - (neg_emb * 0.5)  # 0.5 strength for negative
        return blended / np.linalg.norm(blended, axis=1, keepdims=True)
    else:
        return _prompt_ensemble_embedding(query)


def _video_group_key(result: dict):
    """Return a stable identity for grouping results from one video."""
    video_id = result.get("video_id")
    if video_id is not None:
        return ("id", str(video_id))
    video_path = result.get("video_path")
    if video_path:
        return ("path", os.path.normcase(os.path.abspath(video_path)))
    return ("filename", str(result.get("video_filename") or ""))


def _apply_temporal_boosting(results: list) -> list:
    """
    STAGE 2.5/3: TEMPORAL CONTINUITY BOOST (Option 5)
    Boosts scores of frames that occur in temporal clusters (real objects persist over time).
    Penalizes isolated frames (likely glitches or motion blur false positives).
    """
    if not results:
        return results
        
    video_map = {}
    for r in results:
        vid = _video_group_key(r)
        if vid not in video_map:
            video_map[vid] = []
        video_map[vid].append(r)
        
    for vid, group in video_map.items():
        group.sort(key=lambda x: x.get("timestamp", 0))
        for i, r in enumerate(group):
            ts = r.get("timestamp", 0)
            has_neighbor = False
            
            # Check within 3 seconds
            if i > 0 and abs(ts - group[i-1].get("timestamp", 0)) <= 3.0:
                has_neighbor = True
            if i < len(group) - 1 and abs(group[i+1].get("timestamp", 0) - ts) <= 3.0:
                has_neighbor = True
                
            if has_neighbor:
                r["score"] = r.get("score", 0) * 1.20 # 20% boost
            else:
                r["score"] = r.get("score", 0) * 0.90 # 10% penalty

    results.sort(key=lambda x: x.get("score", 0), reverse=True)
    return results


@_timed_operation("text_search")
def search(query: str, top_k: int = None, mode: str = "hybrid"):
    """
    3-Stage Smart Search:
    
    Stage 1: CLIP vector recall → top candidates (instant)
    Stage 2: YOLO-World re-ranking → precise filtering (~1 sec)
    Stage 3: VLM enrichment → add descriptions to top results (optional)
    """
    if top_k is None:
        top_k = config.DEFAULT_TOP_K

    results = []
    seen_keys = set()

    # ================================================================
    # STAGE 1: CLIP VECTOR RECALL (instant)
    # ================================================================
    # Stricter baseline (from 0.15 -> 0.20) to filter out garbage early
    min_clip_score = 0.20
    index = _load_faiss_index()
    
    if index is not None and index.ntotal > 0 and mode in ("clip", "hybrid"):
        try:
            # Generate the advanced query embedding (combines blending + negative prompts)
            query_emb = _get_query_embedding(query)
            
            # Get more candidates than needed — long CCTV videos can contain
            # tens of thousands of near-similar detections, so keep recall high
            # before YOLO-World and temporal de-duplication filter the list.
            n_candidates = min(
                max(top_k * config.CLIP_CANDIDATE_POOL_FACTOR, config.CLIP_MIN_CANDIDATES),
                index.ntotal,
            )
            scores, indices = index.search(query_emb, n_candidates)

            valid = [(int(indices[0][i]), float(scores[0][i]))
                     for i in range(len(indices[0]))
                     if indices[0][i] >= 0 and float(scores[0][i]) >= min_clip_score]

            faiss_ids = [fid for fid, _ in valid]
            score_map = {fid: score for fid, score in valid}

            print(f"[search] Stage 1 CLIP: {len(valid)} candidates (threshold {min_clip_score})")

            if faiss_ids:
                # Detection-level results (YOLO crops)
                det_results = db.get_detections_by_faiss_ids(faiss_ids)
                for r in det_results:
                    key = f"det_{r['id']}"
                    if key not in seen_keys:
                        seen_keys.add(key)
                        result = _format_detection_result(r, source="clip")
                        result["score"] = score_map.get(r["faiss_id"], 0)
                        results.append(result)

                # Frame-level results
                frame_results = db.get_frame_embeddings_by_faiss_ids(faiss_ids)
                for r in frame_results:
                    key = f"scene_{r['id']}"
                    if key not in seen_keys:
                        seen_keys.add(key)
                        result = _format_scene_result(r, source="clip")
                        result["score"] = score_map.get(r["faiss_id"], 0)
                        results.append(result)

        except Exception as e:
            print(f"[search] Stage 1 error: {e}")

    # Also try text search if captions exist
    if mode in ("text", "hybrid"):
        try:
            fts_query = _build_fts_query(query)
            text_results = db.search_by_text(fts_query, limit=top_k)
            for r in text_results:
                key = f"det_{r['id']}"
                if key not in seen_keys:
                    seen_keys.add(key)
                    results.append(_format_detection_result(r, source="text"))

            scene_results = db.search_frames_by_text(fts_query, limit=top_k)
            for r in scene_results:
                key = f"scene_{r['id']}"
                if key not in seen_keys:
                    seen_keys.add(key)
                    results.append(_format_scene_result(r, source="text"))
        except Exception:
            pass

    # Sort by CLIP score
    results.sort(key=lambda x: x.get("score", 0), reverse=True)

    # ================================================================
    # STAGE 2: YOLO-WORLD RE-RANKING (~1-2 seconds)
    # Only re-rank top candidates for speed, but ensure we have enough
    # ================================================================
    if results and mode in ("clip", "hybrid"):
        # Keep a larger recall pool so open-vocabulary verification can rescue
        # relevant CLIP candidates that initially rank just below top_k.
        rerank_factor = max(1, int(getattr(config, "YOLO_WORLD_RERANK_FACTOR", 3)))
        rerank_cap = max(top_k, int(getattr(config, "YOLO_WORLD_MAX_CANDIDATES", 60)))
        rerank_count = min(len(results), top_k * rerank_factor, rerank_cap)
        candidates_to_rerank = results[:rerank_count]
        print(f"[search] Stage 2 YOLO-World: re-ranking {len(candidates_to_rerank)} candidates")
        
        # We only keep the ones that survived re-ranking
        results = _rerank_with_yolo_world(candidates_to_rerank, query, top_k)

        # Drop only clearly-noise results. YOLO-World misses on small / rare
        # objects, so we do NOT filter out non-verified candidates \u2014 they are
        # already softly demoted by the miss penalty inside the reranker.
        results = [r for r in results if r.get("score", 0) > 0.01]

    # Apply Option 5: Temporal Continuity Boost
    results = _apply_temporal_boosting(results)

    # Collapse near-duplicate thumbnails from the same event
    results = _cluster_near_duplicates(results)

    return results[:top_k]


def _color_histogram(img_bgr, bins=16):
    """Compute a normalized HSV histogram (H+S channels) for color matching."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # Hue (0-179) and Saturation (0-255) capture color identity better than Value
    hist = cv2.calcHist([hsv], [0, 1], None, [bins, bins], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def _rerank_by_color(results, query_image: Image.Image, weight: float = 0.4):
    """Re-rank image search results by color histogram similarity to the query.

    Blends the original CLIP score with an HSV color-correlation score:
        final_score = (1 - weight) * clip_score + weight * color_sim * max_clip_score

    This preserves CLIP's semantic ranking but demotes results with wrong colors.
    """
    if not results:
        return results

    # Convert query to BGR for OpenCV
    query_bgr = cv2.cvtColor(np.array(query_image), cv2.COLOR_RGB2BGR)
    query_hist = _color_histogram(query_bgr)

    max_score = max(r.get("score", 0) for r in results) or 1.0

    for r in results:
        # Load the candidate image
        path = r.get("crop_path") or r.get("frame_path", "")
        if not path or not os.path.exists(path):
            continue
        cand_bgr = cv2.imread(path)
        if cand_bgr is None:
            continue
        cand_hist = _color_histogram(cand_bgr)
        # Correlation ranges from -1 to 1; 1 = identical color distribution
        color_sim = cv2.compareHist(query_hist, cand_hist, cv2.HISTCMP_CORREL)
        color_sim = max(0.0, color_sim)  # clamp negatives

        clip_score = r.get("score", 0)
        r["score"] = (1 - weight) * clip_score + weight * color_sim * max_score
        r["color_similarity"] = round(color_sim, 3)

    results.sort(key=lambda x: x.get("score", 0), reverse=True)
    print(f"[search] Color re-rank applied (weight={weight})")
    return results


def _build_image_query_variants(image: Image.Image, yolo_results) -> list[Image.Image]:
    """Return the full query image plus the largest detected object crops.

    A query photo can contain several relevant objects, so encoding only the
    largest box can silently search for the wrong subject. Ultralytics already
    applies NMS; sorting by area gives a small, deterministic query batch.
    """
    crop_candidates = []
    image_width, image_height = image.size
    for result in yolo_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1 = max(0, min(x1, image_width))
            y1 = max(0, min(y1, image_height))
            x2 = max(0, min(x2, image_width))
            y2 = max(0, min(y2, image_height))
            area = (x2 - x1) * (y2 - y1)
            if area > 0:
                crop_candidates.append((area, (x1, y1, x2, y2)))

    crop_candidates.sort(key=lambda item: item[0], reverse=True)
    max_objects = max(1, int(getattr(config, "IMAGE_SEARCH_MAX_OBJECTS", 3)))
    object_crops = [image.crop(bounds) for _, bounds in crop_candidates[:max_objects]]

    variants = []
    if getattr(config, "IMAGE_SEARCH_INCLUDE_FULL_IMAGE", True) or not object_crops:
        variants.append(image)
    variants.extend(object_crops)
    return variants


@_timed_operation("image_search")
def search_by_image(image: Image.Image, top_k: int = None):
    """
    Search by image — encode uploaded image with CLIP, find nearest matches in FAISS.
    No YOLO-World re-ranking needed (visual similarity is direct).
    """
    if top_k is None:
        top_k = config.DEFAULT_TOP_K

    results = []
    seen_keys = set()
    min_clip_score = 0.15
    best_crop = None

    index = _load_faiss_index()
    if index is None or index.ntotal == 0:
        return results

    try:
        # Option 1: Object-Centric Image Search
        # Use YOLOv11 to find the primary object in the uploaded image and crop out the background.
        # This prevents background noise (e.g. a messy room) from polluting the CLIP embedding.
        print("[search] Finding primary object in uploaded image...")

        # Convert PIL to CV2 format for YOLO
        cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        yolo_model = _load_image_search_yolo()
        yolo_results = yolo_model(cv_img, conf=config.IMAGE_SEARCH_YOLO_CONFIDENCE, verbose=False)
        
        query_images = _build_image_query_variants(image, yolo_results)
        best_crop = query_images[1] if len(query_images) > 1 else None
        if best_crop:
            print(f"[search] Encoding {len(query_images) - 1} object crops plus the full image.")
        else:
            print("[search] No clear object found, using full image.")
        query_embeddings = models.encode_images_clip(query_images)

        n_candidates = min(
            max(top_k * config.IMAGE_CANDIDATE_POOL_FACTOR, config.IMAGE_MIN_CANDIDATES),
            index.ntotal,
        )
        score_map = {}
        for query_embedding in query_embeddings:
            scores, indices = index.search(query_embedding.reshape(1, -1), n_candidates)
            for i in range(len(indices[0])):
                faiss_id = int(indices[0][i])
                score = float(scores[0][i])
                if faiss_id >= 0 and score >= min_clip_score:
                    score_map[faiss_id] = max(score_map.get(faiss_id, float("-inf")), score)

        faiss_ids = list(score_map)
        print(f"[search] Image search: {len(faiss_ids)} unique matches "
              f"from {len(query_images)} query variants (threshold {min_clip_score})")

        if faiss_ids:
            det_results = db.get_detections_by_faiss_ids(faiss_ids)
            for r in det_results:
                key = f"det_{r['id']}"
                if key not in seen_keys:
                    seen_keys.add(key)
                    result = _format_detection_result(r, source="image")
                    result["score"] = score_map.get(r["faiss_id"], 0)
                    results.append(result)

            frame_results = db.get_frame_embeddings_by_faiss_ids(faiss_ids)
            for r in frame_results:
                key = f"scene_{r['id']}"
                if key not in seen_keys:
                    seen_keys.add(key)
                    result = _format_scene_result(r, source="image")
                    result["score"] = score_map.get(r["faiss_id"], 0)
                    results.append(result)

    except Exception as e:
        print(f"[search] Image search error: {e}")

    results.sort(key=lambda x: x.get("score", 0), reverse=True)

    # --- Color histogram re-ranking ---
    # CLIP captures semantics ("bag") but not color well. Re-rank by color
    # similarity so a red-bag query doesn't surface black bags at rank 2.
    results = _rerank_by_color(results, image if best_crop is None else best_crop)

    # Apply Option 5: Temporal Continuity Boost
    results = _apply_temporal_boosting(results)

    results = _cluster_near_duplicates(results)

    return results[:top_k]


def _generate_query_orientations(image: Image.Image) -> list[Image.Image]:
    """Generate orientation variants for rotation-invariant matching.

    Covers all 4 rotations × horizontal flip = 8 variants.  This handles
    ceiling-mounted cameras, rotated phone photos, and mirrored feeds.
    """
    flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
    variants = [
        image,                                         # Original
        image.transpose(Image.ROTATE_90),              # 90° CCW
        image.transpose(Image.ROTATE_180),             # 180°
        image.transpose(Image.ROTATE_270),             # 270° CCW
        flipped,                                       # Horizontal mirror
        flipped.transpose(Image.ROTATE_90),            # Mirror + 90°
        flipped.transpose(Image.ROTATE_180),           # Mirror + 180°
        flipped.transpose(Image.ROTATE_270),           # Mirror + 270°
    ]
    return variants


def _normalize_lighting_cv(img_bgr: np.ndarray) -> np.ndarray:
    """Apply CLAHE normalization for day/night lighting invariance.

    Converts to LAB color space, equalizes the L (lightness) channel while
    preserving color information.  Standard preprocessing in surveillance
    systems — makes a sunlit person produce a similar embedding to the same
    person under dim streetlights.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def _normalize_lighting_pil(image: Image.Image) -> Image.Image:
    """PIL wrapper around CLAHE normalization."""
    cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    normalized = _normalize_lighting_cv(cv_img)
    return Image.fromarray(cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB))


def _build_person_query_variants(image: Image.Image) -> list[dict]:
    """Build person-query variants with explicit orientation metadata."""
    if getattr(config, "PERSON_MULTI_ORIENTATION", False):
        orientations = _generate_query_orientations(image)
        print(f"[search] Multi-orientation: {len(orientations)} variants")
    else:
        orientations = [image]

    variants = []
    for orientation_index, oriented_image in enumerate(orientations):
        variants.append({
            "image": oriented_image,
            "label": f"orient_{orientation_index}",
            "is_original": orientation_index == 0,
            "is_lighting_normalized": False,
        })
        if getattr(config, "PERSON_LIGHTING_NORMALIZE", False):
            variants.append({
                "image": _normalize_lighting_pil(oriented_image),
                "label": f"orient_{orientation_index}_clahe",
                "is_original": orientation_index == 0,
                "is_lighting_normalized": True,
            })
    return variants


def _search_clip_fallback_person(image: Image.Image, top_k: int,
                                  seen_keys: set) -> list:
    """Layer 3: Search the general CLIP index for person-class detections.

    CLIP was trained on 400M image-text pairs and is far more robust to extreme
    viewpoint and lighting changes than OSNet. Results are returned at a lower
    confidence tier (CLIP_FALLBACK_SCORE_WEIGHT) so they never outrank true
    biometric matches but still surface the person when all else fails.
    """
    results = []
    index = _load_faiss_index()
    if index is None or index.ntotal == 0:
        return results

    try:
        query_emb = models.encode_images_clip([image])
        n_candidates = min(
            max(top_k * config.IMAGE_CANDIDATE_POOL_FACTOR, config.IMAGE_MIN_CANDIDATES),
            index.ntotal,
        )
        scores, indices = index.search(query_emb, n_candidates)

        valid = [(int(indices[0][i]), float(scores[0][i]))
                 for i in range(len(indices[0]))
                 if indices[0][i] >= 0 and float(scores[0][i]) >= 0.20]

        if not valid:
            return results

        faiss_ids = [fid for fid, _ in valid]
        score_map = {fid: score for fid, score in valid}

        det_results = db.get_detections_by_faiss_ids(faiss_ids)
        for r in det_results:
            r = dict(r)  # sqlite3.Row → dict for .get()
            # Only keep person-class detections
            if str(r.get("class_name", "")).lower() != "person":
                continue
            key = f"person_{r['id']}"
            if key in seen_keys:
                continue
            seen_keys.add(key)
            result = _format_detection_result(r, source="clip_fallback")
            raw_score = score_map.get(r.get("faiss_id"), 0)
            result["score"] = raw_score * config.CLIP_FALLBACK_SCORE_WEIGHT
            result["match_type"] = "visual"
            results.append(result)

        print(f"[search] CLIP fallback: {len(results)} person matches")
    except Exception as e:
        print(f"[search] CLIP fallback error: {e}")

    return results


@_timed_operation("person_search")
def search_by_person(image: Image.Image, top_k: int = None):
    """Search by person photo — find this specific person across all indexed videos.

    3-Layer cross-camera strategy for real security scenarios
    (phone photo → CCTV match across different cameras/angles/lighting):

    Layer 1: Multi-Orientation — tests 8 rotated/flipped variants of the query
             to handle ceiling mounts, rotated feeds, different phone angles.
    Layer 2: CLAHE Lighting Normalization — equalizes brightness so a daylight
             phone photo matches the same person in dark nighttime CCTV.
    Layer 3: CLIP Fallback — searches the general CLIP visual index for
             person-class matches as a safety net when biometric models fail
             due to extreme viewpoint/resolution differences.

    Biometric models used:
    1. OSNet (Body Re-ID) — matches clothing, build, and appearance patterns.
    2. FaceNet (Face ID) — matches bone structure and facial landmarks.

    Score hierarchy:  face+body > face > body > CLIP visual
    """
    if top_k is None:
        top_k = config.DEFAULT_TOP_K

    seen_keys = set()
    results = []

    # ------------------------------------------------------------------
    # Build query variants (Layer 1 + Layer 2)
    # ------------------------------------------------------------------
    query_variants = _build_person_query_variants(image)

    print(f"[search] Total query variants: {len(query_variants)} "
          f"(orientations={'on' if getattr(config, 'PERSON_MULTI_ORIENTATION', False) else 'off'}, "
          f"CLAHE={'on' if getattr(config, 'PERSON_LIGHTING_NORMALIZE', False) else 'off'})")

    # ------------------------------------------------------------------
    # Detect face — ONLY on the original uploaded image (orient_0).
    #
    # YOLO-Face and FaceNet are trained on upright, properly-oriented
    # faces. Running them on ANY rotated/flipped variant produces false
    # face detections from body textures and clothing patterns. FaceNet
    # then encodes these garbage crops into embeddings that match wrong
    # people with HIGH confidence, poisoning the results.
    #
    # Strategy: Face matching works on the image AS UPLOADED.
    # All 8 rotated variants are handled by OSNet body Re-ID only.
    # ------------------------------------------------------------------
    face_variants = []  # list of (cv_img_bgr, face_bbox, is_face_crop, variant_label)

    # Only process the original orientation for face detection
    original_pil = query_variants[0]["image"]  # orient_0
    cv_img = cv2.cvtColor(np.array(original_pil), cv2.COLOR_RGB2BGR)
    h, w = cv_img.shape[:2]
    img_area = h * w

    faces = models.detect_faces(cv_img)
    if faces:
        best_face = max(faces, key=lambda f: (f["bbox"][2]-f["bbox"][0]) * (f["bbox"][3]-f["bbox"][1]))
        best_bbox = best_face["bbox"]
        face_area = (best_bbox[2]-best_bbox[0]) * (best_bbox[3]-best_bbox[1])
        is_face_crop = face_area > 0.4 * img_area
        face_variants.append((cv_img, best_bbox, is_face_crop, "orient_0"))
        print(f"[search] Face detected in query image (conf: {best_face['confidence']:.2f})")

        # Also try CLAHE-normalized version for lighting-invariant face encoding
        if getattr(config, "PERSON_LIGHTING_NORMALIZE", False):
            clahe_cv = _normalize_lighting_cv(cv_img)
            face_variants.append((clahe_cv, best_bbox, is_face_crop, "orient_0_clahe"))
    else:
        print("[search] No face detected in original query image.")

    print(f"[search] Faces detected in {len(face_variants)}/{len(query_variants)} variants")

    # Determine if the original query is a tight face crop (skip body Re-ID)
    original_is_face_crop = False
    if face_variants:
        # Check if the original (first) orientation is a face crop
        for cv_img, bbox, is_face_crop, label in face_variants:
            if label == "orient_0":
                original_is_face_crop = is_face_crop
                break

    # ------------------------------------------------------------------
    # Stage 1: OSNet Body Re-ID across all orientations (batch-encoded)
    # ------------------------------------------------------------------
    person_index = _get_cached_faiss(config.PERSON_FAISS_INDEX_PATH)

    if not original_is_face_crop and person_index is not None and person_index.ntotal > 0:
        # Batch-encode all variants in a single Re-ID call.
        body_images = [variant["image"] for variant in query_variants]
        body_embeddings = models.encode_person_reid(body_images)

        n_candidates = min(
            max(top_k * config.PERSON_CANDIDATE_POOL_FACTOR, config.PERSON_MIN_CANDIDATES),
            person_index.ntotal,
        )

        # Search with EACH embedding and collect best score per FAISS ID
        best_scores = {}  # faiss_id → best_score
        for emb_idx in range(len(body_embeddings)):
            query_emb = body_embeddings[emb_idx:emb_idx+1]
            scores, indices = person_index.search(query_emb, n_candidates)
            
            is_original = query_variants[emb_idx]["is_original"]
            threshold = config.PERSON_MATCH_THRESHOLD if is_original else config.PERSON_MATCH_THRESHOLD_RELAXED
            discount = 1.0 if is_original else config.BODY_ROTATED_DISCOUNT

            for i in range(len(indices[0])):
                fid = int(indices[0][i])
                score = float(scores[0][i])
                if fid >= 0 and score >= threshold:
                    discounted_score = score * discount
                    if fid not in best_scores or discounted_score > best_scores[fid]:
                        best_scores[fid] = discounted_score

        print(f"[search] Person Re-ID (multi-orient): {len(best_scores)} unique matches")

        if best_scores:
            faiss_ids = list(best_scores.keys())
            person_results = db.get_persons_by_faiss_ids(faiss_ids)
            for r in person_results:
                key = f"person_{r['id']}"
                if key not in seen_keys:
                    seen_keys.add(key)
                    result = _format_detection_result(r, source="person_reid")
                    raw_score = best_scores.get(r["person_faiss_id"], 0)
                    result["score"] = raw_score * config.BODY_ONLY_DISCOUNT
                    result["match_type"] = "body"
                    results.append(result)
    elif original_is_face_crop:
        print("[search] Query is a tight face crop. Skipping Body Re-ID.")
    else:
        print("[search] Person FAISS index empty or missing. Skipping Body Re-ID.")

    # ------------------------------------------------------------------
    # Stage 2: FaceNet across face-detected orientations
    # ------------------------------------------------------------------
    face_index = _get_cached_faiss(config.FACE_FAISS_INDEX_PATH)

    if face_variants and face_index is not None and face_index.ntotal > 0:
        n_candidates = min(
            max(top_k * config.FACE_CANDIDATE_POOL_FACTOR, config.FACE_MIN_CANDIDATES),
            face_index.ntotal,
        )

        # Collect best face scores across all orientations
        best_face_scores = {}  # faiss_id → best_score
        for cv_img, best_bbox, _, label in face_variants:
            query_face_emb = models.encode_face_identity(cv_img, best_bbox)
            if np.all(query_face_emb == 0):
                continue
            query_face_emb = query_face_emb.reshape(1, -1)

            scores, indices = face_index.search(query_face_emb, n_candidates)
            for i in range(len(indices[0])):
                fid = int(indices[0][i])
                score = float(scores[0][i])
                if fid >= 0 and score >= config.FACE_MATCH_THRESHOLD:
                    if fid not in best_face_scores or score > best_face_scores[fid]:
                        best_face_scores[fid] = score

        print(f"[search] Face match (multi-orient): {len(best_face_scores)} unique matches "
              f"(threshold {config.FACE_MATCH_THRESHOLD})")

        if best_face_scores:
            faiss_ids = list(best_face_scores.keys())
            face_results = db.get_faces_by_faiss_ids(faiss_ids)
            for r in face_results:
                r = dict(r)  # sqlite3.Row → dict for .get()
                det_id = r["detection_id"]
                key = f"person_{det_id}"
                face_score = best_face_scores.get(r["face_faiss_id"], 0)

                if key in seen_keys:
                    # Already found via body — boost with face score
                    for existing in results:
                        if existing.get("detection_id") == det_id:
                            existing["score"] = existing["score"] * 0.3 + face_score * 0.7
                            existing["match_type"] = "body+face"
                            break
                else:
                    seen_keys.add(key)
                    result = {
                        "type": "detection", "source": "face",
                        "detection_id": det_id,
                        "track_id": r.get("track_id"),
                        "video_id": r.get("video_id"),
                        "frame_path": r["frame_path"],
                        "crop_path": r.get("person_crop_path", r.get("crop_path", "")),
                        "face_crop_path": r.get("crop_path", ""),
                        "timestamp": r["timestamp"],
                        "frame_number": r["frame_number"],
                        "video_filename": r["video_filename"],
                        "video_path": r["video_path"],
                        "camera_id": r.get("camera_id"),
                        "location": r.get("location"),
                        "recording_start_time": r.get("recording_start_time"),
                        "timezone": r.get("timezone"),
                        "class_name": "person",
                        "confidence": r.get("confidence", 0),
                        "caption": "face match",
                        "bbox": None,
                        "score": face_score * 0.9,  # Face-only match
                        "match_type": "face",
                    }
                    results.append(result)

    # ------------------------------------------------------------------
    # Stage 3: CLIP Fallback (Layer 3 — catches extreme viewpoint changes)
    # ------------------------------------------------------------------
    if getattr(config, "PERSON_CLIP_FALLBACK", False):
        clip_results = _search_clip_fallback_person(image, top_k, seen_keys)
        results.extend(clip_results)

    # ------------------------------------------------------------------
    # Stage 4: Color-Aware Re-ranking ("Same Clothes" Maximization)
    # ------------------------------------------------------------------
    # If the user provided a full body image, color (clothing) is the most
    # reliable feature for tracking the exact same person across cameras.
    if not original_is_face_crop and results:
        # High weight (0.75) to aggressively prioritize clothing color (same clothes)
        results = _rerank_by_color(results, image, weight=0.75)

    # Sort by combined score
    results.sort(key=lambda x: x.get("score", 0), reverse=True)
    
    # Apply temporal boosting
    results = _apply_temporal_boosting(results)

    results = _cluster_near_duplicates(results)

    return results[:top_k]


# ---- Helpers ----

# FTS5 treats these as syntax; feeding raw user input causes OperationalError
# ("no such column", "unterminated string", …). We strip them and quote each
# surviving token so any character inside becomes a literal.
_FTS5_STRIP = str.maketrans({
    '"': " ", "'": " ", "(": " ", ")": " ",
    "*": " ", ":": " ", "^": " ", "-": " ",
    "+": " ", "~": " ", "[": " ", "]": " ",
    "{": " ", "}": " ", "\\": " ",
    # SQLite treats NUL as C-string terminator; passing it into a bound
    # parameter raises "unterminated string" before FTS5 even sees the query.
    "\x00": " ",
})
_FTS5_RESERVED = {"and", "or", "not", "near"}


def _build_fts_query(query: str) -> str:
    """Build a safe FTS5 MATCH query from free-form user text.

    - Strips FTS5 syntax characters (and NUL bytes).
    - Drops any other ASCII control characters.
    - Filters out reserved operator words (case-insensitive).
    - Double-quotes each surviving term so its internal contents are literal.
    - Falls back to a single quoted phrase (or the empty string) when no
      usable tokens remain.
    """
    if not query:
        return '""'
    # Strip control characters first so tokens like ``hello\x01world`` become
    # ``helloworld`` instead of being kept as-is.
    cleaned_chars = []
    for ch in query:
        if ord(ch) < 32 and ch not in "\t\n\r":
            continue
        cleaned_chars.append(ch)
    cleaned = "".join(cleaned_chars).translate(_FTS5_STRIP)
    tokens = [t for t in cleaned.split() if len(t) > 2 and t.lower() not in _FTS5_RESERVED]
    if not tokens:
        fallback = cleaned.strip()
        if not fallback:
            return '""'
        return '"' + fallback.replace('"', "") + '"'
    return " OR ".join(f'"{t}"' for t in tokens)


def _format_detection_result(row, source="text"):
    r = dict(row)
    # FTS5 BM25 rank is negative; smaller value = better match. Normalise to (0,1]
    # so it can be merged with cosine-similarity CLIP scores without one dominating.
    raw_rank = r.get("rank")
    score = 1.0 / (1.0 + abs(raw_rank)) if raw_rank is not None else 0
    return {
        "type": "detection", "source": source,
        "detection_id": r["id"],
        "track_id": r.get("track_id"),
        "video_id": r.get("video_id"),
        "frame_path": r["frame_path"],
        "crop_path": r.get("crop_path", ""),
        "timestamp": r["timestamp"],
        "frame_number": r["frame_number"],
        "video_filename": r["video_filename"],
        "video_path": r["video_path"],
        "camera_id": r.get("camera_id"),
        "location": r.get("location"),
        "recording_start_time": r.get("recording_start_time"),
        "timezone": r.get("timezone"),
        "class_name": r.get("class_name", ""),
        "confidence": r.get("confidence", 0),
        "caption": r.get("caption", ""),
        "bbox": [r.get("bbox_x1", 0), r.get("bbox_y1", 0),
             r.get("bbox_x2", 0), r.get("bbox_y2", 0)] if r.get("bbox_x1") is not None else None,
        "score": score,
    }


def _format_scene_result(row, source="text"):
    r = dict(row)
    raw_rank = r.get("rank")
    score = 1.0 / (1.0 + abs(raw_rank)) if raw_rank is not None else 0
    return {
        "type": "scene", "source": source,
        "video_id": r.get("video_id"),
        "frame_path": r["frame_path"],
        "crop_path": "",
        "timestamp": r["timestamp"],
        "frame_number": r["frame_number"],
        "video_filename": r["video_filename"],
        "video_path": r["video_path"],
        "camera_id": r.get("camera_id"),
        "location": r.get("location"),
        "recording_start_time": r.get("recording_start_time"),
        "timezone": r.get("timezone"),
        "class_name": "scene",
        "confidence": 1.0,
        "caption": r.get("caption", ""),
        "bbox": None,
        "score": score,
    }


def _cluster_near_duplicates(results: list, window_frames: int = 2) -> list:
    """Collapse near-duplicate thumbnails (same video, adjacent frames) into one card.

    A single real event often produces multiple detections across consecutive
    extracted frames; showing all of them wastes result slots. Keeps the best
    scoring member of each cluster and stores siblings in ``_cluster``.
    """
    if not results:
        return results

    # Sort deterministically by (video, timestamp) so adjacent frames are neighbours.
    ordered = sorted(results, key=lambda r: (_video_group_key(r),
                                             float(r.get("frame_number") or 0)))
    kept: list = []
    for r in ordered:
        vid = _video_group_key(r)
        fno = float(r.get("frame_number") or 0)
        merged = False
        for k in kept:
            if _video_group_key(k) == vid and abs(float(k.get("frame_number") or 0) - fno) <= window_frames:
                # Prefer higher-scored representative; append the loser to _cluster.
                loser = r if k.get("score", 0) >= r.get("score", 0) else k
                winner = k if loser is r else r
                if winner is r:
                    kept.remove(k)
                    kept.append(winner)
                winner.setdefault("_cluster", []).append({
                    "frame_number": loser.get("frame_number"),
                    "timestamp": loser.get("timestamp"),
                    "score": loser.get("score"),
                    "frame_path": loser.get("frame_path"),
                })
                merged = True
                break
        if not merged:
            kept.append(r)

    kept.sort(key=lambda r: r.get("score", 0), reverse=True)
    return kept
