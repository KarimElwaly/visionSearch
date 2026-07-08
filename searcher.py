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
import numpy as np
import faiss
from PIL import Image
import cv2

import config
import database as db
import models


def _load_faiss_index():
    if os.path.exists(config.FAISS_INDEX_PATH):
        return faiss.read_index(config.FAISS_INDEX_PATH)
    return None


# ---- YOLO-World for open-vocabulary re-ranking ----

_yolo_world_model = None


def _load_yolo_world():
    global _yolo_world_model
    if _yolo_world_model is None:
        from ultralytics import YOLO
        import clip as clip_pkg
        model_path = config.YOLO_WORLD_MODEL_PATH
        print(f"[search] Loading YOLO-World: {model_path}")
        _yolo_world_model = YOLO(model_path)
        _yolo_world_model.to("cpu")
        # Pre-load YOLO-World's internal CLIP text encoder on CPU
        # (set_classes uses clip.load() which defaults to CUDA and crashes)
        clip_model, _ = clip_pkg.load("ViT-B/32", device="cpu")
        _yolo_world_model.model.clip = clip_model
        print("[search] YOLO-World loaded (CPU).")
    return _yolo_world_model


def _rerank_with_yolo_world(candidates: list, query: str, top_k: int) -> list:
    """
    Stage 2: Re-rank candidates using YOLO-World open-vocabulary detection.
    Uses the FULL query as the detection class (not individual words).
    "white jacket" → detects white jackets, NOT any jacket.
    """
    try:
        model = _load_yolo_world()
    except Exception as e:
        print(f"[search] YOLO-World not available: {e}, skipping re-ranking")
        return candidates

    # Use the full query (e.g., "black bag") and the core object (e.g., "bag")
    # This allows YOLO-World to still find the object if it struggles with the color attribute
    words = query.lower().split()
    core_nouns = [w for w in words if len(w) > 3 and w not in 
                  {"with", "wearing", "that", "this", "from", "into", "near", "next", "have", "some", "black", "white", "red", "blue", "green", "yellow"}]
    
    query_classes = [query.strip()]
    if core_nouns and core_nouns[-1] != query.strip():
        query_classes.append(core_nouns[-1])

    try:
        model.set_classes(query_classes)
    except Exception as e:
        print(f"[search] YOLO-World set_classes failed: {e}, skipping re-ranking")
        return candidates
    
    print(f"[search] YOLO-World classes: {query_classes}")

    for candidate in candidates:
        frame_path = candidate.get("frame_path", "")
        if not frame_path or not os.path.exists(frame_path):
            continue

        try:
            frame = cv2.imread(frame_path)
            if frame is None:
                continue

            # Search with stricter confidence threshold (0.15 instead of 0.10)
            results = model(frame, conf=0.15, verbose=False)
            
            # Count detections and get max confidence
            max_conf = 0.0
            det_count = 0
            best_class_idx = -1
            
            for r in results:
                for box in r.boxes:
                    det_count += 1
                    conf = float(box.conf[0])
                    cls_idx = int(box.cls[0])
                    if conf > max_conf:
                        max_conf = conf
                        best_class_idx = cls_idx

            # Boost score based on YOLO-World detections
            if det_count > 0:
                # Strong boost if it matched the exact full query (class index 0)
                if best_class_idx == 0:
                    candidate["score"] = candidate.get("score", 0) + max_conf * 0.8
                # Minor boost if it only matched the fallback noun
                else:
                    candidate["score"] = candidate.get("score", 0) + max_conf * 0.3
                    
                candidate["yolo_world_match"] = True
                candidate["yolo_world_conf"] = max_conf
                candidate["yolo_world_count"] = det_count
            else:
                # STRONG PENALTY: YOLO-World completely failed to find the object
                # Drop the score aggressively so it falls below actually matched items
                candidate["score"] = candidate.get("score", 0) * 0.1
                candidate["yolo_world_match"] = False

        except Exception as e:
            print(f"[search] YOLO-World error on {frame_path}: {e}")

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


def _get_query_embedding(query: str) -> np.ndarray:
    """
    Parses negative prompts (--no) and applies attribute blending.
    Returns forced unit-length embedding.
    """
    query_lower = query.lower()
    if " --no " in query_lower:
        parts = query_lower.split(" --no ", 1)
        pos_query = parts[0].strip()
        neg_query = parts[1].strip()
        
        pos_emb = _blend_attributes(pos_query)
        neg_emb = _blend_attributes(neg_query)
        
        # Vector Subtraction (Option 2)
        print(f"[search] Negative Prompt applied: '{pos_query}' minus '{neg_query}'")
        blended = pos_emb - (neg_emb * 0.5)  # 0.5 strength for negative
        return blended / np.linalg.norm(blended, axis=1, keepdims=True)
    else:
        return _blend_attributes(query)


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
        vid = r.get("video_filename")
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
            
            # Get more candidates than needed — Stage 2 will filter
            n_candidates = min(top_k * 5, index.ntotal)
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
        # Re-rank the number of results requested by the user
        candidates_to_rerank = results[:top_k]
        print(f"[search] Stage 2 YOLO-World: re-ranking {len(candidates_to_rerank)} candidates")
        
        # We only keep the ones that survived re-ranking
        results = _rerank_with_yolo_world(candidates_to_rerank, query, top_k)
        
        # Filter out results that got the massive penalty (score < 0.05)
        results = [r for r in results if r.get("score", 0) > 0.05]

    # Apply Option 5: Temporal Continuity Boost
    results = _apply_temporal_boosting(results)

    return results[:top_k]


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

    index = _load_faiss_index()
    if index is None or index.ntotal == 0:
        return results

    try:
        # Option 1: Object-Centric Image Search
        # Use YOLOv11 to find the primary object in the uploaded image and crop out the background.
        # This prevents background noise (e.g. a messy room) from polluting the CLIP embedding.
        from ultralytics import YOLO
        print("[search] Finding primary object in uploaded image...")
        
        # Convert PIL to CV2 format for YOLO
        cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        yolo_model = YOLO(config.YOLO_MODEL_PATH)
        yolo_model.to("cpu")
        yolo_results = yolo_model(cv_img, conf=0.2, verbose=False)
        
        best_crop = None
        max_area = 0
        
        for r in yolo_results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                # Find the largest bounding box
                if area > max_area:
                    max_area = area
                    best_crop = image.crop((x1, y1, x2, y2))
                    
        if best_crop:
            print("[search] Found primary object, cropping background.")
            query_emb = models.encode_images_clip([best_crop])
        else:
            print("[search] No clear object found, using full image.")
            query_emb = models.encode_images_clip([image])

        n_candidates = min(top_k * 3, index.ntotal)
        scores, indices = index.search(query_emb, n_candidates)

        valid = [(int(indices[0][i]), float(scores[0][i]))
                 for i in range(len(indices[0]))
                 if indices[0][i] >= 0 and float(scores[0][i]) >= min_clip_score]

        faiss_ids = [fid for fid, _ in valid]
        score_map = {fid: score for fid, score in valid}

        print(f"[search] Image search: {len(valid)} matches (threshold {min_clip_score})")

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
    
    # Apply Option 5: Temporal Continuity Boost
    results = _apply_temporal_boosting(results)
    
    return results[:top_k]


def search_by_person(image: Image.Image, top_k: int = None):
    """
    Search by person photo — find this specific person across all indexed videos.
    Uses dedicated biometric models for high-accuracy security tracking:
    1. DINOv2 (Body Re-ID) — mathematically matches clothing/gait patterns.
    2. FaceNet (Face ID) — matches bone structure and landmarks.
    """
    if top_k is None:
        top_k = config.DEFAULT_TOP_K

    seen_keys = set()
    results = []

    import cv2 as cv2_local
    cv_img = cv2_local.cvtColor(np.array(image), cv2_local.COLOR_RGB2BGR)
    h, w = cv_img.shape[:2]
    img_area = h * w

    # Detect and encode face identity using FaceNet
    query_face_emb = None
    faces = models.detect_faces(cv_img)
    
    is_face_crop = False
    best_bbox = None
    
    if faces:
        best_face = max(faces, key=lambda f: (f["bbox"][2]-f["bbox"][0]) * (f["bbox"][3]-f["bbox"][1]))
        best_bbox = best_face["bbox"]
        face_area = (best_bbox[2]-best_bbox[0]) * (best_bbox[3]-best_bbox[1])
        if face_area > 0.4 * img_area:
            is_face_crop = True
        print(f"[search] Face detected in query image (conf: {best_face['confidence']:.2f})")
    else:
        # YOLO-Face sometimes misses extreme tight crops. Let's assume it *might* be a face
        # if the aspect ratio is face-like, but to be safe we'll try to encode the whole image as a face.
        best_bbox = (0, 0, w, h)
        print("[search] No face detected by YOLO, attempting full-image FaceNet extraction.")

    # Always try to extract face identity from the bbox
    query_face_emb = models.encode_face_identity(cv_img, best_bbox)
    # encode_face_identity returns all zeros if it fails
    if np.all(query_face_emb == 0):
        query_face_emb = None
    else:
        query_face_emb = query_face_emb.reshape(1, -1)

    # Encode with DINOv2 (Biometric Re-ID) ONLY if it's not a tight face crop
    query_emb = None
    if not is_face_crop:
        print("[search] Query image contains a body. Running DINOv2 Re-ID.")
        query_emb = models.encode_person_reid([image])
    else:
        print("[search] Query image is a tight face crop. Disabling Body Re-ID to prevent inaccurate clothing matches.")

    # --- Stage 1: Person Re-ID Index ---
    person_index_path = config.PERSON_FAISS_INDEX_PATH
    if os.path.exists(person_index_path) and query_emb is not None:
        person_index = faiss.read_index(person_index_path)
        if person_index.ntotal > 0:
            n_candidates = min(top_k * 3, person_index.ntotal)
            scores, indices = person_index.search(query_emb, n_candidates)
            
            valid = [(int(indices[0][i]), float(scores[0][i]))
                     for i in range(len(indices[0]))
                     if indices[0][i] >= 0 and float(scores[0][i]) >= config.PERSON_MATCH_THRESHOLD]
            
            print(f"[search] Person Re-ID: {len(valid)} matches (threshold {config.PERSON_MATCH_THRESHOLD})")
            
            if valid:
                faiss_ids = [fid for fid, _ in valid]
                score_map = {fid: score for fid, score in valid}
                
                person_results = db.get_persons_by_faiss_ids(faiss_ids)
                for r in person_results:
                    key = f"person_{r['id']}"
                    if key not in seen_keys:
                        seen_keys.add(key)
                        result = _format_detection_result(r, source="person_reid")
                        result["score"] = score_map.get(r["person_faiss_id"], 0)
                        result["match_type"] = "body"
                        results.append(result)

    # --- Stage 2: Face Index ---
    if query_face_emb is not None:
        face_index_path = config.FACE_FAISS_INDEX_PATH
        if os.path.exists(face_index_path):
            face_index = faiss.read_index(face_index_path)
            if face_index.ntotal > 0:
                n_candidates = min(top_k * 3, face_index.ntotal)
                scores, indices = face_index.search(query_face_emb, n_candidates)
                
                valid = [(int(indices[0][i]), float(scores[0][i]))
                         for i in range(len(indices[0]))
                         if indices[0][i] >= 0 and float(scores[0][i]) >= config.FACE_MATCH_THRESHOLD]
                
                print(f"[search] Face match: {len(valid)} matches (threshold {config.FACE_MATCH_THRESHOLD})")
                
                if valid:
                    faiss_ids = [fid for fid, _ in valid]
                    score_map = {fid: score for fid, score in valid}
                    
                    face_results = db.get_faces_by_faiss_ids(faiss_ids)
                    for r in face_results:
                        r = dict(r)  # sqlite3.Row → dict for .get()
                        det_id = r["detection_id"]
                        key = f"person_{det_id}"
                        face_score = score_map.get(r["face_faiss_id"], 0)
                        
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
                                "frame_path": r["frame_path"],
                                "crop_path": r.get("person_crop_path", r.get("crop_path", "")),
                                "face_crop_path": r.get("crop_path", ""),
                                "timestamp": r["timestamp"],
                                "frame_number": r["frame_number"],
                                "video_filename": r["video_filename"],
                                "video_path": r["video_path"],
                                "class_name": "person",
                                "confidence": r.get("confidence", 0),
                                "caption": "face match",
                                "bbox": None,
                                "score": face_score * 0.9,  # Face-only match
                                "match_type": "face",
                            }
                            results.append(result)

    # Sort by combined score
    results.sort(key=lambda x: x.get("score", 0), reverse=True)
    
    # Apply temporal boosting
    results = _apply_temporal_boosting(results)
    
    return results[:top_k]


# ---- Helpers ----

def _build_fts_query(query: str) -> str:
    words = [w for w in query.strip().split() if len(w) > 2]
    return " OR ".join(words) if words else query


def _format_detection_result(row, source="text"):
    r = dict(row)
    return {
        "type": "detection", "source": source,
        "detection_id": r["id"],
        "frame_path": r["frame_path"],
        "crop_path": r.get("crop_path", ""),
        "timestamp": r["timestamp"],
        "frame_number": r["frame_number"],
        "video_filename": r["video_filename"],
        "video_path": r["video_path"],
        "class_name": r.get("class_name", ""),
        "confidence": r.get("confidence", 0),
        "caption": r.get("caption", ""),
        "bbox": [r.get("bbox_x1", 0), r.get("bbox_y1", 0),
                 r.get("bbox_x2", 0), r.get("bbox_y2", 0)] if r.get("bbox_x1") else None,
        "score": abs(r["rank"]) if "rank" in r else 0,
    }


def _format_scene_result(row, source="text"):
    r = dict(row)
    return {
        "type": "scene", "source": source,
        "frame_path": r["frame_path"],
        "crop_path": "",
        "timestamp": r["timestamp"],
        "frame_number": r["frame_number"],
        "video_filename": r["video_filename"],
        "video_path": r["video_path"],
        "class_name": "scene",
        "confidence": 1.0,
        "caption": r.get("caption", ""),
        "bbox": None,
        "score": abs(r["rank"]) if "rank" in r else 0,
    }
