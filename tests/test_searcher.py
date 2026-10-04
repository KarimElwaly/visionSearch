import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import config
import searcher


class _FakeIndex:
    ntotal = 20

    def search(self, _embedding, count):
        ids = np.arange(count, dtype=np.int64).reshape(1, -1)
        scores = np.linspace(0.95, 0.50, count, dtype=np.float32).reshape(1, -1)
        return scores, ids


class _FakeBox:
    def __init__(self, confidence=0.8, class_index=0):
        self.conf = [confidence]
        self.cls = [class_index]


class _FakeResult:
    def __init__(self, boxes=None):
        self.boxes = boxes if boxes is not None else [_FakeBox()]


class _FakeCropBox:
    def __init__(self, bounds):
        self.xyxy = [bounds]


class _FakeWorldModel:
    def __init__(self):
        self.classes = None
        self.batch_lengths = []

    def set_classes(self, classes):
        self.classes = classes

    def __call__(self, frames, **_kwargs):
        self.batch_lengths.append(len(frames))
        return [_FakeResult() for _ in frames]


def _detection_row(identifier):
    return {
        "id": identifier,
        "faiss_id": identifier,
        "video_id": identifier + 100,
        "frame_path": f"frame_{identifier}.jpg",
        "crop_path": "",
        "timestamp": float(identifier),
        "frame_number": identifier,
        "video_filename": f"video_{identifier}.mp4",
        "video_path": f"/archive/video_{identifier}.mp4",
        "class_name": "person",
        "confidence": 0.9,
        "caption": "person",
        "bbox_x1": 1,
        "bbox_y1": 2,
        "bbox_x2": 3,
        "bbox_y2": 4,
    }


class SearcherRegressionTests(unittest.TestCase):
    def test_search_reranks_larger_recall_pool(self):
        captured = []

        def fake_rerank(candidates, _query, top_k):
            captured.extend(candidates)
            return candidates[:top_k]

        rows = [_detection_row(i) for i in range(20)]
        with (
            patch.object(searcher, "_load_faiss_index", return_value=_FakeIndex()),
            patch.object(searcher, "_get_query_embedding", return_value=np.zeros((1, 768), dtype=np.float32)),
            patch.object(searcher.db, "get_detections_by_faiss_ids", return_value=rows),
            patch.object(searcher.db, "get_frame_embeddings_by_faiss_ids", return_value=[]),
            patch.object(searcher, "_rerank_with_yolo_world", side_effect=fake_rerank),
        ):
            searcher.search("red bag", top_k=2, mode="clip")

        self.assertEqual(len(captured), 6)

    def test_yolo_world_infers_duplicate_frame_once(self):
        model = _FakeWorldModel()
        candidates = [
            {"frame_path": "same.jpg", "score": 0.4},
            {"frame_path": "same.jpg", "score": 0.3},
        ]

        with (
            patch.object(searcher, "_load_yolo_world", return_value=model),
            patch.object(searcher.os.path, "exists", return_value=True),
            patch.object(searcher.cv2, "imread", return_value=np.zeros((32, 32, 3), dtype=np.uint8)),
        ):
            results = searcher._rerank_with_yolo_world(candidates, "red bag", top_k=2)

        self.assertEqual(model.batch_lengths, [1])
        self.assertTrue(all(result["yolo_world_match"] for result in results))

    def test_temporal_boost_does_not_mix_same_filename(self):
        results = [
            {"video_id": 1, "video_filename": "camera.mp4", "timestamp": 1.0, "score": 1.0},
            {"video_id": 2, "video_filename": "camera.mp4", "timestamp": 2.0, "score": 1.0},
        ]

        boosted = searcher._apply_temporal_boosting(results)

        self.assertEqual([result["score"] for result in boosted], [0.9, 0.9])

    def test_duplicate_clustering_does_not_mix_same_filename(self):
        results = [
            {"video_id": 1, "video_filename": "camera.mp4", "frame_number": 10, "score": 1.0},
            {"video_id": 2, "video_filename": "camera.mp4", "frame_number": 11, "score": 0.9},
        ]

        self.assertEqual(len(searcher._cluster_near_duplicates(results)), 2)

    def test_zero_x_coordinate_preserves_bbox(self):
        row = _detection_row(1)
        row["bbox_x1"] = 0

        result = searcher._format_detection_result(row)

        self.assertEqual(result["bbox"], [0, 2, 3, 4])

    def test_rotated_variant_is_not_original_without_clahe(self):
        image = Image.new("RGB", (20, 10), "white")
        with (
            patch.object(config, "PERSON_MULTI_ORIENTATION", True),
            patch.object(config, "PERSON_LIGHTING_NORMALIZE", False),
        ):
            variants = searcher._build_person_query_variants(image)

        self.assertEqual(len(variants), 8)
        self.assertTrue(variants[0]["is_original"])
        self.assertFalse(variants[1]["is_original"])

    def test_image_query_uses_multiple_largest_object_crops(self):
        image = Image.new("RGB", (100, 80), "white")
        yolo_results = [_FakeResult([
            _FakeCropBox([0, 0, 10, 10]),
            _FakeCropBox([10, 10, 90, 70]),
            _FakeCropBox([20, 20, 60, 60]),
        ])]

        with (
            patch.object(config, "IMAGE_SEARCH_MAX_OBJECTS", 2),
            patch.object(config, "IMAGE_SEARCH_INCLUDE_FULL_IMAGE", True),
        ):
            variants = searcher._build_image_query_variants(image, yolo_results)

        self.assertEqual(len(variants), 3)
        self.assertEqual(variants[0].size, (100, 80))
        self.assertEqual(variants[1].size, (80, 60))
        self.assertEqual(variants[2].size, (40, 40))

    def test_prompt_templates_do_not_become_attributes(self):
        encoded_prompts = []

        def fake_encode(prompt):
            encoded_prompts.append(prompt)
            return np.ones((1, 4), dtype=np.float32)

        with patch.object(searcher.models, "encode_text_clip", side_effect=fake_encode):
            searcher._prompt_ensemble_embedding("person")

        self.assertEqual(len(encoded_prompts), len(searcher._PROMPT_TEMPLATES))
        self.assertNotIn("photo of", encoded_prompts)
        self.assertNotIn("picture of", encoded_prompts)


if __name__ == "__main__":
    unittest.main()