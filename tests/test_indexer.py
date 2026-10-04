import unittest
from unittest.mock import Mock, patch

import numpy as np

import indexer


class IndexerConfigurationTests(unittest.TestCase):
    def test_detection_overrides_are_passed_without_mutating_config(self):
        mega_batch = [{
            "frame_bgr_ref": np.zeros((8, 8, 3), dtype=np.uint8),
            "frame_no": 1,
            "timestamp": 0.5,
            "crop_images": [],
            "crop_records": [],
        }]
        tracker = Mock()
        tracker.update.return_value = []
        original_classes = indexer.config.YOLO_CLASSES
        original_confidence = indexer.config.YOLO_CONFIDENCE

        with patch.object(indexer.models, "detect_objects_batch", return_value=[[]]) as detect:
            count = indexer._detect_track_crop_batch(
                mega_batch,
                tracker,
                io_executor=None,
                video_frames_dir="unused",
                yolo_classes=[0, 2],
                yolo_confidence=0.77,
            )

        self.assertEqual(count, 0)
        detect.assert_called_once()
        self.assertEqual(detect.call_args.kwargs["classes"], [0, 2])
        self.assertEqual(detect.call_args.kwargs["confidence"], 0.77)
        self.assertIs(indexer.config.YOLO_CLASSES, original_classes)
        self.assertEqual(indexer.config.YOLO_CONFIDENCE, original_confidence)


if __name__ == "__main__":
    unittest.main()
