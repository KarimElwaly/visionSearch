"""Lightweight tracking helpers for CCTV-style indexing.

This module intentionally avoids heavy tracker dependencies. It provides a
small IoU tracker that groups frame-level YOLO detections into stable local
tracks, giving the rest of the app a track/event layer for long CCTV videos.
"""
from dataclasses import dataclass
from typing import Any


BBox = tuple[float, float, float, float]


def bbox_iou(a: BBox, b: BBox) -> float:
    """Return intersection-over-union for two ``(x1, y1, x2, y2)`` boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return float(inter_area / union)


@dataclass
class _TrackState:
    local_id: int
    class_name: str
    bbox: BBox
    start_frame_number: int
    end_frame_number: int
    start_timestamp: float
    end_timestamp: float
    hits: int = 1
    missed: int = 0
    best_confidence: float = 0.0


class SimpleIOUTracker:
    """Greedy per-class IoU tracker.

    It is not a full ByteTrack replacement, but it is deterministic, fast, and
    good enough to persist a first CCTV event layer without adding new native
    dependencies. Each input detection receives a stable ``track_local_id``.
    """

    def __init__(self, iou_threshold: float = 0.25, max_missed: int = 5, start_local_id: int = 1):
        self.iou_threshold = float(iou_threshold)
        self.max_missed = int(max_missed)
        self._next_local_id = max(1, int(start_local_id))
        self._active: dict[int, _TrackState] = {}

    def update(self, frame_number: int, timestamp: float, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Assign tracks to detections from one frame.

        Args:
            frame_number: Source video frame number.
            timestamp: Source video timestamp in seconds.
            detections: YOLO detection dicts containing ``class_name``, ``bbox``
                and optional ``confidence``.

        Returns:
            One assignment dict per input detection, in input order.
        """
        assignments: list[dict[str, Any] | None] = [None] * len(detections)
        unmatched_track_ids = set(self._active.keys())
        unmatched_det_indices = set(range(len(detections)))

        classes = sorted({str(d.get("class_name") or "object") for d in detections})
        for class_name in classes:
            track_ids = [
                tid for tid, state in self._active.items()
                if state.class_name == class_name and tid in unmatched_track_ids
            ]
            det_indices = [
                i for i in unmatched_det_indices
                if str(detections[i].get("class_name") or "object") == class_name
            ]
            pairs: list[tuple[float, int, int]] = []
            for tid in track_ids:
                state = self._active[tid]
                for di in det_indices:
                    iou = bbox_iou(state.bbox, tuple(detections[di]["bbox"]))
                    if iou >= self.iou_threshold:
                        pairs.append((iou, tid, di))

            pairs.sort(reverse=True, key=lambda p: p[0])
            for iou, tid, di in pairs:
                if tid not in unmatched_track_ids or di not in unmatched_det_indices:
                    continue
                self._update_track(tid, frame_number, timestamp, detections[di])
                assignments[di] = self._assignment(self._active[tid], iou=iou, is_new=False)
                unmatched_track_ids.remove(tid)
                unmatched_det_indices.remove(di)

        # New tracks for unmatched detections.
        for di in sorted(unmatched_det_indices):
            state = self._create_track(frame_number, timestamp, detections[di])
            assignments[di] = self._assignment(state, iou=1.0, is_new=True)

        # Age unmatched active tracks and retire stale ones.
        stale_ids = []
        for tid in list(unmatched_track_ids):
            state = self._active.get(tid)
            if state is None:
                continue
            state.missed += 1
            if state.missed > self.max_missed:
                stale_ids.append(tid)
        for tid in stale_ids:
            self._active.pop(tid, None)

        return [a for a in assignments if a is not None]

    @property
    def active_count(self) -> int:
        return len(self._active)

    def _create_track(self, frame_number: int, timestamp: float, detection: dict[str, Any]) -> _TrackState:
        local_id = self._next_local_id
        self._next_local_id += 1
        confidence = float(detection.get("confidence") or 0.0)
        state = _TrackState(
            local_id=local_id,
            class_name=str(detection.get("class_name") or "object"),
            bbox=tuple(detection["bbox"]),
            start_frame_number=int(frame_number),
            end_frame_number=int(frame_number),
            start_timestamp=float(timestamp),
            end_timestamp=float(timestamp),
            best_confidence=confidence,
        )
        self._active[local_id] = state
        return state

    def _update_track(self, local_id: int, frame_number: int, timestamp: float, detection: dict[str, Any]) -> None:
        state = self._active[local_id]
        state.bbox = tuple(detection["bbox"])
        state.end_frame_number = int(frame_number)
        state.end_timestamp = float(timestamp)
        state.hits += 1
        state.missed = 0
        state.best_confidence = max(state.best_confidence, float(detection.get("confidence") or 0.0))

    @staticmethod
    def _assignment(state: _TrackState, iou: float, is_new: bool) -> dict[str, Any]:
        return {
            "track_local_id": state.local_id,
            "track_class_name": state.class_name,
            "track_iou": float(iou),
            "track_is_new": bool(is_new),
            "track_hits": int(state.hits),
        }
