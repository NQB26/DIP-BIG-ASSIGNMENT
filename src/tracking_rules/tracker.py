"""Wrapper ByteTrack (gói `trackers` của Roboflow). Chỉ track lớp `person`.

lab_coat / cell_phone KHÔNG cần track: chúng được ghép với person theo hình
học ở từng frame -> ít ID switch hơn và code đơn giản hơn.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

from .schemas import Detection, Track


class PersonTracker:
    def __init__(self, fps: float = 30.0, lost_buffer_sec: float = 1.5,
                 activation_threshold: float = 0.5, high_conf_threshold: float = 0.6,
                 min_consecutive_frames: int = 2, min_iou: float = 0.1):
        from trackers import ByteTrackTracker

        self._tracker = ByteTrackTracker(
            lost_track_buffer=max(1, int(lost_buffer_sec * fps)),
            frame_rate=fps,
            track_activation_threshold=activation_threshold,
            high_conf_det_threshold=high_conf_threshold,
            minimum_consecutive_frames=min_consecutive_frames,
            minimum_iou_threshold=min_iou,
        )

    def update(self, detections: Sequence[Detection]) -> List[Track]:
        import supervision as sv

        persons = [d for d in detections if d.class_name == "person"]
        if persons:
            xyxy = np.array([d.bbox for d in persons], dtype=np.float32)
            conf = np.array([d.confidence for d in persons], dtype=np.float32)
            sv_det = sv.Detections(xyxy=xyxy, confidence=conf,
                                   class_id=np.zeros(len(persons), dtype=int))
        else:
            sv_det = sv.Detections.empty()

        out = self._tracker.update(sv_det)
        tracks: List[Track] = []
        for i in range(len(out)):
            tid = int(out.tracker_id[i])
            if tid < 0:        # track chưa được xác nhận
                continue
            c = float(out.confidence[i]) if out.confidence is not None else 1.0
            tracks.append(Track(tid, "person", c, tuple(float(v) for v in out.xyxy[i])))
        return tracks

    def reset(self):
        self._tracker.reset()
