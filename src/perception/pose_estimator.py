"""YOLO pose adapter for COCO-17 human keypoints.

The model returns one pose instance per detected person. Track association is
performed separately after ByteTrack updates, so pose IDs are never treated as
persistent tracking IDs.
"""
from __future__ import annotations

import math
from typing import List

from tracking_rules.schemas import PoseObservation

# COCO-17 keypoint indices used by Ultralytics pose checkpoints.
COCO_KEYPOINTS = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}


class PoseEstimator:
    """Run a YOLO pose model and return bbox + 17 COCO keypoints per person."""

    def __init__(self, weights: str = "yolo11n-pose.pt", conf: float = 0.25,
                 imgsz: int = 512, device: str | None = "cpu"):
        try:
            conf = float(conf)
        except (TypeError, ValueError) as exc:
            raise ValueError("pose confidence must be numeric") from exc
        if not math.isfinite(conf) or not 0.0 <= conf <= 1.0:
            raise ValueError("pose confidence must be finite and in [0, 1]")
        if int(imgsz) < 32:
            raise ValueError("pose imgsz must be >= 32")

        from ultralytics import YOLO

        self.weights = str(weights)
        self.conf = conf
        self.imgsz = int(imgsz)
        self.device = device
        self.model = YOLO(self.weights)

    def __call__(self, frame) -> List[PoseObservation]:
        kwargs = {
            "source": frame,
            "conf": self.conf,
            "imgsz": self.imgsz,
            "verbose": False,
        }
        if self.device is not None:
            kwargs["device"] = self.device
        results = self.model.predict(**kwargs)
        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        keypoints_obj = getattr(result, "keypoints", None)
        if boxes is None or keypoints_obj is None:
            return []

        try:
            box_xyxy = boxes.xyxy.cpu().numpy()
            kp_data = keypoints_obj.data.cpu().numpy()
        except AttributeError:
            # Supports numpy-like values in tests or alternative wrappers.
            box_xyxy = boxes.xyxy
            kp_data = keypoints_obj.data

        observations: List[PoseObservation] = []
        for bbox, kp in zip(box_xyxy, kp_data):
            if len(bbox) != 4 or len(kp) < 17:
                continue
            bbox_tuple = tuple(float(v) for v in bbox)
            if not all(math.isfinite(v) for v in bbox_tuple):
                continue
            if bbox_tuple[2] <= bbox_tuple[0] or bbox_tuple[3] <= bbox_tuple[1]:
                continue

            clean = []
            for point in kp[:17]:
                x = float(point[0])
                y = float(point[1])
                confidence = float(point[2]) if len(point) >= 3 else 1.0
                if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(confidence)):
                    confidence = 0.0
                    x, y = 0.0, 0.0
                confidence = min(1.0, max(0.0, confidence))
                clean.append((x, y, confidence))
            observations.append(PoseObservation(bbox_tuple, tuple(clean)))
        return observations
