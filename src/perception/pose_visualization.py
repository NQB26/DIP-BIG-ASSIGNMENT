"""Small, dependency-light helpers for COCO-17 pose diagnostics."""
from __future__ import annotations

import math
from typing import Sequence

# COCO-17 skeleton: face, shoulders/arms, torso, legs.
COCO_SKELETON_EDGES = (
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)


def keypoint_is_visible(point: Sequence[float], min_conf: float = 0.35) -> bool:
    """Return True for a finite (x, y, confidence) point above threshold."""
    if len(point) < 3:
        return False
    try:
        x, y, conf = float(point[0]), float(point[1]), float(point[2])
        threshold = float(min_conf)
    except (TypeError, ValueError):
        return False
    return (
        math.isfinite(x) and math.isfinite(y) and math.isfinite(conf)
        and math.isfinite(threshold) and 0.0 <= threshold <= 1.0
        and 0.0 <= conf <= 1.0 and conf >= threshold
    )


def pose_visibility_summary(keypoints: Sequence[Sequence[float]], min_conf: float = 0.35) -> dict:
    """Summarize visible keypoints and wrist visibility for one pose."""
    visible = [keypoint_is_visible(point, min_conf) for point in keypoints[:17]]
    visible += [False] * max(0, 17 - len(visible))
    return {
        "visible_keypoints": sum(visible),
        "left_wrist_visible": visible[9],
        "right_wrist_visible": visible[10],
        "both_wrists_visible": visible[9] and visible[10],
    }
