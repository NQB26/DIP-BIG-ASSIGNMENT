from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from perception.pose_visualization import (  # noqa: E402
    COCO_SKELETON_EDGES,
    keypoint_is_visible,
    pose_visibility_summary,
)


def test_coco_skeleton_edges_only_use_valid_coco17_indices():
    assert len(COCO_SKELETON_EDGES) == 16
    assert all(0 <= a < 17 and 0 <= b < 17 for a, b in COCO_SKELETON_EDGES)


def test_keypoint_visibility_threshold():
    assert keypoint_is_visible((10, 20, 0.8), 0.35)
    assert not keypoint_is_visible((10, 20, 0.2), 0.35)


def test_keypoint_rejects_non_finite_values():
    assert not keypoint_is_visible((math.nan, 20, 0.9), 0.35)
    assert not keypoint_is_visible((10, 20, math.inf), 0.35)


def test_pose_visibility_summary_counts_wrists():
    points = [(0.0, 0.0, 0.0)] * 17
    points[5] = (100, 100, 0.9)
    points[9] = (120, 150, 0.8)
    points[10] = (160, 150, 0.7)
    summary = pose_visibility_summary(points, 0.35)
    assert summary["visible_keypoints"] == 3
    assert summary["left_wrist_visible"] is True
    assert summary["right_wrist_visible"] is True
    assert summary["both_wrists_visible"] is True


def test_pose_visibility_summary_handles_missing_points():
    summary = pose_visibility_summary([(1, 2, 0.9)], 0.35)
    assert summary["visible_keypoints"] == 1
    assert summary["left_wrist_visible"] is False
    assert summary["right_wrist_visible"] is False
