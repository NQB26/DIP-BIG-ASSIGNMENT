from __future__ import annotations

from tracking_rules.pose_association import (
    associate_pose_to_tracks,
    box_iou,
    nearest_confident_wrist,
    point_to_bbox_distance,
)
from tracking_rules.schemas import PoseObservation, Track


def pose(bbox, left=(40.0, 50.0, 0.9), right=(0.0, 0.0, 0.0)):
    points = [(0.0, 0.0, 0.0)] * 17
    points[9] = left
    points[10] = right
    return PoseObservation(tuple(bbox), tuple(points))


def tr(tid, bbox):
    return Track(tid, "person", 0.9, tuple(bbox))


def test_iou_exact_overlap():
    assert box_iou((0, 0, 100, 200), (0, 0, 100, 200)) == 1.0


def test_pose_instances_associate_one_to_one_with_tracks():
    poses = [
        pose((0, 0, 100, 200), left=(30, 100, 0.9)),
        pose((200, 0, 300, 200), left=(230, 100, 0.9)),
    ]
    tracks = [tr(11, (0, 0, 100, 200)), tr(22, (200, 0, 300, 200))]
    result = associate_pose_to_tracks(poses, tracks, min_iou=0.3)
    assert set(result) == {11, 22}
    assert result[11] is poses[0]
    assert result[22] is poses[1]


def test_pose_below_iou_threshold_is_not_associated():
    result = associate_pose_to_tracks(
        [pose((200, 200, 300, 300))], [tr(3, (0, 0, 100, 100))], min_iou=0.3
    )
    assert result == {}


def test_pose_matching_is_one_to_one_when_two_pose_boxes_overlap_one_track():
    poses = [pose((0, 0, 100, 200)), pose((5, 5, 105, 205))]
    result = associate_pose_to_tracks(poses, [tr(5, (0, 0, 100, 200))], min_iou=0.1)
    assert len(result) == 1


def test_point_to_bbox_distance_zero_inside_box():
    assert point_to_bbox_distance((50, 50), (40, 40, 60, 60)) == 0.0


def test_nearest_confident_wrist_ignores_low_confidence_keypoint():
    p = pose((0, 0, 100, 200), left=(20, 20, 0.1), right=(50, 50, 0.9))
    result = nearest_confident_wrist(p, (48, 48, 54, 54), 200, min_keypoint_conf=0.35)
    assert result is not None
    assert result["side"] == "right"
    assert result["distance_px"] == 0.0


def test_no_reliable_wrist_returns_none():
    p = pose((0, 0, 100, 200), left=(20, 20, 0.1), right=(50, 50, 0.2))
    assert nearest_confident_wrist(p, (48, 48, 54, 54), 200, 0.35) is None
