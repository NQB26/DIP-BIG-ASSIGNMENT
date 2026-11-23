"""Associate per-frame pose instances to persistent ByteTrack person tracks."""
from __future__ import annotations

import math
from typing import Dict, Iterable, Sequence

from .schemas import PoseObservation, Track
from .geometry import bbox_area, intersection_area


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    inter = intersection_area(a, b)
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def associate_pose_to_tracks(
    poses: Sequence[PoseObservation],
    tracks: Sequence[Track],
    min_iou: float = 0.30,
) -> Dict[int, PoseObservation]:
    """Greedy one-to-one matching by descending bbox IoU.

    Pose detections are frame-local. The returned keys are ByteTrack IDs.
    A pose or a track can participate in at most one match per frame.
    """
    min_iou = float(min_iou)
    if not math.isfinite(min_iou) or not 0.0 <= min_iou <= 1.0:
        raise ValueError("min_iou must be finite and in [0, 1]")

    person_tracks = [t for t in tracks if t.class_name == "person"]
    candidates = []
    for pose_idx, pose in enumerate(poses):
        for track_idx, track in enumerate(person_tracks):
            score = box_iou(pose.bbox, track.bbox)
            if score >= min_iou:
                candidates.append((score, pose_idx, track_idx))

    candidates.sort(reverse=True)
    used_poses = set()
    used_tracks = set()
    matched: Dict[int, PoseObservation] = {}
    for score, pose_idx, track_idx in candidates:
        if pose_idx in used_poses or track_idx in used_tracks:
            continue
        used_poses.add(pose_idx)
        used_tracks.add(track_idx)
        matched[person_tracks[track_idx].track_id] = poses[pose_idx]
    return matched


def point_to_bbox_distance(point: Sequence[float], bbox: Sequence[float]) -> float:
    """Euclidean distance from a point to the nearest point on an xyxy box."""
    x, y = float(point[0]), float(point[1])
    x1, y1, x2, y2 = map(float, bbox)
    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)
    return math.hypot(dx, dy)


def nearest_confident_wrist(
    pose: PoseObservation,
    phone_bbox: Sequence[float],
    person_height: float,
    min_keypoint_conf: float = 0.35,
):
    """Return the nearest reliable COCO wrist to a phone box.

    The distance is normalized by person bbox height, making the threshold
    less sensitive to camera distance. Returns None if neither wrist is
    sufficiently visible.
    """
    if person_height <= 0:
        return None
    min_keypoint_conf = float(min_keypoint_conf)
    if not 0 <= min_keypoint_conf <= 1:
        raise ValueError("min_keypoint_conf must be in [0, 1]")

    candidates = []
    # COCO-17: left wrist=9, right wrist=10.
    for side, index in (("left", 9), ("right", 10)):
        if len(pose.keypoints) <= index:
            continue
        x, y, confidence = pose.keypoints[index]
        if confidence < min_keypoint_conf:
            continue
        distance = point_to_bbox_distance((x, y), phone_bbox)
        ratio = distance / float(person_height)
        candidates.append((ratio, distance, side, (float(x), float(y)), float(confidence)))

    if not candidates:
        return None
    ratio, distance, side, xy, confidence = min(candidates, key=lambda item: item[0])
    return {
        "distance_ratio": ratio,
        "distance_px": distance,
        "side": side,
        "xy": xy,
        "confidence": confidence,
    }
