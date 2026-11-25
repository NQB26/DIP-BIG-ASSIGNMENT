"""Drawing helpers: zones, boxes, track IDs, per-rule FSM state, alert banner, HUD.

cv2.putText only renders ASCII, so every on-video label is ASCII.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import cv2
import numpy as np

RULE_COLORS = {"R1": (200, 120, 0), "R2": (40, 40, 220), "R3": (0, 150, 240)}  # BGR
ZONE_COLORS = {"general": (80, 180, 80), "restricted": (50, 50, 230)}
COAT_COLOR = (230, 160, 40)
PHONE_COLOR = (200, 60, 200)

try:  # optional: skeleton drawing (perception/pose_visualization.py)
    from perception.pose_visualization import COCO_SKELETON_EDGES, keypoint_is_visible
except Exception:  # pragma: no cover - dependency location differs per repo
    try:
        from pose_visualization import COCO_SKELETON_EDGES, keypoint_is_visible
    except Exception:
        COCO_SKELETON_EDGES, keypoint_is_visible = (), None


def _label(img, text: str, x: int, y: int, color, scale: float = 0.45):
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    y = max(th + 2, y)
    cv2.rectangle(img, (x, y - th - 3), (x + tw + 4, y + base - 1), color, -1)
    cv2.putText(img, text, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), 1, cv2.LINE_AA)


def draw_zones(img, zones, alpha: float = 0.18):
    h, w = img.shape[:2]
    layer = img.copy()
    polys = []
    for z in zones:
        pts = np.array(z.polygon_px(w, h), dtype=np.int32)
        col = ZONE_COLORS.get(z.type, (200, 200, 200))
        cv2.fillPoly(layer, [pts], col)
        polys.append((z, pts, col))
    cv2.addWeighted(layer, alpha, img, 1 - alpha, 0, img)
    for z, pts, col in polys:
        cv2.polylines(img, [pts], True, col, 2, cv2.LINE_AA)
        x, y = int(pts[:, 0].min()), int(pts[:, 1].min())
        _label(img, f"{z.type}:{z.zone_id}", x, y + 16, col, 0.42)


def draw_frame(img, zones, tracks, detections, engine, pose_by_track: Optional[dict],
               recent_events: Sequence[dict], hud: Dict[str, object], t: float,
               show_pose: bool = True):
    """Annotate ``img`` in place and return it."""
    draw_zones(img, zones)

    for d in detections:
        if d.class_name == "lab_coat":
            x1, y1, x2, y2 = map(int, d.bbox)
            cv2.rectangle(img, (x1, y1), (x2, y2), COAT_COLOR, 2)
            _label(img, f"coat {d.confidence:.2f}", x1, y1, COAT_COLOR, 0.4)
        elif d.class_name == "cell_phone":
            x1, y1, x2, y2 = map(int, d.bbox)
            cv2.rectangle(img, (x1, y1), (x2, y2), PHONE_COLOR, 2)
            _label(img, f"phone {d.confidence:.2f}", x1, y2 + 14, PHONE_COLOR, 0.4)

    fsm_states = engine.debug_states() if engine is not None else {}
    for tr in tracks:
        if tr.class_name != "person":
            continue
        x1, y1, x2, y2 = map(int, tr.bbox)
        flagged = [rid for (tid, rid), f in fsm_states.items()
                   if tid == tr.track_id and f.state.value in ("PENDING", "CONFIRMED", "COOLDOWN")]
        col = RULE_COLORS[flagged[0]] if flagged else (60, 200, 60)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        cv2.circle(img, (int((x1 + x2) / 2), y2), 4, col, -1)  # bottom-center anchor (R2)
        _label(img, f"ID {tr.track_id} {tr.confidence:.2f}", x1, y1, col)
        line = 1
        for (tid, rid), f in sorted(fsm_states.items()):
            if tid != tr.track_id or f.state.value == "NORMAL":
                continue
            need = engine.rules[rid]["min_duration_sec"]
            txt = f"{rid} {f.state.value}"
            if f.state.value == "PENDING":
                txt += f" {f.elapsed:.1f}/{need:.1f}s"
            _label(img, txt, x1, y2 + 14 * line + 12, RULE_COLORS.get(rid, col), 0.4)
            line += 1
        if show_pose and pose_by_track and keypoint_is_visible and tr.track_id in pose_by_track:
            kps = pose_by_track[tr.track_id].keypoints
            for a, b in COCO_SKELETON_EDGES:
                if keypoint_is_visible(kps[a]) and keypoint_is_visible(kps[b]):
                    cv2.line(img, (int(kps[a][0]), int(kps[a][1])),
                             (int(kps[b][0]), int(kps[b][1])), (255, 255, 0), 1, cv2.LINE_AA)
            for idx in (9, 10):  # wrists drive R3 association
                if keypoint_is_visible(kps[idx]):
                    cv2.circle(img, (int(kps[idx][0]), int(kps[idx][1])), 4, (0, 255, 255), -1)

    # alert banner for events confirmed in the last 3 video-seconds
    banner_y = 30
    for ev in recent_events:
        if 0 <= t - ev["video_time_sec"] <= 3.0:
            col = RULE_COLORS.get(ev["rule_id"], (0, 0, 255))
            cv2.rectangle(img, (0, banner_y - 22), (img.shape[1], banner_y + 6), col, -1)
            cv2.putText(img, f"ALERT {ev['rule_id']} {ev['rule_name']}  ID {ev['track_id']}  {ev['event_id']}",
                        (8, banner_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            banner_y += 30

    h = img.shape[0]
    text = "  ".join(f"{k}:{v}" for k, v in hud.items())
    cv2.rectangle(img, (0, h - 22), (img.shape[1], h), (0, 0, 0), -1)
    cv2.putText(img, text, (8, h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return img
