"""Điều phối R1/R2/R3: hình học -> FSM -> RuleEvent (dict theo events/event_schema.json)."""
from __future__ import annotations

import math
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

import yaml

from .geometry import (bbox_anchor, center, compute_ioa, point_in_bbox,
                       point_in_polygon, intersection_area, bbox_area, upper_body_roi)
from .schemas import Detection, Track, Zone
from .event_ids import EventIdStore
from .state_machine import FSMConfig, State, ViolationFSM
from .pose_association import nearest_confident_wrist

SCHEMA_VERSION = "1.0"
RULE_NAMES = {"R1": "NO_LAB_COAT", "R2": "RESTRICTED_ZONE", "R3": "PHONE_USAGE"}


DEFAULTS = {
    "common": {"person_conf_min": 0.5, "min_person_height_ratio": 0.25,
                "dedup_iou": 0.5, "full_frame_fallback": False,
                "event_id_db": "runs/event_sequence.sqlite3"},
    "tracker": {"lost_buffer_sec": 1.5, "activation_threshold": 0.5, "high_conf_threshold": 0.6,
                "min_consecutive_frames": 2, "min_iou": 0.1},
    "rules": {
        "R1": {"severity": "medium", "min_ioa_threshold": 0.6, "coat_conf_min": 0.4, "coat_hold_sec": 0.5,
               "min_duration_sec": 2.5, "grace_period_sec": 0.4, "cooldown_sec": 15.0},
        "R2": {"severity": "high", "min_duration_sec": 1.5, "grace_period_sec": 0.3, "cooldown_sec": 10.0},
        "R3": {"severity": "medium", "phone_conf_min": 0.3, "upper_body_ratio": 0.65,
               "association_mode": "upper_body_roi",
               "pose_keypoint_conf_min": 0.35, "phone_wrist_max_dist_ratio": 0.18,
               "min_duration_sec": 3.0, "grace_period_sec": 0.4, "cooldown_sec": 15.0},
    },
}


def with_defaults(cfg: dict) -> dict:
    """Merge defaults and reject invalid thresholds before processing video."""
    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise ValueError("rules configuration must be a YAML mapping")
    out = {k: dict(v) if k != "rules" else {} for k, v in DEFAULTS.items()}
    for k in ("common", "tracker"):
        section = cfg.get(k) or {}
        if not isinstance(section, dict):
            raise ValueError(f"{k} must be a YAML mapping")
        out[k].update(section)
    rules_cfg = cfg.get("rules") or {}
    if not isinstance(rules_cfg, dict):
        raise ValueError("rules must be a YAML mapping")
    for rid, dflt in DEFAULTS["rules"].items():
        rule_cfg = rules_cfg.get(rid) or {}
        if not isinstance(rule_cfg, dict):
            raise ValueError(f"{rid} must be a YAML mapping")
        out["rules"][rid] = {**dflt, **rule_cfg}

    if not isinstance(out["common"].get("full_frame_fallback"), bool):
        raise ValueError("common.full_frame_fallback must be true or false")
    for rid, rc in out["rules"].items():
        if not isinstance(rc.get("enabled", True), bool):
            raise ValueError(f"{rid}.enabled must be true or false")
        target_type = rc.get("target_zone_type", "restricted" if rid == "R2" else "general")
        if target_type not in ("general", "restricted"):
            raise ValueError(f"{rid}.target_zone_type must be general or restricted")
        for key in ("min_duration_sec", "grace_period_sec", "cooldown_sec"):
            value = float(rc[key])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{rid}.{key} must be finite and non-negative")
        for key in ("coat_conf_min", "phone_conf_min", "min_ioa_threshold"):
            if key in rc and not 0 <= float(rc[key]) <= 1:
                raise ValueError(f"{rid}.{key} must be in [0, 1]")
        if rid == "R3":
            if not 0 < float(rc["upper_body_ratio"]) <= 1:
                raise ValueError("R3.upper_body_ratio must be in (0, 1]")
            if rc.get("association_mode", "upper_body_roi") not in ("upper_body_roi", "pose_hand"):
                raise ValueError("R3.association_mode must be upper_body_roi or pose_hand")
            for key in ("pose_keypoint_conf_min", "phone_wrist_max_dist_ratio"):
                value = float(rc[key])
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"R3.{key} must be finite and non-negative")
            if float(rc["pose_keypoint_conf_min"]) > 1:
                raise ValueError("R3.pose_keypoint_conf_min must be in [0, 1]")
        if rid == "R2":
            # Validate configured anchor eagerly, not on the first detected person.
            bbox_anchor((0.0, 0.0, 1.0, 1.0), rc.get("anchor_point", "bottom_center"))
    return out


def load_rules_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return with_defaults(yaml.safe_load(f) or {})


def load_zones(path: str) -> List[Zone]:
    """Load and validate normalized [0, 1] polygons from a zones YAML."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    zones: List[Zone] = []
    seen_ids = set()
    for index, raw in enumerate(data.get("zones", [])):
        if not isinstance(raw, dict):
            raise ValueError(f"zones[{index}] must be a mapping")
        zone_id = raw.get("zone_id") or raw.get("id")
        zone_type = raw.get("type")
        polygon = raw.get("polygon")
        if not zone_id or not isinstance(zone_id, str):
            raise ValueError(f"zones[{index}] needs a non-empty zone_id")
        if zone_id in seen_ids:
            raise ValueError(f"Duplicate zone_id: {zone_id}")
        seen_ids.add(zone_id)
        if zone_type not in ("general", "restricted"):
            raise ValueError(f"Zone {zone_id}: unsupported type {zone_type!r}")
        if not isinstance(polygon, list) or len(polygon) < 3:
            raise ValueError(f"Zone {zone_id}: polygon needs at least 3 points")
        points = []
        for point_index, point in enumerate(polygon):
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError(f"Zone {zone_id}: point {point_index} must be [x, y]")
            x, y = float(point[0]), float(point[1])
            if not math.isfinite(x) or not math.isfinite(y) or not (0 <= x <= 1 and 0 <= y <= 1):
                raise ValueError(f"Zone {zone_id}: normalized coordinates must be finite in [0, 1]")
            points.append((x, y))
        # Shoelace area; a repeated final point is allowed (draw_zones emits it optionally).
        area = abs(sum(points[i][0] * points[(i + 1) % len(points)][1]
                       - points[(i + 1) % len(points)][0] * points[i][1]
                       for i in range(len(points))) / 2.0)
        if area <= 1e-9:
            raise ValueError(f"Zone {zone_id}: polygon area is zero")
        zones.append(Zone(zone_id, zone_type, points))
    return zones


def _iou(a, b) -> float:
    inter = intersection_area(a, b)
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def _r(b: Sequence[float]) -> List[float]:
    return [round(float(v), 1) for v in b]


class RuleEngine:
    def __init__(self, zones: List[Zone], cfg: dict, source_id: str = "video",
                 source_type: str = "video_file", start_wall_time: Optional[datetime] = None,
                 event_id_db: Optional[str] = None):
        self.zones = zones
        cfg = with_defaults(cfg)
        self.cfg = cfg
        self.common = cfg.get("common", {})
        self.rules = cfg["rules"]
        self.source = {"id": source_id, "type": source_type}
        self._t0 = start_wall_time or datetime.now()
        self._fsm: Dict[Tuple[int, str], ViolationFSM] = {}
        self._coat_seen: Dict[int, float] = {}
        self._recent_events: List[dict] = []     # để dedup khi ID switch
        db_path = event_id_db or self.common.get("event_id_db", "runs/event_sequence.sqlite3")
        self._event_ids = EventIdStore(db_path)

        zone_types = {z.type for z in zones}
        for rid in ("R1", "R3"):
            rc = self.rules[rid]
            target_type = rc.get("target_zone_type", "general")
            if rc.get("enabled", True) and target_type not in zone_types:
                if not self.common.get("full_frame_fallback", False):
                    warnings.warn(
                        f"{rid} is enabled but zones.yaml has no {target_type!r} zone; "
                        "the rule will not evaluate any person until that zone is configured.",
                        RuntimeWarning,
                    )

    # ------------------------------------------------------------------ API
    def update(self, frame_id: int, t: float, tracks: Sequence[Track],
               detections: Sequence[Detection], frame_size: Tuple[int, int],
               pose_by_track: Optional[dict] = None) -> List[dict]:
        W, H = frame_size
        r1 = self.rules["R1"]
        min_h = r1["min_person_height_px"] if "min_person_height_px" in r1 \
            else self.common.get("min_person_height_ratio", 0.25) * H
        pconf = self.common.get("person_conf_min", 0.5)
        persons = [p for p in tracks if p.class_name == "person" and p.confidence >= pconf]

        zones_by_type = {
            "general": [z for z in self.zones if z.type == "general"],
            "restricted": [z for z in self.zones if z.type == "restricted"],
        }
        zones_px = {
            ztype: {z.zone_id: z.polygon_px(W, H) for z in zone_list}
            for ztype, zone_list in zones_by_type.items()
        }

        # Không tự động áp dụng R1/R3 trên toàn ảnh nếu chưa vẽ general zone.
        # Chỉ bật fallback này khi cấu hình cho phép rõ ràng.
        info: Dict[int, dict] = {}
        for p in persons:
            foot = bbox_anchor(p.bbox, "bottom_center")
            zone_by_type = {}
            for ztype, zone_list in zones_by_type.items():
                z = next((z for z in zone_list
                          if point_in_polygon(foot, zones_px[ztype][z.zone_id])), None)
                zone_by_type[ztype] = z.zone_id if z else None
            if not zones_by_type["general"] and self.common.get("full_frame_fallback", False):
                zone_by_type["general"] = "full_frame"
            info[p.track_id] = {
                "track": p, "foot": foot, "zone_by_type": zone_by_type,
                "general": zone_by_type["general"],
                "restricted": zone_by_type["restricted"],
                "height": p.bbox[3] - p.bbox[1],
            }

        results: Dict[Tuple[int, str], dict] = {}
        if self.rules["R1"].get("enabled", True):
            self._eval_r1(t, info, detections, min_h, results)
        if self.rules["R2"].get("enabled", True):
            self._eval_r2(info, W, H, results)
        if self.rules["R3"].get("enabled", True):
            self._eval_r3(info, detections, results, pose_by_track or {})

        for (tid, _rid), res in results.items():
            res["track"] = info[tid]["track"]

        # FSM: track vắng mặt vẫn được cập nhật với violated=False (để grace/cooldown chạy)
        keys = set(self._fsm) | set(results)
        events: List[dict] = []
        for key in keys:
            tid, rid = key
            fsm = self._fsm.get(key)
            res = results.get(key)
            if fsm is None:
                rc = self.rules[rid]
                fsm = self._fsm[key] = ViolationFSM(FSMConfig(
                    rc["min_duration_sec"], rc.get("grace_period_sec", 0.4), rc.get("cooldown_sec", 10.0)))
            fired = fsm.update(bool(res and res["violated"]), t)
            if fired and res:
                ev = self._make_event(rid, tid, frame_id, t, fsm, res, (W, H))
                if ev is not None:
                    events.append(ev)
            if tid not in info and fsm.state == State.NORMAL:
                del self._fsm[key]
        self._coat_seen = {k: v for k, v in self._coat_seen.items() if t - v < 10.0}
        return events

    def debug_states(self) -> Dict[Tuple[int, str], ViolationFSM]:
        return self._fsm

    # ---------------------------------------------------------------- rules
    def _eval_r1(self, t, info, detections, min_h, results):
        rc = self.rules["R1"]
        coats = [d for d in detections if d.class_name == "lab_coat" and d.confidence >= rc["coat_conf_min"]]
        target_type = rc.get("target_zone_type", "general")
        applicable = {tid: i for tid, i in info.items()
                      if i["zone_by_type"].get(target_type) and i["height"] >= min_h}
        best: Dict[int, float] = {}
        for c in coats:  # mỗi áo chỉ gán cho 1 người (người có IoA cao nhất)
            cand = [(compute_ioa(c.bbox, i["track"].bbox), tid) for tid, i in applicable.items()]
            if not cand:
                continue
            ioa, tid = max(cand)
            if ioa >= rc["min_ioa_threshold"]:
                best[tid] = max(best.get(tid, 0.0), ioa)
                self._coat_seen[tid] = t
        for tid, i in applicable.items():
            has_coat = tid in best or (t - self._coat_seen.get(tid, -1e9)) <= rc.get("coat_hold_sec", 0.0)
            results[(tid, "R1")] = {"violated": not has_coat,
                                    "zone": i["zone_by_type"][target_type], "zone_type": target_type,
                                    "details": {"best_ioa": round(best[tid], 3) if tid in best else None,
                                                "person_height_px": round(i["height"], 1)}}

    def _eval_r2(self, info, W, H, results):
        rc = self.rules["R2"]
        target_type = rc.get("target_zone_type", "restricted")
        anchor_name = rc.get("anchor_point", "bottom_center")
        target_zones = [z for z in self.zones if z.type == target_type]
        target_px = {z.zone_id: z.polygon_px(W, H) for z in target_zones}
        for tid, i in info.items():
            anchor = bbox_anchor(i["track"].bbox, anchor_name)
            zone = next((z for z in target_zones
                         if point_in_polygon(anchor, target_px[z.zone_id])), None)
            if zone:
                ax, ay = anchor
                details = {
                    "anchor_point_type": anchor_name,
                    "anchor_point_px": [round(ax, 1), round(ay, 1)],
                    "anchor_point_norm": [round(ax / W, 4), round(ay / H, 4)],
                }
                # Keep legacy fields for consumers expecting foot_point_*.
                if anchor_name == "bottom_center":
                    details["foot_point_px"] = details["anchor_point_px"]
                    details["foot_point_norm"] = details["anchor_point_norm"]
                results[(tid, "R2")] = {"violated": True, "zone": zone.zone_id,
                                        "zone_type": target_type, "details": details}
            else:
                results[(tid, "R2")] = {"violated": False}

    def _eval_r3(self, info, detections, results, pose_by_track=None):
        """Evaluate phone-use candidates.

        Legacy mode only checks the upper-body ROI. ``pose_hand`` mode is
        stricter: the phone center must remain in the upper-body ROI and a
        sufficiently confident COCO wrist must be close to the phone box.
        The FSM outside this method supplies the temporal persistence filter.
        """
        rc = self.rules["R3"]
        phones = [d for d in detections if d.class_name == "cell_phone" and d.confidence >= rc["phone_conf_min"]]
        target_type = rc.get("target_zone_type", "general")
        applicable = {tid: i for tid, i in info.items()
                      if i["zone_by_type"].get(target_type)}
        mode = rc.get("association_mode", "upper_body_roi")
        pose_by_track = pose_by_track or {}
        hit: Dict[int, Detection] = {}
        hit_details: Dict[int, dict] = {}

        for ph in phones:
            c = center(ph.bbox)
            candidates = []
            for tid, i in applicable.items():
                roi = upper_body_roi(i["track"].bbox, rc["upper_body_ratio"])
                if not point_in_bbox(c, roi):
                    continue

                if mode == "upper_body_roi":
                    roi_center = center(roi)
                    score = math.hypot(c[0] - roi_center[0], c[1] - roi_center[1])
                    candidates.append((score, tid, None))
                    continue

                pose = pose_by_track.get(tid)
                if pose is None:
                    continue
                person_height = max(1.0, i["height"])
                wrist = nearest_confident_wrist(
                    pose, ph.bbox, person_height,
                    min_keypoint_conf=rc["pose_keypoint_conf_min"],
                )
                if wrist is None:
                    continue
                if wrist["distance_ratio"] > rc["phone_wrist_max_dist_ratio"]:
                    continue
                candidates.append((wrist["distance_ratio"], tid, wrist))

            if candidates:
                score, tid, wrist = min(candidates, key=lambda item: item[0])
                if tid not in hit or ph.confidence > hit[tid].confidence:
                    hit[tid] = ph
                    details = {
                        "phone_bbox": _r(ph.bbox),
                        "phone_confidence": round(ph.confidence, 3),
                        "association_mode": mode,
                    }
                    if wrist is not None:
                        details.update({
                            "wrist_side": wrist["side"],
                            "wrist_xy": [round(wrist["xy"][0], 1), round(wrist["xy"][1], 1)],
                            "wrist_confidence": round(wrist["confidence"], 3),
                            "phone_wrist_distance_px": round(wrist["distance_px"], 2),
                            "phone_wrist_distance_ratio": round(wrist["distance_ratio"], 4),
                            "max_phone_wrist_distance_ratio": rc["phone_wrist_max_dist_ratio"],
                            "pose_bbox": _r(pose_by_track[tid].bbox),
                        })
                    hit_details[tid] = details

        for tid, i in applicable.items():
            ph = hit.get(tid)
            res = {"violated": ph is not None,
                   "zone": i["zone_by_type"][target_type], "zone_type": target_type}
            if ph is not None:
                res["details"] = hit_details[tid]
            results[(tid, "R3")] = res

    # ---------------------------------------------------------------- event
    def _make_event(self, rid, tid, frame_id, t, fsm, res, size) -> Optional[dict]:
        W, H = size
        track: Track = res["track"]
        rc = self.rules[rid]

        # Chống trùng khi ID switch: cùng rule + bbox chồng nhau trong thời gian cooldown
        thr = self.common.get("dedup_iou", 0.5)
        cd = rc.get("cooldown_sec", 10.0)
        self._recent_events = [e for e in self._recent_events if t - e["video_time_sec"] <= cd]
        for e in self._recent_events:
            if e["rule_id"] == rid and _iou(e["bbox"], track.bbox) >= thr:
                return None

        wall = self._t0 + timedelta(seconds=t)
        day = wall.strftime("%Y%m%d")
        event_id = self._event_ids.next_id(day)
        ev = {
            "schema_version": SCHEMA_VERSION,
            "event_id": event_id,
            "rule_id": rid,
            "rule_name": rc.get("name", RULE_NAMES[rid]),
            "severity": rc.get("severity", "medium"),
            "status": "CONFIRMED",
            "timestamp": wall.isoformat(timespec="milliseconds"),
            "video_time_sec": round(t, 3),
            "frame_id": frame_id,
            "violation_start_sec": round(fsm.start_t, 3),
            "duration_sec": round(t - fsm.start_t, 3),
            "track_id": tid,
            "zone_id": res["zone"],
            "zone_type": res["zone_type"],
            "bbox": _r(track.bbox),
            "frame_size": {"width": W, "height": H},
            "confidence": round(track.confidence, 3),
            "details": res.get("details", {}),
            "source": self.source,
            "evidence": {"image_path": None, "clip_path": None},
        }
        self._recent_events.append(ev)
        return ev
