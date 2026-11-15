"""Điều phối R1/R2/R3: hình học -> FSM -> RuleEvent (dict theo events/event_schema.json)."""
from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

import yaml

from .geometry import (bottom_center, center, compute_ioa, point_in_bbox,
                       point_in_polygon, intersection_area, bbox_area, upper_body_roi)
from .schemas import Detection, Track, Zone
from .state_machine import FSMConfig, State, ViolationFSM

SCHEMA_VERSION = "1.0"
RULE_NAMES = {"R1": "NO_LAB_COAT", "R2": "RESTRICTED_ZONE", "R3": "PHONE_USAGE"}


DEFAULTS = {
    "common": {"person_conf_min": 0.5, "min_person_height_ratio": 0.25, "dedup_iou": 0.5},
    "tracker": {"lost_buffer_sec": 1.5, "activation_threshold": 0.5, "high_conf_threshold": 0.6,
                "min_consecutive_frames": 2, "min_iou": 0.1},
    "rules": {
        "R1": {"severity": "medium", "min_ioa_threshold": 0.6, "coat_conf_min": 0.4, "coat_hold_sec": 0.5,
               "min_duration_sec": 2.5, "grace_period_sec": 0.4, "cooldown_sec": 15.0},
        "R2": {"severity": "high", "min_duration_sec": 1.5, "grace_period_sec": 0.3, "cooldown_sec": 10.0},
        "R3": {"severity": "medium", "phone_conf_min": 0.3, "upper_body_ratio": 0.65,
               "min_duration_sec": 3.0, "grace_period_sec": 0.4, "cooldown_sec": 15.0},
    },
}


def with_defaults(cfg: dict) -> dict:
    """Trộn giá trị mặc định: file YAML chỉ cần khai báo key muốn đổi."""
    out = {k: dict(v) if k != "rules" else {} for k, v in DEFAULTS.items()}
    for k in ("common", "tracker"):
        out[k].update(cfg.get(k) or {})
    for rid, dflt in DEFAULTS["rules"].items():
        out["rules"][rid] = {**dflt, **((cfg.get("rules") or {}).get(rid) or {})}
    return out


def load_rules_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return with_defaults(yaml.safe_load(f) or {})


def load_zones(path: str) -> List[Zone]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return [Zone(z.get("zone_id") or z["id"], z["type"], [tuple(p) for p in z["polygon"]])
            for z in data.get("zones", [])]


def _iou(a, b) -> float:
    inter = intersection_area(a, b)
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def _r(b: Sequence[float]) -> List[float]:
    return [round(float(v), 1) for v in b]


class RuleEngine:
    def __init__(self, zones: List[Zone], cfg: dict, source_id: str = "video",
                 source_type: str = "video_file", start_wall_time: Optional[datetime] = None):
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
        self._seq: Dict[str, int] = {}

    # ------------------------------------------------------------------ API
    def update(self, frame_id: int, t: float, tracks: Sequence[Track],
               detections: Sequence[Detection], frame_size: Tuple[int, int]) -> List[dict]:
        W, H = frame_size
        r1 = self.rules["R1"]
        min_h = r1["min_person_height_px"] if "min_person_height_px" in r1 \
            else self.common.get("min_person_height_ratio", 0.25) * H
        pconf = self.common.get("person_conf_min", 0.5)
        persons = [p for p in tracks if p.class_name == "person" and p.confidence >= pconf]

        general = [z for z in self.zones if z.type == "general"]
        restricted = [z for z in self.zones if z.type == "restricted"]
        gen_px = {z.zone_id: z.polygon_px(W, H) for z in general}
        res_px = {z.zone_id: z.polygon_px(W, H) for z in restricted}

        # vùng general chứa chân người (không có zone general -> toàn khung hình)
        info: Dict[int, dict] = {}
        for p in persons:
            foot = bottom_center(p.bbox)
            gz = next((z.zone_id for z in general if point_in_polygon(foot, gen_px[z.zone_id])), None)
            if not general:
                gz = "full_frame"
            rz = next((z.zone_id for z in restricted if point_in_polygon(foot, res_px[z.zone_id])), None)
            info[p.track_id] = {"track": p, "foot": foot, "general": gz, "restricted": rz,
                                "height": p.bbox[3] - p.bbox[1]}

        results: Dict[Tuple[int, str], dict] = {}
        if self.rules["R1"].get("enabled", True):
            self._eval_r1(t, info, detections, min_h, results)
        if self.rules["R2"].get("enabled", True):
            self._eval_r2(info, W, H, results)
        if self.rules["R3"].get("enabled", True):
            self._eval_r3(info, detections, results)

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
        applicable = {tid: i for tid, i in info.items() if i["general"] and i["height"] >= min_h}
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
            results[(tid, "R1")] = {"violated": not has_coat, "zone": i["general"], "zone_type": "general",
                                    "details": {"best_ioa": round(best[tid], 3) if tid in best else None,
                                                "person_height_px": round(i["height"], 1)}}

    def _eval_r2(self, info, W, H, results):
        for tid, i in info.items():
            if i["restricted"]:
                fx, fy = i["foot"]
                results[(tid, "R2")] = {"violated": True, "zone": i["restricted"], "zone_type": "restricted",
                                        "details": {"foot_point_px": [round(fx, 1), round(fy, 1)],
                                                    "foot_point_norm": [round(fx / W, 4), round(fy / H, 4)]}}
            else:
                results[(tid, "R2")] = {"violated": False}

    def _eval_r3(self, info, detections, results):
        rc = self.rules["R3"]
        phones = [d for d in detections if d.class_name == "cell_phone" and d.confidence >= rc["phone_conf_min"]]
        applicable = {tid: i for tid, i in info.items() if i["general"]}
        hit: Dict[int, Detection] = {}
        for ph in phones:
            c = center(ph.bbox)
            cand = []
            for tid, i in applicable.items():
                roi = upper_body_roi(i["track"].bbox, rc["upper_body_ratio"])
                if point_in_bbox(c, roi):
                    rc_ = center(roi)
                    cand.append((math.hypot(c[0] - rc_[0], c[1] - rc_[1]), tid))
            if cand:
                tid = min(cand)[1]
                if tid not in hit or ph.confidence > hit[tid].confidence:
                    hit[tid] = ph
        for tid, i in applicable.items():
            ph = hit.get(tid)
            res = {"violated": ph is not None, "zone": i["general"], "zone_type": "general"}
            if ph is not None:
                res["details"] = {"phone_bbox": _r(ph.bbox), "phone_confidence": round(ph.confidence, 3)}
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
        self._seq[day] = self._seq.get(day, 0) + 1
        ev = {
            "schema_version": SCHEMA_VERSION,
            "event_id": f"EVT_{day}_{self._seq[day]:04d}",
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
