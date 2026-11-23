from __future__ import annotations

import copy
import json
import sys
from datetime import datetime
from pathlib import Path

import jsonschema
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tracking_rules.rule_engine import RuleEngine, load_rules_config  # noqa: E402
from tracking_rules.schemas import Detection, PoseObservation, Track, Zone  # noqa: E402

W, H = 1280, 720
PERSON = (300.0, 100.0, 500.0, 600.0)
PHONE = (390.0, 220.0, 410.0, 240.0)
GENERAL = Zone("general_test", "general", [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
SCHEMA = json.loads((ROOT / "events" / "event_schema.json").read_text(encoding="utf-8"))


def config():
    cfg = load_rules_config(str(ROOT / "configs" / "rules_r3_pose_test.yaml"))
    return cfg


def make_pose(left=(400.0, 230.0, 0.9), right=(470.0, 330.0, 0.9)):
    kp = [(0.0, 0.0, 0.0)] * 17
    kp[5] = (360.0, 180.0, 0.9)  # left shoulder
    kp[6] = (440.0, 180.0, 0.9)  # right shoulder
    kp[7] = (380.0, 210.0, 0.9)  # left elbow
    kp[8] = (460.0, 250.0, 0.9)  # right elbow
    kp[9] = left
    kp[10] = right
    return PoseObservation(PERSON, tuple(kp))


def evaluate(duration=3.0, phone_bbox=PHONE, pose=None):
    cfg = config()
    engine = RuleEngine(
        [GENERAL], cfg,
        start_wall_time=datetime(2026, 11, 23, 10, 0, 0),
        event_id_db=":memory:",
    )
    events = []
    count = int(duration * 10) + 1
    for frame in range(count):
        t = frame / 10.0
        tracks = [Track(7, "person", 0.95, PERSON)]
        dets = [Detection("cell_phone", 0.9, phone_bbox)]
        pose_by_track = {7: pose} if pose is not None else {}
        events.extend(engine.update(frame, t, tracks, dets, (W, H), pose_by_track=pose_by_track))
    return events


def test_r3_pose_hand_confirms_phone_near_wrist_after_duration():
    events = evaluate(pose=make_pose(left=(400.0, 230.0, 0.95)))
    assert len(events) == 1
    event = events[0]
    assert event["rule_id"] == "R3"
    assert event["details"]["association_mode"] == "pose_hand"
    assert event["details"]["wrist_side"] == "left"
    assert event["details"]["phone_wrist_distance_ratio"] == 0.0
    jsonschema.validate(event, SCHEMA)


def test_r3_does_not_trigger_for_phone_in_upper_body_but_far_from_wrists():
    far_pose = make_pose(left=(310.0, 150.0, 0.95), right=(490.0, 150.0, 0.95))
    assert evaluate(pose=far_pose) == []


def test_r3_does_not_trigger_without_pose_observation():
    assert evaluate(pose=None) == []


def test_r3_does_not_trigger_when_wrist_confidence_is_too_low():
    low_conf = make_pose(left=(400.0, 230.0, 0.1), right=(470.0, 330.0, 0.1))
    assert evaluate(pose=low_conf) == []


def test_r3_does_not_trigger_for_phone_outside_upper_body_roi_even_if_wrist_near():
    phone_low = (390.0, 500.0, 410.0, 520.0)
    low_pose = make_pose(left=(400.0, 510.0, 0.95))
    assert evaluate(phone_bbox=phone_low, pose=low_pose) == []


def test_r3_legacy_mode_remains_backward_compatible():
    cfg = config()
    cfg["rules"]["R3"]["association_mode"] = "upper_body_roi"
    engine = RuleEngine([GENERAL], cfg, event_id_db=":memory:")
    events = []
    for frame in range(31):
        events.extend(engine.update(
            frame, frame / 10.0,
            [Track(7, "person", 0.95, PERSON)],
            [Detection("cell_phone", 0.9, PHONE)],
            (W, H),
        ))
    assert len(events) == 1
    assert events[0]["details"]["association_mode"] == "upper_body_roi"
