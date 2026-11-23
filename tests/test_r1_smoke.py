import json
import os
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import jsonschema
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tracking_rules.rule_engine import load_rules_config, load_zones, RuleEngine
from tracking_rules.schemas import Detection, Track, Zone

W, H = 1280, 720
PERSON = (300.0, 100.0, 500.0, 600.0)
COAT = (320.0, 220.0, 480.0, 470.0)


def config():
    return load_rules_config(str(ROOT / "configs" / "rules_r1_test.yaml"))


def zone():
    return Zone("lab_general_test", "general", [
        (0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)
    ])


def evaluate(frames, zones=None):
    cfg = config()
    engine = RuleEngine(zones or [zone()], cfg,
                        start_wall_time=datetime(2026, 11, 23, 10, 0, 0),
                        event_id_db=":memory:")
    events = []
    for frame_id, (t, person_bbox, detections) in enumerate(frames):
        tracks = [] if person_bbox is None else [Track(11, "person", 0.95, person_bbox)]
        events.extend(engine.update(frame_id, t, tracks, detections, (W, H)))
    return events


def frames(duration_sec=3.0, person_bbox=PERSON, det_factory=None, dt=0.1):
    result = []
    count = int(round(duration_sec / dt)) + 1
    for i in range(count):
        t = round(i * dt, 6)
        dets = det_factory(t) if det_factory else []
        result.append((t, person_bbox, dets))
    return result


def test_r1_confirms_uncoated_person_after_min_duration():
    events = evaluate(frames(3.0))
    assert len(events) == 1
    assert events[0]["rule_id"] == "R1"
    assert events[0]["track_id"] == 11
    assert events[0]["duration_sec"] >= 2.5
    schema = json.loads((ROOT / "events" / "event_schema.json").read_text(encoding="utf-8"))
    jsonschema.validate(events[0], schema)


def test_r1_stays_quiet_when_coat_is_associated():
    events = evaluate(frames(4.0, det_factory=lambda _t: [Detection("lab_coat", 0.9, COAT)]))
    assert events == []


def test_r1_ignores_person_outside_general_zone():
    central_zone = Zone("central_lab", "general", [
        (0.1, 0.1), (0.5, 0.1), (0.5, 0.9), (0.1, 0.9)
    ])
    outside_person = (800.0, 100.0, 1000.0, 600.0)
    events = evaluate(frames(4.0, person_bbox=outside_person), [central_zone])
    assert events == []


def test_r1_ignores_person_below_minimum_height():
    short_person = (300.0, 400.0, 400.0, 490.0)  # height 90 px < 100 px
    events = evaluate(frames(4.0, person_bbox=short_person))
    assert events == []


def test_r1_coat_hold_prevents_short_detector_dropouts():
    def coat_for_time(t):
        # A short gap below coat_hold_sec must not turn into an R1 violation.
        return [] if 1.0 <= t <= 1.2 else [Detection("lab_coat", 0.9, COAT)]
    events = evaluate(frames(4.0, det_factory=coat_for_time))
    assert events == []
