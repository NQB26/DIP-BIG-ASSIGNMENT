import json
import os
import sys
from datetime import datetime, timezone, timedelta

import jsonschema
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from tracking_rules.geometry import compute_ioa, point_in_polygon, bottom_center
from tracking_rules.rule_engine import RuleEngine, load_rules_config, load_zones
from tracking_rules.schemas import Detection, Track
from tracking_rules.state_machine import FSMConfig, State, ViolationFSM

W, H, FPS = 1280, 720, 30
SCHEMA = json.load(open(os.path.join(ROOT, "events", "event_schema.json"), encoding="utf-8"))
CFG = load_rules_config(os.path.join(ROOT, "configs", "rules.yaml"))
ZONES = load_zones(os.path.join(ROOT, "configs", "zones.yaml"))
T0 = datetime(2026, 11, 15, 10, 0, 0)

PERSON = (300.0, 200.0, 450.0, 600.0)          # chân (375, 600) nằm trong lab_main_area
COAT = (310.0, 260.0, 440.0, 420.0)
PHONE = (360.0, 280.0, 390.0, 320.0)           # tâm trong 65% thân trên


def run(frames, engine=None):
    eng = engine or RuleEngine(ZONES, CFG, start_wall_time=T0)
    events = []
    for i, (person, dets) in enumerate(frames):
        tracks = [Track(1, "person", 0.9, person)] if person else []
        events += eng.update(i, i / FPS, tracks, dets, (W, H))
    return events


def test_geometry():
    assert compute_ioa(COAT, PERSON) == pytest.approx(1.0)
    assert compute_ioa((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(0.5)
    sq = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert point_in_polygon((5, 5), sq) and not point_in_polygon((11, 5), sq)
    assert bottom_center((0, 0, 10, 20)) == (5, 20)


def test_fsm_confirm_grace_cooldown():
    f = ViolationFSM(FSMConfig(1.0, 0.3, 5.0))
    fired = [f.update(True, i / FPS) for i in range(31)]
    assert fired.count(True) == 1 and f.state == State.CONFIRMED
    f.update(True, 1.1)
    assert f.state == State.COOLDOWN
    assert f.update(True, 3.0) is False                 # đang cooldown
    g = ViolationFSM(FSMConfig(1.0, 0.3, 5.0))          # gián đoạn ngắn không reset
    for i in range(15):
        g.update(True, i / FPS)
    for i in range(15, 20):                             # 0.17s mất
        g.update(False, i / FPS)
    assert g.state == State.PENDING
    for i in range(20, 40):
        g.update(True, i / FPS)
    assert g.state in (State.CONFIRMED, State.COOLDOWN)
    h = ViolationFSM(FSMConfig(1.0, 0.3, 5.0))          # gián đoạn dài reset
    for i in range(10):
        h.update(True, i / FPS)
    for i in range(10, 25):
        h.update(False, i / FPS)
    assert h.state == State.NORMAL


def test_r1_no_coat_fires_once_and_validates():
    ev = run([(PERSON, [])] * 120)                      # 4s không áo
    assert [e["rule_id"] for e in ev] == ["R1"]
    jsonschema.validate(ev[0], SCHEMA)
    assert ev[0]["duration_sec"] >= 2.5
    assert ev[0]["timestamp"] == "2026-11-15T10:00:02.500" or ev[0]["timestamp"].startswith("2026-11-15T10:00:02")
    assert ev[0]["zone_id"] == "lab_main_area"


def test_r1_with_coat_and_flicker_is_quiet():
    frames = []
    for i in range(300):                                # áo bị miss 40% frame
        dets = [Detection("lab_coat", 0.8, COAT)] if i % 5 not in (1, 3) else []
        frames.append((PERSON, dets))
    assert run(frames) == []


def test_r1_small_person_ignored():
    far = (300.0, 400.0, 340.0, 490.0)                  # cao 90px < min_person_height_px (100)
    assert run([(far, [])] * 120) == []


def test_r2_restricted_zone():
    inside = (1000.0, 250.0, 1100.0, 470.0)              # chân (1050, 470) trong chemical_storage
    dets = [Detection("lab_coat", 0.9, (1005.0, 280.0, 1095.0, 400.0))]
    ev = run([(inside, dets)] * 60)
    assert [e["rule_id"] for e in ev] == ["R2"]
    jsonschema.validate(ev[0], SCHEMA)


def test_r3_phone():
    dets = [Detection("lab_coat", 0.9, COAT), Detection("cell_phone", 0.7, PHONE)]
    ev = run([(PERSON, dets)] * 120)
    assert [e["rule_id"] for e in ev] == ["R3"]
    jsonschema.validate(ev[0], SCHEMA)
    table_phone = Detection("cell_phone", 0.7, (360.0, 520.0, 390.0, 560.0))  # thấp hơn 65% thân
    assert run([(PERSON, [Detection("lab_coat", 0.9, COAT), table_phone])] * 120) == []


def test_cooldown_then_realert():
    ev = run([(PERSON, [])] * (30 * 40))                # 40s liên tục
    assert 2 <= len(ev) <= 3                            # 2.5s + 15s cooldown lặp lại
    assert len({e["event_id"] for e in ev}) == len(ev)


def test_track_missing_does_not_leak_fsm():
    eng = RuleEngine(ZONES, CFG, start_wall_time=T0)
    run([(PERSON, [])] * 30 + [(None, [])] * 60, eng)
    assert eng.debug_states() == {}
