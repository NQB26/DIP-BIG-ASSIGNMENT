import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tracking_rules.rule_engine import RuleEngine, load_rules_config
from tracking_rules.schemas import Track, Zone

W, H = 1280, 720
INSIDE = (300.0, 100.0, 500.0, 600.0)  # bottom_center=(400,600)
OUTSIDE = (800.0, 100.0, 1000.0, 600.0) # bottom_center=(900,600)


def r2_config():
    return load_rules_config(str(ROOT / "configs" / "rules_r2_test.yaml"))


def restricted_zone():
    # x=[0.2,0.5], y=[0.7,0.95] contains INSIDE foot point after normalization.
    return Zone("restricted_test", "restricted", [
        (0.2, 0.7), (0.5, 0.7), (0.5, 0.95), (0.2, 0.95)
    ])


def run(person_bbox, duration=2.0, dt=0.1, zone=None):
    engine = RuleEngine(
        [zone or restricted_zone()], r2_config(),
        start_wall_time=datetime(2026, 11, 23, 10, 0, 0),
        event_id_db=":memory:",
    )
    events = []
    count = int(round(duration / dt)) + 1
    for idx in range(count):
        t = round(idx * dt, 6)
        tracks = [Track(7, "person", 0.95, person_bbox)]
        events.extend(engine.update(idx, t, tracks, [], (W, H)))
    return events


def test_r2_confirms_person_in_restricted_zone_after_duration():
    events = run(INSIDE)
    assert len(events) == 1
    assert events[0]["rule_id"] == "R2"
    assert events[0]["rule_name"] == "RESTRICTED_ZONE"
    assert events[0]["track_id"] == 7
    assert events[0]["zone_id"] == "restricted_test"
    assert events[0]["duration_sec"] >= 1.5


def test_r2_does_not_trigger_for_person_outside_restricted_zone():
    assert run(OUTSIDE) == []


def test_r2_does_not_trigger_if_only_upper_body_overlaps_zone():
    # The polygon covers upper body x/y, but not the bottom-center anchor.
    zone = Zone("upper_only", "restricted", [
        (0.2, 0.1), (0.5, 0.1), (0.5, 0.5), (0.2, 0.5)
    ])
    assert run(INSIDE, zone=zone) == []


def test_r2_stays_quiet_before_minimum_duration():
    assert run(INSIDE, duration=1.4) == []


def test_r2_does_not_confirm_across_long_observation_gap():
    cfg = r2_config()
    engine = RuleEngine([restricted_zone()], cfg, event_id_db=":memory:")
    events = []
    for idx in range(8):
        t = idx * 0.1
        events.extend(engine.update(idx, t, [Track(7, "person", 0.95, INSIDE)], [], (W, H)))
    # Long gap exceeds grace_period_sec; continuous duration must restart.
    events.extend(engine.update(8, 1.0, [Track(7, "person", 0.95, INSIDE)], [], (W, H)))
    events.extend(engine.update(9, 1.1, [Track(7, "person", 0.95, INSIDE)], [], (W, H)))
    assert events == []
