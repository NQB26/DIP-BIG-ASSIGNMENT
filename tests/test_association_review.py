from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "review_associations.py"
spec = importlib.util.spec_from_file_location("review_associations", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)


def det(name, bbox, conf=0.9):
    return SimpleNamespace(class_name=name, bbox=tuple(bbox), confidence=conf)


def track(track_id, bbox):
    return SimpleNamespace(track_id=track_id, class_name="person", bbox=tuple(bbox), confidence=0.9)


def test_lab_coat_associates_to_person_with_highest_ioa():
    coats = [det("lab_coat", (30, 30, 70, 100))]
    tracks = [track(1, (30, 30, 60, 100)), track(2, (25, 20, 80, 140))]
    result = module.associate_lab_coats(coats, tracks, min_ioa=0.6)
    assert result[0]["matched"] is True
    assert result[0]["track_id"] == 2
    assert result[0]["score"] == 1.0


def test_lab_coat_below_threshold_is_unmatched():
    coats = [det("lab_coat", (0, 0, 100, 100))]
    tracks = [track(1, (75, 75, 150, 150))]
    result = module.associate_lab_coats(coats, tracks, min_ioa=0.6)
    assert result[0]["matched"] is False
    assert result[0]["reason"] == "below_ioa_threshold"


def test_lab_coat_ignores_short_person_tracks():
    coats = [det("lab_coat", (10, 10, 40, 50))]
    tracks = [track(1, (0, 0, 100, 80))]
    result = module.associate_lab_coats(coats, tracks, min_ioa=0.5, min_person_height_px=100)
    assert result[0]["reason"] == "no_eligible_person"


def test_phone_associates_when_center_inside_upper_body_roi():
    phones = [det("cell_phone", (45, 30, 55, 50))]
    tracks = [track(7, (0, 0, 100, 200))]
    result = module.associate_phones(phones, tracks, upper_body_ratio=0.65)
    assert result[0]["matched"] is True
    assert result[0]["track_id"] == 7


def test_phone_outside_upper_body_roi_is_unmatched():
    phones = [det("cell_phone", (45, 160, 55, 180))]
    tracks = [track(7, (0, 0, 100, 200))]
    result = module.associate_phones(phones, tracks, upper_body_ratio=0.65)
    assert result[0]["matched"] is False
    assert result[0]["reason"] == "outside_all_upper_body_rois"


def test_phone_chooses_nearest_roi_center_when_people_overlap():
    phones = [det("cell_phone", (45, 30, 55, 40))]
    tracks = [track(1, (0, 0, 100, 200)), track(2, (20, 0, 120, 200))]
    result = module.associate_phones(phones, tracks, upper_body_ratio=0.65)
    assert result[0]["matched"] is True
    assert result[0]["candidate_count"] == 2
    # The phone center is 50,35; ROI centers are 50,65 and 70,65 respectively.
    assert result[0]["track_id"] == 1
