from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_yolo_dataset import audit_dataset  # noqa: E402


def _write_img(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((32, 32, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def _make_dataset(root: Path, with_test: bool = False) -> Path:
    for split, n in (("train", 2), ("val", 1)):
        for i in range(n):
            _write_img(root / "images" / split / f"{split}_{i}.jpg", 30 + i + (0 if split == "train" else 50))
            label = root / "labels" / split / f"{split}_{i}.txt"
            label.parent.mkdir(parents=True, exist_ok=True)
            label.write_text("0 0.5 0.5 0.3 0.4\n", encoding="utf-8")
    if with_test:
        _write_img(root / "images" / "test" / "test_0.jpg", 99)
        (root / "labels" / "test").mkdir(parents=True, exist_ok=True)
        (root / "labels" / "test" / "test_0.txt").write_text("", encoding="utf-8")
    data = {
        "path": str(root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "lab_coat"},
    }
    if with_test:
        data["test"] = "images/test"
    yaml_path = root / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return yaml_path


def test_valid_dataset_reports_small_set_and_missing_test_as_warnings(tmp_path):
    report = audit_dataset(_make_dataset(tmp_path))
    assert report["status"] == "PASS_WITH_WARNINGS"
    assert report["splits"]["train"]["images"] == 2
    assert report["splits"]["train"]["target_objects"] == 2
    assert any("No test split" in warning for warning in report["warnings"])


def test_invalid_class_mapping_fails(tmp_path):
    yaml_path = _make_dataset(tmp_path)
    content = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    content["names"] = {0: "phone"}
    yaml_path.write_text(yaml.safe_dump(content), encoding="utf-8")
    try:
        audit_dataset(yaml_path)
    except ValueError as exc:
        assert "lab_coat" in str(exc)
    else:
        raise AssertionError("Expected class mapping error")


def test_duplicate_image_across_splits_fails(tmp_path):
    yaml_path = _make_dataset(tmp_path)
    duplicate_source = tmp_path / "images" / "train" / "train_0.jpg"
    duplicate_target = tmp_path / "images" / "val" / "duplicate.jpg"
    duplicate_target.write_bytes(duplicate_source.read_bytes())
    (tmp_path / "labels" / "val" / "duplicate.txt").write_text("0 0.5 0.5 0.3 0.4\n", encoding="utf-8")
    report = audit_dataset(yaml_path)
    assert report["status"] == "FAIL"
    assert report["duplicate_image_hashes_across_splits"]


def test_bad_bbox_fails(tmp_path):
    yaml_path = _make_dataset(tmp_path)
    bad = tmp_path / "labels" / "train" / "train_0.txt"
    bad.write_text("0 1.2 0.5 0.3 0.4\n", encoding="utf-8")
    report = audit_dataset(yaml_path)
    assert report["status"] == "FAIL"
    assert report["splits"]["train"]["out_of_range_boxes"] == 1


def test_empty_negative_image_is_counted(tmp_path):
    yaml_path = _make_dataset(tmp_path, with_test=True)
    report = audit_dataset(yaml_path)
    assert report["splits"]["test"]["empty_negative_images"] == 1
