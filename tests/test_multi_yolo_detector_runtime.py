from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1] / "src" / "perception" / "multi_yolo_detector.py"
spec = importlib.util.spec_from_file_location("multi_yolo_detector_runtime", MODULE)
mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(mod)


class FakeScalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FakeCoords:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class FakeBox:
    def __init__(self, class_id, confidence, bbox):
        self.cls = FakeScalar(class_id)
        self.conf = FakeScalar(confidence)
        self.xyxy = [FakeCoords(bbox)]


class FakeModel:
    instances = []

    def __init__(self, weights):
        self.weights = str(weights)
        self.is_coat = "coat" in self.weights
        self.names = ({0: "lab_coat"} if self.is_coat else
                      {0: "person", 1: "bicycle", 67: "cell phone"})
        self.calls = []
        FakeModel.instances.append(self)

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        if self.is_coat:
            boxes = [FakeBox(0, 0.91, [10, 20, 40, 80])]
        else:
            boxes = [
                FakeBox(0, 0.88, [0, 0, 100, 200]),
                FakeBox(67, 0.77, [40, 60, 55, 80]),
                FakeBox(1, 0.99, [150, 20, 200, 70]),
            ]
        return [types.SimpleNamespace(boxes=boxes)]


def install_fakes(monkeypatch):
    class FakeDetection:
        def __init__(self, class_name, confidence, bbox):
            self.class_name = class_name
            self.confidence = confidence
            self.bbox = bbox

    tracking_pkg = types.ModuleType("tracking_rules")
    tracking_pkg.__path__ = []
    schemas_mod = types.ModuleType("tracking_rules.schemas")
    schemas_mod.Detection = FakeDetection
    monkeypatch.setitem(sys.modules, "tracking_rules", tracking_pkg)
    monkeypatch.setitem(sys.modules, "tracking_rules.schemas", schemas_mod)
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeModel))
    FakeModel.instances = []


def test_dual_detector_fuses_expected_classes_and_uses_separate_thresholds(monkeypatch):
    install_fakes(monkeypatch)
    detector = mod.MultiYoloDetector(
        coco_weights="yolo11n.pt",
        lab_coat_weights="best_coat.pt",
        conf=0.25,
        coco_conf=0.30,
        lab_coat_conf=0.42,
        filter_coco_classes=True,
    )
    detections = detector(frame=object(), frame_id=0)

    assert [d.class_name for d in detections] == ["person", "cell_phone", "lab_coat"]
    assert detections[2].bbox == (10.0, 20.0, 40.0, 80.0)
    coco_model, coat_model = FakeModel.instances
    assert coco_model.calls[0]["conf"] == 0.30
    assert coco_model.calls[0]["classes"] == [0, 67]
    assert coat_model.calls[0]["conf"] == 0.42
    assert "classes" not in coat_model.calls[0]


def test_single_model_compatibility_keeps_all_detected_classes(monkeypatch):
    install_fakes(monkeypatch)
    detector = mod.MultiYoloDetector(
        coco_weights="yolo11n.pt", filter_coco_classes=False, conf=0.25
    )
    detections = detector(frame=object())
    assert [d.class_name for d in detections] == ["person", "cell_phone", "bicycle"]


def test_bad_confidence_is_rejected_before_inference(monkeypatch):
    install_fakes(monkeypatch)
    try:
        mod.MultiYoloDetector(coco_weights="yolo11n.pt", conf=1.1)
    except ValueError as exc:
        assert "[0, 1]" in str(exc)
    else:
        raise AssertionError("Expected invalid confidence to raise ValueError")


def test_invalid_bbox_is_rejected():
    FakeDetection = type("FakeDetection", (), {})
    invalid_box = FakeBox(0, 0.9, [10, 20, 5, 80])
    try:
        mod.MultiYoloDetector._box_to_detection(FakeDetection, "person", invalid_box)
    except ValueError as exc:
        assert "zero/negative-area" in str(exc)
    else:
        raise AssertionError("Expected invalid bbox to raise ValueError")
