from __future__ import annotations

import importlib.util
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1] / "src" / "perception" / "multi_yolo_detector.py"
# Load by path so these tests do not require Ultralytics/PyTorch.
spec = importlib.util.spec_from_file_location("multi_yolo_detector", MODULE)
mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(mod)


def test_normalizes_coco_phone_name():
    assert mod.normalize_class_name("cell phone") == "cell_phone"


def test_find_person_and_phone_ids_from_coco_names():
    names = {0: "person", 67: "cell phone", 2: "car"}
    assert mod.find_class_id(names, "person") == 0
    assert mod.find_class_id(names, "cell_phone") == 67


def test_accepts_one_class_coat_model():
    names = {0: "lab_coat"}
    assert mod.names_to_dict(names) == names
    assert mod.find_class_id(names, "lab_coat") == 0


def test_multi_detector_fuses_person_phone_and_lab_coat(monkeypatch):
    import sys
    import types

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
        def __init__(self, weights):
            self.is_coat = "coat" in str(weights)
            self.names = {0: "lab_coat"} if self.is_coat else {0: "person", 1: "bicycle", 67: "cell phone"}
        def predict(self, **kwargs):
            if self.is_coat:
                boxes = [FakeBox(0, 0.91, [10, 20, 40, 80])]
            else:
                boxes = [
                    FakeBox(0, 0.88, [0, 0, 100, 200]),
                    FakeBox(67, 0.77, [40, 60, 55, 80]),
                    FakeBox(1, 0.99, [150, 20, 200, 70]),
                ]
            return [types.SimpleNamespace(boxes=boxes)]

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
    detector = mod.MultiYoloDetector(
        coco_weights="yolo11n.pt", lab_coat_weights="labcoat_best.pt",
        filter_coco_classes=True,
    )
    detections = detector(frame=object(), frame_id=0)
    assert [d.class_name for d in detections] == ["person", "cell_phone", "lab_coat"]
    assert detections[0].confidence == 0.88
    assert detections[2].bbox == (10.0, 20.0, 40.0, 80.0)
