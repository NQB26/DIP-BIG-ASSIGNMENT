"""Fuse COCO person/phone detections with a custom one-class lab-coat model.

The output contract stays tracking_rules.schemas.Detection. Only person objects
are sent to ByteTrack by PersonTracker; lab_coat and cell_phone remain per-frame
objects for RuleEngine association.
"""
from __future__ import annotations

import math
import re
from typing import Any


def normalize_class_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def names_to_dict(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        result = {int(k): str(v) for k, v in names.items()}
    elif isinstance(names, (list, tuple)):
        result = {i: str(v) for i, v in enumerate(names)}
    else:
        raise ValueError(f"Unsupported model.names value: {type(names).__name__}")
    if not result or any(not name.strip() for name in result.values()):
        raise ValueError("model.names must contain at least one non-empty class name")
    return result


def find_class_id(names: Any, wanted: str) -> int:
    wanted_norm = normalize_class_name(wanted)
    matches = [class_id for class_id, name in names_to_dict(names).items()
               if normalize_class_name(name) == wanted_norm]
    if len(matches) != 1:
        raise ValueError(f"Expected one class named {wanted!r}; found {matches} in {names!r}")
    return matches[0]


def _validate_confidence(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1], got {value!r}")
    return value


class MultiYoloDetector:
    """Run a COCO model and optional custom lab-coat model on each frame.

    ``conf`` is the backwards-compatible fallback. ``coco_conf`` and
    ``lab_coat_conf`` can override it independently because the two models
    are trained on different label distributions and should be calibrated
    separately on validation data.
    """

    def __init__(self, coco_weights: str, lab_coat_weights: str | None = None,
                 conf: float = 0.25, imgsz: int = 640, device: str | None = None,
                 filter_coco_classes: bool = True,
                 coco_conf: float | None = None,
                 lab_coat_conf: float | None = None):
        from ultralytics import YOLO

        self.coco_model = YOLO(coco_weights)
        self.lab_coat_model = YOLO(lab_coat_weights) if lab_coat_weights else None
        self.coco_conf = _validate_confidence("coco_conf", conf if coco_conf is None else coco_conf)
        self.lab_coat_conf = _validate_confidence(
            "lab_coat_conf", conf if lab_coat_conf is None else lab_coat_conf
        )
        self.imgsz = int(imgsz)
        if self.imgsz < 32:
            raise ValueError("imgsz must be >= 32")
        self.device = device
        self.filter_coco_classes = bool(filter_coco_classes)

        self.person_id = None
        self.phone_id = None
        if self.filter_coco_classes:
            self.person_id = find_class_id(self.coco_model.names, "person")
            self.phone_id = find_class_id(self.coco_model.names, "cell_phone")

        self.lab_coat_class_id = None
        if self.lab_coat_model is not None:
            coat_names = names_to_dict(self.lab_coat_model.names)
            coat_ids = [i for i, n in coat_names.items()
                        if normalize_class_name(n) in {"lab_coat", "labcoat"}]
            if len(coat_names) == 1:
                # A single-class model has only class 0 (or its sole declared ID).
                self.lab_coat_class_id = next(iter(coat_names))
            elif len(coat_ids) == 1:
                self.lab_coat_class_id = coat_ids[0]
            else:
                raise ValueError(
                    "Custom coat model must have exactly one class or one class named lab_coat; "
                    f"got {coat_names}"
                )

    def _predict(self, model, frame, conf: float, classes=None):
        kwargs = {"source": frame, "conf": conf, "imgsz": self.imgsz, "verbose": False}
        if self.device is not None:
            kwargs["device"] = self.device
        if classes is not None:
            kwargs["classes"] = classes
        results = model.predict(**kwargs)
        if not results:
            return None
        return results[0]

    @staticmethod
    def _box_to_detection(Detection, class_name: str, box):
        bbox = tuple(float(v) for v in box.xyxy[0].tolist())
        confidence = float(box.conf.item())
        if len(bbox) != 4 or not all(math.isfinite(v) for v in bbox):
            raise ValueError(f"Detector returned an invalid bbox: {bbox!r}")
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Detector returned a zero/negative-area bbox: {bbox!r}")
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Detector returned invalid confidence: {confidence!r}")
        return Detection(class_name, confidence, bbox)

    def __call__(self, frame, frame_id: int = 0):
        from tracking_rules.schemas import Detection

        detections: list[Detection] = []
        coco_classes = None
        if self.filter_coco_classes:
            coco_classes = [self.person_id, self.phone_id]
        coco_result = self._predict(self.coco_model, frame, self.coco_conf, coco_classes)
        coco_names = names_to_dict(self.coco_model.names)
        if coco_result is not None:
            for box in coco_result.boxes:
                class_id = int(box.cls.item())
                if class_id not in coco_names:
                    continue
                class_name = normalize_class_name(coco_names[class_id])
                if self.filter_coco_classes and class_name not in {"person", "cell_phone"}:
                    continue
                detections.append(self._box_to_detection(Detection, class_name, box))

        if self.lab_coat_model is not None:
            coat_result = self._predict(self.lab_coat_model, frame, self.lab_coat_conf)
            if coat_result is not None:
                for box in coat_result.boxes:
                    if int(box.cls.item()) != self.lab_coat_class_id:
                        continue
                    detections.append(self._box_to_detection(Detection, "lab_coat", box))
        return detections
