#!/usr/bin/env python3
"""Inventory local media, YOLO labels, dataset YAML files, and model weights.

Run from the project root:
    python scripts/audit_dataset.py --data-dir data --model-dir models \
        --root-weights yolo11n.pt --output reports/dataset_inventory.json

The script is read-only with respect to source data. It does not modify labels,
move files, extract frames, or upload anything.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv", ".mpeg", ".mpg"}
WEIGHT_EXTS = {".pt", ".onnx", ".engine", ".tflite", ".pb"}
EXCLUDED_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache"}


def rel(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except Exception:
        return str(path)


def load_yaml(path: Path) -> tuple[Any | None, str | None]:
    try:
        import yaml  # type: ignore
    except Exception as exc:
        return None, f"PyYAML unavailable: {exc}"
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")), None
    except Exception as exc:
        return None, str(exc)


def expected_label_path(image_path: Path) -> Path | None:
    parts = list(image_path.parts)
    # Standard YOLO structure: images/<split>/... -> labels/<split>/...
    for i, part in enumerate(parts):
        if part.lower() == "images":
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")
    return None


def inspect_yolo_label(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path), "instances": 0, "class_counts": {},
        "invalid_lines": [], "formats": {"bbox": 0, "polygon_or_other": 0},
    }
    counts: Counter[str] = Counter()
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except Exception as exc:
        result["read_error"] = str(exc)
        return result

    for line_no, raw in enumerate(lines, start=1):
        raw = raw.strip()
        if not raw:
            continue
        fields = raw.split()
        try:
            class_float = float(fields[0])
            if not math.isfinite(class_float) or not class_float.is_integer() or class_float < 0:
                raise ValueError("class id must be a non-negative integer")
            class_id = int(class_float)
            coords = [float(v) for v in fields[1:]]
            if len(coords) < 4:
                raise ValueError("expected at least 4 coordinates")
            if not all(math.isfinite(v) for v in coords):
                raise ValueError("coordinates must be finite")
            if any(v < 0.0 or v > 1.0 for v in coords):
                raise ValueError("normalized coordinates must be within [0, 1]")
            if len(coords) == 4:
                if coords[2] <= 0 or coords[3] <= 0:
                    raise ValueError("bbox width and height must be > 0")
                result["formats"]["bbox"] += 1
            elif len(coords) >= 6 and len(coords) % 2 == 0:
                result["formats"]["polygon_or_other"] += 1
            else:
                raise ValueError("expected 4 bbox values or an even number of polygon coordinates")
            counts[str(class_id)] += 1
        except Exception as exc:
            result["invalid_lines"].append({"line": line_no, "reason": str(exc), "content": raw[:180]})

    result["instances"] = sum(counts.values())
    result["class_counts"] = dict(sorted(counts.items(), key=lambda item: int(item[0])))
    result["empty_label_file"] = result["instances"] == 0 and not result["invalid_lines"]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit local image/video/YOLO dataset files (read-only).")
    parser.add_argument("--data-dir", default="data", help="Directory containing media and datasets")
    parser.add_argument("--model-dir", default="models", help="Directory containing model weights")
    parser.add_argument("--root-weights", nargs="*", default=["yolo11n.pt"], help="Weight files in project root")
    parser.add_argument("--output", default="reports/dataset_inventory.json", help="JSON report path")
    args = parser.parse_args()

    project = Path.cwd()
    data_dir = (project / args.data_dir).resolve()
    model_dir = (project / args.model_dir).resolve()
    try:
        import cv2  # type: ignore
        cv2_version = cv2.__version__
    except Exception as exc:
        cv2 = None  # type: ignore
        cv2_version = None
        cv2_error = str(exc)

    report: dict[str, Any] = {
        "project_root": str(project),
        "data_dir": str(data_dir),
        "model_dir": str(model_dir),
        "opencv_version": cv2_version,
        "data_dir_exists": data_dir.exists(),
        "model_dir_exists": model_dir.exists(),
        "images": [], "videos": [], "labels": [], "dataset_yaml": [],
        "weights": [], "warnings": [],
    }
    if cv2 is None:
        report["warnings"].append(f"OpenCV unavailable; media metadata not read: {cv2_error}")

    files: list[Path] = []
    if data_dir.exists():
        for p in data_dir.rglob("*"):
            if not p.is_file() or any(part in EXCLUDED_DIRS for part in p.parts):
                continue
            files.append(p)
    else:
        report["warnings"].append(f"Data directory not found: {data_dir}")

    image_paths = [p for p in files if p.suffix.lower() in IMAGE_EXTS]
    video_paths = [p for p in files if p.suffix.lower() in VIDEO_EXTS]
    label_paths = [p for p in files if p.suffix.lower() == ".txt" and "labels" in [x.lower() for x in p.parts]]
    yaml_paths = [p for p in files if p.suffix.lower() in {".yaml", ".yml"}]

    for path in sorted(image_paths):
        item: dict[str, Any] = {"path": rel(path, project), "size_bytes": path.stat().st_size}
        if cv2 is not None:
            img = cv2.imread(str(path))
            if img is None:
                item["readable"] = False
            else:
                h, w = img.shape[:2]
                item.update({"readable": True, "width": int(w), "height": int(h), "channels": int(img.shape[2]) if len(img.shape) > 2 else 1})
        lp = expected_label_path(path)
        if lp is not None:
            item["expected_label_path"] = rel(lp, project)
            item["label_exists"] = lp.exists()
        report["images"].append(item)

    for path in sorted(video_paths):
        item = {"path": rel(path, project), "size_bytes": path.stat().st_size}
        if cv2 is not None:
            cap = cv2.VideoCapture(str(path))
            opened = bool(cap.isOpened())
            item["readable"] = opened
            if opened:
                fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                item.update({
                    "width": width, "height": height, "fps": round(fps, 4),
                    "frame_count": frame_count,
                    "duration_sec": round(frame_count / fps, 3) if fps > 0 else None,
                })
            cap.release()
        report["videos"].append(item)

    for path in sorted(label_paths):
        item = inspect_yolo_label(path)
        item["path"] = rel(path, project)
        report["labels"].append(item)

    for path in sorted(yaml_paths):
        content, error = load_yaml(path)
        item: dict[str, Any] = {"path": rel(path, project)}
        if error:
            item["parse_error"] = error
        else:
            item["top_level_keys"] = list(content.keys()) if isinstance(content, dict) else []
            if isinstance(content, dict) and "names" in content:
                item["names"] = content["names"]
            if isinstance(content, dict):
                item["train"] = content.get("train")
                item["val"] = content.get("val")
                item["test"] = content.get("test")
                item["path_config"] = content.get("path")
        report["dataset_yaml"].append(item)

    weight_paths: set[Path] = set()
    if model_dir.exists():
        weight_paths.update(p for p in model_dir.rglob("*") if p.is_file() and p.suffix.lower() in WEIGHT_EXTS)
    for value in args.root_weights:
        p = (project / value).resolve()
        if p.exists() and p.is_file():
            weight_paths.add(p)
    for path in sorted(weight_paths):
        report["weights"].append({"path": rel(path, project), "size_bytes": path.stat().st_size})

    # Dataset-level warnings, deliberately conservative: labels may be intentionally empty.
    missing_labels = [x["path"] for x in report["images"] if x.get("expected_label_path") and not x.get("label_exists")]
    invalid_labels = [x["path"] for x in report["labels"] if x.get("invalid_lines") or x.get("read_error")]
    report["summary"] = {
        "image_count": len(report["images"]),
        "video_count": len(report["videos"]),
        "label_file_count": len(report["labels"]),
        "label_instance_count": sum(x.get("instances", 0) for x in report["labels"]),
        "dataset_yaml_count": len(report["dataset_yaml"]),
        "weight_count": len(report["weights"]),
        "images_with_missing_label_file": len(missing_labels),
        "label_files_with_invalid_lines": len(invalid_labels),
    }
    if missing_labels:
        report["warnings"].append("Some images under an images/ folder have no matching labels/*.txt. Confirm they are intentional negative samples; otherwise create/fix labels.")
        report["missing_label_examples"] = missing_labels[:30]
    if invalid_labels:
        report["warnings"].append("Some label files contain malformed rows. See labels[].invalid_lines.")
        report["invalid_label_examples"] = invalid_labels[:30]
    if not report["images"] and not report["videos"]:
        report["warnings"].append("No images or videos found under the data directory. The folder may be empty or the media may be stored elsewhere.")

    output = (project / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== DIP DATASET INVENTORY ===")
    print(f"Project:       {project}")
    print(f"Images:        {report['summary']['image_count']}")
    print(f"Videos:        {report['summary']['video_count']}")
    print(f"Label files:   {report['summary']['label_file_count']}")
    print(f"Label objects: {report['summary']['label_instance_count']}")
    print(f"Dataset YAML:  {report['summary']['dataset_yaml_count']}")
    print(f"Model weights: {report['summary']['weight_count']}")
    print(f"Report:        {output}")
    for warning in report["warnings"]:
        print(f"WARNING: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
