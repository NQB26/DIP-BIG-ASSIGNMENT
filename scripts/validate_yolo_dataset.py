"""Audit a one-class Ultralytics YOLO detection dataset before training.

Checks split paths, image/label pairing, YOLO box format, class IDs, exact duplicate
images across splits, and reports missing test/negative data as warnings (not errors).
The script is read-only except for writing the JSON report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _names_to_dict(raw: Any) -> dict[int, str]:
    if isinstance(raw, list):
        return {i: str(v) for i, v in enumerate(raw)}
    if isinstance(raw, dict):
        out: dict[int, str] = {}
        for k, v in raw.items():
            try:
                out[int(k)] = str(v)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid class ID in names: {k!r}") from exc
        return out
    raise ValueError("data.yaml must define 'names' as a list or mapping")


def _resolve_root(data: dict[str, Any], yaml_path: Path) -> Path:
    raw = data.get("path", str(yaml_path.parent))
    root = Path(str(raw)).expanduser()
    if not root.is_absolute():
        root = (yaml_path.parent / root).resolve()
    return root.resolve()


def _resolve_split(root: Path, value: Any) -> Path:
    p = Path(str(value)).expanduser()
    return p.resolve() if p.is_absolute() else (root / p).resolve()


def _label_dir_for_images(image_dir: Path) -> Path:
    parts = list(image_dir.parts)
    indexes = [i for i, part in enumerate(parts) if part.lower() == "images"]
    if indexes:
        parts[indexes[-1]] = "labels"
        return Path(*parts)
    # Conventional fallback: split/images -> split/labels
    return image_dir.parent / "labels"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_dataset(data_yaml: Path, expected_class: str = "lab_coat") -> dict[str, Any]:
    data_yaml = data_yaml.expanduser().resolve()
    if not data_yaml.is_file():
        raise FileNotFoundError(data_yaml)
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    names = _names_to_dict(data.get("names"))
    normalized = {cid: name.strip().lower().replace("-", "_").replace(" ", "_") for cid, name in names.items()}
    target_ids = [cid for cid, name in normalized.items() if name == expected_class]
    if len(target_ids) != 1:
        raise ValueError(f"Expected exactly one class {expected_class!r}; got {names}")
    target_id = target_ids[0]
    root = _resolve_root(data, data_yaml)

    report: dict[str, Any] = {
        "data_yaml": str(data_yaml),
        "dataset_root": str(root),
        "class_names": {str(k): v for k, v in names.items()},
        "target_class": expected_class,
        "target_class_id": target_id,
        "splits": {},
        "duplicate_image_hashes_across_splits": [],
        "warnings": [],
        "errors": [],
    }
    split_hashes: dict[str, dict[str, str]] = {}
    resolved_dirs: dict[str, Path] = {}

    for split in ("train", "val", "test"):
        if split not in data or data.get(split) in (None, ""):
            if split == "test":
                report["warnings"].append("No test split is defined; validation metrics must not be reported as final test metrics.")
            else:
                report["errors"].append(f"Required split {split!r} is missing from data.yaml.")
            continue
        image_dir = _resolve_split(root, data[split])
        resolved_dirs[split] = image_dir
        if not image_dir.is_dir():
            report["errors"].append(f"{split} image directory does not exist: {image_dir}")
            continue
        label_dir = _label_dir_for_images(image_dir)
        images = sorted(p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        if not images:
            report["errors"].append(f"No images found in {split}: {image_dir}")
            continue

        counts = {
            "images": len(images),
            "label_files": 0,
            "missing_label_files": 0,
            "empty_negative_images": 0,
            "images_with_target": 0,
            "target_objects": 0,
            "other_class_objects": 0,
            "invalid_label_lines": 0,
            "out_of_range_boxes": 0,
            "class_id_counts": {},
        }
        class_counter: Counter[str] = Counter()
        hashes: dict[str, str] = {}
        expected_labels: set[Path] = set()

        for image_path in images:
            rel = image_path.relative_to(image_dir)
            label_path = (label_dir / rel).with_suffix(".txt")
            expected_labels.add(label_path.resolve())
            if not label_path.is_file():
                counts["missing_label_files"] += 1
                continue
            counts["label_files"] += 1
            lines = label_path.read_text(encoding="utf-8-sig").splitlines()
            image_target_count = 0
            for line_no, raw_line in enumerate(lines, 1):
                line = raw_line.strip()
                if not line:
                    continue
                fields = line.split()
                if len(fields) != 5:
                    counts["invalid_label_lines"] += 1
                    report["errors"].append(f"{label_path}:{line_no}: expected 5 fields, got {len(fields)}")
                    continue
                try:
                    cid_float = float(fields[0])
                    coords = [float(x) for x in fields[1:]]
                    cid = int(cid_float)
                    if cid_float != cid:
                        raise ValueError("class ID is not an integer")
                except ValueError as exc:
                    counts["invalid_label_lines"] += 1
                    report["errors"].append(f"{label_path}:{line_no}: invalid numeric label ({exc})")
                    continue
                class_counter[str(cid)] += 1
                if cid == target_id:
                    image_target_count += 1
                    counts["target_objects"] += 1
                else:
                    counts["other_class_objects"] += 1
                x, y, w, h = coords
                if any(v < 0 or v > 1 for v in coords) or w <= 0 or h <= 0 or x-w/2 < -1e-6 or x+w/2 > 1+1e-6 or y-h/2 < -1e-6 or y+h/2 > 1+1e-6:
                    counts["out_of_range_boxes"] += 1
                    report["errors"].append(f"{label_path}:{line_no}: bbox is outside normalized image bounds")
            if image_target_count:
                counts["images_with_target"] += 1
            else:
                counts["empty_negative_images"] += 1
            try:
                hashes[_sha256(image_path)] = str(image_path)
            except OSError as exc:
                report["errors"].append(f"Cannot hash image {image_path}: {exc}")

        actual_label_files = {p.resolve() for p in label_dir.rglob("*.txt")} if label_dir.is_dir() else set()
        orphan_labels = actual_label_files - expected_labels
        counts["orphan_label_files"] = len(orphan_labels)
        counts["class_id_counts"] = dict(class_counter)
        counts["label_directory"] = str(label_dir)
        counts["image_directory"] = str(image_dir)
        report["splits"][split] = counts
        split_hashes[split] = hashes

        if counts["missing_label_files"]:
            report["warnings"].append(f"{split}: {counts['missing_label_files']} images have no matching label file.")
        if counts["empty_negative_images"] == 0:
            report["warnings"].append(f"{split}: no empty negative images (all images contain the target class).")
        if counts["orphan_label_files"]:
            report["warnings"].append(f"{split}: {counts['orphan_label_files']} label files have no matching image.")
        if counts["other_class_objects"]:
            report["warnings"].append(f"{split}: found {counts['other_class_objects']} objects from unexpected class IDs.")

    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if a in resolved_dirs and b in resolved_dirs and resolved_dirs[a] == resolved_dirs[b]:
            report["errors"].append(f"Split path collision: {a} and {b} resolve to the same directory.")
    for a, hashes_a in split_hashes.items():
        for b, hashes_b in split_hashes.items():
            if list(split_hashes).index(a) >= list(split_hashes).index(b):
                continue
            shared = set(hashes_a) & set(hashes_b)
            if shared:
                report["duplicate_image_hashes_across_splits"].append({"split_a": a, "split_b": b, "count": len(shared)})
                report["errors"].append(f"Exact duplicate images exist across {a} and {b}: {len(shared)}")

    train_count = report.get("splits", {}).get("train", {}).get("images", 0)
    val_count = report.get("splits", {}).get("val", {}).get("images", 0)
    if train_count < 100:
        report["warnings"].append(f"Training set has only {train_count} images; treat the first model as a pipeline baseline, not a robust detector.")
    if val_count < 30:
        report["warnings"].append(f"Validation set has only {val_count} images; validation metrics may have high uncertainty.")

    report["status"] = "FAIL" if report["errors"] else ("PASS_WITH_WARNINGS" if report["warnings"] else "PASS")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Prepared one-class data.yaml")
    parser.add_argument("--expected-class", default="lab_coat")
    parser.add_argument("--output", type=Path, default=Path("reports/dataset_validation/lab_coat.json"))
    args = parser.parse_args()
    try:
        report = audit_dataset(args.data, args.expected_class)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"\nValidation status: {report['status']}")
        print(f"Report saved: {args.output.resolve()}")
        return 2 if report["status"] == "FAIL" else 0
    except Exception as exc:
        print(f"DATASET VALIDATION ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
