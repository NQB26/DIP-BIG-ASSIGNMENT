"""Normalize YOLO Detection and YOLO Segmentation exports to one-class lab_coat detection.

Source data is read-only. The target class is remapped to class 0. Detection labels
(5 fields) are preserved; polygon segmentation labels (class + x/y pairs) are
converted to tight enclosing bounding boxes. All non-target classes are validated
then excluded. Images without a source label file are rejected by default because
an unlabeled image is not automatically a confirmed negative.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
# Small annotation-export rounding errors are clipped; larger excursions fail.
DEFAULT_BBOX_BOUNDARY_EPSILON = 1e-4
SPLIT_ALIASES = {
    "train": "train", "val": "val", "valid": "val",
    "validation": "val", "test": "test",
}


def norm_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def normalize_names(raw: Any) -> dict[int, str]:
    if isinstance(raw, list):
        return {i: str(name) for i, name in enumerate(raw)}
    if isinstance(raw, dict):
        out: dict[int, str] = {}
        for key, value in raw.items():
            try:
                out[int(key)] = str(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid class id in data.yaml names: {key!r}") from exc
        return out
    raise ValueError("data.yaml must define 'names' as a list or mapping")


def locate_data_yaml(source: Path, explicit: Path | None) -> Path:
    if explicit:
        candidate = explicit.expanduser().resolve()
    else:
        candidates = [source / "data.yaml", source / "data.yml"]
        candidate = next((p for p in candidates if p.is_file()), candidates[0])
    if not candidate.is_file():
        raise FileNotFoundError(
            f"Cannot find data.yaml/data.yml at {candidate}. "
            "Use --data-yaml with the exact YAML path if it is nested."
        )
    return candidate.resolve()


def split_hint(split_value: Any) -> str | None:
    tokens = [t.lower() for t in re.split(r"[\\/]+", str(split_value).strip()) if t and t not in {".", ".."}]
    for token in tokens:
        if token in SPLIT_ALIASES:
            return SPLIT_ALIASES[token]
    return None


def resolve_image_dir(yaml_dir: Path, dataset_path: Path, split_value: Any, source: Path) -> Path | None:
    if not split_value:
        return None
    p = Path(str(split_value))
    candidates = [p] if p.is_absolute() else [dataset_path / p, yaml_dir / p, source / p]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_dir():
            return resolved
    hint = split_hint(split_value)
    if hint:
        folder = {"train": "train", "val": "valid", "test": "test"}[hint]
        candidate = source / folder / "images"
        if candidate.is_dir():
            return candidate.resolve()
        if hint == "val":
            candidate = source / "val" / "images"
            if candidate.is_dir():
                return candidate.resolve()
    return None


def discover_splits(source: Path, yaml_path: Path, data: dict) -> tuple[dict[str, Path], list[str]]:
    dataset_path = Path(str(data.get("path", ".")))
    if not dataset_path.is_absolute():
        dataset_path = (yaml_path.parent / dataset_path).resolve()
    splits: dict[str, Path] = {}
    missing: list[str] = []
    for source_key in ("train", "val", "valid", "validation", "test"):
        if source_key not in data:
            continue
        split_name = SPLIT_ALIASES[source_key]
        if split_name in splits:
            continue
        d = resolve_image_dir(yaml_path.parent, dataset_path, data[source_key], source)
        if d is not None:
            splits[split_name] = d
        else:
            missing.append(split_name)
    convention = {"train": ("train",), "val": ("valid", "val"), "test": ("test",)}
    declared = {SPLIT_ALIASES[k] for k in data if k in SPLIT_ALIASES}
    for canonical, folders in convention.items():
        if canonical in splits or canonical in declared:
            continue
        for folder in folders:
            candidate = source / folder / "images"
            if candidate.is_dir():
                splits[canonical] = candidate.resolve()
                break
    return splits, sorted(set(missing))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_floats(path: Path, line_no: int, raw: list[str]) -> list[float]:
    try:
        vals = [float(x) for x in raw]
    except ValueError as exc:
        raise ValueError(f"{path}:{line_no}: non-numeric YOLO coordinates") from exc
    if not all(math.isfinite(v) for v in vals):
        raise ValueError(f"{path}:{line_no}: coordinates must be finite")
    if any(v < 0.0 or v > 1.0 for v in vals):
        raise ValueError(f"{path}:{line_no}: normalized coordinates must be in [0,1]")
    return vals


def parse_yolo_row_detailed(
    parts: list[str], path: Path, line_no: int,
    bbox_boundary_epsilon: float = DEFAULT_BBOX_BOUNDARY_EPSILON,
) -> tuple[int, str, list[float], bool]:
    """Parse one YOLO row and report whether a tiny bbox boundary error was clipped.

    YOLO bbox center/size values must be normalized. A box may extend outside
    [0, 1] only by at most `bbox_boundary_epsilon` due to export rounding; in
    that case its corners are clipped and center/size are recomputed. Larger
    excursions are rejected. Polygon coordinates themselves must remain in
    [0, 1].
    """
    if not math.isfinite(bbox_boundary_epsilon) or bbox_boundary_epsilon < 0:
        raise ValueError("bbox_boundary_epsilon must be a finite non-negative number")
    if len(parts) < 5:
        raise ValueError(f"{path}:{line_no}: expected 5 bbox fields or polygon fields, got {len(parts)}")
    try:
        class_float = float(parts[0])
        class_id = int(class_float)
    except ValueError as exc:
        raise ValueError(f"{path}:{line_no}: invalid class id {parts[0]!r}") from exc
    if not math.isfinite(class_float) or class_float != class_id or class_id < 0:
        raise ValueError(f"{path}:{line_no}: class id must be a non-negative integer")

    if len(parts) == 5:
        coords = _validate_floats(path, line_no, parts[1:])
        x, y, w, h = coords
        if w <= 0 or h <= 0:
            raise ValueError(f"{path}:{line_no}: bbox width and height must be > 0")
        xmin, xmax = x - w / 2, x + w / 2
        ymin, ymax = y - h / 2, y + h / 2
        if (xmin < -bbox_boundary_epsilon or ymin < -bbox_boundary_epsilon or
                xmax > 1 + bbox_boundary_epsilon or ymax > 1 + bbox_boundary_epsilon):
            raise ValueError(
                f"{path}:{line_no}: bbox extends outside normalized image bounds "
                f"beyond tolerance {bbox_boundary_epsilon:g} "
                f"(bounds=({xmin:.9f}, {ymin:.9f}, {xmax:.9f}, {ymax:.9f}))"
            )
        clipped_bounds = (
            max(0.0, xmin), max(0.0, ymin),
            min(1.0, xmax), min(1.0, ymax),
        )
        did_clip = any(a != b for a, b in zip((xmin, ymin, xmax, ymax), clipped_bounds))
        if not did_clip:
            # Preserve exact source values when no correction is needed.
            return class_id, "bbox", coords, False
        cx0, cy0, cx1, cy1 = clipped_bounds
        clipped_w, clipped_h = cx1 - cx0, cy1 - cy0
        if clipped_w <= 0 or clipped_h <= 0:
            raise ValueError(f"{path}:{line_no}: bbox becomes empty after boundary clipping")
        clipped_coords = [
            (cx0 + cx1) / 2, (cy0 + cy1) / 2, clipped_w, clipped_h
        ]
        return class_id, "bbox", clipped_coords, True

    # YOLO polygon rows: class id followed by x1,y1,...,xn,yn (n >= 3).
    if len(parts) >= 7 and (len(parts) - 1) % 2 == 0:
        coords = _validate_floats(path, line_no, parts[1:])
        points = list(zip(coords[0::2], coords[1::2]))
        if len(set(points)) < 3:
            raise ValueError(f"{path}:{line_no}: polygon must contain at least 3 distinct points")
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        w, h = xmax - xmin, ymax - ymin
        if w <= 0 or h <= 0:
            raise ValueError(f"{path}:{line_no}: polygon has zero-width or zero-height bounds")
        bbox = [(xmin + xmax) / 2, (ymin + ymax) / 2, w, h]
        return class_id, "polygon", bbox, False

    raise ValueError(
        f"{path}:{line_no}: unsupported YOLO row with {len(parts)} fields; "
        "expected 5-field detection bbox or class + an even number of polygon coordinates (>= 6)"
    )


def parse_yolo_row(parts: list[str], path: Path, line_no: int) -> tuple[int, str, list[float]]:
    """Backward-compatible parser API; detailed clipping status is internal."""
    class_id, fmt, coords, _clipped = parse_yolo_row_detailed(parts, path, line_no)
    return class_id, fmt, coords


def parse_label_file_detailed(
    label_path: Path, source_names: dict[int, str], target_id: int,
    bbox_boundary_epsilon: float = DEFAULT_BBOX_BOUNDARY_EPSILON,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "rows": [], "target_count": 0, "bbox_count": 0,
        "polygon_count": 0, "ignored_non_target_objects": 0,
        "source_bbox_rows": 0, "source_polygon_rows": 0,
        "clipped_bbox_count": 0,
    }
    if not label_path.exists():
        return result
    for line_no, raw_line in enumerate(label_path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        class_id, fmt, coords, was_clipped = parse_yolo_row_detailed(
            parts, label_path, line_no, bbox_boundary_epsilon
        )
        if class_id not in source_names:
            raise ValueError(f"{label_path}:{line_no}: class id {class_id} absent from data.yaml")
        if fmt == "bbox":
            result["source_bbox_rows"] += 1
        else:
            result["source_polygon_rows"] += 1
        if class_id != target_id:
            result["ignored_non_target_objects"] += 1
            continue
        result["rows"].append("0 " + " ".join(f"{v:.8f}" for v in coords))
        result["target_count"] += 1
        result["bbox_count" if fmt == "bbox" else "polygon_count"] += 1
        if was_clipped:
            result["clipped_bbox_count"] += 1
    return result


def parse_label_file(label_path: Path, source_names: dict[int, str], target_id: int) -> tuple[list[str], int]:
    """Backward-compatible two-value parser API used by existing tests/tools."""
    parsed = parse_label_file_detailed(label_path, source_names, target_id)
    return parsed["rows"], parsed["target_count"]


def prepare(source: Path, output: Path, yaml_file: Path | None = None,
            class_name: str = "lab_coat", drop_empty: bool = False,
            source_class_id: int | None = None,
            bbox_boundary_epsilon: float = DEFAULT_BBOX_BOUNDARY_EPSILON) -> dict:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if not source.is_dir():
        raise NotADirectoryError(source)
    yaml_path = locate_data_yaml(source, yaml_file)
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    source_names = normalize_names(raw.get("names"))

    if source_class_id is not None:
        if source_class_id not in source_names:
            raise ValueError(f"--source-class-id {source_class_id} is not declared in data.yaml. Available classes: {source_names}")
        target_id = source_class_id
        target_source_name = source_names[target_id]
    else:
        target_key = norm_name(class_name)
        matches = [cid for cid, name in source_names.items() if norm_name(name) == target_key]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one class matching {class_name!r}; found {matches}. Available classes: {source_names}. "
                "If the class name is opaque, inspect source annotations first and use --source-class-id."
            )
        target_id = matches[0]
        target_source_name = source_names[target_id]

    splits, missing_splits = discover_splits(source, yaml_path, raw)
    if "train" not in splits or "val" not in splits:
        raise ValueError(f"Need distinct train and val image directories; discovered: {splits}; missing: {missing_splits}")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if a in splits and b in splits and splits[a] == splits[b]:
            raise ValueError(f"Split path collision: {a} and {b} both resolve to {splits[a]}")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}. Choose a new --output; source data is never overwritten.")

    # Preflight all images/labels and parse every annotation before creating output.
    plans: dict[str, list[tuple[Path, Path, Path, dict[str, Any], str]]] = {}
    for split, image_dir in splits.items():
        image_files = sorted(p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        if not image_files:
            if split in ("train", "val"):
                raise ValueError(f"No images found in required split {split}: {image_dir}")
            continue
        split_plans = []
        used_names: set[str] = set()
        for image_path in image_files:
            rel = image_path.relative_to(image_dir)
            out_name = "__".join(rel.with_suffix("").parts) + image_path.suffix.lower()
            if out_name in used_names:
                raise ValueError(f"Image filename collision after flattening: {image_path}")
            used_names.add(out_name)
            label_path = image_path.parent.parent / "labels" / rel.with_suffix(".txt")
            if not label_path.exists():
                label_path = image_dir.parent / "labels" / rel.with_suffix(".txt")
            if not label_path.is_file():
                raise FileNotFoundError(
                    f"Image has no label file: {image_path}. If this is a verified negative image, "
                    f"create an empty label file explicitly: {label_path}"
                )
            parsed = parse_label_file_detailed(
                label_path, source_names, target_id, bbox_boundary_epsilon
            )
            if drop_empty and parsed["target_count"] == 0:
                continue
            split_plans.append((image_path, label_path, output, parsed, out_name))
        plans[split] = split_plans

    output.mkdir(parents=True)
    summary: dict[str, Any] = {
        "schema_version": 2,
        "source": str(source), "source_data_yaml": str(yaml_path),
        "source_classes": source_names, "target_class": class_name,
        "source_target_class_id": target_id, "source_target_class_name": target_source_name,
        "output_class_id": 0, "missing_source_splits": missing_splits,
        "annotation_formats": {"bbox_rows": 0, "polygon_rows": 0},
        "boundary_policy": {
            "bbox_boundary_epsilon": bbox_boundary_epsilon,
            "action": "clip_bbox_corners_and_recompute_xywh_within_tolerance; reject_larger_excursions",
            "clipped_target_bbox_rows": 0,
        },
        "splits": {}, "duplicate_image_hashes_across_splits": [],
    }
    hashes_by_split: dict[str, set[str]] = {}

    for split, split_plans in plans.items():
        out_images, out_labels = output / "images" / split, output / "labels" / split
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)
        counts = {
            "images": 0, "images_with_lab_coat": 0, "empty_negative_images": 0,
            "lab_coat_objects": 0, "lab_coat_bbox_objects": 0,
            "lab_coat_polygon_objects_converted_to_bbox": 0,
            "ignored_non_target_objects": 0, "missing_source_label_files": 0,
            "lab_coat_bbox_objects_clipped_to_image_bounds": 0,
        }
        split_hashes: set[str] = set()
        for image_path, label_path, _unused, parsed, out_name in split_plans:
            shutil.copy2(image_path, out_images / out_name)
            (out_labels / (Path(out_name).stem + ".txt")).write_text(
                "\n".join(parsed["rows"]) + ("\n" if parsed["rows"] else ""), encoding="utf-8"
            )
            counts["images"] += 1
            counts["lab_coat_objects"] += parsed["target_count"]
            counts["lab_coat_bbox_objects"] += parsed["bbox_count"]
            counts["lab_coat_polygon_objects_converted_to_bbox"] += parsed["polygon_count"]
            counts["lab_coat_bbox_objects_clipped_to_image_bounds"] += parsed["clipped_bbox_count"]
            summary["boundary_policy"]["clipped_target_bbox_rows"] += parsed["clipped_bbox_count"]
            counts["ignored_non_target_objects"] += parsed["ignored_non_target_objects"]
            summary["annotation_formats"]["bbox_rows"] += parsed["source_bbox_rows"]
            summary["annotation_formats"]["polygon_rows"] += parsed["source_polygon_rows"]
            if parsed["target_count"]:
                counts["images_with_lab_coat"] += 1
            else:
                counts["empty_negative_images"] += 1
            split_hashes.add(sha256_file(image_path))
        hashes_by_split[split] = split_hashes
        summary["splits"][split] = counts

    for i, a in enumerate(hashes_by_split):
        for b in list(hashes_by_split)[i + 1:]:
            shared = hashes_by_split[a] & hashes_by_split[b]
            if shared:
                summary["duplicate_image_hashes_across_splits"].append({"split_a": a, "split_b": b, "count": len(shared)})

    yaml_data: dict[str, Any] = {
        "path": output.as_posix(), "train": "images/train", "val": "images/val",
        "names": {0: "lab_coat"},
    }
    if "test" in summary["splits"]:
        yaml_data["test"] = "images/test"
    (output / "data.yaml").write_text(yaml.safe_dump(yaml_data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    (output / "conversion_report.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--data-yaml", type=Path, default=None)
    ap.add_argument("--class-name", default="lab_coat")
    ap.add_argument("--source-class-id", type=int, default=None)
    ap.add_argument("--drop-empty-images", action="store_true")
    ap.add_argument(
        "--bbox-boundary-epsilon", type=float,
        default=DEFAULT_BBOX_BOUNDARY_EPSILON,
        help="Maximum normalized bbox-boundary overshoot to clip (default: 1e-4). Larger errors fail.",
    )
    args = ap.parse_args()
    try:
        summary = prepare(
            args.source, args.output, args.data_yaml, args.class_name,
            args.drop_empty_images, args.source_class_id, args.bbox_boundary_epsilon
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
