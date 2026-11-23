"""Merge independently normalized, single-class YOLO lab_coat datasets.

Never merge raw exports directly. First run prepare_labcoat_dataset.py for each
source so each input has class 0 == lab_coat and its own train/val/test folders.
This script copies source data read-only, prefixes filenames, validates labels,
prevents exact-image leakage across splits, and writes a canonical data.yaml.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val", "test")


def _names(raw: Any) -> dict[int, str]:
    if isinstance(raw, list):
        return {i: str(v) for i, v in enumerate(raw)}
    if isinstance(raw, dict):
        try:
            return {int(k): str(v) for k, v in raw.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid class mapping: {raw!r}") from exc
    raise ValueError("Each normalized data.yaml must define names as a list or mapping")


def _load_dataset(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve()
    yaml_path = root / "data.yaml"
    if not root.is_dir():
        raise NotADirectoryError(root)
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Missing normalized data.yaml: {yaml_path}")
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    names = _names(data.get("names"))
    if names != {0: "lab_coat"}:
        raise ValueError(
            f"Input {root} is not normalized: expected names {{0: 'lab_coat'}}, got {names}. "
            "Run prepare_labcoat_dataset.py first; do not merge raw class IDs."
        )
    base_value = data.get("path", ".")
    base = Path(str(base_value))
    if not base.is_absolute():
        base = (yaml_path.parent / base).resolve()
    split_dirs: dict[str, Path] = {}
    for split in SPLITS:
        if split not in data:
            continue
        p = Path(str(data[split]))
        candidate = p if p.is_absolute() else base / p
        candidate = candidate.resolve()
        if not candidate.is_dir():
            raise FileNotFoundError(f"{split} image directory missing in {root}: {candidate}")
        split_dirs[split] = candidate
    if "train" not in split_dirs or "val" not in split_dirs:
        raise ValueError(f"Dataset {root} must have distinct train and val splits")
    if split_dirs["train"] == split_dirs["val"]:
        raise ValueError(f"Train and val point to the same directory in {root}")
    return {"root": root, "yaml": yaml_path, "split_dirs": split_dirs}


def _label_dir(image_dir: Path) -> Path:
    if image_dir.parent.name.lower() != "images":
        raise ValueError(f"Expected image directory to end in images/<split>: {image_dir}")
    return image_dir.parent.parent / "labels" / image_dir.name


def _read_label(path: Path) -> tuple[str, int]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing label file: {path}")
    out: list[str] = []
    objects = 0
    for line_no, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{line_no}: expected 5 YOLO fields, got {len(parts)}")
        try:
            cls = int(float(parts[0]))
            vals = [float(x) for x in parts[1:]]
        except ValueError as exc:
            raise ValueError(f"{path}:{line_no}: non-numeric YOLO label") from exc
        if cls != 0:
            raise ValueError(f"{path}:{line_no}: normalized class ID must be 0, got {cls}")
        x, y, w, h = vals
        if any(v < 0 or v > 1 for v in vals) or w <= 0 or h <= 0:
            raise ValueError(f"{path}:{line_no}: invalid normalized bbox values {vals}")
        if x - w / 2 < -1e-6 or x + w / 2 > 1 + 1e-6 or y - h / 2 < -1e-6 or y + h / 2 > 1 + 1e-6:
            raise ValueError(f"{path}:{line_no}: bbox exceeds image bounds")
        out.append("0 " + " ".join(f"{v:.8f}" for v in vals))
        objects += 1
    return "\n".join(out) + ("\n" if out else ""), objects


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_").lower()
    return slug or "dataset"


def merge(inputs: list[Path], output: Path) -> dict[str, Any]:
    if len(inputs) < 1:
        raise ValueError("Provide at least one normalized dataset with --inputs")
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists; refusing to overwrite: {output}")
    datasets = [_load_dataset(p) for p in inputs]
    resolved_roots = [d["root"] for d in datasets]
    if len(set(resolved_roots)) != len(resolved_roots):
        raise ValueError("The same input dataset was supplied more than once")
    if any(output == root or root in output.parents for root in resolved_roots):
        raise ValueError("Output must not be inside any source dataset")

    # Stable and collision-resistant source prefixes, even if folders share a basename.
    prefix_by_root: dict[Path, str] = {}
    for index, root in enumerate(resolved_roots, 1):
        prefix_by_root[root] = f"src{index:02d}_{_slug(root.name)}"

    output.mkdir(parents=True)
    report: dict[str, Any] = {
        "schema_version": 1,
        "target_class": "lab_coat",
        "target_class_id": 0,
        "sources": [],
        "splits": {},
        "exact_duplicate_images_within_split_skipped": [],
        "cross_split_duplicate_images": [],
        "test_available": False,
    }
    # Track hash globally so exact copies never leak across train/val/test.
    hash_owner: dict[str, tuple[str, str, str]] = {}
    all_source_names: set[str] = set()
    for d in datasets:
        source_record: dict[str, Any] = {
            "root": str(d["root"]),
            "data_yaml": str(d["yaml"]),
            "prefix": prefix_by_root[d["root"]],
            "splits": {},
        }
        conv_report = d["root"] / "conversion_report.json"
        if conv_report.is_file():
            try:
                cr = json.loads(conv_report.read_text(encoding="utf-8"))
                source_record["original_source"] = cr.get("source")
                source_record["source_data_yaml"] = cr.get("source_data_yaml")
                source_record["source_target_class_id"] = cr.get("source_target_class_id")
                source_record["source_target_class_name"] = cr.get("source_target_class_name")
                # Missing source labels are ambiguous, not safe negatives.
                missing_total = sum(
                    int(v.get("missing_source_label_files", 0))
                    for v in (cr.get("splits") or {}).values()
                )
                if missing_total:
                    raise ValueError(
                        f"{d['root']} conversion report has {missing_total} missing source label files. "
                        "Resolve these before merging; unlabeled images must not silently become negatives."
                    )
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid conversion_report.json in {d['root']}") from exc
        report["sources"].append(source_record)

    aggregate = {s: {"images": 0, "positive_images": 0, "negative_images": 0,
                     "lab_coat_objects": 0, "exact_duplicates_skipped": 0} for s in SPLITS}

    for d, source_record in zip(datasets, report["sources"]):
        root = d["root"]
        prefix = prefix_by_root[root]
        for split, image_dir in d["split_dirs"].items():
            label_dir = _label_dir(image_dir)
            if not label_dir.is_dir():
                raise FileNotFoundError(f"Missing label directory for {split}: {label_dir}")
            images = sorted(p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
            if split in ("train", "val") and not images:
                raise ValueError(f"No images in required {split} directory: {image_dir}")
            split_out_img = output / "images" / split
            split_out_lbl = output / "labels" / split
            split_out_img.mkdir(parents=True, exist_ok=True)
            split_out_lbl.mkdir(parents=True, exist_ok=True)
            src_counts = {"images": 0, "positive_images": 0, "negative_images": 0,
                          "lab_coat_objects": 0, "missing_label_files": 0}
            for image in images:
                rel = image.relative_to(image_dir)
                label = label_dir / rel.with_suffix(".txt")
                if not label.is_file():
                    src_counts["missing_label_files"] += 1
                    raise FileNotFoundError(
                        f"Image {image} has no corresponding label {label}. "
                        "Create an empty .txt only after confirming this is a fully reviewed negative image."
                    )
                label_text, obj_count = _read_label(label)
                digest = _sha(image)
                owner = hash_owner.get(digest)
                if owner:
                    old_split, old_source, old_label_text = owner
                    if old_split != split:
                        report["cross_split_duplicate_images"].append({
                            "sha256": digest, "split_a": old_split, "split_b": split,
                            "source_a": old_source, "source_b": str(root),
                        })
                        raise ValueError(
                            f"Exact duplicate image crosses splits ({old_split} vs {split}): {image}. "
                            "Resolve leakage before merging."
                        )
                    if old_label_text != label_text:
                        raise ValueError(
                            f"Duplicate image has conflicting annotations in {split}: {image}"
                        )
                    src_counts["exact_duplicates_skipped"] = src_counts.get("exact_duplicates_skipped", 0) + 1
                    aggregate[split]["exact_duplicates_skipped"] += 1
                    continue
                hash_owner[digest] = (split, str(root), label_text)
                # Prefix every output name by source and preserve nested relative path.
                rel_stem = "__".join(rel.with_suffix("").parts)
                out_stem = f"{prefix}__{rel_stem}"
                out_image = split_out_img / (out_stem + image.suffix.lower())
                out_label = split_out_lbl / (out_stem + ".txt")
                if out_image.exists() or out_label.exists():
                    raise FileExistsError(f"Output filename collision: {out_stem}")
                shutil.copy2(image, out_image)
                out_label.write_text(label_text, encoding="utf-8")
                src_counts["images"] += 1
                src_counts["lab_coat_objects"] += obj_count
                if obj_count:
                    src_counts["positive_images"] += 1
                else:
                    src_counts["negative_images"] += 1
                aggregate[split]["images"] += 1
                aggregate[split]["lab_coat_objects"] += obj_count
                if obj_count:
                    aggregate[split]["positive_images"] += 1
                else:
                    aggregate[split]["negative_images"] += 1
            if src_counts["missing_label_files"]:
                raise ValueError(f"{src_counts['missing_label_files']} missing labels in {root}/{split}")
            source_record["splits"][split] = src_counts

    if aggregate["train"]["images"] == 0 or aggregate["val"]["images"] == 0:
        raise ValueError("Merged dataset must contain non-empty train and val splits")

    yaml_data: dict[str, Any] = {
        "path": output.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "lab_coat"},
    }
    if aggregate["test"]["images"] > 0:
        yaml_data["test"] = "images/test"
        report["test_available"] = True
    report["splits"] = aggregate
    (output / "data.yaml").write_text(
        yaml.safe_dump(yaml_data, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    (output / "merge_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, type=Path,
                        help="One or more normalized one-class YOLO lab_coat dataset directories")
    parser.add_argument("--output", required=True, type=Path,
                        help="New merged dataset directory; must not already exist")
    args = parser.parse_args()
    try:
        report = merge(args.inputs, args.output)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nMerged data.yaml: {(args.output.resolve() / 'data.yaml')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
