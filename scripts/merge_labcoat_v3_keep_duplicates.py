#!/usr/bin/env python3
"""Merge approved curated Lab Coat V1 + V2 into YOLO V3 WITHOUT deduplication.

This utility intentionally retains every approved image, including exact duplicate
images across source datasets and splits. It reports duplicate groups and possible
cross-split leakage but does not fail solely because duplicates exist.
Source curated datasets are never modified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_VERSION = "1.13.0-keep-duplicates"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
APPROVED = {"verified_positive", "verified_negative"}
SPLITS = {"train", "val", "test"}


@dataclass
class Item:
    source: str
    root: Path
    split: str
    image: Path
    label: Path
    image_rel: str
    label_rel: str
    status: str
    sha256: str
    boxes: list[list[float]]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_child(root: Path, relative: str, field_name: str) -> Path:
    rel = Path(relative)
    if rel.is_absolute():
        raise ValueError(f"{field_name} must be relative, got: {relative}")
    candidate = (root / rel).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError(f"{field_name} escapes dataset root: {relative}")
    return candidate


def read_yolo_label(path: Path) -> list[list[float]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing label file: {path}")
    boxes: list[list[float]] = []
    text = path.read_text(encoding="utf-8-sig")
    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_no}: expected 5 YOLO fields, found {len(fields)}")
        try:
            cls, xc, yc, width, height = map(float, fields)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_no}: non-numeric YOLO annotation") from exc
        if abs(cls) > 1e-6:
            raise ValueError(f"{path}:{line_no}: expected class 0, found {cls}")
        vals = (xc, yc, width, height)
        if not all(0.0 <= value <= 1.0 for value in vals) or width <= 0 or height <= 0:
            raise ValueError(f"{path}:{line_no}: invalid normalized xywh {vals}")
        if (xc - width / 2 < -1e-6 or yc - height / 2 < -1e-6
                or xc + width / 2 > 1 + 1e-6 or yc + height / 2 > 1 + 1e-6):
            raise ValueError(f"{path}:{line_no}: bounding box exceeds image boundary")
        boxes.append([xc, yc, width, height])
    return boxes


def load_curated(source: str, root_arg: Path) -> tuple[list[Item], dict[str, Any]]:
    root = root_arg.resolve()
    summary_path = root / "review_summary.json"
    manifest_path = root / "review_manifest.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(f"{source}: missing {summary_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{source}: missing {manifest_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"image", "label", "split", "status"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{source}: review_manifest.csv missing columns {sorted(missing)}")
        rows = list(reader)

    status_counts = Counter((row.get("status") or "pending").strip() for row in rows)
    blockers = {status: status_counts[status] for status in ("pending", "needs_review") if status_counts[status]}
    if blockers:
        raise RuntimeError(f"{source}: unresolved QA statuses remain: {blockers}")

    approved_count = status_counts["verified_positive"] + status_counts["verified_negative"]
    if approved_count != int(summary.get("approved_total", -1)):
        raise RuntimeError(f"{source}: approved_total in summary does not match manifest")
    if len(rows) != int(summary.get("total_images", -1)):
        raise RuntimeError(f"{source}: total_images in summary does not match manifest")

    items: list[Item] = []
    for row_num, row in enumerate(rows, start=2):
        status = (row.get("status") or "pending").strip()
        if status not in APPROVED:
            # Rejected rows are intentionally excluded; their files should not be copied.
            if status not in {"rejected", ""}:
                raise ValueError(f"{source}: unsupported review status {status!r} at CSV row {row_num}")
            continue

        image_rel = (row.get("image") or "").strip()
        label_rel = (row.get("label") or "").strip()
        if not image_rel or not label_rel:
            raise ValueError(f"{source}: blank image/label path at CSV row {row_num}")

        image = safe_child(root, image_rel, "image path")
        label = safe_child(root, label_rel, "label path")
        if image.suffix.lower() not in IMAGE_EXTS:
            raise ValueError(f"{source}: unsupported image extension: {image}")
        if not image.is_file():
            raise FileNotFoundError(f"{source}: approved image is missing: {image}")
        boxes = read_yolo_label(label)
        if status == "verified_positive" and not boxes:
            raise ValueError(f"{source}: positive image has empty annotation: {label}")
        if status == "verified_negative" and boxes:
            raise ValueError(f"{source}: negative image has non-empty annotation: {label}")
        parts = Path(image_rel).parts
        if len(parts) < 3 or parts[0].lower() != "images":
            raise ValueError(f"{source}: expected image path images/<split>/<filename>, got {image_rel}")
        split = parts[1].lower()
        if split not in SPLITS:
            raise ValueError(f"{source}: unsupported split {split!r} in {image_rel}")
        if Path(label_rel).suffix.lower() != ".txt":
            raise ValueError(f"{source}: YOLO label must end in .txt: {label_rel}")
        if "box_count" in row and row["box_count"].strip():
            try:
                manifest_box_count = int(row["box_count"])
            except ValueError as exc:
                raise ValueError(f"{source}: invalid box_count in manifest row {row_num}") from exc
            if manifest_box_count != len(boxes):
                raise ValueError(
                    f"{source}: box_count mismatch for {image.name}: manifest={manifest_box_count}, file={len(boxes)}"
                )

        items.append(Item(
            source=source, root=root, split=split, image=image, label=label,
            image_rel=Path(image_rel).as_posix(), label_rel=Path(label_rel).as_posix(),
            status=status, sha256=sha256_file(image), boxes=boxes,
        ))

    if len(items) != approved_count:
        raise RuntimeError(f"{source}: found {len(items)} approved image rows; expected {approved_count}")
    return items, summary


def duplicate_report(items: list[Item]) -> list[dict[str, Any]]:
    by_hash: dict[str, list[Item]] = defaultdict(list)
    for item in items:
        by_hash[item.sha256].append(item)
    groups = []
    for digest, group in by_hash.items():
        if len(group) < 2:
            continue
        splits = sorted({item.split for item in group})
        groups.append({
            "sha256": digest,
            "copies": len(group),
            "sources": [f"{x.source}:{x.split}:{x.image.name}" for x in group],
            "splits": splits,
            "cross_split_leakage": len(splits) > 1,
        })
    return sorted(groups, key=lambda g: g["sha256"])


def target_filename(item: Item) -> str:
    # Always prefix source to prevent filename collisions; content hashes are unchanged.
    return f"{item.source}__{item.image.name}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge approved Lab Coat V1/V2 curated data into V3 while KEEPING exact duplicates."
    )
    parser.add_argument("--v1", type=Path, required=True, help="Curated V1 directory")
    parser.add_argument("--v2", type=Path, required=True, help="Curated V2 directory")
    parser.add_argument("--output", type=Path, required=True, help="New V3 output directory")
    parser.add_argument("--report", type=Path, help="Optional external JSON summary path")
    parser.add_argument("--dry-run", action="store_true", help="Validate/plan only; do not copy files")
    args = parser.parse_args()

    print(f"MERGE SCRIPT VERSION: {SCRIPT_VERSION}")
    try:
        v1_items, v1_summary = load_curated("v1", args.v1)
        v2_items, v2_summary = load_curated("v2", args.v2)
        items = v1_items + v2_items

        split_counts = Counter(x.split for x in items)
        object_counts = Counter()
        for item in items:
            object_counts[item.split] += len(item.boxes)
        duplicate_groups = duplicate_report(items)
        cross_split_groups = [g for g in duplicate_groups if g["cross_split_leakage"]]

        # A fresh destination is mandatory. Never delete or overwrite prior output.
        output = args.output.resolve()
        if output.exists() and any(output.iterdir()) and not args.dry_run:
            raise FileExistsError(
                f"Output directory is not empty: {output}. Choose a new destination or move the old output aside."
            )

        report: dict[str, Any] = {
            "schema_version": 1,
            "script_version": SCRIPT_VERSION,
            "deduplication_performed": False,
            "duplicates_policy": "KEEP_ALL_APPROVED_IMAGES; exact duplicates are reported but not removed",
            "sources": {
                "v1": {"root": str(args.v1.resolve()), "review_summary": v1_summary},
                "v2": {"root": str(args.v2.resolve()), "review_summary": v2_summary},
            },
            "total_images": len(items),
            "images_by_split": {split: split_counts.get(split, 0) for split in ("train", "val", "test")},
            "objects_by_split": {split: object_counts.get(split, 0) for split in ("train", "val", "test")},
            "positive_images": sum(1 for x in items if x.status == "verified_positive"),
            "negative_images": sum(1 for x in items if x.status == "verified_negative"),
            "exact_duplicate_groups": len(duplicate_groups),
            "exact_duplicate_extra_copies": sum(g["copies"] - 1 for g in duplicate_groups),
            "cross_split_duplicate_groups": len(cross_split_groups),
            "duplicate_groups": duplicate_groups,
            "warning": (
                "Exact duplicate images are intentionally retained. If duplicates span train/val/test, "
                "validation/test metrics may be optimistic due to data leakage."
            ),
            "dry_run": bool(args.dry_run),
            "output": str(output),
        }

        print("\n=== V3 MERGE PLAN (KEEP DUPLICATES) ===")
        print(f"V1 approved images: {len(v1_items)}")
        print(f"V2 approved images: {len(v2_items)}")
        print(f"Total V3 images: {len(items)}")
        print(f"Images by split: {dict(report['images_by_split'])}")
        print(f"Objects by split: {dict(report['objects_by_split'])}")
        print(f"Exact duplicate groups retained: {len(duplicate_groups)}")
        print(f"Cross-split duplicate groups: {len(cross_split_groups)}")
        for index, group in enumerate(duplicate_groups, start=1):
            print(f"  Duplicate {index}: copies={group['copies']}, splits={group['splits']}")
            for source in group["sources"]:
                print(f"    - {source}")
        print("WARNING: duplicates are intentionally retained; evaluation metrics may be optimistic.")

        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"Plan report: {args.report.resolve()}")

        if args.dry_run:
            print("\nDRY RUN ONLY: no dataset files copied.")
            return 0

        output.mkdir(parents=True, exist_ok=True)
        manifest_rows: list[dict[str, Any]] = []
        seen_targets: set[tuple[str, str]] = set()
        for item in sorted(items, key=lambda x: ({"train": 0, "val": 1, "test": 2}[x.split], x.source, x.image.name.lower())):
            filename = target_filename(item)
            target_key = (item.split, filename.lower())
            if target_key in seen_targets:
                raise RuntimeError(f"Filename collision despite source prefix: {target_key}")
            seen_targets.add(target_key)

            target_image = output / "images" / item.split / filename
            target_label = output / "labels" / item.split / f"{Path(filename).stem}.txt"
            target_image.parent.mkdir(parents=True, exist_ok=True)
            target_label.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item.image, target_image)
            shutil.copy2(item.label, target_label)

            copied_boxes = read_yolo_label(target_label)
            if copied_boxes != item.boxes:
                raise RuntimeError(f"Copied annotation differs from source: {target_label}")
            manifest_rows.append({
                "split": item.split,
                "image": target_image.relative_to(output).as_posix(),
                "label": target_label.relative_to(output).as_posix(),
                "source": item.source,
                "source_image": item.image_rel,
                "source_label": item.label_rel,
                "status": item.status,
                "sha256": item.sha256,
                "object_count": len(copied_boxes),
                "duplicate_retained": False,  # updated below from SHA multiplicity
            })

        hash_counts = Counter(row["sha256"] for row in manifest_rows)
        for row in manifest_rows:
            row["duplicate_retained"] = hash_counts[row["sha256"]] > 1

        manifest_path = output / "merge_manifest.csv"
        with manifest_path.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(manifest_rows[0].keys()) if manifest_rows else [
                "split", "image", "label", "source", "source_image", "source_label", "status",
                "sha256", "object_count", "duplicate_retained",
            ])
            writer.writeheader()
            writer.writerows(manifest_rows)

        yaml_text = (
            "path: .\n"
            "train: images/train\n"
            "val: images/val\n"
            "test: images/test\n"
            "nc: 1\n"
            "names: ['lab_coat']\n"
        )
        (output / "data.yaml").write_text(yaml_text, encoding="utf-8")

        # Verify copied structure and report, but do not reject duplicate image hashes.
        copied_images = 0
        copied_hashes: dict[str, list[str]] = defaultdict(list)
        for split in ("train", "val", "test"):
            image_dir = output / "images" / split
            label_dir = output / "labels" / split
            if not image_dir.exists():
                continue
            for image_path in image_dir.iterdir():
                if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTS:
                    continue
                label_path = label_dir / f"{image_path.stem}.txt"
                if not label_path.is_file():
                    raise RuntimeError(f"V3 label missing for image: {image_path}")
                digest = sha256_file(image_path)
                copied_hashes[digest].append(f"{split}/{image_path.name}")
                copied_images += 1
        if copied_images != len(items):
            raise RuntimeError(f"Final image count mismatch: planned {len(items)}, copied {copied_images}")

        final_duplicate_groups = [
            {"sha256": digest, "copies": len(paths), "files": paths,
             "cross_split_leakage": len({p.split("/", 1)[0] for p in paths}) > 1}
            for digest, paths in copied_hashes.items() if len(paths) > 1
        ]
        report["dry_run"] = False
        report["dataset_yaml"] = str((output / "data.yaml").resolve())
        report["manifest"] = str(manifest_path.resolve())
        report["final_copied_images"] = copied_images
        report["final_duplicate_groups"] = final_duplicate_groups
        (output / "merge_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        if args.report:
            args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

        print(f"\nV3 successfully created: {output}")
        print(f"Copied images: {copied_images}")
        print(f"Exact duplicate groups retained: {len(final_duplicate_groups)}")
        print(f"Dataset YAML: {output / 'data.yaml'}")
        print(f"Merge manifest: {manifest_path}")
        return 0

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
