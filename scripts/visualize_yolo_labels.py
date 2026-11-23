from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import cv2

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw YOLO detection labels over images and write an audit report."
    )
    parser.add_argument("--images", required=True, type=Path, help="Directory containing images")
    parser.add_argument("--labels", required=True, type=Path, help="Directory containing YOLO .txt labels")
    parser.add_argument("--output", required=True, type=Path, help="Output directory for annotated copies")
    parser.add_argument("--class-id", type=int, default=None, help="Only draw this class ID; default draws all")
    parser.add_argument("--max-images", type=int, default=0, help="Maximum images to process; 0 means all")
    parser.add_argument("--line-width", type=int, default=2, help="Bounding-box line width")
    return parser.parse_args()


def parse_label_file(label_path: Path, image_width: int, image_height: int):
    boxes = []
    errors = []
    if not label_path.exists():
        return boxes, ["missing_label_file"]

    try:
        lines = label_path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        return boxes, [f"read_error:{exc}"]

    for line_number, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) < 5:
            errors.append(f"line_{line_number}:expected_at_least_5_fields")
            continue
        try:
            class_id = int(float(fields[0]))
            xc, yc, bw, bh = map(float, fields[1:5])
        except ValueError:
            errors.append(f"line_{line_number}:non_numeric_value")
            continue

        if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0):
            errors.append(f"line_{line_number}:coordinates_out_of_normalized_range")
            continue

        x1 = int(round((xc - bw / 2.0) * image_width))
        y1 = int(round((yc - bh / 2.0) * image_height))
        x2 = int(round((xc + bw / 2.0) * image_width))
        y2 = int(round((yc + bh / 2.0) * image_height))
        x1 = max(0, min(image_width - 1, x1))
        y1 = max(0, min(image_height - 1, y1))
        x2 = max(0, min(image_width - 1, x2))
        y2 = max(0, min(image_height - 1, y2))
        if x2 <= x1 or y2 <= y1:
            errors.append(f"line_{line_number}:zero_area_after_clipping")
            continue
        boxes.append((class_id, x1, y1, x2, y2, line_number))
    return boxes, errors


def main() -> int:
    args = parse_args()
    if not args.images.is_dir():
        print(f"ERROR: images directory not found: {args.images}", file=sys.stderr)
        return 2
    if not args.labels.is_dir():
        print(f"ERROR: labels directory not found: {args.labels}", file=sys.stderr)
        return 2
    if args.max_images < 0 or args.line_width < 1:
        print("ERROR: --max-images must be >= 0 and --line-width must be >= 1", file=sys.stderr)
        return 2

    image_paths = sorted(p for p in args.images.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if args.max_images:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        print(f"ERROR: no supported images found in {args.images}", file=sys.stderr)
        return 2

    args.output.mkdir(parents=True, exist_ok=True)
    class_counts: Counter[int] = Counter()
    total_boxes = 0
    missing_labels = 0
    invalid_rows = 0
    unreadable_images = 0
    rows = []

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            unreadable_images += 1
            rows.append({"image": image_path.name, "label": image_path.stem + ".txt", "boxes_total": 0,
                         "boxes_drawn": 0, "classes": "", "status": "unreadable_image", "errors": ""})
            continue
        height, width = image.shape[:2]
        label_path = args.labels / f"{image_path.stem}.txt"
        boxes, errors = parse_label_file(label_path, width, height)
        if not label_path.exists():
            missing_labels += 1

        drawn = 0
        found_classes = []
        for class_id, x1, y1, x2, y2, line_number in boxes:
            class_counts[class_id] += 1
            total_boxes += 1
            found_classes.append(class_id)
            if args.class_id is not None and class_id != args.class_id:
                continue
            color = (0, 220, 0) if args.class_id is None or class_id == args.class_id else (255, 180, 0)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, args.line_width)
            label = f"class {class_id}"
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            top = max(0, y1 - text_h - baseline - 4)
            cv2.rectangle(image, (x1, top), (min(width - 1, x1 + text_w + 6), y1), color, -1)
            cv2.putText(image, label, (x1 + 3, max(text_h, y1 - baseline - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
            drawn += 1

        output_path = args.output / image_path.name
        if not cv2.imwrite(str(output_path), image):
            errors.append("write_failed")
        status = "ok" if not errors else ";".join(errors)
        rows.append({"image": image_path.name, "label": label_path.name, "boxes_total": len(boxes),
                     "boxes_drawn": drawn, "classes": ";".join(map(str, sorted(set(found_classes))),),
                     "status": status, "errors": " | ".join(errors)})
        invalid_rows += sum(1 for e in errors if not e.startswith("missing_label_file"))

    report_path = args.output / "annotation_visualization_report.csv"
    with report_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "label", "boxes_total", "boxes_drawn", "classes", "status", "errors"])
        writer.writeheader()
        writer.writerows(rows)

    print("=== YOLO ANNOTATION VISUALIZATION ===")
    print(f"Images processed : {len(image_paths)}")
    print(f"Total boxes      : {total_boxes}")
    print(f"Class counts     : {dict(sorted(class_counts.items()))}")
    print(f"Missing labels   : {missing_labels}")
    print(f"Invalid rows     : {invalid_rows}")
    print(f"Unreadable imgs  : {unreadable_images}")
    print(f"Annotated output : {args.output.resolve()}")
    print(f"CSV report       : {report_path.resolve()}")
    if total_boxes == 0:
        print("WARNING: No valid YOLO boxes were found. Check the label format and file-name matching.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
