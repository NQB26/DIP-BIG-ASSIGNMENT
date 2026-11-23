#!/usr/bin/env python3
"""Human-in-the-loop review/editor for a normalized one-class YOLO dataset.

Expected layout:
  dataset/images/{train,val,test}/*.{jpg,png,...}
  dataset/labels/{train,val,test}/*.txt

The source dataset is never modified. Approved images/labels are copied into
an output dataset with the same split folder names. Review state is resumable.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
WINDOW = "YOLO Annotation QA - V1"


@dataclass
class Item:
    split: str
    image_path: Path
    label_path: Path
    rel_image: str
    rel_label: str
    original_boxes: list[list[float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review/edit normalized class-0 YOLO annotations.")
    parser.add_argument("--dataset", type=Path, required=True,
                        help="Normalized dataset root containing images/ and labels/.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Separate curated output root; source data is not modified.")
    parser.add_argument("--class-id", type=int, default=0,
                        help="Class ID expected in normalized labels (default: 0).")
    parser.add_argument("--max-width", type=int, default=1280)
    parser.add_argument("--max-height", type=int, default=760)
    return parser.parse_args()


def read_yolo_label(path: Path, class_id: int) -> list[list[float]]:
    boxes: list[list[float]] = []
    if not path.exists():
        raise FileNotFoundError(
            f"Missing label file: {path}. An empty label file means a verified negative; "
            "a missing file must not be interpreted as a negative image."
        )
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_no}: expected 5 fields, got {len(fields)}")
        try:
            cls_f, xc, yc, w, h = map(float, fields)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_no}: non-numeric value") from exc
        cls = int(cls_f)
        if abs(cls_f - cls) > 1e-6:
            raise ValueError(f"{path}:{line_no}: class id must be integer")
        if cls != class_id:
            raise ValueError(f"{path}:{line_no}: class {cls} found; expected normalized class {class_id}")
        if not all(map(lambda v: 0.0 <= v <= 1.0, (xc, yc, w, h))) or w <= 0 or h <= 0:
            raise ValueError(f"{path}:{line_no}: invalid normalized xywh {(xc, yc, w, h)}")
        boxes.append([xc, yc, w, h])
    return boxes


def discover_items(dataset: Path, class_id: int) -> list[Item]:
    image_root = dataset / "images"
    label_root = dataset / "labels"
    if not image_root.is_dir() or not label_root.is_dir():
        raise FileNotFoundError(
            f"Dataset must contain images/ and labels/ folders: {dataset}"
        )
    items: list[Item] = []
    for split_dir in sorted(p for p in image_root.iterdir() if p.is_dir()):
        split = split_dir.name
        for image_path in sorted(p for p in split_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS):
            rel_inside_split = image_path.relative_to(split_dir).with_suffix(".txt")
            label_path = label_root / split / rel_inside_split
            boxes = read_yolo_label(label_path, class_id)
            items.append(Item(
                split=split,
                image_path=image_path,
                label_path=label_path,
                rel_image=image_path.relative_to(dataset).as_posix(),
                rel_label=label_path.relative_to(dataset).as_posix(),
                original_boxes=boxes,
            ))
    if not items:
        raise RuntimeError(f"No images found under {image_root}")
    return items


def box_to_pixels(box: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    xc, yc, bw, bh = box
    x1 = round((xc - bw / 2) * width)
    y1 = round((yc - bh / 2) * height)
    x2 = round((xc + bw / 2) * width)
    y2 = round((yc + bh / 2) * height)
    return max(0, x1), max(0, y1), min(width - 1, x2), min(height - 1, y2)


def pixels_to_box(rect: tuple[int, int, int, int], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = rect
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width - 1, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height - 1, y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Box must have positive width and height")
    return [
        ((x1 + x2) / 2) / width,
        ((y1 + y2) / 2) / height,
        (x2 - x1) / width,
        (y2 - y1) / height,
    ]


class Reviewer:
    def __init__(self, args: argparse.Namespace, items: list[Item]):
        self.args = args
        self.items = items
        self.output = args.output.resolve()
        self.state_path = self.output / "review_state.json"
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state: dict[str, dict] = {}
        if self.state_path.exists():
            try:
                self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid review state JSON: {self.state_path}") from exc
        self.index = 0
        self.edit_mode = False
        self.current_boxes: list[list[float]] = []
        self.drag_start: Optional[tuple[int, int]] = None
        self.drag_end: Optional[tuple[int, int]] = None
        self.display_scale = 1.0
        self.display_size = (0, 0)
        self.image_size = (0, 0)
        self.display = None

    def state_for(self, item: Item) -> dict:
        saved = self.state.get(item.rel_image)
        if saved:
            return saved
        return {"status": "pending", "boxes": item.original_boxes, "note": ""}

    def persist_state(self) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.state_path)
        self.write_manifest()

    def write_manifest(self) -> None:
        path = self.output / "review_manifest.csv"
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["image", "label", "split", "status", "box_count", "note"])
            writer.writeheader()
            for item in self.items:
                entry = self.state_for(item)
                writer.writerow({
                    "image": item.rel_image,
                    "label": item.rel_label,
                    "split": item.split,
                    "status": entry.get("status", "pending"),
                    "box_count": len(entry.get("boxes", [])),
                    "note": entry.get("note", ""),
                })

    def export_approved(self, item: Item, status: str, boxes: list[list[float]]) -> None:
        image_out = self.output / "images" / item.split / item.image_path.name
        label_rel = Path(item.rel_label)
        label_out = self.output / label_rel
        image_out.parent.mkdir(parents=True, exist_ok=True)
        label_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item.image_path, image_out)
        if status == "verified_positive" and not boxes:
            raise ValueError("Positive annotation cannot be empty; use the negative action only for a verified no-coat image.")
        lines = ["0 " + " ".join(f"{v:.8f}" for v in b) for b in boxes]
        label_out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def set_status(self, item: Item, status: str, note: str = "") -> None:
        boxes = [list(map(float, b)) for b in self.current_boxes]
        self.state[item.rel_image] = {"status": status, "boxes": boxes, "note": note}
        self.persist_state()
        if status in {"verified_positive", "verified_negative"}:
            self.export_approved(item, status, boxes)
        elif status in {"rejected", "needs_review"}:
            # Remove any stale curated copy if the reviewer changes a previous decision.
            out_img = self.output / "images" / item.split / item.image_path.name
            out_lbl = self.output / item.rel_label
            out_img.unlink(missing_ok=True)
            out_lbl.unlink(missing_ok=True)

    def load_current(self) -> None:
        item = self.items[self.index]
        entry = self.state_for(item)
        self.current_boxes = [list(map(float, b)) for b in entry.get("boxes", item.original_boxes)]
        self.edit_mode = False
        self.drag_start = None
        self.drag_end = None

    def draw(self) -> None:
        item = self.items[self.index]
        img = cv2.imread(str(item.image_path))
        if img is None:
            raise RuntimeError(f"Cannot read image: {item.image_path}")
        h, w = img.shape[:2]
        self.image_size = (w, h)
        scale = min(self.args.max_width / w, self.args.max_height / h, 1.0)
        disp_w, disp_h = max(1, round(w * scale)), max(1, round(h * scale))
        self.display_scale = scale
        self.display_size = (disp_w, disp_h)
        canvas = cv2.resize(img, (disp_w, disp_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)

        for i, box in enumerate(self.current_boxes, 1):
            x1, y1, x2, y2 = box_to_pixels(box, w, h)
            p1 = (round(x1 * scale), round(y1 * scale))
            p2 = (round(x2 * scale), round(y2 * scale))
            cv2.rectangle(canvas, p1, p2, (40, 220, 40), 2)
            cv2.putText(canvas, f"coat {i}", (p1[0], max(18, p1[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 220, 40), 2)

        if self.drag_start and self.drag_end:
            cv2.rectangle(canvas, self.drag_start, self.drag_end, (0, 220, 255), 2)

        entry = self.state_for(item)
        reviewed = sum(1 for it in self.items if self.state_for(it).get("status") not in {"pending", "needs_review"})
        approved = sum(1 for it in self.items if self.state_for(it).get("status") in {"verified_positive", "verified_negative"})
        header_h = 116
        panel = cv2.copyMakeBorder(canvas, header_h, 0, 0, 0, cv2.BORDER_CONSTANT, value=(28, 28, 28))
        lines = [
            f"{self.index + 1}/{len(self.items)} | split={item.split} | reviewed={reviewed}/{len(self.items)} | approved={approved}",
            f"{item.image_path.name} | boxes={len(self.current_boxes)} | status={entry.get('status', 'pending')} | {'EDIT MODE' if self.edit_mode else 'VIEW MODE'}",
            "A approve current  E edit  C clear  U undo  S save edits  B verified-negative  F needs-review  X reject",
            "N next  P previous  H help  Q quit | In edit mode: left mouse drag adds a box",
        ]
        for i, line in enumerate(lines):
            cv2.putText(panel, line, (12, 25 + i * 27), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (245, 245, 245), 1, cv2.LINE_AA)
        self.display = panel
        cv2.imshow(WINDOW, panel)

    def on_mouse(self, event: int, x: int, y: int, flags: int, _param) -> None:
        if not self.edit_mode or y < 116:
            return
        y -= 116
        disp_w, disp_h = self.display_size
        if x < 0 or y < 0 or x >= disp_w or y >= disp_h:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
            self.drag_end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_start is not None:
            self.drag_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.drag_start is not None:
            self.drag_end = (x, y)
            x1d, y1d = self.drag_start
            x2d, y2d = self.drag_end
            sx1, sx2 = sorted((x1d, x2d))
            sy1, sy2 = sorted((y1d, y2d))
            s = self.display_scale
            rect = (round(sx1 / s), round(sy1 / s), round(sx2 / s), round(sy2 / s))
            try:
                self.current_boxes.append(pixels_to_box(rect, *self.image_size))
            except ValueError:
                pass
            self.drag_start = None
            self.drag_end = None
            self.draw()

    def help(self) -> None:
        print("""
KEYS
  A : approve the currently displayed boxes as correct (positive image)
  E : toggle edit mode; drag with left mouse to add a bounding box
  C : clear all current boxes (only in edit mode)
  U : undo the most recently added/current box (only in edit mode)
  S : save edited boxes as verified positive (only in edit mode; at least one box)
  B : verify that the image contains NO lab coat (save an empty label)
  F : flag image for further review; it is not exported to curated dataset
  X : reject unusable image; it is not exported
  N/P : next/previous image
  Q : quit; review progress is saved after every decision

Green = current annotation; yellow = box currently being drawn.
The source dataset is never edited. Approved results are copied to --output.
""")

    def run(self) -> None:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        self.load_current()
        self.help()
        while True:
            self.draw()
            key = cv2.waitKey(0) & 0xFF
            item = self.items[self.index]
            key_char = chr(key).lower() if 0 <= key < 256 else ""
            if key_char == "q":
                break
            elif key_char == "n":
                self.index = min(len(self.items) - 1, self.index + 1)
                self.load_current()
            elif key_char == "p":
                self.index = max(0, self.index - 1)
                self.load_current()
            elif key_char == "e":
                self.edit_mode = not self.edit_mode
            elif key_char == "c" and self.edit_mode:
                self.current_boxes = []
            elif key_char == "u" and self.edit_mode and self.current_boxes:
                self.current_boxes.pop()
            elif key_char == "a":
                if not self.current_boxes:
                    print("No boxes currently present. Use B only if you visually confirmed no lab coat.")
                else:
                    self.set_status(item, "verified_positive", "accepted_existing_or_current_boxes")
            elif key_char == "s" and self.edit_mode:
                if not self.current_boxes:
                    print("Cannot save an empty positive annotation. Use B only for a verified negative image.")
                else:
                    self.set_status(item, "verified_positive", "manually_reviewed_or_corrected")
                    self.edit_mode = False
            elif key_char == "b":
                self.current_boxes = []
                self.set_status(item, "verified_negative", "visually_confirmed_no_lab_coat")
            elif key_char == "f":
                self.set_status(item, "needs_review", "requires_second_review")
            elif key_char == "x":
                self.set_status(item, "rejected", "unusable_or_unresolvable")
            elif key_char == "h":
                self.help()
        cv2.destroyAllWindows()
        self.write_manifest()
        statuses = [self.state_for(it).get("status", "pending") for it in self.items]
        summary = {
            "dataset": str(self.args.dataset.resolve()),
            "output": str(self.output),
            "total_images": len(self.items),
            "verified_positive": statuses.count("verified_positive"),
            "verified_negative": statuses.count("verified_negative"),
            "needs_review": statuses.count("needs_review"),
            "rejected": statuses.count("rejected"),
            "pending": statuses.count("pending"),
            "approved_total": statuses.count("verified_positive") + statuses.count("verified_negative"),
        }
        (self.output / "review_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        print(f"Review state: {self.state_path}")
        print(f"Manifest: {self.output / 'review_manifest.csv'}")
        print(f"Curated dataset (approved items only): {self.output}")


def main() -> int:
    args = parse_args()
    args.dataset = args.dataset.resolve()
    args.output = args.output.resolve()
    if args.dataset == args.output or args.dataset in args.output.parents:
        print("ERROR: output must be separate from source dataset.", file=sys.stderr)
        return 2
    try:
        items = discover_items(args.dataset, args.class_id)
        print(f"Found {len(items)} images in {args.dataset}")
        Reviewer(args, items).run()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
