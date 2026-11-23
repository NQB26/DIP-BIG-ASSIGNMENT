#!/usr/bin/env python3
"""Inspect source videos and create contact sheets without changing originals.

Run from the project root:
    python scripts/inspect_videos.py --data-dir data --output-dir reports/video_review

Outputs:
    video_inventory.json / .csv
    video_review_manifest.csv (blank fields for human review)
    contact_sheets/<video-stem>.jpg
    contact_sheet_overview.jpg
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv", ".mpeg", ".mpg"}
EXCLUDED_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache"}


def safe_stem(path: Path) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("._")
    return value or "video"


def fit_thumbnail(frame, width: int, height: int, cv2):
    h, w = frame.shape[:2]
    if h <= 0 or w <= 0:
        return None
    scale = min(width / w, height / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = cv2.copyMakeBorder(
        resized,
        (height - new_h) // 2,
        height - new_h - (height - new_h) // 2,
        (width - new_w) // 2,
        width - new_w - (width - new_w) // 2,
        cv2.BORDER_CONSTANT,
        value=(28, 28, 28),
    )
    return canvas


def timestamp_text(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "time unknown"
    total_ms = max(0, int(round(seconds * 1000)))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def sample_positions(frame_count: int, requested: int) -> list[int]:
    if frame_count <= 0:
        return []
    n = max(1, min(requested, frame_count))
    if n == 1:
        return [0]
    values = [int(round(i * (frame_count - 1) / (n - 1))) for i in range(n)]
    return list(dict.fromkeys(values))


def inspect_one(path: Path, project: Path, cv2, samples_per_video: int,
                thumb_w: int, thumb_h: int, output_dir: Path) -> tuple[dict[str, Any], Any | None]:
    rel = str(path.resolve().relative_to(project.resolve()))
    item: dict[str, Any] = {
        "path": rel,
        "size_bytes": path.stat().st_size,
        "readable": False,
        "width": None,
        "height": None,
        "fps": None,
        "frame_count": None,
        "duration_sec": None,
        "codec": None,
        "sample_frames_read": 0,
        "error": None,
    }
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        item["error"] = "OpenCV could not open this video"
        cap.release()
        return item, None

    try:
        fps_raw = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fourcc_num = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
        codec = "".join(chr((fourcc_num >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00 ")
        fps = fps_raw if math.isfinite(fps_raw) and fps_raw > 0 else None
        duration = frame_count / fps if fps and frame_count > 0 else None
        item.update({
            "readable": True,
            "width": width or None,
            "height": height or None,
            "fps": round(fps, 4) if fps else None,
            "frame_count": frame_count or None,
            "duration_sec": round(duration, 3) if duration is not None else None,
            "codec": codec or None,
        })

        positions = sample_positions(frame_count, samples_per_video)
        cards = []
        for idx, frame_idx in enumerate(positions):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            seconds = frame_idx / fps if fps else None
            thumb = fit_thumbnail(frame, thumb_w, thumb_h, cv2)
            if thumb is None:
                continue
            cv2.putText(thumb, f"Frame {frame_idx} | {timestamp_text(seconds)}", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cards.append(thumb)

        item["sample_frames_read"] = len(cards)
        if cards:
            gap = 10
            label_h = 38
            cols = min(2, len(cards))
            rows = math.ceil(len(cards) / cols)
            sheet = cv2.copyMakeBorder(cards[0], 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=(28, 28, 28))
            sheet_h = rows * (thumb_h + label_h) + (rows + 1) * gap
            sheet_w = cols * thumb_w + (cols + 1) * gap
            sheet = __import__("numpy").full((sheet_h, sheet_w, 3), 28, dtype="uint8")
            for i, thumb in enumerate(cards):
                row, col = divmod(i, cols)
                x, y = gap + col * thumb_w, gap + row * (thumb_h + label_h)
                sheet[y:y + thumb_h, x:x + thumb_w] = thumb
            title = f"{path.name} | {width}x{height} | {fps or 'unknown'} FPS | {timestamp_text(duration)}"
            cv2.putText(sheet, title[:120], (gap, sheet_h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (230, 230, 230), 1, cv2.LINE_AA)
            out = output_dir / "contact_sheets" / f"{safe_stem(path)}.jpg"
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            item["contact_sheet"] = str(out.resolve().relative_to(project.resolve()))
            # Midpoint thumbnail for the combined overview.
            overview = fit_thumbnail(cards[len(cards) // 2], 480, 270, cv2)
            if overview is not None:
                cv2.putText(overview, path.name[:65], (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                            0.58, (255, 255, 255), 1, cv2.LINE_AA)
                return item, overview
        return item, None
    except Exception as exc:
        item["error"] = f"Inspection error: {type(exc).__name__}: {exc}"
        return item, None
    finally:
        cap.release()


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect video metadata and create contact sheets; source videos remain unchanged.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="reports/video_review")
    parser.add_argument("--samples-per-video", type=int, default=5)
    parser.add_argument("--thumbnail-width", type=int, default=640)
    parser.add_argument("--thumbnail-height", type=int, default=360)
    args = parser.parse_args()

    if args.samples_per_video < 1:
        parser.error("--samples-per-video must be >= 1")
    if args.thumbnail_width < 160 or args.thumbnail_height < 90:
        parser.error("thumbnail dimensions are too small")

    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except Exception as exc:
        raise SystemExit(f"OpenCV and NumPy are required. Install project requirements first. Details: {exc}")

    project = Path.cwd().resolve()
    data_dir = (project / args.data_dir).resolve()
    output_dir = (project / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not data_dir.exists():
        raise SystemExit(f"Data directory does not exist: {data_dir}")

    videos = sorted(p for p in data_dir.rglob("*")
                    if p.is_file() and p.suffix.lower() in VIDEO_EXTS
                    and not any(part in EXCLUDED_DIRS for part in p.parts))
    records: list[dict[str, Any]] = []
    overview_cards = []
    for i, path in enumerate(videos, start=1):
        print(f"[{i}/{len(videos)}] Inspecting: {path}")
        record, overview = inspect_one(path, project, cv2, args.samples_per_video,
                                       args.thumbnail_width, args.thumbnail_height, output_dir)
        records.append(record)
        if overview is not None:
            overview_cards.append((record["path"], overview))

    json_path = output_dir / "video_inventory.json"
    json_path.write_text(json.dumps({
        "project_root": str(project),
        "data_dir": str(data_dir),
        "opencv_version": cv2.__version__,
        "video_count": len(records),
        "videos": records,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = output_dir / "video_inventory.csv"
    fieldnames = ["path", "size_bytes", "readable", "width", "height", "fps", "frame_count",
                  "duration_sec", "codec", "sample_frames_read", "contact_sheet", "error"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    manifest_path = output_dir / "video_review_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as handle:
        fields = ["video_path", "session_group", "split", "usable", "contains_person", "contains_lab_coat",
                  "contains_phone", "phone_small_or_blurry", "notes"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({"video_path": record["path"]})

    if overview_cards:
        cols = 3
        card_w, card_h, label_h, gap = 480, 270, 30, 12
        rows = math.ceil(len(overview_cards) / cols)
        canvas = np.full((rows * (card_h + label_h) + (rows + 1) * gap,
                          cols * card_w + (cols + 1) * gap, 3), 28, dtype=np.uint8)
        for index, (name, thumb) in enumerate(overview_cards):
            row, col = divmod(index, cols)
            x, y = gap + col * card_w, gap + row * (card_h + label_h)
            canvas[y:y + card_h, x:x + card_w] = thumb
            cv2.putText(canvas, name[:72], (x + 4, y + card_h + 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, (230, 230, 230), 1, cv2.LINE_AA)
        overview_path = output_dir / "contact_sheet_overview.jpg"
        cv2.imwrite(str(overview_path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    else:
        overview_path = None

    readable = sum(bool(r["readable"]) for r in records)
    print("\n=== VIDEO REVIEW INVENTORY ===")
    print(f"Videos found:       {len(records)}")
    print(f"Readable videos:    {readable}")
    print(f"Unreadable videos:  {len(records) - readable}")
    print(f"JSON:               {json_path}")
    print(f"CSV:                {csv_path}")
    print(f"Review manifest:    {manifest_path}")
    if overview_path:
        print(f"Overview sheet:     {overview_path}")
    print(f"Per-video sheets:   {output_dir / 'contact_sheets'}")
    return 0 if records and readable == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
