"""Review per-frame association of lab_coat / cell_phone detections to person tracks.

This is a diagnostic pipeline only. It does not trigger R1/R2/R3 or write RuleEvents.
It uses the same geometric rules as the current RuleEngine:
- lab_coat: IoA(coat bbox, person bbox) >= R1.min_ioa_threshold; choose max IoA.
- cell_phone: phone center inside person's upper-body ROI; choose nearest ROI center.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from perception import MultiYoloDetector  # noqa: E402
from tracking_rules import PersonTracker, load_rules_config  # noqa: E402
from tracking_rules.geometry import compute_ioa, center, point_in_bbox, upper_body_roi  # noqa: E402


def _center_of_bbox(box: Sequence[float]) -> tuple[float, float]:
    return center(box)


def associate_lab_coats(coats, tracks, min_ioa: float, min_person_height_px: float = 0.0):
    """Return one association record per coat detection using RuleEngine R1 geometry."""
    output = []
    persons = [t for t in tracks if t.class_name == "person" and
               (t.bbox[3] - t.bbox[1]) >= min_person_height_px]
    for det_index, coat in enumerate(coats):
        candidates = []
        for track in persons:
            score = compute_ioa(coat.bbox, track.bbox)
            candidates.append((float(score), track))
        candidates.sort(key=lambda x: (x[0], x[1].track_id), reverse=True)
        best_score, best_track = candidates[0] if candidates else (0.0, None)
        matched = best_track is not None and best_score >= min_ioa
        output.append({
            "class_name": "lab_coat",
            "detection_index": det_index,
            "detection": coat,
            "track_id": best_track.track_id if matched else None,
            "score": best_score if best_track is not None else 0.0,
            "candidate_count": len(candidates),
            "matched": matched,
            "reason": "ioa_threshold_met" if matched else (
                "below_ioa_threshold" if best_track is not None else "no_eligible_person"
            ),
        })
    return output


def associate_phones(phones, tracks, upper_body_ratio: float):
    """Return one association record per phone using RuleEngine R3 geometry."""
    output = []
    persons = [t for t in tracks if t.class_name == "person"]
    for det_index, phone in enumerate(phones):
        p_center = _center_of_bbox(phone.bbox)
        candidates = []
        for track in persons:
            roi = upper_body_roi(track.bbox, upper_body_ratio)
            if point_in_bbox(p_center, roi):
                roi_center = _center_of_bbox(roi)
                distance = math.hypot(p_center[0] - roi_center[0], p_center[1] - roi_center[1])
                candidates.append((float(distance), track))
        candidates.sort(key=lambda x: (x[0], x[1].track_id))
        best_distance, best_track = candidates[0] if candidates else (None, None)
        output.append({
            "class_name": "cell_phone",
            "detection_index": det_index,
            "detection": phone,
            "track_id": best_track.track_id if best_track is not None else None,
            "score": best_distance,
            "candidate_count": len(candidates),
            "matched": best_track is not None,
            "reason": "center_inside_upper_body_roi" if best_track is not None else "outside_all_upper_body_rois",
        })
    return output


def _draw_box(frame, bbox, color, label, thickness=2):
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    h, w = frame.shape[:2]
    x1, x2 = max(0, min(w - 1, x1)), max(0, min(w - 1, x2))
    y1, y2 = max(0, min(h - 1, y1)), max(0, min(h - 1, y2))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    text_y = max(18, y1 - 6)
    cv2.putText(frame, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)


def _draw_link(frame, det_bbox, track, color, label):
    det_center = tuple(int(round(v)) for v in _center_of_bbox(det_bbox))
    person_center = tuple(int(round(v)) for v in _center_of_bbox(track.bbox))
    cv2.line(frame, det_center, person_center, color, 2, cv2.LINE_AA)
    mx = (det_center[0] + person_center[0]) // 2
    my = (det_center[1] + person_center[1]) // 2
    cv2.putText(frame, label, (mx + 4, my - 4), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, color, 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description="Visual audit of lab_coat/phone to person-track associations")
    ap.add_argument("--video", required=True)
    ap.add_argument("--weights", required=True, help="COCO yolo11n.pt checkpoint")
    ap.add_argument("--lab-coat-weights", required=True, help="custom lab_coat best.pt checkpoint")
    ap.add_argument("--rules", default=str(ROOT / "configs" / "rules.yaml"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "association_review_v1"))
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--coco-conf", type=float, default=0.25)
    ap.add_argument("--coat-conf", type=float, default=0.25)
    ap.add_argument("--min-ioa", type=float, default=None)
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    if args.max_frames < 0:
        ap.error("--max-frames must be >= 0 (0 means all frames)")

    cfg = load_rules_config(args.rules)
    r1 = cfg["rules"]["R1"]
    r3 = cfg["rules"]["R3"]
    min_ioa = float(r1.get("min_ioa_threshold", 0.6) if args.min_ioa is None else args.min_ioa)
    if not 0.0 <= min_ioa <= 1.0:
        ap.error("--min-ioa must be in [0, 1]")
    upper_ratio = float(r3.get("upper_body_ratio", 0.65))
    if not 0.0 < upper_ratio <= 1.0:
        ap.error("R3.upper_body_ratio in rules YAML must be in (0, 1]")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input video: {args.video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Invalid video size: {width}x{height}")

    tracker_cfg = cfg["tracker"]
    tracker = PersonTracker(
        fps=fps,
        lost_buffer_sec=tracker_cfg["lost_buffer_sec"],
        activation_threshold=tracker_cfg["activation_threshold"],
        high_conf_threshold=tracker_cfg["high_conf_threshold"],
        min_consecutive_frames=tracker_cfg["min_consecutive_frames"],
        min_iou=tracker_cfg["min_iou"],
    )
    detector = MultiYoloDetector(
        coco_weights=args.weights,
        lab_coat_weights=args.lab_coat_weights,
        imgsz=args.imgsz,
        device=args.device,
        filter_coco_classes=True,
        coco_conf=args.coco_conf,
        lab_coat_conf=args.coat_conf,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_dir / "association_review.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create output video in {out_dir}; check codec/path")

    csv_path = out_dir / "associations.csv"
    summary_path = out_dir / "summary.json"
    fieldnames = ["frame_id", "time_sec", "class_name", "det_confidence",
                  "x1", "y1", "x2", "y2", "track_id", "association_score",
                  "candidate_count", "matched", "reason"]
    count = Counter()
    associated_tracks = defaultdict(Counter)
    total_tracks = 0
    frames = 0
    started = time.perf_counter()

    try:
        with csv_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            csv_writer.writeheader()
            while True:
                ok, frame = cap.read()
                if not ok or (args.max_frames and frames >= args.max_frames):
                    break
                detections = detector(frame, frames)
                tracks = tracker.update(detections)
                total_tracks += len(tracks)

                coats = [d for d in detections if d.class_name == "lab_coat" and
                         d.confidence >= float(r1.get("coat_conf_min", 0.4))]
                phones = [d for d in detections if d.class_name == "cell_phone" and
                          d.confidence >= float(r3.get("phone_conf_min", 0.3))]
                coat_links = associate_lab_coats(
                    coats, tracks, min_ioa,
                    float(r1.get("min_person_height_px", cfg["common"].get("min_person_height_ratio", 0.25) * height)),
                )
                phone_links = associate_phones(phones, tracks, upper_ratio)
                all_links = coat_links + phone_links

                for item in all_links:
                    det = item["detection"]
                    x1, y1, x2, y2 = det.bbox
                    score = item["score"]
                    csv_writer.writerow({
                        "frame_id": frames,
                        "time_sec": round(frames / fps, 4),
                        "class_name": item["class_name"],
                        "det_confidence": round(float(det.confidence), 5),
                        "x1": round(x1, 2), "y1": round(y1, 2),
                        "x2": round(x2, 2), "y2": round(y2, 2),
                        "track_id": item["track_id"] if item["track_id"] is not None else "",
                        "association_score": round(float(score), 5) if score is not None else "",
                        "candidate_count": item["candidate_count"],
                        "matched": int(item["matched"]),
                        "reason": item["reason"],
                    })
                    count[f"{item['class_name']}_detections"] += 1
                    count[f"{item['class_name']}_matched"] += int(item["matched"])
                    count[f"{item['class_name']}_unmatched"] += int(not item["matched"])
                    if item["matched"]:
                        associated_tracks[item["class_name"]][item["track_id"]] += 1

                # Person tracks: green; coat: orange; phone: magenta.
                frame_copy = frame.copy()
                track_by_id = {t.track_id: t for t in tracks}
                for track in tracks:
                    _draw_box(frame_copy, track.bbox, (0, 220, 0), f"PERSON ID {track.track_id}")
                for item in all_links:
                    det = item["detection"]
                    class_name = item["class_name"]
                    color = (0, 165, 255) if class_name == "lab_coat" else (255, 0, 255)
                    if item["matched"]:
                        label = (f"{class_name} -> ID {item['track_id']} "
                                 f"score={item['score']:.2f}")
                        _draw_box(frame_copy, det.bbox, color, label, 2)
                        _draw_link(frame_copy, det.bbox, track_by_id[item["track_id"]], color, label)
                    else:
                        _draw_box(frame_copy, det.bbox, (0, 0, 255),
                                  f"{class_name} UNMATCHED ({item['reason']})", 2)
                cv2.putText(frame_copy, f"frame={frames} t={frames / fps:.2f}s", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
                writer.write(frame_copy)
                if args.show:
                    cv2.imshow("Association review - press q to stop", frame_copy)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                frames += 1
    finally:
        cap.release()
        writer.release()
        if args.show:
            cv2.destroyAllWindows()

    elapsed = max(time.perf_counter() - started, 1e-9)
    summary = {
        "video": str(Path(args.video).resolve()),
        "frames_processed": frames,
        "input_fps": fps,
        "processing_seconds": round(elapsed, 3),
        "processing_fps": round(frames / elapsed, 3),
        "mean_active_tracks_per_frame": round(total_tracks / frames, 3) if frames else 0.0,
        "min_lab_coat_ioa": min_ioa,
        "phone_upper_body_ratio": upper_ratio,
        "counts": dict(count),
        "matched_track_ids_by_class": {
            cls: {str(tid): n for tid, n in sorted(per_track.items())}
            for cls, per_track in associated_tracks.items()
        },
        "note": "Diagnostic association only; this script does not emit safety RuleEvents.",
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Outputs: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
