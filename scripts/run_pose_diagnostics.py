"""Run YOLO COCO-17 pose inference alone and export an annotated video + CSV.

This isolates pose inference quality/performance from ByteTrack and the Safety
Rule Engine. It does not create safety events or modify source videos.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from perception.pose_estimator import PoseEstimator  # noqa: E402
from perception.pose_visualization import (  # noqa: E402
    COCO_SKELETON_EDGES,
    keypoint_is_visible,
    pose_visibility_summary,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone YOLO pose diagnostic")
    parser.add_argument("--video", required=True)
    parser.add_argument("--weights", default="yolo11n-pose.pt")
    parser.add_argument("--out", default="runs/pose_diagnostics_v1")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--keypoint-conf", type=float, default=0.35)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if args.imgsz < 32:
        parser.error("--imgsz must be >= 32")
    if not math.isfinite(args.conf) or not 0 <= args.conf <= 1:
        parser.error("--conf must be in [0, 1]")
    if not math.isfinite(args.keypoint_conf) or not 0 <= args.keypoint_conf <= 1:
        parser.error("--keypoint-conf must be in [0, 1]")
    if args.max_frames < 0:
        parser.error("--max-frames must be >= 0 (0 means entire video)")
    return args


def draw_pose(frame, pose, pose_index: int, keypoint_conf: float):
    x1, y1, x2, y2 = map(int, pose.bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
    cv2.putText(frame, f"POSE {pose_index}", (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 2)

    points = pose.keypoints
    for a_idx, b_idx in COCO_SKELETON_EDGES:
        if a_idx >= len(points) or b_idx >= len(points):
            continue
        a, b = points[a_idx], points[b_idx]
        if keypoint_is_visible(a, keypoint_conf) and keypoint_is_visible(b, keypoint_conf):
            cv2.line(frame, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (255, 220, 0), 2)

    for index, point in enumerate(points[:17]):
        if keypoint_is_visible(point, keypoint_conf):
            x, y = int(point[0]), int(point[1])
            color = (0, 255, 255) if index in (9, 10) else (255, 100, 0)
            cv2.circle(frame, (x, y), 4, color, -1)
            if index in (9, 10):
                side = "L_WRIST" if index == 9 else "R_WRIST"
                cv2.putText(frame, side, (x + 5, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

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
        raise RuntimeError(f"Invalid video dimensions: {width}x{height}")

    writer = cv2.VideoWriter(
        str(out_dir / "pose_annotated.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Cannot create pose_annotated.mp4; check OpenCV codec/output path")

    estimator = PoseEstimator(
        weights=args.weights, conf=args.conf, imgsz=args.imgsz, device=args.device
    )

    csv_path = out_dir / "pose_detections.csv"
    csv_file = csv_path.open("w", newline="", encoding="utf-8-sig")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "frame_id", "video_time_sec", "pose_index", "x1", "y1", "x2", "y2",
        "visible_keypoints", "left_wrist_x", "left_wrist_y", "left_wrist_conf",
        "right_wrist_x", "right_wrist_y", "right_wrist_conf",
    ])

    frames = 0
    frames_with_pose = 0
    pose_count = 0
    left_wrist_visible = 0
    right_wrist_visible = 0
    both_wrists_visible = 0
    total_visible_keypoints = 0
    started = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok or (args.max_frames and frames >= args.max_frames):
                break

            poses = estimator(frame)
            if poses:
                frames_with_pose += 1
            pose_count += len(poses)

            for pose_index, pose in enumerate(poses):
                draw_pose(frame, pose, pose_index, args.keypoint_conf)
                summary = pose_visibility_summary(pose.keypoints, args.keypoint_conf)
                total_visible_keypoints += summary["visible_keypoints"]
                left_wrist_visible += int(summary["left_wrist_visible"])
                right_wrist_visible += int(summary["right_wrist_visible"])
                both_wrists_visible += int(summary["both_wrists_visible"])

                left = pose.keypoints[9] if len(pose.keypoints) > 9 else (0.0, 0.0, 0.0)
                right = pose.keypoints[10] if len(pose.keypoints) > 10 else (0.0, 0.0, 0.0)
                csv_writer.writerow([
                    frames, round(frames / fps, 4), pose_index,
                    *[round(float(v), 2) for v in pose.bbox],
                    summary["visible_keypoints"],
                    *[round(float(v), 2) for v in left],
                    *[round(float(v), 2) for v in right],
                ])

            cv2.putText(frame, f"poses={len(poses)} frame={frames}", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            writer.write(frame)
            if args.show:
                cv2.imshow("YOLO Pose Diagnostic", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    frames += 1
                    break
            frames += 1
    finally:
        elapsed = max(time.perf_counter() - started, 1e-9)
        cap.release()
        writer.release()
        csv_file.close()
        if args.show:
            cv2.destroyAllWindows()

    summary = {
        "video": str(Path(args.video).resolve()),
        "weights": args.weights,
        "frames": frames,
        "input_fps": round(fps, 3),
        "processing_seconds": round(elapsed, 3),
        "processing_fps": round(frames / elapsed, 3),
        "frames_with_pose": frames_with_pose,
        "pose_instances": pose_count,
        "mean_pose_instances_per_frame": round(pose_count / frames, 4) if frames else 0.0,
        "left_wrist_visible_instances": left_wrist_visible,
        "right_wrist_visible_instances": right_wrist_visible,
        "both_wrists_visible_instances": both_wrists_visible,
        "mean_visible_keypoints_per_pose": round(total_visible_keypoints / pose_count, 3) if pose_count else 0.0,
        "keypoint_conf_threshold": args.keypoint_conf,
        "pose_conf_threshold": args.conf,
        "imgsz": args.imgsz,
        "device": args.device,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Outputs: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
