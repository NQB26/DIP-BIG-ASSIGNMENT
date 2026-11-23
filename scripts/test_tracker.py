"""Chạy toàn bộ pipeline Detection -> ByteTrack -> Rule Engine trên video và xuất:
  <out>/annotated.mp4   video có overlay (zone, ID, trạng thái FSM, cảnh báo)
  <out>/events.jsonl    mỗi dòng 1 RuleEvent
  <out>/tracks.csv      frame_id,t,track_id,x1,y1,x2,y2,conf   (để phân tích ID switch / đứt track)
  <out>/summary.json    thống kê nhanh

Detector: --weights best.pt (ultralytics YOLO)  HOẶC  --dets-json file.jsonl
(mỗi dòng: {"frame_id": 0, "detections": [{"class_name","confidence","bbox"}]} - đúng data contract).
"""
import argparse
import csv
import json
import os
import statistics
import sys
import time
import math
from collections import defaultdict

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from tracking_rules import (Detection, PersonTracker, RuleEngine, load_rules_config, load_zones)  # noqa: E402
from tracking_rules.geometry import bottom_center, upper_body_roi  # noqa: E402
from tracking_rules.state_machine import State  # noqa: E402
from perception import MultiYoloDetector  # noqa: E402
from perception.pose_estimator import PoseEstimator, COCO_KEYPOINTS  # noqa: E402
from tracking_rules.pose_association import associate_pose_to_tracks  # noqa: E402


def norm_name(n):
    return n.strip().lower().replace(" ", "_").replace("-", "_")


class YoloDetector:
    """Detection adapter preserving the existing Detection data contract.

    With --lab-coat-weights, the base COCO model emits person + cell_phone,
    while the custom one-class model emits lab_coat. Without it, behavior
    remains compatible with the original single-model mode.
    """
    def __init__(self, weights, conf, imgsz, device, lab_coat_weights=None,
                 coco_conf=None, coat_conf=None):
        self.detector = MultiYoloDetector(
            coco_weights=weights,
            lab_coat_weights=lab_coat_weights,
            conf=conf,
            imgsz=imgsz,
            device=device,
            filter_coco_classes=bool(lab_coat_weights),
            coco_conf=coco_conf,
            lab_coat_conf=coat_conf,
        )

    def __call__(self, frame, frame_id):
        return self.detector(frame, frame_id)


class ReplayDetector:
    def __init__(self, path):
        self.data = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    j = json.loads(line)
                    self.data[j["frame_id"]] = [Detection(norm_name(d.get("class_name") or d["class"]), d["confidence"],
                                                          tuple(d["bbox"])) for d in j.get("detections", []) + j.get("tracks", []) + j.get("objects", [])]

    def __call__(self, frame, frame_id):
        return self.data.get(frame_id, [])


STATE_COLOR = {State.NORMAL: (255, 255, 255), State.PENDING: (0, 215, 255),
               State.CONFIRMED: (0, 0, 255), State.COOLDOWN: (0, 0, 255)}


def draw(frame, zones, dets, tracks, engine, events, banners, t, rcfg, pose_by_track=None):
    h, w = frame.shape[:2]
    ov = frame.copy()
    for z in zones:
        pts = np.array(z.polygon_px(w, h), dtype=np.int32)
        col = (0, 200, 0) if z.type == "general" else (0, 0, 255)
        cv2.fillPoly(ov, [pts], col)
    frame[:] = cv2.addWeighted(ov, 0.2, frame, 0.8, 0)
    for z in zones:
        pts = np.array(z.polygon_px(w, h), dtype=np.int32)
        cv2.polylines(frame, [pts], True, (0, 200, 0) if z.type == "general" else (0, 0, 255), 2)
        cv2.putText(frame, z.zone_id, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    for d in dets:
        if d.class_name in ("lab_coat", "cell_phone"):
            col = (255, 128, 0) if d.class_name == "lab_coat" else (255, 0, 255)
            x1, y1, x2, y2 = map(int, d.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 1)
            cv2.putText(frame, f"{d.class_name} {d.confidence:.2f}", (x1, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
    # Pose overlay: COCO-17 shoulder -> elbow -> wrist chains.
    pose_by_track = pose_by_track or {}
    pose_edges = ((5, 7), (7, 9), (6, 8), (8, 10), (5, 6))
    for track_id, pose in pose_by_track.items():
        kp = pose.keypoints
        for a_idx, b_idx in pose_edges:
            if len(kp) <= max(a_idx, b_idx):
                continue
            ax, ay, ac = kp[a_idx]
            bx, by, bc = kp[b_idx]
            if ac >= 0.35 and bc >= 0.35:
                cv2.line(frame, (int(ax), int(ay)), (int(bx), int(by)), (255, 255, 0), 2)
        for wrist_idx in (9, 10):
            if len(kp) > wrist_idx:
                wx, wy, wc = kp[wrist_idx]
                if wc >= 0.35:
                    cv2.circle(frame, (int(wx), int(wy)), 5, (0, 255, 255), -1)
                    side = "L" if wrist_idx == 9 else "R"
                    cv2.putText(frame, f"{side}:{wc:.2f}", (int(wx) + 5, int(wy) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

    states = engine.debug_states()
    for tr in tracks:
        fsms = {rid: states.get((tr.track_id, rid)) for rid in ("R1", "R2", "R3")}
        worst = State.NORMAL
        lines = []
        for rid, f in fsms.items():
            if f is None or f.state == State.NORMAL:
                continue
            if f.state == State.PENDING:
                lines.append(f"{rid} {f.elapsed:.1f}/{rcfg[rid]['min_duration_sec']:.1f}s")
                worst = State.PENDING if worst == State.NORMAL else worst
            else:
                lines.append(f"{rid} ALERT")
                worst = State.COOLDOWN
        col = STATE_COLOR[worst]
        x1, y1, x2, y2 = map(int, tr.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        cv2.putText(frame, f"ID {tr.track_id} {tr.confidence:.2f}", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
        for i, s in enumerate(lines):
            cv2.putText(frame, s, (x1, y2 + 16 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        fx, fy = map(int, bottom_center(tr.bbox))
        cv2.circle(frame, (fx, fy), 5, (0, 255, 255), -1)          # điểm chân (R2)
        rx1, ry1, rx2, ry2 = map(int, upper_body_roi(tr.bbox, rcfg["R3"]["upper_body_ratio"]))
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 0, 255), 1)  # ROI thân trên (R3)
    for e in events:
        banners.append((t + 2.0, f'{e["rule_name"]} ID{e["track_id"]} @ {e["video_time_sec"]:.1f}s'))
    banners[:] = [b for b in banners if b[0] >= t]
    for i, (_, text) in enumerate(banners[-4:]):
        cv2.putText(frame, text, (10, 24 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"t={t:6.2f}s", (w - 140, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--weights", help="Base detector weights; use yolo11n.pt for COCO person/cell_phone")
    ap.add_argument("--lab-coat-weights", help="Optional fine-tuned one-class lab_coat checkpoint")
    ap.add_argument("--dets-json")
    ap.add_argument("--zones", default=os.path.join(ROOT, "configs", "zones.yaml"))
    ap.add_argument("--rules", default=os.path.join(ROOT, "configs", "rules.yaml"))
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "test01"))
    ap.add_argument("--det-conf", type=float, default=0.25, help="ngưỡng fallback cho cả hai detector")
    ap.add_argument("--coco-conf", type=float, default=None, help="ngưỡng riêng cho person/cell_phone; mặc định dùng --det-conf")
    ap.add_argument("--coat-conf", type=float, default=None, help="ngưỡng riêng cho lab_coat; mặc định dùng --det-conf")
    ap.add_argument("--pose-weights", default=None, help="Optional YOLO11 pose checkpoint, e.g. yolo11n-pose.pt")
    ap.add_argument("--pose-conf", type=float, default=0.25)
    ap.add_argument("--pose-match-iou", type=float, default=0.30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    if not a.weights and not a.dets_json:
        ap.error("cần --weights hoặc --dets-json")

    os.makedirs(a.out, exist_ok=True)
    cfg = load_rules_config(a.rules)
    zones = load_zones(a.zones)
    r3_cfg = cfg["rules"]["R3"]
    if r3_cfg.get("enabled", True) and r3_cfg.get("association_mode", "upper_body_roi") == "pose_hand" and not a.pose_weights:
        ap.error("R3 is configured for pose_hand; provide --pose-weights yolo11n-pose.pt")
    pose_estimator = (PoseEstimator(a.pose_weights, conf=a.pose_conf, imgsz=a.imgsz, device=a.device)
                      if a.pose_weights else None)
    detector = ReplayDetector(a.dets_json) if a.dets_json else YoloDetector(
        a.weights, a.det_conf, a.imgsz, a.device, a.lab_coat_weights,
        a.coco_conf, a.coat_conf
    )

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input video: {a.video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if W <= 0 or H <= 0:
        cap.release()
        raise RuntimeError(f"Invalid video dimensions: {W}x{H}")
    tk = cfg["tracker"]
    tracker = PersonTracker(fps=fps, lost_buffer_sec=tk["lost_buffer_sec"],
                            activation_threshold=tk["activation_threshold"],
                            high_conf_threshold=tk["high_conf_threshold"],
                            min_consecutive_frames=tk["min_consecutive_frames"], min_iou=tk["min_iou"])
    engine = RuleEngine(zones, cfg, source_id=os.path.basename(a.video))
    writer = cv2.VideoWriter(os.path.join(a.out, "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Cannot create annotated.mp4; check output path and OpenCV video codec")
    ev_f = open(os.path.join(a.out, "events.jsonl"), "w", encoding="utf-8")
    tr_f = open(os.path.join(a.out, "tracks.csv"), "w", newline="")
    csv_w = csv.writer(tr_f)
    csv_w.writerow(["frame_id", "t", "track_id", "x1", "y1", "x2", "y2", "conf"])
    pose_f = open(os.path.join(a.out, "pose_tracks.csv"), "w", newline="", encoding="utf-8") if pose_estimator else None
    pose_csv = csv.writer(pose_f) if pose_f else None
    if pose_csv:
        pose_csv.writerow(["frame_id", "t", "track_id", "left_wrist_x", "left_wrist_y", "left_wrist_conf",
                           "right_wrist_x", "right_wrist_y", "right_wrist_conf"])

    life = defaultdict(list)
    n_events = defaultdict(int)
    n_detections = defaultdict(int)
    total_tracks = 0
    pose_instances_total = 0
    pose_matches_total = 0
    banners = []
    fid = 0
    processing_started = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok or (a.max_frames and fid >= a.max_frames):
            break
        t = fid / fps
        dets = detector(frame, fid)
        for detection in dets:
            n_detections[detection.class_name] += 1
        tracks = tracker.update(dets)
        total_tracks += len(tracks)
        pose_by_track = {}
        if pose_estimator is not None:
            pose_instances = pose_estimator(frame)
            pose_instances_total += len(pose_instances)
            pose_by_track = associate_pose_to_tracks(pose_instances, tracks, min_iou=a.pose_match_iou)
            pose_matches_total += len(pose_by_track)
            if pose_csv:
                for track_id, pose in pose_by_track.items():
                    left = pose.keypoints[9] if len(pose.keypoints) > 9 else (0.0, 0.0, 0.0)
                    right = pose.keypoints[10] if len(pose.keypoints) > 10 else (0.0, 0.0, 0.0)
                    pose_csv.writerow([fid, round(t, 3), track_id, *[round(v, 2) for v in left],
                                       *[round(v, 2) for v in right]])
        events = engine.update(fid, t, tracks, dets, (W, H), pose_by_track=pose_by_track)
        for tr in tracks:
            life[tr.track_id].append(t)
            csv_w.writerow([fid, round(t, 3), tr.track_id, *[round(v, 1) for v in tr.bbox], round(tr.confidence, 3)])
        for e in events:
            ev_f.write(json.dumps(e, ensure_ascii=False) + "\n")
            n_events[e["rule_id"]] += 1
        draw(frame, zones, dets, tracks, engine, events, banners, t, cfg["rules"], pose_by_track)
        writer.write(frame)
        if a.show:
            cv2.imshow("tracker test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        fid += 1
    elapsed_processing = max(time.perf_counter() - processing_started, 1e-9)
    cap.release(); writer.release(); ev_f.close(); tr_f.close()
    if pose_f:
        pose_f.close()
    if a.show:
        cv2.destroyAllWindows()

    spans = {k: (v[-1] - v[0] + 1 / fps) for k, v in life.items()}
    summary = {
        "frames": fid, "fps": fps, "input_fps": fps, "processing_seconds": round(elapsed_processing, 3),
        "processing_fps": round(fid / elapsed_processing, 3),
        "detections_by_class": dict(n_detections),
        "mean_tracks_per_frame": round(total_tracks / fid, 3) if fid else 0.0,
        "unique_track_ids": len(life),
        "median_track_life_sec": round(statistics.median(spans.values()), 2) if spans else 0,
        "short_tracks_lt_1s": sum(1 for s in spans.values() if s < 1.0),
        "events_by_rule": dict(n_events),
        "pose": {
            "enabled": pose_estimator is not None,
            "weights": a.pose_weights,
            "pose_instances": pose_instances_total,
            "matched_to_person_tracks": pose_matches_total,
            "pose_track_match_rate": round(pose_matches_total / pose_instances_total, 4) if pose_instances_total else 0.0,
            "match_iou_threshold": a.pose_match_iou if pose_estimator else None,
        },
    }
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print("Kết quả:", a.out)


if __name__ == "__main__":
    main()
