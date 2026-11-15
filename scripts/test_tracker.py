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
from collections import defaultdict

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from tracking_rules import (Detection, PersonTracker, RuleEngine, load_rules_config, load_zones)  # noqa: E402
from tracking_rules.geometry import bottom_center, upper_body_roi  # noqa: E402
from tracking_rules.state_machine import State  # noqa: E402


def norm_name(n):
    return n.strip().lower().replace(" ", "_").replace("-", "_")


class YoloDetector:
    def __init__(self, weights, conf, imgsz, device):
        from ultralytics import YOLO
        self.model, self.conf, self.imgsz, self.device = YOLO(weights), conf, imgsz, device

    def __call__(self, frame, frame_id):
        r = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        out = []
        for b in r.boxes:
            out.append(Detection(norm_name(self.model.names[int(b.cls)]), float(b.conf),
                                 tuple(float(v) for v in b.xyxy[0].tolist())))
        return out


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


def draw(frame, zones, dets, tracks, engine, events, banners, t, rcfg):
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
    ap.add_argument("--weights")
    ap.add_argument("--dets-json")
    ap.add_argument("--zones", default=os.path.join(ROOT, "configs", "zones.yaml"))
    ap.add_argument("--rules", default=os.path.join(ROOT, "configs", "rules.yaml"))
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "test01"))
    ap.add_argument("--det-conf", type=float, default=0.25, help="ngưỡng detector (thấp, để ByteTrack dùng detection conf thấp)")
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
    detector = ReplayDetector(a.dets_json) if a.dets_json else YoloDetector(a.weights, a.det_conf, a.imgsz, a.device)

    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tk = cfg["tracker"]
    tracker = PersonTracker(fps=fps, lost_buffer_sec=tk["lost_buffer_sec"],
                            activation_threshold=tk["activation_threshold"],
                            high_conf_threshold=tk["high_conf_threshold"],
                            min_consecutive_frames=tk["min_consecutive_frames"], min_iou=tk["min_iou"])
    engine = RuleEngine(zones, cfg, source_id=os.path.basename(a.video))
    writer = cv2.VideoWriter(os.path.join(a.out, "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    ev_f = open(os.path.join(a.out, "events.jsonl"), "w", encoding="utf-8")
    tr_f = open(os.path.join(a.out, "tracks.csv"), "w", newline="")
    csv_w = csv.writer(tr_f)
    csv_w.writerow(["frame_id", "t", "track_id", "x1", "y1", "x2", "y2", "conf"])

    life = defaultdict(list)
    n_events = defaultdict(int)
    banners = []
    fid = 0
    while True:
        ok, frame = cap.read()
        if not ok or (a.max_frames and fid >= a.max_frames):
            break
        t = fid / fps
        dets = detector(frame, fid)
        tracks = tracker.update(dets)
        events = engine.update(fid, t, tracks, dets, (W, H))
        for tr in tracks:
            life[tr.track_id].append(t)
            csv_w.writerow([fid, round(t, 3), tr.track_id, *[round(v, 1) for v in tr.bbox], round(tr.confidence, 3)])
        for e in events:
            ev_f.write(json.dumps(e, ensure_ascii=False) + "\n")
            n_events[e["rule_id"]] += 1
        draw(frame, zones, dets, tracks, engine, events, banners, t, cfg["rules"])
        writer.write(frame)
        if a.show:
            cv2.imshow("tracker test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        fid += 1
    cap.release(); writer.release(); ev_f.close(); tr_f.close()

    spans = {k: (v[-1] - v[0] + 1 / fps) for k, v in life.items()}
    summary = {
        "frames": fid, "fps": fps, "unique_track_ids": len(life),
        "median_track_life_sec": round(statistics.median(spans.values()), 2) if spans else 0,
        "short_tracks_lt_1s": sum(1 for s in spans.values() if s < 1.0),
        "events_by_rule": dict(n_events),
    }
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print("Kết quả:", a.out)


if __name__ == "__main__":
    main()
