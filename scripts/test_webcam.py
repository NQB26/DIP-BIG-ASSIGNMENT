"""Test pipeline Detection -> ByteTrack -> Rule Engine realtime bằng webcam.

  py scripts/test_webcam.py --weights models/best.pt
  py scripts/test_webcam.py --weights models/best.pt --source 1 --flip --imgsz 480 --save-video runs/cam.mp4

--source nhận chỉ số camera (0, 1, ...) hoặc đường dẫn video/RTSP.
Phím: q/ESC thoát | z bật/tắt vẽ zone | p tạm dừng | s lưu ảnh chụp.
Thời gian của FSM = đồng hồ thực (time.monotonic), nên "2.5s" nghĩa là 2.5s thật dù FPS máy thấp.
Lưu ý: nếu dùng --flip (gương) thì vẽ zone bằng draw_zones.py --camera N --flip cho khớp.
"""
import argparse
import json
import os
import sys
import time

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tracking_rules import PersonTracker, RuleEngine, load_rules_config, load_zones  # noqa: E402
from test_tracker import YoloDetector, draw  # noqa: E402


def open_source(src, width, height):
    if src.isdigit():
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        cap = cv2.VideoCapture(int(src), backend)
        if width and height:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    else:
        cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Không mở được nguồn '{src}'. Thử --source 1, đóng app khác đang dùng webcam, "
                         "hoặc kiểm tra quyền camera của Windows.")
    return cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", default="0")
    ap.add_argument("--zones", default=os.path.join(ROOT, "configs", "zones.yaml"))
    ap.add_argument("--no-zones", action="store_true", help="không dùng zone (coi cả khung hình là general)")
    ap.add_argument("--rules", default=os.path.join(ROOT, "configs", "rules.yaml"))
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "webcam"))
    ap.add_argument("--det-conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None, help="vd: cpu, 0")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--assumed-fps", type=float, default=15.0,
                    help="FPS ước lượng của pipeline, dùng cho bộ đệm mất track của ByteTrack")
    ap.add_argument("--flip", action="store_true", help="lật gương ngang")
    ap.add_argument("--save-video", default=None)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--no-show", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    cfg = load_rules_config(a.rules)
    zones = [] if a.no_zones else load_zones(a.zones)
    detector = YoloDetector(a.weights, a.det_conf, a.imgsz, a.device)
    tk = cfg["tracker"]
    tracker = PersonTracker(fps=a.assumed_fps, lost_buffer_sec=tk["lost_buffer_sec"],
                            activation_threshold=tk["activation_threshold"],
                            high_conf_threshold=tk["high_conf_threshold"],
                            min_consecutive_frames=tk["min_consecutive_frames"], min_iou=tk["min_iou"])
    engine = RuleEngine(zones, cfg, source_id=f"webcam_{a.source}", source_type="camera")

    cap = open_source(a.source, a.width, a.height)
    writer = None
    ev_f = open(os.path.join(a.out, "events.jsonl"), "a", encoding="utf-8")
    banners, show_zones, paused = [], True, False
    t0, fid, fps_ema, last = time.monotonic(), 0, 0.0, time.monotonic()
    print("Đang chạy. Nhấn q để thoát.")
    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                print("Hết khung hình / mất kết nối camera.")
                break
            if a.flip:
                frame = cv2.flip(frame, 1)
            H, W = frame.shape[:2]
            now = time.monotonic()
            t = now - t0
            dets = detector(frame, fid)
            tracks = tracker.update(dets)
            events = engine.update(fid, t, tracks, dets, (W, H))
            for e in events:
                ev_f.write(json.dumps(e, ensure_ascii=False) + "\n"); ev_f.flush()
                print(f'[EVENT] {e["rule_id"]} {e["rule_name"]} track={e["track_id"]} zone={e["zone_id"]} '
                      f'duration={e["duration_sec"]:.1f}s')
            draw(frame, zones if show_zones else [], dets, tracks, engine, events, banners, t, cfg["rules"])
            dt = now - last
            last = now
            fps_ema = 1 / dt if fps_ema == 0 else 0.9 * fps_ema + 0.1 * (1 / dt)
            cv2.putText(frame, f"FPS {fps_ema:4.1f}", (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if a.save_video:
                if writer is None:
                    os.makedirs(os.path.dirname(os.path.abspath(a.save_video)), exist_ok=True)
                    writer = cv2.VideoWriter(a.save_video, cv2.VideoWriter_fourcc(*"mp4v"),
                                             max(5.0, fps_ema), (W, H))
                writer.write(frame)
            fid += 1
            if a.max_frames and fid >= a.max_frames:
                break
        if not a.no_show:
            cv2.imshow("webcam tracker test", frame)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            elif k == ord("z"):
                show_zones = not show_zones
            elif k == ord("p"):
                paused = not paused
            elif k == ord("s"):
                path = os.path.join(a.out, f"snap_{fid:06d}.jpg")
                cv2.imwrite(path, frame)
                print("Đã lưu", path)
    cap.release()
    if writer:
        writer.release()
    ev_f.close()
    if not a.no_show:
        cv2.destroyAllWindows()
    print(f"Xong: {fid} frame, FPS trung bình ~{fps_ema:.1f}. Event: {os.path.join(a.out, 'events.jsonl')}")


if __name__ == "__main__":
    main()
