"""Vẽ zone lên 1 frame của video và lưu toạ độ chuẩn hoá [0,1] vào configs/zones.yaml.

Phím tắt (cửa sổ OpenCV):
  Chuột trái  : thêm đỉnh          Chuột phải / u : xoá đỉnh cuối
  g           : đóng đa giác -> zone 'general'   (áp dụng R1, R3)
  r           : đóng đa giác -> zone 'restricted' (áp dụng R2)
  d           : xoá zone vừa tạo gần nhất        c : xoá các đỉnh đang vẽ
  s           : lưu file                          q / ESC : thoát
"""
import argparse
import os

import cv2
import numpy as np
import yaml

COLORS = {"general": (0, 200, 0), "restricted": (0, 0, 255)}


def to_norm(points_px, w, h):
    return [[round(x / w, 4), round(y / h, 4)] for x, y in points_px]


def load_existing(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return (yaml.safe_load(f) or {}).get("zones", [])


def save_zones(path, zones, camera_id, w, h):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    data = {"camera_id": camera_id, "reference_frame": {"width": w, "height": h}, "zones": zones}
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True, default_flow_style=None)


def read_frame(video, idx):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Không đọc được frame {idx} của {video}")
    return frame


def render(base, zones, current, scale, w, h):
    img = base.copy()
    overlay = img.copy()
    for z in zones:
        pts = (np.array(z["polygon"]) * [w, h] * scale).astype(np.int32)
        cv2.fillPoly(overlay, [pts], COLORS[z["type"]])
    img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)
    for z in zones:
        pts = (np.array(z["polygon"]) * [w, h] * scale).astype(np.int32)
        cv2.polylines(img, [pts], True, COLORS[z["type"]], 2)
        cv2.putText(img, f'{z["zone_id"]} ({z["type"]})', tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, COLORS[z["type"]], 2)
    if current:
        pts = np.array(current, dtype=np.int32)
        cv2.polylines(img, [pts], False, (0, 255, 255), 2)
        for p in current:
            cv2.circle(img, tuple(int(v) for v in p), 4, (0, 255, 255), -1)
    cv2.putText(img, "LMB add | RMB/u undo | g=general r=restricted | d del zone | s save | q quit",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frame-idx", type=int, default=0)
    ap.add_argument("--out", default="configs/zones.yaml")
    ap.add_argument("--camera-id", default="cam01")
    ap.add_argument("--max-display", type=int, nargs=2, default=[1280, 800], metavar=("W", "H"))
    a = ap.parse_args()

    frame = read_frame(a.video, a.frame_idx)
    h, w = frame.shape[:2]
    scale = min(1.0, a.max_display[0] / w, a.max_display[1] / h)
    base = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1 else frame
    zones = load_existing(a.out)
    current = []                 # đỉnh đang vẽ (toạ độ hiển thị)
    counters = {"general": 0, "restricted": 0}
    for z in zones:
        counters[z["type"]] += 1

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            current.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and current:
            current.pop()

    win = "draw zones"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        cv2.imshow(win, render(base, zones, current, scale, w, h))
        k = cv2.waitKey(30) & 0xFF
        if k in (ord("q"), 27):
            break
        elif k == ord("u") and current:
            current.pop()
        elif k == ord("c"):
            current.clear()
        elif k == ord("d") and zones:
            removed = zones.pop()
            counters[removed["type"]] -= 1
        elif k in (ord("g"), ord("r")) and len(current) >= 3:
            ztype = "general" if k == ord("g") else "restricted"
            counters[ztype] += 1
            pts_px = [(x / scale, y / scale) for x, y in current]
            zid = f"{ztype}_{counters[ztype]}"
            zones.append({"zone_id": zid, "name": zid, "type": ztype,
                          "polygon": to_norm(pts_px, w, h)})
            current.clear()
        elif k == ord("s"):
            save_zones(a.out, zones, a.camera_id, w, h)
            print(f"Đã lưu {len(zones)} zone -> {a.out}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
