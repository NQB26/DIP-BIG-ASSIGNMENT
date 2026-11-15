"""Sinh video + detection giả (không cần camera/model) để test tracker và R1-R3.

  python scripts/make_synthetic.py --out data/synthetic
  python scripts/test_tracker.py --video data/synthetic/no_coat.mp4 --dets-json data/synthetic/no_coat.jsonl --out runs/no_coat

Mỗi kịch bản có thêm nhiễu: rung bbox, conf dao động, detection rớt ngẫu nhiên.
Toạ độ theo khung 1280x720 và zones.yaml mẫu (general: x 64-1216, y 72-684; restricted: x 832-1178, y 144-540).
"""
import argparse
import json
import os
import random

import cv2
import numpy as np

W, H, FPS = 1280, 720, 30


def box(cx, foot_y, w=150, h=400):
    return [cx - w / 2, foot_y - h, cx + w / 2, foot_y]


def person(cx, fy, rng, drop=0.03):
    if rng.random() < drop:
        return None
    j = lambda: rng.uniform(-3, 3)
    b = box(cx, fy)
    return {"class_name": "person", "confidence": round(rng.uniform(0.75, 0.95), 2), "bbox": [b[0] + j(), b[1] + j(), b[2] + j(), b[3] + j()]}


def coat(cx, fy, rng, drop=0.2):
    if rng.random() < drop:
        return None
    b = box(cx, fy)
    return {"class_name": "lab_coat", "confidence": round(rng.uniform(0.5, 0.9), 2),
            "bbox": [b[0] + 10, b[1] + 60, b[2] - 10, b[1] + 240]}


def phone(cx, fy, rng, on_table=False, drop=0.15):
    if rng.random() < drop:
        return None
    b = box(cx, fy)
    y = b[3] - 60 if on_table else b[1] + 90          # trên bàn: thấp hơn 65% thân
    return {"class_name": "cell_phone", "confidence": round(rng.uniform(0.4, 0.85), 2),
            "bbox": [cx - 15, y, cx + 15, y + 40]}


def scenario(name, seconds, fn, seed=0):
    """fn(t) -> list[(cx, foot_y, has_coat, phone_mode)] với phone_mode in (None,'hand','table')."""
    rng = random.Random(seed)
    vw = cv2.VideoWriter(f"{args.out}/{name}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    with open(f"{args.out}/{name}.jsonl", "w", encoding="utf-8") as f:
        for i in range(int(seconds * FPS)):
            t = i / FPS
            dets = []
            frame = np.full((H, W, 3), 50, np.uint8)
            for (cx, fy, has_coat, ph) in fn(t):
                for d in (person(cx, fy, rng), coat(cx, fy, rng) if has_coat else None,
                          phone(cx, fy, rng, ph == "table") if ph else None):
                    if d:
                        dets.append(d)
                        x1, y1, x2, y2 = map(int, d["bbox"])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 1)
            vw.write(frame)
            f.write(json.dumps({"frame_id": i, "detections": dets}) + "\n")
    vw.release()
    print("tạo", name)


def main():
    global args
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    scenario("coat_ok", 20, lambda t: [(300 + 30 * t, 600, True, None)])                    # kỳ vọng: 0 event
    scenario("no_coat", 20, lambda t: [(300 + 30 * t, 600, False, None)])                   # kỳ vọng: R1 sau ~2.5s (clip dài hơn 17.5s mới lặp lại)
    scenario("restricted", 20, lambda t: [(500 + 60 * t, 520, True, None)])                 # đi vào vùng cấm ở ~x>832: R2
    scenario("restricted_brief", 10, lambda t: [(780 + 150 * max(0.0, 1 - abs(t - 3.5) / 0.5), 520, True, None)])  # chân chạm vùng cấm ~0.65s: 0 event
    scenario("phone_hand", 20, lambda t: [(400, 600, True, "hand")])                        # R3
    scenario("phone_table", 20, lambda t: [(400, 600, True, "table")])                      # 0 event
    scenario("cross_two_people", 20, lambda t: [(250 + 50 * t, 600, True, None),
                                                (1000 - 50 * t, 610, True, None)])           # kiểm tra ID khi hai người đi ngang
    scenario("leave_return", 24, lambda t: [(400, 600, False, None)] if (t < 6 or t > 12) else [])  # mất 6s rồi quay lại: ID mới, không báo trùng


if __name__ == "__main__":
    main()
