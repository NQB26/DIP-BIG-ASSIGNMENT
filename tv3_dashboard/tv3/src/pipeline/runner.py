#!/usr/bin/env python
"""End-to-end pipeline: video/webcam -> detect -> track -> (pose) -> rules -> evidence.

Example (PowerShell, repo root):
    python src\\pipeline\\runner.py --source .\\data\\zone_test.mp4 `
        --weights .\\yolo11n.pt --lab-coat-weights $coat `
        --zones .\\configs\\zones.yaml --rules .\\configs\\rules.yaml `
        --out .\\runs\\demo --device cpu --imgsz 512

Webcam:  --source 0

Open the dashboard in a second terminal:
    python src\\dashboard\\server.py --runs-dir .\\runs
"""
from __future__ import annotations

import argparse
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

_SRC = Path(__file__).resolve().parents[1]
for _p in (_SRC, _SRC.parent):  # make `pipeline`, `tracking_rules`, `perception` importable
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import cv2  # noqa: E402

from pipeline.event_manager import EvidenceManager  # noqa: E402
from pipeline.eventlog import (LIVE_FILE, STATUS_FILE, SUMMARY_FILE,  # noqa: E402
                               write_json_atomic)
from pipeline.metrics import PipelineMetrics  # noqa: E402
from pipeline.overlay import draw_frame  # noqa: E402
from pipeline.tracker_adapter import build_tracker  # noqa: E402


class FrameSource:
    """cv2.VideoCapture wrapper that yields (ok, frame, video_time_sec, frame_idx).

    Files use video time (frame_idx / fps) so rule timing is deterministic and
    independent of processing speed. Webcams use the wall clock.
    """

    def __init__(self, source: str):
        self.is_live = str(source).isdigit()
        self.label = f"webcam:{source}" if self.is_live else Path(source).name
        if self.is_live:
            backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
            self.cap = cv2.VideoCapture(int(source), backend)
        else:
            if not Path(source).exists():
                raise FileNotFoundError(f"video not found: {source}")
            self.cap = cv2.VideoCapture(str(source))
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open source: {source}")
        fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.fps = fps if 1.0 <= fps <= 240.0 else 25.0
        total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.total_frames = total if (total > 0 and not self.is_live) else 0
        self._idx = 0
        self._t0 = time.perf_counter()

    def read(self):
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return False, None, 0.0, self._idx
        idx = self._idx
        self._idx += 1
        t = (time.perf_counter() - self._t0) if self.is_live else idx / self.fps
        return True, frame, t, idx

    def release(self):
        self.cap.release()


class SafetyPipeline:
    """Orchestrates one run. Components are injected so tests can fake them."""

    def __init__(self, source, detector, tracker, engine, evidence: EvidenceManager,
                 out_dir: str, zones, pose_estimator=None, pose_min_iou: float = 0.30,
                 show: bool = False, frame_stride: int = 1, realtime: bool = False,
                 live_every_sec: float = 0.25, status_every_sec: float = 0.5,
                 live_width: int = 720, save_video: bool = True):
        self.source, self.detector, self.tracker, self.engine = source, detector, tracker, engine
        self.evidence, self.out_dir, self.zones = evidence, Path(out_dir), zones
        self.pose_estimator, self.pose_min_iou = pose_estimator, pose_min_iou
        self.show, self.stride, self.realtime = show, max(1, int(frame_stride)), realtime
        self.live_every, self.status_every = live_every_sec, status_every_sec
        self.live_width, self.save_video = live_width, save_video
        self.metrics = PipelineMetrics()
        self.events_total = 0
        self.by_rule = {"R1": 0, "R2": 0, "R3": 0}
        self._recent: list = []
        self._stop = False
        self.started_at = datetime.now()

    def stop(self, *_):
        self._stop = True

    # ------------------------------------------------------------------ run
    def run(self, max_frames: int = 0) -> dict:
        from tracking_rules.pose_association import associate_pose_to_tracks

        src = self.source
        self.out_dir.mkdir(parents=True, exist_ok=True)
        writer = None
        processed, last_t, last_live, last_status = 0, 0.0, 0.0, 0.0
        wall0 = time.perf_counter()
        state, error = "running", None
        self._write_status(state, 0, 0.0, src, processed)
        try:
            while not self._stop:
                ok, frame, t, idx = src.read()
                if not ok:
                    break
                if max_frames and processed >= max_frames:
                    break
                if idx % self.stride != 0:
                    continue
                if self.realtime and not src.is_live:
                    ahead = (wall0 + t) - time.perf_counter()
                    if ahead > 0:
                        time.sleep(ahead)

                t_frame = time.perf_counter()
                H, W = frame.shape[:2]

                t0 = time.perf_counter()
                detections = self.detector(frame, idx)
                t1 = time.perf_counter()
                tracks = self.tracker.update(detections, idx, t, frame, (W, H))
                t2 = time.perf_counter()
                pose_by_track = {}
                if self.pose_estimator is not None:
                    poses = self.pose_estimator(frame)
                    pose_by_track = associate_pose_to_tracks(poses, tracks, self.pose_min_iou)
                t3 = time.perf_counter()
                events = self.engine.update(idx, t, tracks, detections, (W, H), pose_by_track)
                t4 = time.perf_counter()

                for ev in events:
                    self._recent.append(ev)
                self._recent = [e for e in self._recent if t - e["video_time_sec"] <= 5.0]

                hud = {"t": f"{t:6.1f}s", "frame": idx, "fps": f"{self.metrics.fps_recent:4.1f}",
                       "alerts": self.events_total + len(events)}
                annotated = draw_frame(frame, self.zones, tracks, detections, self.engine,
                                       pose_by_track, self._recent, hud, t)
                t5 = time.perf_counter()

                self.evidence.push_frame(t, annotated)
                for ev in events:
                    ev.setdefault("details", {})["pipeline_latency_ms"] = round(
                        (time.perf_counter() - t_frame) * 1000.0, 1)
                    self.evidence.on_event(ev, annotated)
                    self.events_total += 1
                    self.by_rule[ev["rule_id"]] = self.by_rule.get(ev["rule_id"], 0) + 1

                if self.save_video:
                    if writer is None:
                        writer = cv2.VideoWriter(str(self.out_dir / "pipeline_annotated.mp4"),
                                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                                 max(1.0, src.fps / self.stride), (W, H))
                    writer.write(annotated)

                processed += 1
                last_t = t
                m = self.metrics
                m.record("detect", t1 - t0)
                m.record("track", t2 - t1)
                m.record("pose", t3 - t2)
                m.record("rules", t4 - t3)
                m.record("draw", t5 - t4)
                m.record("total", time.perf_counter() - t_frame)
                m.frame_done()

                now = time.perf_counter()
                if now - last_live >= self.live_every:
                    last_live = now
                    self._write_live(annotated)
                if now - last_status >= self.status_every:
                    last_status = now
                    self._write_status(state, idx, t, src, processed)

                if self.show:
                    cv2.imshow("Safety Monitor", annotated)
                    if (cv2.waitKey(1) & 0xFF) in (ord("q"), ord("Q"), 27):
                        break
        except KeyboardInterrupt:
            pass
        except Exception as exc:
            state, error = "error", f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if state != "error":
                state = "finished"
            if writer is not None:
                writer.release()
            if self.show:
                cv2.destroyAllWindows()
            src.release()
            self.evidence.close()
            summary = self._summary(src, processed, last_t, error)
            write_json_atomic(self.out_dir / SUMMARY_FILE, summary)
            self._write_status(state, max(0, processed - 1), last_t, src, processed, error=error)
        return summary

    # ------------------------------------------------------------- outputs
    def _write_live(self, img):
        h, w = img.shape[:2]
        if w > self.live_width:
            img = cv2.resize(img, (self.live_width, int(h * self.live_width / w)),
                             interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ok:
            return
        tmp = self.out_dir / (LIVE_FILE + ".tmp")
        try:
            tmp.write_bytes(buf.tobytes())
            tmp.replace(self.out_dir / LIVE_FILE)
        except OSError:
            pass  # reader has it open (Windows); next tick will succeed

    def _write_status(self, state, frame_idx, t, src, processed, error=None):
        m = self.metrics
        total = src.total_frames
        write_json_atomic(self.out_dir / STATUS_FILE, {
            "state": state, "error": error, "source": src.label, "is_live": src.is_live,
            "started_at": self.started_at.isoformat(timespec="seconds"),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "frame_id": frame_idx, "processed_frames": processed,
            "total_frames": total or None,
            "progress": round(frame_idx / total, 4) if total else None,
            "video_time_sec": round(t, 2), "source_fps": round(src.fps, 2),
            "fps": round(m.fps_recent, 2), "fps_avg": round(m.fps_avg, 2),
            "latency": m.summary(), "events_total": self.events_total,
            "events_by_rule": self.by_rule, "evidence": dict(self.evidence.stats),
        })

    def _summary(self, src, processed, last_t, error) -> dict:
        m = self.metrics
        # analysed span: each processed frame covers 1/fps (times the stride)
        span = (last_t + self.stride / src.fps) if processed else 0.0
        return {
            "source": src.label, "started_at": self.started_at.isoformat(timespec="seconds"),
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "processed_frames": processed, "frame_stride": self.stride,
            "source_fps": round(src.fps, 2), "analyzed_duration_sec": round(span, 3),
            "fps_avg": round(m.fps_avg, 2), "latency": m.summary(),
            "events_total": self.events_total, "events_by_rule": self.by_rule,
            "evidence": dict(self.evidence.stats), "error": error,
            "note": ("pipeline_latency_ms in each event = time from frame decode to alert "
                     "being logged; it excludes camera/driver capture delay."),
        }


# --------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Lab-safety monitoring pipeline (TV3 integration)")
    p.add_argument("--source", required=True, help="video file path, or webcam index such as 0")
    p.add_argument("--weights", default="yolo11n.pt", help="COCO model (person + cell phone)")
    p.add_argument("--lab-coat-weights", default=None, help="custom lab-coat model")
    p.add_argument("--pose-weights", default="yolo11n-pose.pt",
                   help="used only if R3.association_mode == pose_hand")
    p.add_argument("--zones", default="configs/zones.yaml")
    p.add_argument("--rules", default="configs/rules.yaml")
    p.add_argument("--out", default=None, help="run folder (default runs/run_YYYYmmdd_HHMMSS)")
    p.add_argument("--imgsz", type=int, default=512)
    p.add_argument("--device", default="cpu")
    p.add_argument("--coco-conf", type=float, default=0.25)
    p.add_argument("--coat-conf", type=float, default=0.25)
    p.add_argument("--pose-conf", type=float, default=0.25)
    p.add_argument("--tracker", choices=("auto", "simple"), default="auto",
                   help="auto = tracking_rules.PersonTracker (ByteTrack); simple = IoU fallback")
    p.add_argument("--max-frames", type=int, default=0, help="0 = whole video")
    p.add_argument("--frame-stride", type=int, default=1, help="process every Nth frame")
    p.add_argument("--realtime", action="store_true", help="play a video file at its native speed")
    p.add_argument("--show", action="store_true", help="OpenCV preview window (Q to quit)")
    p.add_argument("--no-save-video", action="store_true", help="skip pipeline_annotated.mp4")
    p.add_argument("--clip-pre", type=float, default=3.0, help="seconds before the event")
    p.add_argument("--clip-post", type=float, default=2.0, help="seconds after the event")
    p.add_argument("--clip-fps", type=float, default=12.0, help="buffer/clip sampling fps")
    p.add_argument("--clip-width", type=int, default=960)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    from tracking_rules import RuleEngine, load_rules_config, load_zones
    from perception import MultiYoloDetector, PoseEstimator

    cfg = load_rules_config(args.rules)
    zones = load_zones(args.zones)
    source = FrameSource(args.source)
    out_dir = Path(args.out or Path("runs") / f"run_{datetime.now():%Y%m%d_%H%M%S}")
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = MultiYoloDetector(args.weights, args.lab_coat_weights, imgsz=args.imgsz,
                                 device=args.device, coco_conf=args.coco_conf,
                                 lab_coat_conf=args.coat_conf)
    tracker = build_tracker(cfg["tracker"], source.fps / max(1, args.frame_stride), args.tracker)
    r3 = cfg["rules"]["R3"]
    pose = None
    if r3.get("enabled", True) and r3.get("association_mode") == "pose_hand":
        pose = PoseEstimator(args.pose_weights, args.pose_conf, args.imgsz, args.device)
    engine = RuleEngine(zones, cfg, source_id=source.label,
                        source_type="webcam" if source.is_live else "video_file")
    evidence = EvidenceManager(str(out_dir), pre_sec=args.clip_pre, post_sec=args.clip_post,
                               buffer_fps=args.clip_fps, max_width=args.clip_width)
    pipe = SafetyPipeline(source, detector, tracker, engine, evidence, str(out_dir), zones,
                          pose_estimator=pose, show=args.show, frame_stride=args.frame_stride,
                          realtime=args.realtime, save_video=not args.no_save_video)
    signal.signal(signal.SIGINT, pipe.stop)
    print(f"[pipeline] run folder: {out_dir.resolve()}")
    summary = pipe.run(args.max_frames)
    print(f"[pipeline] done: {summary['processed_frames']} frames, "
          f"{summary['events_total']} events, {summary['fps_avg']} FPS avg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
