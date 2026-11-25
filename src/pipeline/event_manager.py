"""Asynchronous evidence manager: snapshot + pre/post-event clip, never blocks inference.

Flow per confirmed event (called from the main loop):
  1. the event is appended to events.jsonl immediately (clip_status=PENDING);
  2. a snapshot job is queued;
  3. the pre-event frames from a ring buffer are attached to a "pending clip",
     which keeps collecting frames for ``post_sec`` more seconds;
  4. once complete, the clip is handed to a background writer thread, which
     encodes the video and appends an "_update" record (clip_status=READY).

If the writer queue is full the clip is dropped (clip_status=DROPPED) rather
than stalling the pipeline.
"""
from __future__ import annotations

import queue
import shutil
import subprocess
import threading
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional, Tuple

import cv2

from .eventlog import EVENTS_FILE, append_record

Frame = Tuple[float, "cv2.Mat"]  # (video time sec, resized BGR frame)


def _resize_to_width(img, max_width: int):
    h, w = img.shape[:2]
    if max_width and w > max_width:
        scale = max_width / float(w)
        return cv2.resize(img, (max_width, max(1, int(round(h * scale)))),
                          interpolation=cv2.INTER_AREA)
    return img.copy()


def _fmt_time(t: float) -> str:
    t = max(0.0, float(t))
    return f"{int(t // 60):02d}:{t % 60:05.2f}"


class _PendingClip:
    def __init__(self, event: dict, frames: List[Frame], end_t: float):
        self.event = event
        self.frames = frames
        self.end_t = end_t


class EvidenceManager:
    def __init__(self, out_dir: str, pre_sec: float = 3.0, post_sec: float = 2.0,
                 buffer_fps: float = 12.0, max_width: int = 960,
                 jpeg_quality: int = 88, queue_size: int = 8,
                 events_file: Optional[str] = None):
        if pre_sec < 0 or post_sec < 0 or buffer_fps <= 0:
            raise ValueError("pre_sec/post_sec must be >= 0 and buffer_fps > 0")
        self.out_dir = Path(out_dir)
        self.evidence_dir = self.out_dir / "evidence"
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = Path(events_file) if events_file else self.out_dir / EVENTS_FILE
        self.pre_sec = float(pre_sec)
        self.post_sec = float(post_sec)
        self.buffer_fps = float(buffer_fps)
        self.max_width = int(max_width)
        self.jpeg_quality = int(jpeg_quality)

        self._ring: Deque[Frame] = deque()
        self._last_kept_t = -1e9
        self._pending: List[_PendingClip] = []
        self._queue: "queue.Queue[Optional[tuple]]" = queue.Queue(maxsize=queue_size)
        self._closed = False
        self.stats = {"events": 0, "clips_ready": 0, "clips_dropped": 0, "clips_failed": 0}
        self._worker = threading.Thread(target=self._run_worker, name="evidence-writer", daemon=True)
        self._worker.start()
        self._ffmpeg = shutil.which("ffmpeg")

    # ------------------------------------------------------------ main thread
    def push_frame(self, t: float, frame) -> None:
        """Offer every processed frame (already annotated). Cheap: sub-samples."""
        if self._closed:
            return
        if t - self._last_kept_t < (1.0 / self.buffer_fps) - 1e-6 and self._last_kept_t > -1e8:
            return
        self._last_kept_t = t
        item = (t, _resize_to_width(frame, self.max_width))
        self._ring.append(item)
        horizon = t - self.pre_sec - 1.0
        while self._ring and self._ring[0][0] < horizon:
            self._ring.popleft()

        still: List[_PendingClip] = []
        for p in self._pending:
            p.frames.append(item)
            if t >= p.end_t:
                self._submit_clip(p)
            else:
                still.append(p)
        self._pending = still

    def on_event(self, event: dict, frame) -> dict:
        """Register a confirmed event. Call after push_frame for the same frame."""
        if self._closed:
            raise RuntimeError("EvidenceManager is closed")
        eid, rid = event["event_id"], event["rule_id"]
        base = f"{eid}_{rid}"
        t = float(event["video_time_sec"])
        image_rel = f"evidence/{base}.jpg"
        clip_rel = f"evidence/{base}.mp4"
        event["evidence"] = {"image_path": image_rel, "clip_path": clip_rel,
                             "clip_status": "PENDING"}
        append_record(self.events_path, event)
        self.stats["events"] += 1

        snap = _resize_to_width(frame, self.max_width)
        self._enqueue(("snapshot", self.out_dir / image_rel, snap, eid))

        pre = [f for f in self._ring if f[0] >= t - self.pre_sec]
        self._pending.append(_PendingClip(event, list(pre), t + self.post_sec))
        return event

    def close(self, timeout: float = 120.0) -> None:
        """Flush pending clips (with whatever frames exist) and stop the worker."""
        if self._closed:
            return
        self._closed = True
        for p in self._pending:
            self._submit_clip(p, blocking=True)
        self._pending = []
        self._queue.put(None)
        self._worker.join(timeout=timeout)

    # -------------------------------------------------------------- internals
    def _enqueue(self, job: tuple) -> bool:
        try:
            self._queue.put_nowait(job)
            return True
        except queue.Full:
            return False

    def _submit_clip(self, p: _PendingClip, blocking: bool = False) -> None:
        ev = p.event
        job = ("clip", self.out_dir / ev["evidence"]["clip_path"], p.frames, ev)
        if blocking:
            self._queue.put(job)
            return
        if not self._enqueue(job):
            self.stats["clips_dropped"] += 1
            append_record(self.events_path, {
                "_update": True, "event_id": ev["event_id"],
                "evidence": {"clip_path": None, "clip_status": "DROPPED"}})

    def _run_worker(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                return
            try:
                if job[0] == "snapshot":
                    _, path, img, _eid = job
                    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                else:
                    self._write_clip(job[1], job[2], job[3])
            except Exception as exc:  # never kill the worker
                if job[0] == "clip":
                    self.stats["clips_failed"] += 1
                    append_record(self.events_path, {
                        "_update": True, "event_id": job[3]["event_id"],
                        "evidence": {"clip_status": "FAILED", "clip_error": str(exc)[:200]}})

    def _write_clip(self, path: Path, frames: List[Frame], ev: dict) -> None:
        if not frames:
            raise RuntimeError("no frames buffered for clip")
        h, w = frames[0][1].shape[:2]
        span = frames[-1][0] - frames[0][0]
        fps = (len(frames) - 1) / span if span > 0 and len(frames) > 1 else self.buffer_fps
        fps = min(30.0, max(2.0, fps))
        banner = f"{ev['rule_id']} {ev['rule_name']} | ID {ev['track_id']} | {ev['event_id']}"

        tmp = path.with_suffix(".tmp.mp4")
        writer, codec = None, None
        for cc in ("avc1", "H264", "mp4v"):
            wr = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*cc), fps, (w, h))
            if wr.isOpened():
                writer, codec = wr, cc
                break
            wr.release()
        if writer is None:
            raise RuntimeError("no usable VideoWriter codec")
        event_t = float(ev["video_time_sec"])
        try:
            for t, img in frames:
                out = img.copy()  # ring-buffer frames are shared between clips
                cv2.rectangle(out, (0, 0), (w, 26), (0, 0, 0), -1)
                cv2.putText(out, f"{banner}  t={_fmt_time(t)}", (8, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                if abs(t - event_t) < 0.5 / fps + 1e-6 or (t >= event_t and t - event_t < 1.0 / fps):
                    cv2.putText(out, "ALERT", (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 0, 255), 2, cv2.LINE_AA)
                writer.write(out)
        finally:
            writer.release()

        final_codec = codec
        if codec == "mp4v" and self._ffmpeg:
            # mp4v is not playable in most browsers; transcode to H.264 when possible.
            h264_tmp = path.with_suffix(".h264.mp4")
            try:
                r = subprocess.run(
                    [self._ffmpeg, "-y", "-loglevel", "error", "-i", str(tmp),
                     "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                     str(h264_tmp)], capture_output=True, timeout=120)
                if r.returncode == 0 and h264_tmp.exists() and h264_tmp.stat().st_size > 0:
                    tmp.unlink()
                    tmp = h264_tmp
                    final_codec = "h264"
            except (OSError, subprocess.SubprocessError):
                pass
            finally:
                if h264_tmp.exists() and tmp != h264_tmp:
                    h264_tmp.unlink()
        tmp.replace(path)
        self.stats["clips_ready"] += 1
        append_record(self.events_path, {
            "_update": True, "event_id": ev["event_id"],
            "evidence": {"clip_status": "READY", "clip_codec": final_codec,
                         "clip_duration_sec": round(frames[-1][0] - frames[0][0], 2),
                         "browser_playable": final_codec in ("avc1", "H264", "h264")}})
