"""Light-weight per-stage timing and FPS tracking for the pipeline."""
from __future__ import annotations

import time
from collections import deque
from typing import Dict


def _pct(sorted_vals, q: float) -> float:
    if not sorted_vals:
        return 0.0
    i = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[i]


class PipelineMetrics:
    """Rolling window (default 300 frames) of per-stage latencies in ms."""

    STAGES = ("detect", "track", "pose", "rules", "draw", "total")

    def __init__(self, window: int = 300):
        self._stage: Dict[str, deque] = {s: deque(maxlen=window) for s in self.STAGES}
        self._frame_times: deque = deque(maxlen=window)
        self.start = time.perf_counter()
        self.frames = 0

    def record(self, stage: str, seconds: float) -> None:
        self._stage.setdefault(stage, deque(maxlen=300)).append(seconds * 1000.0)

    def frame_done(self) -> None:
        self.frames += 1
        self._frame_times.append(time.perf_counter())

    @property
    def fps_recent(self) -> float:
        if len(self._frame_times) < 2:
            return 0.0
        span = self._frame_times[-1] - self._frame_times[0]
        return (len(self._frame_times) - 1) / span if span > 0 else 0.0

    @property
    def fps_avg(self) -> float:
        elapsed = time.perf_counter() - self.start
        return self.frames / elapsed if elapsed > 0 else 0.0

    def summary(self) -> dict:
        out = {}
        for name, vals in self._stage.items():
            if not vals:
                continue
            s = sorted(vals)
            out[name] = {"mean_ms": round(sum(s) / len(s), 2),
                         "p50_ms": round(_pct(s, 0.5), 2),
                         "p95_ms": round(_pct(s, 0.95), 2)}
        return out
