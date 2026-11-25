"""Aggregate statistics shown on the dashboard (pure functions, unit-tested)."""
from __future__ import annotations

import math
from typing import Dict, List, Optional

RULES = ("R1", "R2", "R3")


def _nice_bucket(duration: float, max_buckets: int = 48) -> int:
    for b in (5, 10, 15, 30, 60, 120, 300, 600, 1800, 3600):
        if duration / b <= max_buckets:
            return b
    return 3600


def analysed_duration(status: Optional[dict], summary: Optional[dict],
                      events: List[dict]) -> Optional[float]:
    if summary and summary.get("analyzed_duration_sec"):
        return float(summary["analyzed_duration_sec"])
    if status and status.get("video_time_sec"):
        return float(status["video_time_sec"])
    if events:
        return max(float(e.get("video_time_sec", 0.0)) for e in events) or None
    return None


def compute_stats(events: List[dict], status: Optional[dict], summary: Optional[dict],
                  reviews: Dict[str, dict]) -> dict:
    by_rule = {r: 0 for r in RULES}
    by_sev: Dict[str, int] = {}
    tracks = set()
    for e in events:
        by_rule[e.get("rule_id", "?")] = by_rule.get(e.get("rule_id", "?"), 0) + 1
        by_sev[e.get("severity", "?")] = by_sev.get(e.get("severity", "?"), 0) + 1
        if e.get("track_id") is not None:
            tracks.add(e["track_id"])

    duration = analysed_duration(status, summary, events)
    per_hour = None
    if duration and duration > 0:
        per_hour = round(len(events) / (duration / 3600.0), 1)

    tp = fp = 0
    for e in events:
        r = reviews.get(e.get("event_id"))
        if r:
            tp += r["label"] == "TP"
            fp += r["label"] == "FP"
    reviewed = tp + fp
    review = {
        "reviewed": reviewed, "tp": tp, "fp": fp,
        "unreviewed": len(events) - reviewed,
        "precision": round(tp / reviewed, 3) if reviewed else None,
        "fp_per_hour": round(fp / (duration / 3600.0), 2) if (duration and fp is not None) else None,
    }

    timeline = {"bucket_sec": None, "duration_sec": duration, "buckets": []}
    if duration and duration > 0:
        b = _nice_bucket(duration)
        n = max(1, int(math.ceil(duration / b)))
        buckets = [{"start": i * b, **{r: 0 for r in RULES}} for i in range(n)]
        for e in events:
            i = min(n - 1, max(0, int(float(e.get("video_time_sec", 0.0)) // b)))
            if e.get("rule_id") in RULES:
                buckets[i][e["rule_id"]] += 1
        timeline = {"bucket_sec": b, "duration_sec": round(duration, 2), "buckets": buckets}

    return {
        "total": len(events), "by_rule": by_rule, "by_severity": by_sev,
        "distinct_tracks": len(tracks), "duration_sec": duration,
        "alerts_per_hour": per_hour, "review": review, "timeline": timeline,
    }
