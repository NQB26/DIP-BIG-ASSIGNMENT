"""Append-only event log (events.jsonl) + human review labels (reviews.json).

events.jsonl is append-only so the pipeline never rewrites a file the dashboard
is reading. A confirmed event is written immediately; when its clip finishes
(asynchronously) an "_update" record with the same event_id is appended. Readers
merge records by event_id with ``read_events``.

The same reader also loads events.jsonl produced by scripts/test_tracker.py,
which has no "_update" records and usually has no evidence paths.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

EVENTS_FILE = "events.jsonl"
REVIEWS_FILE = "reviews.json"
STATUS_FILE = "status.json"
SUMMARY_FILE = "pipeline_summary.json"
LIVE_FILE = "live.jpg"

_write_lock = threading.Lock()


def append_record(path: str | os.PathLike, record: dict) -> None:
    """Append one JSON line (thread-safe within a process)."""
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with _write_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()


def read_events(path: str | os.PathLike) -> List[dict]:
    """Read and merge events.jsonl. Tolerates a truncated last line and BOM."""
    p = Path(path)
    if not p.exists():
        return []
    events: Dict[str, dict] = {}
    order: List[str] = []
    with open(p, "r", encoding="utf-8-sig", errors="replace") as f:
        for n, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # partially written line
            if not isinstance(rec, dict):
                continue
            eid = rec.get("event_id")
            if not eid:
                eid = f"_line{n}"
                rec["event_id"] = eid
            if rec.get("_update"):
                base = events.get(eid)
                if base is None:
                    continue
                patch = {k: v for k, v in rec.items() if k != "_update"}
                ev_patch = patch.pop("evidence", None)
                base.update(patch)
                if isinstance(ev_patch, dict):
                    base["evidence"] = {**(base.get("evidence") or {}), **ev_patch}
            else:
                if eid not in events:
                    order.append(eid)
                events[eid] = rec
    out = []
    for eid in order:
        ev = events[eid]
        if not isinstance(ev.get("evidence"), dict):
            ev["evidence"] = {"image_path": None, "clip_path": None}
        out.append(ev)
    return out


def read_json(path: str | os.PathLike) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def write_json_atomic(path: str | os.PathLike, data: dict) -> bool:
    """Write JSON via temp file + replace. Returns False if the target is locked
    (e.g. a Windows reader has it open); callers simply retry on the next tick."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".{os.getpid()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, p)
        return True
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass
        return False


def load_reviews(run_dir: str | os.PathLike) -> Dict[str, dict]:
    data = read_json(Path(run_dir) / REVIEWS_FILE) or {}
    return {k: v for k, v in data.items()
            if isinstance(v, dict) and v.get("label") in ("TP", "FP")}


def save_review(run_dir: str | os.PathLike, event_id: str, label: Optional[str],
                note: str = "") -> Dict[str, dict]:
    """Set (TP/FP) or clear (None) a human review label for one event."""
    if label not in (None, "TP", "FP"):
        raise ValueError("label must be TP, FP or None")
    with _write_lock:
        reviews = load_reviews(run_dir)
        if label is None:
            reviews.pop(event_id, None)
        else:
            reviews[event_id] = {"label": label, "note": str(note)[:500],
                                 "reviewed_at": datetime.now().isoformat(timespec="seconds")}
        if not write_json_atomic(Path(run_dir) / REVIEWS_FILE, reviews):
            raise OSError("could not write reviews.json")
    return reviews
