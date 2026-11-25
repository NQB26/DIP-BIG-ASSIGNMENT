#!/usr/bin/env python
"""Alert dashboard server (Python standard library only – no pip install needed).

    python src\\dashboard\\server.py --runs-dir .\\runs --port 8000
    then open http://127.0.0.1:8000

It reads each run folder written by pipeline/runner.py (events.jsonl,
status.json, live.jpg, evidence/…) and also any events.jsonl produced by
scripts/test_tracker.py, so older validation runs can be browsed too.

Binds to 127.0.0.1 by default: there is no authentication.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, unquote, urlparse

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dashboard.stats import compute_stats  # noqa: E402
from pipeline.eventlog import (EVENTS_FILE, LIVE_FILE, STATUS_FILE, SUMMARY_FILE,  # noqa: E402
                               load_reviews, read_events, read_json, save_review)

STATIC_DIR = Path(__file__).resolve().parent / "static"
RUN_NAME_RE = re.compile(r"^[\w.\- ]{1,120}$")
mimetypes.add_type("video/mp4", ".mp4")


class DashboardApp:
    def __init__(self, runs_dir: str, default_run: Optional[str] = None):
        self.runs_dir = Path(runs_dir).resolve()
        self.default_run = default_run
        self._cache = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------- runs
    def list_runs(self):
        runs = []
        if not self.runs_dir.is_dir():
            return runs
        for d in self.runs_dir.iterdir():
            if not d.is_dir():
                continue
            ev, st = d / EVENTS_FILE, d / STATUS_FILE
            if not (ev.exists() or st.exists()):
                continue
            mt = max(p.stat().st_mtime for p in (ev, st) if p.exists())
            status = read_json(st) or {}
            runs.append({"name": d.name, "mtime": mt, "state": status.get("state"),
                         "has_pipeline_status": st.exists()})
        runs.sort(key=lambda r: r["mtime"], reverse=True)
        return runs

    def resolve_run(self, name: Optional[str]) -> Optional[Path]:
        name = name or self.default_run
        if not name:
            runs = self.list_runs()
            name = runs[0]["name"] if runs else None
        if not name or not RUN_NAME_RE.match(name) or name in (".", ".."):
            return None
        d = (self.runs_dir / name).resolve()
        if d.parent != self.runs_dir or not d.is_dir():
            return None
        return d

    # ------------------------------------------------------------ state
    def _events_cached(self, run_dir: Path):
        p = run_dir / EVENTS_FILE
        try:
            st = p.stat()
            key = (st.st_mtime_ns, st.st_size)
        except OSError:
            return []
        with self._lock:
            hit = self._cache.get(str(p))
            if hit and hit[0] == key:
                return hit[1]
        events = read_events(p)
        with self._lock:
            self._cache[str(p)] = (key, events)
        return events

    def state(self, run_dir: Path) -> dict:
        events = self._events_cached(run_dir)
        status = read_json(run_dir / STATUS_FILE)
        summary = read_json(run_dir / SUMMARY_FILE)
        reviews = load_reviews(run_dir)
        stats = compute_stats(events, status, summary, reviews)
        out_events = []
        for e in reversed(events):  # newest first
            item = dict(e)
            item["review"] = reviews.get(e.get("event_id"))
            out_events.append(item)
        return {
            "run": run_dir.name, "status": status, "summary": summary, "stats": stats,
            "events": out_events,
            "has_live": (run_dir / LIVE_FILE).exists(),
            "extra_videos": [n for n in ("pipeline_annotated.mp4", "annotated.mp4")
                             if (run_dir / n).exists()],
        }

    def safe_file(self, run_dir: Path, rel: str) -> Optional[Path]:
        """Resolve ``rel`` strictly inside the run folder (blocks ../ and symlinks out)."""
        try:
            p = (run_dir / rel).resolve()
            p.relative_to(run_dir)
        except (ValueError, OSError):
            return None
        return p if p.is_file() else None


def make_handler(app: DashboardApp):
    class Handler(BaseHTTPRequestHandler):
        server_version = "SafetyDashboard/1.0"

        def log_message(self, fmt, *args):  # keep the console quiet
            pass

        # ------------------------------------------------------ helpers
        def _json(self, obj, code=200):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _error(self, code, msg):
            self._json({"error": msg}, code)

        def _send_file(self, path: Path, cache="no-store"):
            size = path.stat().st_size
            ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            start, end, code = 0, size - 1, 200
            rng = self.headers.get("Range")
            if rng:
                m = re.match(r"bytes=(\d*)-(\d*)$", rng.strip())
                if m and (m.group(1) or m.group(2)):
                    if m.group(1):
                        start = int(m.group(1))
                        end = int(m.group(2)) if m.group(2) else size - 1
                    else:  # suffix range: last N bytes
                        start = max(0, size - int(m.group(2)))
                    end = min(end, size - 1)
                    if start > end or start >= size:
                        self.send_response(416)
                        self.send_header("Content-Range", f"bytes */{size}")
                        self.end_headers()
                        return
                    code = 206
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", cache)
            self.send_header("Content-Length", str(end - start + 1))
            if code == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            try:
                with open(path, "rb") as f:
                    f.seek(start)
                    remaining = end - start + 1
                    while remaining > 0:
                        chunk = f.read(min(64 * 1024, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass  # browser closed the stream (normal for video seeking)

        # --------------------------------------------------------- GET
        def do_GET(self):
            url = urlparse(self.path)
            q = parse_qs(url.query)
            path = unquote(url.path)
            run_q = (q.get("run") or [None])[0]

            if path in ("/", "/index.html"):
                return self._send_file(STATIC_DIR / "index.html")
            if path == "/api/runs":
                return self._json({"runs": app.list_runs(), "default": app.default_run})
            if path == "/api/state":
                d = app.resolve_run(run_q)
                if d is None:
                    return self._json({"run": None, "events": [], "stats": None,
                                       "status": None, "summary": None,
                                       "has_live": False, "extra_videos": []})
                return self._json(app.state(d))
            if path == "/live.jpg":
                d = app.resolve_run(run_q)
                f = app.safe_file(d, LIVE_FILE) if d else None
                return self._send_file(f) if f else self._error(404, "no live frame")
            if path == "/api/export.csv":
                d = app.resolve_run(run_q)
                if d is None:
                    return self._error(404, "run not found")
                return self._export_csv(d)
            if path.startswith("/evidence/"):
                parts = path[len("/evidence/"):].split("/", 1)
                if len(parts) != 2:
                    return self._error(404, "not found")
                d = app.resolve_run(parts[0])
                f = app.safe_file(d, parts[1]) if d else None
                if f is None or f.suffix.lower() not in (".jpg", ".jpeg", ".png", ".mp4"):
                    return self._error(404, "not found")
                return self._send_file(f)
            if path.startswith("/file/"):  # annotated videos of a run
                parts = path[len("/file/"):].split("/", 1)
                d = app.resolve_run(parts[0]) if len(parts) == 2 else None
                f = app.safe_file(d, parts[1]) if d else None
                if f is None or f.name not in ("pipeline_annotated.mp4", "annotated.mp4"):
                    return self._error(404, "not found")
                return self._send_file(f)
            return self._error(404, "not found")

        def _export_csv(self, d: Path):
            import csv
            import io
            events = app._events_cached(d)
            reviews = load_reviews(d)
            buf = io.StringIO()
            w = csv.writer(buf)
            w.writerow(["event_id", "rule_id", "rule_name", "severity", "timestamp",
                        "video_time_sec", "duration_sec", "track_id", "zone_id", "confidence",
                        "image_path", "clip_path", "clip_status", "review", "latency_ms"])
            for e in events:
                ev = e.get("evidence") or {}
                w.writerow([e.get("event_id"), e.get("rule_id"), e.get("rule_name"),
                            e.get("severity"), e.get("timestamp"), e.get("video_time_sec"),
                            e.get("duration_sec"), e.get("track_id"), e.get("zone_id"),
                            e.get("confidence"), ev.get("image_path"), ev.get("clip_path"),
                            ev.get("clip_status"),
                            (reviews.get(e.get("event_id")) or {}).get("label", ""),
                            (e.get("details") or {}).get("pipeline_latency_ms")])
            body = ("﻿" + buf.getvalue()).encode("utf-8")  # BOM so Excel reads UTF-8
            self.send_response(200)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header("Content-Disposition", f'attachment; filename="{d.name}_events.csv"')
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        # -------------------------------------------------------- POST
        def do_POST(self):
            url = urlparse(self.path)
            if url.path != "/api/review":
                return self._error(404, "not found")
            # Requiring JSON forces a CORS preflight, so other websites can't POST here.
            if (self.headers.get("Content-Type") or "").split(";")[0].strip() != "application/json":
                return self._error(415, "Content-Type must be application/json")
            try:
                n = int(self.headers.get("Content-Length") or 0)
                if not 0 < n <= 10_000:
                    raise ValueError
                data = json.loads(self.rfile.read(n).decode("utf-8"))
            except (ValueError, json.JSONDecodeError):
                return self._error(400, "invalid JSON body")
            d = app.resolve_run(data.get("run"))
            eid, label = data.get("event_id"), data.get("label")
            if d is None:
                return self._error(404, "run not found")
            if not isinstance(eid, str) or label not in ("TP", "FP", None):
                return self._error(400, "event_id (str) and label (TP/FP/null) required")
            if eid not in {e["event_id"] for e in app._events_cached(d)}:
                return self._error(404, "unknown event_id")
            try:
                save_review(d, eid, label, data.get("note", ""))
            except OSError as exc:
                return self._error(500, str(exc))
            return self._json({"ok": True})

    return Handler


def serve(runs_dir: str, host: str = "127.0.0.1", port: int = 8000,
          default_run: Optional[str] = None) -> ThreadingHTTPServer:
    app = DashboardApp(runs_dir, default_run)
    httpd = ThreadingHTTPServer((host, port), make_handler(app))
    httpd.daemon_threads = True
    return httpd


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Lab-safety alert dashboard")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--run", default=None, help="run folder name to open by default")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args(argv)
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print("WARNING: dashboard has no authentication; exposing it on a network "
              "shows evidence images/clips to anyone who can reach this port.")
    httpd = serve(args.runs_dir, args.host, args.port, args.run)
    print(f"Dashboard: http://{args.host}:{args.port}   (runs dir: {Path(args.runs_dir).resolve()})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
