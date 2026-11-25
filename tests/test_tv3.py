"""TV3 tests. Run from the repo root:  python -m unittest discover -s tests -v

They need numpy + opencv (already required by the project) and the TV2 package
`tracking_rules` on the path (src/ or repo root). No YOLO weights are used: the
detector is faked so the tests are fast and deterministic.
"""
from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tracking_rules import RuleEngine, load_rules_config  # noqa: E402
from tracking_rules.schemas import Detection, Zone  # noqa: E402

from dashboard.server import serve  # noqa: E402
from dashboard.stats import compute_stats  # noqa: E402
from evaluation.evaluate import evaluate_run, match_rule  # noqa: E402
from pipeline.event_manager import EvidenceManager  # noqa: E402
from pipeline.eventlog import (append_record, load_reviews, read_events,  # noqa: E402
                               save_review)
from pipeline.runner import SafetyPipeline  # noqa: E402
from pipeline.tracker_adapter import (PersonTrackerAdapter, SimpleIoUTracker,  # noqa: E402
                                      to_tracks)

W, H, FPS = 320, 240, 10.0


class FakeSource:
    """Same interface as runner.FrameSource, no camera needed."""
    is_live = False
    label = "fake.mp4"

    def __init__(self, n_frames: int, fps: float = FPS):
        self.fps, self.total_frames, self.n, self.i = fps, n_frames, n_frames, 0

    def read(self):
        if self.i >= self.n:
            return False, None, 0.0, self.i
        frame = np.full((H, W, 3), 60, np.uint8)
        cv2.putText(frame, str(self.i), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        idx, self.i = self.i, self.i + 1
        return True, frame, idx / self.fps, idx

    def release(self):
        pass


class FakeDetector:
    """A person standing at the bottom-right, i.e. inside the restricted zone."""

    def __init__(self, bbox=(200, 120, 260, 230), conf=0.9):
        self.bbox, self.conf = bbox, conf

    def __call__(self, frame, idx):
        return [Detection("person", self.conf, self.bbox)]


def restricted_cfg(min_dur=1.0):
    return {"common": {"person_conf_min": 0.5, "min_person_height_ratio": 0.1, "event_id_db": None},
            "rules": {"R1": {"enabled": False}, "R3": {"enabled": False},
                      "R2": {"enabled": True, "min_duration_sec": min_dur,
                             "grace_period_sec": 0.3, "cooldown_sec": 30.0}}}


ZONES = [Zone("lab_floor", "general", [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]),
         Zone("forbidden", "restricted", [(0.5, 0.4), (1.0, 0.4), (1.0, 1.0), (0.5, 1.0)])]


class EventLogTests(unittest.TestCase):
    def test_merge_update_and_partial_line(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "events.jsonl"
            append_record(p, {"event_id": "E1", "rule_id": "R2", "evidence": {"image_path": "a.jpg", "clip_status": "PENDING"}})
            append_record(p, {"event_id": "E2", "rule_id": "R1", "evidence": None})
            append_record(p, {"_update": True, "event_id": "E1", "evidence": {"clip_status": "READY"}})
            append_record(p, {"_update": True, "event_id": "GHOST", "evidence": {"clip_status": "READY"}})
            with open(p, "a", encoding="utf-8") as f:
                f.write('{"event_id": "E3", "rule')  # truncated write
            ev = read_events(p)
            self.assertEqual([e["event_id"] for e in ev], ["E1", "E2"])
            self.assertEqual(ev[0]["evidence"], {"image_path": "a.jpg", "clip_status": "READY"})
            self.assertIsInstance(ev[1]["evidence"], dict)  # legacy None normalised

    def test_reviews_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            save_review(d, "E1", "TP")
            save_review(d, "E2", "FP", "reflection")
            self.assertEqual(load_reviews(d)["E2"]["label"], "FP")
            save_review(d, "E1", None)
            self.assertNotIn("E1", load_reviews(d))
            with self.assertRaises(ValueError):
                save_review(d, "E1", "MAYBE")


class TrackerAdapterTests(unittest.TestCase):
    def test_signature_discovery(self):
        seen = {}

        class T:
            def __init__(self, lost_buffer_sec=1.0, frame_rate=30, unknown=5):
                seen.update(lost=lost_buffer_sec, fr=frame_rate, unk=unknown)

            def update(self, detections, frame_idx, frame):
                seen["frame_shape"] = frame.shape
                return [{"track_id": 3, "class": "person", "confidence": 0.8, "bbox": [1, 2, 3, 4]}]

        a = PersonTrackerAdapter(T, {"lost_buffer_sec": 2.5, "unrelated": 1}, fps=12.0)
        out = a.update([], 4, 0.4, np.zeros((5, 6, 3), np.uint8), (6, 5))
        self.assertEqual((seen["lost"], seen["fr"], seen["unk"]), (2.5, 12.0, 5))
        self.assertEqual(seen["frame_shape"], (5, 6, 3))
        self.assertEqual(out[0].track_id, 3)

    def test_to_tracks_rejects_garbage(self):
        with self.assertRaises(TypeError):
            to_tracks([42])

    def test_simple_tracker_keeps_id(self):
        t = SimpleIoUTracker()
        a = t.update([Detection("person", .9, (10, 10, 50, 100))], 0, 0, None, (W, H))
        b = t.update([Detection("person", .9, (12, 11, 52, 101))], 1, .1, None, (W, H))
        self.assertEqual(a[0].track_id, b[0].track_id)


class EvidenceTests(unittest.TestCase):
    def test_snapshot_clip_and_nonblocking(self):
        with tempfile.TemporaryDirectory() as d:
            em = EvidenceManager(d, pre_sec=1.0, post_sec=1.0, buffer_fps=10, max_width=200)
            ev = None
            slowest = 0.0
            for i in range(60):
                t = i / 10.0
                frame = np.full((H, W, 3), i * 3 % 255, np.uint8)
                t0 = time.perf_counter()
                em.push_frame(t, frame)
                if i == 25:
                    ev = {"event_id": "EVT_X_0001", "rule_id": "R2", "rule_name": "RESTRICTED_ZONE",
                          "track_id": 7, "video_time_sec": t}
                    em.on_event(ev, frame)
                slowest = max(slowest, time.perf_counter() - t0)
            em.close()
            self.assertLess(slowest, 0.25, "evidence hand-off must not block the pipeline")
            self.assertTrue((Path(d) / "evidence/EVT_X_0001_R2.jpg").exists())
            clip = Path(d) / "evidence/EVT_X_0001_R2.mp4"
            self.assertTrue(clip.exists())
            cap = cv2.VideoCapture(str(clip))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            self.assertGreaterEqual(n, 15)  # ~1 s before + ~1 s after at 10 fps
            merged = read_events(Path(d) / "events.jsonl")[0]
            self.assertEqual(merged["evidence"]["clip_status"], "READY")
            self.assertEqual(em.stats["clips_ready"], 1)

    def test_close_flushes_incomplete_clip(self):
        with tempfile.TemporaryDirectory() as d:
            em = EvidenceManager(d, pre_sec=1.0, post_sec=5.0, buffer_fps=10)
            for i in range(10):
                em.push_frame(i / 10, np.zeros((H, W, 3), np.uint8))
            em.on_event({"event_id": "E", "rule_id": "R1", "rule_name": "NO_LAB_COAT",
                         "track_id": 1, "video_time_sec": 0.9}, np.zeros((H, W, 3), np.uint8))
            em.close()
            self.assertTrue((Path(d) / "evidence/E_R1.mp4").exists())


class PipelineE2ETests(unittest.TestCase):
    def test_restricted_zone_event_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = restricted_cfg(min_dur=1.0)
            cfg["common"]["event_id_db"] = str(Path(d) / "seq.sqlite3")
            engine = RuleEngine(ZONES, cfg, event_id_db=str(Path(d) / "seq.sqlite3"))
            ev_mgr = EvidenceManager(d, pre_sec=1.0, post_sec=1.0, buffer_fps=10)
            pipe = SafetyPipeline(FakeSource(60), FakeDetector(), SimpleIoUTracker(), engine,
                                  ev_mgr, d, ZONES, live_every_sec=0.0, status_every_sec=0.0)
            summary = pipe.run()
            self.assertEqual(summary["events_total"], 1)  # one alert per intrusion (cooldown)
            self.assertEqual(summary["events_by_rule"]["R2"], 1)
            self.assertAlmostEqual(summary["analyzed_duration_sec"], 6.0, places=1)

            events = read_events(Path(d) / "events.jsonl")
            self.assertEqual(len(events), 1)
            e = events[0]
            self.assertEqual(e["rule_id"], "R2")
            self.assertGreaterEqual(e["duration_sec"], 1.0)
            self.assertIn("pipeline_latency_ms", e["details"])
            self.assertEqual(e["evidence"]["clip_status"], "READY")
            self.assertTrue((Path(d) / e["evidence"]["image_path"]).exists())
            self.assertTrue((Path(d) / e["evidence"]["clip_path"]).exists())
            status = json.loads((Path(d) / "status.json").read_text())
            self.assertEqual(status["state"], "finished")
            self.assertTrue((Path(d) / "live.jpg").exists())

    def test_person_outside_zone_no_event(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = restricted_cfg()
            engine = RuleEngine(ZONES, cfg, event_id_db=str(Path(d) / "seq.sqlite3"))
            pipe = SafetyPipeline(FakeSource(40), FakeDetector((10, 100, 60, 200)), SimpleIoUTracker(),
                                  engine, EvidenceManager(d), d, ZONES, save_video=False)
            self.assertEqual(pipe.run()["events_total"], 0)

    def test_error_state_is_reported(self):
        with tempfile.TemporaryDirectory() as d:
            def boom(frame, idx):
                raise RuntimeError("detector exploded")
            engine = RuleEngine(ZONES, restricted_cfg(), event_id_db=str(Path(d) / "s.sqlite3"))
            pipe = SafetyPipeline(FakeSource(5), boom, SimpleIoUTracker(), engine,
                                  EvidenceManager(d), d, ZONES, save_video=False)
            with self.assertRaises(RuntimeError):
                pipe.run()
            self.assertEqual(json.loads((Path(d) / "status.json").read_text())["state"], "error")


class DashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.runs = Path(cls.tmp.name) / "runs"
        run = cls.runs / "demo"
        cfg = restricted_cfg()
        engine = RuleEngine(ZONES, cfg, event_id_db=str(Path(cls.tmp.name) / "s.sqlite3"))
        SafetyPipeline(FakeSource(50), FakeDetector(), SimpleIoUTracker(), engine,
                       EvidenceManager(str(run), pre_sec=1, post_sec=1, buffer_fps=10), str(run), ZONES,
                       save_video=False).run()
        (cls.runs / "secret.txt").write_text("top secret")
        cls.httpd = serve(str(cls.runs), "127.0.0.1", 0)
        cls.port = cls.httpd.server_address[1]
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.port}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.tmp.cleanup()

    def get(self, path, headers=None):
        req = urllib.request.Request(self.base + path, headers=headers or {})
        return urllib.request.urlopen(req, timeout=5)

    def test_index_runs_state(self):
        self.assertIn(b"Gi\xc3\xa1m s\xc3\xa1t", self.get("/").read())
        self.assertEqual(json.loads(self.get("/api/runs").read())["runs"][0]["name"], "demo")
        st = json.loads(self.get("/api/state?run=demo").read())
        self.assertEqual(st["stats"]["total"], 1)
        self.assertTrue(st["has_live"])
        self.assertGreater(st["stats"]["alerts_per_hour"], 0)

    def test_video_range_request(self):
        st = json.loads(self.get("/api/state?run=demo").read())
        clip = st["events"][0]["evidence"]["clip_path"]
        r = self.get(f"/evidence/demo/{clip}", {"Range": "bytes=0-99"})
        self.assertEqual(r.status, 206)
        self.assertEqual(len(r.read()), 100)
        self.assertEqual(r.headers["Content-Range"].split("/")[0], "bytes 0-99")

    def test_path_traversal_blocked(self):
        for bad in ("/evidence/demo/../secret.txt", "/evidence/demo/%2e%2e/secret.txt",
                    "/evidence/..%2Fsecret.txt/x", "/evidence/demo/events.jsonl",
                    "/api/state?run=../runs", "/api/state?run=..%2F"):
            try:
                body = self.get(bad).read()
                self.assertNotIn(b"top secret", body, bad)
            except urllib.error.HTTPError as e:
                self.assertIn(e.code, (400, 404), bad)

    def test_review_requires_json_and_valid_event(self):
        st = json.loads(self.get("/api/state?run=demo").read())
        eid = st["events"][0]["event_id"]

        def post(body, ctype="application/json"):
            req = urllib.request.Request(self.base + "/api/review", data=json.dumps(body).encode(),
                                         headers={"Content-Type": ctype}, method="POST")
            return urllib.request.urlopen(req, timeout=5)

        with self.assertRaises(urllib.error.HTTPError) as c:
            post({"run": "demo", "event_id": eid, "label": "TP"}, "text/plain")
        self.assertEqual(c.exception.code, 415)
        with self.assertRaises(urllib.error.HTTPError) as c:
            post({"run": "demo", "event_id": "NOPE", "label": "TP"})
        self.assertEqual(c.exception.code, 404)
        self.assertEqual(post({"run": "demo", "event_id": eid, "label": "FP"}).status, 200)
        st = json.loads(self.get("/api/state?run=demo").read())
        self.assertEqual(st["stats"]["review"]["fp"], 1)
        self.assertEqual(st["events"][0]["review"]["label"], "FP")

    def test_csv_export(self):
        body = self.get("/api/export.csv?run=demo").read().decode("utf-8-sig")
        self.assertTrue(body.startswith("event_id,rule_id"))
        self.assertIn("R2", body)


class StatsAndEvalTests(unittest.TestCase):
    def test_stats_timeline_and_rate(self):
        events = [{"event_id": "a", "rule_id": "R1", "severity": "medium", "video_time_sec": 3, "track_id": 1},
                  {"event_id": "b", "rule_id": "R2", "severity": "high", "video_time_sec": 70, "track_id": 2}]
        s = compute_stats(events, None, {"analyzed_duration_sec": 120.0}, {"a": {"label": "TP"}})
        self.assertEqual(s["alerts_per_hour"], 60.0)
        self.assertEqual(s["review"]["precision"], 1.0)
        self.assertEqual(sum(b["R1"] + b["R2"] for b in s["timeline"]["buckets"]), 2)

    def test_matching_is_one_to_one_and_duplicates_are_fp(self):
        alerts = [{"event_id": "1", "video_time_sec": 12.0, "violation_start_sec": 10.5},
                  {"event_id": "2", "video_time_sec": 14.0, "violation_start_sec": 13.0},   # duplicate
                  {"event_id": "3", "video_time_sec": 80.0, "violation_start_sec": 78.0}]  # false alarm
        gts = [{"rule_id": "R2", "start_sec": 10.0, "end_sec": 15.0},
               {"rule_id": "R2", "start_sec": 40.0, "end_sec": 45.0}]                      # missed
        pairs, un_a, un_g, dup = match_rule(alerts, gts, tolerance=1.0)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(sorted(un_a), [1, 2])
        self.assertEqual(un_g, [1])
        self.assertEqual(dup, [1])

    def test_evaluate_run_metrics(self):
        with tempfile.TemporaryDirectory() as d:
            for i, (t0, t1) in enumerate([(10.5, 12.0), (80.0, 82.0)]):
                append_record(Path(d) / "events.jsonl",
                              {"event_id": f"E{i}", "rule_id": "R2", "violation_start_sec": t0,
                               "video_time_sec": t1, "evidence": {}})
            gt = [{"rule_id": "R2", "start_sec": 10.0, "end_sec": 15.0},
                  {"rule_id": "R2", "start_sec": 40.0, "end_sec": 45.0}]
            r = evaluate_run(d, gt, 1.0, 3600.0)
            r2 = r["rules"]["R2"]
            self.assertEqual((r2["TP"], r2["FP"], r2["FN"]), (1, 1, 1))
            self.assertEqual((r2["precision"], r2["recall"]), (0.5, 0.5))
            self.assertEqual(r2["false_alerts_per_hour"], 1.0)
            self.assertIsNone(r["rules"]["R1"]["precision"])  # no alerts and no GT: undefined, not 0/1
            r_nodur = evaluate_run(d, gt, 1.0, None)
            self.assertIsNone(r_nodur["rules"]["R2"]["false_alerts_per_hour"])  # never guesses duration


if __name__ == "__main__":
    unittest.main(verbosity=2)
