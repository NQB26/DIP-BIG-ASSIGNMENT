#!/usr/bin/env python
"""Precision / Recall / false-alerts-per-hour from run folders.

Two label sources (use either or both):
  * --gt  ground-truth file (JSON or CSV) listing the real violation intervals;
          gives Precision AND Recall.
  * reviews.json written by the dashboard (TP/FP per alert); gives Precision
    over the reviewed alerts only. Recall needs ground truth.

Compare several runs side by side (e.g. before/after temporal filter,
baseline vs fine-tuned model):

    python src\\evaluation\\evaluate.py ^
        --run no_debounce=runs\\r2_nodebounce --run debounce=runs\\r2_validation_v3 ^
        --gt data\\test_ground_truth\\zone_test.json --tolerance 2.0 --out reports

Ground-truth JSON:
    {"video_duration_sec": 600,
     "events": [{"rule_id": "R2", "start_sec": 12.0, "end_sec": 20.5}, ...]}
Ground-truth CSV columns: rule_id,start_sec,end_sec  (duration via --duration-sec).

Matching rule (agree this with the team BEFORE computing metrics, see plan §5):
an alert covers the interval [violation_start_sec, video_time_sec]; it matches a
ground-truth interval of the same rule when the two intervals overlap after
widening the ground truth by ±tolerance seconds. Matching is one-to-one, so a
second alert for the same real violation counts as a false positive (duplicate).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pipeline.eventlog import (EVENTS_FILE, SUMMARY_FILE, load_reviews, read_events,  # noqa: E402
                               read_json)

RULES = ("R1", "R2", "R3")


# ----------------------------------------------------------------- loading
def load_ground_truth(path: str) -> Tuple[List[dict], Optional[float]]:
    p = Path(path)
    duration = None
    if p.suffix.lower() == ".csv":
        with open(p, "r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    else:
        with open(p, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        if isinstance(data, list):
            rows = data
        else:
            rows = data.get("events", [])
            duration = data.get("video_duration_sec")
    gt = []
    for i, r in enumerate(rows):
        try:
            rid = str(r["rule_id"]).strip().upper()
            s, e = float(r["start_sec"]), float(r["end_sec"])
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError(f"ground truth row {i} invalid: {r!r}") from exc
        if rid not in RULES:
            raise ValueError(f"ground truth row {i}: unknown rule_id {rid!r}")
        if e < s:
            raise ValueError(f"ground truth row {i}: end_sec < start_sec")
        gt.append({"rule_id": rid, "start_sec": s, "end_sec": e})
    return gt, (float(duration) if duration else None)


def alert_interval(ev: dict) -> Tuple[float, float]:
    end = float(ev.get("video_time_sec", 0.0))
    start = float(ev.get("violation_start_sec", end))
    return min(start, end), end


# ---------------------------------------------------------------- matching
def match_rule(alerts: List[dict], gts: List[dict], tolerance: float):
    """One-to-one greedy matching, earliest alert first.

    Returns (matched pairs [(alert_idx, gt_idx)], unmatched alert idx, unmatched gt idx,
             duplicate alert idx = unmatched alerts that overlap an already-matched GT).
    """
    cands = []
    for ai, a in enumerate(alerts):
        a0, a1 = alert_interval(a)
        for gi, g in enumerate(gts):
            if a0 <= g["end_sec"] + tolerance and a1 >= g["start_sec"] - tolerance:
                dist = abs((a0 + a1) / 2 - (g["start_sec"] + g["end_sec"]) / 2)
                cands.append((a1, dist, ai, gi))
    # Earliest alert wins its violation (the first detection is the TP; later alerts
    # for the same violation are duplicates); ties go to the nearest ground truth.
    cands.sort()
    used_a, used_g, pairs = set(), set(), []
    for _, _, ai, gi in cands:
        if ai in used_a or gi in used_g:
            continue
        used_a.add(ai)
        used_g.add(gi)
        pairs.append((ai, gi))
    un_a = [i for i in range(len(alerts)) if i not in used_a]
    un_g = [i for i in range(len(gts)) if i not in used_g]
    overlap_with_matched = {ai for _, _, ai, gi in cands if gi in used_g and ai not in used_a}
    return pairs, un_a, un_g, sorted(overlap_with_matched)


def _safe_div(a, b):
    return round(a / b, 4) if b else None


def f1(p, r):
    return round(2 * p * r / (p + r), 4) if p and r else (0.0 if p is not None and r is not None else None)


def evaluate_run(run_dir: str, gt: Optional[List[dict]], tolerance: float,
                 duration_sec: Optional[float]) -> dict:
    d = Path(run_dir)
    events = read_events(d / EVENTS_FILE)
    reviews = load_reviews(d)
    summary = read_json(d / SUMMARY_FILE) or {}
    # FA/hour needs a real duration; it is never guessed (None -> reported as unknown).
    dur = duration_sec or summary.get("analyzed_duration_sec") or None
    result = {"run_dir": str(d), "alerts_total": len(events),
              "duration_sec": dur, "tolerance_sec": tolerance, "rules": {}}
    if not events and not (d / EVENTS_FILE).exists():
        result["warning"] = f"{EVENTS_FILE} not found"

    for rid in RULES:
        alerts = [e for e in events if e.get("rule_id") == rid]
        row = {"alerts": len(alerts)}
        if gt is not None:
            gts = [g for g in gt if g["rule_id"] == rid]
            pairs, un_a, un_g, dup = match_rule(alerts, gts, tolerance)
            tp, fp, fn = len(pairs), len(un_a), len(un_g)
            p, r = _safe_div(tp, tp + fp), _safe_div(tp, tp + fn)
            row.update({"gt_events": len(gts), "TP": tp, "FP": fp, "FN": fn,
                        "duplicate_FP": len(dup), "precision": p, "recall": r,
                        "f1": f1(p, r),
                        "fp_event_ids": [alerts[i]["event_id"] for i in un_a],
                        "fn_intervals": [[gts[i]["start_sec"], gts[i]["end_sec"]] for i in un_g]})
            if pairs:
                lat = [alert_interval(alerts[ai])[1] - gts[gi]["start_sec"] for ai, gi in pairs]
                row["mean_detection_delay_sec"] = round(sum(lat) / len(lat), 2)
            fp_for_rate = fp
        else:
            fp_for_rate = None
        rv = [reviews[e["event_id"]]["label"] for e in alerts if e["event_id"] in reviews]
        if rv:
            rtp, rfp = rv.count("TP"), rv.count("FP")
            row["review"] = {"reviewed": len(rv), "TP": rtp, "FP": rfp,
                             "precision": _safe_div(rtp, rtp + rfp),
                             "unreviewed": len(alerts) - len(rv)}
            if gt is None:
                fp_for_rate = rfp
        row["false_alerts_per_hour"] = (round(fp_for_rate / (dur / 3600.0), 2)
                                        if (fp_for_rate is not None and dur) else None)
        result["rules"][rid] = row

    if gt is not None:
        tot = {k: sum(result["rules"][r].get(k, 0) for r in RULES) for k in ("TP", "FP", "FN")}
        p, r = _safe_div(tot["TP"], tot["TP"] + tot["FP"]), _safe_div(tot["TP"], tot["TP"] + tot["FN"])
        result["overall"] = {**tot, "precision": p, "recall": r, "f1": f1(p, r),
                             "false_alerts_per_hour": round(tot["FP"] / (dur / 3600.0), 2) if dur else None}
    lat = (summary.get("latency") or {}).get("total")
    result["performance"] = {"fps_avg": summary.get("fps_avg"),
                             "frame_latency_p50_ms": lat.get("p50_ms") if lat else None,
                             "frame_latency_p95_ms": lat.get("p95_ms") if lat else None}
    alert_lat = [e.get("details", {}).get("pipeline_latency_ms") for e in events]
    alert_lat = sorted(v for v in alert_lat if isinstance(v, (int, float)))
    if alert_lat:
        result["performance"]["alert_latency_median_ms"] = alert_lat[len(alert_lat) // 2]
    return result


# ----------------------------------------------------------------- output
def _fmt(v, pct=False):
    if v is None:
        return "–"
    return f"{v*100:.1f}%" if pct else str(v)


def to_markdown(results: Dict[str, dict], has_gt: bool) -> str:
    lines = ["# Kết quả đánh giá", ""]
    for name, res in results.items():
        dur = res["duration_sec"]
        lines += [f"## {name}", "",
                  f"- Thư mục: `{res['run_dir']}`",
                  f"- Thời lượng video phân tích: {f'{dur/60:.2f} phút ({dur/3600:.4f} giờ)' if dur else 'không rõ (không tính được cảnh báo giả/giờ)'}",
                  f"- Dung sai ghép cảnh báo–ground truth: ±{res['tolerance_sec']} s",
                  f"- Tổng cảnh báo: {res['alerts_total']}"]
        if res.get("warning"):
            lines.append(f"- ⚠ {res['warning']}")
        perf = res["performance"]
        if perf.get("fps_avg") is not None:
            lines.append(f"- FPS trung bình: {perf['fps_avg']}; độ trễ khung p50/p95: "
                         f"{perf.get('frame_latency_p50_ms')}/{perf.get('frame_latency_p95_ms')} ms")
        lines.append("")
        if has_gt:
            lines += ["| Rule | GT | Cảnh báo | TP | FP | FN | Precision | Recall | F1 | FA/giờ |",
                      "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
            for rid in RULES:
                r = res["rules"][rid]
                lines.append(f"| {rid} | {r['gt_events']} | {r['alerts']} | {r['TP']} | {r['FP']} | {r['FN']} | "
                             f"{_fmt(r['precision'], True)} | {_fmt(r['recall'], True)} | {_fmt(r['f1'], True)} | "
                             f"{_fmt(r['false_alerts_per_hour'])} |")
            o = res["overall"]
            lines.append(f"| **Tổng** | | {res['alerts_total']} | {o['TP']} | {o['FP']} | {o['FN']} | "
                         f"{_fmt(o['precision'], True)} | {_fmt(o['recall'], True)} | {_fmt(o['f1'], True)} | "
                         f"{_fmt(o['false_alerts_per_hour'])} |")
        else:
            lines += ["| Rule | Cảnh báo | Đã duyệt | TP | FP | Precision (đã duyệt) | FA/giờ |",
                      "|---|--:|--:|--:|--:|--:|--:|"]
            for rid in RULES:
                r = res["rules"][rid]
                rv = r.get("review") or {}
                lines.append(f"| {rid} | {r['alerts']} | {rv.get('reviewed', 0)} | {rv.get('TP', '–')} | "
                             f"{rv.get('FP', '–')} | {_fmt(rv.get('precision'), True)} | {_fmt(r['false_alerts_per_hour'])} |")
            lines += ["", "_Không có ground truth: chỉ tính Precision trên các cảnh báo đã duyệt; Recall cần `--gt`._"]
        lines.append("")

    if len(results) > 1:
        lines += ["## So sánh các run", "",
                  "| Run | Rule | Cảnh báo | " + ("TP | FP | FN | Precision | Recall | " if has_gt else "") + "FA/giờ |",
                  "|---|---|--:|" + ("--:|--:|--:|--:|--:|" if has_gt else "") + "--:|"]
        for name, res in results.items():
            for rid in RULES:
                r = res["rules"][rid]
                if has_gt:
                    lines.append(f"| {name} | {rid} | {r['alerts']} | {r['TP']} | {r['FP']} | {r['FN']} | "
                                 f"{_fmt(r['precision'], True)} | {_fmt(r['recall'], True)} | {_fmt(r['false_alerts_per_hour'])} |")
                else:
                    lines.append(f"| {name} | {rid} | {r['alerts']} | {_fmt(r['false_alerts_per_hour'])} |")
        lines.append("")
    return "\n".join(lines)


def to_csv(results: Dict[str, dict], path: Path, has_gt: bool):
    cols = ["run", "rule", "alerts"] + (["gt_events", "TP", "FP", "FN", "precision", "recall", "f1"] if has_gt else []) \
        + ["false_alerts_per_hour"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for name, res in results.items():
            for rid in RULES:
                r = res["rules"][rid]
                w.writerow([name, rid, r["alerts"]]
                           + ([r["gt_events"], r["TP"], r["FP"], r["FN"], r["precision"], r["recall"], r["f1"]] if has_gt else [])
                           + [r["false_alerts_per_hour"]])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate alerts: Precision, Recall, false alerts/hour")
    ap.add_argument("--run", action="append", required=True,
                    help="run folder, or name=folder; repeat to compare")
    ap.add_argument("--gt", default=None, help="ground-truth JSON/CSV")
    ap.add_argument("--tolerance", type=float, default=2.0, help="matching tolerance in seconds")
    ap.add_argument("--duration-sec", type=float, default=None,
                    help="video duration (overrides GT/summary); needed for false alerts/hour")
    ap.add_argument("--out", default="reports")
    args = ap.parse_args(argv)
    if args.tolerance < 0:
        ap.error("--tolerance must be >= 0")

    gt, gt_dur = (load_ground_truth(args.gt) if args.gt else (None, None))
    results = {}
    for spec in args.run:
        name, _, folder = spec.partition("=") if "=" in spec else (Path(spec).name, "", spec)
        results[name] = evaluate_run(folder, gt, args.tolerance, args.duration_sec or gt_dur)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    md = to_markdown(results, gt is not None)
    (out / "metrics.md").write_text(md, encoding="utf-8")
    to_csv(results, out / "metrics.csv", gt is not None)
    print(md)
    print(f"\nSaved: {out / 'metrics.json'}, {out / 'metrics.md'}, {out / 'metrics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
