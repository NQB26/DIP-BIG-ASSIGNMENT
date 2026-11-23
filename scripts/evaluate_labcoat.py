"""Evaluate a trained lab_coat detector on the held-out test split."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--split", choices=("val", "test"), default="test")
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--project", default="reports/evaluation")
    ap.add_argument("--name", default="labcoat_detector")
    args = ap.parse_args()
    try:
        if not args.weights.is_file():
            raise FileNotFoundError(args.weights)
        data = yaml.safe_load(args.data.read_text(encoding="utf-8")) or {}
        if args.split not in data:
            raise ValueError(f"Dataset YAML has no {args.split!r} split. Available keys: {list(data)}")
        from ultralytics import YOLO
        model = YOLO(str(args.weights))
        metrics = model.val(
            data=str(args.data.resolve()), split=args.split,
            imgsz=args.imgsz, batch=args.batch, device=args.device,
            project=args.project, name=args.name, plots=True, verbose=True,
        )
        summary = {str(k): float(v) for k, v in metrics.results_dict.items()}
        output_dir = Path(metrics.save_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        print(f"Saved metrics: {output_dir / 'metrics_summary.json'}")
        return 0
    except Exception as exc:
        print(f"EVALUATION ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
