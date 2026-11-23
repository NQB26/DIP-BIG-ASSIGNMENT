"""Preflight checks for the isolated pose-assisted R3 profile."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tracking_rules.rule_engine import load_rules_config, load_zones  # noqa: E402


def check_weights(value: str, label: str, allow_official_alias: bool = False) -> str:
    candidate = Path(value)
    if candidate.is_file():
        return f"OK: {label}: {candidate.resolve()}"
    if allow_official_alias and candidate.name == "yolo11n-pose.pt":
        return f"INFO: {label}: official Ultralytics alias; first YOLO load may download it"
    raise FileNotFoundError(f"{label} not found: {value}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zones", required=True)
    parser.add_argument("--rules", required=True)
    parser.add_argument("--coco-weights", required=True)
    parser.add_argument("--lab-coat-weights", required=True)
    parser.add_argument("--pose-weights", default="yolo11n-pose.pt")
    args = parser.parse_args()

    cfg = load_rules_config(args.rules)
    zones = load_zones(args.zones)
    r3 = cfg["rules"]["R3"]
    if not r3.get("enabled", False):
        raise ValueError("R3 must be enabled in the selected profile")
    if r3.get("association_mode") != "pose_hand":
        raise ValueError("R3.association_mode must be pose_hand")
    if not any(z.type == "general" for z in zones):
        raise ValueError("zones.yaml needs at least one general polygon for R3")
    if cfg["rules"]["R1"].get("enabled", False) or cfg["rules"]["R2"].get("enabled", False):
        raise ValueError("R1 and R2 must be disabled in the isolated R3 profile")

    print("=== R3 POSE PREFLIGHT ===")
    print(f"R1 enabled: {cfg['rules']['R1'].get('enabled', False)}")
    print(f"R2 enabled: {cfg['rules']['R2'].get('enabled', False)}")
    print(f"R3 enabled: {r3['enabled']}")
    print(f"R3 association mode: {r3['association_mode']}")
    print(f"General zones: {[z.zone_id for z in zones if z.type == 'general']}")
    print(f"Minimum duration: {r3['min_duration_sec']}s")
    print(f"Maximum phone-wrist distance ratio: {r3['phone_wrist_max_dist_ratio']}")
    print(check_weights(args.coco_weights, "COCO detector"))
    print(check_weights(args.lab_coat_weights, "Lab coat detector"))
    print(check_weights(args.pose_weights, "Pose model", allow_official_alias=True))

    try:
        import ultralytics
        print(f"Ultralytics: {ultralytics.__version__}")
    except Exception as exc:
        raise RuntimeError(f"Cannot import ultralytics: {exc}") from exc
    print("PASS: R3 pose-assisted configuration is ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
