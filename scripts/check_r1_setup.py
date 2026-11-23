"""Preflight checks before running R1 (NO_LAB_COAT) on a real video."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tracking_rules.rule_engine import load_rules_config, load_zones  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zones", default=str(ROOT / "configs" / "zones.yaml"))
    parser.add_argument("--rules", default=str(ROOT / "configs" / "rules_r1_test.yaml"))
    parser.add_argument("--video")
    parser.add_argument("--coco-weights")
    parser.add_argument("--lab-coat-weights")
    args = parser.parse_args()

    errors: list[str] = []
    warnings: list[str] = []
    for name, path in (("zones", args.zones), ("rules", args.rules)):
        if not Path(path).is_file():
            errors.append(f"Missing {name} file: {path}")

    if errors:
        for item in errors:
            print(f"ERROR: {item}")
        return 2

    try:
        cfg = load_rules_config(args.rules)
        zones = load_zones(args.zones)
    except Exception as exc:
        print(f"ERROR: cannot load configuration: {exc}")
        return 2

    r1 = cfg["rules"]["R1"]
    common = cfg["common"]
    general = [zone for zone in zones if zone.type == "general"]
    restricted = [zone for zone in zones if zone.type == "restricted"]

    if not r1.get("enabled", False):
        errors.append("R1 must be enabled in the selected rules profile")
    for rid in ("R2", "R3"):
        if cfg["rules"][rid].get("enabled", False):
            errors.append(f"{rid} must remain disabled during the isolated R1 test")
    if common.get("full_frame_fallback", False):
        errors.append("full_frame_fallback must be false for real-camera R1 validation")
    if not general:
        errors.append("zones.yaml has no general zone; draw one using scripts/draw_zones.py before R1")
    if float(r1.get("min_person_height_px", 0)) <= 0:
        errors.append("R1.min_person_height_px must be positive")
    if not 0 <= float(r1.get("min_ioa_threshold", -1)) <= 1:
        errors.append("R1.min_ioa_threshold must be within [0, 1]")

    for label, path in (("video", args.video), ("COCO weights", args.coco_weights),
                        ("lab-coat weights", args.lab_coat_weights)):
        if path and not Path(path).is_file():
            errors.append(f"Missing {label}: {path}")

    print("=== R1 PRE-FLIGHT ===")
    print(f"R1 enabled: {r1.get('enabled')}")
    print(f"R1 target zone: {r1.get('target_zone_type')}")
    print(f"General zones: {[z.zone_id for z in general]}")
    print(f"Restricted zones preserved: {[z.zone_id for z in restricted]}")
    print(f"Minimum duration: {r1.get('min_duration_sec')} s")
    print(f"Coat confidence: {r1.get('coat_conf_min')}")
    print(f"Minimum IoA: {r1.get('min_ioa_threshold')}")
    print(f"Minimum person height: {r1.get('min_person_height_px')} px")
    print(f"Coat hold: {r1.get('coat_hold_sec')} s")

    if errors:
        for item in errors:
            print(f"ERROR: {item}")
        return 2

    if not restricted:
        warnings.append("No restricted zones are configured. This is acceptable for an R1-only test.")
    if args.video is None:
        warnings.append("No video path supplied; video existence was not checked.")
    for item in warnings:
        print(f"WARNING: {item}")
    print("PASS: R1 configuration is ready for an isolated test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
