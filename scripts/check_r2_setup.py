"""Preflight check before an isolated R2 restricted-zone test."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tracking_rules.rule_engine import load_rules_config, load_zones  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zones", default=str(ROOT / "configs" / "zones.yaml"))
    p.add_argument("--rules", default=str(ROOT / "configs" / "rules_r2_test.yaml"))
    p.add_argument("--video")
    p.add_argument("--coco-weights")
    p.add_argument("--lab-coat-weights")
    a = p.parse_args()

    errors: list[str] = []
    for label, path in (("zones", a.zones), ("rules", a.rules)):
        if not Path(path).is_file():
            errors.append(f"Missing {label} file: {path}")
    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        return 2

    try:
        cfg = load_rules_config(a.rules)
        zones = load_zones(a.zones)
    except Exception as exc:
        print(f"ERROR: cannot load configuration: {exc}")
        return 2

    rules = cfg["rules"]
    common = cfg["common"]
    restricted = [z for z in zones if z.type == "restricted"]
    r2 = rules["R2"]

    if rules["R1"].get("enabled", False):
        errors.append("R1 must remain disabled during the isolated R2 test")
    if not r2.get("enabled", False):
        errors.append("R2 must be enabled")
    if rules["R3"].get("enabled", False):
        errors.append("R3 must remain disabled during the isolated R2 test")
    if r2.get("target_zone_type") != "restricted":
        errors.append("R2.target_zone_type must be 'restricted'")
    if not restricted:
        errors.append("zones.yaml has no restricted polygon")
    if common.get("full_frame_fallback", False):
        errors.append("full_frame_fallback must be false")
    if r2.get("anchor_point", "bottom_center") != "bottom_center":
        print("WARNING: R2 uses a non-default anchor_point; verify its placement visually")
    if float(r2.get("min_duration_sec", 0)) <= 0:
        errors.append("R2.min_duration_sec must be positive")

    for label, path in (("video", a.video), ("COCO weights", a.coco_weights),
                        ("lab-coat weights", a.lab_coat_weights)):
        if path and not Path(path).is_file():
            errors.append(f"Missing {label}: {path}")

    print("=== R2 PRE-FLIGHT ===")
    print(f"R1 enabled: {rules['R1'].get('enabled')}")
    print(f"R2 enabled: {r2.get('enabled')}")
    print(f"R3 enabled: {rules['R3'].get('enabled')}")
    print(f"R2 target zone: {r2.get('target_zone_type')}")
    print(f"R2 anchor: {r2.get('anchor_point')}")
    print(f"Restricted zones: {[z.zone_id for z in restricted]}")
    print(f"Minimum duration: {r2.get('min_duration_sec')} s")
    print(f"Grace period: {r2.get('grace_period_sec')} s")

    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        return 2
    print("PASS: isolated R2 configuration is ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
