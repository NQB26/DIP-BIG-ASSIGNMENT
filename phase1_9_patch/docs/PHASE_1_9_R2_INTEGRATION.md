# Phase 1.9 – R2 isolated integration

R2 is the restricted-zone intrusion rule. The Phase 1.8 profile enabled R1 only,
so it was expected not to emit R2 events.

## Behavior
- Uses the person's `bottom_center` anchor, not any overlap of the full person box.
- The anchor must be inside a polygon whose `type: restricted`.
- The condition must persist for `min_duration_sec` (default 1.5 s), allowing
  brief gaps no longer than `grace_period_sec` (default 0.3 s).
- R1 and R3 are disabled in this profile.

## Commands (PowerShell)

```powershell
$env:PYTHONPATH = "src"
python -m pytest tests/test_r2_smoke.py -q
python scripts/check_r2_setup.py `
  --zones configs/zones.yaml `
  --rules configs/rules_r2_test.yaml `
  --video $video `
  --coco-weights yolo11n.pt `
  --lab-coat-weights $coat
```

Then run the existing `scripts/test_tracker.py` with:

```powershell
python scripts/test_tracker.py `
  --video $video `
  --weights yolo11n.pt `
  --lab-coat-weights $coat `
  --zones configs/zones.yaml `
  --rules configs/rules_r2_test.yaml `
  --out runs/r2_validation_v1 `
  --imgsz 512 --device cpu `
  --coco-conf 0.25 --coat-conf 0.25 `
  --max-frames 900 --show
```

Before interpreting a negative result, inspect `configs/zones.yaml` against the
video reference frame and verify that the normalized restricted polygon covers
the floor area where the person's feet are expected. A person's torso entering
while their bottom-center remains outside does not trigger this implementation.
