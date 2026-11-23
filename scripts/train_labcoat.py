"""Conservative Windows-safe baseline fine-tuning for a one-class lab_coat model.

This is intended for an early pipeline baseline on a small dataset. It does not
make the resulting checkpoint a production-ready detector.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _resolve_split(root: Path, value: Any) -> Path:
    p = Path(str(value)).expanduser()
    return p.resolve() if p.is_absolute() else (root / p).resolve()


def validate_one_class_data(data_yaml: Path) -> dict[str, Any]:
    if not data_yaml.is_file():
        raise FileNotFoundError(data_yaml)
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    raw_names = data.get("names")
    if isinstance(raw_names, list):
        names = {i: str(v) for i, v in enumerate(raw_names)}
    elif isinstance(raw_names, dict):
        names = {int(k): str(v) for k, v in raw_names.items()}
    else:
        raise ValueError("data.yaml must define names as a list or mapping")
    if len(names) != 1 or names.get(0, "").strip().lower().replace("-", "_").replace(" ", "_") != "lab_coat":
        raise ValueError(f"Expected exactly one class ID 0 named lab_coat; got {names}")

    root = Path(data.get("path", data_yaml.parent)).expanduser()
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    split_info: dict[str, Any] = {}
    dirs: dict[str, Path] = {}
    for key in ("train", "val"):
        split_value = data.get(key)
        if not split_value:
            raise ValueError(f"data.yaml missing required {key!r} split")
        split_dir = _resolve_split(root, split_value)
        if not split_dir.is_dir():
            raise FileNotFoundError(f"{key} split path does not exist: {split_dir}")
        dirs[key] = split_dir
        images = [p for p in split_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        if not images:
            raise ValueError(f"No images found in {key}: {split_dir}")
        label_parts = list(split_dir.parts)
        image_idx = [i for i, part in enumerate(label_parts) if part.lower() == "images"]
        if image_idx:
            label_parts[image_idx[-1]] = "labels"
            label_dir = Path(*label_parts)
        else:
            label_dir = split_dir.parent / "labels"
        missing = [p for p in images if not (label_dir / p.relative_to(split_dir)).with_suffix(".txt").is_file()]
        split_info[key] = {"images": len(images), "missing_label_files": len(missing), "image_dir": str(split_dir), "label_dir": str(label_dir)}
        if missing:
            raise ValueError(f"{key}: {len(missing)} images do not have matching label files")
    if dirs["train"] == dirs["val"]:
        raise ValueError("Train and validation resolve to the same directory; refusing data leakage")

    if "test" not in data:
        print("WARNING: no independent test split exists. Do not report validation metrics as final test metrics.")
    if split_info["train"]["images"] < 100:
        print(f"WARNING: only {split_info['train']['images']} training images; run is an experimental baseline.")
    if split_info["val"]["images"] < 30:
        print(f"WARNING: only {split_info['val']['images']} validation images; metrics may vary substantially.")
    return {"data_yaml": str(data_yaml), "class_names": names, "splits": split_info, "has_test": "test" in data}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--freeze", type=int, default=10)
    ap.add_argument("--project", default="runs/train")
    ap.add_argument("--name", default="labcoat_baseline_v1")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    try:
        data_yaml = args.data.expanduser().resolve()
        audit = validate_one_class_data(data_yaml)
        model_path = Path(args.model).expanduser()
        if model_path.suffix.lower() == ".pt" and not model_path.is_file():
            raise FileNotFoundError(f"Pretrained weights not found: {model_path.resolve()}")
        if args.epochs < 1 or args.imgsz < 32 or args.batch < 1:
            raise ValueError("epochs, imgsz and batch must be positive (imgsz >= 32)")

        from ultralytics import YOLO
        model = YOLO(str(model_path) if model_path.exists() else args.model)
        results = model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=args.device,
            patience=args.patience,
            freeze=args.freeze,
            project=args.project,
            name=args.name,
            exist_ok=False,
            seed=args.seed,
            deterministic=True,
            cache=False,
            plots=True,
            save=True,
            save_period=10,
            optimizer="AdamW",
            lr0=0.001,
            close_mosaic=min(10, max(0, args.epochs // 4)),
            mosaic=0.5,
            mixup=0.0,
            degrees=2.0,
            translate=0.05,
            scale=0.2,
            fliplr=0.5,
            flipud=0.0,
            amp=False if str(args.device).lower() == "cpu" else True,
            verbose=True,
        )
        save_dir = Path(results.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        run_meta = {
            "audit": audit,
            "model_initialization": str(model_path.resolve()) if model_path.exists() else args.model,
            "epochs_requested": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
            "optimizer": "AdamW",
            "lr0": 0.001,
            "freeze": args.freeze,
            "seed": args.seed,
            "run_dir": str(save_dir.resolve()),
            "best_checkpoint": str((save_dir / "weights" / "best.pt").resolve()),
            "test_split_available": audit["has_test"],
        }
        (save_dir / "run_metadata.json").write_text(json.dumps(run_meta, indent=2, default=str), encoding="utf-8")
        print(json.dumps(run_meta, indent=2, default=str))
        print(f"Training completed. Best checkpoint: {save_dir / 'weights' / 'best.pt'}")
        return 0
    except Exception as exc:
        print(f"TRAINING ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
