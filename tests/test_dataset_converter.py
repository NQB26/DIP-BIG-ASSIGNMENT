from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_labcoat_dataset.py"
spec = importlib.util.spec_from_file_location("prepare_labcoat_dataset", SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(mod)


def make_source(root: Path, roboflow_paths: bool = False, include_test: bool = True):
    names = ["goggles", "person", "gloves", "Lab Coat"]
    folders = ["train", "valid"] + (["test"] if include_test else [])
    for split in folders:
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        img = root / split / "images" / f"{split}_01.jpg"
        img.write_bytes(f"fake-image-{split}".encode())
        label_lines = [
            "1 0.50 0.50 0.40 0.80",
            "3 0.50 0.55 0.20 0.35",
            "0 0.20 0.20 0.10 0.10",
        ]
        (root / split / "labels" / f"{split}_01.txt").write_text("\n".join(label_lines), encoding="utf-8")
    train_path = "../train/images" if roboflow_paths else "train/images"
    val_path = "../valid/images" if roboflow_paths else "valid/images"
    test_path = "../test/images" if roboflow_paths else "test/images"
    data = {"train": train_path, "val": val_path, "test": test_path, "nc": 4, "names": names}
    (root / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def test_normalize_class_names():
    assert mod.norm_name("Lab Coat") == "labcoat"
    assert mod.norm_name("lab_coat") == "labcoat"
    assert mod.norm_name("LAB-COAT") == "labcoat"


def test_split_hint_handles_parent_relative_roboflow_paths():
    assert mod.split_hint("../train/images") == "train"
    assert mod.split_hint("../valid/images") == "val"
    assert mod.split_hint("../test/images") == "test"


def test_discover_splits_never_maps_valid_or_missing_test_to_train(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    make_source(source, roboflow_paths=True, include_test=False)
    data_yaml = source / "data.yaml"
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    splits, missing = mod.discover_splits(source, data_yaml, data)
    assert splits["train"] == (source / "train" / "images").resolve()
    assert splits["val"] == (source / "valid" / "images").resolve()
    assert "test" not in splits
    assert "test" in missing


def test_prepare_converts_roboflow_parent_relative_train_val_without_test(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "prepared"
    source.mkdir()
    make_source(source, roboflow_paths=True, include_test=False)
    summary = mod.prepare(source, output)
    assert summary["splits"]["train"]["images"] == 1
    assert summary["splits"]["val"]["images"] == 1
    assert "test" not in summary["splits"]
    assert (output / "images" / "val" / "valid_01.jpg").is_file()
    config = yaml.safe_load((output / "data.yaml").read_text(encoding="utf-8"))
    assert config["names"] == {0: "lab_coat"}
    assert "test" not in config


def test_prepare_all_splits_when_source_has_test(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "prepared"
    source.mkdir()
    make_source(source, roboflow_paths=True, include_test=True)
    summary = mod.prepare(source, output)
    assert set(summary["splits"]) == {"train", "val", "test"}
    assert summary["splits"]["train"]["images"] == 1
    assert summary["splits"]["val"]["images"] == 1
    assert summary["splits"]["test"]["images"] == 1


def test_explicit_source_class_id_supports_opaque_class_names(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "prepared"
    source.mkdir()
    (source / "train" / "images").mkdir(parents=True)
    (source / "train" / "labels").mkdir(parents=True)
    (source / "valid" / "images").mkdir(parents=True)
    (source / "valid" / "labels").mkdir(parents=True)
    for split in ("train", "valid"):
        (source / split / "images" / f"{split}.jpg").write_bytes(f"img-{split}".encode())
        (source / split / "labels" / f"{split}.txt").write_text("0 0.5 0.5 0.2 0.3\n", encoding="utf-8")
    (source / "data.yaml").write_text("train: train/images\nval: valid/images\nnc: 1\nnames: ['2']\n", encoding="utf-8")
    summary = mod.prepare(source, output, source_class_id=0)
    assert summary["source_target_class_name"] == "2"
    label = (output / "labels" / "train" / "train.txt").read_text(encoding="utf-8").strip()
    assert label.startswith("0 ")


def test_converter_refuses_existing_output(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "prepared"
    source.mkdir()
    make_source(source)
    output.mkdir()
    try:
        mod.prepare(source, output)
    except FileExistsError:
        pass
    else:
        raise AssertionError("expected FileExistsError")


def test_converter_rejects_bad_bbox(tmp_path):
    label = tmp_path / "bad.txt"
    label.write_text("0 0.5 0.5 1.2 0.4\n", encoding="utf-8")
    try:
        mod.parse_label_file(label, {0: "Lab Coat"}, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected invalid bbox to be rejected")
