from __future__ import annotations
import importlib.util
from pathlib import Path
import yaml
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_labcoat_dataset.py"
spec = importlib.util.spec_from_file_location("prepare_labcoat_dataset", SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(mod)


def test_detection_bbox_is_preserved():
    cls, fmt, coords = mod.parse_yolo_row("2 0.5 0.5 0.4 0.8".split(), Path("x.txt"), 1)
    assert cls == 2 and fmt == "bbox"
    assert coords == [0.5, 0.5, 0.4, 0.8]


def test_polygon_is_converted_to_enclosing_bbox():
    # Rectangle polygon: xmin=.1, xmax=.7, ymin=.2, ymax=.8
    row = "2 .1 .2 .7 .2 .7 .8 .1 .8".split()
    cls, fmt, coords = mod.parse_yolo_row(row, Path("poly.txt"), 1)
    assert cls == 2 and fmt == "polygon"
    assert coords == pytest.approx([0.4, 0.5, 0.6, 0.6])


def test_mixed_bbox_and_polygon_only_keep_target_and_remap_to_zero(tmp_path):
    label = tmp_path / "mixed.txt"
    label.write_text(
        "0 0.5 0.5 0.2 0.3\n"  # gloves bbox: ignored
        "2 0.5 0.5 0.4 0.8\n"  # lab coat bbox
        "2 .1 .2 .7 .2 .7 .8 .1 .8\n"  # lab coat polygon
        "1 .2 .2 .3 .2 .3 .3 .2 .3\n",  # goggles polygon: ignored
        encoding="utf-8"
    )
    parsed = mod.parse_label_file_detailed(label, {0:"Gloves",1:"Goggles",2:"Lab_coat"}, 2)
    assert parsed["target_count"] == 2
    assert parsed["bbox_count"] == 1
    assert parsed["polygon_count"] == 1
    assert parsed["ignored_non_target_objects"] == 2
    assert all(line.startswith("0 ") for line in parsed["rows"])


def test_polygon_requires_pairs_and_at_least_three_points():
    bad_rows = [
        "2 .1 .2 .3 .4 .5",  # too few points / invalid odd field count
        "2 .1 .2 .3 .4 .5 .6 .7",  # odd number of coordinate values
    ]
    for row in bad_rows:
        try:
            mod.parse_yolo_row(row.split(), Path("bad.txt"), 1)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected invalid polygon to fail: {row}")


def test_polygon_rejects_out_of_range_and_non_finite():
    for row in ("2 0 .1 1.1 .2 .3 .4 .5 .6", "2 0 .1 nan .2 .3 .4 .5 .6"):
        try:
            mod.parse_yolo_row(row.split(), Path("bad.txt"), 1)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected invalid polygon to fail: {row}")


def _make_source(root: Path):
    names = ["Gloves", "Goggles", "Lab_coat"]
    for split in ("train", "valid", "test"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        (root / split / "images" / f"{split}.jpg").write_bytes(f"img-{split}".encode())
        (root / split / "labels" / f"{split}.txt").write_text(
            "0 0.2 0.2 0.1 0.1\n"
            "2 0.5 0.5 0.4 0.8\n"
            "2 .1 .2 .7 .2 .7 .8 .1 .8\n", encoding="utf-8"
        )
    (root / "data.yaml").write_text(yaml.safe_dump({
        "train":"../train/images", "val":"../valid/images", "test":"../test/images",
        "nc":3, "names":names
    }, sort_keys=False), encoding="utf-8")


def test_prepare_mixed_formats_all_splits_and_report(tmp_path):
    source, output = tmp_path / "raw", tmp_path / "normalized"
    source.mkdir()
    _make_source(source)
    report = mod.prepare(source, output, source_class_id=2)
    assert report["splits"]["train"]["images"] == 1
    assert report["splits"]["train"]["lab_coat_objects"] == 2
    assert report["splits"]["train"]["lab_coat_bbox_objects"] == 1
    assert report["splits"]["train"]["lab_coat_polygon_objects_converted_to_bbox"] == 1
    assert report["annotation_formats"]["bbox_rows"] == 6
    assert report["annotation_formats"]["polygon_rows"] == 3
    out_label = (output / "labels/train/train.txt").read_text(encoding="utf-8").splitlines()
    assert len(out_label) == 2
    assert all(len(x.split()) == 5 and x.split()[0] == "0" for x in out_label)
    cfg = yaml.safe_load((output / "data.yaml").read_text(encoding="utf-8"))
    assert cfg["names"] == {0:"lab_coat"}
    assert cfg["test"] == "images/test"


def test_missing_label_file_fails_before_output_creation(tmp_path):
    source, output = tmp_path / "raw", tmp_path / "normalized"
    source.mkdir()
    _make_source(source)
    (source / "train/labels/train.txt").unlink()
    try:
        mod.prepare(source, output, source_class_id=2)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Missing label file must fail")
    assert not output.exists()


def test_bbox_with_tiny_rounding_overshoot_is_clipped():
    row = "2 0.533218954248366 0.7026225490196079 0.5108986928104575 0.5947794117647058".split()
    cls, fmt, coords = mod.parse_yolo_row(row, Path("rounding.txt"), 1)
    assert cls == 2 and fmt == "bbox"
    x, y, w, h = coords
    assert y + h / 2 == pytest.approx(1.0)
    assert y - h / 2 == pytest.approx(0.40523284313725494)
    assert x - w / 2 >= 0 and x + w / 2 <= 1


def test_bbox_with_material_out_of_bounds_is_rejected():
    row = "2 0.5 0.8 0.4 0.5".split()  # bottom = 1.05
    with pytest.raises(ValueError, match="beyond tolerance"):
        mod.parse_yolo_row(row, Path("bad_bbox.txt"), 1)


def test_prepare_reports_clipped_bbox_and_writes_valid_yolo_box(tmp_path):
    source, output = tmp_path / "raw", tmp_path / "normalized"
    source.mkdir()
    _make_source(source)
    edge_label = source / "train" / "labels" / "train.txt"
    edge_label.write_text(
        "2 0.533218954248366 0.7026225490196079 0.5108986928104575 0.5947794117647058\n",
        encoding="utf-8",
    )
    report = mod.prepare(source, output, source_class_id=2)
    train = report["splits"]["train"]
    assert train["lab_coat_bbox_objects_clipped_to_image_bounds"] == 1
    assert report["boundary_policy"]["clipped_target_bbox_rows"] == 1
    output_row = (output / "labels" / "train" / "train.txt").read_text().split()
    assert output_row[0] == "0"
    x, y, w, h = map(float, output_row[1:])
    assert x - w / 2 >= 0 and y - h / 2 >= 0
    assert x + w / 2 <= 1 and y + h / 2 <= 1
