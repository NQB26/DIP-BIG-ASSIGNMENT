from pathlib import Path
import sys
import yaml
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from merge_labcoat_datasets import merge


def make_ds(root: Path, name="lab_coat"):
    root.mkdir(parents=True)
    (root / "images" / "train").mkdir(parents=True)
    (root / "labels" / "train").mkdir(parents=True)
    (root / "images" / "val").mkdir(parents=True)
    (root / "labels" / "val").mkdir(parents=True)
    # Distinct synthetic image bytes per split/file.
    for split, stem, content, label in [
        ("train", "a", b"image-a", "0 0.5 0.5 0.4 0.4\n"),
        ("val", "b", b"image-b", ""),
    ]:
        (root / "images" / split / f"{stem}.jpg").write_bytes(content)
        (root / "labels" / split / f"{stem}.txt").write_text(label, encoding="utf-8")
    data = {"path": root.as_posix(), "train": "images/train", "val": "images/val", "names": {0: name}}
    (root / "data.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")


def test_merge_normalized_sources_and_preserve_empty_negative(tmp_path):
    a, b, out = tmp_path / "source_a", tmp_path / "source_b", tmp_path / "merged"
    make_ds(a)
    make_ds(b)
    # Change bytes to avoid exact duplicate images across sources.
    (b / "images/train/a.jpg").write_bytes(b"image-a-2")
    (b / "images/val/b.jpg").write_bytes(b"image-b-2")
    report = merge([a, b], out)
    assert report["splits"]["train"]["images"] == 2
    assert report["splits"]["val"]["images"] == 2
    assert report["splits"]["val"]["negative_images"] == 2
    assert report["splits"]["train"]["lab_coat_objects"] == 2
    data = yaml.safe_load((out / "data.yaml").read_text())
    assert data["names"] == {0: "lab_coat"}
    assert len(list((out / "labels/val").glob("*.txt"))) == 2


def test_reject_wrong_class_mapping(tmp_path):
    a, out = tmp_path / "source_a", tmp_path / "merged"
    make_ds(a, name="coat")
    with pytest.raises(ValueError, match="not normalized"):
        merge([a], out)


def test_reject_missing_label_file(tmp_path):
    a, out = tmp_path / "source_a", tmp_path / "merged"
    make_ds(a)
    (a / "labels/train/a.txt").unlink()
    with pytest.raises(FileNotFoundError, match="no corresponding label"):
        merge([a], out)


def test_reject_cross_split_duplicate(tmp_path):
    a, out = tmp_path / "source_a", tmp_path / "merged"
    make_ds(a)
    # Make val image byte-identical to train image.
    (a / "images/val/b.jpg").write_bytes(b"image-a")
    with pytest.raises(ValueError, match="crosses splits"):
        merge([a], out)


def test_reject_nonzero_output_class_id(tmp_path):
    a, out = tmp_path / "source_a", tmp_path / "merged"
    make_ds(a)
    (a / "labels/train/a.txt").write_text("2 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    with pytest.raises(ValueError, match="class ID must be 0"):
        merge([a], out)


def test_reject_existing_output(tmp_path):
    a, out = tmp_path / "source_a", tmp_path / "merged"
    make_ds(a)
    out.mkdir()
    with pytest.raises(FileExistsError):
        merge([a], out)
