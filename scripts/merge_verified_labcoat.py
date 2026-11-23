#!/usr/bin/env python3
"""Build a one-class YOLO V3 dataset from human-reviewed V1 and V2.

The source processed/curated datasets are never modified. A dry run validates
review status, image/label integrity, exact duplicate policy and split hygiene.
The actual run copies approved images/labels into a new output directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

SCRIPT_VERSION = "1.12.2"

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
APPROVED = {'verified_positive', 'verified_negative'}

# The five exact duplicate pairs previously found by SHA-256 across V1 and V2.
# Rule: keep the image's higher-priority split; use the manually verified V1
# annotation as the label donor for all five same-image pairs.
DUPLICATE_PAIRS = [
    {
        'group': 1,
        'v1_split': 'train', 'v1_stem': 'LC30_jpg.rf.ff59cede0fcca3e8bd218730d455506b',
        'v2_split': 'val',   'v2_stem': 'LC30_jpg.rf.b5e80e52d247c6fc5690ef9563d54820',
        'keep_source': 'v2', 'keep_split': 'val', 'label_source': 'v1',
    },
    {
        'group': 2,
        'v1_split': 'val', 'v1_stem': 'LC11_jpg.rf.95a0a7da389361000556f58747011c95',
        'v2_split': 'train', 'v2_stem': 'LC11_jpg.rf.175ce455fa4296a331f7eeaa2b2fbab4c',
        'keep_source': 'v1', 'keep_split': 'val', 'label_source': 'v1',
    },
    {
        'group': 3,
        'v1_split': 'val', 'v1_stem': 'LC12_jpg.rf.9592025042bc941f25daea190f99091e',
        'v2_split': 'test', 'v2_stem': 'LC12_jpg.rf.315aeef4e52131c9a5c83c51c2aa0155',
        'keep_source': 'v2', 'keep_split': 'test', 'label_source': 'v1',
    },
    {
        'group': 4,
        'v1_split': 'val', 'v1_stem': 'LC15_jpg.rf.fcf4ecd2bfb3ebb9128aeb42b3014581',
        'v2_split': 'train', 'v2_stem': 'LC15_jpg.rf.5261869a5a430f4f1b542d6bf3b29ec7',
        'keep_source': 'v1', 'keep_split': 'val', 'label_source': 'v1',
    },
    {
        'group': 5,
        'v1_split': 'val', 'v1_stem': 'LC1_jpg.rf.5ab37413bce3ea0541c1a28d4adc5c78',
        'v2_split': 'train', 'v2_stem': 'LC1_jpg.rf.12404f684c59f61749a38cde330e3e2d',
        'keep_source': 'v1', 'keep_split': 'val', 'label_source': 'v1',
    },
]

@dataclass
class Item:
    source: str
    root: Path
    split: str
    image: Path
    label: Path
    sha256: str
    boxes: List[List[float]]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_boxes(label_path: Path) -> List[List[float]]:
    if not label_path.is_file():
        raise FileNotFoundError(f'Missing label file for image: {label_path}')
    boxes: List[List[float]] = []
    for line_no, raw in enumerate(label_path.read_text(encoding='utf-8').splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f'{label_path}:{line_no}: expected 5 YOLO fields, got {len(fields)}')
        cls_f, xc, yc, w, h = map(float, fields)
        if abs(cls_f) > 1e-6:
            raise ValueError(f'{label_path}:{line_no}: expected normalized class 0, got {cls_f}')
        if not all(0.0 <= v <= 1.0 for v in (xc, yc, w, h)) or w <= 0 or h <= 0:
            raise ValueError(f'{label_path}:{line_no}: invalid normalized xywh {(xc, yc, w, h)}')
        if xc - w / 2 < -1e-6 or yc - h / 2 < -1e-6 or xc + w / 2 > 1 + 1e-6 or yc + h / 2 > 1 + 1e-6:
            raise ValueError(f'{label_path}:{line_no}: box corners outside image bounds')
        boxes.append([xc, yc, w, h])
    return boxes


def read_reviewed_dataset(source: str, root: Path) -> Tuple[List[Item], dict]:
    root = root.resolve()
    summary_path = root / 'review_summary.json'
    manifest_path = root / 'review_manifest.csv'
    if not summary_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f'{source}: expected review_summary.json and review_manifest.csv in {root}')
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    with manifest_path.open('r', newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    status_counts = Counter((r.get('status') or 'pending') for r in rows)
    # Treat pending/needs_review as hard blockers. Rejected images are intentionally excluded.
    blockers = {s: status_counts[s] for s in ('pending', 'needs_review') if status_counts[s]}
    if blockers:
        raise RuntimeError(f'{source}: review is not fully approved; unresolved statuses: {blockers}')
    if status_counts['verified_positive'] + status_counts['verified_negative'] != int(summary.get('approved_total', -1)):
        raise RuntimeError(f'{source}: review_summary.json does not match review_manifest.csv approved counts')
    if len(rows) != int(summary.get('total_images', -1)):
        raise RuntimeError(f'{source}: review_summary.json total_images does not match manifest rows')

    items: List[Item] = []
    for row in rows:
        status = row.get('status', 'pending')
        if status not in APPROVED:
            continue
        image_rel = Path(row['image'])
        label_rel = Path(row['label'])
        image = root / image_rel
        label = root / label_rel
        if not image.is_file():
            raise FileNotFoundError(f'{source}: approved image missing from curated output: {image}')
        boxes = read_boxes(label)
        if status == 'verified_positive' and not boxes:
            raise ValueError(f'{source}: positive image has empty label: {label}')
        if status == 'verified_negative' and boxes:
            raise ValueError(f'{source}: negative image has non-empty label: {label}')
        # image path layout must be images/<split>/...
        parts = image_rel.parts
        if len(parts) < 3 or parts[0] != 'images':
            raise ValueError(f'{source}: unexpected image path in manifest: {image_rel}')
        split = parts[1]
        if split not in {'train', 'val', 'test'}:
            raise ValueError(f'{source}: unsupported split {split!r} for {image}')
        items.append(Item(source, root, split, image, label, sha256_file(image), boxes))

    # Ensure all approved manifest rows have corresponding curated copies.
    expected = status_counts['verified_positive'] + status_counts['verified_negative']
    if len(items) != expected:
        raise RuntimeError(f'{source}: found {len(items)} curated approved images but manifest expects {expected}')
    return items, summary


def find_optional(items: List[Item], source: str, split: str, stem: str):
    matches = [x for x in items if x.source == source and x.split == split and x.image.stem == stem]
    if len(matches) > 1:
        raise RuntimeError(f'Expected at most one curated item for {source}/{split}/{stem}, found {len(matches)}')
    return matches[0] if matches else None


def item_key(item: Item) -> str:
    return f'{item.source}:{item.split}:{item.image.name}'


def _matches_duplicate_alias(item: Item, source: str, split: str, stem: str) -> bool:
    return item.source == source and item.split == split and item.image.stem == stem


def make_plan(v1_items: List[Item], v2_items: List[Item]):
    all_items = v1_items + v2_items
    by_hash: Dict[str, List[Item]] = defaultdict(list)
    for item in all_items:
        by_hash[item.sha256].append(item)

    pair_specs = []
    handled_hashes = set()
    for spec in DUPLICATE_PAIRS:
        # The keeper and annotation donor must exist in curated data.
        keeper_alias = (
            (spec['v1_split'], spec['v1_stem'])
            if spec['keep_source'] == 'v1'
            else (spec['v2_split'], spec['v2_stem'])
        )
        donor_alias = (
            (spec['v1_split'], spec['v1_stem'])
            if spec['label_source'] == 'v1'
            else (spec['v2_split'], spec['v2_stem'])
        )
        keeper = find_optional(all_items, spec['keep_source'], keeper_alias[0], keeper_alias[1])
        donor = find_optional(all_items, spec['label_source'], donor_alias[0], donor_alias[1])
        if keeper is None:
            raise RuntimeError(
                f"Group {spec['group']}: required keeper is missing from curated data; "
                f"expected {spec['keep_source']}/{keeper_alias[0]}/{keeper_alias[1]}"
            )
        if donor is None:
            raise RuntimeError(
                f"Group {spec['group']}: required annotation donor is missing from curated data; "
                f"expected {spec['label_source']}/{donor_alias[0]}/{donor_alias[1]}"
            )

        # Resolve the duplicate group using the keeper's SHA-256. This is more
        # robust than relying only on a second curated manifest row: a copy can
        # be physically present and approved even when its alias lookup differs
        # (e.g. manifest path normalization), or it can be absent after QA.
        hash_group = by_hash[keeper.sha256]
        # Both source aliases are explicitly allowed, whether or not both
        # copies were exported to curated during manual review. A lower-priority
        # duplicate may be present in curated (approved) or absent (rejected / not
        # exported); either case is valid as long as no third alias shares the hash.
        expected_aliases = {
            ('v1', spec['v1_split'], spec['v1_stem']),
            ('v2', spec['v2_split'], spec['v2_stem']),
        }
        actual_aliases = {
            (x.source, x.split, x.image.stem)
            for x in hash_group
        }
        unexpected_aliases = actual_aliases.difference(expected_aliases)
        if unexpected_aliases:
            raise RuntimeError(
                f"Group {spec['group']}: unexpected curated item(s) share keeper SHA-256: "
                f"{sorted(unexpected_aliases)}"
            )
        if not hash_group or len(hash_group) > 2:
            raise RuntimeError(
                f"Group {spec['group']}: expected one or two curated copies for this exact duplicate, "
                f"found {len(hash_group)}"
            )
        if not any(x is keeper for x in hash_group):
            raise RuntimeError(f"Group {spec['group']}: keeper is not in its own SHA-256 group")

        # If a named member is present in curated data, it must have the same
        # SHA-256 as the keeper. If it is absent, it is not reintroduced.
        a = find_optional(all_items, 'v1', spec['v1_split'], spec['v1_stem'])
        b = find_optional(all_items, 'v2', spec['v2_split'], spec['v2_stem'])
        for candidate, alias_name in ((a, 'V1'), (b, 'V2')):
            if candidate is not None and candidate.sha256 != keeper.sha256:
                raise RuntimeError(
                    f"Group {spec['group']}: {alias_name} duplicate candidate no longer matches keeper SHA-256"
                )

        present_members = list(hash_group)
        missing_members = []
        if not any(x.source == 'v1' and x.split == spec['v1_split'] and x.image.stem == spec['v1_stem'] for x in present_members):
            missing_members.append(f"v1/{spec['v1_split']}/{spec['v1_stem']}")
        if not any(x.source == 'v2' and x.split == spec['v2_split'] and x.image.stem == spec['v2_stem'] for x in present_members):
            missing_members.append(f"v2/{spec['v2_split']}/{spec['v2_stem']}")

        # The pre-QA cross-source hash scan established the identity of each
        # known pair. Only the selected keeper and label donor are mandatory.
        if not any(item_key(x) == item_key(keeper) for x in present_members):
            raise RuntimeError(f"Group {spec['group']}: keeper missing from duplicate group")
        if donor.sha256 != keeper.sha256:
            raise RuntimeError(f"Group {spec['group']}: label donor is not the same image as keeper (SHA mismatch)")

        if keeper_hash := keeper.sha256:
            if keeper_hash in handled_hashes:
                raise RuntimeError(f"Duplicate SHA-256 assigned more than once: {keeper_hash}")
            handled_hashes.add(keeper_hash)

        if keeper.split != spec['keep_split']:
            raise RuntimeError(f"Group {spec['group']}: keeper split mismatch ({keeper.split} != {spec['keep_split']})")
        pair_specs.append((spec, keeper, donor, a, b, present_members, missing_members))

    unexpected = {h: xs for h, xs in by_hash.items() if len(xs) > 1 and h not in handled_hashes}
    if unexpected:
        details = {h[:12]: [item_key(x) for x in xs] for h, xs in unexpected.items()}
        raise RuntimeError('Unexpected exact duplicates found. Review these before merging: ' + json.dumps(details, ensure_ascii=False))

    duplicate_members = {
        item_key(x)
        for _spec, _keeper, _donor, _a, _b, present_members, _missing in pair_specs
        for x in present_members
    }
    selected = []
    for item in all_items:
        if item_key(item) not in duplicate_members:
            selected.append((item, item))  # image source, label source
    for _spec, keeper, donor, _a, _b, _present, _missing in pair_specs:
        selected.append((keeper, donor))

    # Preserve stable ordering: split, then image name.
    split_order = {'train': 0, 'val': 1, 'test': 2}
    selected.sort(key=lambda pair: (split_order[pair[0].split], pair[0].image.name.lower()))

    # Validate output-path uniqueness; add source prefix only on a name collision.
    path_counts = Counter((img_item.split, img_item.image.stem.lower()) for img_item, _ in selected)
    return selected, pair_specs, path_counts

def make_target_name(image_item: Item, collision: bool) -> str:
    return f'{image_item.source}__{image_item.image.name}' if collision else image_item.image.name


def build_or_report(args, v1_items, v2_items, v1_summary, v2_summary):
    selected, pair_specs, path_counts = make_plan(v1_items, v2_items)
    split_counts = Counter(img.split for img, _ in selected)
    object_counts = Counter()
    image_status_counts = Counter()
    for img, label in selected:
        object_counts[img.split] += len(label.boxes)
        image_status_counts['verified_negative' if len(label.boxes) == 0 else 'verified_positive'] += 1

    report = {
        'schema_version': 1,
        'sources': {
            'v1': {'root': str(args.v1.resolve()), 'review_summary': v1_summary},
            'v2': {'root': str(args.v2.resolve()), 'review_summary': v2_summary},
        },
        'duplicate_policy': 'exact SHA-256 only; split priority Test > Validation > Train; use verified V1 annotation for the five known cross-source duplicates',
        'known_duplicate_groups': [],
        'unexpected_exact_duplicates': 0,
        'selected_total_images': len(selected),
        'selected_images_by_split': dict(split_counts),
        'selected_objects_by_split': dict(object_counts),
        'selected_positive_images': image_status_counts['verified_positive'],
        'selected_negative_images': image_status_counts['verified_negative'],
        'dry_run': bool(args.dry_run),
        'output': str(args.output.resolve()),
    }
    for spec, keeper, donor, a, b, present_members, missing_members in pair_specs:
        report['known_duplicate_groups'].append({
            'group': spec['group'],
            'sha256': keeper.sha256,
            'members_present_in_curated': [item_key(x) for x in present_members],
            'members_absent_from_curated': missing_members,
            'kept_image': item_key(keeper),
            'label_donor': item_key(donor),
            'kept_split': keeper.split,
            'reason': 'split priority and verified V1 annotation provenance; an absent lower-priority duplicate is not reintroduced',
        })

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')

    print('\n=== V3 MERGE PLAN ===')
    print(f"V1 approved images: {v1_summary['approved_total']} (verified positive={v1_summary['verified_positive']}, negative={v1_summary['verified_negative']}, rejected={v1_summary['rejected']})")
    print(f"V2 approved images: {v2_summary['approved_total']} (verified positive={v2_summary['verified_positive']}, negative={v2_summary['verified_negative']}, rejected={v2_summary['rejected']})")
    print(f"Known exact duplicate groups resolved: {len(pair_specs)}")
    for spec, keeper, donor, a, b, present_members, missing_members in pair_specs:
        missing_note = f"; absent from curated={missing_members}" if missing_members else ''
        print(f"  Group {spec['group']}: keep {item_key(keeper)}; annotation from {item_key(donor)}; SHA={keeper.sha256[:12]}{missing_note}")
    print(f"V3 images: {len(selected)}")
    print(f"Split counts: {dict(split_counts)}")
    print(f"Object counts: {dict(object_counts)}")
    print(f"Positive images: {image_status_counts['verified_positive']} | negative images: {image_status_counts['verified_negative']}")
    if args.report:
        print(f"Plan report saved: {args.report.resolve()}")

    if args.dry_run:
        print('\nDRY RUN ONLY: no V3 image/label files were copied.')
        return

    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Output directory is not empty: {out}. Choose a new output path or move it aside.')
    out.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for img_item, label_item in selected:
        split = img_item.split
        collision = path_counts[(split, img_item.image.stem.lower())] > 1
        target_name = make_target_name(img_item, collision)
        target_image = out / 'images' / split / target_name
        target_label = out / 'labels' / split / (Path(target_name).stem + '.txt')
        target_image.parent.mkdir(parents=True, exist_ok=True)
        target_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_item.image, target_image)
        shutil.copy2(label_item.label, target_label)
        # Re-read copied label and ensure its format remains valid.
        copied_boxes = read_boxes(target_label)
        if copied_boxes != label_item.boxes:
            raise RuntimeError(f'Copied label verification failed: {target_label}')
        manifest_rows.append({
            'split': split,
            'image': target_image.relative_to(out).as_posix(),
            'label': target_label.relative_to(out).as_posix(),
            'image_source': item_key(img_item),
            'label_source': item_key(label_item),
            'sha256': img_item.sha256,
            'object_count': len(copied_boxes),
            'is_negative': str(len(copied_boxes) == 0).lower(),
        })

    with (out / 'merge_manifest.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    yaml_text = (
        'path: .\n'
        'train: images/train\n'
        'val: images/val\n'
        'test: images/test\n'
        'nc: 1\n'
        "names: ['lab_coat']\n"
    )
    (out / 'data.yaml').write_text(yaml_text, encoding='utf-8')
    final_report = dict(report)
    final_report['dry_run'] = False
    final_report['dataset_yaml'] = str((out / 'data.yaml').resolve())
    final_report['manifest'] = str((out / 'merge_manifest.csv').resolve())
    (out / 'merge_summary.json').write_text(json.dumps(final_report, indent=2, ensure_ascii=False), encoding='utf-8')

    # Final structural and cross-split leakage check.
    final_hashes: Dict[str, List[str]] = defaultdict(list)
    final_images = 0
    for split in ('train', 'val', 'test'):
        image_dir = out / 'images' / split
        label_dir = out / 'labels' / split
        for image_path in image_dir.glob('*'):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTS:
                final_images += 1
                label_path = label_dir / f'{image_path.stem}.txt'
                if not label_path.is_file():
                    raise RuntimeError(f'Final dataset missing label: {label_path}')
                final_hashes[sha256_file(image_path)].append(f'{split}/{image_path.name}')
    leaks = {h[:12]: paths for h, paths in final_hashes.items() if len(paths) > 1}
    if leaks:
        raise RuntimeError('Final V3 has exact duplicate image leakage: ' + json.dumps(leaks, ensure_ascii=False))
    if final_images != len(selected):
        raise RuntimeError(f'Final V3 image count mismatch: expected {len(selected)}, got {final_images}')

    print(f'\nV3 successfully created: {out}')
    print(f'Final image count: {final_images}; exact duplicate leakage: none')
    print(f'Dataset YAML: {out / "data.yaml"}')


def parse_args():
    p = argparse.ArgumentParser(description='Merge verified lab-coat YOLO datasets V1 and V2.')
    p.add_argument('--v1', type=Path, required=True, help='Curated V1 root')
    p.add_argument('--v2', type=Path, required=True, help='Curated V2 root')
    p.add_argument('--output', type=Path, required=True, help='New V3 destination')
    p.add_argument('--report', type=Path, help='Optional JSON merge-plan report')
    p.add_argument('--dry-run', action='store_true', help='Validate and print plan without copying V3 files')
    return p.parse_args()


def main():
    print(f'MERGE SCRIPT VERSION: {SCRIPT_VERSION}')
    args = parse_args()
    try:
        v1_items, v1_summary = read_reviewed_dataset('v1', args.v1)
        v2_items, v2_summary = read_reviewed_dataset('v2', args.v2)
        build_or_report(args, v1_items, v2_items, v1_summary, v2_summary)
        return 0
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 2

if __name__ == '__main__':
    raise SystemExit(main())
