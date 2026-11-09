#!/usr/bin/env python3
"""Cap the validation side of a Scheme-C split without touching train.

The current ``schemeC_split_bal`` val is dominated by two red-outpost capture
groups (1258 of 3985 images), so ``best.pt`` is selected on a metric where
outpost is ~38% of the set.  This script keeps every train row and keeps at
most ``--max-per-group`` images from each val capture group.  The discarded
frames are **not** moved to train, so the group-level train/val separation
stays leakage-free.

The output is a new symlink tree; the source split is never modified.

Usage:
    python scripts/cap_schemeC_val.py \
        --src datasets/schemeC_split_bal \
        --out datasets/schemeC_split_bal_capval \
        --max-per-group 120
"""
from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path("/home/wbk/ultralytics_2027")
CLEAN = ROOT / "datasets/clean_schemeC"


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--src", default=str(ROOT / "datasets/schemeC_split_bal"))
    p.add_argument("--out", default=str(ROOT / "datasets/schemeC_split_bal_capval"))
    p.add_argument("--max-per-group", type=int, default=120, help="max val images kept per capture group")
    p.add_argument("--min-per-class", type=int, default=40, help="warn when a val class has fewer instances")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def read_rows(src: Path) -> tuple[list[dict], list[str]]:
    manifest = src / "split_manifest.tsv"
    with manifest.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
        header = reader.fieldnames or []
    if not rows:
        raise SystemExit(f"no rows in {manifest}")
    for key in ("group", "split", "image", "label"):
        if key not in header:
            raise SystemExit(f"{manifest} is missing the {key!r} column")
    return rows, header


def class_counts(rows: list[dict], src: Path) -> Counter:
    counts = Counter()
    for row in rows:
        label = src / "labels" / row["split"] / row["label"]
        if not label.is_file():
            continue
        for line in label.read_text().splitlines():
            if line.strip():
                counts[int(line.split()[0])] += 1
    return counts


def main() -> None:
    a = parse()
    src = Path(a.src).resolve()
    out = Path(a.out).resolve()
    if not (src / "split_manifest.tsv").is_file():
        raise SystemExit(f"missing {src}/split_manifest.tsv")
    if CLEAN == out or CLEAN in out.parents:
        raise SystemExit("refusing to write inside clean_schemeC")
    if out.exists():
        if not a.force:
            raise SystemExit(f"output exists: {out}; use --force to rebuild")
        if src in out.parents or out in src.parents:
            raise SystemExit("refusing to delete a path related to --src")
        shutil.rmtree(out)

    rows, header = read_rows(src)
    by_group: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["split"] == "val":
            by_group[row["group"]].append(row)

    rng = random.Random(a.seed)
    kept: list[dict] = []
    dropped = 0
    for group in sorted(by_group):
        group_rows = by_group[group]
        rng.shuffle(group_rows)
        keep = group_rows[: max(1, a.max_per_group)]
        dropped += len(group_rows) - len(keep)
        kept.extend(keep)
    kept.extend(row for row in rows if row["split"] == "train")

    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    for row in kept:
        split = row["split"]
        src_img = src / "images" / split / row["image"]
        src_lab = src / "labels" / split / row["label"]
        if not src_img.exists() or not src_lab.is_file():
            raise SystemExit(f"missing source for {split}/{row['image']}")
        dst_img = out / "images" / split / row["image"]
        dst_lab = out / "labels" / split / row["label"]
        dst_img.symlink_to(src_img)
        dst_lab.symlink_to(src_lab)

    with (out / "split_manifest.tsv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header, delimiter="\t")
        writer.writeheader()
        writer.writerows(kept)

    data = yaml.safe_load((src / "data.yaml").read_text())
    data["path"] = str(out)
    (out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    shutil.copy2(src / "classes.txt", out / "classes.txt")

    names = data.get("names", [])
    train_rows = [r for r in kept if r["split"] == "train"]
    val_rows = [r for r in kept if r["split"] == "val"]
    train_counts = class_counts(train_rows, out)
    val_counts = class_counts(val_rows, out)
    print(f"[cap_val] src={src}")
    print(f"[cap_val] out={out}")
    print(f"[cap_val] val groups={len(by_group)} kept={len(val_rows)} dropped={dropped} "
          f"max_per_group={a.max_per_group}")
    print(f"[cap_val] train images={len(train_rows)} val images={len(val_rows)}")
    print("[cap_val] val instances per class:")
    items = names.items() if isinstance(names, dict) else enumerate(names)
    for index, name in items:
        index = int(index)
        flag = "  <-- low" if val_counts[index] < a.min_per_class else ""
        print(f"  {index:2d} {str(name):16s} val={val_counts[index]:5d}  train={train_counts[index]:6d}{flag}")
    if any(val_counts[i] < a.min_per_class for i in range(len(names))):
        print(f"[cap_val] WARNING: some val classes have fewer than {a.min_per_class} instances; "
              "move a whole capture group if that class matters")


if __name__ == "__main__":
    main()
