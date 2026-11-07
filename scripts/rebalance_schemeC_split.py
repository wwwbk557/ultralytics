#!/usr/bin/env python3
"""Rebuild the Scheme-C split with a leakage-safe blue_sentry rebalance.

The current split puts almost all small/hard blue_sentry clips in val (median
15x10 px) while train keeps the large lab shots (median 35x15 px), so the class
is evaluated on a domain it barely sees. This script moves **whole capture roots**
between train and val, never individual frames, so a scene can not appear on both
sides.

A root merges everything that can leak into each other: the trailing segment of a
`drone_..._2` id is dropped (`drone_20260801_05_01_1` + `..._2` -> one root), and
`rmcv:<seq>` / `own_*` groups stay as they are. Roots already split across
train/val are consolidated on one side.

Output is a new symlink tree (the source dataset is untouched):
    datasets/schemeC_split_bal/{images,labels}/{train,val} + data.yaml

Usage:
    python scripts/rebalance_schemeC_split.py --src datasets/schemeC_split \
        --out datasets/schemeC_split_bal --small-px 20 --val-ratio 0.18
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path("/home/wbk/ultralytics_2027")
TARGET_CLS = 0  # blue_sentry


def root_of(group: str) -> str:
    """Everything that can leak into everything else in the same capture.

    `drone_20260801_05_01_1/2` are segments of one session and must move together;
    `xjtu_0500` chunks and `rmcv:<seq>` / `own_*` groups are already leakage units.
    """
    if group.startswith(("rmcv:", "own_", "xjtu_")):
        return group
    m = re.match(r"^(.*)_(\d+)$", group)
    return m.group(1) if m else group


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--src", default=str(ROOT / "datasets/schemeC_split"))
    p.add_argument("--out", default=str(ROOT / "datasets/schemeC_split_bal"))
    p.add_argument("--small-px", type=float, default=20.0, help="blue_sentry height threshold for 'hard/small'")
    p.add_argument("--val-ratio", type=float, default=0.18, help="max fraction of images allowed in val")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def label_sizes(path: Path, ref=(480, 640)) -> list[tuple[int, float]]:
    out = []
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        p = line.split()
        if len(p) == 13:
            out.append((int(p[0]), float(p[3]) * ref[0]))  # (cls, height px)
    return out


def main() -> None:
    a = parse()
    src, out = Path(a.src), Path(a.out)
    if out.exists():
        if not a.force:
            raise SystemExit(f"output exists: {out}; use --force")
        shutil.rmtree(out)
    rows = list(csv.DictReader((src / "split_manifest.tsv").open(), delimiter="\t"))

    roots: dict[str, dict] = defaultdict(lambda: {"split": set(), "rows": [], "small": 0, "b0": 0})
    for r in rows:
        rt = roots[root_of(r["group"])]
        rt["split"].add(r["split"])
        rt["rows"].append(r)
        for cls, h in label_sizes(src / "labels" / r["split"] / r["label"]):
            if cls == TARGET_CLS:
                rt["b0"] += 1
                rt["small"] += int(h < a.small_px)

    # ---- current stats, counted per row (never per root) -------------------
    def stats() -> tuple[int, int, int]:
        v = sum(1 for r in rows if r["split"] == "val")
        vs = sum(r["_small"] for r in rows if r["split"] == "val")
        ts = sum(r["_small"] for r in rows if r["split"] == "train")
        return v, vs, ts

    for r in rows:  # cache per-row small flag (cheap, avoids re-reading labels)
        r["_small"] = sum(1 for cls, h in label_sizes(src / "labels" / r["split"] / r["label"])
                          if cls == TARGET_CLS and h < a.small_px)

    val_imgs, val_small, train_small = stats()
    total_imgs = len(rows)
    print(f"before: val_small={val_small} train_small={train_small} "
          f"val_imgs={val_imgs}/{total_imgs} ({100 * val_imgs / total_imgs:.1f}%)")

    # ---- consolidate pre-existing session leakage -------------------------
    by_root = defaultdict(list)
    for r in rows:
        by_root[root_of(r["group"])].append(r)
    consolidated = []
    for rt, rs in by_root.items():
        n_val = sum(1 for r in rs if r["split"] == "val")
        n_train = len(rs) - n_val
        if n_val and n_train:
            keep = "val" if n_val >= n_train else "train"
            for r in rs:
                r["_new_split"] = keep
            consolidated.append((rt, n_train, n_val, keep))
    for r in rows:
        r.setdefault("_new_split", r["split"])
    print(f"pre-existing session overlap consolidated: {len(consolidated)} roots")
    for rt, nt, nv, keep in sorted(consolidated):
        print(f"  {rt:<40} train={nt:>4} val={nv:>4} -> {keep}")

    # ---- rebalance the small/hard blue_sentry ------------------------------
    v_imgs = sum(1 for r in rows if r["_new_split"] == "val")
    v_small = sum(r["_small"] for r in rows if r["_new_split"] == "val")
    t_small = sum(r["_small"] for r in rows if r["_new_split"] == "train")
    target = (v_small + t_small) // 2
    budget = int(total_imgs * a.val_ratio)
    print(f"\nafter consolidation: val_small={v_small} train_small={t_small} target={target} "
          f"val_imgs={v_imgs} budget={budget}")

    movable = []
    for rt, rs in by_root.items():
        splits = {r["_new_split"] for r in rs}
        small = sum(r["_small"] for r in rs)
        if splits == {"train"} and small and len(rs) <= 250:
            movable.append((rt, small, len(rs)))
    movable.sort(key=lambda x: (-x[1], x[2]))
    print(f"{len(movable)} movable train roots with small sentries")
    moved = []
    for rt, small, n in movable:
        if v_small >= target - 6:
            break
        if v_imgs + n > budget or v_small + small > target + 10:
            continue
        for r in by_root[rt]:
            r["_new_split"] = "val"
        v_small += small
        t_small -= small
        v_imgs += n
        moved.append((rt, small, n))
    print(f"after:  val_small={v_small} train_small={t_small} val_imgs={v_imgs}/{total_imgs} "
          f"({100 * v_imgs / total_imgs:.1f}%)")
    for rt, small, n in moved:
        print(f"  moved train->val: {rt:<40} small={small:>3} imgs={n:>4}")

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    manifest = ["source\tstem\tgroup\troot\tsplit\timage\tlabel\tout_stem"]
    cls_count = Counter()
    for rt, v in roots.items():
        for r in v["rows"]:
            split = r["_new_split"]
            img_link = out / "images" / split / r["image"]
            lab_link = out / "labels" / split / r["label"]
            os.symlink(os.path.relpath((src / "images" / r["split"] / r["image"]).resolve(), img_link.parent), img_link)
            os.symlink(os.path.relpath((src / "labels" / r["split"] / r["label"]).resolve(), lab_link.parent), lab_link)
            for c in (src / "labels" / r["split"] / r["label"]).read_text().splitlines():
                if c.strip():
                    cls_count[(split, int(c.split()[0]))] += 1
            manifest.append(
                f"{r['source']}\t{r['stem']}\t{r['group']}\t{rt}\t{split}\t{r['image']}\t{r['label']}\t{r['out_stem']}"
            )
    (out / "split_manifest.tsv").write_text("\n".join(manifest) + "\n")

    y = yaml.safe_load((src / "data.yaml").read_text())
    y["path"] = str(out.resolve())
    y["train"] = "images/train"
    y["val"] = "images/val"
    (out / "data.yaml").write_text(yaml.safe_dump(y, sort_keys=False))
    shutil.copy2(src / "classes.txt", out / "classes.txt")

    print(f"\nwrote {out} (symlinks only; clean_schemeC untouched)")
    for split in ("train", "val"):
        n = sum(1 for _ in (out / "images" / split).iterdir())
        print(f"  {split}: {n} images, {sum(v for (s, c), v in cls_count.items() if s == split)} instances")


if __name__ == "__main__":
    main()
