#!/usr/bin/env python3
"""Build train/val symlink trees over clean_schemeC, without touching the source.

The output contains only symlinks and a data.yaml.  Delete the output directory
at any time; clean_schemeC is never modified.

Examples:
  python scripts/build_schemeC_split.py --out datasets/schemeC_split \
      --sources rm_pose_old,rmcv_armor,ws5 --val-ratio 0.15
  python scripts/build_schemeC_split.py --out datasets/schemeC_split_all \
      --sources all --val-ratio 0.15
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path("/home/wbk/ultralytics_2027")
CLEAN = ROOT / "datasets/clean_schemeC"
ALL_SOURCES = ("ws1", "ws2", "ws3", "ws5", "rmcv_armor", "rm_pose_old")
CORE_SOURCES = ("rm_pose_old", "rmcv_armor", "ws5")
IMG_EXT = (".jpg", ".jpeg", ".png")


def group_for(source: str, stem: str, rmcv_seq: dict[str, str]) -> str:
    if source == "rmcv_armor":
        seq = rmcv_seq.get(stem)
        return f"rmcv:{seq}" if seq else f"rmcv:{stem.split('__')[0]}"
    if source == "rm_pose_old":
        if stem.startswith("xjtu_"):
            n = int(re.search(r"(\d+)$", stem).group(1))
            return f"xjtu_{n // 500 * 500:04d}"
        if stem.startswith("own_"):
            return re.sub(r"_\d+$", "", stem)
    if "__" in stem:
        return stem.split("__")[0]
    return re.sub(r"_\d+$", "", stem)


def load_rmcv_sequences() -> dict[str, str]:
    p = ROOT / "datasets/rmcv_armor_14cls_grouped/samples.tsv"
    out = {}
    if p.exists():
        for r in csv.DictReader(p.open(), delimiter="\t"):
            if r.get("file") and r.get("sequence"):
                out[r["file"]] = f"{r.get('source','?')}:{r['sequence']}"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sources", default="core", help="all, core, or comma list")
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.sources == "all":
        sources = list(ALL_SOURCES)
    elif args.sources == "core":
        sources = list(CORE_SOURCES)
    else:
        sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    bad = [s for s in sources if s not in ALL_SOURCES]
    if bad:
        raise SystemExit(f"unknown sources: {bad}")
    if not (0.05 <= args.val_ratio <= 0.4):
        raise SystemExit("val-ratio should be in [0.05, 0.4]")

    out = Path(args.out)
    if out.exists():
        if not args.force:
            raise SystemExit(f"output exists: {out}; use --force to rebuild")
        if CLEAN == out or CLEAN in out.parents:
            raise SystemExit("refusing to delete inside clean_schemeC")
        shutil.rmtree(out)
    (out / "images/train").mkdir(parents=True)
    (out / "images/val").mkdir(parents=True)
    (out / "labels/train").mkdir(parents=True)
    (out / "labels/val").mkdir(parents=True)

    rmcv_seq = load_rmcv_sequences()
    records = []
    for source in sources:
        sdir = CLEAN / source
        for img in sorted(p for p in (sdir / "images").glob("*") if p.suffix.lower() in IMG_EXT):
            lp = sdir / "labels" / f"{img.stem}.txt"
            if not lp.exists():
                continue
            records.append({
                "source": source,
                "stem": img.stem,
                "img": img,
                "label": lp,
                "group": group_for(source, img.stem, rmcv_seq),
            })
    if not records:
        raise SystemExit("no records")
    groups = defaultdict(list)
    for r in records:
        groups[r["group"]].append(r)

    names = [g for g in sorted(groups)]
    random.Random(args.seed).shuffle(names)
    target = int(round(len(records) * args.val_ratio))
    val_groups = set()
    val_n = 0
    for g in names:
        if val_n >= target:
            break
        val_groups.add(g)
        val_n += len(groups[g])
    if not val_groups:
        val_groups.add(names[0])

    # collision-safe output names
    used = set()
    for r in records:
        base = r["img"].name
        if base in used:
            base = f"{r['source']}__{base}"
        used.add(base)
        r["out_name"] = base
        r["out_stem"] = Path(base).stem
        r["split"] = "val" if r["group"] in val_groups else "train"

    manifest = ["source\tstem\tgroup\tsplit\timage\tlabel\tout_stem"]
    cls = Counter()
    for r in records:
        split = r["split"]
        img_link = out / "images" / split / r["out_name"]
        lab_link = out / "labels" / split / f"{r['out_stem']}.txt"
        img_target = os.path.relpath(r["img"], img_link.parent)
        lab_target = os.path.relpath(r["label"], lab_link.parent)
        os.symlink(img_target, img_link)
        os.symlink(lab_target, lab_link)
        for ln in r["label"].read_text().splitlines():
            if ln.strip():
                cls[int(ln.split()[0])] += 1
        manifest.append(
            f"{r['source']}\t{r['stem']}\t{r['group']}\t{split}\t{r['out_name']}\t{r['out_stem']}.txt\t{r['out_stem']}"
        )

    classes = [x.strip() for x in (CLEAN / "classes.txt").read_text().splitlines() if x.strip()]
    y = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "kpt_shape": [4, 2],
        "flip_idx": [1, 0, 3, 2],
        "nc": len(classes),
        "names": {i: n for i, n in enumerate(classes)},
    }
    (out / "data.yaml").write_text(yaml.safe_dump(y, sort_keys=False, allow_unicode=True))
    (out / "split_manifest.tsv").write_text("\n".join(manifest) + "\n")
    (out / "classes.txt").write_text("\n".join(classes) + "\n")

    tr = sum(1 for r in records if r["split"] == "train")
    va = len(records) - tr
    ov = {r["group"] for r in records if r["split"] == "train"} & val_groups
    print(f"out={out}")
    print(f"sources={sources}")
    print(f"images train={tr} val={va} total={len(records)}")
    print(f"groups train={len(groups)-len(val_groups)} val={len(val_groups)} overlap={len(ov)}")
    print(f"objects={sum(cls.values())} classes={dict(sorted(cls.items()))}")
    print(f"wrote {out/'data.yaml'}")


if __name__ == "__main__":
    main()
