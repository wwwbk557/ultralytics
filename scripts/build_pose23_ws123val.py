#!/usr/bin/env python3
"""Scheme-C 14-class view with ws1/ws2/ws3 as val.

Rows come from ``schemeC_pose23_full_v5`` (white / 5 / unknown labels already
dropped, visibility=0 objects removed).  The split is unit-level and
exhaustive:

* every row whose unit is under ``workspaces/ws1|ws2|ws3`` goes to val;
* every other row (pose, source_json, ws5, other ws sessions) goes to train.

Images are relative symlinks back into ``pose23_current``; labels are copied.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path("/home/wbk/ultralytics_2027")
SRC = ROOT / "datasets/schemeC_pose23_full_v5"
OUT = ROOT / "datasets/schemeC_pose23_ws123val"
VAL_UNITS = ("workspaces/ws1/", "workspaces/ws2/", "workspaces/ws3/")
NAMES = [
    "blue_sentry", "blue_hero", "blue_engineer", "blue_3", "blue_4", "blue_outpost", "blue_base",
    "red_sentry", "red_hero", "red_engineer", "red_3", "red_4", "red_outpost", "red_base",
]


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=str(SRC))
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main():
    a = parse()
    src, out = Path(a.src), Path(a.out)
    rows = list(__import__("csv").DictReader((src / "split_manifest.tsv").open(), delimiter="\t"))
    if out.exists():
        if a.dry_run:
            pass
        elif a.force:
            shutil.rmtree(out)
        else:
            raise SystemExit(f"{out} exists; use --force")
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        if not a.dry_run:
            (out / sub).mkdir(parents=True, exist_ok=True)
    c_train, c_val = Counter(), Counter()
    n_train = n_val = 0
    for r in rows:
        is_val = r["unit"].startswith(VAL_UNITS)
        split = "val" if is_val else "train"
        img = src / "images" / r["split"] / r["image"]
        lab = src / "labels" / r["split"] / r["label"]
        dst_img = out / "images" / split / r["image"]
        if a.dry_run:
            pass
        else:
            os.symlink(os.path.relpath(img.resolve(), dst_img.parent), dst_img)
            shutil.copy2(lab, out / "labels" / split / r["label"])
        counts = c_val if is_val else c_train
        for line in lab.read_text().splitlines():
            if line.strip():
                counts[int(line.split()[0])] += 1
        if is_val:
            n_val += 1
        else:
            n_train += 1
    data = {
        "path": str(out.resolve()), "train": "images/train", "val": "images/val",
        "kpt_shape": [4, 2], "flip_idx": [1, 0, 3, 2], "nc": 14,
        "names": {i: n for i, n in enumerate(NAMES)},
    }
    summary = {
        "source": str(src),
        "output": str(out),
        "rule": "ws1/ws2/ws3 -> val only; all other units -> train",
        "images": {"train": n_train, "val": n_val},
        "objects": {"train": sum(c_train.values()), "val": sum(c_val.values())},
        "classes": {
            "train": {NAMES[i]: c_train[i] for i in range(14)},
            "val": {NAMES[i]: c_val[i] for i in range(14)},
        },
    }
    if not a.dry_run:
        (out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
        (out / "classes.txt").write_text("\n".join(NAMES) + "\n")
        (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        (out / "README.md").write_text(
            "# Scheme-C ws1/ws2/ws3 val\n\n"
            "14-class, kpt_shape [4,2]. Every ws1/ws2/ws3 row is val; every other\n"
            "unit (pose, source_json, ws5, other ws sessions) is train. Unit-level\n"
            "and exhaustive, no overlap. Images are relative symlinks into\n"
            "datasets/pose23_current; labels are real files.\n"
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
