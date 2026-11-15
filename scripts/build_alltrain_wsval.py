#!/usr/bin/env python3
"""All clean full data to train; ws1/2/3/5 as a sanity val.

The user does not use val for model selection/curves, so ws1/2/3/5 are kept in
train and also linked as val (overlap is intentional). This gives one final
sanity check rather than a leakage-free holdout.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path("/home/wbk/ultralytics_2027")
SRC = ROOT / "datasets/schemeC_pose23_full_v5"
OUT = ROOT / "datasets/schemeC_pose23_alltrain_wsval"
NAMES = [
    "blue_sentry", "blue_hero", "blue_engineer", "blue_3", "blue_4", "blue_outpost", "blue_base",
    "red_sentry", "red_hero", "red_engineer", "red_3", "red_4", "red_outpost", "red_base",
]
WS_UNITS = ("workspaces/ws1/", "workspaces/ws2/", "workspaces/ws3/", "workspaces/ws5/")


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--force", action="store_true")
    a.add_argument("--dry-run", action="store_true")
    args = a.parse_args()
    rows = list(csv.DictReader((SRC / "split_manifest.tsv").open(), delimiter="\t"))
    if OUT.exists():
        if not args.force:
            raise SystemExit(f"{OUT} exists; use --force")
        shutil.rmtree(OUT)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    c_train = Counter(); c_val = Counter()
    train = 0; val = 0
    for r in rows:
        img = SRC / "images" / r["split"] / r["image"]
        lab = SRC / "labels" / r["split"] / r["label"]
        # all clean rows -> train
        dst = OUT / "images" / "train" / r["image"]
        os.symlink(os.path.relpath(img.resolve(), dst.parent), dst)
        shutil.copy2(lab, OUT / "labels" / "train" / r["label"])
        train += 1
        for line in lab.read_text().splitlines():
            if line.strip():
                c_train[int(line.split()[0])] += 1
        # ws1/2/3/5 -> sanity val (also still in train)
        if any(r["unit"].startswith(u) for u in WS_UNITS):
            dstv = OUT / "images" / "val" / r["image"]
            os.symlink(os.path.relpath(img.resolve(), dstv.parent), dstv)
            shutil.copy2(lab, OUT / "labels" / "val" / r["label"])
            val += 1
            for line in lab.read_text().splitlines():
                if line.strip():
                    c_val[int(line.split()[0])] += 1
    data = {
        "path": str(OUT.resolve()), "train": "images/train", "val": "images/val",
        "kpt_shape": [4, 2], "flip_idx": [1, 0, 3, 2], "nc": 14,
        "names": {i: n for i, n in enumerate(NAMES)},
    }
    (OUT / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    (OUT / "classes.txt").write_text("\n".join(NAMES) + "\n")
    summary = {
        "source": str(SRC), "output": str(OUT),
        "rule": "all clean full rows -> train; ws1/ws2/ws3/ws5 also -> val (intentional overlap; no val curves)",
        "images": {"train": train, "val": val},
        "objects": {"train": sum(c_train.values()), "val": sum(c_val.values())},
        "classes": {"train": {NAMES[i]: c_train[i] for i in range(14)},
                    "val": {NAMES[i]: c_val[i] for i in range(14)}},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (OUT / "README.md").write_text(
        "# All-train + ws sanity val\n\n"
        "All clean 14-class rows from schemeC_pose23_full_v5 go to train.\n"
        "ws1/ws2/ws3/ws5 are additionally linked to val. They also remain in train;\n"
        "this is intentional because val is only a final sanity check, not used for\n"
        "model selection or curves. Do not report val as a clean holdout.\n"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
