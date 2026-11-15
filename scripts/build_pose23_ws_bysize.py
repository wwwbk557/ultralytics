#!/usr/bin/env python3
"""Scheme-C 14-class view with ws split by target size.

Rows come from ``schemeC_pose23_full_v5`` (white / 5 / unknown labels already
dropped, visibility=0 objects removed), so the only rules added here are:

* every non-ws row (pose + source_json) goes to train;
* a ws capture session goes to train when the median of its per-image "largest
  object long side" is >= 24 px on the 480x640 canvas, otherwise to val;
* ws objects with a long side < 12 px are dropped (P3 cannot learn them) and ws
  images left with no object are removed from the dataset.

Sizes are measured after the letterbox resize to the training canvas
(``s = min(480 / H, 640 / W)``).  Non-ws rows keep every object.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image

ROOT = Path("/home/wbk/ultralytics_2027")
SRC = ROOT / "datasets/schemeC_pose23_full_v5"
OUT = ROOT / "datasets/schemeC_pose23_ws_bysize"
NAMES = [
    "blue_sentry", "blue_hero", "blue_engineer", "blue_3", "blue_4", "blue_outpost", "blue_base",
    "red_sentry", "red_hero", "red_engineer", "red_3", "red_4", "red_outpost", "red_base",
]
CANVAS = (480, 640)
WS_TRAIN_MIN_PX = 24.0  # session median of the per-image largest object
MIN_OBJ_PX = 12.0  # per-object floor on ws rows


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=str(SRC))
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--ws-train-px", type=float, default=WS_TRAIN_MIN_PX)
    p.add_argument("--min-obj-px", type=float, default=MIN_OBJ_PX)
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def is_ws(unit: str) -> bool:
    return unit.startswith("workspaces/")


def object_long_sides(label_path: Path, width: int, height: int, scale: float) -> list[float]:
    """Long side of every 14-class box on the 480x640 canvas."""
    sides = []
    for line in label_path.read_text().splitlines():
        f = line.split()
        if len(f) != 13:
            continue
        sides.append(max(float(f[3]) * width, float(f[4]) * height) * scale)
    return sides


def main():
    a = parse()
    src, out = Path(a.src), Path(a.out)
    rows = list(csv.DictReader((src / "split_manifest.tsv").open(), delimiter="\t"))

    # Pass 1: ws sizes per image, then one decision per capture session.
    ws_max: dict[int, float] = {}
    scale_of: dict[int, float] = {}
    for i, r in enumerate(rows):
        if not is_ws(r["unit"]):
            continue
        img = src / "images" / r["split"] / r["image"]
        with Image.open(img) as im:
            w, h = im.size
        scale = min(CANVAS[0] / h, CANVAS[1] / w)
        scale_of[i] = scale
        sides = object_long_sides(src / "labels" / r["split"] / r["label"], w, h, scale)
        if sides:
            ws_max[i] = max(sides)
    per_session = defaultdict(list)
    for i, m in ws_max.items():
        per_session[rows[i]["unit"]].append(m)
    session_train = {}
    for unit, vals in per_session.items():
        vals.sort()
        med = vals[len(vals) // 2] if len(vals) % 2 else 0.5 * (vals[len(vals) // 2 - 1] + vals[len(vals) // 2])
        session_train[unit] = med >= a.ws_train_px

    # Pass 2: split + label filter.
    plan = []
    dropped_obj = Counter()
    dropped_img = Counter()
    for i, r in enumerate(rows):
        unit = r["unit"]
        if is_ws(unit):
            split = "train" if session_train[unit] else "val"
            img = src / "images" / r["split"] / r["image"]
            with Image.open(img) as im:
                w, h = im.size
            scale = min(CANVAS[0] / h, CANVAS[1] / w)
            kept = []
            for line in (src / "labels" / r["split"] / r["label"]).read_text().splitlines():
                f = line.split()
                if len(f) != 13:
                    continue
                if max(float(f[3]) * w, float(f[4]) * h) * scale < a.min_obj_px:
                    dropped_obj["small_ws_object"] += 1
                    continue
                kept.append(line)
            if not kept:
                dropped_img["ws_image_without_object"] += 1
                continue
            plan.append((r, split, kept))
        else:
            text = (src / "labels" / r["split"] / r["label"]).read_text()
            plan.append((r, "train", [ln for ln in text.splitlines() if ln.strip()]))

    stats = {
        "source": str(src), "output": str(out),
        "rule": "non-ws -> train; ws session median long side >= "
                f"{a.ws_train_px:g}px -> train else val; ws objects < {a.min_obj_px:g}px dropped; "
                "ws images left empty dropped",
        "canvas": list(CANVAS),
        "ws_sessions": {"train": sum(session_train.values()), "val": sum(not v for v in session_train.values())},
        "images": Counter(sp for _, sp, _ in plan),
        "dropped": dict(dropped_obj + dropped_img),
        "objects": Counter(),
        "classes": {sp: Counter() for sp in ("train", "val")},
    }
    for r, sp, kept in plan:
        for line in kept:
            c = int(line.split()[0])
            stats["objects"][sp] += 1
            stats["classes"][sp][c] += 1
    stats["images"] = dict(stats["images"])
    stats["objects"] = dict(stats["objects"])
    stats["classes"] = {
        sp: {NAMES[i]: c[i] for i in range(14)} for sp, c in stats["classes"].items()
    }
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    if a.dry_run:
        return

    if out.exists():
        if not a.force:
            raise SystemExit(f"{out} exists; use --force")
        shutil.rmtree(out)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    manifest = []
    for r, sp, kept in plan:
        img = src / "images" / r["split"] / r["image"]
        dst_img = out / "images" / sp / r["image"]
        os.symlink(os.path.relpath(img.resolve(), dst_img.parent), dst_img)
        (out / "labels" / sp / r["label"]).write_text("\n".join(kept) + "\n")
        manifest.append((r["source_rel"], r["unit"], r["split"], sp, r["image"], r["label"], len(kept)))
    data = {
        "path": str(out.resolve()), "train": "images/train", "val": "images/val",
        "kpt_shape": [4, 2], "flip_idx": [1, 0, 3, 2], "nc": 14,
        "names": {i: n for i, n in enumerate(NAMES)},
    }
    (out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    (out / "classes.txt").write_text("\n".join(NAMES) + "\n")
    with (out / "split_manifest.tsv").open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["source_rel", "unit", "source_split", "split", "image", "label", "objects"])
        w.writerows(manifest)
    (out / "summary.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    (out / "README.md").write_text(
        "# Scheme-C 14-class view: ws split by target size\n\n"
        f"Source rows: `{src.name}` (white / 5 / unknown labels already dropped).\n\n"
        "Every non-ws row (pose train+val, source_json) goes to train.\n"
        "A ws capture session goes to train when the median of its per-image largest\n"
        f"object (long side on {CANVAS[0]}x{CANVAS[1]}) is >= {a.ws_train_px:g} px, else to val.\n"
        f"ws objects with a long side < {a.min_obj_px:g} px are dropped and ws images left\n"
        "without objects are removed. Non-ws rows keep every object.\n"
    )
    print("wrote", out)


if __name__ == "__main__":
    main()
