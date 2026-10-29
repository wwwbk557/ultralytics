#!/usr/bin/env python3
"""Convert the auto-labelled YOLO pose txt files to Dextera JSON for browsing.

Dextera's data-annotation page reads Dextera JSON only.  Stems are unique across
own_kicked sources, so the JSON files are written flat into one label directory.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ext.armor_pose.ref_infer import TARGET_NAMES


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=str(ROOT / "own_kicked_20261025"))
    p.add_argument("--labels", default=str(ROOT / "own_kicked_20261025_auto14" / "labels"))
    p.add_argument("--out", default=str(ROOT / "own_kicked_dextera_labels"))
    p.add_argument("--workers", type=int, default=8)
    return p.parse_args()


def convert(task):
    lab_path, img_path, out_dir = task
    lines = [ln.split() for ln in lab_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    anns = []
    cats = {}
    for p in lines:
        if len(p) != 13:
            continue
        cid = int(p[0])
        cx, cy, bw, bh = (float(v) for v in p[1:5])
        pts = [(float(p[5 + 2 * i]), float(p[6 + 2 * i])) for i in range(4)]
        cats[cid] = TARGET_NAMES[cid] if 0 <= cid < len(TARGET_NAMES) else str(cid)
        anns.append({
            "category_id": cid,
            "bbox": [(cx - bw / 2), (cy - bh / 2), (cx + bw / 2), (cy + bh / 2)],  # normalized; fixed below
            "keypoints": {f"p{i + 1}": [x, y] for i, (x, y) in enumerate(pts)},
            "keypoints_visible": {f"p{i + 1}": True for i in range(4)},
        })
    if img_path is None or not img_path.exists():
        return None
    with Image.open(img_path) as im:
        w, h = im.size
    for a in anns:
        x1, y1, x2, y2 = a["bbox"]
        a["bbox"] = [x1 * w, y1 * h, x2 * w, y2 * h]
        a["keypoints"] = {k: [v[0] * w, v[1] * h] for k, v in a["keypoints"].items()}
    data = {
        "image": {"file": img_path.name, "width": w, "height": h},
        "categories": [{"id": c, "name": n} for c, n in sorted(cats.items())],
        "annotations": anns,
    }
    (out_dir / (lab_path.stem + ".json")).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return lab_path.stem


def main():
    a = parse()
    src_root, lab_root, out = Path(a.src), Path(a.labels), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    img_by_stem = {}
    for p in src_root.glob("*/*.jpg"):
        img_by_stem[p.stem] = p
    tasks = []
    for lab in lab_root.glob("*/*.txt"):
        tasks.append((lab, img_by_stem.get(lab.stem), out))
    print("labels", len(tasks), flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        done = sum(1 for r in ex.map(convert, tasks, chunksize=32) if r)
    print("written", done, "out", out)


if __name__ == "__main__":
    main()
