"""Contact sheet of one class across train/val, sorted by object size.

Answers "do train and val look like the same data?" for a single armour class:
picks a size-stratified sample from both splits (largest to smallest), crops each
instance with context, and stamps split / size / scene on every tile.

Usage:
    python scripts/sentry_gallery.py --cls 0 --per-split 12
    python scripts/sentry_gallery.py --cls 0 --data datasets/schemeC_split/data.yaml --out /tmp/g.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TILE_W, TILE_H = 210, 150
REF = (480, 640)  # training input H, W


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data", default=str(ROOT / "datasets/schemeC_split/data.yaml"))
    p.add_argument("--cls", type=int, default=0, help="class id to sample (0 = blue_sentry)")
    p.add_argument("--per-split", type=int, default=12)
    p.add_argument("--pad", type=float, default=0.55, help="context padding around the box, fraction of max side")
    p.add_argument("--out", default="")
    return p.parse_args()


def read_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text())


def load_instances(data_yaml: Path, cls: int) -> dict[str, list[dict]]:
    cfg = read_yaml(data_yaml)
    root = Path(cfg.get("path", data_yaml.parent))
    out: dict[str, list[dict]] = {}
    for split in ("train", "val"):
        rel = str(cfg.get(split, f"images/{split}"))
        img_dir = Path(rel) if Path(rel).is_absolute() else root / rel
        lab_dir = Path(str(img_dir).replace("/images/", "/labels/"))
        items = []
        for lab in sorted(lab_dir.glob("*.txt")):
            img_path = img_dir / f"{lab.stem}.jpg"
            if not img_path.is_file():
                continue
            for line in lab.read_text().splitlines():
                p = line.split()
                if len(p) != 13 or int(p[0]) != cls:
                    continue
                v = np.array(p[1:], dtype=np.float64)
                items.append(
                    {
                        "img": img_path,
                        "img_hw": None,
                        "box": v[0:4],  # cx cy w h, normalized
                        "kpts": v[4:].reshape(4, 2),
                        "scene": lab.stem.split("__")[0],
                        "size": (v[2] * REF[1], v[3] * REF[0]),  # px in the 480x640 training frame
                    }
                )
        out[split] = items
    return out


def pick(items: list[dict], k: int) -> list[dict]:
    """Size-stratified pick with distinct scenes, largest first."""
    items = sorted(items, key=lambda d: -d["size"][1])
    n = len(items)
    if n <= k:
        return items
    idx = np.linspace(0, n - 1, k).round().astype(int)
    chosen, seen = [], set()
    for i in idx:  # first pass: distinct scenes
        it = items[int(i)]
        if it["scene"] not in seen:
            chosen.append(it)
            seen.add(it["scene"])
    for i in idx:  # fill up if a scene repeats
        if len(chosen) >= k:
            break
        it = items[int(i)]
        if it not in chosen:
            chosen.append(it)
    return sorted(chosen, key=lambda d: -d["size"][1])[:k]


def tile(inst: dict, split: str) -> np.ndarray:
    img = cv2.imread(str(inst["img"]))
    h, w = img.shape[:2]
    cx, cy = inst["box"][0] * w, inst["box"][1] * h
    bw, bh = inst["box"][2] * w, inst["box"][3] * h
    pad = inst["pad"] * max(bw, bh) if "pad" in inst else 0.55 * max(bw, bh)
    x0, y0 = int(max(cx - bw / 2 - pad, 0)), int(max(cy - bh / 2 - pad, 0))
    x1, y1 = int(min(cx + bw / 2 + pad, w - 1)), int(min(cy + bh / 2 + pad, h - 1))
    crop = img[y0:y1, x0:x1].copy()
    if crop.size == 0:
        crop = np.zeros((TILE_H, TILE_W, 3), np.uint8)
    crop = cv2.resize(crop, (TILE_W, TILE_H), interpolation=cv2.INTER_LINEAR)

    sx, sy = TILE_W / max(x1 - x0, 1), TILE_H / max(y1 - y0, 1)
    col = (255, 160, 40) if split == "train" else (60, 220, 60)
    cv2.rectangle(
        crop,
        (int((cx - bw / 2 - x0) * sx), int((cy - bh / 2 - y0) * sy)),
        (int((cx + bw / 2 - x0) * sx), int((cy + bh / 2 - y0) * sy)),
        col,
        1,
    )
    for k in range(4):
        px = int((inst["kpts"][k, 0] * w - x0) * sx)
        py = int((inst["kpts"][k, 1] * h - y0) * sy)
        cv2.circle(crop, (px, py), 2, (0, 0, 255), -1)
        cv2.putText(crop, str(k), (px + 2, py - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(crop, (0, 0), (TILE_W - 1, TILE_H - 1), (70, 70, 70), 1)
    cv2.putText(crop, f"{split}", (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)
    cv2.putText(
        crop,
        f"{inst['size'][0]:.0f}x{inst['size'][1]:.0f}px  {inst['scene'][:20]}",
        (4, TILE_H - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.33,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    return crop


def main() -> None:
    a = parse()
    data = Path(a.data)
    per = load_instances(data, a.cls)
    rows = []
    for split in ("train", "val"):
        items = per[split]
        picked = pick(items, a.per_split)
        print(f"\n{split}: {len(items)} instances, showing {len(picked)} (largest -> smallest)")
        for it in picked:
            print(f"  {it['size'][0]:5.1f}x{it['size'][1]:4.1f}px  {it['scene'][:34]:<36} {it['img']}")
        rows.append([tile(it, split) for it in picked])
    ncol = max(len(r) for r in rows)
    rows = [r + [np.zeros((TILE_H, TILE_W, 3), np.uint8)] * (ncol - len(r)) for r in rows]
    sheet = cv2.vconcat([cv2.hconcat(r) for r in rows])
    out = Path(a.out) if a.out else ROOT / "runs/pose/_eval" / f"class{a.cls}_train_vs_val.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet)
    print(f"\nwrote {out}  ({sheet.shape[1]}x{sheet.shape[0]})")


if __name__ == "__main__":
    main()
