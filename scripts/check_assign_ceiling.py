"""How many anchor centres can a GT box contain, at a given input size?

TAL can only pick positive anchors whose centre falls inside the GT box, so this
is the hard ceiling on "anchors per GT" - independent of training. Use it to
check whether an input-size change can fix small-object assignment starvation.

Usage:
    python scripts/check_assign_ceiling.py --n 1200
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "datasets/schemeC_split/images/train"
LABELS = ROOT / "datasets/schemeC_split/labels/train"

CANDIDATES = [((480, 640), (8, 16, 32)), ((640, 640), (8, 16, 32)), ((640, 960), (8, 16, 32)),
              ((720, 1280), (8, 16, 32)), ((480, 640), (4, 8, 16, 32))]
EDGES = (32.0**2, 96.0**2, 192.0**2)
BUCKETS = ("tiny", "small", "medium", "large")


def centres(n: int, stride: int) -> np.ndarray:
    return (np.arange(n // stride) + 0.5) * stride


def pool_size(box: np.ndarray, hw: tuple[int, int], strides: tuple[int, ...]) -> int:
    """Number of anchor centres (all levels) inside an xyxy box, in input pixels."""
    h, w = hw
    x1, y1, x2, y2 = box
    x1, x2 = np.clip([x1, x2], 0, w)
    y1, y2 = np.clip([y1, y2], 0, h)
    total = 0
    for s in strides:
        xs = centres(w, s)
        ys = centres(h, s)
        total += int(((xs >= x1) & (xs <= x2)).sum() * ((ys >= y1) & (ys <= y2)).sum())
    return total


def letterbox_box(
    lb: np.ndarray, orig: tuple[int, int], hw: tuple[int, int], strides: tuple[int, ...] = (8, 16, 32)
) -> np.ndarray:
    """normalised xywh -> xyxy in the letterboxed input."""
    w0, h0 = orig
    h, w = hw
    s = min(h / h0, w / w0)
    w1, h1 = w0 * s, h0 * s
    pad_x, pad_y = (w - w1) / 2, (h - h1) / 2
    cx, cy, bw, bh = lb
    x1 = pad_x + (cx - bw / 2) * w1
    y1 = pad_y + (cy - bh / 2) * h1
    x2 = pad_x + (cx + bw / 2) * w1
    y2 = pad_y + (cy + bh / 2) * h1
    # ultralytics select_candidates_in_gts expands any side < stride[0] to stride_val=16
    side_w, side_h = x2 - x1, y2 - y1
    s0 = min(strides)
    sval = strides[1] if len(strides) > 1 else s0
    exp_w = float(sval) if side_w < s0 else side_w
    exp_h = float(sval) if side_h < s0 else side_h
    cx2, cy2 = (x1 + x2) / 2, (y1 + y2) / 2
    return np.array([cx2 - exp_w / 2, cy2 - exp_h / 2, cx2 + exp_w / 2, cy2 + exp_h / 2], dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1200, help="number of images to sample")
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    files = sorted(IMAGES.glob("*.jpg"))
    random.seed(0)
    files = random.sample(files, min(args.n, len(files)))
    data = []
    for f in files:
        with Image.open(f) as im:
            w0, h0 = im.size
        lb = LABELS / (f.stem + ".txt")
        if not lb.is_file():
            continue
        a = np.loadtxt(lb, ndmin=2)
        if a.size == 0:
            continue
        data.append((w0, h0, a[:, 1:5].astype(np.float64)))
    print(f"images={len(data)}", flush=True)

    aspects = np.array([w / h for w, h, _ in data])
    print("aspect w/h: median %.2f  min %.2f  max %.2f" % (np.median(aspects), aspects.min(), aspects.max()))

    base = (480, 640)
    base_areas = []
    base_strides = (8, 16, 32)
    for w0, h0, boxes in data:
        for b in boxes:
            px = letterbox_box(b, (w0, h0), base, base_strides)
            base_areas.append(max((px[2] - px[0]) * (px[3] - px[1]), 0.0))
    base_areas = np.array(base_areas)
    bucket = np.digitize(base_areas, EDGES)

    for hw, strides in CANDIDATES:
        pools = []
        for w0, h0, boxes in data:
            for b in boxes:
                px = letterbox_box(b, (w0, h0), hw, strides)
                pools.append(pool_size(px, hw, strides))
        pools = np.array(pools)
        tag = f"{hw[0]}x{hw[1]} strides={strides}"
        print(f"\n{tag}")
        print(f"  {'bucket':<7}{'n':>7}{'median':>8}{'mean':>8}{'<topk':>8}")
        for i, name in enumerate(BUCKETS):
            p = pools[bucket == i]
            if p.size == 0:
                continue
            print(f"  {name:<7}{p.size:>7}{np.median(p):>8.1f}{p.mean():>8.1f}{(p < args.topk).mean():>8.1%}")
        p = pools[bucket == 0]
        print(f"  tiny pool median={np.median(p):.1f}, >=topk share={(p >= args.topk).mean():.1%}")


if __name__ == "__main__":
    main()
