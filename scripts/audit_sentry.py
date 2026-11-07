"""Audit the Scheme-C split, with focus on sentry (classes 0 / 7).

Reports per-class counts, input-space box sizes at 480x640, sentry sources,
label sanity and exact train/val duplicates (leakage check).

Usage:
    python scripts/audit_sentry.py
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT / "datasets/schemeC_split"
NAMES = [
    "blue_sentry", "blue_hero", "blue_engineer", "blue_3", "blue_4", "blue_outpost", "blue_base",
    "red_sentry", "red_hero", "red_engineer", "red_3", "red_4", "red_outpost", "red_base",
]
SENTRY = {0, 7}
HW = (480, 640)


def source_of(stem: str) -> str:
    s = stem.split("__")[0]
    for p in ("xjtu", "mixed_armor", "drone", "infantry", "qdu_gimbal", "qdu_damo", "own"):
        if s == p or s.startswith(p + "_"):
            if p == "own":
                return "_".join(s.split("_")[:2]) + ("_" + s.split("_")[2] if s.count("_") >= 2 else "")
            return p
    return s


def size_of(img: Path) -> tuple[int, int]:
    with Image.open(img) as im:
        return im.size


def scan(split: str):
    lbl_dir = SPLIT / "labels" / split
    img_dir = SPLIT / "images" / split
    per_class = defaultdict(lambda: {"n": 0, "imgs": set(), "areas": [], "src": Counter(),
                                     "wh": [], "bad_kpt": 0, "bad_box": 0})
    md5 = {}
    stems = set()
    img_sizes = {}
    files = sorted(lbl_dir.glob("*.txt"))
    for i, f in enumerate(files):
        stem = f.stem
        stems.add(stem)
        img = img_dir / f"{stem}.jpg"
        if not img.is_file():
            img = next(iter(img_dir.glob(f"{stem}.*")), None)
        if img is None:
            continue
        if stem not in img_sizes:
            img_sizes[stem] = size_of(img)
        w0, h0 = img_sizes[stem]
        s = min(HW[0] / h0, HW[1] / w0)
        rows = np.loadtxt(f, ndmin=2)
        if rows.size == 0:
            continue
        for r in rows:
            c = int(r[0])
            cx, cy, bw, bh = r[1:5]
            d = per_class[c]
            d["n"] += 1
            d["imgs"].add(stem)
            w_px, h_px = bw * w0 * s, bh * h0 * s
            d["areas"].append(w_px * h_px)
            d["wh"].append((w_px, h_px))
            d["src"][source_of(stem)] += 1
            kp = r[5:].reshape(4, 2)
            # keypoints in the original image frame, compare against the box
            kx, ky = kp[:, 0] * w0 * s, kp[:, 1] * h0 * s
            bx1, by1 = (cx - bw / 2) * w0 * s, (cy - bh / 2) * h0 * s
            bx2, by2 = (cx + bw / 2) * w0 * s, (cy + bh / 2) * h0 * s
            tol = 2.0
            if ((kx < bx1 - tol) | (kx > bx2 + tol) | (ky < by1 - tol) | (ky > by2 + tol)).any():
                d["bad_kpt"] += 1
            if w_px < 1 or h_px < 1:
                d["bad_box"] += 1
        md5[stem] = hashlib.md5((img).read_bytes()).hexdigest()
    return per_class, md5, stems


def main() -> None:
    tr, tr_md5, tr_stems = scan("train")
    va, va_md5, va_stems = scan("val")

    print(f"train images={len(tr_stems)}  val images={len(va_stems)}")
    print(f"\n{'class':<15}{'train':>8}{'val':>7}{'tr_imgs':>9}{'va_imgs':>9}"
          f"{'med_w':>7}{'med_h':>7}{'tiny%':>7}{'<8px%':>7}{'badKpt':>7}")
    for c, name in enumerate(NAMES):
        a, b = tr[c], va[c]
        wa = np.array([w for w, h in a["wh"]]) if a["wh"] else np.array([0.0])
        ha = np.array([h for w, h in a["wh"]]) if a["wh"] else np.array([0.0])
        tiny = float(np.mean(np.array(a["areas"]) < 32**2)) if a["areas"] else 0.0
        lt8 = float(np.mean((wa < 8) | (ha < 8))) if a["areas"] else 0.0
        print(f"{name:<15}{a['n']:>8}{b['n']:>7}{len(a['imgs']):>9}{len(b['imgs']):>9}"
              f"{np.median(wa):>7.1f}{np.median(ha):>7.1f}{tiny*100:>6.1f}%{lt8*100:>6.1f}%{a['bad_kpt']:>7}")

    print("\nsentry sources (train instances / val instances)")
    for c in sorted(SENTRY):
        print(f"  {NAMES[c]:<12} train {dict(tr[c]['src'])}")
        print(f"  {'':<12} val   {dict(va[c]['src'])}")
    print("\nsentry co-occurrence in train images (other classes in the same image)")
    co = Counter()
    lbl_dir = SPLIT / "labels/train"
    for stem in tr[0]["imgs"] | tr[7]["imgs"]:
        rows = np.loadtxt(lbl_dir / f"{stem}.txt", ndmin=2)
        others = {int(r[0]) for r in rows} - SENTRY
        for o in others:
            co[NAMES[o]] += 1
    print("  " + ", ".join(f"{k}:{v}" for k, v in co.most_common(8)))

    # leakage: exact duplicates across splits
    tr_hashes = set(tr_md5.values())
    dup = [s for s, h in va_md5.items() if h in tr_hashes]
    print(f"\nexact duplicate images val∩train: {len(dup)} / {len(va_md5)}")
    if dup:
        print("  examples:", dup[:5])
    common_stems = tr_stems & va_stems
    print(f"same filename in both splits: {len(common_stems)}")

    # sentry GT count by split per source, to see if a whole source is only in val
    print("\nsentry instances per source (train/val)")
    srcs = sorted(set(tr[0]["src"]) | set(tr[7]["src"]) | set(va[0]["src"]) | set(va[7]["src"]))
    for s in srcs:
        a = tr[0]["src"][s] + tr[7]["src"][s]
        b = va[0]["src"][s] + va[7]["src"][s]
        print(f"  {s:<12} {a:>6} / {b:>5}")


if __name__ == "__main__":
    main()
