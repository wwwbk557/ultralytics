#!/usr/bin/env python3
"""Build a cleaned Scheme-C v3 split.

What it changes (everything else is left on the original split):
  * sentry labels that are too small / too dark to be readable are removed;
  * sentry labels whose plate shows a normal 3/4 digit are removed
    (uses the glyph audit CSV when available);
  * base labels without a visible light bar are removed
    (base is treated as out-of-scope for the class-balance work);
  * most sentry val capture roots are moved to train, so train finally sees
    the small/far drone domain.  ``rmcv:mixed_armor:seq0007`` is split
    temporally 55/35 with a 10% gap because it is one long sequence and has
    almost all of the red sentry val.

The source tree is never modified.  Images are symlinked, labels are copied
(filtered), so deleting a label here cannot damage the original dataset.

Usage:
    python scripts/build_schemeC_v3.py --dry-run
    python scripts/build_schemeC_v3.py --out datasets/schemeC_clean_v3
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

ROOT = Path("/home/wbk/ultralytics_2027")
HW = (480, 640)
NAMES = [
    "blue_sentry", "blue_hero", "blue_engineer", "blue_3", "blue_4", "blue_outpost", "blue_base",
    "red_sentry", "red_hero", "red_engineer", "red_3", "red_4", "red_outpost", "red_base",
]
SENTRY = {0, 7}
BASE = {6, 13}
# ws5 roots kept for val: one small/dark and one medium, so val still measures
# the hard domain instead of becoming an easy close-up set.
KEEP_WS5_VAL = {"drone_20260804_02_09", "drone_20260805_03_35"}
# one known bad label: the crop shows a clear "4"
KNOWN_WRONG = {"drone_20260801_05_01_2__b16ba53b__f004369__1785561095090000384"}


def parse():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--src", default=str(ROOT / "datasets/schemeC_split_bal"))
    p.add_argument("--out", default=str(ROOT / "datasets/schemeC_clean_v3"))
    p.add_argument("--glyph-csv", default=str(ROOT / "runs/audit/sentry_digit.csv"))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--sentry-min-area", type=float, default=50.0)
    p.add_argument("--sentry-min-side", type=float, default=3.0)
    p.add_argument("--sentry-min-mean", type=float, default=8.0)
    p.add_argument("--sentry-min-max", type=float, default=40.0)
    p.add_argument("--sentry-p34", type=float, default=0.60, help="delete if glyph p3/p4 exceeds this and p_sentry is low")
    p.add_argument("--base-min-area", type=float, default=60.0)
    p.add_argument("--base-min-max", type=float, default=200.0)
    return p.parse_args()


def image_path(src: Path, split: str, stem: str) -> Path:
    p = src / "images" / split / f"{stem}.jpg"
    if p.is_file():
        return p
    cand = sorted((src / "images" / split).glob(f"{stem}.*"))
    return cand[0]


def load_glyph(path: Path):
    out = {}
    if not path.is_file():
        return out
    for r in csv.DictReader(path.open()):
        key = (r["split"], r["stem"], int(r["scheme_cls"]), round(float(r["area480"]), 1))
        out[key] = r
    return out


def sentry_stats(img: Image.Image, cx, cy, bw, bh):
    w0, h0 = img.size
    s = min(HW[0] / h0, HW[1] / w0)
    wp, hp = bw * w0 * s, bh * h0 * s
    x1 = max(0, int((cx - bw / 2) * w0)); x2 = min(w0, int((cx + bw / 2) * w0) + 1)
    y1 = max(0, int((cy - bh / 2) * h0)); y2 = min(h0, int((cy + bh / 2) * h0) + 1)
    g = np.asarray(img.crop((x1, y1, x2, y2)).convert("L"))
    return wp, hp, float(g.mean()), float(g.max())


def load_frames():
    p = ROOT / "datasets/rmcv_armor_14cls_grouped/samples.tsv"
    out = {}
    if p.is_file():
        for r in csv.DictReader(p.open(), delimiter="\t"):
            try:
                out[r["file"]] = float(r["frame"]) if r.get("frame") not in (None, "", "None") else None
            except Exception:
                out[r["file"]] = None
    return out


def seq_split(rows, frames):
    """Temporal 55/35 split of one long sequence with a 10% gap."""
    def key(r):
        f = frames.get(r["stem"])
        return (f is None, f if f is not None else 0.0, r["stem"])
    rs = sorted(rows, key=key)
    n = len(rs)
    a = int(n * 0.55)
    b = int(n * 0.65)
    out = {}
    for i, r in enumerate(rs):
        if i < a:
            out[r["stem"]] = "train"
        elif i >= b:
            out[r["stem"]] = "val"
        else:
            out[r["stem"]] = "gap"
    return out




def root_sentry_counts(rows, src: Path):
    cnt = defaultdict(int)
    for r in rows:
        if r["split"] != "val":
            continue
        for line in (src / "labels" / r["split"] / r["label"]).read_text().splitlines():
            p = line.split()
            if p and int(p[0]) in SENTRY:
                cnt[r["root"]] += 1
    return cnt


def main():
    a = parse()
    src, out = Path(a.src), Path(a.out)
    rows = list(csv.DictReader((src / "split_manifest.tsv").open(), delimiter="\t"))
    glyph = load_glyph(Path(a.glyph_csv))
    frames = load_frames()

    # ---- decide the new split per image --------------------------------
    by_root = defaultdict(list)
    for r in rows:
        by_root[r["root"]].append(r)
    new_split = {}
    seq_roots = set()
    root_sentry = root_sentry_counts(rows, src)
    for root, rs in by_root.items():
        if root == "rmcv:mixed_armor:seq0007":
            seq_roots.add(root)
            new_split.update(seq_split(rs, frames))
            continue
        old = {r["split"] for r in rs}
        # Move only ws5 roots that actually contain sentry.  Moving every val
        # root would drag hundreds of good-class images along for 12 labels.
        if old == {"val"} and root.startswith("drone_") and root not in KEEP_WS5_VAL and root_sentry.get(root, 0) > 0:
            for r in rs:
                new_split[r["stem"]] = "train"
        else:
            for r in rs:
                new_split[r["stem"]] = r["split"]
    if seq_roots:
        print(f"temporal split: {sorted(seq_roots)}")
    print("split moves:", Counter((r["split"], new_split[r["stem"]]) for r in rows))

    # ---- filter labels --------------------------------------------------
    report = []
    kept_counts = Counter()
    del_counts = Counter()
    del_reason = Counter()
    out_rows = []
    for r in rows:
        sp = new_split[r["stem"]]
        if sp == "gap":
            continue
        lines = (src / "labels" / r["split"] / r["label"]).read_text().splitlines()
        classes = [int(x.split()[0]) for x in lines if len(x.split()) >= 13]
        if not any(c in SENTRY or c in BASE for c in classes):
            keep_lines = []
            seen = set()
            for line in lines:
                if len(line.split()) < 13:
                    continue
                if line in seen:
                    del_counts[int(line.split()[0])] += 1
                    del_reason["duplicate"] += 1
                    continue
                seen.add(line)
                keep_lines.append(line)
                kept_counts[int(line.split()[0])] += 1
            out_rows.append((r, sp, keep_lines))
            continue
        img = image_path(src, r["split"], r["stem"])
        had_labels = any(len(x.split()) >= 13 for x in lines)
        with Image.open(img) as im:
            keep_lines = []
            seen_lines = set()
            for li, line in enumerate(lines):
                p = line.split()
                if len(p) < 13:
                    continue
                if line in seen_lines:
                    del_counts[int(p[0])] += 1
                    del_reason["duplicate"] += 1
                    report.append([r["split"], sp, r["stem"], li, int(p[0]), "", "", "",
                                   "delete", "duplicate", "", "", ""])
                    continue
                seen_lines.add(line)
                c = int(p[0])
                cx, cy, bw, bh = map(float, p[1:5])
                if c in SENTRY or c in BASE:
                    wp, hp, mean, mx = sentry_stats(im, cx, cy, bw, bh)
                else:
                    wp = hp = mean = mx = 0.0
                area = wp * hp
                g = None
                reason = ""
                if c in SENTRY:
                    key = (r["split"], r["stem"], c, round(area, 1))
                    g = glyph.get(key)
                    if r["stem"] in KNOWN_WRONG:
                        reason = "known_wrong"
                    elif g and float(g["p_sentry"]) < 0.5 and max(float(g["p3"]), float(g["p4"])) >= a.sentry_p34:
                        reason = "wrong_digit"
                    elif area < a.sentry_min_area or min(wp, hp) < a.sentry_min_side:
                        reason = "too_small"
                    elif mean < a.sentry_min_mean or mx < a.sentry_min_max:
                        reason = "too_dark"
                elif c in BASE:
                    if area < a.base_min_area or mx < a.base_min_max:
                        reason = "no_light_bar"
                if reason:
                    del_counts[c] += 1
                    del_reason[reason] += 1
                    report.append([r["split"], sp, r["stem"], li, c, f"{area:.1f}", f"{mean:.1f}", f"{mx:.0f}",
                                   "delete", reason,
                                   g["p_sentry"] if g else "", g["p3"] if g else "", g["p4"] if g else ""])
                else:
                    keep_lines.append(line)
                    kept_counts[c] += 1
                    report.append([r["split"], sp, r["stem"], li, c, f"{area:.1f}", f"{mean:.1f}", f"{mx:.0f}",
                                   "keep", "", g["p_sentry"] if g else "", g["p3"] if g else "", g["p4"] if g else ""])
        if had_labels and not keep_lines:
            # Every label on this image was removed.  Writing an empty label
            # would turn a visible (but unreadable/wrong) robot into a
            # background-only negative, which is worse than dropping the image.
            continue
        out_rows.append((r, sp, keep_lines))

    print("kept instances:", dict(sorted(kept_counts.items())))
    print("deleted instances:", dict(sorted(del_counts.items())))
    print("delete reasons:", dict(del_reason))
    for split in ("train", "val"):
        print(f"  {split}: images={sum(1 for r, sp, _ in out_rows if sp == split)} "
              f"instances={sum(len(k) for r, sp, k in out_rows if sp == split)}")

    if a.dry_run:
        return
    if out.exists():
        if not a.force:
            raise SystemExit(f"output exists: {out}; use --force")
        shutil.rmtree(out)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    # ---- write symlinks + filtered labels ------------------------------
    manifest = ["source\tstem\tgroup\troot\tsplit\timage\tlabel\tout_stem"]
    seen = set()
    for r, sp, keep_lines in out_rows:
        if r["stem"] in seen:
            continue
        seen.add(r["stem"])
        img = image_path(src, r["split"], r["stem"])
        dst_img = out / "images" / sp / img.name
        os.symlink(os.path.relpath(img.resolve(), dst_img.parent), dst_img)
        dst_lab = out / "labels" / sp / f"{r['stem']}.txt"
        dst_lab.parent.mkdir(parents=True, exist_ok=True)
        dst_lab.write_text("\n".join(keep_lines) + ("\n" if keep_lines else ""))
        manifest.append(f"{r['source']}\t{r['stem']}\t{r['group']}\t{r['root']}\t{sp}\t{img.name}\t{dst_lab.name}\t{r['stem']}")
    (out / "split_manifest.tsv").write_text("\n".join(manifest) + "\n")

    y = yaml.safe_load((src / "data.yaml").read_text())
    y["path"] = str(out.resolve())
    y["train"] = "images/train"
    y["val"] = "images/val"
    (out / "data.yaml").write_text(yaml.safe_dump(y, sort_keys=False, allow_unicode=True))
    shutil.copy2(src / "classes.txt", out / "classes.txt")

    with (out / "label_report.tsv").open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["old_split", "new_split", "stem", "line", "cls", "area480", "mean", "max",
                    "decision", "reason", "p_sentry", "p3", "p4"])
        w.writerows(report)
    summary = {
        "src": str(src), "out": str(out),
        "kept": dict(sorted(kept_counts.items())), "deleted": dict(sorted(del_counts.items())),
        "reasons": dict(del_reason),
        "per_split": {sp: {
            "images": sum(1 for r, s, _ in out_rows if s == sp),
            "instances": sum(len(k) for r, s, k in out_rows if s == sp)} for sp in ("train", "val")},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
