#!/usr/bin/env python3
"""Build a leak-safe 14-class Scheme-C full view from pose23_current.

Label-only filtering: keep mixed images and drop white/5/unknown labels. Drop
objects with a missing (visibility=0) keypoint because the local compact head
has no visibility channel. The train/val split is unit-level: pose source
train/val dirs, source_json capture groups, and workspace capture sessions.
The strict proposal lists are only used as an initial assignment; units that
appear on both sides are consolidated onto one side.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path("/home/wbk/ultralytics_2027")
SRC = ROOT / "datasets/pose23_current"
OUT = ROOT / "datasets/schemeC_pose23_full_v5"
SPLIT_DIR = SRC / "split"
MAP = {0: 1, 1: 2, 2: 3, 3: 4, 5: 0, 6: 5, 14: 6,
       7: 8, 8: 9, 9: 10, 10: 11, 12: 7, 13: 12, 15: 13}
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


def rel_of(path: Path) -> str:
    return path.relative_to(SRC / "images").as_posix()


def unit_of(rel: str) -> str:
    parts = rel.split("/")
    if parts[0] == "pose" and len(parts) >= 4 and parts[2] in {"train", "val"}:
        return "/".join(parts[:3])
    if parts[0] == "workspaces" and len(parts) >= 3:
        base = Path(parts[-1]).stem.split("__")[0]
        m = re.match(r"^(.*)_\d+$", base)
        base = m.group(1) if m else base
        return f"workspaces/{parts[1]}/{base}"
    if parts[0] == "source_json" and len(parts) >= 3:
        return f"source_json/{parts[1]}"
    return "/".join(parts[:2])


def convert(text: str):
    kept = []
    oldc = Counter(); newc = Counter(); drop = Counter()
    for line in text.splitlines():
        f = line.split()
        if not f:
            continue
        if len(f) != 17:
            drop["bad_fields"] += 1; continue
        try:
            old = int(f[0]); vals = [float(x) for x in f[1:]]
        except ValueError:
            drop["bad_number"] += 1; continue
        if old not in MAP:
            drop["excluded_class"] += 1; continue
        vis = [vals[4 + 3 * k + 2] for k in range(4)]
        if any(v == 0 for v in vis):
            drop["missing_keypoint"] += 1; continue
        cx, cy, bw, bh = vals[:4]
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
            drop["bad_bbox"] += 1; continue
        coords = []
        for k in range(4):
            x, y = vals[4 + 3 * k], vals[4 + 3 * k + 1]
            if not (0 <= x <= 1 and 0 <= y <= 1):
                drop["bad_keypoint"] += 1; break
            coords.extend((x, y))
        else:
            kept.append(" ".join([str(MAP[old]), *(f"{x:.8f}" for x in (cx, cy, bw, bh)),
                                  *(f"{x:.8f}" for x in coords)]))
            oldc[old] += 1; newc[MAP[old]] += 1
    return kept, oldc, newc, drop


def main():
    a = parse()
    src, out = Path(a.src), Path(a.out)
    # Initial unit assignment from the strict proposal.
    unit_votes = defaultdict(Counter)
    for split, name in (("train", "train.txt"), ("val", "val.txt")):
        p = SPLIT_DIR / name
        if not p.is_file():
            raise SystemExit(f"missing split metadata: {p}")
        for raw in p.read_text().splitlines():
            if not raw.strip():
                continue
            rel = raw.split("/images/", 1)[1]
            unit_votes[unit_of(rel)][split] += 1
    unit_split = {}
    conflicts = []
    for u, votes in unit_votes.items():
        if votes["train"] and votes["val"]:
            # consolidate the whole unit on the majority side
            side = "train" if votes["train"] >= votes["val"] else "val"
            conflicts.append((u, votes["train"], votes["val"], side))
        else:
            side = "train" if votes["train"] else "val"
        unit_split[u] = side
    print(f"units={len(unit_split)} conflicts={len(conflicts)}")
    for c in conflicts:
        print("  conflict", c)

    all_images = sorted(p for p in (src / "images").rglob("*") if p.is_file() or p.is_symlink())
    rows = []
    class_counts = {sp: Counter() for sp in ("train", "val")}
    stats = defaultdict(Counter)
    used = set()
    for img in all_images:
        rel = rel_of(img)
        u = unit_of(rel)
        split = unit_split.get(u)
        if split is None:
            # No strict-proposal image for this unit. Pose source dirs have an
            # explicit split; otherwise deterministic unit hash, never per-image.
            parts = rel.split("/")
            if parts[0] == "pose" and len(parts) >= 4 and parts[2] in {"train", "val"}:
                split = parts[2]
            else:
                split = "val" if int(hashlib.sha256(u.encode()).hexdigest()[:8], 16) % 100 < 15 else "train"
            unit_split[u] = split
        lp = src / "labels" / Path(rel).with_suffix(".txt")
        if not lp.is_file():
            stats["skip_no_label"][u] += 1; continue
        if not img.exists():
            stats["skip_missing_image"][u] += 1; continue
        text = lp.read_text()
        kept, oldc, newc, drop = convert(text)
        for k, v in drop.items():
            stats[k][u] += v
        if not kept and text.strip():
            # Image still contains excluded / unreadable objects; do not turn it
            # into a background-only negative.
            stats["skip_no_14_label"][u] += 1; continue
        stem = img.stem
        out_name = img.name
        if stem in used:
            out_name = f"{u.replace('/', '_')}__{img.name}"
            stem = Path(out_name).stem
        used.add(stem)
        rows.append((rel, u, split, out_name, f"{stem}.txt", len(kept), oldc, newc, bool(text.strip())))
        class_counts[split].update(newc)
        stats[f"images_{split}"][u] += 1

    train_units = {u for u, s in unit_split.items() if s == "train"}
    val_units = {u for u, s in unit_split.items() if s == "val"}
    overlap = train_units & val_units
    assert not overlap, overlap
    print("images", {sp: sum(1 for r in rows if r[2] == sp) for sp in ("train", "val")})
    print("objects", {sp: sum(c.values()) for sp, c in class_counts.items()})
    print("classes", {sp: dict(sorted(c.items())) for sp, c in class_counts.items()})
    print("stats", {k: sum(v.values()) for k, v in stats.items()})
    if a.dry_run:
        return
    if out.exists():
        if not a.force:
            raise SystemExit(f"{out} exists; use --force")
        shutil.rmtree(out)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    manifest = []
    for rel, u, split, out_name, lab_name, n, oldc, newc, had in rows:
        img = src / "images" / rel
        lp = src / "labels" / Path(rel).with_suffix(".txt")
        kept, _, _, _ = convert(lp.read_text())
        dst_img = out / "images" / split / out_name
        os.symlink(os.path.relpath(img.resolve(), dst_img.parent), dst_img)
        (out / "labels" / split / lab_name).write_text("\n".join(kept) + ("\n" if kept else ""))
        manifest.append((rel, u, split, out_name, lab_name, n))
    data = {
        "path": str(out.resolve()), "train": "images/train", "val": "images/val",
        "kpt_shape": [4, 2], "flip_idx": [1, 0, 3, 2], "nc": 14,
        "names": {i: n for i, n in enumerate(NAMES)},
    }
    (out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    (out / "classes.txt").write_text("\n".join(NAMES) + "\n")
    with (out / "split_manifest.tsv").open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["source_rel", "unit", "split", "image", "label", "objects"])
        w.writerows(manifest)
    summary = {
        "source": str(src), "output": str(out),
        "rule": "label-only filter: drop white/5/unknown; drop visibility=0 objects; keep mixed images; unit-level leakage-safe split",
        "mapping": {str(k): v for k, v in MAP.items()},
        "units": {"train": len(train_units), "val": len(val_units), "overlap": len(overlap)},
        "conflicts_consolidated": conflicts,
        "images": {sp: sum(1 for r in rows if r[2] == sp) for sp in ("train", "val")},
        "objects": {sp: sum(c.values()) for sp, c in class_counts.items()},
        "class_counts": {sp: {NAMES[i]: c[i] for i in range(14)} for sp, c in class_counts.items()},
        "stats": {k: sum(v.values()) for k, v in stats.items()},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (out / "README.md").write_text(
        "# Scheme-C 14-class full view from pose23_current\n\n"
        "Label-only: mixed images are kept and only white/5/unknown labels are dropped.\n"
        "Objects with visibility=0 are dropped (no coordinate; local head has no visibility).\n"
        "Visibility=1 objects are kept. Classes are mapped to local sentry/hero/engineer/3/4/outpost/base.\n"
        "Keypoint order stays LT,RT,RB,LB; the third value is removed.\n"
        "Split is unit-level: pose source-train/val dirs, source_json capture groups, workspace\n"
        "capture sessions. Units seen on both sides of the strict proposal were consolidated.\n"
    )
    print("wrote", out)


if __name__ == "__main__":
    main()
