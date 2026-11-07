#!/usr/bin/env python3
"""Build one clean, per-source Scheme-C dataset without train/val split.

Scheme C (14 classes, one base, no 5, no white):
    color 0/1 = blue/red
    role 0..6 = sentry, hero, engineer, 3, 4, outpost, base
    class = color * 7 + role

Rules:
  * white_* annotations are removed;
  * blue_5 / red_5 annotations are removed;
  * if a source label contains ONLY white/5 objects, the image is omitted;
  * if a source label mixes white/5 with kept objects, only kept objects are written;
  * empty JSON labels are kept as reviewed negatives (empty .txt);
  * unlabeled images are placed under unlabeled/images, never as negatives;
  * no train/val split is created.

The raw source directories are not modified. Images are hard-linked into the
clean tree when possible, so the clean output does not duplicate 3+ GB.
"""
from __future__ import annotations

import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path("/home/wbk/ultralytics_2027/datasets")
OUT = ROOT / "clean_schemeC"

SCHEME_NAMES = [
    "blue_sentry", "blue_hero", "blue_engineer", "blue_3", "blue_4", "blue_outpost", "blue_base",
    "red_sentry", "red_hero", "red_engineer", "red_3", "red_4", "red_outpost", "red_base",
]
NAME_TO_SCHEME = {
    "blue_guard": 0, "blue_1": 1, "blue_2": 2, "blue_3": 3, "blue_4": 4,
    "blue_outpost": 5, "blue_base": 6,
    "red_guard": 7, "red_1": 8, "red_2": 9, "red_3": 10, "red_4": 11,
    "red_outpost": 12, "red_base": 13,
}
DROP_NAMES = {"blue_5", "red_5"}
KP_ORDER = ("left_top", "right_top", "right_bottom", "left_bottom")
IMG_EXT = {".jpg", ".jpeg", ".png"}


def is_white(name: str) -> bool:
    return name.startswith("white_")


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src.resolve(), dst)
    except OSError:
        shutil.copy2(src, dst)


def fmt(v: float) -> str:
    return f"{min(1.0, max(0.0, float(v))):.6f}"


def kpt_line(ann: dict, w: int, h: int) -> tuple[str, str] | None:
    kp = ann.get("keypoints") or {}
    vis = ann.get("keypoints_visible") or {}
    pts = []
    for name in KP_ORDER:
        p = kp.get(name)
        if not (isinstance(p, (list, tuple)) and len(p) >= 2):
            return None, "bad_kpt"
        if vis.get(name, True) is False:
            return None, "invisible"
        pts.append((float(p[0]), float(p[1])))
    return " ".join(f"{fmt(x / w)} {fmt(y / h)}" for x, y in pts), ""


def bbox_cxcywh(ann: dict, pts: list[tuple[float, float]], w: int, h: int) -> tuple[float, float, float, float]:
    """Tight bbox from the four keypoints; the source bbox is deliberately ignored."""
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
    return (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h


def write_txt(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_source_readme(dst: Path, source: str, counts: Counter, note: str = "") -> None:
    lines = [
        f"# {source} — Scheme C clean source",
        "",
        "No train/val split. This directory is one source only.",
        "",
        "```text",
        "images/            labeled images (positive or reviewed empty-negative)",
        "labels/            YOLO pose .txt, 14 Scheme-C classes",
        "unlabeled/images/  images with no JSON label at all (not negatives)",
        "manifest.tsv       per-image audit: kept/dropped counts and status",
        "classes.txt        14 class names",
        "```",
        "",
        "Label line: `class cx cy w h x1 y1 x2 y2 x3 y3 x4 y4` (all normalized).",
        "Keypoint order: left_top, right_top, right_bottom, left_bottom.",
        "",
        "## Rules applied",
        "",
        "- `white_*` annotations removed.",
        "- `blue_5` / `red_5` annotations removed.",
        "- Image omitted when its JSON contained only white/5 objects.",
        "- If white/5 were mixed with kept objects, only kept objects are written.",
        "- Empty JSON is kept as an empty .txt negative.",
        "- Images without JSON are in `unlabeled/`, not in `images/`.",
        "- Raw source is untouched.",
        "",
        "## Counts",
        "",
        "| item | count |",
        "|---|---:|",
    ]
    for k in ("images_total", "labels", "images_in_images", "labels_in_labels", "unlabeled", "deleted_white5_only", "deleted_no_valid", "negatives", "positive", "ann_kept", "white_dropped", "five_dropped", "invalid_dropped"):
        lines.append(f"| {k} | {counts.get(k, 0)} |")
    if note:
        lines += ["", note]
    (dst / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def convert_ws(ws: str) -> dict:
    src = ROOT / ws
    dst = OUT / ws
    for sub in ("images", "labels", "unlabeled/images"):
        (dst / sub).mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in (src / "images").glob("*") if p.suffix.lower() in IMG_EXT)
    label_paths: dict[str, Path] = {}
    for p in sorted((src / "labels").glob("*.json")):
        label_paths.setdefault(p.stem, p)
    for p in sorted((src / "images").glob("*.json")):
        label_paths.setdefault(p.stem, p)

    manifest = ["stem\tgroup\tstatus\tn_original\tn_kept\tn_white\tn_five\tn_invalid\tn_other\timage\tlabel"]
    counts = Counter(images_total=len(images), labels=len(label_paths))
    labels_by_stem = set(label_paths)
    images_by_stem = {p.stem for p in images}

    for img in images:
        stem = img.stem
        group = stem.split("__")[0]
        lp = label_paths.get(stem)
        if lp is None:
            link_or_copy(img, dst / "unlabeled/images" / img.name)
            manifest.append(f"{stem}\t{group}\tunlabeled\t0\t0\t0\t0\t0\t0\t{img.name}\t")
            counts["unlabeled"] += 1
            continue

        try:
            data = json.loads(lp.read_text(encoding="utf-8"))
        except Exception:
            counts["bad_json"] += 1
            manifest.append(f"{stem}\t{group}\tbad_json\t0\t0\t0\t0\t0\t0\t{img.name}\t{lp.name}")
            continue

        w = int(data.get("image", {}).get("width") or 0)
        h = int(data.get("image", {}).get("height") or 0)
        if not w or not h:
            counts["bad_size"] += 1
            manifest.append(f"{stem}\t{group}\tbad_size\t0\t0\t0\t0\t0\t0\t{img.name}\t{lp.name}")
            continue

        cats = {c.get("id"): c.get("name") for c in data.get("categories", [])}
        anns = data.get("annotations", [])
        lines = []
        n_white = n_five = n_invalid = n_other = 0
        for ann in anns:
            name = cats.get(ann.get("category_id"))
            if name is None:
                n_other += 1
                continue
            if is_white(name):
                n_white += 1
                continue
            if name in DROP_NAMES:
                n_five += 1
                continue
            cid = NAME_TO_SCHEME.get(name)
            if cid is None:
                n_other += 1
                continue
            kline, why = kpt_line(ann, w, h)
            if kline is None:
                n_invalid += 1
                continue
            kp = ann.get("keypoints") or {}
            pts = [(float(kp[k][0]), float(kp[k][1])) for k in KP_ORDER]
            cx, cy, bw, bh = bbox_cxcywh(ann, pts, w, h)
            lines.append(f"{cid} {fmt(cx)} {fmt(cy)} {fmt(bw)} {fmt(bh)} {kline}")

        n_orig = len(anns)
        counts["white_dropped"] += n_white
        counts["five_dropped"] += n_five
        counts["invalid_dropped"] += n_invalid
        counts["other_dropped"] += n_other

        if lines:
            write_txt(dst / "labels" / f"{stem}.txt", lines)
            link_or_copy(img, dst / "images" / img.name)
            counts["images_in_images"] += 1
            counts["labels_in_labels"] += 1
            counts["positive"] += 1
            counts["ann_kept"] += len(lines)
            status = "ok" if n_white + n_five + n_invalid + n_other == 0 else "filtered"
            manifest.append(f"{stem}\t{group}\t{status}\t{n_orig}\t{len(lines)}\t{n_white}\t{n_five}\t{n_invalid}\t{n_other}\t{img.name}\t{stem}.txt")
            continue

        if n_orig == 0:
            # Reviewed empty label: keep as a negative sample.
            write_txt(dst / "labels" / f"{stem}.txt", [])
            link_or_copy(img, dst / "images" / img.name)
            counts["images_in_images"] += 1
            counts["labels_in_labels"] += 1
            counts["negatives"] += 1
            manifest.append(f"{stem}\t{group}\tnegative\t0\t0\t0\t0\t0\t0\t{img.name}\t{stem}.txt")
        elif n_white + n_five == n_orig:
            # The requested deletion: only white/5 objects in this label.
            counts["deleted_white5_only"] += 1
            manifest.append(f"{stem}\t{group}\tdeleted_white5_only\t{n_orig}\t0\t{n_white}\t{n_five}\t{n_invalid}\t{n_other}\t{img.name}\t{lp.name}")
        else:
            # Had objects but none usable after filtering (invalid/unknown only).
            counts["deleted_no_valid"] += 1
            manifest.append(f"{stem}\t{group}\tdeleted_no_valid\t{n_orig}\t0\t{n_white}\t{n_five}\t{n_invalid}\t{n_other}\t{img.name}\t{lp.name}")

    for stem in sorted(labels_by_stem - images_by_stem):
        manifest.append(f"{stem}\t{stem.split('__')[0]}\tmissing_image\t0\t0\t0\t0\t0\t0\t\t{label_paths[stem].name}")
        counts["missing_image"] += 1

    (dst / "classes.txt").write_text("\n".join(SCHEME_NAMES) + "\n", encoding="utf-8")
    (dst / "manifest.tsv").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    write_source_readme(dst, ws, counts, f"Source: `{src}`")
    return dict(counts)


def load_names(yaml_path: Path) -> list[str]:
    y = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    names = y.get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=lambda x: int(x))]
    return [str(n) for n in names]


def convert_yolo_source(name: str, src: Path, out_name: str | None = None) -> dict:
    dst = OUT / (out_name or name)
    for sub in ("images", "labels", "unlabeled/images"):
        (dst / sub).mkdir(parents=True, exist_ok=True)
    old_names = load_names(src / "data.yaml")
    manifest = ["stem\tgroup\tstatus\tn_original\tn_kept\tn_white\tn_five\tn_invalid\tn_other\timage\tlabel"]
    counts = Counter()
    for split in ("train", "val"):
        idir, ldir = src / "images" / split, src / "labels" / split
        if not idir.is_dir():
            continue
        for img in sorted(p for p in idir.glob("*") if p.suffix.lower() in IMG_EXT):
            stem = img.stem
            group = stem.split("__")[0]
            lp = ldir / f"{stem}.txt"
            counts["images_total"] += 1
            if not lp.exists():
                link_or_copy(img, dst / "unlabeled/images" / img.name)
                counts["unlabeled"] += 1
                manifest.append(f"{stem}\t{group}\tunlabeled\t0\t0\t0\t0\t0\t0\t{img.name}\t")
                continue
            raw = [ln for ln in lp.read_text(encoding="utf-8").splitlines() if ln.strip()]
            lines = []
            n_white = n_five = n_invalid = n_other = 0
            for ln in raw:
                parts = ln.split()
                if len(parts) < 13:
                    n_invalid += 1
                    continue
                try:
                    old_id = int(float(parts[0]))
                except ValueError:
                    n_invalid += 1
                    continue
                old_name = old_names[old_id] if 0 <= old_id < len(old_names) else None
                if old_name is None:
                    n_other += 1
                    continue
                if is_white(old_name):
                    n_white += 1
                    continue
                if old_name in DROP_NAMES:
                    n_five += 1
                    continue
                new_id = NAME_TO_SCHEME.get(old_name)
                if new_id is None:
                    n_other += 1
                    continue
                vals = [float(x) for x in parts[1:13]]
                xs, ys = vals[4:12:2], vals[5:12:2]
                x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
                cx, cy, bw, bh = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
                lines.append(
                    f"{new_id} {fmt(cx)} {fmt(cy)} {fmt(bw)} {fmt(bh)} "
                    + " ".join(fmt(v) for v in vals)
                )
            counts["white_dropped"] += n_white
            counts["five_dropped"] += n_five
            counts["invalid_dropped"] += n_invalid
            counts["other_dropped"] += n_other
            if lines:
                write_txt(dst / "labels" / f"{stem}.txt", lines)
                link_or_copy(img, dst / "images" / img.name)
                counts["images_in_images"] += 1
                counts["labels_in_labels"] += 1
                counts["positive"] += 1
                counts["ann_kept"] += len(lines)
                status = "ok" if n_white + n_five + n_invalid + n_other == 0 else "filtered"
                manifest.append(f"{stem}\t{group}\t{status}\t{len(raw)}\t{len(lines)}\t{n_white}\t{n_five}\t{n_invalid}\t{n_other}\t{img.name}\t{stem}.txt")
            elif len(raw) == 0:
                write_txt(dst / "labels" / f"{stem}.txt", [])
                link_or_copy(img, dst / "images" / img.name)
                counts["images_in_images"] += 1
                counts["labels_in_labels"] += 1
                counts["negatives"] += 1
                manifest.append(f"{stem}\t{group}\tnegative\t0\t0\t0\t0\t0\t0\t{img.name}\t{stem}.txt")
            elif n_white + n_five == len(raw):
                counts["deleted_white5_only"] += 1
                manifest.append(f"{stem}\t{group}\tdeleted_white5_only\t{len(raw)}\t0\t{n_white}\t{n_five}\t{n_invalid}\t{n_other}\t{img.name}\t{stem}.txt")
            else:
                counts["deleted_no_valid"] += 1
                manifest.append(f"{stem}\t{group}\tdeleted_no_valid\t{len(raw)}\t0\t{n_white}\t{n_five}\t{n_invalid}\t{n_other}\t{img.name}\t{stem}.txt")
    (dst / "classes.txt").write_text("\n".join(SCHEME_NAMES) + "\n", encoding="utf-8")
    (dst / "manifest.tsv").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    write_source_readme(dst, name, counts, f"Source: `{src}`\n\nOriginal class names: {old_names}")
    return dict(counts)


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"output already exists: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "README.md").write_text(
        "# Clean Scheme-C sources\n\n"
        "Per-source images/ + labels/ (YOLO txt), no train/val split.\n\n"
        "Classes: 2 colours x 7 roles = 14.\n"
        "`white_*` and `blue_5/red_5` annotations are removed. "
        "Images whose labels contain only those objects are omitted. "
        "Raw sources are untouched.\n",
        encoding="utf-8",
    )
    (OUT / "classes.txt").write_text("\n".join(SCHEME_NAMES) + "\n", encoding="utf-8")

    summary = {}
    for ws in ("ws1", "ws2", "ws3", "ws5"):
        print(f"converting {ws} ...", flush=True)
        summary[ws] = convert_ws(ws)
    print("converting rmcv_armor ...", flush=True)
    summary["rmcv_armor"] = convert_yolo_source("rmcv_armor", ROOT / "rmcv_armor_14cls_grouped", "rmcv_armor")

    # rm_pose is intentionally not emitted here: its local images are broken
    # symlinks and 192.168.1.100 is currently unreachable.
    (OUT / "RM_POSE_BLOCKED.md").write_text(
        "# rm_pose_14cls_grouped is not included\n\n"
        "Local `rm_pose_14cls_grouped/images` contains 15,638 symlinks to "
        "`/home/dynamicx/...`; 0 images are readable. The source host "
        "192.168.1.100 was unreachable during this run, so images could not be "
        "copied. Labels remain in the original source directory.\n",
        encoding="utf-8",
    )

    rows = ["source\timages_total\tlabels_in_labels\timages_in_images\tunlabeled\tnegatives\tdeleted_white5_only\tdeleted_no_valid\tann_kept\twhite_dropped\tfive_dropped\tinvalid_dropped"]
    for name, c in summary.items():
        rows.append(
            f"{name}\t{c.get('images_total',0)}\t{c.get('labels_in_labels',0)}\t{c.get('images_in_images',0)}\t"
            f"{c.get('unlabeled',0)}\t{c.get('negatives',0)}\t{c.get('deleted_white5_only',0)}\t"
            f"{c.get('deleted_no_valid',0)}\t{c.get('ann_kept',0)}\t{c.get('white_dropped',0)}\t"
            f"{c.get('five_dropped',0)}\t{c.get('invalid_dropped',0)}"
        )
    (OUT / "source_summary.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print((OUT / "source_summary.tsv").read_text())
    print("WROTE", OUT)


if __name__ == "__main__":
    main()
