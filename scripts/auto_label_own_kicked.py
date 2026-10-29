#!/usr/bin/env python3
"""Auto-label own_kicked with last.pt (21-class reference) into 14-class YOLO pose.

Outputs a self-contained dataset next to the source:

    own_kicked_20261025_auto14/
      images/<source>/<name>.jpg   (symlink to the source image)
      labels/<source>/<name>.txt   (14-class YOLO pose, keypoint hull as bbox)
      manifest.tsv
      dataset.yaml

Images where the model finds nothing keep the old annotation converted from the
16-class ids (marked fallback=gt in the manifest) instead of becoming negatives.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ext.armor_pose.ref_infer import RefArmorPose, TARGET_NAMES

OLD16_TO_14 = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 8: 7, 9: 8, 10: 9, 11: 10, 12: 11, 13: 12, 14: 13}


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=str(ROOT / "own_kicked_20261025"))
    p.add_argument("--out", default=str(ROOT / "own_kicked_20261025_auto14"))
    p.add_argument("--weights", default=str(ROOT / "last.pt"))
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="auto")
    p.add_argument("--limit", type=int, default=0, help="only process N images, for smoke tests")
    p.add_argument("--no-gt-fallback", action="store_true")
    p.add_argument("--exist-ok", action="store_true")
    return p.parse_args()


def read_old_gt(path: Path) -> list[dict]:
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 13:
            continue
        old = int(parts[0])
        if old not in OLD16_TO_14:
            continue
        out.append({
            "cls": OLD16_TO_14[old],
            "kpts": np.array([[float(parts[5 + 2 * i]), float(parts[6 + 2 * i])] for i in range(4)], dtype=np.float64),
        })
    return out


def norm_line(cls: int, kpts: np.ndarray, w: int, h: int) -> str:
    pts = np.clip(kpts / np.array([w, h], dtype=np.float64), 0.0, 1.0)
    x1, y1 = pts[:, 0].min(), pts[:, 1].min()
    x2, y2 = pts[:, 0].max(), pts[:, 1].max()
    vals = [cls, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]
    vals += pts.reshape(-1).tolist()
    return " ".join(f"{v:.6f}" if i else str(int(v)) for i, v in enumerate(vals))


def link_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    rel = os.path.relpath(src.resolve(), dst.parent.resolve())
    os.symlink(rel, dst)


def main() -> None:
    a = parse()
    src_root, out_root = Path(a.src), Path(a.out)
    if out_root.exists() and not a.exist_ok:
        raise SystemExit(f"{out_root} exists; pass --exist-ok to merge")
    (out_root / "labels").mkdir(parents=True, exist_ok=True)
    (out_root / "images").mkdir(parents=True, exist_ok=True)

    model = RefArmorPose(a.weights, device=a.device, imgsz=a.imgsz, conf=a.conf, iou=a.iou)
    print(f"model loaded on {model.device}", flush=True)

    items = sorted(src_root.glob("*/*.jpg"))
    if a.limit:
        items = items[: a.limit]
    print("images", len(items), flush=True)

    rows = ["source\tfile\tnpred\tngt\tkept\tfallback\tclasses"]
    cls_counts = Counter()
    n_fallback = n_empty = n_pred_images = 0
    total_pred = 0

    for start in range(0, len(items), a.batch):
        chunk = items[start:start + a.batch]
        imgs, metas = [], []
        for p in chunk:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                imgs.append(None)
                metas.append((p, None, None, []))
                continue
            gt = read_old_gt(p.with_suffix(".txt"))
            imgs.append(img)
            metas.append((p, img.shape[1], img.shape[0], gt))

        valid = [i for i, im in enumerate(imgs) if im is not None]
        preds = model.predict_batch([imgs[i] for i in valid]) if valid else []
        pred_map = dict(zip(valid, preds))

        for i, (p, w, h, gt) in enumerate(metas):
            source = p.parent.name
            if w is None:
                print("unreadable", p, flush=True)
                rows.append(f"{source}\t{p.name}\t0\t0\t0\tunreadable\t")
                continue
            pr = pred_map.get(i, [])
            lab_dir = out_root / "labels" / source
            lab_dir.mkdir(parents=True, exist_ok=True)
            lab_path = lab_dir / (p.stem + ".txt")
            link_image(p, out_root / "images" / source / p.name)

            lines = []
            for o in pr:
                lines.append(norm_line(o["class_id"], np.array(o["keypoints"]), w, h))
                cls_counts[o["name"]] += 1
            fallback = ""
            if not lines and gt and not a.no_gt_fallback:
                for g in gt:
                    lines.append(norm_line(g["cls"], g["kpts"] * [w, h], w, h))
                    cls_counts[TARGET_NAMES[g["cls"]]] += 1
                fallback = "gt"
                n_fallback += 1
            if not lines:
                n_empty += 1
            if pr:
                n_pred_images += 1
                total_pred += len(pr)
            lab_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            shown = pr if pr else [{"name": TARGET_NAMES[g["cls"]], "score": None} for g in gt]
            cls_txt = ",".join(f"{o['name']}:{o['score']:.2f}" if o.get("score") is not None else f"{o['name']}:gt"
                               for o in shown)
            rows.append(f"{source}\t{p.name}\t{len(pr)}\t{len(gt)}\t{len(lines)}\t{fallback}\t{cls_txt}")
        done = min(start + a.batch, len(items))
        if done % 80 == 0 or done == len(items):
            print(f"  {done}/{len(items)}  pred_imgs={n_pred_images} fallback={n_fallback} empty={n_empty}", flush=True)

    (out_root / "manifest.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(TARGET_NAMES))
    (out_root / "dataset.yaml").write_text(
        f"path: {out_root}\ntrain: images\nval: images\nkpt_shape: [4, 2]\nflip_idx: [1, 0, 3, 2]\nnames:\n{names}\n",
        encoding="utf-8",
    )
    print("\nsummary")
    print(f"  images with preds: {n_pred_images}/{len(items)}")
    print(f"  total pred objects: {total_pred}")
    print(f"  gt fallback: {n_fallback}  empty: {n_empty}")
    print("  classes:", dict(cls_counts))
    print("out", out_root)


if __name__ == "__main__":
    main()
