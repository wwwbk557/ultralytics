"""Diagnose the low box precision of the Pico pose head.

The head exposes three different class-score paths:

  A. colour(softmax) x role(softmax) x quality(sigmoid)  <- what inference uses
  B. colour(softmax) x role(softmax)                     <- no quality gate
  C. joint classifier sigmoid                            <- the branch TAL/cls loss trains
  D. joint classifier x quality

``criterion`` optimises C (TAL assignment + BCE), but ``_inference`` emits A.
This script measures AP50 / precision / recall for each variant on a val subset
and breaks the false positives down into "background" vs "duplicate/offset"
so it is clear whether the box branch or the score path is at fault.

Usage:
    python scripts/diag_pico_precision.py --weights runs/.../best.pt --n 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402
from ultralytics.cfg import DEFAULT_CFG_DICT  # noqa: E402
from ultralytics.data.build import build_dataloader, build_yolo_dataset  # noqa: E402
from ultralytics.data.utils import check_det_dataset  # noqa: E402
from ultralytics.utils.metrics import ap_per_class  # noqa: E402
from ultralytics.utils.nms import non_max_suppression  # noqa: E402

from ext.pico_pose.rect_train import install_rect_patches  # noqa: E402

HW = (480, 640)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=str(ROOT / "runs/pose/runs/pose/schemeC_480x640/weights/best.pt"))
    p.add_argument("--data", default=str(ROOT / "datasets/schemeC_split/data.yaml"))
    p.add_argument("--n", type=int, default=200, help="number of val images")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--conf", type=float, default=0.001, help="NMS conf floor (val default)")
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--nc", type=int, default=14)
    p.add_argument("--workers", type=int, default=4)
    return p.parse_args()


def build_loader(data_path: str, n: int, batch: int, workers: int = 4):
    install_rect_patches()
    data = check_det_dataset(data_path)
    cfg = SimpleNamespace(**{**DEFAULT_CFG_DICT, "task": "pose", "imgsz": HW, "rect": False, "cache": False, "augment": False})
    ds = build_yolo_dataset(cfg, data["val"], batch=batch, data=data, mode="val", rect=False, stride=32)
    idx = np.random.default_rng(0).choice(len(ds.labels), size=min(n, len(ds.labels)), replace=False)
    idx.sort()
    ds.labels = [ds.labels[i] for i in idx]
    ds.im_files = [ds.im_files[i] for i in idx]
    ds.ni = len(ds.labels)
    return build_dataloader(ds, batch=batch, workers=workers, shuffle=False), data


def score_variants(y: torch.Tensor, head_preds: dict) -> dict[str, torch.Tensor]:
    """Return class-score blocks for A/B/C/D, each (B, nc, A)."""
    scores_a = y[:, 4:18]  # already colour x role x quality
    color = head_preds["color"].softmax(1)
    number = head_preds["number"].softmax(1)
    q = head_preds["quality"].sigmoid()
    scores_b = (color[:, :, None] * number[:, None, :]).reshape(y.shape[0], -1, y.shape[2])
    scores_c = head_preds["scores"].sigmoid()
    return {"A colour*role*q": scores_a, "B colour*role": scores_b, "C joint": scores_c, "D joint*q": scores_c * q}


def run_nms(boxes: torch.Tensor, scores: torch.Tensor, conf: float, iou: float, nc: int):
    pred = torch.cat((boxes, scores), 1)  # (B, 4+nc, A), boxes are decoded xywh pixels
    return non_max_suppression(pred, conf_thres=conf, iou_thres=iou, nc=nc, max_det=300)


def xywh_to_xyxy(b: np.ndarray) -> np.ndarray:
    out = b.copy()
    out[:, 0] = b[:, 0] - b[:, 2] / 2
    out[:, 1] = b[:, 1] - b[:, 3] / 2
    out[:, 2] = b[:, 0] + b[:, 2] / 2
    out[:, 3] = b[:, 1] + b[:, 3] / 2
    return out


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix (len(a), len(b)) for xyxy boxes."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None] - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def match_image(pred: np.ndarray, gt: np.ndarray, nc: int, iou_thr: float = 0.5):
    """Greedy per-class matching. Returns (tp flags, best_iou_any, matched_cls)."""
    p_boxes, g_boxes = pred[:, :4], gt[:, :4]
    iou = box_iou(p_boxes, g_boxes)  # (P, G)
    best_iou_any = iou.max(1) if iou.size else np.zeros(len(pred), np.float32)
    tp = np.zeros(len(pred), bool)
    used = np.zeros(len(gt), bool)
    order = np.argsort(-pred[:, 4])
    for i in order:
        same = (gt[:, 4] == pred[i, 5]) & (~used)
        if not same.any():
            continue
        j = np.argmax(np.where(same, iou[i], -1))
        if iou[i, j] >= iou_thr:
            tp[i] = True
            used[j] = True
    return tp, best_iou_any


def ap50(tp: list[np.ndarray], conf: list[np.ndarray], cls: list[np.ndarray], n_gt: np.ndarray, nc: int):
    """101-point AP50 per class (mean over classes)."""
    tp = np.concatenate(tp) if tp else np.zeros(0, bool)
    conf = np.concatenate(conf) if conf else np.zeros(0, np.float32)
    cls = np.concatenate(cls) if cls else np.zeros(0, np.int64)
    aps = []
    for c in range(nc):
        m = cls == c
        n = n_gt[c]
        if n == 0:
            continue
        p_tp, p_conf = tp[m], conf[m]
        order = np.argsort(-p_conf)
        p_tp = p_tp[order]
        cum_tp = np.cumsum(p_tp)
        cum_fp = np.cumsum(~p_tp)
        rec = cum_tp / n
        prec = cum_tp / np.maximum(cum_tp + cum_fp, 1)
        mrec = np.linspace(0, 1, 101)
        ap = np.mean([prec[rec >= r].max() if (rec >= r).any() else 0.0 for r in mrec])
        aps.append(ap)
    return float(np.mean(aps)) if aps else 0.0


def main():
    args = parse()
    torch.set_num_threads(max(1, (torch.get_num_threads() or 8)))
    loader, data = build_loader(args.data, args.n, args.batch, args.workers)
    model = YOLO(args.weights).model.eval()
    names = data["names"]

    per_variant = {k: {"tp": [], "conf": [], "cls": [], "conf_all": [], "tp_all": []} for k in
                   ("A colour*role*q", "B colour*role", "C joint", "D joint*q")}
    per_variant["oracle cls"] = {"tp": [], "conf": [], "cls": [], "conf_all": [], "tp_all": []}
    fp_kind = {"背景(与任何GT IoU<0.5)": 0, "重复框/定位差(IoU>=0.5)": 0}
    n_gt = np.zeros(args.nc, np.int64)
    max_scores = []
    n_images = 0

    with torch.no_grad():
        for batch in loader:
            if n_images >= args.n:
                break
            img = batch["img"].float() / 255.0
            y, head_preds = model(img)
            variants = score_variants(y, head_preds)
            dets = {k: run_nms(y[:, :4], v, args.conf, args.iou, args.nc) for k, v in variants.items()}
            dets["oracle cls"] = run_nms(y[:, :4], variants["A colour*role*q"], args.conf, args.iou, args.nc)

            bboxes = batch["bboxes"].numpy().reshape(-1, 4)
            bcls = batch["cls"].numpy().reshape(-1).astype(int)
            bidx = batch["batch_idx"].numpy().reshape(-1).astype(int)
            H, W = img.shape[2], img.shape[3]
            for b in range(img.shape[0]):
                if n_images >= args.n:
                    break
                m = bidx == b
                g = bboxes[m]
                gt = np.zeros((len(g), 5), np.float32)
                gx = xywh_to_xyxy(g * np.array([W, H, W, H], np.float32))
                gt[:, :4] = gx
                gt[:, 4] = bcls[m]
                n_gt += np.bincount(gt[:, 4].astype(int), minlength=args.nc)
                max_scores.append(float(variants["A colour*role*q"][b].max()))
                for k, d in dets.items():
                    p = d[b].numpy() if len(d[b]) else np.zeros((0, 6), np.float32)
                    if k == "oracle cls" and len(p):
                        iou = box_iou(p[:, :4], gt[:, :4]) if len(gt) else np.zeros((len(p), 0))
                        if iou.size:
                            j = iou.argmax(1)
                            ok = iou[np.arange(len(p)), j] >= 0.5
                            p[ok, 5] = gt[j[ok], 4]
                    tp, best_iou = match_image(p, gt, args.nc)
                    per_variant[k]["tp"].append(tp)
                    per_variant[k]["conf"].append(p[:, 4])
                    per_variant[k]["cls"].append(p[:, 5].astype(int))
                    if k == "A colour*role*q":
                        n_pred_pos = (p[:, 4] >= 0.25)
                        for i in np.where(n_pred_pos)[0]:
                            if not tp[i]:
                                if best_iou[i] >= 0.5:
                                    fp_kind["重复框/定位差(IoU>=0.5)"] += 1
                                else:
                                    fp_kind["背景(与任何GT IoU<0.5)"] += 1
                n_images += 1
            if n_images % 50 < args.batch:
                print(f"  scanned {n_images} images", flush=True)

    total_gt = int(n_gt.sum())
    print(f"\nimages={n_images}  gt={total_gt}  per-class gt={dict(zip(names.values(), n_gt.tolist()))}")
    print(f"mean max class score: {np.mean(max_scores):.3f}")
    print(f"\n{'variant':<20}{'AP50':>8}{'P':>8}{'R':>8}{'F1':>8}   (ultralytics ap_per_class semantics)")
    for k, d in per_variant.items():
        tp = np.concatenate(d["tp"]) if d["tp"] else np.zeros(0, bool)
        cf = np.concatenate(d["conf"]) if d["conf"] else np.zeros(0, np.float32)
        cl = np.concatenate(d["cls"]) if d["cls"] else np.zeros(0, np.int64)
        target_cls = np.repeat(np.arange(args.nc), n_gt)
        tp_, fp_, p_, r_, f1_, ap_, *_ = ap_per_class(tp[:, None], cf, cl, target_cls)
        print(f"{k:<20}{ap_[:, 0].mean():>8.3f}{p_.mean():>8.3f}{r_.mean():>8.3f}{f1_.mean():>8.3f}")
    print(f"\nconf>=0.25 FP breakdown (variant A): {fp_kind}")
    print("\nper-class AP50 (variant A vs C):")
    for c, name in names.items():
        line = f"  {name:<16}"
        for k in ("A colour*role*q", "C joint"):
            d = per_variant[k]
            # per-class AP50 over all images
            tp = np.concatenate(d["tp"]); cf = np.concatenate(d["conf"]); cl = np.concatenate(d["cls"])
            m = cl == c
            n = n_gt[c]
            if n == 0:
                line += f"{k:>18}: {'-':>6}"
                continue
            o = np.argsort(-cf[m]); t = tp[m][o]
            ct = np.cumsum(t); cfp = np.cumsum(~t)
            rec = ct / n
            prec = ct / np.maximum(ct + cfp, 1)
            mrec = np.linspace(0, 1, 101)
            ap = np.mean([prec[rec >= r].max() if (rec >= r).any() else 0.0 for r in mrec])
            line += f"{k:>18}: {ap:>6.3f}"
        print(line)


if __name__ == "__main__":
    main()
