"""Is low precision caused by tiny objects? Break FPs down by GT size bucket.

Matches predictions to GT (IoU 0.5, class-aware, greedy by score) on a val
subset and reports, at several confidence thresholds, how many false positives
fall near tiny/small/medium/large GTs vs pure background.

Usage:
    python scripts/check_precision_by_size.py --weights runs/.../epoch10.pt --n 600
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

EDGES = (32.0**2, 96.0**2, 192.0**2)
BUCKETS = ("tiny", "small", "medium", "large")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--data", default=str(ROOT / "datasets/schemeC_split/data.yaml"))
    p.add_argument("--n", type=int, default=600)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--multi-label", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--obj-thresh", type=float, default=None, help="enable the objectness gate at this threshold")
    return p.parse_args()


def box_iou(a, b):
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


def build_loader(data_path: str, n: int, batch: int, workers: int):
    from types import SimpleNamespace

    from ultralytics.cfg import DEFAULT_CFG_DICT
    from ultralytics.data.build import build_dataloader, build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset

    from ext.pico_pose.rect_train import install_rect_patches

    install_rect_patches()
    data = check_det_dataset(data_path)
    cfg = SimpleNamespace(**{**DEFAULT_CFG_DICT, "task": "pose", "imgsz": (480, 640), "rect": False, "cache": False})
    ds = build_yolo_dataset(cfg, data["val"], batch=batch, data=data, mode="val", rect=False, stride=32)
    idx = np.random.default_rng(0).choice(len(ds.labels), size=min(n, len(ds.labels)), replace=False)
    idx.sort()
    ds.labels = [ds.labels[i] for i in idx]
    ds.im_files = [ds.im_files[i] for i in idx]
    ds.ni = len(ds.labels)
    return build_dataloader(ds, batch=batch, workers=workers, shuffle=False), data


def main():
    args = parse_args()
    from ultralytics import YOLO
    from ultralytics.utils.nms import non_max_suppression

    loader, data = build_loader(args.data, args.n, args.batch, args.workers)
    model = YOLO(args.weights).model.eval()
    if args.obj_thresh is not None:
        head = model.model[-1]
        head.obj_threshold = args.obj_thresh
        head.deploy_filter = True

    thr_list = (0.1, 0.25, 0.5, 0.7, 0.8)
    tp = {t: {b: 0 for b in BUCKETS + ("background", "dup")} for t in thr_list}
    fp = {t: {b: 0 for b in BUCKETS + ("background", "dup")} for t in thr_list}
    n_gt = {b: 0 for b in BUCKETS}

    with torch.no_grad():
        for batch in loader:
            img = batch["img"].float() / 255
            y, _ = model(img)
            dets = non_max_suppression(y[:, :18], args.conf, args.iou, nc=14, multi_label=args.multi_label, max_det=300)
            bboxes = batch["bboxes"].numpy().reshape(-1, 4)
            bcls = batch["cls"].numpy().reshape(-1).astype(int)
            bidx = batch["batch_idx"].numpy().reshape(-1).astype(int)
            h, w = img.shape[2], img.shape[3]
            for b in range(img.shape[0]):
                m = bidx == b
                g = bboxes[m]
                if len(g):
                    gx = np.stack(
                        ((g[:, 0] - g[:, 2] / 2) * w, (g[:, 1] - g[:, 3] / 2) * h,
                         (g[:, 0] + g[:, 2] / 2) * w, (g[:, 1] + g[:, 3] / 2) * h), -1)
                    gc = bcls[m]
                    area = (gx[:, 2] - gx[:, 0]) * (gx[:, 3] - gx[:, 1])
                    gbucket = np.digitize(area, EDGES)
                    for k in range(len(BUCKETS)):
                        n_gt[BUCKETS[k]] += int((gbucket == k).sum())
                else:
                    gx, gc = np.zeros((0, 4)), np.zeros(0, int)

                p = dets[b].numpy()
                if len(p) == 0:
                    continue
                iou = box_iou(p[:, :4], gx)
                used = np.zeros(len(gx), bool)
                for i in np.argsort(-p[:, 4]):
                    same = (gc == int(p[i, 5])) & (~used)
                    hit = -1
                    if same.any():
                        j = np.argmax(np.where(same, iou[i], -1))
                        if iou[i, j] >= 0.5:
                            hit = int(j)
                    if hit >= 0:
                        used[hit] = True
                        bucket = BUCKETS[gbucket[hit]]
                        kind = "tp"
                    else:
                        if len(gx) and iou[i].max() >= 0.5:
                            bucket = BUCKETS[gbucket[int(iou[i].argmax())]]
                            kind = "dup"
                        else:
                            bucket = "background"
                            kind = "fp"
                    for t in thr_list:
                        if p[i, 4] >= t:
                            (tp if kind == "tp" else fp)[t][bucket] += 1

    total_gt = sum(n_gt.values())
    print(f"\n{args.weights}   images={len(loader.dataset) if hasattr(loader, 'dataset') else args.n}  gt={total_gt}")
    print(f"GT per bucket: {n_gt}")
    print(f"\n{'conf':>6}{'TP':>7}{'FP':>7}{'P':>7}{'R':>7}   FP breakdown (count)")
    for t in thr_list:
        tpn = sum(tp[t].values())
        fpn = sum(fp[t].values())
        p = tpn / max(tpn + fpn, 1)
        r = tpn / max(total_gt, 1)
        parts = {k: tp[t][k] + fp[t][k] for k in BUCKETS + ("background", "dup")}
        print(f"{t:>6.2f}{tpn:>7}{fpn:>7}{p:>7.3f}{r:>7.3f}   {parts}")
    # precision per bucket at each threshold (TP in bucket / (TP+FP) in bucket)
    print("\nprecision per bucket (TP / detections in that bucket)")
    for t in thr_list:
        row = []
        for b in BUCKETS + ("background", "dup"):
            d = tp[t][b] + fp[t][b]
            row.append(f"{b}={tp[t][b] / d:.2f}" if d else f"{b}=-")
        print(f"  conf>={t:.2f}: " + "  ".join(row))


if __name__ == "__main__":
    main()
