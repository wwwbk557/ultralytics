"""Per-class / per-attribute error breakdown for the Pico pose head.

Answers "which class is dragging precision down": runs the stock PoseValidator
(for the official per-class AP50 / AP50-95) and a second pass with class-agnostic
OKS matching that separates the four ways a detection can be wrong:

    tp        correct class, OKS >= 0.5
    wrong_cls correct localization, wrong class  (colour/role confusion)
    dup       OKS >= 0.5 but that GT is already taken (NMS did not merge)
    bg_fp     no GT with OKS >= 0.5 (background / localisation failure)
    fn        GT with no detection at all

Usage:
    python scripts/pico_per_class.py --weights ... --n 768 --mode kpts --iou 0.45
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics.cfg import DEFAULT_CFG_DICT, get_cfg  # noqa: E402
from ultralytics.data.build import build_dataloader, build_yolo_dataset  # noqa: E402
from ultralytics.data.utils import check_det_dataset  # noqa: E402
from ultralytics.models.yolo.pose import PoseValidator  # noqa: E402
from ultralytics.utils.metrics import kpt_iou  # noqa: E402
from ultralytics.utils.nms import non_max_suppression  # noqa: E402

from ext.pico_pose.attrs import COLOR_NAMES, ROLE_NAMES  # noqa: E402
from ext.pico_pose.rect_train import install_rect_patches  # noqa: E402
from scripts.val_pico_deploy_parity import MODES  # noqa: E402

HW = (480, 640)
OKS_THR = 0.5


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--data", default=str(ROOT / "datasets/schemeC_split/data.yaml"))
    p.add_argument("--n", type=int, default=768)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--mode", choices=sorted(MODES), default="kpts")
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--conf", type=float, default=0.25, help="operating point for the error breakdown")
    p.add_argument("--obj-thresh", type=float, default=None, help="emulate the deploy objectness gate")
    return p.parse_args()


def build_subset(args):
    data = check_det_dataset(args.data)
    cfg = SimpleNamespace(**{**DEFAULT_CFG_DICT, "task": "pose", "imgsz": HW, "rect": False, "cache": False})
    ds = build_yolo_dataset(cfg, data["val"], batch=args.batch, data=data, mode="val", rect=False, stride=32)
    idx = np.random.default_rng(0).choice(len(ds.labels), size=min(args.n, len(ds.labels)), replace=False)
    idx.sort()
    ds.labels = [ds.labels[i] for i in idx]
    ds.im_files = [ds.im_files[i] for i in idx]
    ds.ni = len(ds.labels)
    return data, ds


@torch.no_grad()
def error_breakdown(model, loader, conf: float, iou: float, kpt_shape=(4, 2)):
    head = model.model[-1]
    nc = head.nc
    nk = int(np.prod(kpt_shape))
    sigma = torch.full((kpt_shape[0],), 0.05)
    stats = {k: Counter() for k in ("gt", "tp", "wrong", "dup", "bg_fp", "fn")}
    curve = {t: {"tp": 0, "fp": 0} for t in (0.3, 0.4, 0.5, 0.6)}
    color_conf = np.zeros((2, 2), np.int64)  # [pred, gt] for wrong-class matches
    role_conf = np.zeros((7, 7), np.int64)
    for batch in loader:
        img = batch["img"].float() / 255.0
        out = model(img)
        out = out[0] if isinstance(out, tuple) else out
        det_all = non_max_suppression(out, conf, iou, nc=nc)
        for b, det in enumerate(det_all):
            m = batch["batch_idx"] == b
            gb = batch["bboxes"][m].float()
            gk = batch["keypoints"][m].float()
            gc = batch["cls"][m].view(-1).long()
            h, w = img.shape[2], img.shape[3]
            gxyxy = torch.cat(((gb[:, :2] - gb[:, 2:] / 2) * torch.tensor([w, h]), (gb[:, :2] + gb[:, 2:] / 2) * torch.tensor([w, h])), 1)
            gpx = gk[..., :2] * torch.tensor([w, h])
            area = ((gxyxy[:, 2] - gxyxy[:, 0]) * (gxyxy[:, 3] - gxyxy[:, 1])).clamp_min(1.0)
            for c in gc.tolist():
                stats["gt"][c] += 1
            if det is None or len(det) == 0:
                for c in gc.tolist():
                    stats["fn"][c] += 1
                continue
            pcls = det[:, 5].long()
            pk = det[:, 6 : 6 + nk].view(len(det), kpt_shape[0], 2)
            oks = kpt_iou(
                torch.cat((gpx, torch.ones(len(gpx), kpt_shape[0], 1)), -1),
                torch.cat((pk, torch.ones(len(pk), kpt_shape[0], 1)), -1),
                area,
                sigma.tolist(),
            )  # (G, D)
            used = np.zeros(len(gpx), bool)
            for i in range(len(det)):
                cls_i = int(pcls[i])
                if len(gpx) == 0:
                    stats["bg_fp"][cls_i] += 1
                    for t, c in curve.items():
                        if float(det[i, 4]) >= t:
                            c["fp"] += 1
                    continue
                col = oks[:, i].numpy()
                free = col.copy()
                free[used] = -1.0
                j = int(np.argmax(free))
                if free[j] < OKS_THR:
                    stats["dup" if col.max() >= OKS_THR else "bg_fp"][cls_i] += 1
                    for t, c in curve.items():
                        if float(det[i, 4]) >= t:
                            c["fp"] += 1
                    continue
                used[j] = True
                gc_j = int(gc[j])
                for t, c in curve.items():
                    if float(det[i, 4]) >= t:
                        c["tp" if cls_i == gc_j else "fp"] += 1
                if cls_i == gc_j:
                    stats["tp"][cls_i] += 1
                else:
                    stats["wrong"][cls_i] += 1
                    color_conf[cls_i // 7, gc_j // 7] += 1
                    role_conf[cls_i % 7, gc_j % 7] += 1
            for j in range(len(gpx)):
                if not used[j]:
                    stats["fn"][int(gc[j])] += 1
    return stats, color_conf, role_conf, curve


def main():
    args = parse()
    install_rect_patches()
    data, ds = build_subset(args)
    loader = build_dataloader(ds, batch=args.batch, workers=args.workers, shuffle=False)
    from ultralytics import YOLO

    model = YOLO(args.weights).model.eval()
    model.model[-1]._inference = MethodType(MODES[args.mode], model.model[-1])
    if args.obj_thresh is not None:
        model.model[-1].obj_threshold = float(args.obj_thresh)
        model.model[-1].deploy_filter = True

    vargs = get_cfg(
        overrides=dict(model=args.weights, data=str(args.data), imgsz=HW, conf=0.001, iou=args.iou, task="pose",
                       device="cpu", batch=args.batch, plots=False, save_json=False, verbose=False, half=False, rect=False)
    )
    validator = PoseValidator(args=vargs)
    validator.dataloader = loader
    validator.data = data
    orig_post = validator.postprocess
    validator.postprocess = lambda preds: orig_post(preds[0] if isinstance(preds, tuple) else preds)
    m = validator(model=model)
    vm = validator.metrics

    stats, color_conf, role_conf, curve = error_breakdown(model, loader, args.conf, args.iou)
    names = data["names"]
    print(f"\nweights={Path(args.weights).parent.name} mode={args.mode} iou={args.iou} n={len(ds)} "
          f"conf={args.conf} obj_gate={args.obj_thresh}  ({sum(stats['gt'].values())} GT)")
    print(f"{'cls':<16}{'n_gt':>6}{'TP':>6}{'wrong':>7}{'dup':>6}{'bgFP':>6}{'FN':>6}{'P':>7}{'R':>7}"
          f"{'AP50':>7}{'AP50-95':>8}")
    for c in range(len(names)):
        g = stats["gt"][c]
        tp, wr, dp, bg, fn = (stats[k][c] for k in ("tp", "wrong", "dup", "bg_fp", "fn"))
        prec = tp / max(tp + wr + dp + bg, 1)
        rec = tp / max(g, 1)
        print(f"{c:>2} {names[c]:<13}{g:>6}{tp:>6}{wr:>7}{dp:>6}{bg:>6}{fn:>6}{prec:>7.3f}{rec:>7.3f}"
              f"{vm.pose.ap50[c]:>7.3f}{vm.pose.ap[c]:>8.3f}")
    tot = {k: sum(v.values()) for k, v in stats.items()}
    print(f"\ntotal GT={tot['gt']} TP={tot['tp']} wrong={tot['wrong']} dup={tot['dup']} "
          f"bgFP={tot['bg_fp']} FN={tot['fn']}  P={tot['tp']/max(tot['tp']+tot['wrong']+tot['dup']+tot['bg_fp'],1):.3f} "
          f"R={tot['tp']/max(tot['gt'],1):.3f}")

    print("\ndeploy operating curve (kpt boxes):")
    print(f"{'conf>=':>7}{'TP':>7}{'FP':>7}{'P':>7}{'R':>7}")
    for t, c in curve.items():
        print(f"{t:>7.1f}{c['tp']:>7}{c['fp']:>7}{c['tp']/max(c['tp']+c['fp'],1):>7.3f}{c['tp']/max(tot['gt'],1):>7.3f}")

    print("\ncolour confusion (wrong-class matches, pred x gt):")
    print("        " + "".join(f"{n:>7}" for n in COLOR_NAMES))
    for i, n in enumerate(COLOR_NAMES):
        print(f"  {n:<6}" + "".join(f"{color_conf[i, j]:>7}" for j in range(2)))
    print("\nrole confusion (wrong-class matches, pred x gt):")
    print("           " + "".join(f"{n:>9}" for n in ROLE_NAMES))
    for i, n in enumerate(ROLE_NAMES):
        print(f"  {n:<9}" + "".join(f"{role_conf[i, j]:>9}" for j in range(7)))


if __name__ == "__main__":
    main()
