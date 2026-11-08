"""Measure pose-metric parity between the *training* score path and the *export* path.

The compact head keeps an auxiliary box branch (``cv2`` + DFL) for TAL assignment
and for validation. The exported graph only runs ``cv_compact`` (raw18), so the
deployment decoder builds the NMS box from the four predicted corners:

    bbox = (min(x), min(y), max(x), max(y)) over the 4 keypoints

This script runs the stock ``PoseValidator`` on a fixed random val subset three
times, changing only how the NMS box is produced:

* ``box``     - box branch + DFL (what ``metrics/mAP50-95(P)`` currently reports)
* ``kpts``    - box derived from the 4 decoded corners (what deployment does)
* ``joint``   - box branch, scores from the train-only ``cv_joint`` head (reference)

Same weights, same subset, same seed, so the delta is purely the box source.

Usage:
    python scripts/val_pico_deploy_parity.py --weights runs/pose/.../weights/last.pt --n 192
    python scripts/val_pico_deploy_parity.py --weights ... --mode kpts --obj-thresh 0.05
"""

from __future__ import annotations

import argparse
import json
import sys
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
from ultralytics.utils.tal import make_anchors  # noqa: E402

from ext.pico_pose.rect_train import install_rect_patches  # noqa: E402

HW = (480, 640)


def _decode(module, x):
    """Shared part of ``PicoPoseCompactAuxBox._inference``: anchors + keypoints."""
    shape = x["feats"][0].shape
    if module.dynamic or module.shape != shape:
        module.anchors, module.strides = (t.transpose(0, 1) for t in make_anchors(x["feats"], module.stride, 0.5))
        module.shape = shape
    bs = x["kpts"].shape[0]
    kpts = module.kpts_decode(x["kpts"]).view(bs, module.nkpt, module.ndim, -1)
    # ``shape`` is the P3 *feature* grid; convert back to input pixels with its stride.
    s0 = float(module.stride[0])
    return bs, kpts, shape[2] * s0, shape[3] * s0


def kpt_box_inference(self, x):
    """Export-parity inference: NMS box = axis-aligned extent of the 4 corners."""
    bs, kpts, h, w = _decode(self, x)
    xs, ys = kpts[:, :, 0], kpts[:, :, 1]
    x0 = xs.min(1).values.clamp(0, w)
    y0 = ys.min(1).values.clamp(0, h)
    x1 = xs.max(1).values.clamp(0, w)
    y1 = ys.max(1).values.clamp(0, h)
    # non_max_suppression expects xywh (it calls xywh2xyxy internally).
    dbox = torch.cat(
        (
            ((x0 + x1) * 0.5)[:, None],
            ((y0 + y1) * 0.5)[:, None],
            (x1 - x0)[:, None],
            (y1 - y0)[:, None],
        ),
        1,
    )
    scores = self._class_scores(x["quality"].sigmoid(), x["color"].softmax(1), x["number"].softmax(1))
    return torch.cat((dbox, scores, kpts.reshape(bs, self.nk, -1)), 1)


def box_branch_inference(self, x):
    """Stock validation inference: NMS box from cv2 + DFL (baseline)."""
    bs, kpts, _, _ = _decode(self, x)
    dbox = self.decode_bboxes(self.dfl(x["boxes"]), self.anchors.unsqueeze(0)) * self.strides
    scores = self._class_scores(x["quality"].sigmoid(), x["color"].softmax(1), x["number"].softmax(1))
    return torch.cat((dbox, scores, kpts.reshape(bs, self.nk, -1)), 1)


def joint_inference(self, x):
    """Reference: train-only joint classifier provides the scores (not deployable)."""
    bs, kpts, _, _ = _decode(self, x)
    dbox = self.decode_bboxes(self.dfl(x["boxes"]), self.anchors.unsqueeze(0)) * self.strides
    return torch.cat((dbox, x["scores"].sigmoid(), kpts.reshape(bs, self.nk, -1)), 1)


MODES = {"box": box_branch_inference, "kpts": kpt_box_inference, "joint": joint_inference}


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--data", default=str(ROOT / "datasets/schemeC_split/data.yaml"))
    p.add_argument("--n", type=int, default=192)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--mode", choices=sorted(MODES), default="box")
    p.add_argument("--obj-thresh", type=float, default=None, help="emulate the deploy quality gate")
    p.add_argument("--obj-bias", type=float, default=None, help="objectness logit offset (log pos_weight)")
    p.add_argument("--builtin", action="store_true", help="use the head's own deploy_boxes path (applies obj_logit_bias)")
    p.add_argument("--iou", type=float, default=0.7, help="NMS IoU (deploy node defaults to 0.45)")
    p.add_argument("--conf", type=float, default=0.001, help="pre-NMS conf used by the validator")
    p.add_argument("--json", action="store_true", help="print one json line instead of a table")
    return p.parse_args()


def main() -> None:
    args = parse()
    install_rect_patches()
    data = check_det_dataset(args.data)
    cfg = SimpleNamespace(**{**DEFAULT_CFG_DICT, "task": "pose", "imgsz": HW, "rect": False, "cache": False})
    ds = build_yolo_dataset(cfg, data["val"], batch=args.batch, data=data, mode="val", rect=False, stride=32)
    idx = np.random.default_rng(0).choice(len(ds.labels), size=min(args.n, len(ds.labels)), replace=False)
    idx.sort()
    ds.labels = [ds.labels[i] for i in idx]
    ds.im_files = [ds.im_files[i] for i in idx]
    ds.ni = len(ds.labels)
    loader = build_dataloader(ds, batch=args.batch, workers=args.workers, shuffle=False)

    from ultralytics import YOLO

    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    model = YOLO(args.weights).model.eval()
    head = model.model[-1]
    if args.builtin:
        head.deploy_boxes = True   # head._inference -> _corner_bboxes + obj_prob(bias)
    else:
        head._inference = MethodType(MODES[args.mode], head)
    if args.obj_thresh is not None:
        head.obj_threshold = float(args.obj_thresh)
        head.deploy_filter = True
    if args.obj_bias is not None:
        head.obj_logit_bias = float(args.obj_bias)

    vargs = get_cfg(
        overrides=dict(
            model=args.weights,
            data=str(args.data),
            imgsz=HW,
            conf=args.conf,
            iou=args.iou,
            task="pose",
            device="cpu",
            batch=args.batch,
            plots=False,
            save_json=False,
            verbose=False,
            half=False,
            rect=False,
        )
    )
    validator = PoseValidator(args=vargs)
    validator.dataloader = loader
    validator.data = data
    orig_post = validator.postprocess

    def postprocess(preds):
        return orig_post(preds[0] if isinstance(preds, tuple) else preds)

    validator.postprocess = postprocess
    metrics = validator(model=model)

    keep = ("metrics/precision(P)", "metrics/recall(P)", "metrics/mAP50(P)", "metrics/mAP50-95(P)",
            "metrics/mAP50(B)", "metrics/mAP50-95(B)")
    out = {"weights": args.weights, "mode": args.mode, "n": len(ds), "epoch": ckpt.get("epoch"),
           "obj_thresh": args.obj_thresh, "iou": args.iou}
    out.update({k.split("/")[-1]: round(float(metrics[k]), 5) for k in keep})
    if args.json:
        print(json.dumps(out))
    else:
        print(f"\n{Path(args.weights).parent.name}  epoch={out['epoch']}  mode={args.mode}  n={out['n']}  gate={args.obj_thresh}")
        for k in keep:
            print(f"  {k:24s} {float(metrics[k]):.4f}")


if __name__ == "__main__":
    main()
