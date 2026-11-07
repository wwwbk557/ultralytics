"""Run the official PoseValidator on a random val subset, optionally with the
joint-classifier score path instead of colour x role x quality.

This answers whether the low precision comes from the box branch (no: the same
boxes get much better precision with a better-trained score head) and gives the
exact P/R/mAP numbers the trainer prints.

Usage:
    python scripts/val_pico_subset.py --n 400                       # current scores
    python scripts/val_pico_subset.py --n 400 --score joint         # joint scores
"""

from __future__ import annotations

import argparse
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
from ultralytics.nn.modules.head import Pose  # noqa: E402
from ultralytics.utils.tal import make_anchors  # noqa: E402

from ext.pico_pose.rect_train import install_rect_patches  # noqa: E402

HW = (480, 640)


def joint_inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
    """Pico head inference that scores anchors with the jointly trained cv_joint branch."""
    shape = x["feats"][0].shape
    if self.dynamic or self.shape != shape:
        self.anchors, self.strides = (t.transpose(0, 1) for t in make_anchors(x["feats"], self.stride, 0.5))
        self.shape = shape
    bs = x["kpts"].shape[0]
    if self.export:
        return x["raw18"].permute(0, 2, 1).contiguous()
    kpts = self.kpts_decode(x["kpts"]).view(bs, self.nkpt, self.ndim, -1)
    dbox = self.decode_bboxes(self.dfl(x["boxes"]), self.anchors.unsqueeze(0)) * self.strides
    scores = x["scores"].sigmoid()
    return torch.cat((dbox, scores, kpts.reshape(bs, self.nk, -1)), 1)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=str(ROOT / "runs/pose/runs/pose/schemeC_480x640/weights/best.pt"))
    p.add_argument("--data", default=str(ROOT / "datasets/schemeC_split/data.yaml"))
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--score", choices=["compact", "joint"], default="compact")
    p.add_argument(
        "--obj-thresh",
        type=float,
        default=None,
        help="deployment gate: drop anchors with objectness below this value (scores stay soft)",
    )
    return p.parse_args()


def main():
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

    model = YOLO(args.weights).model.eval()
    head = model.model[-1]
    if args.obj_thresh is not None:
        # Reproduce the deployment filter: membership only, ranking score stays obj*class_prob.
        head.obj_threshold = args.obj_thresh
        head.deploy_filter = True
    if args.score == "joint":
        head._inference = MethodType(joint_inference, head)

    vargs = get_cfg(
        overrides=dict(
            model=args.weights,
            data=str(args.data),
            imgsz=HW,
            conf=0.001,
            iou=0.7,
            task="pose",
            device="cpu",
            batch=args.batch,
            plots=False,
            save_json=False,
            verbose=True,
            half=False,
            rect=False,
        )
    )
    validator = PoseValidator(args=vargs)
    validator.dataloader = loader
    validator.data = data

    orig_post = validator.postprocess

    def postprocess(preds):
        if isinstance(preds, tuple):
            preds = preds[0]
        return orig_post(preds)

    validator.postprocess = postprocess
    metrics = validator(model=model)
    keys = list(metrics.keys())
    gate = f" obj>={args.obj_thresh}" if args.obj_thresh is not None else " none (soft ranking)"
    print(f"\nscore path: {args.score}  images={len(ds)}  deploy filter:{gate}")
    for k in keys:
        if k.startswith("metrics/") or k.startswith("fitness"):
            print(f"  {k}: {float(metrics[k]):.4f}")


if __name__ == "__main__":
    main()
