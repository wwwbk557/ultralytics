"""Run the official PoseValidator on a random val subset with the Pico scoring path.

The head scores anchors as ``colour x role sigmoid``; validation with
``deploy_boxes`` builds the same corner box the exported graph implies, so these
are the P/R/mAP numbers the trainer prints.

Usage:
    python scripts/val_pico_subset.py --n 400
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics.cfg import DEFAULT_CFG_DICT, get_cfg  # noqa: E402
from ultralytics.data.build import build_dataloader, build_yolo_dataset  # noqa: E402
from ultralytics.data.utils import check_det_dataset  # noqa: E402
from ultralytics.models.yolo.pose import PoseValidator  # noqa: E402

from ext.pico_pose.rect_train import install_rect_patches  # noqa: E402

HW = (480, 640)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=str(ROOT / "runs/pose/runs/pose/schemeC_480x640/weights/best.pt"))
    p.add_argument("--data", default=str(ROOT / "datasets/schemeC_split/data.yaml"))
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=6)
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
    box_mode = "corner-AABB" if model.model[-1].deploy_boxes else "DFL"
    print(f"\nscore: colour x role sigmoid  images={len(ds)}  boxes: {box_mode}")
    for k in keys:
        if k.startswith("metrics/") or k.startswith("fitness"):
            print(f"  {k}: {float(metrics[k]):.4f}")


if __name__ == "__main__":
    main()
