#!/usr/bin/env python3
"""Train the compact four-keypoint pose head with configurable YOLO arguments."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from ultralytics import YOLO

CFG = Path(__file__).resolve().parent / "cfg" / "yolo11-pose-pico-compact.yaml"


def _bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _cache(value: str):
    v = str(value).strip().lower()
    if v in {"", "0", "false", "none", "no", "off"}:
        return False
    if v in {"1", "true", "yes", "on"}:
        return True
    return v


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_argument_group("run")
    g.add_argument("--data", required=True, help="dataset yaml")
    g.add_argument("--cfg", default=str(CFG), help="model yaml")
    g.add_argument("--weights", default="", help="optional checkpoint or backbone weights")
    g.add_argument("--project", default="", help="results root; empty uses runs/<task>")
    g.add_argument("--name", default="compact_640")
    g.add_argument("--exist-ok", action="store_true")
    g.add_argument("--resume", default="", help="resume checkpoint path or empty")
    g.add_argument("--device", default="0")
    g.add_argument("--workers", type=int, default=12)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)

    g = p.add_argument_group("schedule")
    g.add_argument("--epochs", type=int, default=250)
    g.add_argument("--batch", type=int, default=32)
    g.add_argument("--imgsz", type=int, default=640)
    g.add_argument("--optimizer", default="AdamW")
    g.add_argument("--lr0", type=float, default=5e-4)
    g.add_argument("--lrf", type=float, default=0.01)
    g.add_argument("--momentum", type=float, default=0.937)
    g.add_argument("--weight-decay", type=float, default=5e-4)
    g.add_argument("--warmup-epochs", type=float, default=5.0)
    g.add_argument("--warmup-momentum", type=float, default=0.8)
    g.add_argument("--warmup-bias-lr", type=float, default=0.1)
    g.add_argument("--cos-lr", action=argparse.BooleanOptionalAction, default=True)
    g.add_argument("--close-mosaic", type=int, default=20)
    g.add_argument("--patience", type=int, default=0)
    g.add_argument("--nbs", type=int, default=64)
    g.add_argument("--fraction", type=float, default=1.0)
    g.add_argument("--dropout", type=float, default=0.0)
    g.add_argument("--freeze", type=int, default=0)
    g.add_argument("--single-cls", action=argparse.BooleanOptionalAction, default=False)

    g = p.add_argument_group("runtime")
    g.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    g.add_argument("--rect", action=argparse.BooleanOptionalAction, default=False)
    g.add_argument("--multi-scale", type=float, default=0.0)
    g.add_argument("--cache", default="false")
    g.add_argument("--val", action=argparse.BooleanOptionalAction, default=True)
    g.add_argument("--save", action=argparse.BooleanOptionalAction, default=True)
    g.add_argument("--save-period", type=int, default=10)
    g.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    g.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    g.add_argument("--profile", action=argparse.BooleanOptionalAction, default=False)
    g.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    g.add_argument("--max-det", type=int, default=300)

    g = p.add_argument_group("augmentation")
    g.add_argument("--hsv-h", type=float, default=0.0)
    g.add_argument("--hsv-s", type=float, default=0.05)
    g.add_argument("--hsv-v", type=float, default=0.5)
    g.add_argument("--degrees", type=float, default=4.0)
    g.add_argument("--translate", type=float, default=0.1)
    g.add_argument("--scale", type=float, default=0.3)
    g.add_argument("--shear", type=float, default=0.0)
    g.add_argument("--perspective", type=float, default=0.0)
    g.add_argument("--flipud", type=float, default=0.0)
    g.add_argument("--fliplr", type=float, default=0.0)
    g.add_argument("--bgr", type=float, default=0.0)
    g.add_argument("--mosaic", type=float, default=0.0)
    g.add_argument("--mixup", type=float, default=0.0)
    g.add_argument("--cutmix", type=float, default=0.0)
    g.add_argument("--copy-paste", type=float, default=0.0)
    g.add_argument("--auto-augment", default="none")
    g.add_argument("--erasing", type=float, default=0.4)

    g = p.add_argument_group("loss")
    g.add_argument("--box", type=float, default=7.5)
    g.add_argument("--cls", type=float, default=0.5)
    g.add_argument("--dfl", type=float, default=1.5)
    g.add_argument("--pose", type=float, default=12.0)
    g.add_argument("--kobj", type=float, default=1.0)
    g.add_argument("--color", type=float, default=0.1)
    g.add_argument("--number", type=float, default=0.1)
    g.add_argument("--quality", type=float, default=0.05)
    g.add_argument("--geom", type=float, default=0.0, help="geometry gain; 0 disables")
    g.add_argument("--geom-par", type=float, default=0.10)
    g.add_argument("--geom-opp", type=float, default=0.33)
    g.add_argument("--geom-angle", type=float, default=0.23)
    g.add_argument("--geom-axis", type=float, default=0.44)
    return p.parse_args()


def main() -> None:
    a = parse()
    model = YOLO(a.cfg)
    if a.weights and Path(a.weights).is_file():
        model.load(a.weights)
    elif a.weights:
        raise FileNotFoundError(a.weights)

    auto_augment = None if str(a.auto_augment).lower() in {"none", "null", ""} else a.auto_augment
    args = vars(a)
    for key in ("cfg", "weights"):
        args.pop(key)
    args["cache"] = _cache(a.cache)
    args["auto_augment"] = auto_augment
    args["resume"] = a.resume or False
    model.train(**args)


if __name__ == "__main__":
    main()
