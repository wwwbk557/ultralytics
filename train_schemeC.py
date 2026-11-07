#!/usr/bin/env python3
"""Local Scheme-C Pico pose trainer (480x640 by default).

Training data is read through the throw-away symlink tree:
    datasets/schemeC_split/        (core: rm_pose_old + rmcv + ws5)
    datasets/schemeC_split_all/    (all clean sources)

Nothing under datasets/clean_schemeC is modified.

Examples:
    python train_schemeC.py                       # scratch, core split, 480x640
    python train_schemeC.py --data datasets/schemeC_split_all/data.yaml
    python train_schemeC.py --epochs 100 --batch 16 --lr0 3e-4
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from ultralytics import YOLO
from ultralytics.models.yolo.pose import PoseTrainer

from ext.pico_pose.rect_train import PicoRectTrainer

CFG = ROOT / "ext/pico_pose/cfg/yolo11-pose-pico-compact.yaml"
DEFAULT_DATA = ROOT / "datasets/schemeC_split/data.yaml"


def _imgsz(value: str):
    vals = [int(v) for v in str(value).replace("x", ",").split(",") if v.strip()]
    if len(vals) == 1:
        return vals[0]
    if len(vals) == 2:
        return tuple(vals)
    raise argparse.ArgumentTypeError(f"imgsz must be one or two ints, got {value!r}")


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Train Scheme-C Pico pose head locally (rectangular by default).",
    )
    g = p.add_argument_group("run")
    g.add_argument("--data", default=str(DEFAULT_DATA), help="data yaml (symlink split)")
    g.add_argument("--cfg", default=str(CFG), help="model yaml")
    g.add_argument("--weights", default="", help="optional init weights; empty = from scratch")
    g.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        metavar="LAST_PT",
        help="continue an interrupted run from weights/last.pt (keeps optimizer/EMA/schedule); "
        "omit the value to pick the newest runs/pose/*/weights/last.pt",
    )
    g.add_argument("--project", default="runs/pose", help="output project")
    g.add_argument("--name", default="schemeC_480x640", help="run name")
    g.add_argument("--exist-ok", action="store_true", help="reuse output dir")
    g.add_argument("--device", default="0", help="cuda device / cpu")
    g.add_argument("--workers", type=int, default=8)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)

    g = p.add_argument_group("schedule")
    g.add_argument("--epochs", type=int, default=250)
    g.add_argument("--batch", type=int, default=16)
    g.add_argument("--imgsz", type=_imgsz, default="480,640", help="H,W or one int; rectangular uses symlink split")
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
    g.add_argument("--max-det", type=int, default=300)

    g = p.add_argument_group("runtime")
    g.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    g.add_argument("--cache", default="false", choices=["false", "ram", "disk"], help="dataloader cache")
    g.add_argument("--multi-scale", type=float, default=0.0)
    g.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    g.add_argument("--val", action=argparse.BooleanOptionalAction, default=True)
    g.add_argument("--save", action=argparse.BooleanOptionalAction, default=True)
    g.add_argument("--save-period", type=int, default=10)
    g.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    g.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    g.add_argument("--profile", action=argparse.BooleanOptionalAction, default=False)

    g = p.add_argument_group("augmentation (rect mode keeps mosaic via RectMosaic)")
    g.add_argument("--hsv-h", type=float, default=0.015, help="hue gain; keep small for red/blue armour")
    g.add_argument("--hsv-s", type=float, default=0.7)
    g.add_argument("--hsv-v", type=float, default=0.4)
    g.add_argument("--degrees", type=float, default=5.0)
    g.add_argument("--translate", type=float, default=0.1)
    g.add_argument("--scale", type=float, default=0.5)
    g.add_argument("--shear", type=float, default=0.0)
    g.add_argument("--perspective", type=float, default=0.0)
    g.add_argument("--flipud", type=float, default=0.0, help="keep 0: vertical flip mirrors armour digits")
    g.add_argument("--fliplr", type=float, default=0.0, help="keep 0: horizontal flip mirrors armour digits")
    g.add_argument("--bgr", type=float, default=0.0)
    g.add_argument("--mosaic", type=float, default=1.0, help="rect mode uses a 2H x 2W mosaic canvas")
    g.add_argument("--mixup", type=float, default=0.0)
    g.add_argument("--cutmix", type=float, default=0.0)
    g.add_argument("--copy-paste", type=float, default=0.0, help="no-op without segment labels")
    g.add_argument("--auto-augment", default="none", help="classification-only upstream; ignored for pose")
    g.add_argument("--erasing", type=float, default=0.4)

    g = p.add_argument_group("loss")
    g.add_argument("--box", type=float, default=7.5)
    g.add_argument("--cls", type=float, default=0.5)
    g.add_argument("--dfl", type=float, default=1.5)
    g.add_argument("--pose", type=float, default=12.0)
    g.add_argument("--kobj", type=float, default=1.0)
    g.add_argument("--color", type=float, default=0.1)
    g.add_argument("--number", type=float, default=0.1, help="role loss weight")
    g.add_argument("--quality", type=float, default=0.05, help="objectness (channel 8) loss weight")
    g.add_argument("--obj-pw-pow", type=float, default=0.5, help="exponent on objectness pos_weight; 1=balanced, 0.5 background-heavy")
    g.add_argument("--quality-soft", type=float, default=0.0,
                   help="blend the deployment NMS IoU into the objectness target: tgt=(1-s)+s*IoU; 0 = hard 1/0")
    g.add_argument("--obj-logit-bias", default="auto",
                   help="objectness logit offset so sigmoid(z-bias) is calibrated; 'auto' = log(pos_weight) from the "
                        "train labels, a number fixes it, 0 disables")
    g.add_argument("--deploy-boxes", action=argparse.BooleanOptionalAction, default=True,
                   help="validate/NMS with the box built from the 4 predicted corners (deployment parity)")
    g.add_argument("--geom", type=float, default=0.0)
    g.add_argument("--geom-par", type=float, default=0.10)
    g.add_argument("--geom-opp", type=float, default=0.33)
    g.add_argument("--geom-angle", type=float, default=0.23)
    g.add_argument("--geom-axis", type=float, default=0.44)
    return p.parse_args()


def _resolve_resume(spec: str, project: str) -> Path:
    """Turn ``--resume`` into an existing last.pt path."""
    if spec in {"auto", "true", "True", ""}:
        root = Path(project)
        root = root if root.is_absolute() else ROOT / root
        cands = list(root.glob("*/weights/last.pt")) + list(root.glob("*/*/weights/last.pt"))
        if not cands:
            raise SystemExit(f"no runs/*/weights/last.pt under {root}")
        return max(cands, key=lambda p: p.stat().st_mtime)
    p = Path(spec)
    p = p if p.is_absolute() else ROOT / p
    if not p.is_file():
        raise SystemExit(f"resume checkpoint not found: {p}")
    return p


def _prepare_csv_for_resume(last_pt: Path) -> None:
    """Move a results.csv whose header predates the new monitor columns.

    ultralytics only writes the CSV header when the file does not exist, so
    appending new keys (val/P@0.5, ...) to an old header would misalign columns.
    """
    run = last_pt.parent.parent
    csv_path = run / "results.csv"
    if not csv_path.is_file() or csv_path.stat().st_size == 0:
        return
    header = csv_path.read_text().splitlines()[0]
    if "val/P@0.5" in header:
        return
    dst = run / "results_pre_resume.csv"
    i = 1
    while dst.exists():
        dst = run / f"results_pre_resume_{i}.csv"
        i += 1
    csv_path.rename(dst)
    print(f"[train_schemeC] moved {csv_path.name} -> {dst.name} (fresh header for the new monitor columns)")


def _resolve_obj_logit_bias(spec, data_yaml: Path, imgsz, pw_pow: float, topk: int = 10) -> float:
    """Turn ``--obj-logit-bias auto`` into ``log(pos_weight)`` from the train labels.

    ``pos_weight = (n_neg / n_pos) ** pw_pow`` with ``n_pos ≈ n_gt * anchors_per_gt``
    and ``n_neg ≈ n_img * anchors_per_img``. Subtracting ``log(pos_weight)`` from the
    objectness logit turns the inflated training probability back into a calibrated
    foreground posterior, so ``obj >= 0.5`` means 50% again.
    """
    if str(spec).strip().lower() not in {"auto", "true", ""}:
        return float(spec)
    try:
        import yaml

        cfg = yaml.safe_load(data_yaml.read_text())
        root = Path(cfg.get("path", data_yaml.parent))
        split = str(cfg.get("train", "images/train"))
        img_dir = Path(split) if Path(split).is_absolute() else root / split
        imgs = [p for p in img_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
        if not imgs:
            raise ValueError(f"no images under {img_dir} (check data.yaml path/split)")
        lab_dir = Path(str(img_dir).replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"))
        n_gt = sum(1 for f in lab_dir.glob("*.txt") for line in f.open() if line.strip())
        if n_gt == 0:
            raise ValueError(f"no labels under {lab_dir}")
        h, w = imgsz if isinstance(imgsz, (tuple, list)) else (imgsz, imgsz)
        anchors = sum((h // s) * (w // s) for s in (8, 16, 32))
        # TAL keeps the top-k anchors per GT but tiny/ambiguous GTs receive fewer.
        p = min(max(n_gt * topk * 2 / 3 / (len(imgs) * anchors), 1e-9), 0.5)
        return float(math.log(((1.0 - p) / p) ** pw_pow))
    except Exception as e:  # never block training because of the calibration estimate
        print(f"[train_schemeC] obj_logit_bias auto failed ({e}); using 0.0")
        return 0.0


def main() -> None:
    a = parse()

    if a.resume:
        last = _resolve_resume(a.resume, a.project)
        _prepare_csv_for_resume(last)
        ck = torch.load(last, map_location="cpu", weights_only=False)
        ck_args = ck.get("train_args", {})
        imgsz_ck = ck_args.get("imgsz", [480, 640])
        rect = isinstance(imgsz_ck, (list, tuple)) and len(imgsz_ck) == 2
        trainer = PicoRectTrainer if rect else PoseTrainer
        print(f"[train_schemeC] RESUME from {last}")
        print(f"[train_schemeC] ckpt epoch={ck.get('epoch', -1)} epochs={ck_args.get('epochs')} "
              f"imgsz={imgsz_ck} lr0={ck_args.get('lr0')} name={ck_args.get('name')}")
        print(f"[train_schemeC] trainer={trainer.__name__} (device={a.device}, workers={a.workers})")
        print(f"[train_schemeC] pico eval: deploy_boxes={ck_args.get('deploy_boxes', True)} "
              f"quality_soft={ck_args.get('quality_soft', 0.0)} obj_logit_bias={ck_args.get('obj_logit_bias', 0.0)}")
        model = YOLO(str(last))
        model.train(trainer=trainer, resume=str(last), device=a.device, workers=a.workers)
        return

    data = Path(a.data)
    if not data.is_file():
        raise SystemExit(f"data yaml not found: {data}")
    a.obj_logit_bias = _resolve_obj_logit_bias(a.obj_logit_bias, data, a.imgsz, a.obj_pw_pow)
    cache = False if str(a.cache).lower() == "false" else a.cache
    auto_augment = None if str(a.auto_augment).lower() in {"none", "null", ""} else a.auto_augment

    init_weights = a.weights
    model = YOLO(a.cfg)
    if init_weights:
        model.load(init_weights)

    args = vars(a)
    for key in ("cfg", "weights"):
        args.pop(key)
    args["cache"] = cache
    args["auto_augment"] = auto_augment

    rect = isinstance(a.imgsz, tuple)
    trainer = PicoRectTrainer if rect else PoseTrainer
    print(f"[train_schemeC] data={a.data}")
    print(f"[train_schemeC] imgsz={a.imgsz} rect={rect} trainer={trainer.__name__}")
    print(f"[train_schemeC] init_weights={init_weights or 'FROM SCRATCH'}")
    print(f"[train_schemeC] quality_soft={a.quality_soft} obj_logit_bias={a.obj_logit_bias:+.3f} "
          f"(channel8: sigmoid(z - bias)) deploy_boxes={a.deploy_boxes}")
    aug_keys = ("hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "fliplr", "mosaic", "mixup", "cutmix", "erasing")
    print("[train_schemeC] aug=" + " ".join(f"{k}={args[k]}" for k in aug_keys))
    model.train(trainer=trainer, **args)


if __name__ == "__main__":
    main()
