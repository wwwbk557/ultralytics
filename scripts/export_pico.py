"""Export the compact Pico pose head with the deployment-parity settings baked in.

The export graph already drops ``cv2`` (box) and ``cv_joint`` (train-only scores).
This wrapper additionally applies the objectness calibration chosen at training
time (``obj_logit_bias``), by subtracting it from channel 8, so the device's
``sigmoid(z)`` is a calibrated foreground posterior again.

Usage:
    python scripts/export_pico.py --weights runs/pose/.../weights/best.pt --format onnx
    python scripts/export_pico.py --weights ... --format engine --imgsz 480,640 --half
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--weights", required=True, help="trained checkpoint (best.pt / last.pt)")
    p.add_argument("--format", default="onnx")
    p.add_argument("--imgsz", default="480,640", help="H,W (the exported graph has no letterbox)")
    p.add_argument("--obj-logit-bias", type=float, default=None, help="override the checkpoint's value")
    p.add_argument("--half", action="store_true")
    p.add_argument("--simplify", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main() -> None:
    a = parse()
    model = YOLO(a.weights)
    head = model.model.model[-1]
    ckpt_args = getattr(model.model, "args", None)
    bias = a.obj_logit_bias
    if bias is None:
        bias = float(getattr(ckpt_args, "obj_logit_bias", 0.0) or 0.0)
    head.obj_logit_bias = float(bias)
    head.export = True

    h, w = (int(v) for v in a.imgsz.replace("x", ",").split(","))
    print(
        f"[export_pico] weights={a.weights}\n"
        f"[export_pico] imgsz={(h, w)} obj_logit_bias={bias:+.3f} "
        f"(channel 8 = logit - bias, deployment applies sigmoid)"
    )
    path = model.export(format=a.format, imgsz=(h, w), half=a.half, simplify=a.simplify)
    print(f"[export_pico] -> {path}")


if __name__ == "__main__":
    main()
