"""Export the compact Pico pose head with the objectness calibration applied.

The export graph already drops ``cv2`` (box) and ``cv_joint`` (train-only scores).
This wrapper applies ``obj_logit_bias`` (= ``log(pos_weight)`` used during training)
to channel 8, so the device's ``sigmoid(z)`` is a calibrated foreground posterior.

Calibration is a one-shot additive offset in logit space: ``p = sigmoid(z - log w)``.
It may be applied exactly once per logit: the head keeps ``_obj_logit_calibrated``
False here and both the runtime path (``obj_prob``) and the graph path
(``_inference``) refuse to subtract a second time. The checkpoint file is never
written; only the exported graph carries the offset (one Sub node on channel 8).

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
    p.add_argument("--obj-logit-bias", type=float, default=None, help="override the checkpoint's value; 0 disables")
    p.add_argument("--half", action="store_true")
    p.add_argument("--simplify", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--opset", type=int, default=17, help="17 keeps onnxruntime<=1.19 and OpenVINO happy")
    return p.parse_args()


def resolve_bias(args, weights: Path) -> float:
    """CLI value, else ``train_args.obj_logit_bias`` from the checkpoint.

    ``model.model.args`` is a plain dict rebuilt from the default cfg and does not
    carry the custom key, so the checkpoint's ``train_args`` is the source of truth.
    """
    if args.obj_logit_bias is not None:
        return float(args.obj_logit_bias)
    try:
        import torch

        ta = torch.load(weights, map_location="cpu", weights_only=False).get("train_args", {}) or {}
        raw = ta.get("obj_logit_bias", 0.0) if hasattr(ta, "get") else getattr(ta, "obj_logit_bias", 0.0)
        return float(raw or 0.0)
    except Exception as e:  # noqa: BLE001
        print(f"[export_pico] could not read train_args ({e}); obj_logit_bias defaults to 0")
        return 0.0


def main() -> None:
    a = parse()
    weights = Path(a.weights)
    model = YOLO(str(weights))
    head = model.model.model[-1]
    head.export = True
    bias = resolve_bias(a, weights)
    h, w = (int(v) for v in a.imgsz.replace("x", ",").split(","))

    if not bias:
        # The checkpoint pickles the whole head, so a v2 checkpoint arrives with
        # obj_logit_bias already set; an explicit 0 must clear it or the "raw"
        # export silently stays calibrated.
        head.obj_logit_bias = 0.0
        print("[export_pico] obj_logit_bias=0, exporting without calibration")
        path = model.export(format=a.format, imgsz=(h, w), half=a.half, simplify=a.simplify, opset=a.opset)
        print(f"[export_pico] -> {path}")
        return

    # One-shot: the offset is applied by the export branch of `_inference`, and the
    # calibrated flag stays False so a second subtraction would raise elsewhere.
    head.obj_logit_bias = float(bias)
    head._obj_logit_calibrated = False
    print(
        f"[export_pico] weights={weights}\n"
        f"[export_pico] imgsz={(h, w)} obj_logit_bias={bias:+.3f} applied to channel {head.nk} "
        f"(one-shot: sigmoid(z - bias) == sigmoid(z_baked), checkpoint untouched)"
    )
    path = model.export(format=a.format, imgsz=(h, w), half=a.half, simplify=a.simplify, opset=a.opset)
    print(f"[export_pico] -> {path}")


if __name__ == "__main__":
    main()
