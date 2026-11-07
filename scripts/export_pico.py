"""Export the compact Pico pose head with the deployment-parity settings baked in.

The export graph already drops ``cv2`` (box) and ``cv_joint`` (train-only scores).
This wrapper additionally folds the objectness calibration ``obj_logit_bias``
(``log(pos_weight)``) into ``cv_compact.bias[8]`` on a **copy** of the state dict,
so the device's ``sigmoid(z)`` is a calibrated foreground posterior and the ONNX
graph carries no extra subtract node.

Calibration is a one-shot additive offset in logit space: ``p = sigmoid(z - log w)``.
It may be applied exactly once per logit. The head therefore records
``_obj_logit_calibrated`` while the baked weights are loaded, and both the
runtime path (``obj_prob``) and the graph path (``_inference``) refuse to subtract
a second time. The checkpoint file itself is never written.

Usage:
    python scripts/export_pico.py --weights runs/pose/.../weights/best.pt --format onnx
    python scripts/export_pico.py --weights ... --format engine --imgsz 480,640 --half
"""

from __future__ import annotations

import argparse
import copy
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


def bake_obj_bias(head, bias: float):
    """Fold ``bias`` into ``cv_compact.bias[8]`` on a deep copy of the state dict.

    Returns ``(restore, baked_state)``; the caller must call ``restore()`` in a
    ``finally`` block so the in-memory model goes back to the uncalibrated state.
    """
    original = copy.deepcopy(head.state_dict())
    baked = copy.deepcopy(original)
    for i in range(head.nl):
        key = f"cv_compact.{i}.bias"
        if key not in baked:
            raise SystemExit(f"unexpected head layout: {key} missing")
        baked[key][head.nk] -= bias

    def restore() -> None:
        head.load_state_dict(original)
        head._obj_logit_calibrated = False

    return restore, baked


def main() -> None:
    a = parse()
    model = YOLO(a.weights)
    head = model.model.model[-1]
    ckpt_args = getattr(model.model, "args", None)
    bias = a.obj_logit_bias
    if bias is None:
        bias = float(getattr(ckpt_args, "obj_logit_bias", 0.0) or 0.0)

    h, w = (int(v) for v in a.imgsz.replace("x", ",").split(","))
    head.export = True

    if not bias:
        print("[export_pico] obj_logit_bias=0, exporting without calibration")
        path = model.export(format=a.format, imgsz=(h, w), half=a.half, simplify=a.simplify)
        print(f"[export_pico] -> {path}")
        return

    restore, baked = bake_obj_bias(head, bias)
    try:
        head.load_state_dict(baked)
        # The bias now lives in the weights: the runtime offset must be off, and
        # the flag makes any second subtraction fail loudly.
        head.obj_logit_bias = 0.0
        head._obj_logit_calibrated = True
        print(
            f"[export_pico] weights={a.weights}\n"
            f"[export_pico] imgsz={(h, w)} obj_logit_bias={bias:+.3f} baked into cv_compact.bias[{head.nk}] "
            f"(one-shot: sigmoid(z - bias) == sigmoid(z_baked), checkpoint untouched)"
        )
        path = model.export(format=a.format, imgsz=(h, w), half=a.half, simplify=a.simplify)
    finally:
        restore()
        head.obj_logit_bias = bias
    print(f"[export_pico] -> {path}")


if __name__ == "__main__":
    main()
