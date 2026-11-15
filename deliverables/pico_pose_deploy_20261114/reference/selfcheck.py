"""Self-check the ONNX contract and the reference decoder against the golden sample.

Checks:
  1. md5 of each ONNX file matches ``../weights/md5sums.txt``;
  2. graph I/O names, shapes and dtypes match the contract;
  3. decoding ``sample.jpg`` reproduces ``expected_detections.json``.

Usage:
    python selfcheck.py
    python selfcheck.py --onnx ../weights/schemeC_ws123val_scratch250_20261114_last_fp16.onnx
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from decode import INPUT_H, INPUT_W, infer, COLOR_NAMES, ROLE_NAMES, CLASS_NAMES  # noqa: E402

BOX_ATOL = 0.5   # px
KPT_ATOL = 0.5   # px
SCORE_ATOL = 5e-3


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_weights(path: Path, errors: list[str]) -> None:
    sums = HERE.parent / "weights" / "md5sums.txt"
    if not sums.exists():
        errors.append("weights/md5sums.txt missing")
        return
    expected = {}
    for line in sums.read_text().splitlines():
        digest, name = line.split()
        expected[name] = digest
    digest = md5(path)
    if path.name in expected:
        if expected[path.name] != digest:
            errors.append(f"md5 mismatch for {path.name}: {digest} != {expected[path.name]}")
        else:
            print(f"[ok] md5 {path.name} {digest}")
    else:
        print(f"[warn] {path.name} not listed in md5sums.txt (md5 {digest})")


def check_io(session: ort.InferenceSession, errors: list[str]) -> None:
    inp, out = session.get_inputs()[0], session.get_outputs()[0]
    if inp.name != "images" or inp.shape != [1, 3, INPUT_H, INPUT_W] or inp.type != "tensor(float)":
        errors.append(f"unexpected input: {inp.name} {inp.shape} {inp.type}")
    else:
        print(f"[ok] input  {inp.name} {inp.shape} {inp.type}")
    if out.name != "output0" or out.shape != [1, 6300, 16] or out.type != "tensor(float)":
        errors.append(f"unexpected output: {out.name} {out.shape} {out.type}")
    else:
        print(f"[ok] output {out.name} {out.shape} {out.type}")


def _iou(a, b):
    lt = np.maximum(a[:2], b[:2])
    rb = np.minimum(a[2:], b[2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[0] * wh[1]
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def check_sample(session: ort.InferenceSession, errors: list[str]) -> None:
    img = cv2.imread(str(HERE / "sample.jpg"))
    if img is None:
        errors.append("sample.jpg missing")
        return
    expected = json.loads((HERE / "expected_detections.json").read_text())
    got = infer(session, img)
    if len(got) != len(expected):
        errors.append(f"detection count {len(got)} != {len(expected)}")
    used = set()
    for exp in expected:
        best, best_iou = None, -1.0
        for i, det in enumerate(got):
            if i in used or det["class_id"] != exp["class_id"]:
                continue
            iou = _iou(np.array(det["bbox_xyxy"]), np.array(exp["bbox_xyxy"]))
            if iou > best_iou:
                best, best_iou = i, iou
        if best is None or best_iou < 0.99:
            errors.append(f'{exp["class_name"]} not reproduced (best IoU {best_iou:.3f})')
            continue
        used.add(best)
        det = got[best]
        if abs(det["score"] - exp["score"]) > SCORE_ATOL:
            errors.append(f'{exp["class_name"]} score {det["score"]:.4f} != {exp["score"]:.4f}')
        box_diff = float(np.max(np.abs(np.array(det["bbox_xyxy"]) - np.array(exp["bbox_xyxy"]))))
        if box_diff > BOX_ATOL:
            errors.append(f'{exp["class_name"]} box max diff {box_diff:.2f}px > {BOX_ATOL}')
        kpt_diff = float(np.max(np.abs(np.array(det["kpts_xy"]) - np.array(exp["kpts_xy"]))))
        if kpt_diff > KPT_ATOL:
            errors.append(f'{exp["class_name"]} kpt max diff {kpt_diff:.2f}px > {KPT_ATOL}')
    if not any(e.startswith("detection count") for e in errors):
        print(f"[ok] {len(got)} detections reproduce expected_detections.json")
        for det in got:
            print(f'     {det["class_name"]:<14} score={det["score"]:.3f} box={det["bbox_xyxy"]}')


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", default=str(HERE.parent / "weights" / "schemeC_ws123val_scratch250_20261114_best_fp16.onnx"))
    return p.parse_args()


def main() -> None:
    args = parse()
    path = Path(args.onnx)
    print(f"model: {path}")
    errors: list[str] = []
    check_weights(path, errors)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    check_io(session, errors)
    print("classes:", ", ".join(f"{i}:{n}" for i, n in enumerate(CLASS_NAMES)))
    print(f"colours: {COLOR_NAMES}  roles: {ROLE_NAMES}")
    check_sample(session, errors)
    if errors:
        print("\nFAIL")
        for e in errors:
            print("  -", e)
        raise SystemExit(1)
    print("\nPASS")


if __name__ == "__main__":
    main()
