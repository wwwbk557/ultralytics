"""Reference decoder for the Pico pose 16-channel ONNX export.

The graph takes one letterboxed image and returns ``output0`` of shape
``[1, 6300, 16]``:

    ch 0..7    raw keypoint coordinates, 4 corners x (x, y), order TL, TR, BR, BL
    ch 8       colour logit z            -> p_red = sigmoid(z)
    ch 9..15   role logits r[0..6]       -> role_p = sigmoid(r)  (multi-label, NOT softmax)

Anchor confidence and the 14 class scores are

    score[c * 7 + r] = colour_p[c] * role_p[r],   colour_p = [1 - p_red, p_red]

and there is no objectness and no box branch: the NMS box is the axis-aligned
extent of the four decoded corners.

Usage:
    python decode.py --onnx ../weights/schemeC_ws123val_scratch250_20261114_best_fp16.onnx \
        --image sample.jpg --json dets.json --draw dets.jpg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

INPUT_H, INPUT_W = 480, 640          # network input, (height, width)
STRIDES = (8, 16, 32)                # P3, P4, P5
PAD_VALUE = 114                      # letterbox grey, BGR and RGB
NUM_COLORS, NUM_ROLES, NUM_CLASSES = 2, 7, 14
KPT_NAMES = ("TL", "TR", "BR", "BL")  # keypoint order around the plate
COLOR_NAMES = ("blue", "red")
ROLE_NAMES = ("sentry", "hero", "engineer", "3", "4", "outpost", "base")
CLASS_NAMES = tuple(f"{c}_{r}" for c in COLOR_NAMES for r in ROLE_NAMES)

# Deployment defaults, matching the C++ node (conf 0.25 / IoU 0.45 / 300 dets).
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45
DEFAULT_MAX_DET = 300


def letterbox(img: np.ndarray, new_shape: tuple[int, int] = (INPUT_H, INPUT_W)) -> tuple[np.ndarray, float, int, int]:
    """Resize with unchanged aspect ratio and pad to ``new_shape`` (ultralytics LetterBox).

    Returns the padded image, the scale ratio ``r`` and the padding offsets that
    must be removed from decoded coordinates: ``(img, r, pad_left, pad_top)``.
    """
    h, w = img.shape[:2]
    new_h, new_w = new_shape
    r = min(new_h / h, new_w / w)
    unpad_w, unpad_h = int(round(w * r)), int(round(h * r))
    dw, dh = (new_w - unpad_w) / 2.0, (new_h - unpad_h) / 2.0
    if (w, h) != (unpad_w, unpad_h):
        img = cv2.resize(img, (unpad_w, unpad_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(PAD_VALUE,) * 3)
    return img, r, left, top


def preprocess(img_bgr: np.ndarray) -> tuple[np.ndarray, float, int, int]:
    """BGR uint8 image -> float32 ``[1, 3, 480, 640]`` RGB in ``[0, 1]``."""
    padded, r, pad_left, pad_top = letterbox(img_bgr)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    blob = rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray(blob), r, pad_left, pad_top


def anchor_grid() -> tuple[np.ndarray, np.ndarray]:
    """Anchor centres in grid units and their strides, level-major, row-major.

    Anchor ``i`` of a level sits at ``(x + 0.5, y + 0.5)`` for the cell index
    ``x = i % w``, ``y = i // w``. Level order is stride 8, 16, 32
    (4800 + 1200 + 300 = 6300 anchors).
    """
    points, strides = [], []
    for stride in STRIDES:
        h, w = INPUT_H // stride, INPUT_W // stride
        ys, xs = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
        points.append(np.stack((xs.reshape(-1) + 0.5, ys.reshape(-1) + 0.5), 1))
        strides.append(np.full((h * w,), stride, np.float32))
    return np.concatenate(points, 0), np.concatenate(strides, 0)


ANCHORS, ANCHOR_STRIDES = anchor_grid()


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def decode(raw: np.ndarray, conf: float = DEFAULT_CONF, iou: float = DEFAULT_IOU, max_det: int = DEFAULT_MAX_DET):
    """Decode ``output0[0]`` (6300, 16) into detections in network-input pixels.

    Returns a dict of arrays: ``boxes`` (N, 4) xyxy, ``scores`` (N,),
    ``classes`` (N,), ``kpts`` (N, 4, 2), ``color_p`` (N, 2), ``role_p`` (N, 7).
    """
    kpt_raw = raw[:, :8].reshape(-1, 4, 2).astype(np.float32)
    # kpts_decode in the head: (raw + anchor) * stride, anchors already carry +0.5.
    kpts = (kpt_raw + ANCHORS[:, None, :]) * ANCHOR_STRIDES[:, None, None]

    color_logit = raw[:, 8].astype(np.float32)
    role_logits = raw[:, 9:16].astype(np.float32)
    p_red = sigmoid(color_logit)
    color_p = np.stack((1.0 - p_red, p_red), 1)           # (A, 2): blue, red
    role_p = sigmoid(role_logits)                          # (A, 7), independent
    scores = (color_p[:, :, None] * role_p[:, None, :]).reshape(-1, NUM_CLASSES)

    best_score = scores.max(1)
    best_class = scores.argmax(1)
    keep = best_score > conf
    if not keep.any():
        empty = np.zeros((0,), np.float32)
        return {
            "boxes": np.zeros((0, 4), np.float32),
            "scores": empty,
            "classes": np.zeros((0,), np.int64),
            "kpts": np.zeros((0, 4, 2), np.float32),
            "color_p": np.zeros((0, 2), np.float32),
            "role_p": np.zeros((0, 7), np.float32),
        }

    xs, ys = kpts[keep][:, :, 0], kpts[keep][:, :, 1]
    boxes = np.stack(
        (xs.min(1), ys.min(1), xs.max(1), ys.max(1)), 1
    ).astype(np.float32)
    keep_idx = nms(boxes, best_score[keep], best_class[keep], iou, max_det)
    sel = np.flatnonzero(keep)[keep_idx]
    return {
        "boxes": boxes[keep_idx],
        "scores": best_score[sel],
        "classes": best_class[sel],
        "kpts": kpts[sel],
        "color_p": color_p[sel],
        "role_p": role_p[sel],
    }


def _iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    lt = np.maximum(box[:2], boxes[:, :2])
    rb = np.minimum(box[2:], boxes[:, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[:, 0] * wh[:, 1]
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area + areas - inter
    return np.where(union > 0, inter / union, 0.0)


def nms(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou: float, max_det: int) -> np.ndarray:
    """Greedy class-aware NMS; same semantics as ultralytics ``non_max_suppression`` (multi_label=False)."""
    order = np.argsort(-scores, kind="stable")
    suppressed = np.zeros(len(boxes), bool)
    kept: list[int] = []
    for pos, i in enumerate(order):
        if suppressed[i]:
            continue
        kept.append(int(i))
        if len(kept) >= max_det:
            break
        rest = order[pos + 1 :]
        overlap = _iou_one_to_many(boxes[i], boxes[rest]) > iou
        suppressed[rest[overlap & (classes[rest] == classes[i])]] = True
    return np.asarray(kept, np.int64)


def unmap(dets: dict, r: float, pad_left: int, pad_top: int, orig_w: int, orig_h: int) -> dict:
    """Network-input pixels -> original image pixels, clipping boxes to the image."""
    out = {k: v.copy() for k, v in dets.items()}
    out["boxes"][:, [0, 2]] = (out["boxes"][:, [0, 2]] - pad_left) / r
    out["boxes"][:, [1, 3]] = (out["boxes"][:, [1, 3]] - pad_top) / r
    out["kpts"][:, :, 0] = (out["kpts"][:, :, 0] - pad_left) / r
    out["kpts"][:, :, 1] = (out["kpts"][:, :, 1] - pad_top) / r
    out["boxes"][:, [0, 2]] = np.clip(out["boxes"][:, [0, 2]], 0, orig_w)
    out["boxes"][:, [1, 3]] = np.clip(out["boxes"][:, [1, 3]], 0, orig_h)
    return out


def infer(session: ort.InferenceSession, img_bgr: np.ndarray, conf=DEFAULT_CONF, iou=DEFAULT_IOU,
          max_det=DEFAULT_MAX_DET) -> list[dict]:
    """Full pipeline: image -> list of detection dicts in original-image coordinates."""
    blob, r, pad_left, pad_top = preprocess(img_bgr)
    name = session.get_inputs()[0].name
    raw = session.run(None, {name: blob})[0][0]
    dets = unmap(decode(raw, conf, iou, max_det), r, pad_left, pad_top, img_bgr.shape[1], img_bgr.shape[0])
    results = []
    for i in range(len(dets["scores"])):
        cls = int(dets["classes"][i])
        color, role = divmod(cls, NUM_ROLES)
        results.append(
            {
                "class_id": cls,
                "class_name": CLASS_NAMES[cls],
                "color": COLOR_NAMES[color],
                "role": ROLE_NAMES[role],
                "score": float(dets["scores"][i]),
                "bbox_xyxy": [round(float(v), 2) for v in dets["boxes"][i]],
                "kpts_xy": [[round(float(x), 2), round(float(y), 2)] for x, y in dets["kpts"][i]],
                "color_p": [round(float(v), 4) for v in dets["color_p"][i]],
                "role_p": [round(float(v), 4) for v in dets["role_p"][i]],
            }
        )
    return results


def draw(img: np.ndarray, dets: list[dict]) -> np.ndarray:
    vis = img.copy()
    for d in dets:
        x0, y0, x1, y1 = (int(round(v)) for v in d["bbox_xyxy"])
        color = (255, 80, 80) if d["color"] == "blue" else (80, 80, 255)
        cv2.rectangle(vis, (x0, y0), (x1, y1), color, 2)
        cv2.putText(vis, f'{d["class_name"]} {d["score"]:.2f}', (x0, max(y0 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        for name, (kx, ky) in zip(KPT_NAMES, d["kpts_xy"]):
            cv2.circle(vis, (int(round(kx)), int(round(ky))), 3, (0, 255, 0), -1)
            cv2.putText(vis, name, (int(kx) + 4, int(ky) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    return vis


def parse():
    p = argparse.ArgumentParser(description="Reference decoder for Pico pose 16-channel ONNX")
    p.add_argument("--onnx", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--conf", type=float, default=DEFAULT_CONF)
    p.add_argument("--iou", type=float, default=DEFAULT_IOU)
    p.add_argument("--max-det", type=int, default=DEFAULT_MAX_DET)
    p.add_argument("--json", help="write detections as JSON")
    p.add_argument("--draw", help="write an annotated image")
    return p.parse_args()


def main() -> None:
    args = parse()
    session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"cannot read image: {args.image}")
    dets = infer(session, img, args.conf, args.iou, args.max_det)
    print(f"{len(dets)} detections")
    for d in dets:
        print(f'  {d["class_name"]:<14} score={d["score"]:.3f} box={d["bbox_xyxy"]}')
    if args.json:
        Path(args.json).write_text(json.dumps(dets, ensure_ascii=False, indent=2))
    if args.draw:
        cv2.imwrite(args.draw, draw(img, dets))


if __name__ == "__main__":
    main()
