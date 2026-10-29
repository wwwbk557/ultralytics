#!/usr/bin/env python3
"""Dextera custom detector: last.pt (21-class ArmorPose) -> 14-class boxes + 4 points.

Use this script in Dextera's Auto-Annotation dialog with
"处理方式 = 自定义脚本" and the absolute path to this file.

Params (JSON, all optional):
    {"weights": "/home/wbk/ultralytics_2027/last.pt",
     "conf": 0.25, "iou": 0.45, "imgsz": 640, "device": "auto",
     "max_det": 100, "classes": ["blue_1", "red_guard"]}
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

WORKSPACE = Path("/home/wbk/ultralytics_2027")
if str(WORKSPACE) not in sys.path:
    # append, so the backend venv's own ultralytics wins over the workspace copy
    sys.path.append(str(WORKSPACE))

_MODEL = None
_MODEL_KEY = None


def _get_model(params: dict):
    global _MODEL, _MODEL_KEY
    key = (
        params.get("weights", str(WORKSPACE / "last.pt")),
        float(params.get("conf", 0.25)),
        float(params.get("iou", 0.45)),
        int(params.get("imgsz", 640)),
        str(params.get("device", "auto")),
    )
    if _MODEL is None or _MODEL_KEY != key:
        from ext.armor_pose.ref_infer import RefArmorPose

        _MODEL = RefArmorPose(key[0], device=key[4], imgsz=key[3], conf=key[1], iou=key[2])
        _MODEL_KEY = key
    return _MODEL


def process(image: np.ndarray, params: dict) -> list[dict]:
    h, w = image.shape[:2]
    model = _get_model(params or {})
    keep = set(params.get("classes", [])) if params else set()
    max_det = int(params.get("max_det", 100)) if params else 100

    results = []
    for o in model.predict(image):
        if keep and o["name"] not in keep:
            continue
        x1, y1, x2, y2 = o["box"]
        results.append({
            "box": {
                "x1": float(np.clip(x1 / w, 0, 1)),
                "y1": float(np.clip(y1 / h, 0, 1)),
                "x2": float(np.clip(x2 / w, 0, 1)),
                "y2": float(np.clip(y2 / h, 0, 1)),
                "class_name": o["name"],
                "class_id": int(o["class_id"]),
            },
            "keypoints": [
                {
                    "name": f"p{i + 1}",
                    "x": float(np.clip(x / w, 0, 1)),
                    "y": float(np.clip(y / h, 0, 1)),
                    "visible": True,
                }
                for i, (x, y) in enumerate(o["keypoints"])
            ],
        })
        if len(results) >= max_det:
            break
    return results
