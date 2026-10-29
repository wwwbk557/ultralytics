#!/usr/bin/env python3
"""Inference wrapper for a 21-class ArmorPose checkpoint (last.pt).

Checkpoints store the head class as ``ultralytics.nn.modules.head.ArmorPose``;
``_ArmorPoseStub`` injects an attribute-compatible stand-in so ``torch.load`` can
rebuild them without that class.

Decoding uses the YOLO pose keypoint formula ``(2 * raw + anchor - 0.5) * stride``
and maps keypoint order ``GT[0,3,2,1]``.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

import ultralytics.nn.modules.head as _head_mod
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.tal import make_anchors

REF_NAMES = (
    "red_hero", "red_engineer", "red_3", "red_4", "red_sentry", "red_outpost", "red_base",
    "blue_hero", "blue_engineer", "blue_3", "blue_4", "blue_sentry", "blue_outpost", "blue_base",
    "gray_hero", "gray_engineer", "gray_3", "gray_4", "gray_sentry", "gray_outpost", "gray_base",
)

TARGET_NAMES = (
    "blue_1", "blue_2", "blue_3", "blue_4", "blue_5", "blue_guard", "blue_outpost",
    "red_1", "red_2", "red_3", "red_4", "red_5", "red_guard", "red_outpost",
)

# Reference slot -> 14-class dataset id.  Dataset digits 1..5 are
# hero/engineer/infantry3/infantry4/infantry5; the reference has no infantry5 and
# the dataset has no base, so base and all gray slots are dropped.
_MAP21 = (
    7, 8, 9, 10, 12, 13, -1,   # red hero/engineer/3/4/sentry/outpost/base
    0, 1, 2, 3, 5, 6, -1,      # blue hero/engineer/3/4/sentry/outpost/base
    -1, -1, -1, -1, -1, -1, -1,
)
REF_TO_TARGET = _MAP21

_KPT_ORDER = (0, 3, 2, 1)


class _ArmorPoseStub(nn.Module):
    """Attribute-compatible stand-in so torch.load can rebuild the checkpoint."""

    def forward(self, x):
        bs = x[0].shape[0]
        kpts, quality, color, number, joint, boxes = [], [], [], [], [], []
        for i in range(self.nl):
            f = self.cv_shared[i](x[i])
            y = self.cv_compact[i](f).view(bs, 19, -1)
            kpts.append(y[:, 0:8])
            quality.append(y[:, 8:9])
            color.append(y[:, 9:12])
            number.append(y[:, 12:19])
            joint.append(self.cv_joint[i](f).view(bs, 21, -1))
            boxes.append(self.cv2[i](x[i]).view(bs, 4 * self.reg_max, -1))
        return {
            "kpts": torch.cat(kpts, -1),
            "quality": torch.cat(quality, -1),
            "color": torch.cat(color, -1),
            "number": torch.cat(number, -1),
            "joint": torch.cat(joint, -1),
            "boxes": torch.cat(boxes, -1),
            "feats": x,
        }


def _ensure_stub() -> None:
    if not hasattr(_head_mod, "ArmorPose"):
        _head_mod.ArmorPose = _ArmorPoseStub


class RefArmorPose:
    """Run the 21-class reference model and emit 14-class boxes + 4 keypoints."""

    def __init__(
        self,
        weights: str | Path,
        device: str | None = None,
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.45,
        max_det: int = 100,
    ) -> None:
        _ensure_stub()
        if device is None or device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.max_det = int(max_det)

        ck = torch.load(str(weights), map_location="cpu", weights_only=False)
        self.model = ck["model"].float().eval().to(self.device)
        self.head = self.model.model[-1]
        self.stride = self.head.stride.float().to(self.device)
        self.reg_max = int(self.head.reg_max)

        map_t = torch.full((21,), -1, dtype=torch.long)
        for ref_id, tgt in enumerate(_MAP21):
            map_t[ref_id] = tgt
        self._map = map_t.to(self.device)

    def _letterbox(self, img: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        h, w = img.shape[:2]
        r = min(self.imgsz / w, self.imgsz / h)
        nw, nh = int(round(w * r)), int(round(h * r))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        left, top = (self.imgsz - nw) // 2, (self.imgsz - nh) // 2
        canvas[top:top + nh, left:left + nw] = resized
        return canvas, r, left, top

    @torch.inference_mode()
    def predict_batch(self, images: list[np.ndarray]) -> list[list[dict]]:
        metas, tensors = [], []
        for img in images:
            canvas, r, left, top = self._letterbox(img)
            metas.append((img.shape[1], img.shape[0], r, left, top))
            tensors.append(canvas[:, :, ::-1].transpose(2, 0, 1))
        x = torch.from_numpy(np.ascontiguousarray(np.stack(tensors))).to(self.device)
        x = x.float() / 255.0

        out = self.model(x)
        feats = out["feats"]
        anchors, stride_t = make_anchors(feats, self.stride, 0.5)  # (A,2), (A,1)
        a = anchors.unsqueeze(0)  # (1,A,2)

        raw = out["kpts"].view(x.shape[0], 4, 2, -1)
        kx = (raw[:, :, 0] * 2 + a[..., 0] - 0.5) * stride_t[:, 0]
        ky = (raw[:, :, 1] * 2 + a[..., 1] - 0.5) * stride_t[:, 0]
        kpts = torch.stack((kx, ky), -1).permute(0, 2, 1, 3).reshape(x.shape[0], -1, 8)  # (B,A,8)
        kpts = kpts.view(x.shape[0], -1, 4, 2)[:, :, list(_KPT_ORDER), :].reshape(x.shape[0], -1, 8)
        k4 = kpts.view(x.shape[0], -1, 4, 2)
        hx1, hx2 = k4[..., 0].min(-1).values, k4[..., 0].max(-1).values
        hy1, hy2 = k4[..., 1].min(-1).values, k4[..., 1].max(-1).values
        hull = torch.stack(((hx1 + hx2) / 2, (hy1 + hy2) / 2, hx2 - hx1, hy2 - hy1), -1)  # (B,A,4)

        q = out["quality"].sigmoid()  # (B,1,A)
        cprob = out["color"].softmax(1)  # (B,3,A)
        nprob = out["number"].softmax(1)  # (B,7,A)
        ref_ids = torch.arange(21, device=self.device)
        score21 = q * cprob[:, ref_ids // 7] * nprob[:, ref_ids % 7]  # (B,21,A)

        bs = x.shape[0]
        scores14 = torch.zeros((bs, len(TARGET_NAMES), score21.shape[-1]), device=self.device)
        for ref_id, tgt in enumerate(_MAP21):
            if tgt >= 0:
                scores14[:, tgt] = score21[:, ref_id]

        pred = torch.cat((hull.transpose(1, 2), scores14, kpts.transpose(1, 2)), 1)  # (B, 4+14+8, A)
        dets, _ = non_max_suppression(
            pred, self.conf, self.iou, nc=len(TARGET_NAMES), max_det=self.max_det, return_idxs=True
        )

        results: list[list[dict]] = []
        for bi, det in enumerate(dets):
            w, h, r, left, top = metas[bi]
            det = det.cpu().numpy()
            objs = []
            for row in det:
                box = row[:4]
                cls_id = int(row[5])
                score = float(row[4])
                pts = row[6:].reshape(4, 2)
                pts[:, 0] = np.clip((pts[:, 0] - left) / r, 0, w - 1)
                pts[:, 1] = np.clip((pts[:, 1] - top) / r, 0, h - 1)
                # Dataset convention: bbox is the keypoint hull, not the regression box.
                x1, y1 = pts[:, 0].min(), pts[:, 1].min()
                x2, y2 = pts[:, 0].max(), pts[:, 1].max()
                objs.append({
                    "class_id": cls_id,
                    "name": TARGET_NAMES[cls_id],
                    "score": score,
                    "box": [float(x1), float(y1), float(x2), float(y2)],
                    "keypoints": pts.tolist(),
                })
            objs.sort(key=lambda o: o["score"], reverse=True)
            results.append(objs)
        return results

    def predict(self, image: np.ndarray, **kwargs) -> list[dict]:
        return self.predict_batch([image], **kwargs)[0]
