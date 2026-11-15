"""Compact four-keypoint pose head with colour and role attribute branches.

Per pyramid level the head contains:

* ``cv2``: Conv3x3(64) -> Conv3x3(64) -> Conv1x1(4 * reg_max) box distribution.
* ``cv_shared``: depthwise Conv3x3 -> Conv1x1(32), BN + SiLU.
* ``cv_compact``: Conv1x1(32 -> 9): 8 keypoint offsets, 1 colour logit.
* ``cv3``: DWConv3x3(32) -> DWConv3x3(32) -> Conv1x1(32 -> 7) role logits.

There is no objectness channel. The seven role logits are the classification
head in the YOLOv8 sense: ``sigmoid`` is trained with BCE over all anchors and
doubles as the anchor confidence, so the TAL assigner and the exported graph use
the same scores. Colour is a conditional attribute on the shared feature:
``p_red = sigmoid(z)`` and ``color_p = [1 - p_red, p_red]``.

Scoring (validation, NMS and mAP all use the same score):

    score[c * 7 + r] = color_p[c] * sigmoid(role_logit[r])
    class = argmax(color_p) * 7 + argmax(role)

``cv2``/``dfl`` are training/validation branches. Exporting runs only
``cv_shared``, ``cv_compact`` and ``cv3`` and returns ``[B, A, 16]`` with raw
keypoint offsets, the colour logit and the seven role logits.

``deploy_boxes=True`` (set by ``PicoRectTrainer`` from the run args) makes
validation build the NMS box from the four decoded corners instead of the
train-only DFL branch, so the reported mAP is the metric the exported graph
actually achieves. The exported 16-channel tensor already contains no box.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ultralytics.nn.modules.block import DFL
from ultralytics.nn.modules.conv import Conv, DWConv
from ultralytics.nn.modules.head import Pose
from ultralytics.utils.tal import make_anchors

__all__ = ("PicoPoseCompactAuxBox",)


class PicoPoseCompactAuxBox(Pose):
    """Four-keypoint compact pose head with an auxiliary box branch."""

    # Class-level default keeps checkpoints pickled before this flag existed
    # loadable: unpickling restores __dict__ only, never the new __init__ keys.
    deploy_boxes = False

    def __init__(
        self,
        nc: int = 14,
        kpt_shape: tuple = (4, 2),
        reg_max: int = 16,
        end2end: bool = False,
        ch: tuple = (),
        head_channels: int = 32,
        **kwargs,
    ):
        nn.Module.__init__(self)
        if tuple(kpt_shape) != (4, 2):
            raise ValueError("PicoPoseCompactAuxBox requires kpt_shape=(4, 2)")
        self.nc = int(nc)
        self.kpt_shape = tuple(kpt_shape)
        self.nkpt, self.ndim = self.kpt_shape
        self.nk = self.nkpt * self.ndim
        self.nl = len(ch)
        self.reg_max = int(reg_max)
        self.num_colors = 2
        self.num_roles = 7
        if self.nc != self.num_colors * self.num_roles:
            raise ValueError("PicoPoseCompactAuxBox requires nc = num_colors * num_roles = 14")
        self.compact_no = self.nk + 1
        self.no_export = self.compact_no + self.num_roles
        self.no = self.nc + 4 * self.reg_max
        self.head_channels = int(head_channels)

        self.aux_box = True
        self.multi_head = True
        self.export_xy_only = True
        self.export_row_major = True
        self.export = False
        # Deployment parity: the exported graph has no box branch, so deployment
        # builds the NMS box from the four decoded corners. `deploy_boxes` makes
        # validation/NMS use that same box instead of the train-only DFL branch.
        self.deploy_boxes = bool(kwargs.pop("deploy_boxes", False))
        self.extra_loss_names = ("color_loss", "geom_loss", "consistency_loss")
        self.cv4 = None
        self._end2end = False
        self.stride = torch.zeros(self.nl)
        self.shape = None
        self.dynamic = False

        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(c, 64, 3),
                Conv(64, 64, 3),
                nn.Conv2d(64, 4 * self.reg_max, 1),
            )
            for c in ch
        )
        self.cv_shared = nn.ModuleList(
            nn.Sequential(
                Conv(c, c, 3, 1, 1, g=c),
                Conv(c, self.head_channels, 1),
            )
            for c in ch
        )
        # Role head: Pose26-style depthwise stack. It is also the exported
        # confidence branch, so it cannot be train-only.
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                DWConv(self.head_channels, self.head_channels, 3),
                DWConv(self.head_channels, self.head_channels, 3),
                nn.Conv2d(self.head_channels, self.num_roles, 1),
            )
            for _ in ch
        )
        self.cv_compact = nn.ModuleList(nn.Conv2d(self.head_channels, self.compact_no, 1) for _ in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x: list[torch.Tensor]):
        """Return training predictions, or ``(decoded, raw)`` outside training."""
        bs = x[0].shape[0]
        kpts, color, number = [], [], []
        boxes, raw = [], [] if self.export else None
        aux_active = self.aux_box and (self.training or not self.export)
        for i in range(self.nl):
            f = self.cv_shared[i](x[i])
            y = self.cv_compact[i](f).view(bs, self.compact_no, -1)
            n = self.cv3[i](f).view(bs, self.num_roles, -1)
            kpts.append(y[:, 0 : self.nk])
            color.append(y[:, self.nk : self.nk + 1])
            number.append(n)
            if raw is not None:
                raw.append(torch.cat((y, n), 1))
            if aux_active:
                boxes.append(self.cv2[i](x[i]).view(bs, 4 * self.reg_max, -1))

        preds = {
            "kpts": torch.cat(kpts, -1),
            "color": torch.cat(color, -1),
            "number": torch.cat(number, -1),
            # TAL and the exported graph share the role logits as the confidence.
            "scores": torch.cat(number, -1),
            "feats": x,
        }
        if raw is not None:
            preds["raw16"] = torch.cat(raw, -1)
        if aux_active:
            preds["boxes"] = torch.cat(boxes, -1)
        if self.training:
            return preds
        y = self._inference(preds)
        return y if self.export else (y, preds)

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode the validation tensor or return the raw export tensor."""
        shape = x["feats"][0].shape
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (t.transpose(0, 1) for t in make_anchors(x["feats"], self.stride, 0.5))
            self.shape = shape
        bs = x["kpts"].shape[0]
        if self.export:
            return x["raw16"].permute(0, 2, 1).contiguous()

        kpts = self.kpts_decode(x["kpts"]).view(bs, self.nkpt, self.ndim, -1)
        if self.deploy_boxes:
            dbox = self._corner_bboxes(kpts)
        else:
            dbox = self.decode_bboxes(self.dfl(x["boxes"]), self.anchors.unsqueeze(0)) * self.strides
        scores = self._class_scores(self.color_prob(x["color"]), x["number"].sigmoid())
        return torch.cat((dbox, scores, kpts.reshape(bs, self.nk, -1)), 1)

    @staticmethod
    def color_prob(logits: torch.Tensor) -> torch.Tensor:
        """Two-colour probability from one logit: ``[1 - sigmoid(z), sigmoid(z)]``."""
        p_red = logits.sigmoid()
        return torch.cat((1.0 - p_red, p_red), 1)

    @staticmethod
    def _corner_bboxes(kpts: torch.Tensor) -> torch.Tensor:
        """Axis-aligned box (xywh, input pixels) around the four decoded corners.

        Args:
            kpts: ``(B, 4, 2, A)`` decoded corners in input-image pixels.

        Returns:
            ``(B, 4, A)`` boxes in xywh, the format ``non_max_suppression`` expects.
            This is exactly the box the deployment decoder builds, and it is also
            exactly how the ground-truth boxes were derived from the labels.
        """
        xs, ys = kpts[:, :, 0], kpts[:, :, 1]
        x0, x1 = xs.min(1).values, xs.max(1).values
        y0, y1 = ys.min(1).values, ys.max(1).values
        return torch.cat(
            (((x0 + x1) * 0.5)[:, None], ((y0 + y1) * 0.5)[:, None], (x1 - x0)[:, None], (y1 - y0)[:, None]), 1
        )

    def _class_scores(self, color_p: torch.Tensor, number_p: torch.Tensor) -> torch.Tensor:
        """Score anchors as colour x role, the 14-class product the validator expects.

        Args:
            color_p: (B, num_colors, A) colour probabilities summing to 1.
            number_p: (B, num_roles, A) role sigmoid probabilities from the cls head.

        Returns:
            (B, nc, A) class scores with ``score[c * 7 + r] = color_p[c] * number_p[r]``.
            The role head is multi-label BCE, so the seven role scores are not
            normalised; the class, not the vector sum, is what NMS/mAP consume.
        """
        return (color_p[:, :, None] * number_p[:, None, :]).reshape(color_p.shape[0], self.nc, -1)

    def kpts_decode(self, kpts: torch.Tensor) -> torch.Tensor:
        """Decode raw offsets to grid coordinates as ``(anchor + raw) * stride``."""
        bs = kpts.shape[0]
        y = kpts.view(bs, self.nkpt, self.ndim, -1).clone()
        y[:, :, 0] = (y[:, :, 0] + self.anchors[0]) * self.strides
        y[:, :, 1] = (y[:, :, 1] + self.anchors[1]) * self.strides
        return y.reshape(bs, self.nk, -1)

    def bias_init(self) -> None:
        """Initialize branch biases from the compact and role output layouts."""
        for head in self.cv_compact:
            head.bias.data.zero_()
        for i, head in enumerate(self.cv3):
            s = float(self.stride[i]) if float(self.stride[i]) > 0 else 8.0 * (2**i)
            head[-1].bias.data.fill_(math.log(5 / self.num_roles / (640 / s) ** 2))
        for head in self.cv2:
            head[-1].bias.data.fill_(2.0)

    def fuse(self) -> None:
        """Keep all head branches; BN folding is handled by ``BaseModel.fuse``."""
        return
