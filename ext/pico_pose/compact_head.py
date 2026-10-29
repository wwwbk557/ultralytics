"""Compact four-keypoint pose head with colour and number attribute branches.

Per pyramid level the head contains:

* ``cv2``: Conv3x3(64) -> Conv3x3(64) -> Conv1x1(4 * reg_max) box distribution.
* ``cv_shared``: depthwise Conv3x3 -> Conv1x1(32), BN + SiLU.
* ``cv_compact``: Conv1x1(32 -> 18): 8 keypoint, 1 quality, 2 colour, 7 number.
* ``cv_joint``: Conv1x1(32 -> nc) class logits used by the assigner.

``cv2`` and ``dfl`` are training/validation branches. Exporting uses only
``cv_compact`` and returns ``[B, A, 18]`` with raw keypoint offsets, quality
logit, colour logits and number logits.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ultralytics.nn.modules.block import DFL
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.head import Pose
from ultralytics.utils.tal import make_anchors

__all__ = ("PicoPoseCompactAuxBox",)


class PicoPoseCompactAuxBox(Pose):
    """Four-keypoint compact pose head with an auxiliary box branch."""

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
        self.num_numbers = 7
        if self.nc != self.num_colors * self.num_numbers:
            raise ValueError("PicoPoseCompactAuxBox requires nc = num_colors * num_numbers = 14")
        self.compact_no = self.nk + 1 + self.num_colors + self.num_numbers
        self.no_export = self.compact_no
        self.no = self.nc + 4 * self.reg_max
        self.head_channels = int(head_channels)
        self.aux_box = True
        self.multi_head = True
        self.export_xy_only = True
        self.export_row_major = True
        self.export = False
        self.cv3 = None
        self.cv4 = None
        self._end2end = False
        self.legacy = False
        self.xyxy = False
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
        self.cv_compact = nn.ModuleList(nn.Conv2d(self.head_channels, self.compact_no, 1) for _ in ch)
        self.cv_joint = nn.ModuleList(nn.Conv2d(self.head_channels, self.nc, 1) for _ in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x: list[torch.Tensor]):
        """Return training predictions, or ``(decoded, raw)`` outside training."""
        bs = x[0].shape[0]
        kpts, quality, color, number = [], [], [], []
        scores, boxes, raw = [], [], [] if self.export else None
        aux_active = self.aux_box and (self.training or not self.export)
        for i in range(self.nl):
            f = self.cv_shared[i](x[i])
            y = self.cv_compact[i](f).view(bs, self.compact_no, -1)
            kpts.append(y[:, 0 : self.nk])
            quality.append(y[:, self.nk : self.nk + 1])
            color.append(y[:, self.nk + 1 : self.nk + 1 + self.num_colors])
            number.append(y[:, self.nk + 1 + self.num_colors :])
            if raw is not None:
                raw.append(y)
            if aux_active:
                scores.append(self.cv_joint[i](f).view(bs, self.nc, -1))
                boxes.append(self.cv2[i](x[i]).view(bs, 4 * self.reg_max, -1))

        preds = {
            "kpts": torch.cat(kpts, -1),
            "quality": torch.cat(quality, -1),
            "color": torch.cat(color, -1),
            "number": torch.cat(number, -1),
            "feats": x,
        }
        if raw is not None:
            preds["raw18"] = torch.cat(raw, -1)
        if aux_active:
            preds["scores"] = torch.cat(scores, -1)
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
            return x["raw18"].permute(0, 2, 1).contiguous()

        kpts = self.kpts_decode(x["kpts"]).view(bs, self.nkpt, self.ndim, -1)
        dbox = self.decode_bboxes(self.dfl(x["boxes"]), self.anchors.unsqueeze(0)) * self.strides
        color_p = x["color"].softmax(1)
        number_p = x["number"].softmax(1)
        scores = (color_p[:, :, None] * number_p[:, None, :]).reshape(bs, self.nc, -1)
        scores = scores * x["quality"].sigmoid()
        return torch.cat((dbox, scores, kpts.reshape(bs, self.nk, -1)), 1)

    def kpts_decode(self, kpts: torch.Tensor) -> torch.Tensor:
        """Decode raw offsets to grid coordinates as ``(anchor + raw) * stride``."""
        bs = kpts.shape[0]
        y = kpts.view(bs, self.nkpt, self.ndim, -1).clone()
        y[:, :, 0] = (y[:, :, 0] + self.anchors[0]) * self.strides
        y[:, :, 1] = (y[:, :, 1] + self.anchors[1]) * self.strides
        return y.reshape(bs, self.nk, -1)

    def bias_init(self) -> None:
        """Initialize branch biases from the compact and joint output layouts."""
        for head in self.cv_compact:
            b = head.bias.data
            b[0 : self.nk].zero_()
            b[self.nk] = -2.0
            b[self.nk + 1 :].zero_()
        for i, head in enumerate(self.cv_joint):
            s = float(self.stride[i]) if float(self.stride[i]) > 0 else 8.0 * (2**i)
            head.bias.data.fill_(math.log(5 / self.nc / (640 / s) ** 2))
        for head in self.cv2:
            head[-1].bias.data.fill_(2.0)

    def fuse(self) -> None:
        """Keep all head branches; BN folding is handled by ``BaseModel.fuse``."""
        return
