"""Compact four-keypoint pose head with colour and role attribute branches.

Per pyramid level the head contains:

* ``cv2``: Conv3x3(64) -> Conv3x3(64) -> Conv1x1(4 * reg_max) box distribution.
* ``cv_shared``: depthwise Conv3x3 -> Conv1x1(32), BN + SiLU.
* ``cv_compact``: Conv1x1(32 -> 18): 8 keypoint, 1 objectness, 2 colour, 7 role.
* ``cv_joint``: Conv3x3(32->64) -> Conv3x3(64->64) -> Conv1x1(64->nc) class logits\n  used by the assigner (TAL).

``cv2`` and ``dfl`` are training/validation branches. Exporting uses only
``cv_compact`` and returns ``[B, A, 18]`` with raw keypoint offsets, the
objectness logit (channel 8, still named ``quality`` in the export), colour logits and
role logits.

Scoring (validation, NMS and mAP all use the same soft score):

    obj   = sigmoid(objectness)                       # trained as foreground BCE
    score = obj * softmax(colour)[c] * softmax(role)[r]
    class = argmax(colour) * 7 + argmax(role)

``deploy_boxes=True`` (set by ``PicoRectTrainer`` from the run args) makes
validation build the NMS box from the four decoded corners instead of the
train-only DFL branch, so the reported mAP is the metric the exported graph
actually achieves. The exported 18-channel tensor already contains no box at all.

Objectness is a multiplier, not a 0/1 gate: the gate would drop the objectness
ordering (obj=0.51 and obj=0.99 would tie) and leave NMS/mAP sorting on
P(colour, role | object) alone. Deployment may additionally drop anchors below
``obj_threshold``; survivors keep the same soft score.
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

    # Class-level defaults keep checkpoints pickled before these flags existed
    # loadable: unpickling restores __dict__ only, never the new __init__ keys.
    obj_threshold = 0.5
    deploy_filter = False
    deploy_boxes = False
    obj_logit_bias = 0.0
    # Set by the export baker once the bias is folded into cv_compact.bias[8].
    # Subtracting the runtime bias again would double-calibrate (mAP drops while
    # the loss looks normal), so every calibration site asserts on this flag.
    _obj_logit_calibrated = False

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
        self.compact_no = self.nk + 1 + self.num_colors + self.num_roles
        self.no_export = self.compact_no
        self.no = self.nc + 4 * self.reg_max
        self.head_channels = int(head_channels)

        self.aux_box = True
        self.multi_head = True
        self.export_xy_only = True
        self.export_row_major = True
        self.export = False
        # Channel 8 is trained as objectness and always multiplies the class
        # probability. `obj_threshold`/`deploy_filter` only exist to reproduce
        # the deployment gate in validation runs.
        self.obj_threshold = float(kwargs.pop("obj_threshold", 0.5))
        self.deploy_filter = bool(kwargs.pop("deploy_filter", False))
        # Deployment parity: the exported graph has no box branch, so deployment
        # builds the NMS box from the four decoded corners. `deploy_boxes` makes
        # validation/NMS use that same box instead of the train-only DFL branch.
        self.deploy_boxes = bool(kwargs.pop("deploy_boxes", False))
        # Optional objectness calibration: sigmoid(z - obj_logit_bias) in [0, 1]
        # lands on the true foreground posterior when the bias equals
        # log(pos_weight) used during training. It is subtracted from the exported
        # channel-8 logit so the deployment sigmoid is calibrated too.
        self.obj_logit_bias = float(kwargs.pop("obj_logit_bias", 0.0))
        self.extra_loss_names = ("color_loss", "number_loss", "quality_loss", "geom_loss")
        self.cv3 = None
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
        # Same capacity as a YOLOv8 cls branch: 3x3(32->64) -> 3x3(64->64) -> 1x1(64->nc).
        self.cv_joint = nn.ModuleList(
            nn.Sequential(
                Conv(self.head_channels, 64, 3),
                Conv(64, 64, 3),
                nn.Conv2d(64, self.nc, 1),
            )
            for _ in ch
        )
        self.cv_compact = nn.ModuleList(nn.Conv2d(self.head_channels, self.compact_no, 1) for _ in ch)
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
            raw = x["raw18"]
            if self.obj_logit_bias:
                assert not self._obj_logit_calibrated, (
                    "objectness logits are already calibrated; refusing to subtract obj_logit_bias twice"
                )
                raw = raw.clone()
                raw[:, self.nk] -= self.obj_logit_bias
            return raw.permute(0, 2, 1).contiguous()

        kpts = self.kpts_decode(x["kpts"]).view(bs, self.nkpt, self.ndim, -1)
        if self.deploy_boxes:
            dbox = self._corner_bboxes(kpts)
        else:
            dbox = self.decode_bboxes(self.dfl(x["boxes"]), self.anchors.unsqueeze(0)) * self.strides
        scores = self._class_scores(self.obj_prob(x["quality"]), x["color"].softmax(1), x["number"].softmax(1))
        return torch.cat((dbox, scores, kpts.reshape(bs, self.nk, -1)), 1)

    def obj_prob(self, logits: torch.Tensor) -> torch.Tensor:
        """Objectness probability, optionally de-biased by ``obj_logit_bias``."""
        assert not self._obj_logit_calibrated, (
            "objectness logits are already calibrated (bias baked into the weights); "
            "subtracting obj_logit_bias again would double-calibrate"
        )
        return (logits - self.obj_logit_bias).sigmoid()

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

    def _class_scores(
        self, obj: torch.Tensor, color_p: torch.Tensor, number_p: torch.Tensor
    ) -> torch.Tensor:
        """Score anchors as objectness x colour x role (soft, no 0/1 gate).

        Args:
            obj: (B, 1, A) objectness probabilities (sigmoid of channel 8).
            color_p: (B, num_colors, A) colour softmax probabilities.
            number_p: (B, num_roles, A) role softmax probabilities.

        Returns:
            (B, nc, A) class scores. ``scores.sum(1) == obj`` because each
            softmax is normalised before the product, so the 14-class product is
            a distribution over the class set conditioned on the anchor being an
            object.

        Notes:
            ``score = obj * P(colour, role)`` keeps the objectness ordering: an
            obj=0.99 anchor always outranks an obj=0.51 anchor even when they
            share the same class probabilities. A hard ``(obj >= threshold)``
            gate would throw that ordering away and leave NMS/mAP sorting on
            ``colour x role`` alone, which is only P(class | object).

            When ``deploy_filter`` is True the anchor still scores ``obj *
            class_prob``; the gate only decides membership (deployment parity).
        """
        class_prob = (color_p[:, :, None] * number_p[:, None, :]).reshape(obj.shape[0], self.nc, -1)
        scores = obj * class_prob
        if self.deploy_filter:
            scores = scores * (obj >= self.obj_threshold).to(scores.dtype)
        return scores

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
        # for i, head in enumerate(self.cv_joint):
        #     s = float(self.stride[i]) if float(self.stride[i]) > 0 else 8.0 * (2**i)
        #     head.bias.data.fill_(math.log(5 / self.nc / (640 / s) ** 2))
        for i, head in enumerate(self.cv_joint):
            s = float(self.stride[i]) if float(self.stride[i]) > 0 else 8.0 * (2**i)
            head[-1].bias.data.fill_(math.log(5 / self.nc / (640 / s) ** 2))
        for head in self.cv2:
            head[-1].bias.data.fill_(2.0)

    def fuse(self) -> None:
        """Keep all head branches; BN folding is handled by ``BaseModel.fuse``."""
        return
