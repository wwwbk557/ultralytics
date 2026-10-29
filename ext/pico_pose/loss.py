"""Loss functions for the compact four-keypoint pose head.

``PicoCompactPoseLoss`` combines TAL assignment, CIoU/DFL box loss,
OKS keypoint loss and visibility BCE with three attribute terms (colour,
number, quality) and an optional pose-geometry regularizer. The returned dict
holds ``box, pose, kobj, cls, dfl, color, number, quality, geom``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.loss import v8PoseLoss

from .attrs import COLOR_BY_CLASS, NUMBER_BY_CLASS

__all__ = ("PicoCompactPoseLoss", "pose_geometry_loss")

LOSS_NAMES = (
    "box_loss",
    "pose_loss",
    "kobj_loss",
    "cls_loss",
    "dfl_loss",
    "color_loss",
    "number_loss",
    "quality_loss",
    "geom_loss",
)


def pose_geometry_loss(
    points: torch.Tensor,
    par: float = 0.10,
    opp: float = 0.33,
    angle: float = 0.23,
    axis: float = 0.44,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Regularize a quadrilateral described by four ordered points.

    Args:
        points: ``(P, 4, 2)`` pixel coordinates in polygon order.
        par: Weight of the parallelogram residual.
        opp: Weight of the opposite-edge length residual.
        angle: Weight of the right-angle residual.
        axis: Weight of the horizontal first-edge residual.
        eps: Small constant for normalization.

    Returns:
        Scalar loss. Every term is scale invariant.
    """
    p0, p1, p2, p3 = points.unbind(1)
    e01, e12, e23, e30 = p1 - p0, p2 - p1, p3 - p2, p0 - p3
    edges = torch.stack((e01, e12, e23, e30), 1)
    scale = edges.norm(dim=-1).mean(1).clamp_min(1.0)  # (P,)

    par_term = ((p0 + p2 - p1 - p3).norm(dim=-1) / scale).pow(2).mean()
    opp_term = (
        ((e01.norm(dim=-1) - e23.norm(dim=-1)) / scale).pow(2)
        + ((e30.norm(dim=-1) - e12.norm(dim=-1)) / scale).pow(2)
    ).mean()

    v_in = torch.stack((e30, e01, e12, e23), 1)
    v_out = torch.stack((e01, e12, e23, e30), 1)
    cos = (v_in * v_out).sum(-1) / (v_in.norm(dim=-1) * v_out.norm(dim=-1) + eps)
    angle_term = cos.pow(2).mean()

    axis_term = (e01[:, 1].pow(2) / (e01.pow(2).sum(-1) + eps)).mean()

    return par * par_term + opp * opp_term + angle * angle_term + axis * axis_term


def _oks_quality(pred_xy: torch.Tensor, gt_xy: torch.Tensor, area: torch.Tensor, k: float = 0.1) -> torch.Tensor:
    """Return an OKS-like quality target from detached predicted keypoints."""
    d2 = ((pred_xy - gt_xy) ** 2).sum(-1)
    sigma2 = (2.0 * area[:, None] * (k**2)).clamp_min(1e-6)
    return torch.exp(-d2 / sigma2).mean(1).clamp(0, 1)


class PicoCompactPoseLoss(v8PoseLoss):
    """TAL box/pose loss with colour, number, quality and geometry terms."""

    def __init__(self, model):
        super().__init__(model)
        self.color_map = torch.tensor(COLOR_BY_CLASS, device=self.device, dtype=torch.long)
        self.number_map = torch.tensor(NUMBER_BY_CLASS, device=self.device, dtype=torch.long)

    def loss(self, preds: dict, batch: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return the weighted loss vector and its detached per-term dict."""
        pred_kpts = preds["kpts"].permute(0, 2, 1).contiguous()
        bs = pred_kpts.shape[0]
        loss = torch.zeros(5, device=self.device, dtype=torch.float32)
        (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor), det_loss, _ = (
            self.get_assigned_targets_and_loss(preds, batch)
        )
        loss[0], loss[3], loss[4] = det_loss[0], det_loss[1], det_loss[2]

        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=pred_kpts.dtype) * self.stride[0]
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(bs, -1, *self.kpt_shape))

        target_bboxes_px = target_bboxes.clone()
        keypoints = self._batch_keypoints(batch, bs, imgsz) if fg_mask.any() else None
        if keypoints is not None:
            loss[1], loss[2] = self.calculate_keypoints_loss(
                fg_mask,
                target_gt_idx,
                keypoints,
                stride_tensor,
                target_bboxes,
                pred_kpts,
            )
        loss[1] *= self.hyp.pose
        loss[2] *= self.hyp.kobj
        loss = loss * bs

        n_pos = fg_mask.sum().clamp_min(1)
        extra = torch.zeros(4, device=self.device, dtype=torch.float32)
        if fg_mask.any():
            for b in range(bs):
                m = fg_mask[b]
                if not m.any():
                    continue
                gt_idx = target_gt_idx[b][m]
                gt_cls = batch["cls"].view(-1)[batch["batch_idx"].view(-1) == b].long()[gt_idx]
                color_logits = preds["color"][b].permute(1, 0)[m].float()
                number_logits = preds["number"][b].permute(1, 0)[m].float()
                extra[0] += F.cross_entropy(color_logits, self.color_map[gt_cls], reduction="sum")
                extra[1] += F.cross_entropy(number_logits, self.number_map[gt_cls], reduction="sum")

            pred_px = pred_kpts * stride_tensor.view(1, -1, 1, 1)
            gt_kpts = keypoints[torch.arange(bs, device=self.device)[:, None], target_gt_idx]
            pred_pos = pred_px[fg_mask][..., :2].float().detach()
            gt_pos = gt_kpts[fg_mask][..., :2].float()
            tb = target_bboxes_px[fg_mask]
            area = ((tb[:, 2] - tb[:, 0]) * (tb[:, 3] - tb[:, 1])).clamp_min(1.0)
            q_target = _oks_quality(pred_pos, gt_pos, area)
            q_logits = preds["quality"].permute(0, 2, 1).reshape(-1).float()
            q_target_all = torch.zeros_like(q_logits)
            q_target_all[fg_mask.reshape(-1)] = q_target
            q_pos_weight = ((q_logits.numel() - n_pos) / n_pos).clamp(1.0, 10000.0)
            extra[2] = F.binary_cross_entropy_with_logits(
                q_logits, q_target_all, reduction="mean", pos_weight=q_pos_weight
            )

            geom_gain = float(getattr(self.hyp, "geom", 0.0))
            if geom_gain > 0:
                pts = pred_px[fg_mask][..., :2].float()
                extra[3] = pose_geometry_loss(
                    pts,
                    par=float(getattr(self.hyp, "geom_par", 0.10)),
                    opp=float(getattr(self.hyp, "geom_opp", 0.33)),
                    angle=float(getattr(self.hyp, "geom_angle", 0.23)),
                    axis=float(getattr(self.hyp, "geom_axis", 0.44)),
                )

        extra[0] *= getattr(self.hyp, "color", 0.1) / n_pos
        extra[1] *= getattr(self.hyp, "number", 0.1) / n_pos
        extra[2] *= getattr(self.hyp, "quality", 0.05)
        extra[3] *= float(getattr(self.hyp, "geom", 0.0))
        extra = extra * bs

        total = torch.cat((loss, extra))
        return total, dict(zip(LOSS_NAMES, total.detach().tolist()))

    @staticmethod
    def kpts_decode(anchor_points: torch.Tensor, pred_kpts: torch.Tensor) -> torch.Tensor:
        """Decode Pico keypoint offsets to grid coordinates as ``anchor + raw``."""
        y = pred_kpts.clone()
        y[..., 0] += anchor_points[:, [0]]
        y[..., 1] += anchor_points[:, [1]]
        return y
