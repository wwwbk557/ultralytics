"""Loss functions for the compact four-keypoint pose head.

``PicoCompactPoseLoss`` combines TAL assignment, CIoU/DFL box loss,
OKS keypoint loss and visibility BCE with three attribute terms (colour, role,
objectness) and an optional pose-geometry regularizer. The returned vector is
``[box, pose, kobj, cls, dfl, color, number, quality, geom]``.

The ``quality`` slot (channel 8 of the compact export) is trained as
*objectness*: 1 on every TAL-positive anchor, 0 elsewhere. ``colour`` and
``number`` are cross-entropy trained on those positives only, and at inference
they decide the class while objectness only gates the anchor.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.loss import v8PoseLoss

from .attrs import COLOR_BY_CLASS, NUM_ROLE, ROLE_BY_CLASS
from .monitor import assignment_summary, hist_auc

__all__ = ("PicoCompactPoseLoss", "pose_geometry_loss")


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
    opp_term = (((e01.norm(dim=-1) - e23.norm(dim=-1)) / scale).pow(2)
                + ((e30.norm(dim=-1) - e12.norm(dim=-1)) / scale).pow(2)).mean()

    v_in = torch.stack((e30, e01, e12, e23), 1)
    v_out = torch.stack((e01, e12, e23, e30), 1)
    cos = (v_in * v_out).sum(-1) / (v_in.norm(dim=-1) * v_out.norm(dim=-1) + eps)
    angle_term = cos.pow(2).mean()

    axis_term = (e01[:, 1].pow(2) / (e01.pow(2).sum(-1) + eps)).mean()

    return par * par_term + opp * opp_term + angle * angle_term + axis * axis_term


class PicoCompactPoseLoss(v8PoseLoss):
    """TAL box/pose loss with colour, role, objectness (channel 8) and geometry terms."""

    def __init__(self, model):
        super().__init__(model)
        # A calibrated head has obj_logit_bias folded into channel 8, so its raw
        # logit is no longer the quantity the BCE target is defined against.
        assert not getattr(model.model[-1], "_obj_logit_calibrated", False), (
            "refusing to train against a calibrated objectness head (bias already baked into the weights)"
        )
        self.color_map = torch.tensor(COLOR_BY_CLASS, device=self.device, dtype=torch.long)
        self.number_map = torch.tensor(ROLE_BY_CLASS, device=self.device, dtype=torch.long)

    # ---- epoch-level monitors (TAL ranking / cv_joint vs attribute agreement) ----
    def _monitors(self) -> dict:
        if not hasattr(self, "_mon") or self._mon is None:
            dev = self.device

            def z():
                return torch.zeros((), device=dev, dtype=torch.float64)

            self._mon = {
                "pos_hist": torch.zeros(100, device=dev, dtype=torch.float64),
                "neg_hist": torch.zeros(100, device=dev, dtype=torch.float64),
                "pos_sum": z(),
                "neg_sum": z(),
                "pos_n": z(),
                "neg_n": z(),
                "agree": z(),
                "agree_n": z(),
                "assign_areas": [],
                "assign_counts": [],
            }
        return self._mon

    def pop_monitors(self) -> dict[str, float]:
        """Return the epoch's TAL/attribute monitors and start a fresh window."""
        if not hasattr(self, "_mon") or self._mon is None:
            return {}
        m = self._mon
        self._mon = None
        nan = float("nan")
        # keep the keys stable so results.csv headers never shift between epochs
        out: dict[str, float] = {"mon/s_auc": nan, "mon/s_gap": nan, "mon/attr_agree": nan}
        pos_n, neg_n = float(m["pos_n"]), float(m["neg_n"])
        if pos_n > 0 and neg_n > 0:
            out["mon/s_auc"] = hist_auc(m["pos_hist"].cpu().numpy(), m["neg_hist"].cpu().numpy())
            out["mon/s_gap"] = float(m["pos_sum"] / pos_n - m["neg_sum"] / neg_n)
        if float(m["agree_n"]) > 0:
            out["mon/attr_agree"] = float(m["agree"] / m["agree_n"])
        if m.get("assign_areas"):
            out.update(
                assignment_summary(
                    np.concatenate(m["assign_areas"]),
                    np.concatenate(m["assign_counts"]),
                    int(self.assigner.topk),
                )
            )
        return out

    def _track_assignment(self, batch: dict, fg_mask: torch.Tensor, target_gt_idx: torch.Tensor, imgsz: torch.Tensor) -> None:
        """Count, for every GT, how many anchors TAL actually assigned to it."""
        with torch.no_grad():
            boxes = batch["bboxes"].reshape(-1, 4).float()
            if boxes.numel() == 0:
                return
            bidx = batch["batch_idx"].reshape(-1).long()
            w, h = float(imgsz[1]), float(imgsz[0])
            areas = (boxes[:, 2] * w) * (boxes[:, 3] * h)
            counts = torch.zeros(len(boxes), device=boxes.device, dtype=torch.float32)
            for b in range(fg_mask.shape[0]):
                sel = (bidx == b).nonzero(as_tuple=True)[0]
                if sel.numel() == 0 or not fg_mask[b].any():
                    continue
                idx = target_gt_idx[b][fg_mask[b]].long()
                n_gt = int(sel.numel())
                counts[sel] = torch.bincount(idx, minlength=n_gt)[:n_gt].float()
            mon = self._monitors()
            mon["assign_areas"].append(areas.detach().cpu().numpy())
            mon["assign_counts"].append(counts.detach().cpu().numpy())

    def loss(self, preds: dict, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the weighted loss and its detached per-term vector."""
        pred_kpts = preds["kpts"].permute(0, 2, 1).contiguous()
        bs = pred_kpts.shape[0]
        loss = torch.zeros(5, device=self.device, dtype=torch.float32)
        (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor), det_loss, _ = (
            self.get_assigned_targets_and_loss(preds, batch)
        )
        # The TAL assigner returns float zeros when the whole batch has no GT
        # (n_max_boxes == 0), everything else returns bool; normalise here because
        # bool masks are used as indices below and `~mask` needs a bool tensor.
        fg_mask = fg_mask.bool()
        loss[0], loss[3], loss[4] = det_loss[0], det_loss[1], det_loss[2]

        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=pred_kpts.dtype) * self.stride[0]
        self._track_assignment(batch, fg_mask, target_gt_idx, imgsz)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(bs, -1, *self.kpt_shape))

        # Soft objectness target for `quality_soft > 0`: IoU between the box that
        # the four predicted corners would produce at deployment and the assigned
        # GT box. That is the same quantity the deployment NMS uses to merge
        # duplicates, so the objectness channel becomes a localisation-quality
        # score instead of a plain foreground flag.
        q_iou = None
        q_soft = float(getattr(self.hyp, "quality_soft", 0.0))
        if q_soft > 0 and fg_mask.any():
            with torch.no_grad():
                tb = (target_bboxes / stride_tensor.view(1, -1, 1))[fg_mask]  # xyxy, grid units
                pk = pred_kpts[fg_mask][..., :2]
                x0, y0 = pk[..., 0].min(-1).values, pk[..., 1].min(-1).values
                x1, y1 = pk[..., 0].max(-1).values, pk[..., 1].max(-1).values
                iw = (torch.minimum(x1, tb[:, 2]) - torch.maximum(x0, tb[:, 0])).clamp_min(0)
                ih = (torch.minimum(y1, tb[:, 3]) - torch.maximum(y0, tb[:, 1])).clamp_min(0)
                inter = iw * ih
                union = (x1 - x0) * (y1 - y0) + (tb[:, 2] - tb[:, 0]) * (tb[:, 3] - tb[:, 1]) - inter
                q_iou = inter / union.clamp_min(1e-9)

        keypoints = None
        if fg_mask.sum():
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]
            loss[1], loss[2] = self.calculate_keypoints_loss(
                fg_mask,
                target_gt_idx,
                keypoints,
                batch["batch_idx"].view(-1, 1),
                stride_tensor,
                target_bboxes,
                pred_kpts,
            )
        loss[1] *= self.hyp.pose
        loss[2] *= self.hyp.kobj
        loss = loss * bs

        with torch.no_grad():
            mon = self._monitors()
            s_max = preds["scores"].permute(0, 2, 1).sigmoid().max(-1).values  # (B, A)
            pos, neg = fg_mask, ~fg_mask
            bins = 100
            # bincount instead of histc: histc has no deterministic CUDA kernel, which
            # warns on every batch when --deterministic is on.
            idx = (s_max.detach() * (bins - 1)).long().clamp_(0, bins - 1)
            if pos.any():
                mon["pos_hist"] += torch.bincount(idx[pos], minlength=bins).double()
                mon["pos_sum"] += s_max[pos].double().sum()
                mon["pos_n"] += pos.sum()
            if neg.any():
                mon["neg_hist"] += torch.bincount(idx[neg], minlength=bins).double()
                mon["neg_sum"] += s_max[neg].double().sum()
                mon["neg_n"] += neg.sum()

        n_pos = fg_mask.sum().clamp_min(1)
        extra = torch.zeros(4, device=self.device, dtype=torch.float32)

        # Channel 8 is objectness, not a quality regression: 1 on TAL positives,
        # 0 on every other anchor. It is used as a gate at inference, so it must
        # learn background suppression rather than localisation quality.
        #
        # pos_weight = (n_neg / n_pos) ** obj_pw_pow. With mean reduction an
        # exponent of 1 is class-balanced, so every anchor with a true
        # posterior above ~1/w is pushed above 0.5 (background near objects
        # included). The default 0.5 keeps positives high while letting
        # negatives dominate the total, which is what suppresses background.
        #
        # This term is deliberately outside the `fg_mask.any()` block: an
        # all-negative batch (no GT, or no assigned anchor) is the cleanest
        # background signal and must still push objectness down everywhere.
        q_logits = preds["quality"].permute(0, 2, 1).reshape(-1).float()
        q_target_all = torch.zeros_like(q_logits)
        if fg_mask.any():
            q_target_all[fg_mask.reshape(-1)] = 1.0
            if q_iou is not None:
                # (1 - s) + s * IoU keeps every positive well above the negatives
                # while ordering them by the IoU the deployment NMS will see.
                q_target_all[fg_mask.reshape(-1)] = (1.0 - q_soft) + q_soft * q_iou
        pw_pow = float(getattr(self.hyp, "obj_pw_pow", 0.5))
        q_pos_weight = ((q_logits.numel() - n_pos) / n_pos).clamp(1.0, 10000.0).pow(pw_pow)
        extra[2] = F.binary_cross_entropy_with_logits(
            q_logits, q_target_all, reduction="mean", pos_weight=q_pos_weight
        )

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
                with torch.no_grad():
                    # do the two classification paths agree on the positive anchors?
                    joint_cls = preds["scores"][b].permute(1, 0)[m].argmax(-1)
                    attr_cls = color_logits.argmax(-1) * NUM_ROLE + number_logits.argmax(-1)
                    mon = self._monitors()
                    mon["agree"] += (joint_cls == attr_cls).sum()
                    mon["agree_n"] += m.sum()

            pred_px = pred_kpts * stride_tensor.view(1, -1, 1, 1)

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

        return torch.cat((loss, extra)), torch.cat((loss.detach(), extra.detach()))

    @staticmethod
    def kpts_decode(anchor_points: torch.Tensor, pred_kpts: torch.Tensor) -> torch.Tensor:
        """Decode Pico keypoint offsets to grid coordinates as ``anchor + raw``."""
        y = pred_kpts.clone()
        y[..., 0] += anchor_points[:, [0]]
        y[..., 1] += anchor_points[:, [1]]
        return y
