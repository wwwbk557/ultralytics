"""Loss functions for the compact four-keypoint pose head.

``PicoCompactPoseLoss`` combines TAL assignment, CIoU/DFL box loss,
OKS keypoint loss and three extra terms (colour BCE, pose geometry, photometric
consistency). The returned vector is
``[box, pose, kobj, cls, dfl, color, geom, consistency]``.

``cls`` is the seven-role head: BCE over all anchors against TAL soft targets,
the same quantity the exported graph uses as confidence. There is no objectness
channel. ``colour`` is a one-logit BCE trained on TAL positives only, weighted
by the role mask; at inference ``score[c * 7 + r] = color_p[c] * sigmoid(role_logit[r])``.
"""

from __future__ import annotations

import weakref

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.loss import v8PoseLoss

from .attrs import COLOR_BY_CLASS, NUM_ROLE, ROLE_BY_CLASS
from .monitor import assignment_summary, hist_auc

__all__ = ("PicoCompactPoseLoss", "pose_geometry_loss", "role_mask_weight")

# The criterion is a submodule of the model it scores.  Keeping a weakref *on
# the criterion* would either register the model as a child (reference cycle)
# or make the checkpoint unpicklable (weakref objects cannot be pickled), so
# the live model is tracked by criterion id at module scope instead.
_MODEL_REFS = {}


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


def _as_pair(value, default: tuple[float, float]) -> tuple[float, float]:
    """Parse a ``(lo, hi)`` pair; an empty value disables the term as ``(0, 0)``."""
    if value is None:
        return default
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return 0.0, 0.0
        parts = [p for p in text.replace("x", ",").split(",") if p.strip()]
    elif isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        parts = [value]
    if not parts:
        return 0.0, 0.0
    try:
        vals = [float(v) for v in parts]
    except (TypeError, ValueError):
        return default
    if len(vals) == 1:
        vals *= 2
    if len(vals) != 2:
        return default
    return vals[0], vals[1]


def role_mask_weight(
    points: torch.Tensor,
    area: tuple[float, float] = (20.0, 80.0),
    ratio: tuple[float, float] = (0.10, 0.20),
) -> torch.Tensor:
    """Soft weight in ``[0, 1]`` for the colour/role loss of each GT armour.

    A plate that is too small or too oblique to carry a readable number should
    not teach the classifier a guess.  Box/pose/objectness still learn from it;
    only the attribute cross-entropy is down-weighted.

    Args:
        points: ``(P, 4, 2)`` GT armour corners in input pixels, order LT RT RB LB.
        area: ``(lo, hi)`` ramp in ``px^2``; ``hi <= lo`` disables the term.
        ratio: ``(lo, hi)`` ramp on ``min(short, long) / max(short, long)``;
            ``hi <= lo`` disables the term.

    Returns:
        ``(P,)`` weights on the same device/dtype as ``points``.
    """
    if points.numel() == 0:
        return points.new_zeros((0,))
    x, y = points[..., 0], points[..., 1]
    area_px = 0.5 * torch.abs((x * y.roll(-1, 1) - y * x.roll(-1, 1)).sum(-1))
    width = 0.5 * ((points[:, 1] - points[:, 0]).norm(dim=-1) + (points[:, 2] - points[:, 3]).norm(dim=-1))
    height = 0.5 * ((points[:, 2] - points[:, 1]).norm(dim=-1) + (points[:, 3] - points[:, 0]).norm(dim=-1))
    edge_ratio = torch.minimum(width, height) / torch.maximum(width, height).clamp_min(1e-6)

    a_lo, a_hi = area
    r_lo, r_hi = ratio
    area_w = ((area_px - a_lo) / (a_hi - a_lo)).clamp(0.0, 1.0) if a_hi > a_lo else torch.ones_like(area_px)
    ratio_w = (
        ((edge_ratio - r_lo) / (r_hi - r_lo)).clamp(0.0, 1.0) if r_hi > r_lo else torch.ones_like(edge_ratio)
    )
    return area_w * ratio_w


class PicoCompactPoseLoss(v8PoseLoss):
    """TAL box/pose loss with colour, role (cls) and geometry terms."""

    def __init__(self, model):
        super().__init__(model)
        # ``preds["scores"]`` is the seven-role head now, so TAL must allocate
        # seven target-score columns and assign against labels mapped 14 -> 7.
        self.assigner.num_classes = NUM_ROLE
        # A module-scope weakref avoids registering the model as a child of its
        # own criterion (the criterion is itself a submodule of the model: a
        # strong reference would create a cycle and break train()/state_dict()).
        _MODEL_REFS[id(self)] = weakref.ref(model)
        self.color_map = torch.tensor(COLOR_BY_CLASS, device=self.device, dtype=torch.long)
        self.number_map = torch.tensor(ROLE_BY_CLASS, device=self.device, dtype=torch.long)

    # ---- epoch-level monitors (TAL score ranking / role-mask coverage) ----
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
                "role_w": z(),
                "role_n": z(),
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
        out: dict[str, float] = {"mon/s_auc": nan, "mon/s_gap": nan, "mon/role_mask": nan}
        pos_n, neg_n = float(m["pos_n"]), float(m["neg_n"])
        if pos_n > 0 and neg_n > 0:
            out["mon/s_auc"] = hist_auc(m["pos_hist"].cpu().numpy(), m["neg_hist"].cpu().numpy())
            out["mon/s_gap"] = float(m["pos_sum"] / pos_n - m["neg_sum"] / neg_n)
        if float(m["role_n"]) > 0:
            out["mon/role_mask"] = float(m["role_w"] / m["role_n"])
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

    @staticmethod
    def _photometric_tensor(img: torch.Tensor) -> torch.Tensor:
        """Differentiable exposure/blur view of a normalized ``(B,3,H,W)`` batch.

        Gamma is sampled per image (0.5 = brighter, 2.5 = darker), so the view
        covers the deployment exposure without a CPU round-trip.  Half the batch
        also gets a 3x3 box blur as a cheap motion-blur proxy.
        """
        bs = img.shape[0]
        dtype, device = img.dtype, img.device
        gamma = torch.empty(bs, 1, 1, 1, dtype=dtype, device=device).uniform_(0.5, 2.5)
        x = img.clamp(0.0, 1.0) ** gamma
        alpha = torch.empty(bs, 1, 1, 1, dtype=dtype, device=device).uniform_(0.8, 1.2)
        beta = torch.empty(bs, 1, 1, 1, dtype=dtype, device=device).uniform_(-0.05, 0.05)
        x = (x * alpha + beta).clamp(0.0, 1.0)
        blurred = F.avg_pool2d(x, 3, stride=1, padding=1)
        select = (torch.rand(bs, 1, 1, 1, dtype=dtype, device=device) < 0.5).to(dtype)
        return x * (1.0 - select) + blurred * select

    def _augmented_attrs(self, img: torch.Tensor) -> dict | None:
        """Run the backbone/neck/head once on a photometric view, attributes only.

        ``aux_box=False`` skips the train-only ``cv2`` branch, so the extra
        forward only pays for the compact colour/role head. Returns ``None``
        outside training or when the head layout is not the Pico head.
        """
        ref = _MODEL_REFS.get(id(self))
        model = ref() if ref is not None else None
        inner = getattr(model, "model", None)
        if (
            inner is None
            or not getattr(model, "training", False)
            or not hasattr(model, "_predict_once")
            or not hasattr(inner[-1], "aux_box")
        ):
            return None
        head = inner[-1]
        previous = head.aux_box
        head.aux_box = False
        try:
            return model._predict_once(self._photometric_tensor(img))
        finally:
            head.aux_box = previous

    @staticmethod
    def _consistency_kl(
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        weight: torch.Tensor,
        temperature: float = 2.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Weighted per-logit binary KL(student || teacher), teacher detached.

        Colour is one complementary logit and the seven roles are independent
        binaries, so this is a sum of Bernoulli KLs rather than a softmax KL.
        """
        student = student_logits / temperature
        teacher_p = torch.sigmoid(teacher_logits.detach() / temperature).clamp(1e-6, 1.0 - 1e-6)
        kl = teacher_p * (teacher_p.log() - F.logsigmoid(student)) + (1.0 - teacher_p) * (
            torch.log1p(-teacher_p) - F.logsigmoid(-student)
        )
        kl = kl.sum(-1) * (temperature**2)
        weight = weight.clamp_min(0.0)
        return (kl * weight).sum(), weight.sum()

    def loss(self, preds: dict, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the weighted loss and its detached per-term vector."""
        pred_kpts = preds["kpts"].permute(0, 2, 1).contiguous()
        bs = pred_kpts.shape[0]
        loss = torch.zeros(5, device=self.device, dtype=torch.float32)
        # TAL assigns against the seven role classes; the 14-class labels feed
        # only the colour head below, so map them for this call alone.
        tal_batch = {
            **batch,
            "cls": self.number_map[batch["cls"].view(-1).long()].view_as(batch["cls"]).to(batch["cls"].dtype),
        }
        (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor), det_loss, _ = (
            self.get_assigned_targets_and_loss(preds, tal_batch)
        )
        # The TAL assigner returns float zeros when the whole batch has no GT
        # (n_max_boxes == 0), everything else returns bool; normalise here because
        # bool masks are used as indices below and `~mask` needs a bool tensor.
        fg_mask = fg_mask.bool()
        loss[0], loss[3], loss[4] = det_loss[0], det_loss[1], det_loss[2]

        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=pred_kpts.dtype) * self.stride[0]
        self._track_assignment(batch, fg_mask, target_gt_idx, imgsz)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(bs, -1, *self.kpt_shape))

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
        extra = torch.zeros(3, device=self.device, dtype=torch.float32)

        if fg_mask.any():
            label_smoothing = float(getattr(self.hyp, "role_label_smoothing", 0.0))
            area_ramp = _as_pair(getattr(self.hyp, "role_mask_area", (20.0, 80.0)), (20.0, 80.0))
            ratio_ramp = _as_pair(getattr(self.hyp, "role_mask_ratio", (0.10, 0.20)), (0.10, 0.20))
            consistency_gain = float(getattr(self.hyp, "consistency", 0.0))
            preds_aug = self._augmented_attrs(batch["img"]) if consistency_gain > 0 else None
            consistency_sum = torch.zeros((), device=self.device, dtype=torch.float32)
            consistency_w = torch.zeros((), device=self.device, dtype=torch.float32)
            keypoint_scale = torch.tensor(
                (float(imgsz[1]), float(imgsz[0])), device=self.device, dtype=pred_kpts.dtype
            )
            for b in range(bs):
                m = fg_mask[b]
                if not m.any():
                    continue
                gt_idx = target_gt_idx[b][m]
                gt_sel = batch["batch_idx"].view(-1) == b
                gt_cls = batch["cls"].view(-1)[gt_sel].long()[gt_idx]
                color_logits = preds["color"][b].permute(1, 0)[m].float()
                number_logits = preds["number"][b].permute(1, 0)[m].float()
                # Colour is a property of the plate, not of an anchor: a plate
                # too small or too oblique to read still trains box/pose/role
                # through TAL, but should not teach the colour head a guess.
                gt_kpts = batch["keypoints"][gt_sel][gt_idx].to(self.device).float()[..., :2] * keypoint_scale
                role_w = role_mask_weight(gt_kpts, area_ramp, ratio_ramp)
                target = self.color_map[gt_cls].float()
                if label_smoothing > 0:
                    target = target * (1.0 - label_smoothing) + 0.5 * label_smoothing
                color_bce = F.binary_cross_entropy_with_logits(color_logits[:, 0], target, reduction="none")
                extra[0] += (color_bce * role_w).sum()
                if preds_aug is not None:
                    # The clean, labelled view is the teacher; the photometric view
                    # is the student, so the exported colour/role heads learn to
                    # produce the same posterior under dark/blurred input.
                    aug_color = preds_aug["color"][b].permute(1, 0)[m].float()
                    aug_number = preds_aug["number"][b].permute(1, 0)[m].float()
                    batch_sum, batch_w = self._consistency_kl(aug_color, color_logits, role_w)
                    consistency_sum += batch_sum
                    consistency_w += batch_w
                    batch_sum, batch_w = self._consistency_kl(aug_number, number_logits, role_w)
                    consistency_sum += batch_sum
                    consistency_w += batch_w
                with torch.no_grad():
                    mon = self._monitors()
                    mon["role_w"] += role_w.sum().double()
                    mon["role_n"] += role_w.numel()
            if preds_aug is not None:
                extra[2] = consistency_sum / consistency_w.clamp_min(1.0)

            pred_px = pred_kpts * stride_tensor.view(1, -1, 1, 1)

            geom_gain = float(getattr(self.hyp, "geom", 0.0))
            if geom_gain > 0:
                pts = pred_px[fg_mask][..., :2].float()
                extra[1] = pose_geometry_loss(
                    pts,
                    par=float(getattr(self.hyp, "geom_par", 0.10)),
                    opp=float(getattr(self.hyp, "geom_opp", 0.33)),
                    angle=float(getattr(self.hyp, "geom_angle", 0.23)),
                    axis=float(getattr(self.hyp, "geom_axis", 0.44)),
                )

        extra[0] *= getattr(self.hyp, "color", 0.1) / n_pos
        extra[1] *= float(getattr(self.hyp, "geom", 0.0))
        extra[2] *= float(getattr(self.hyp, "consistency", 0.0))
        extra = extra * bs

        return torch.cat((loss, extra)), torch.cat((loss.detach(), extra.detach()))

    @staticmethod
    def kpts_decode(anchor_points: torch.Tensor, pred_kpts: torch.Tensor) -> torch.Tensor:
        """Decode Pico keypoint offsets to grid coordinates as ``anchor + raw``."""
        y = pred_kpts.clone()
        y[..., 0] += anchor_points[:, [0]]
        y[..., 1] += anchor_points[:, [1]]
        return y
