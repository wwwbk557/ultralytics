# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""无框 Pose 损失: ATSS(纯几何) + VFL + Wing 四点 + 可见性 BCE.

为什么不能用 Ultralytics 默认的 TaskAlignedAssigner
--------------------------------------------------
TAL 的对齐度量是 ``t = s^alpha * IoU(pred_box, gt)^beta``, 依赖 **预测框**;
丢掉框分支后它用不了。这里换成 PicoDet 的做法:

* 分配(assigner): **ATSS** —— 只用 "锚框 vs GT" 的几何 IoU + 中心距离,
  不看网络输出, 所以完全不需要框回归;
* 分类(loss): **VFL**, 正样本 target = 该锚框与 GT 的 IoU(同样来自几何, 不是预测框),
  负样本 target = 0;
* 四点回归: **Wing loss**, 作用在 "按 stride 归一化的偏移量" 上 —— 与 PicoDet 的
  ``bbox2distance(center/stride, gt/stride)`` 同一量纲, w=10 / sigma=2 / C=1;
* 可见性: 第 3 个通道做 BCE。

损失槽位与 Ultralytics 的打印保持一致:
``[box_loss, pose_loss, kobj_loss, cls_loss, dfl_loss]``, 其中 box / dfl 恒为 0
—— 训练表格里这两列为 0 就是 "框分支已移除" 的直接证据。

``PicoAuxPoseLoss`` 是另一条路线: 训练时挂一个辅助框分支, 直接复用上游
``v8PoseLoss``(TAL + box/DFL + KeypointLoss), 只把关键点解码换成 Pico 的
``(anchor + raw)`` 约定; 导出时框分支不进入计算图。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.loss import v8PoseLoss
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.tal import make_anchors

__all__ = ("PicoPoseNoBoxLoss", "PicoAuxPoseLoss", "atss_assign", "wing_loss", "varifocal_loss", "pairwise_iou")


def pairwise_iou(box_a: torch.Tensor, box_b: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """两两 IoU, 输入 xyxy: ``(N, 4) x (M, 4) -> (N, M)``."""
    a, b = box_a[:, None, :], box_b[None, :, :]
    lt = torch.maximum(a[..., :2], b[..., :2])
    rb = torch.minimum(a[..., 2:], b[..., 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[..., 2] - a[..., 0]).clamp(min=0) * (a[..., 3] - a[..., 1]).clamp(min=0)
    area_b = (b[..., 2] - b[..., 0]).clamp(min=0) * (b[..., 3] - b[..., 1]).clamp(min=0)
    return inter / (area_a + area_b - inter + eps)


def varifocal_loss(pred_scores: torch.Tensor, target_scores: torch.Tensor, alpha: float = 0.75, gamma: float = 2.0):
    """VFL: 正样本用 IoU 加权, 负样本用 ``p^gamma`` 降权.

    Args:
        pred_scores: logits, shape (A, C)
        target_scores: 目标分数(正样本为 IoU, 负样本为 0), shape (A, C)
    """
    with torch.autocast(device_type=pred_scores.device.type, enabled=False):
        bce = F.binary_cross_entropy_with_logits(pred_scores.float(), target_scores, reduction="none")
        weight = alpha * pred_scores.sigmoid().pow(gamma) * (1 - target_scores) + target_scores
        return (bce * weight).sum() / target_scores.sum().clamp(min=1)


def wing_loss(delta: torch.Tensor, w: float = 10.0, sigma: float = 2.0, c: float = 1.0) -> torch.Tensor:
    """Wing loss(逐元素): ``w * ln(1 + x/sigma)`` (x < w) 否则 ``x - c``."""
    x = delta.abs()
    return torch.where(x < w, w * torch.log1p(x / sigma), x - c)


def atss_assign(
    anchor_points: torch.Tensor,
    stride_tensor: torch.Tensor,
    level_sizes: list[int],
    gt_boxes: torch.Tensor,
    topk: int = 9,
    grid_cell_scale: float = 5.0,
    force_gt: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ATSS 正样本分配(单张图, 只用几何信息).

    Args:
        anchor_points: (A, 2) 网格坐标(含 +0.5)
        stride_tensor: (A, 1) 每层 stride
        level_sizes: 每层 anchor 数量
        gt_boxes: (n, 4) 像素 xyxy
        topk: 每层每 GT 取多少个候选
        grid_cell_scale: 锚框边长 = ``stride * grid_cell_scale``(PicoDet 同款)
        force_gt: 若某 GT 一个正样本都没有, 强制给它 IoU 最大的锚框

    Returns:
        pos_mask (A,) bool, assigned_gt (A,) long(-1 为背景), assigned_iou (A,)
    """
    a = anchor_points.shape[0]
    device, dtype = anchor_points.device, anchor_points.dtype
    empty = (
        torch.zeros(a, dtype=torch.bool, device=device),
        torch.full((a,), -1, dtype=torch.long, device=device),
        torch.zeros(a, dtype=dtype, device=device),
    )
    n = gt_boxes.shape[0]
    if n == 0 or a == 0:
        return empty

    centers = anchor_points * stride_tensor  # (A, 2) 像素
    half = stride_tensor * grid_cell_scale * 0.5  # (A, 1)
    anchor_boxes = torch.cat((centers - half, centers + half), 1)  # (A, 4) xyxy

    ious = pairwise_iou(anchor_boxes, gt_boxes)  # (A, n)
    gt_centers = (gt_boxes[:, :2] + gt_boxes[:, 2:]) * 0.5
    dist = (centers[:, None, :] - gt_centers[None]).norm(dim=-1)  # (A, n)

    # 1) 每层每 GT 取中心距离最近的 topk 个候选
    cand = torch.zeros(a, n, dtype=torch.bool, device=device)
    cols = torch.arange(n, device=device)
    off = 0
    for size in level_sizes:
        if size <= 0:
            continue
        k = max(min(topk, size), 1)
        idx = dist[off : off + size].topk(k, dim=0, largest=False).indices  # (k, n)
        cand[off + idx, cols[None, :].expand(k, n)] = True
        off += size

    # 2) 每 GT 的候选 IoU 均值 + 标准差作为阈值
    thr = torch.zeros(n, dtype=dtype, device=device)
    for j in range(n):
        v = ious[cand[:, j], j]
        if v.numel():
            thr[j] = v.mean() + v.std()

    # 3) 阈值 + 锚点中心落在 GT 内
    inside = (
        (centers[:, None, 0] >= gt_boxes[None, :, 0])
        & (centers[:, None, 0] <= gt_boxes[None, :, 2])
        & (centers[:, None, 1] >= gt_boxes[None, :, 1])
        & (centers[:, None, 1] <= gt_boxes[None, :, 3])
    )
    pos = cand & (ious > thr[None, :]) & inside  # (A, n)

    # 4) 一个锚框只保留 IoU 最大的那个 GT
    best_iou, best_idx = ious.masked_fill(~pos, -1).max(dim=1)
    pos_mask = best_iou > 0

    # 5) 兜底: 每个 GT 至少有一个正样本(小目标没有中心落入时非常关键)
    if force_gt:
        covered = torch.zeros(n, dtype=torch.bool, device=device)
        if pos_mask.any():
            covered[best_idx[pos_mask]] = True
        for j in torch.nonzero(~covered).view(-1).tolist():
            i = int(ious[:, j].argmax())
            pos_mask[i] = True
            best_idx[i] = j
            best_iou[i] = ious[i, j]

    assigned = torch.where(pos_mask, best_idx, torch.full_like(best_idx, -1))
    assigned_iou = torch.where(pos_mask, best_iou.clamp(min=0), torch.zeros_like(best_iou))
    return pos_mask, assigned, assigned_iou


class PicoPoseNoBoxLoss:
    """无框 Pose 训练准则(配合 ``PicoPoseNoBox`` 头使用)."""

    def __init__(self, model, topk: int = 9, grid_cell_scale: float = 5.0, wing_w: float = 10.0, wing_sigma: float = 2.0):
        """从模型里取出 stride / nc / kpt_shape 等必要信息."""
        device = next(model.parameters()).device
        m = model.model[-1]
        self.hyp = model.args
        self.stride = m.stride
        self.nc = m.nc
        self.device = device
        self.kpt_shape = tuple(m.kpt_shape)
        self.nkpt, self.ndim = self.kpt_shape
        self.topk = topk
        self.grid_cell_scale = grid_cell_scale
        self.wing_w = wing_w
        self.wing_sigma = wing_sigma
        self.bce_vis = nn.BCEWithLogitsLoss()

    def __call__(self, preds, batch: dict):
        """兼容 trainer 的直接调用(训练态 preds 已是 dict)."""
        if isinstance(preds, (list, tuple)) and len(preds) == 2 and isinstance(preds[1], dict):
            preds = preds[1]
        return self.loss(preds, batch)

    def loss(self, preds: dict, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 ``(loss * bs, loss.detach())``, 槽位 = box/pose/kobj/cls/dfl."""
        device = self.device
        scores = preds["scores"]  # (bs, nc, A) logits
        kpts = preds["kpts"]  # (bs, nk, A) 原始回归(按 stride 归一化)
        feats = preds["feats"]
        bs, nc, a = scores.shape

        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)  # (A,2), (A,1)
        level_sizes = [int(f.shape[2] * f.shape[3]) for f in feats]
        imgsz = torch.tensor(feats[0].shape[2:], device=device, dtype=scores.dtype) * self.stride[0]
        w, h = imgsz[1], imgsz[0]

        batch_idx = batch["batch_idx"].long().to(device)
        gt_cls = batch["cls"].float().to(device).view(-1)
        gt_boxes = xywh2xyxy(batch["bboxes"].float().to(device) * torch.tensor([w, h, w, h], device=device))
        gt_kpts = batch["keypoints"].float().to(device).clone()
        if gt_kpts.numel():
            gt_kpts[..., 0] *= w
            gt_kpts[..., 1] *= h

        loss = torch.zeros(5, device=device)  # box, pose, kobj, cls, dfl
        n_pos = 0
        for bi in range(bs):
            sel = batch_idx == bi
            if not sel.any():
                continue
            gb, gc, gk = gt_boxes[sel], gt_cls[sel].long(), gt_kpts[sel]

            pos, assigned, pos_iou = atss_assign(
                anchor_points,
                stride_tensor,
                level_sizes,
                gb,
                topk=self.topk,
                grid_cell_scale=self.grid_cell_scale,
            )

            # --- 分类: VFL, 正样本 target 用锚框-GT 的 IoU(PicoDet 的 assigned_scores 同款)
            target = torch.zeros(a, nc, device=device, dtype=scores.dtype)
            if pos.any():
                target[pos, gc[assigned[pos]]] = pos_iou[pos]
            loss[3] += varifocal_loss(scores[bi].transpose(0, 1).float(), target)

            if not pos.any():
                continue
            n_p = int(pos.sum())
            n_pos += n_p

            raw = kpts[bi].view(self.nkpt, self.ndim, a)[:, :, pos]  # (nkpt, ndim, P)
            a0, a1 = anchor_points[pos, 0], anchor_points[pos, 1]  # (P,)
            st = stride_tensor[pos, 0].view(-1, 1, 1)  # (P, 1, 1)
            # 与 head 的 p = (anchor + raw) * stride 对齐 -> raw 目标 = gt/stride - anchor
            target_xy = (gk[assigned[pos]][..., :2] / st).permute(1, 2, 0)  # (nkpt, 2, P)
            tgt_x = target_xy[:, 0] - a0  # (nkpt, P)
            tgt_y = target_xy[:, 1] - a1
            loss[1] += (
                wing_loss(raw[:, 0] - tgt_x, self.wing_w, self.wing_sigma)
                + wing_loss(raw[:, 1] - tgt_y, self.wing_w, self.wing_sigma)
            ).mean() * n_p

            # --- 可见性
            if self.ndim == 3:
                vis_logits = raw[:, 2]  # (nkpt, P)
                vis_target = (gk[assigned[pos]][..., 2] > 0).permute(1, 0).float()
                loss[2] += self.bce_vis(vis_logits, vis_target) * n_p

        denom = max(n_pos, 1)
        loss[1] = loss[1] / denom * self.hyp.pose
        loss[2] = loss[2] / denom * self.hyp.kobj
        loss[3] = loss[3] / bs * self.hyp.cls
        # loss[0](box) 与 loss[4](dfl) 保持 0: 这就是 "去掉框分支" 在损失里的体现
        return loss * bs, loss.detach()


class PicoAuxPoseLoss(v8PoseLoss):
    """Pico 头 + 训练期辅助框的损失：完全复用上游 TAL + box/DFL + KeypointLoss。

    唯一需要适配的是关键点解码约定：Pico 用 ``(anchor + raw) * stride``，
    而上游 ``Pose`` 用 ``(raw * 2 + anchor - 0.5) * stride``。这里只重写
    ``kpts_decode``，其余 (TaskAlignedAssigner / BboxLoss / DFL / KeypointLoss)
    都走官方实现，避免自己维护一套分配逻辑。
    """

    @staticmethod
    def kpts_decode(anchor_points: torch.Tensor, pred_kpts: torch.Tensor) -> torch.Tensor:
        """把 Pico 原始关键点解码到 grid 坐标，供 KeypointLoss 使用。"""
        y = pred_kpts.clone()
        y[..., 0] += anchor_points[:, [0]]
        y[..., 1] += anchor_points[:, [1]]
        return y
