"""Validation-time objectness diagnostics for the Pico pose head.

The compact export keeps only channel 8 as the candidate source, so its
ranking must be measured on its own, independently of TAL:

* anchors are labelled by their *centre*: positive when the centre falls in
  exactly one GT box, ambiguous when it falls in several (excluded), negative
  otherwise;
* ``obj/AP``, ``obj/P@0.5`` and ``obj/R@0.5`` are anchor-level classification
  metrics computed from those labels;
* ``obj/GT_R@0.5`` is the per-GT recall (fraction of GT boxes that have at least
  one anchor with objectness > 0.5 whose centre falls inside) - this is the
  recall ceiling of the exported candidate set.

The monitor hooks the head of the EMA model during validation and wraps
``validator.update_metrics`` so no extra forward pass is needed.
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = ("AnchorStats", "ObjectnessMonitor", "hist_auc", "assignment_summary", "ASSIGN_BUCKETS")

BINS = 1000

# GT side-length thresholds in pixels: 32 px, 96 px, 192 px (area = side^2).
ASSIGN_BUCKETS = ("tiny", "small", "medium", "large")
ASSIGN_EDGES = (32.0**2, 96.0**2, 192.0**2)


def hist_auc(pos_hist: np.ndarray, neg_hist: np.ndarray) -> float:
    """AUC from two histograms over the same [0, 1] bins (descending thresholds)."""
    n_pos, n_neg = float(pos_hist.sum()), float(neg_hist.sum())
    if n_pos <= 0 or n_neg <= 0:
        return float("nan")
    tpr = np.concatenate(([0.0], np.cumsum(pos_hist[::-1]) / n_pos))
    fpr = np.concatenate(([0.0], np.cumsum(neg_hist[::-1]) / n_neg))
    return float(np.trapz(tpr, fpr))


def assignment_summary(
    areas: np.ndarray, counts: np.ndarray, topk: int, edges: tuple[float, ...] = ASSIGN_EDGES
) -> dict[str, float]:
    """Per-area-bucket summary of how many anchors each GT received from TAL.

    Args:
        areas: (G,) GT box areas in input-image pixels.
        counts: (G,) number of assigned positive anchors per GT.
        topk: the assigner's topk setting (reference for starvation).
        edges: area thresholds between buckets (default 32^2, 96^2, 192^2).

    Returns:
        Flattened dict with ``assign/<bucket>_med`` (median anchors), ``_iqr``
        (q75 - q25), ``_n`` (number of GTs) plus ``assign/zero_frac`` (GTs with
        0 anchors) and ``assign/mean_ratio`` (mean anchors / topk).
    """
    nan = float("nan")
    out: dict[str, float] = {}
    areas = np.asarray(areas, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    if areas.size == 0:
        for name in ASSIGN_BUCKETS:
            out[f"assign/{name}_med"] = nan
            out[f"assign/{name}_iqr"] = nan
            out[f"assign/{name}_n"] = 0
        out["assign/zero_frac"] = nan
        out["assign/mean_ratio"] = nan
        return out

    bucket = np.digitize(areas, edges)
    for i, name in enumerate(ASSIGN_BUCKETS):
        c = counts[bucket == i]
        out[f"assign/{name}_n"] = int(c.size)
        if c.size:
            out[f"assign/{name}_med"] = float(np.median(c))
            out[f"assign/{name}_iqr"] = float(np.percentile(c, 75) - np.percentile(c, 25))
        else:
            out[f"assign/{name}_med"] = nan
            out[f"assign/{name}_iqr"] = nan
    out["assign/zero_frac"] = float((counts == 0).mean())
    out["assign/mean_ratio"] = float(counts.mean() / max(topk, 1))
    return out


class AnchorStats:
    """Accumulate anchor-level objectness labels and scores over a val pass."""

    def __init__(self, thr: float = 0.5):
        self.thr = float(thr)
        self.pos_hist = np.zeros(BINS, np.float64)
        self.neg_hist = np.zeros(BINS, np.float64)
        self.n_pos = 0
        self.n_neg = 0
        self.n_amb = 0
        self.tp = 0  # positives with obj > thr
        self.fp = 0  # negatives with obj > thr
        self.gt_total = 0
        self.gt_hit = 0

    @torch.no_grad()
    def add(self, obj: torch.Tensor, centers: torch.Tensor, boxes_xyxy: torch.Tensor) -> None:
        """Add one image.

        Args:
            obj: (A,) objectness probabilities.
            centers: (A, 2) anchor centres in input-image pixels.
            boxes_xyxy: (G, 4) GT boxes in input-image pixels (empty allowed).
        """
        if obj.numel() == 0:
            return
        if boxes_xyxy.numel():
            cx, cy = centers[:, 0:1], centers[:, 1:2]
            inside = (
                (cx >= boxes_xyxy[None, :, 0])
                & (cx <= boxes_xyxy[None, :, 2])
                & (cy >= boxes_xyxy[None, :, 1])
                & (cy <= boxes_xyxy[None, :, 3])
            )
            count = inside.sum(1)
        else:
            inside = torch.zeros((obj.numel(), 0), dtype=torch.bool, device=obj.device)
            count = torch.zeros(obj.numel(), dtype=torch.long, device=obj.device)

        pos = count == 1
        neg = count == 0
        self.n_pos += int(pos.sum())
        self.n_neg += int(neg.sum())
        self.n_amb += int((count > 1).sum())
        if pos.any():
            self.pos_hist += np.histogram(obj[pos].float().cpu().numpy(), bins=BINS, range=(0.0, 1.0))[0]
        if neg.any():
            self.neg_hist += np.histogram(obj[neg].float().cpu().numpy(), bins=BINS, range=(0.0, 1.0))[0]

        hit = obj > self.thr
        self.tp += int((pos & hit).sum())
        self.fp += int((neg & hit).sum())
        if inside.numel():
            self.gt_total += int(inside.shape[1])
            self.gt_hit += int((inside & hit[:, None]).any(0).sum())

    def result(self) -> dict[str, float]:
        out = {
            "obj/P@0.5": self.tp / max(self.tp + self.fp, 1),
            "obj/R@0.5": self.tp / max(self.n_pos, 1),
            "obj/GT_R@0.5": self.gt_hit / max(self.gt_total, 1),
        }
        n_pos, n_neg = self.pos_hist.sum(), self.neg_hist.sum()
        if n_pos > 0 and n_neg > 0:
            pos = np.cumsum(self.pos_hist[::-1])
            neg = np.cumsum(self.neg_hist[::-1])
            rec = pos / n_pos
            prec = pos / np.maximum(pos + neg, 1.0)
            mrec = np.concatenate(([0.0], rec, [1.0]))
            mpre = np.concatenate(([1.0], prec, [0.0]))
            ap = np.mean([mpre[mrec >= r].max() if (mrec >= r).any() else 0.0 for r in np.linspace(0.0, 1.0, 101)])
            out["obj/AP"] = float(ap)
        else:
            out["obj/AP"] = float("nan")
        return out


def _box_iou_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None] - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


class ObjectnessMonitor:
    """Attach to the EMA head + validator for the duration of one validate()."""

    PR_THRESHOLDS = (0.5, 0.8)  # fixed score thresholds for deployment-style P/R
    PR_IOU = 0.5

    def __init__(self, trainer, thr: float = 0.5):
        self.trainer = trainer
        self.stats = AnchorStats(thr)
        self._handle = None
        self._orig_update = None
        self._last: tuple | None = None
        self.pr = {t: {"tp": 0, "fp": 0} for t in self.PR_THRESHOLDS}
        self.n_gt = 0

    def attach(self) -> None:
        model = getattr(self.trainer.ema, "ema", None) or self.trainer.model
        head = model.model[-1]
        self._handle = head.register_forward_hook(self._hook)
        validator = self.trainer.validator
        self._orig_update = validator.update_metrics
        validator.update_metrics = self._update

    def detach(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        if self._orig_update is not None:
            self.trainer.validator.update_metrics = self._orig_update
            self._orig_update = None

    def _hook(self, module, inputs, output) -> None:
        if isinstance(output, tuple) and len(output) == 2 and isinstance(output[1], dict):
            self._last = (module, output[1])

    def _update(self, preds, batch) -> None:
        self._orig_update(preds, batch)
        if self._last is None:
            return
        module, p = self._last
        self._last = None
        with torch.no_grad():
            obj = p["quality"][:, 0].sigmoid()  # (B, A)
            anchors, strides = module.anchors, module.strides  # (2, A), (1, A)
            if anchors is None:
                return
            centers = torch.stack((anchors[0] * strides[0], anchors[1] * strides[0]), -1)  # (A, 2)
            h, w = batch["img"].shape[2], batch["img"].shape[3]
            boxes = batch["bboxes"].reshape(-1, 4).float().to(centers.device)
            bidx = batch["batch_idx"].reshape(-1).long().to(centers.device)
            gcls = batch["cls"].reshape(-1).long()
            for b in range(obj.shape[0]):
                g = boxes[bidx == b]
                if len(g):
                    cx, cy, gw, gh = g[:, 0] * w, g[:, 1] * h, g[:, 2] * w, g[:, 3] * h
                    gxyxy = torch.stack((cx - gw / 2, cy - gh / 2, cx + gw / 2, cy + gh / 2), -1)
                else:
                    gxyxy = g.reshape(0, 4)
                self.n_gt += len(g)
                self.stats.add(obj[b], centers, gxyxy)
                self._match_fixed_conf(preds[b] if preds is not None and b < len(preds) else None, gxyxy, gcls[bidx == b])

    def _match_fixed_conf(self, det: dict | None, gxyxy: torch.Tensor, gcls: torch.Tensor) -> None:
        """Greedy class-aware matching at IoU 0.5, accumulated at fixed score thresholds."""
        if det is None or len(det.get("bboxes", ())) == 0:
            return
        pb = det["bboxes"].detach().float().cpu().numpy()
        pc = det["conf"].detach().float().cpu().numpy()
        pcls = det["cls"].detach().long().cpu().numpy()
        gx = gxyxy.detach().float().cpu().numpy()
        gc = gcls.detach().long().cpu().numpy()
        iou = _box_iou_np(pb, gx)
        used = np.zeros(len(gx), bool)
        for i in np.argsort(-pc):
            hit = -1
            if len(gx):
                same = (gc == pcls[i]) & (~used)
                if same.any():
                    j = int(np.argmax(np.where(same, iou[i], -1.0)))
                    if iou[i, j] >= self.PR_IOU:
                        hit = j
            if hit >= 0:
                used[hit] = True
            for t in self.PR_THRESHOLDS:
                if pc[i] >= t:
                    self.pr[t]["tp" if hit >= 0 else "fp"] += 1

    def result(self) -> dict[str, float]:
        out = self.stats.result()
        for t in self.PR_THRESHOLDS:
            tp, fp = self.pr[t]["tp"], self.pr[t]["fp"]
            out[f"val/P@{t}"] = tp / max(tp + fp, 1) if (tp + fp) else float("nan")
            out[f"val/R@{t}"] = tp / max(self.n_gt, 1) if self.n_gt else float("nan")
        return out
