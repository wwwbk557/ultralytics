"""Environment-decorrelation augmentation for the Pico pose pipeline.

The role head learned the scene as a shortcut.  Putting a plate patch on a
flat canvas drops role accuracy from 97.8% to 47.2% while colour stays at
~100%; erasing the sibling plate costs only ~2 points.  This transform attacks
that shortcut on the train split only.

Four operations run after mosaic/affine/HSV/flip, i.e. on the final H,W canvas:

* ``relocate`` - cut one plate with its quad mask, inpaint the hole, and paste
  the plate at a plate-free location (under mosaic this is often another scene).
* ``context``  - replace a ``k x plate`` neighbourhood with a plate-free crop
  from elsewhere in the canvas, then paste the plate back.
* ``solo``     - keep one random plate and inpaint the rest together with their
  labels, so a plate cannot be classified from its sibling.
* ``scene``    - replace the whole canvas outside the kept plates with a donor
  scene (a plate-free crop of another training image) or a synthetic gradient.

Everything is opt-in (``p=0``), train-only, and no-op for datasets with
segments or without pose keypoints.
"""

from __future__ import annotations

import random

import cv2
import numpy as np

from ultralytics.data.augment import BaseTransform, Compose, Format, LetterBox

from .photometric import as_pair

__all__ = ("RandomEnvDecouple", "build_background_bank", "install_env_decouple")


def _aabb(points: np.ndarray) -> np.ndarray:
    """Axis-aligned box ``xyxy`` from four corner points."""
    return np.array(
        [points[:, 0].min(), points[:, 1].min(), points[:, 0].max(), points[:, 1].max()], dtype=np.float32
    )


def _rect_overlap(a, b, pad: int = 0) -> bool:
    """True when two ``xyxy`` rectangles intersect (optionally with a pad)."""
    return a[0] < b[2] + pad and a[2] > b[0] - pad and a[1] < b[3] + pad and a[3] > b[1] - pad


def _expand_quad(points: np.ndarray, grow_u: float, grow_v: float) -> np.ndarray:
    """Grow the 4-point plate quad outward in its own frame.

    ``u`` follows p1->p2 (the LED-bar axis) and ``v`` p1->p4, so a small
    ``grow_u`` keeps the cut tight to the bars while ``grow_v`` can include the
    digit that sticks out above/below them, at any roll angle.
    """
    p = np.asarray(points, np.float32)[:, :2]
    c = p.mean(0)
    u = (p[1] - p[0]) + (p[2] - p[3])
    v = (p[3] - p[0]) + (p[2] - p[1])
    u = u / max(float(np.linalg.norm(u)), 1e-6)
    v = v / max(float(np.linalg.norm(v)), 1e-6)
    uu = (p - c) @ u
    vv = (p - c) @ v
    u0, u1 = float(uu.min()) - grow_u, float(uu.max()) + grow_u
    v0, v1 = float(vv.min()) - grow_v, float(vv.max()) + grow_v
    return c + np.outer([u0, u1, u1, u0], u) + np.outer([v0, v0, v1, v1], v)


def _plate_mask(shape, points: np.ndarray, grow: int = 2, grow_u: float | None = None) -> np.ndarray:
    """Binary quad mask for one plate, grown in the plate's own frame."""
    if grow_u is None:
        grow_u = grow
    poly = np.round(_expand_quad(points, grow_u, grow)).astype(np.int32)
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    return mask


def _cut(img: np.ndarray, points: np.ndarray, grow: int = 1, grow_u: float | None = None):
    """Copy the plate pixels and return ``(patch, mask, (x, y))``."""
    poly = _expand_quad(points, grow if grow_u is None else grow_u, grow)
    h, w = img.shape[:2]
    x0 = max(0, int(np.floor(poly[:, 0].min())) - 1)
    y0 = max(0, int(np.floor(poly[:, 1].min())) - 1)
    x1 = min(w, int(np.ceil(poly[:, 0].max())) + 1)
    y1 = min(h, int(np.ceil(poly[:, 1].max())) + 1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    patch = img[y0:y1, x0:x1].copy()
    mask = np.zeros(patch.shape[:2], np.uint8)
    local = np.round(poly - np.array([x0, y0], np.float32)).astype(np.int32)
    cv2.fillPoly(mask, [local], 255)
    if not mask.any():
        return None
    return patch, mask, (x0, y0)


RELOCATE_MARGIN_FRAC = 0.20  # up/down: digit overhang beyond the LED bars
RELOCATE_MARGIN_MAX = 64
RELOCATE_SIDE_FRAC = 0.06  # left/right: just the bar + its glow
RELOCATE_SIDE_MAX = 14


def _painted_extent(points: np.ndarray, grow: int = 0) -> np.ndarray:
    """AABB of the plate's visible pixels: quad plus the uncapped digit overhang.

    ``_relocate_margins`` shrinks the cut to stay clear of siblings, so a plate
    cut only with ``grow=1`` loses the digit that sticks out above/below the
    keypoint quad.  Every paste-back and every "do not touch" check has to use
    this full extent, otherwise the label survives while the plate is clipped.
    """
    box = _aabb(points)
    gu, gv = _relocate_margins(box, [])
    return _aabb(_expand_quad(points, gu + grow, gv + grow))


def _relocate_margins(box, others) -> tuple[int, int]:
    """Cut margins ``(side, top_bottom)`` in px for relocate.

    Digits stick out above/below the 4-keypoint quad (~20% of the longer side
    on average) but stay inside the LED bars horizontally, so only the
    vertical margin follows the plate size; the horizontal one just covers the
    bar.  Both are capped by the nearest sibling so the cut never captures
    another labelled plate.
    """
    w = float(box[2]) - float(box[0])
    h = float(box[3]) - float(box[1])
    long_side = max(w, h)
    gu = max(2, min(int(round(RELOCATE_SIDE_FRAC * long_side)), RELOCATE_SIDE_MAX))
    gv = max(2, min(int(round(RELOCATE_MARGIN_FRAC * long_side)), RELOCATE_MARGIN_MAX))
    gaps = []
    for b in others:
        dx = max(float(b[0]) - float(box[2]), float(box[0]) - float(b[2]), 0.0)
        dy = max(float(b[1]) - float(box[3]), float(box[1]) - float(b[3]), 0.0)
        gaps.append(max(dx, dy))
    if gaps:
        cap = int(min(gaps)) - 5
        gu, gv = min(gu, cap), min(gv, cap)
    return max(0, gu), max(0, gv)


def _paste(img: np.ndarray, patch: np.ndarray, mask: np.ndarray, x: int, y: int) -> bool:
    """Paste only the masked plate pixels at ``(x, y)``; False when out of bounds."""
    h, w = img.shape[:2]
    ph, pw = patch.shape[:2]
    if x < 0 or y < 0 or x + pw > w or y + ph > h:
        return False
    roi = img[y : y + ph, x : x + pw]
    np.copyto(roi, patch, where=mask[..., None].astype(bool))
    return True


def _synthetic_background(h: int, w: int) -> np.ndarray:
    """Flat colour + low-frequency shading + light sensor noise."""
    base = np.array([random.randint(50, 200) for _ in range(3)], np.float32)
    yy = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None, None]
    xx = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :, None]
    img = np.clip(base[None, None, :] + 14.0 * xx + 10.0 * yy, 0.0, 255.0)
    noise = np.random.normal(0.0, 4.0, (h, w, 3)).astype(np.float32)
    return np.clip(img + noise, 0.0, 255.0).astype(np.uint8)


def _record_boxes(record: dict, w: int, h: int) -> np.ndarray:
    """GT boxes of one dataset cache record as absolute ``xyxy``."""
    boxes = record.get("bboxes")
    if boxes is None:
        return np.zeros((0, 4), np.float32)
    arr = np.asarray(boxes, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, 4), np.float32)
    if arr.ndim == 1:
        arr = arr[None]
    if arr.shape[1] < 4:
        return np.zeros((0, 4), np.float32)
    cx, cy, bw, bh = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    if record.get("normalized", True):
        cx, cy, bw, bh = cx * w, cy * h, bw * w, bh * h
    return np.stack((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), 1)


def _free_crop(boxes: np.ndarray, iw: int, ih: int, cw: int, ch: int, tries: int = 50):
    """Random ``ch x cw`` crop that contains no GT box, or None."""
    for _ in range(tries):
        x = random.randint(0, iw - cw)
        y = random.randint(0, ih - ch)
        crop = (x, y, x + cw, y + ch)
        if not any(_rect_overlap(crop, b, pad=2) for b in boxes):
            return x, y
    return None


def build_background_bank(dataset, hw, n: int = 32):
    """Sample ``n`` plate-free crops from the train images, downscaled 4x.

    The bank is a small uint8 array inherited by DataLoader workers, so scene
    replacement does not touch the disk during training.
    """
    if n <= 0:
        return None
    labels = getattr(dataset, "labels", None)
    if not labels:
        return None
    if isinstance(hw, (list, tuple)) and len(hw) == 2:
        h, w = int(hw[0]), int(hw[1])
    else:
        h = w = int(hw)
    th, tw = max(1, h // 4), max(1, w // 4)
    order = list(range(len(labels)))
    random.shuffle(order)
    bank = []
    for idx in order[: min(len(order), n * 10 + 20)]:
        if len(bank) >= n:
            break
        record = labels[idx]
        im_file = record.get("im_file")
        img = cv2.imread(str(im_file)) if im_file else None
        if img is None or img.ndim != 3:
            continue
        ih, iw = img.shape[:2]
        ch, cw = max(8, ih // 2), max(8, iw // 2)
        if cw >= iw or ch >= ih:
            continue
        spot = _free_crop(_record_boxes(record, iw, ih), iw, ih, cw, ch)
        if spot is None:
            continue
        x, y = spot
        crop = img[y : y + ch, x : x + cw]
        if crop.size == 0:
            continue
        bank.append(cv2.resize(crop, (tw, th), interpolation=cv2.INTER_AREA))
    return np.stack(bank) if bank else None


class RandomEnvDecouple(BaseTransform):
    """Relocate / erase / re-scene plates to remove the scene->role shortcut.

    Args:
        p: Probability of applying any operation to one image.
        relocate_p: Per-plate probability of cutting the plate out and pasting
            it at a random plate-free location.
        context_p: Per-plate probability of replacing a neighbourhood around
            the plate with background sampled elsewhere in the canvas.
        solo_p: Per-image probability of keeping one plate and erasing the rest
            (only for images with two or more plates).
        scene_p: Per-image probability of replacing the whole background
            outside the kept plates.
        flat_p: Within a scene replacement, probability of a synthetic flat
            background instead of a donor scene.
        donors: ``(N, h, w, 3)`` uint8 background bank, or None.
        context_scale: ``(lo, hi)`` neighbourhood size as a multiple of the
            plate's larger side.
        max_tries: Random-placement retries before an operation is skipped.
    """

    def __init__(
        self,
        p: float = 0.0,
        relocate_p: float = 0.35,
        context_p: float = 0.5,
        solo_p: float = 0.25,
        scene_p: float = 0.35,
        flat_p: float = 0.4,
        donors=None,
        context_scale=(1.6, 2.6),
        max_tries: int = 24,
    ) -> None:
        super().__init__()
        self.p = float(p)
        self.relocate_p = float(relocate_p)
        self.context_p = float(context_p)
        self.solo_p = float(solo_p)
        self.scene_p = float(scene_p)
        self.flat_p = float(flat_p)
        self.donors = donors
        self.context_scale = as_pair(context_scale, (1.6, 2.6))
        self.max_tries = int(max_tries)

    def __call__(self, labels: dict) -> dict:
        """Apply the enabled operations and keep labels, boxes and keypoints consistent."""
        if self.p <= 0.0 or random.random() >= self.p:
            return labels
        img = labels.get("img")
        instances = labels.get("instances")
        if not isinstance(img, np.ndarray) or img.ndim != 3 or img.dtype != np.uint8:
            return labels
        if instances is None or len(instances) == 0 or instances.keypoints is None:
            return labels
        segments = getattr(instances, "segments", None)
        if segments is not None and len(segments):
            return labels  # segmentation labels have no plate quad to cut

        h, w = img.shape[:2]
        if getattr(instances, "normalized", False):
            instances.denormalize(w, h)
        instances.convert_bbox("xyxy")
        kpts = np.array(instances.keypoints, dtype=np.float32, copy=True)
        cls = labels.get("cls")
        n = len(kpts)
        keep = list(range(n))
        boxes = [_aabb(kpts[i]) for i in range(n)]

        if n >= 2 and random.random() < self.solo_p:
            kept = random.randrange(n)
            for i in range(n):
                if i != kept:
                    gu, gv = _relocate_margins(boxes[i], [_painted_extent(kpts[kept])])
                    mask = _plate_mask((h, w), kpts[i], grow=gv, grow_u=gu)
                    img[:] = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
            keep = [kept]

        order = keep[:]
        random.shuffle(order)
        for i in order:
            if random.random() < self.relocate_p and self._relocate(img, kpts, i, boxes, keep):
                continue
            if random.random() < self.context_p:
                avoid = [_painted_extent(kpts[j]) for j in keep if j != i]
                self._replace_context(img, kpts[i], avoid)

        if random.random() < self.scene_p:
            self._replace_scene(img, [kpts[i] for i in keep])

        kept_kpts = kpts[np.asarray(keep)]
        kept_boxes = np.stack([_aabb(points) for points in kept_kpts]).astype(np.float32)
        if cls is not None:
            labels["cls"] = cls[np.asarray(keep)]
        instances.update(kept_boxes, keypoints=kept_kpts)
        labels["img"] = np.ascontiguousarray(img)
        return labels

    def _relocate(self, img, kpts, index, boxes, keep) -> bool:
        """Cut plate ``index`` out, inpaint its hole and paste it elsewhere."""
        h, w = img.shape[:2]
        others = [_painted_extent(kpts[j]) for j in keep if j != index]
        gu, gv = _relocate_margins(boxes[index], others)
        cut = _cut(img, kpts[index], grow=gv, grow_u=gu)
        if cut is None:
            return False
        patch, pmask, (px, py) = cut
        ph, pw = patch.shape[:2]
        if ph >= h or pw >= w:
            return False
        avoid = [_painted_extent(kpts[j]) for j in keep]
        for _ in range(self.max_tries):
            dx = random.randint(-px, w - px - pw)
            dy = random.randint(-py, h - py - ph)
            new_box = (px + dx, py + dy, px + dx + pw, py + dy + ph)
            if any(_rect_overlap(new_box, box, pad=2) for box in avoid):
                continue
            source_mask = _plate_mask((h, w), kpts[index], grow=gv + 1, grow_u=gu + 1)
            img[:] = cv2.inpaint(img, source_mask, 3, cv2.INPAINT_TELEA)
            if not _paste(img, patch, pmask, px + dx, py + dy):
                return False
            kpts[index, :, 0] += dx
            kpts[index, :, 1] += dy
            boxes[index] = _aabb(kpts[index])
            return True
        return False

    def _replace_context(self, img, points, avoid) -> None:
        """Replace the neighbourhood around one plate with background elsewhere."""
        h, w = img.shape[:2]
        box = _aabb(points)
        size = max(float(box[2] - box[0]), float(box[3] - box[1]), 4.0)
        pad = int(round(random.uniform(*self.context_scale) * size))
        x0 = max(0, int(box[0]) - pad)
        y0 = max(0, int(box[1]) - pad)
        x1 = min(w, int(np.ceil(box[2])) + pad)
        y1 = min(h, int(np.ceil(box[3])) + pad)
        rw, rh = x1 - x0, y1 - y0
        if rw < 8 or rh < 8 or rw >= w or rh >= h:
            return
        region = (x0, y0, x1, y1)
        if any(_rect_overlap(region, other, pad=1) for other in avoid):
            return

        donor = None
        for _ in range(self.max_tries):
            sx = random.randint(0, w - rw)
            sy = random.randint(0, h - rh)
            crop = (sx, sy, sx + rw, sy + rh)
            if _rect_overlap(crop, region, pad=2):
                continue
            if any(_rect_overlap(crop, other, pad=2) for other in avoid):
                continue
            donor = img[sy : sy + rh, sx : sx + rw].copy()
            break

        gu, gv = _relocate_margins(_aabb(points), avoid)
        cut = _cut(img, points, grow=gv, grow_u=gu)
        if cut is None:
            return
        patch, pmask, (px, py) = cut
        if donor is None:
            donor = cv2.GaussianBlur(img[y0:y1, x0:x1], (0, 0), sigmaX=max(2.0, size / 3.0))
        img[y0:y1, x0:x1] = donor[:rh, :rw]
        _paste(img, patch, pmask, px, py)

    def _replace_scene(self, img, plates) -> None:
        """Replace everything outside the kept plates with a donor/synthetic scene."""
        h, w = img.shape[:2]
        cuts = []
        for points in plates:
            gu, gv = _relocate_margins(_aabb(points), [])
            cut = _cut(img, points, grow=gv, grow_u=gu)
            if cut is not None:
                cuts.append(cut)
        if not cuts:
            return
        if self.donors is not None and len(self.donors) and random.random() >= self.flat_p:
            donor = self.donors[random.randrange(len(self.donors))]
            background = cv2.resize(donor, (w, h), interpolation=cv2.INTER_LINEAR)
            if random.random() < 0.5:
                background = background[:, ::-1]
            img[:] = np.ascontiguousarray(background)
        else:
            img[:] = _synthetic_background(h, w)
        for patch, pmask, (px, py) in cuts:
            _paste(img, patch, pmask, px, py)


def install_env_decouple(
    dataset,
    hw=(480, 640),
    p: float = 0.0,
    relocate_p: float = 0.35,
    context_p: float = 0.5,
    solo_p: float = 0.25,
    scene_p: float = 0.35,
    flat_p: float = 0.4,
    donors: int = 32,
    context_scale=(1.6, 2.6),
) -> int:
    """Insert :class:`RandomEnvDecouple` right before the final ``Format``.

    Returns 1 when inserted and 0 when the pipeline already carries the
    transform (or is not a ``Compose`` pipeline).
    """
    transforms = getattr(dataset, "transforms", None)
    if not isinstance(transforms, Compose):
        return 0
    if any(isinstance(t, RandomEnvDecouple) for t in transforms.transforms):
        return 0
    bank = build_background_bank(dataset, hw, donors) if scene_p > 0 else None
    index = next((i for i, t in enumerate(transforms.transforms) if isinstance(t, Format)), None)
    if index is None:
        index = next(
            (i for i, t in enumerate(transforms.transforms) if isinstance(t, LetterBox)),
            len(transforms.transforms),
        )
    transforms.insert(
        index,
        RandomEnvDecouple(
            p=p,
            relocate_p=relocate_p,
            context_p=context_p,
            solo_p=solo_p,
            scene_p=scene_p,
            flat_p=flat_p,
            donors=bank,
            context_scale=context_scale,
        ),
    )
    return 1
