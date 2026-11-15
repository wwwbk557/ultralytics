"""Target-level scale jitter for the Pico pose pipeline.

``--scale`` in ``RandomPerspective`` zooms the whole canvas, so every plate in
the sample grows or shrinks together with the scene.  This transform changes
the apparent size of one plate instead: cut it with the same margins
``RandomEnvDecouple`` uses (the digit overhangs the keypoint quad vertically),
resize the patch, paste it back on its own centre and move the keypoints with
it.  Train only, no-op for segmentation labels and datasets without keypoints.
"""

from __future__ import annotations

import random

import cv2
import numpy as np

from ultralytics.data.augment import BaseTransform, Compose, Format

from .env_decouple import (
    _aabb,
    _cut,
    _expand_quad,
    _paste,
    _painted_extent,
    _plate_mask,
    _rect_overlap,
    _relocate_margins,
)
from .photometric import as_pair

__all__ = ("RandomPlateScale", "install_plate_scale")


class RandomPlateScale(BaseTransform):
    """Resize up to ``max_plates`` plates per image by a random factor.

    Args:
        p: Per-image probability of touching any plate.
        scale: ``(lo, hi)`` uniform factor applied to the picked plate.
        max_plates: Plates resized per image; the rest keep their size.
        min_px: Plates whose keypoint AABB long side is below this many canvas
            pixels are only kept or enlarged, never shrunk.
    """

    def __init__(self, p: float = 0.0, scale=(0.85, 1.15), max_plates: int = 2, min_px: float = 16.0) -> None:
        super().__init__()
        self.p = float(p)
        self.scale = as_pair(scale, (0.85, 1.15))
        self.max_plates = int(max_plates)
        self.min_px = float(min_px)

    def __call__(self, labels: dict) -> dict:
        """Resize plates in place and keep boxes/keypoints consistent."""
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
        boxes = [_aabb(points) for points in kpts]
        order = list(range(len(kpts)))
        random.shuffle(order)
        changed = 0
        for i in order:
            if changed >= self.max_plates:
                break
            if self._resize(img, kpts, i, boxes):
                changed += 1
        if not changed:
            return labels
        instances.update(np.stack(boxes).astype(np.float32), keypoints=kpts)
        return labels

    def _resize(self, img: np.ndarray, kpts: np.ndarray, index: int, boxes: list) -> bool:
        """Scale plate ``index`` about the centre of its cut patch."""
        h, w = img.shape[:2]
        others = [_painted_extent(kpts[j]) for j in range(len(kpts)) if j != index]
        gu, gv = _relocate_margins(boxes[index], others)
        cut = _cut(img, kpts[index], grow=gv, grow_u=gu)
        if cut is None:
            return False
        patch, pmask, (px, py) = cut
        ph, pw = patch.shape[:2]
        long_side = max(float(boxes[index][2]) - float(boxes[index][0]), float(boxes[index][3]) - float(boxes[index][1]))
        lo, hi = self.scale
        if long_side < self.min_px:
            lo, hi = max(lo, 1.0), max(hi, 1.0)
            if hi - lo < 1e-6:
                return False
        factor = random.uniform(lo, hi)
        nw, nh = int(round(pw * factor)), int(round(ph * factor))
        if nw < 4 or nh < 4 or nw > w or nh > h:
            return False
        sx, sy = nw / pw, nh / ph
        nx = int(round(px + pw / 2.0 - nw / 2.0))
        ny = int(round(py + ph / 2.0 - nh / 2.0))
        if nx < 0 or ny < 0 or nx + nw > w or ny + nh > h:
            return False  # checked before the inpaint, else the plate would be erased with its label kept
        moved = kpts[index].copy()
        moved[:, 0] = nx + (kpts[index][:, 0] - px) * sx
        moved[:, 1] = ny + (kpts[index][:, 1] - py) * sy
        new_box = _aabb(moved)
        if new_box[0] < 0 or new_box[1] < 0 or new_box[2] > w or new_box[3] > h:
            return False
        paste = _expand_quad(kpts[index], gu, gv)
        paste = np.stack([nx + (paste[:, 0] - px) * sx, ny + (paste[:, 1] - py) * sy], -1)
        paste = _aabb(paste)
        if any(_rect_overlap(paste, _painted_extent(kpts[j]), pad=2) for j in range(len(kpts)) if j != index):
            return False
        interp = cv2.INTER_AREA if factor < 1.0 else cv2.INTER_LINEAR
        patch_r = cv2.resize(patch, (nw, nh), interpolation=interp)
        mask_r = cv2.resize(pmask, (nw, nh), interpolation=cv2.INTER_NEAREST) > 127
        hole = _plate_mask((h, w), kpts[index], grow=gv + 1, grow_u=gu + 1)
        img[:] = cv2.inpaint(img, hole, 3, cv2.INPAINT_TELEA)
        if not _paste(img, patch_r, mask_r, nx, ny):
            return False
        kpts[index] = moved
        boxes[index] = new_box
        return True


def install_plate_scale(
    dataset, p: float = 0.0, scale=(0.85, 1.15), max_plates: int = 2, min_px: float = 16.0
) -> int:
    """Insert :class:`RandomPlateScale` right before the final ``Format``."""
    transforms = getattr(dataset, "transforms", None)
    if not isinstance(transforms, Compose):
        return 0
    if any(isinstance(t, RandomPlateScale) for t in transforms.transforms):
        return 0
    index = next((i for i, t in enumerate(transforms.transforms) if isinstance(t, Format)), None)
    if index is None:
        index = len(transforms.transforms)
    transforms.insert(index, RandomPlateScale(p=p, scale=scale, max_plates=max_plates, min_px=min_px))
    return 1
