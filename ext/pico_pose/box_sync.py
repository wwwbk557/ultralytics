"""Restore ``box == four-keypoint AABB`` after rotation/shear in the train pipeline.

Upstream ``RandomPerspective`` transforms the *box corners* and takes their
axis-aligned hull.  The Pico labels store the box as the AABB of the four plate
corners, so the two agree only at zero rotation: the quad sits inside its box,
and rotating the box inflates the target.  Measured on the capval train split
at ``degrees=60`` the GT box / keypoint-AABB IoU drops to 0.83 mean (60% of
instances below 0.9), which directly skews the TAL alignment for the
deploy-time corner boxes.

This transform recomputes the box from the (already transformed) keypoints for
instances whose four corners are all visible.  Truncated instances keep the
upstream clipped box: their corner coordinates were clamped to the border, so
a keypoint AABB would under-cover the visible plate.  Train-only, no-op for
datasets without keypoints and for segmentation labels.
"""

from __future__ import annotations

import numpy as np

from ultralytics.data.augment import BaseTransform, Compose, Format

__all__ = ("KeypointBoxSync", "install_keypoint_box_sync")


class KeypointBoxSync(BaseTransform):
    """Set ``instances.bboxes`` to the AABB of the four keypoints."""

    def __call__(self, labels: dict) -> dict:
        """Sync boxes in place and return ``labels``."""
        instances = labels.get("instances")
        if instances is None or instances.keypoints is None or len(instances) == 0:
            return labels
        segments = getattr(instances, "segments", None)
        if segments is not None and len(segments):
            return labels  # segmentation boxes come from polygons, not corners
        img = labels.get("img")
        if not isinstance(img, np.ndarray):
            return labels
        h, w = img.shape[:2]
        if getattr(instances, "normalized", False):
            instances.denormalize(w, h)
        kpts = np.asarray(instances.keypoints, dtype=np.float32)
        if kpts.shape[-1] >= 3:
            full = kpts[..., 2].min(1) >= 0.5
        else:
            full = np.ones(len(kpts), dtype=bool)
        if not full.any():
            return labels
        instances.convert_bbox("xyxy")
        boxes = np.asarray(instances.bboxes, dtype=np.float32).copy()
        xy = kpts[..., :2]
        boxes[full] = np.stack(
            (xy[..., 0].min(1), xy[..., 1].min(1), xy[..., 0].max(1), xy[..., 1].max(1)), 1
        )[full]
        instances.update(boxes, keypoints=kpts)
        return labels


def install_keypoint_box_sync(dataset) -> int:
    """Insert :class:`KeypointBoxSync` right before the final ``Format``."""
    transforms = getattr(dataset, "transforms", None)
    if not isinstance(transforms, Compose):
        return 0
    if any(isinstance(t, KeypointBoxSync) for t in transforms.transforms):
        return 0
    index = next((i for i, t in enumerate(transforms.transforms) if isinstance(t, Format)), None)
    if index is None:
        index = len(transforms.transforms)
    transforms.insert(index, KeypointBoxSync())
    return 1
