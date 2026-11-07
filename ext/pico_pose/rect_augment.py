"""Rectangular (H, W) mosaic for the Pico pose pipeline.

Upstream ``Mosaic`` builds a square ``2*imgsz`` canvas and expects the affine
stage to crop it back to a square.  For 480x640 training we keep the same
4-tile layout but compose it on a ``2H x 2W`` canvas and set
``border = (-H//2, -W//2)``.  Upstream ``RandomPerspective`` then computes
``size = (canvas_w + 2*border_w, canvas_h + 2*border_h) = (W, H)`` and crops the
mosaic directly to the rectangular training size, with no letterbox padding.

The mosaic is installed by ``PicoRectTrainer.build_dataset`` only when
``mosaic > 0``; when it is 0 the stock square ``Mosaic`` is left untouched
(probability 0, so it never runs).
"""

from __future__ import annotations

import random

import numpy as np

from ultralytics.data.augment import Mosaic
from ultralytics.utils import LOGGER
from ultralytics.utils.instance import Instances

__all__ = ("RectMosaic", "install_rect_mosaic")


class RectMosaic(Mosaic):
    """Mosaic augmentation that composes into a ``2H x 2W`` canvas.

    The 2x2 layout, label padding and clipping mirror upstream ``Mosaic``;
    only the canvas size, the random centre range and the final clip box are
    rectangular so the downstream crop is exactly ``(H, W)``.
    """

    def __init__(self, dataset, imgsz=(480, 640), p: float = 1.0, n: int = 4):
        """Build a rect mosaic; ``imgsz`` is the (H, W) training size."""
        h, w = (int(imgsz[0]), int(imgsz[1])) if isinstance(imgsz, (list, tuple)) else (int(imgsz), int(imgsz))
        super().__init__(dataset, imgsz=max(h, w), p=p, n=n)
        self.rect_hw = (h, w)
        # Upstream crops with size = canvas + 2 * border, so this yields (W, H).
        self.border = (-h // 2, -w // 2)

    def _mix_transform(self, labels: dict) -> dict:
        """Use the rect 2x2 mosaic for n=4, otherwise defer to upstream."""
        if self.n != 4:
            LOGGER.warning(f"RectMosaic supports n=4 only, got n={self.n}; falling back to upstream mosaic")
            return super()._mix_transform(labels)
        return self._mosaic4_rect(labels)

    def _mosaic4_rect(self, labels: dict) -> dict:
        """Place four patches on a ``2H x 2W`` canvas around a jittered centre."""
        h_t, w_t = self.rect_hw
        s_h, s_w = 2 * h_t, 2 * w_t
        # Same relative jitter as upstream: centre lands in [0.5s, 1.5s].
        xc = int(random.uniform(w_t * 0.5, w_t * 1.5))
        yc = int(random.uniform(h_t * 0.5, h_t * 1.5))

        mosaic_labels = []
        for i in range(4):
            labels_patch = labels if i == 0 else labels["mix_labels"][i - 1]
            img = labels_patch["img"]
            h, w = labels_patch.pop("resized_shape")  # noqa: F841 (h/w are the patch size)

            if i == 0:  # top left
                img4 = np.full((s_h, s_w, img.shape[2]), 114, dtype=np.uint8)
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s_w), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(yc + h, s_h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            else:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s_w), min(yc + h, s_h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(h, y2a - y1a)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            padw, padh = x1a - x1b, y1a - y1b
            mosaic_labels.append(self._update_labels(labels_patch, padw, padh))

        final_labels = self._cat_labels_rect(mosaic_labels)
        final_labels["img"] = img4
        return final_labels

    def _cat_labels_rect(self, mosaic_labels: list[dict]) -> dict:
        """Concatenate patches, clip to the rectangular canvas and keep keypoints."""
        if not mosaic_labels:
            return {}
        h_t, w_t = self.rect_hw
        s_h, s_w = 2 * h_t, 2 * w_t
        cls = [labels["cls"] for labels in mosaic_labels]
        instances = [labels["instances"] for labels in mosaic_labels]
        final_labels = {
            "im_file": mosaic_labels[0]["im_file"],
            "ori_shape": mosaic_labels[0]["ori_shape"],
            "resized_shape": (s_h, s_w),
            "cls": np.concatenate(cls, 0),
            "instances": Instances.concatenate(instances, axis=0),
            "mosaic_border": self.border,
        }
        final_labels["instances"].clip(s_w, s_h)  # Instances.clip(width, height)
        good = final_labels["instances"].remove_zero_area_boxes()
        final_labels["cls"] = final_labels["cls"][good]
        if "texts" in mosaic_labels[0]:
            final_labels["texts"] = mosaic_labels[0]["texts"]
        return final_labels


def install_rect_mosaic(dataset, imgsz) -> int:
    """Swap every ``Mosaic`` in a built dataset for a ``RectMosaic``.

    Args:
        dataset: A built ``YOLODataset`` whose ``transforms`` already contain
            the upstream ``Mosaic`` (built with the (H, W) tuple).
        imgsz: Rectangular training size as ``(H, W)``.

    Returns:
        Number of swapped mosaics (0 or 1 for the stock pipeline).
    """
    if not isinstance(imgsz, (list, tuple)) or len(imgsz) != 2:
        return 0
    swapped = 0

    def walk(obj) -> None:
        nonlocal swapped
        items = getattr(obj, "transforms", None)
        if items is None:
            return
        for i, t in enumerate(items):
            if type(t) is Mosaic:  # exact type: do not re-wrap a RectMosaic
                items[i] = RectMosaic(t.dataset, imgsz=imgsz, p=t.p, n=t.n)
                swapped += 1
            else:
                walk(t)

    walk(dataset.transforms)
    return swapped
