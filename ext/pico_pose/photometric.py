"""Target-mean photometric and motion-blur augmentation for the Pico pipeline.

The deployment camera is much darker than most of the training corpus
(``mean`` 5-10 vs 30-45), and the exported role head learned a shortcut where
a dark, digit-less plate looks like ``outpost``.  Uniformly brightening the
input is not an option for the deployed graph, so the training distribution
has to cover the deployment exposure instead.

``RandomPhotometric`` samples a target mean brightness per image and solves the
gamma that maps the image mean to that target.  Sampling the target from the
full deployment-to-training range (default 5-60) means every class is seen at
every exposure, which removes the class/brightness correlation rather than
just adding more bright samples.  Motion blur and sensor noise are applied
independently to mimic a moving target.

The transform only runs on the train split and is inserted after
mosaic/affine/flip but before the final ``Format`` tensor conversion, so it
still sees a uint8 HWC canvas.  The corpus is 4:3, so the 480x640 letterbox
adds no padding; on other aspect ratios the 114 padding is gamma-corrected too.
"""

from __future__ import annotations

import math
import random

import cv2
import numpy as np

from ultralytics.data.augment import BaseTransform, Compose, Format, LetterBox

__all__ = ("RandomPhotometric", "install_photometric")


def as_pair(value, default: tuple[float, float]) -> tuple[float, float]:
    """Parse ``"5,60"`` / ``[5, 60]`` / a scalar into a two-float tuple."""
    if value is None:
        return default
    if isinstance(value, str):
        parts = [p for p in value.replace("x", ",").split(",") if p.strip()]
        vals = [float(p) for p in parts]
    elif isinstance(value, (list, tuple)):
        vals = [float(v) for v in value]
    else:
        vals = [float(value), float(value)]
    if not vals:
        return default
    if len(vals) == 1:
        vals *= 2
    if len(vals) != 2:
        raise ValueError(f"expected one or two numbers, got {value!r}")
    return vals[0], vals[1]


class RandomPhotometric(BaseTransform):
    """Exposure/blur/noise augmentation with a target-mean gamma.

    Args:
        p: Probability of applying the whole photometric block.
        target: ``(lo, hi)`` target mean brightness in 8-bit gray levels.
        blur_p: Probability of a linear motion blur after the exposure change.
        noise_p: Probability of additive Gaussian sensor noise.
    """

    def __init__(
        self,
        p: float = 0.5,
        target: tuple[float, float] = (5.0, 60.0),
        blur_p: float = 0.3,
        noise_p: float = 0.2,
    ) -> None:
        super().__init__()
        self.p = float(p)
        self.target = as_pair(target, (5.0, 60.0))
        self.blur_p = float(blur_p)
        self.noise_p = float(noise_p)

    def __call__(self, labels: dict) -> dict:
        """Apply the augmentation to ``labels['img']`` in place and return labels."""
        if random.random() >= self.p:
            return labels
        img = labels.get("img")
        if not isinstance(img, np.ndarray) or img.ndim != 3 or img.dtype != np.uint8:
            return labels

        img = self._gamma_to_target(img)
        img = self._brightness_contrast(img)
        if random.random() < self.blur_p:
            img = self._motion_blur(img)
        if random.random() < self.noise_p:
            img = self._noise(img)
        labels["img"] = np.ascontiguousarray(img)
        return labels

    def _gamma_to_target(self, img: np.ndarray) -> np.ndarray:
        """Map the image mean into ``target`` with a single gamma LUT."""
        mean = float(img.mean())
        # A nearly black or saturated frame carries no usable exposure estimate.
        if mean < 1.0 or mean > 250.0:
            return img
        target = random.uniform(*self.target)
        target = min(max(target, 1.0), 254.0)
        gamma = math.log(target / 255.0) / math.log(mean / 255.0)
        gamma = min(max(gamma, 0.35), 3.0)
        lut = (np.arange(256, dtype=np.float32) / 255.0) ** gamma * 255.0
        return cv2.LUT(img, np.clip(lut, 0.0, 255.0).astype(np.uint8))

    @staticmethod
    def _brightness_contrast(img: np.ndarray) -> np.ndarray:
        """Small gain/offset jitter; keeps the target-mean exposure distribution."""
        alpha = random.uniform(0.80, 1.20)
        beta = random.uniform(-12.0, 12.0)
        return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    @staticmethod
    def _motion_blur(img: np.ndarray) -> np.ndarray:
        """Linear motion blur at a random angle and 3/5/7/9 px kernel."""
        size = random.choice((3, 5, 7, 9))
        kernel = np.zeros((size, size), dtype=np.float32)
        kernel[size // 2, :] = 1.0
        center = (size / 2.0 - 0.5, size / 2.0 - 0.5)
        matrix = cv2.getRotationMatrix2D(center, random.uniform(0.0, 180.0), 1.0)
        kernel = cv2.warpAffine(kernel, matrix, (size, size))
        total = float(kernel.sum())
        if total <= 0.0:
            return img
        return cv2.filter2D(img, -1, kernel / total)

    @staticmethod
    def _noise(img: np.ndarray) -> np.ndarray:
        """Additive Gaussian noise in 8-bit gray levels."""
        sigma = random.uniform(1.5, 7.0)
        noise = np.random.normal(0.0, sigma, img.shape).astype(np.float32)
        return np.clip(img.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)


def install_photometric(
    dataset,
    p: float,
    target=(5.0, 60.0),
    blur_p: float = 0.3,
    noise_p: float = 0.2,
) -> int:
    """Insert a :class:`RandomPhotometric` before the final ``Format`` transform.

    ``Format`` is the tensor conversion appended by ``YOLODataset``; inserting
    before it keeps the image a uint8 HWC array (``cv2`` LUT/filter2D) and puts
    the augmentation after mosaic/affine/flip, i.e. on the actual training
    canvas.  A bare ``LetterBox`` pipeline is handled as a fallback.

    Returns 1 when a transform was inserted, 0 when the dataset has no
    ``Compose`` pipeline or already carries the transform.
    """
    transforms = getattr(dataset, "transforms", None)
    if not isinstance(transforms, Compose):
        return 0
    if any(isinstance(t, RandomPhotometric) for t in transforms.transforms):
        return 0
    index = next(
        (i for i, t in enumerate(transforms.transforms) if isinstance(t, Format)),
        None,
    )
    if index is None:
        index = next(
            (i for i, t in enumerate(transforms.transforms) if isinstance(t, LetterBox)),
            len(transforms.transforms),
        )
    transforms.insert(index, RandomPhotometric(p=p, target=target, blur_p=blur_p, noise_p=noise_p))
    return 1
