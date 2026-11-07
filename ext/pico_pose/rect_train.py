"""Rectangular 480x640 training support for the compact Pico pose head.

Ultralytics' BaseTrainer/BaseValidator normally force ``imgsz`` to one square
integer.  This module keeps the *external* H,W pair for Pico training:

    imgsz=(480, 640)  ->  every train/val batch is (B, 3, 480, 640)

The implementation is deliberately scoped to training:
  * check_imgsz in trainer/validator preserves a two-int list;
  * LetterBox receives (H, W), not ((H,W),(H,W));
  * BaseDataset.load_image pre-resizes with max(H,W), then LetterBox does H,W;
  * mosaic/cutmix/mixup stay usable: the stock square ``Mosaic`` is swapped for
    ``RectMosaic`` (see ``rect_augment.py``) so the 2x2 canvas is 2H x 2W and the
    affine stage crops straight to (H, W).

No upstream file is modified; the patches are installed when
``PicoRectTrainer`` is constructed.
"""
from __future__ import annotations

import math

from ultralytics.cfg import DEFAULT_CFG
from ultralytics.data.augment import LetterBox, Mosaic
from ultralytics.data.base import BaseDataset
from ultralytics.data.build import build_yolo_dataset
from ultralytics.models.yolo.pose import PoseTrainer
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import unwrap_model

from .rect_augment import install_rect_mosaic

import ultralytics.engine.trainer as _trainer_mod
import ultralytics.engine.validator as _validator_mod

_ORIG_CHECK = _trainer_mod.check_imgsz
_ORIG_LETTERBOX = LetterBox.__init__
_ORIG_MOSAIC = Mosaic.__init__
_ORIG_LOAD_IMAGE = BaseDataset.load_image
_ORIG_WARMUP = AutoBackend.warmup
_INSTALLED = False


def _check_imgsz_rect(imgsz, stride=32, min_dim=1, max_dim=2, floor=0):
    """Preserve an explicit (H, W) for train/val instead of max(H, W)."""
    if max_dim == 1 and isinstance(imgsz, (list, tuple)) and len(imgsz) == 2:
        s = int(stride.max()) if hasattr(stride, "max") else int(stride)
        return [max(math.ceil(int(x) / s) * s, floor) for x in imgsz]
    return _ORIG_CHECK(imgsz, stride=stride, min_dim=min_dim, max_dim=max_dim, floor=floor)


def _letterbox_init(self, new_shape=(640, 640), *args, **kwargs):
    if (
        isinstance(new_shape, (list, tuple))
        and len(new_shape) == 2
        and isinstance(new_shape[0], (list, tuple))
    ):
        new_shape = tuple(new_shape[0])
    _ORIG_LETTERBOX(self, new_shape, *args, **kwargs)


def _mosaic_init(self, dataset, imgsz=640, p=1.0, n=4):
    if isinstance(imgsz, (list, tuple)) and len(imgsz) == 2:
        h, w = int(imgsz[0]), int(imgsz[1])
        _ORIG_MOSAIC(self, dataset, max(h, w), p, n)
        self.border = (-h // 2, -w // 2)
        self.rect_hw = (h, w)
    else:
        _ORIG_MOSAIC(self, dataset, imgsz, p, n)


def _load_image(self, index, rect_mode=True):
    imgsz = self.imgsz
    if isinstance(imgsz, (list, tuple)):
        # load at the long side, then the fixed LetterBox produces H,W.
        self.imgsz = max(int(x) for x in imgsz)
    try:
        return _ORIG_LOAD_IMAGE(self, index, rect_mode=rect_mode)
    finally:
        self.imgsz = imgsz  # never leak the scalar back into the dataset


def _warmup(self, imgsz=(1, 3, 640, 640)):
    if isinstance(imgsz, (list, tuple)) and len(imgsz) == 4 and isinstance(imgsz[2], (list, tuple)):
        imgsz = (imgsz[0], imgsz[1], int(imgsz[2][0]), int(imgsz[2][1]))
    return _ORIG_WARMUP(self, imgsz)


def install_rect_patches() -> None:
    """Install the rectangular train/val patches once."""
    global _INSTALLED
    if _INSTALLED:
        return
    _trainer_mod.check_imgsz = _check_imgsz_rect
    _validator_mod.check_imgsz = _check_imgsz_rect
    LetterBox.__init__ = _letterbox_init
    Mosaic.__init__ = _mosaic_init
    BaseDataset.load_image = _load_image
    AutoBackend.warmup = _warmup
    _INSTALLED = True


class PicoRectTrainer(PoseTrainer):
    """PoseTrainer that trains/validates at a fixed rectangular H,W.

    ``overrides['imgsz']`` must be a two-int tuple/list, e.g. ``(480, 640)``.
    Batches are always (B, 3, H, W); ``rect`` batching stays off because the
    shape is already fixed.  Mosaic/mixup/cutmix are supported (mosaic is
    swapped for ``RectMosaic``), copy_paste is a no-op without segments.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        overrides = dict(overrides or {})
        imgsz = overrides.get("imgsz")
        if isinstance(imgsz, (list, tuple)) and len(imgsz) == 2:
            install_rect_patches()
            overrides["imgsz"] = (int(imgsz[0]), int(imgsz[1]))
            overrides["rect"] = False  # fixed H,W batches, not adaptive rect batches
            if float(overrides.get("mosaic", 0) or 0) > 0:
                LOGGER.info("rectangular training: mosaic -> RectMosaic (2H x 2W canvas, cropped to H,W)")
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        # Force rect=False for both train and val: fixed H,W, not adaptive batches.
        dataset = build_yolo_dataset(
            self.args,
            img_path,
            batch,
            self.data,
            mode=mode,
            rect=False,
            stride=gs,
        )
        if mode == "train" and float(getattr(self.args, "mosaic", 0) or 0) > 0:
            n = install_rect_mosaic(dataset, self.args.imgsz)
            LOGGER.info(f"rectangular training: installed {n} RectMosaic for imgsz={tuple(int(x) for x in self.args.imgsz)}")
        return dataset

    def validate(self):
        """Run the stock validation plus the objectness/TAL monitors."""
        from .monitor import ObjectnessMonitor

        monitor = ObjectnessMonitor(self)
        try:
            monitor.attach()
        except Exception as e:  # never break training because of diagnostics
            LOGGER.warning(f"objectness monitor disabled: {e}")
            monitor = None
        try:
            metrics, fitness = super().validate()
        finally:
            if monitor is not None:
                monitor.detach()
        if metrics is None:
            return None, None
        metrics = dict(metrics)
        if monitor is not None:
            metrics.update(monitor.result())
        metrics.update(self._pop_loss_monitors())
        return metrics, fitness

    def _pop_loss_monitors(self) -> dict:
        """Per-epoch TAL/attribute monitors accumulated by the criterion."""
        try:
            criterion = unwrap_model(self.model).criterion
        except AttributeError:
            return {}
        return criterion.pop_monitors() if hasattr(criterion, "pop_monitors") else {}
