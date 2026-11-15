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
from copy import copy

from ultralytics.cfg import DEFAULT_CFG
from ultralytics.data.augment import LetterBox, Mosaic
from ultralytics.data.base import BaseDataset
from ultralytics.data.build import build_yolo_dataset
from ultralytics.models.yolo.pose import PoseTrainer
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import unwrap_model

from .box_sync import install_keypoint_box_sync
from .env_decouple import install_env_decouple
from .photometric import install_photometric
from .plate_scale import install_plate_scale
from .rect_augment import install_rect_mosaic

import ultralytics.engine.trainer as _trainer_mod
import ultralytics.engine.validator as _validator_mod

_ORIG_CHECK = _trainer_mod.check_imgsz
_ORIG_LETTERBOX = LetterBox.__init__
_ORIG_MOSAIC = Mosaic.__init__
_ORIG_LOAD_IMAGE = BaseDataset.load_image
_ORIG_WARMUP = AutoBackend.warmup
_INSTALLED = False

# Every train-time augmentation hyperparameter; the final close_mosaic epochs zero all of them.
_AUG_KEYS = (
    "mosaic",
    "mixup",
    "cutmix",
    "copy_paste",
    "degrees",
    "translate",
    "scale",
    "shear",
    "perspective",
    "fliplr",
    "flipud",
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "erasing",
    "photometric_p",
    "env_p",
    "plate_scale_p",
)


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
    Train-only extras are opt-in: ``photometric_p``, ``env_p`` (environment
    decorrelation: relocate / context / solo / scene) and ``plate_scale_p``
    (per-plate target scale jitter).  ``degrees > 0`` also installs the
    keypoint-box sync so rotation keeps ``box == 4-kpt AABB``.
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
        if mode == "train":
            p = float(getattr(self.args, "photometric_p", 0.0) or 0.0)
            if p > 0:
                n = install_photometric(
                    dataset,
                    p=p,
                    target=getattr(self.args, "photometric_target", (5.0, 60.0)),
                    blur_p=float(getattr(self.args, "photometric_blur_p", 0.3) or 0.0),
                    noise_p=float(getattr(self.args, "photometric_noise_p", 0.2) or 0.0),
                )
                LOGGER.info(
                    f"photometric augmentation: installed {n} transform(s), "
                    f"p={p:.2f} target={getattr(self.args, 'photometric_target', (5.0, 60.0))}"
                )
            env_p = float(getattr(self.args, "env_p", 0.0) or 0.0)
            if env_p > 0:
                n = install_env_decouple(
                    dataset,
                    hw=self.args.imgsz,
                    p=env_p,
                    relocate_p=float(getattr(self.args, "env_relocate_p", 0.35) or 0.0),
                    context_p=float(getattr(self.args, "env_context_p", 0.5) or 0.0),
                    solo_p=float(getattr(self.args, "env_solo_p", 0.25) or 0.0),
                    scene_p=float(getattr(self.args, "env_scene_p", 0.35) or 0.0),
                    flat_p=float(getattr(self.args, "env_flat_p", 0.4) or 0.0),
                    donors=int(getattr(self.args, "env_donors", 32) or 0),
                    context_scale=getattr(self.args, "env_context_scale", (1.6, 2.6)),
                )
                LOGGER.info(
                    f"environment decorrelation: installed {n} transform(s), p={env_p:.2f} "
                    f"relocate={getattr(self.args, 'env_relocate_p', 0.35)} "
                    f"context={getattr(self.args, 'env_context_p', 0.5)} "
                    f"solo={getattr(self.args, 'env_solo_p', 0.25)} "
                    f"scene={getattr(self.args, 'env_scene_p', 0.35)}"
                )
            plate_p = float(getattr(self.args, "plate_scale_p", 0.0) or 0.0)
            if plate_p > 0:
                n = install_plate_scale(
                    dataset,
                    p=plate_p,
                    scale=getattr(self.args, "plate_scale_range", (0.85, 1.15)),
                    max_plates=int(getattr(self.args, "plate_scale_max", 2) or 1),
                    min_px=float(getattr(self.args, "plate_scale_min_px", 16.0) or 0.0),
                )
                LOGGER.info(
                    f"plate scale jitter: installed {n} transform(s), p={plate_p:.2f} "
                    f"range={getattr(self.args, 'plate_scale_range', (0.85, 1.15))} "
                    f"max_plates={getattr(self.args, 'plate_scale_max', 2)} "
                    f"min_px={getattr(self.args, 'plate_scale_min_px', 16.0)}"
                )
            degrees = float(getattr(self.args, "degrees", 0.0) or 0.0)
            if degrees > 0:
                n = install_keypoint_box_sync(dataset)
                LOGGER.info(f"keypoint box sync: installed {n} transform(s), degrees={degrees:.1f}")
        return dataset

    def _close_dataloader_mosaic(self):
        """Final ``close_mosaic`` epochs train unaugmented; keep the box == 4-kpt AABB rule.

        Upstream only zeros mosaic/mixup/cutmix/copy_paste and rebuilds the
        transforms, which silently drops the custom env/plate-scale transforms
        but leaves rotation, translation, scaling and HSV active.  Zero those
        too and re-install the keypoint-box sync so labels stay deployment-shaped.
        """
        dataset = self.train_loader.dataset
        hyp = copy(self.args)
        for key in _AUG_KEYS:
            if hasattr(hyp, key):
                setattr(hyp, key, 0.0)
        if hasattr(dataset, "mosaic"):
            dataset.mosaic = False
        if hasattr(dataset, "close_mosaic"):
            LOGGER.info("closing mosaic and every train augmentation for the final epochs")
            dataset.close_mosaic(hyp=hyp)
            install_keypoint_box_sync(dataset)

    def setup_model(self):
        """Load the model and push the Pico eval flags from the run args onto the head.

        The flags must live on the head *before* ``ModelEMA`` copies the model,
        so the EMA used for validation inherits them (and a resumed EMA too).
        """
        ckpt = super().setup_model()
        self._apply_pico_eval_flags(self.model)
        # A checkpoint may carry the old criterion with the old 4-slot loss and
        # the old args namespace.  Drop it so the first loss call rebuilds it
        # against self.args (new colour/number/consistency weights included).
        inner = unwrap_model(self.model)
        if getattr(inner, "criterion", None) is not None:
            inner.criterion = None
        return ckpt

    def _apply_pico_eval_flags(self, model) -> None:
        """Align validation with the exported graph (corner boxes + objectness bias)."""
        head = getattr(unwrap_model(model), "model", None)
        head = head[-1] if head is not None else None
        if head is None or not hasattr(head, "deploy_boxes"):
            return
        head.deploy_boxes = bool(getattr(self.args, "deploy_boxes", False))
        head.obj_logit_bias = float(getattr(self.args, "obj_logit_bias", 0.0))
        # The checkpoint pickles the old 4-tuple; the loss now returns 5 extra
        # slots, and PoseTrainer.get_validator builds the CSV header from this.
        head.extra_loss_names = ("color_loss", "number_loss", "quality_loss", "geom_loss", "consistency_loss")
        LOGGER.info(
            f"pico eval parity: deploy_boxes={head.deploy_boxes} "
            f"obj_logit_bias={head.obj_logit_bias:+.3f} (sigmoid(z - bias) on channel 8)"
        )

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
