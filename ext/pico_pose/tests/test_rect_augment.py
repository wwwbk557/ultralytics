"""Tests for rectangular (480x640) training augmentation.

Covers the parts that differ from upstream:
  * the stock square ``Mosaic`` is swapped for ``RectMosaic`` when mosaic > 0;
  * a mosaic sample still ends up as exactly (3, H, W) with labels in range;
  * ``close_mosaic`` rebuilds transforms without crashing and without mosaic;
  * keypoint/bbox normalisation survives the mosaic path.

These tests only build the dataset and run the transform pipeline; they do not
train.  They need ``datasets/schemeC_split/data.yaml`` (skipped otherwise).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import copy

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from ultralytics.cfg import DEFAULT_CFG_DICT  # noqa: E402
from ultralytics.data.augment import Mosaic  # noqa: E402
from ultralytics.data.build import build_yolo_dataset  # noqa: E402
from ultralytics.data.utils import check_det_dataset  # noqa: E402

from ext.pico_pose.rect_augment import RectMosaic, install_rect_mosaic  # noqa: E402
from ext.pico_pose.rect_train import install_rect_patches  # noqa: E402

DATA = ROOT / "datasets/schemeC_split/data.yaml"
HW = (480, 640)
AUG = dict(
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=5.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,
    bgr=0.0,
    mosaic=1.0,
    mixup=0.0,
    cutmix=0.0,
    copy_paste=0.0,
    erasing=0.4,
)

pytestmark = pytest.mark.skipif(not DATA.is_file(), reason=f"missing {DATA}")


def _build(mosaic: float):
    """Build the train dataset exactly like PicoRectTrainer.build_dataset does."""
    install_rect_patches()
    data = check_det_dataset(str(DATA))
    cfg = SimpleNamespace(
        **{
            **DEFAULT_CFG_DICT,
            "task": "pose",
            "imgsz": HW,
            "rect": False,
            "cache": False,
            **{**AUG, "mosaic": mosaic},
        }
    )
    ds = build_yolo_dataset(cfg, data["train"], batch=4, data=data, mode="train", rect=False, stride=32)
    if mosaic > 0:
        assert install_rect_mosaic(ds, HW) == 1
    return ds, data, cfg


def _check_sample(sample: dict) -> int:
    img = sample["img"]
    assert tuple(img.shape) == (3, *HW), f"bad image shape {tuple(img.shape)}"
    kpts = sample["keypoints"]
    boxes = sample["bboxes"]
    n = len(sample["cls"])
    if n:
        assert np.isfinite(kpts).all() and np.isfinite(boxes).all()
        assert kpts[..., :2].min() >= 0.0 and kpts[..., :2].max() <= 1.0, "keypoints outside the image"
        assert kpts[..., 2].min() >= 0.0 and kpts[..., 2].max() <= 1.0
        assert boxes.min() >= 0.0 and boxes.max() <= 1.0, "boxes outside the image"
        assert sample["cls"].min() >= 0 and sample["cls"].max() < 14
    return n


def test_mosaic_is_installed_and_is_rect():
    ds, _, _ = _build(1.0)
    mosaics = []

    def walk(obj):
        for t in getattr(obj, "transforms", []):
            if isinstance(t, Mosaic):
                mosaics.append(t)
            else:
                walk(t)

    walk(ds.transforms)
    assert mosaics and all(isinstance(m, RectMosaic) for m in mosaics)
    assert mosaics[0].rect_hw == HW
    assert mosaics[0].border == (-HW[0] // 2, -HW[1] // 2)


def test_mosaic_sample_shape_and_labels():
    ds, _, _ = _build(1.0)
    np.random.seed(0)
    counts = [_check_sample(ds[i]) for i in range(24)]
    # mosaic crops can legitimately drop every instance (upstream does the same),
    # so require objects in most samples rather than all of them.
    assert sum(c > 0 for c in counts) >= 12, f"too many empty mosaic samples: {counts}"


def test_mosaic_stage_canvas_and_border():
    ds, _, cfg = _build(1.0)
    mosaic = ds.transforms.transforms[0].transforms[0]
    assert isinstance(mosaic, RectMosaic)
    labels = ds.get_image_and_label(0)
    m = copy.deepcopy(mosaic)
    m.p = 1.0
    np.random.seed(0)
    out = m(copy.deepcopy(labels))
    h, w = HW
    assert out["img"].shape[:2] == (2 * h, 2 * w), "mosaic canvas must be 2H x 2W"
    assert out["mosaic_border"] == (-h // 2, -w // 2), "affine crop must land on exactly H,W"
    assert len(out["cls"]) >= 1
    boxes = out["instances"].bboxes
    assert boxes[:, [0, 2]].max() <= 2 * w and boxes[:, [1, 3]].max() <= 2 * h


def test_close_mosaic_rebuilds_transforms():
    ds, _, cfg = _build(1.0)
    np.random.seed(1)
    for i in range(4):
        _check_sample(ds[i])
    cfg.mosaic = 0.0  # the trainer sets this for the last `close_mosaic` epochs
    ds.close_mosaic(cfg)
    np.random.seed(1)
    for i in range(4):
        sample = ds[i]
        assert tuple(sample["img"].shape) == (3, *HW)
