"""Tests that env decoupling never keeps a label on top of clipped plate pixels."""

from __future__ import annotations

import random

import cv2
import numpy as np

from ext.pico_pose.env_decouple import RandomEnvDecouple
from ultralytics.utils.instance import Instances


def _plate(cx: float, cy: float, half: float) -> np.ndarray:
    return np.array(
        [[cx - half, cy - half], [cx + half, cy - half], [cx + half, cy + half], [cx - half, cy + half]], np.float32
    )


def _labels(centres, half: float = 20.0, shape=(480, 640)):
    """Green plate quad plus a red digit band above/below it (20% of the long side)."""
    img = np.zeros((*shape, 3), np.uint8)
    kpts, digit = [], np.zeros(shape, bool)
    for cx, cy in centres:
        p = _plate(cx, cy, half)
        kpts.append(np.concatenate([p, np.ones((4, 1), np.float32)], -1))
        poly = np.round(p).astype(np.int32)
        mask = np.zeros(shape, np.uint8)
        cv2.fillPoly(mask, [poly.reshape(-1, 1, 2)], 255)
        img[mask > 0] = (0, 200, 0)
        gv = int(round(0.2 * 2 * half))
        x0, x1 = int(cx - 0.4 * half), int(cx + 0.4 * half)
        for y0, y1 in ((int(cy - half) - gv, int(cy - half)), (int(cy + half), int(cy + half) + gv)):
            img[max(0, y0) : y1, x0:x1] = (0, 0, 255)
            digit[max(0, y0) : y1, x0:x1] = True
    kpts = np.stack(kpts)
    boxes = np.stack([kpts[..., 0].min(1), kpts[..., 1].min(1), kpts[..., 0].max(1), kpts[..., 1].max(1)], 1)
    inst = Instances(boxes.astype(np.float32), np.zeros((0, 1000, 2)), kpts, bbox_format="xyxy", normalized=False)
    return {"img": img, "instances": inst, "cls": np.zeros(len(centres), np.int64)}, digit


def _keep_first(monkeypatch):
    monkeypatch.setattr(random, "randrange", lambda n: 0)


def test_scene_pastes_the_digit_overhang_back():
    labels, digit = _labels([(320.0, 240.0)])
    before = labels["img"].copy()
    RandomEnvDecouple(p=1.0, relocate_p=0.0, context_p=0.0, solo_p=0.0, scene_p=1.0, flat_p=1.0)(labels)
    assert np.array_equal(labels["img"][digit], before[digit])
    assert not np.array_equal(labels["img"], before)  # the background did change


def test_context_pastes_the_digit_overhang_back():
    labels, digit = _labels([(320.0, 240.0)])
    before = labels["img"].copy()
    RandomEnvDecouple(
        p=1.0, relocate_p=0.0, context_p=1.0, solo_p=0.0, scene_p=0.0, context_scale=(0.5, 0.6)
    )(labels)
    assert np.array_equal(labels["img"][digit], before[digit])


def test_solo_erases_the_digit_overhang_of_the_removed_plate(monkeypatch):
    labels, digit = _labels([(160.0, 240.0), (480.0, 240.0)])
    before = labels["img"].copy()
    kept = labels["instances"].keypoints[0].copy()
    _keep_first(monkeypatch)
    RandomEnvDecouple(p=1.0, relocate_p=0.0, context_p=0.0, solo_p=1.0, scene_p=0.0)(labels)
    assert len(labels["instances"]) == 1
    assert np.allclose(labels["instances"].keypoints[0], kept)
    left = digit.copy()
    left[:, 300:] = False  # only the kept plate's band
    assert np.array_equal(labels["img"][left], before[left])
    gone = digit.copy()
    gone[:, :300] = False  # only the erased plate's band
    assert labels["img"][gone].max() < 30  # no unlabelled digit fragment left behind
