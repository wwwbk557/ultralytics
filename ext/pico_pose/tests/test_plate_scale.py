"""Tests that plate scale jitter only ever rescales the plate it moved."""

from __future__ import annotations

import random

import cv2
import numpy as np

from ext.pico_pose.env_decouple import _aabb as _kp_aabb
from ext.pico_pose.env_decouple import _expand_quad, _relocate_margins
from ext.pico_pose.plate_scale import RandomPlateScale
from ultralytics.utils.instance import Instances


def _plate(cx: float, cy: float, half: float) -> np.ndarray:
    return np.array(
        [[cx - half, cy - half], [cx + half, cy - half], [cx + half, cy + half], [cx - half, cy + half]], np.float32
    )


def _labels(centres, half=12.0, shape=(480, 640)):
    kpts = np.stack([_plate(cx, cy, half) for cx, cy in centres])
    kpts = np.concatenate([kpts, np.ones((*kpts.shape[:2], 1), np.float32)], -1)
    boxes = np.stack([kpts[..., 0].min(1), kpts[..., 1].min(1), kpts[..., 0].max(1), kpts[..., 1].max(1)], 1)
    inst = Instances(boxes.astype(np.float32), np.zeros((0, 1000, 2)), kpts, bbox_format="xyxy", normalized=False)
    img = np.random.default_rng(0).integers(0, 255, (*shape, 3), dtype=np.uint8)
    return {"img": img, "instances": inst, "cls": np.zeros(len(centres), np.int64)}


def _aabb(kpts: np.ndarray) -> np.ndarray:
    return np.array([kpts[:, 0].min(), kpts[:, 1].min(), kpts[:, 0].max(), kpts[:, 1].max()], np.float32)


def _moved(before, after):
    return [i for i in range(len(before)) if not np.allclose(before[i], after[i])]


def test_paste_stops_before_a_neighbours_digit_overhang(monkeypatch):
    """The old kp-AABB test allowed a resize that painted over the neighbour's digit."""
    labels = _labels([(320.0, 240.0), (320.0, 190.0)], half=20.0)
    before_k = labels["instances"].keypoints.copy()
    before_img = labels["img"].copy()
    monkeypatch.setattr(random, "shuffle", lambda x: None)  # target plate first
    monkeypatch.setattr(random, "uniform", lambda a, b: b)  # always the largest factor
    RandomPlateScale(p=1.0, scale=(1.3, 1.3), max_plates=1, min_px=0.0)(labels)
    assert np.allclose(labels["instances"].keypoints, before_k)
    assert np.array_equal(labels["img"], before_img)  # rejected before the inpaint, not after


def test_resize_never_touches_a_neighbours_pixels():
    """Neither the paste nor the inpaint hole may eat into another plate's pixels."""
    moves = 0
    for trial in range(80):
        rng = random.Random(trial)
        centres = [(rng.uniform(120, 520), rng.uniform(100, 380)), (rng.uniform(120, 520), rng.uniform(100, 380))]
        half = rng.uniform(15, 40)
        labels = _labels(centres, half=half)
        labels["img"][:] = 0
        masks = []
        for i, k in enumerate(labels["instances"].keypoints):
            gu, gv = _relocate_margins(_kp_aabb(k), [])
            poly = np.round(_expand_quad(k, gu, gv)).astype(np.int32)
            m = np.zeros(labels["img"].shape[:2], np.uint8)
            cv2.fillPoly(m, [poly.reshape(-1, 1, 2)], 255)
            labels["img"][m > 0] = (0, 200, 0) if i == 0 else (0, 255, 255)
            masks.append(m > 0)
        before_k = labels["instances"].keypoints.copy()
        before_img = labels["img"].copy()
        random.seed(trial)
        RandomPlateScale(p=1.0, scale=(0.8, 1.25), max_plates=1, min_px=0.0)(labels)
        moved = set(_moved(before_k, labels["instances"].keypoints))
        moves += len(moved)
        for j in (0, 1):
            if j not in moved:
                assert np.array_equal(labels["img"][masks[j]], before_img[masks[j]])
    assert moves > 10


def test_scale_up_grows_aabb_and_keeps_centre():
    labels = _labels([(120.0, 200.0), (420.0, 200.0)], half=20.0)
    before = [_aabb(k) for k in labels["instances"].keypoints]
    RandomPlateScale(p=1.0, scale=(1.3, 1.3), max_plates=1)(labels)
    after = [_aabb(k) for k in labels["instances"].keypoints]
    grown = [i for i in (0, 1) if abs(after[i][2] - after[i][0]) > abs(before[i][2] - before[i][0]) + 1]
    assert len(grown) == 1
    i = grown[0]
    assert np.allclose((before[i][:2] + before[i][2:]) / 2, (after[i][:2] + after[i][2:]) / 2, atol=1.5)
    ratio = (after[i][2] - after[i][0]) / (before[i][2] - before[i][0])
    assert 1.2 < ratio < 1.4
    other = 1 - i
    assert np.allclose(before[other], after[other])
    assert np.allclose(labels["instances"].bboxes, np.stack(after), atol=1e-3)


def test_scale_down_shrinks_aabb():
    labels = _labels([(300.0, 240.0)], half=40.0)
    before = _aabb(labels["instances"].keypoints[0])
    RandomPlateScale(p=1.0, scale=(0.8, 0.8), max_plates=1)(labels)
    after = _aabb(labels["instances"].keypoints[0])
    ratio = (after[2] - after[0]) / (before[2] - before[0])
    assert 0.7 < ratio < 0.9
    assert np.allclose(labels["instances"].bboxes[0], after, atol=1e-3)


def test_small_plate_is_never_shrunk():
    labels = _labels([(300.0, 240.0)], half=6.0)  # 12 px long side < min_px
    before = labels["instances"].keypoints.copy()
    RandomPlateScale(p=1.0, scale=(0.8, 0.8), max_plates=1, min_px=16.0)(labels)
    assert np.allclose(labels["instances"].keypoints, before)

    labels = _labels([(300.0, 240.0)], half=6.0)
    RandomPlateScale(p=1.0, scale=(0.8, 1.2), max_plates=1, min_px=16.0)(labels)
    after = _aabb(labels["instances"].keypoints[0])
    assert after[2] - after[0] > 12.0  # only the enlarge half of the range is used


def test_rejected_resize_leaves_pixels_untouched():
    labels = _labels([(24.0, 240.0)], half=20.0)  # enlarged patch would leave the canvas
    before_img = labels["img"].copy()
    before_k = labels["instances"].keypoints.copy()
    random.seed(0)
    RandomPlateScale(p=1.0, scale=(1.15, 1.15), max_plates=1, min_px=0.0)(labels)
    assert np.allclose(labels["instances"].keypoints, before_k)
    assert np.array_equal(labels["img"], before_img)


def test_noop_when_disabled_or_without_instances():
    labels = _labels([(120.0, 200.0)])
    before = labels["instances"].keypoints.copy()
    RandomPlateScale(p=0.0)(labels)
    assert np.allclose(labels["instances"].keypoints, before)
    labels["instances"].keypoints = None
    RandomPlateScale(p=1.0)(labels)
    assert labels["instances"].keypoints is None
