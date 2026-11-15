"""Tests for the train-time keypoint/box sync."""

from __future__ import annotations

import numpy as np

from ext.pico_pose.box_sync import KeypointBoxSync, install_keypoint_box_sync
from ultralytics.utils.instance import Instances

KPT = np.array([[[10.0, 20.0, 1.0], [50.0, 24.0, 1.0], [48.0, 44.0, 1.0], [12.0, 40.0, 1.0]]], np.float32)


def _labels(kpts, boxes, normalized=False, shape=(64, 64)):
    inst = Instances(boxes, np.zeros((0, 1000, 2)), kpts, bbox_format="xyxy", normalized=normalized)
    return {"img": np.zeros((*shape, 3), np.uint8), "instances": inst}


def test_sync_sets_box_to_keypoint_aabb():
    labels = _labels(KPT, np.array([[8.0, 18.0, 52.0, 46.0]], np.float32))
    out = KeypointBoxSync()(labels)
    assert out is labels
    assert np.allclose(labels["instances"].bboxes, [[10.0, 20.0, 50.0, 44.0]])


def test_truncated_instance_keeps_upstream_box():
    kpts = KPT.copy()
    kpts[0, 2, 2] = 0.0  # corner clipped by the affine border
    boxes = np.array([[8.0, 18.0, 52.0, 46.0]], np.float32)
    labels = _labels(kpts, boxes)
    KeypointBoxSync()(labels)
    assert np.allclose(labels["instances"].bboxes, boxes)


def test_normalized_instances_are_denormalized():
    kpts = KPT / np.array([64.0, 64.0, 1.0], np.float32)
    boxes = np.array([[8.0 / 64, 18.0 / 64, 52.0 / 64, 46.0 / 64]], np.float32)
    labels = _labels(kpts, boxes, normalized=True)
    KeypointBoxSync()(labels)
    inst = labels["instances"]
    assert not inst.normalized
    assert np.allclose(inst.bboxes, [[10.0, 20.0, 50.0, 44.0]], atol=1e-4)


def test_noop_without_keypoints():
    labels = _labels(KPT, np.array([[8.0, 18.0, 52.0, 46.0]], np.float32))
    labels["instances"].keypoints = None
    KeypointBoxSync()(labels)
    assert np.allclose(labels["instances"].bboxes, [[8.0, 18.0, 52.0, 46.0]])


def test_install_before_format_is_idempotent():
    from ultralytics.data.augment import Compose, Format, LetterBox

    class Dataset:
        def __init__(self):
            self.transforms = Compose(
                [
                    LetterBox((480, 640)),
                    Format(bbox_format="xywh", normalize=True, return_keypoint=True, batch_idx=True),
                ]
            )

    dataset = Dataset()
    assert install_keypoint_box_sync(dataset) == 1
    assert isinstance(dataset.transforms.transforms[-2], KeypointBoxSync)
    assert install_keypoint_box_sync(dataset) == 0
