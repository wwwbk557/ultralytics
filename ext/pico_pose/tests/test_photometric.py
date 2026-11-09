"""Tests for the target-mean photometric augmentation and role mask."""

from __future__ import annotations

import numpy as np
import pytest

from ext.pico_pose.photometric import RandomPhotometric, as_pair, install_photometric

torch = pytest.importorskip("torch")


def test_as_pair_parses_cli_and_yaml_forms():
    assert as_pair("5,60", (1.0, 2.0)) == (5.0, 60.0)
    assert as_pair("5x60", (1.0, 2.0)) == (5.0, 60.0)
    assert as_pair([5, 60], (1.0, 2.0)) == (5.0, 60.0)
    assert as_pair(12, (1.0, 2.0)) == (12.0, 12.0)
    assert as_pair("", (1.0, 2.0)) == (1.0, 2.0)


def test_gamma_maps_mean_to_target():
    image = np.full((64, 64, 3), 80, dtype=np.uint8)
    aug = RandomPhotometric(p=1.0, target=(30.0, 30.0), blur_p=0.0, noise_p=0.0)
    out = aug._gamma_to_target(image)
    assert abs(float(out.mean()) - 30.0) < 1.5

    aug = RandomPhotometric(p=1.0, target=(10.0, 10.0), blur_p=0.0, noise_p=0.0)
    out = aug._gamma_to_target(image)
    assert abs(float(out.mean()) - 10.0) < 1.5


def test_transform_is_a_noop_at_p_zero():
    image = np.full((32, 32, 3), 40, dtype=np.uint8)
    aug = RandomPhotometric(p=0.0)
    labels = {"img": image}
    assert aug(labels) is labels
    assert labels["img"] is image


def test_motion_blur_keeps_shape_and_dtype():
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    image[:, 32] = 255
    out = RandomPhotometric._motion_blur(image)
    assert out.shape == image.shape and out.dtype == np.uint8
    assert out.max() > 0


def test_install_photometric_goes_before_format():
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
    assert install_photometric(dataset, p=0.5) == 1
    assert isinstance(dataset.transforms.transforms[-2], RandomPhotometric)
    assert install_photometric(dataset, p=0.5) == 0  # idempotent


def test_install_photometric_letterbox_fallback():
    from ultralytics.data.augment import Compose, LetterBox

    class Dataset:
        def __init__(self):
            self.transforms = Compose([LetterBox((480, 640))])

    dataset = Dataset()
    assert install_photometric(dataset, p=0.5) == 1
    assert isinstance(dataset.transforms.transforms[0], RandomPhotometric)


def test_role_mask_weight_ramps_area_and_obliquity():
    from ext.pico_pose.loss import role_mask_weight

    frontal = torch.tensor([[[0.0, 0.0], [40.0, 0.0], [40.0, 30.0], [0.0, 30.0]]])
    tiny = torch.tensor([[[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]]])
    edge_on = torch.tensor([[[0.0, 0.0], [40.0, 0.0], [40.0, 4.0], [0.0, 4.0]]])

    assert role_mask_weight(frontal).item() == pytest.approx(1.0)
    assert role_mask_weight(tiny).item() == pytest.approx(0.0)
    assert role_mask_weight(edge_on).item() == pytest.approx(0.0)


def test_role_mask_weight_empty_input():
    from ext.pico_pose.loss import role_mask_weight

    empty = torch.zeros((0, 4, 2))
    assert role_mask_weight(empty).shape == (0,)
