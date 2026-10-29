"""Smoke tests for the compact four-keypoint pose head and its loss."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

CFG = Path(__file__).resolve().parents[1] / "cfg" / "yolo11-pose-pico-compact.yaml"
NC, NKPT, NDIM = 14, 4, 2
NK = NKPT * NDIM
A = (480 // 8) * (640 // 8) + (480 // 16) * (640 // 16) + (480 // 32) * (640 // 32)


def _model():
    from ultralytics.nn.tasks import PoseModel

    return PoseModel(str(CFG), nc=NC, data_kpt_shape=(NKPT, NDIM), verbose=False)


def _batch(n=2):
    return {
        "batch_idx": torch.zeros(n, dtype=torch.long),
        "cls": torch.tensor([[0.0], [13.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.3, 0.3, 0.1, 0.1]]),
        "keypoints": torch.rand(n, NKPT, NDIM),
    }


def test_head_builds_and_training_forward():
    model = _model()
    head = model.model[-1]
    assert head.__class__.__name__ == "PicoPoseCompactAuxBox"
    assert head.kpt_shape == (NKPT, NDIM)
    assert head.compact_no == 18

    model.train()
    out = model(torch.randn(1, 3, 480, 640))
    assert set(out) == {"kpts", "quality", "color", "number", "feats", "scores", "boxes"}
    assert out["kpts"].shape == (1, NK, A)
    assert out["quality"].shape == (1, 1, A)
    assert out["color"].shape == (1, 2, A)
    assert out["number"].shape == (1, 7, A)
    assert out["scores"].shape == (1, NC, A)
    assert out["boxes"].shape == (1, 4 * head.reg_max, A)


def test_eval_shape():
    model = _model().eval()
    with torch.no_grad():
        y, preds = model(torch.randn(1, 3, 480, 640))
    assert y.shape == (1, 4 + NC + NK, A)
    assert "boxes" in preds


def test_head_structure():
    model = _model()
    head = model.model[-1]
    assert head.no == NC + 4 * 16
    assert head.compact_no == 18
    assert not hasattr(head, "p3_extra")

    for i, c in enumerate((64, 128, 256)):
        cv2 = head.cv2[i]
        assert len(cv2) == 3
        assert cv2[0].conv.kernel_size == (3, 3) and cv2[0].conv.in_channels == c
        assert cv2[1].conv.kernel_size == (3, 3) and cv2[1].conv.in_channels == 64
        assert cv2[2].out_channels == 64 and cv2[2].kernel_size == (1, 1)

        shared = head.cv_shared[i]
        assert shared[0].conv.kernel_size == (3, 3) and shared[0].conv.groups == c
        assert shared[1].conv.out_channels == 32
        assert head.cv_compact[i].in_channels == 32 and head.cv_compact[i].out_channels == 18
        assert head.cv_joint[i].in_channels == 32 and head.cv_joint[i].out_channels == NC


def test_export_is_raw18_row_major():
    model = _model().eval()
    head = model.model[-1]
    head.export = True
    try:
        with torch.no_grad():
            y = model(torch.randn(1, 3, 480, 640))
    finally:
        head.export = False
    assert y.shape == (1, A, 18)


def test_export_skips_box_and_joint_branches():
    model = _model().eval()
    head = model.model[-1]
    calls = []
    handles = [m.register_forward_hook(lambda *_: calls.append(1)) for m in list(head.cv2) + list(head.cv_joint)]
    handles.append(head.dfl.register_forward_hook(lambda *_: calls.append(1)))
    head.export = True
    try:
        with torch.no_grad():
            model(torch.randn(1, 3, 480, 640))
    finally:
        head.export = False
        for h in handles:
            h.remove()
    assert not calls


def test_loss_vector_and_geometry_switch():
    model = _model()
    model.args = SimpleNamespace(
        box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0,
        color=0.1, number=0.1, quality=0.05, geom=0.0,
    )
    model.train()
    preds = model(torch.randn(1, 3, 480, 640))
    criterion = model.init_criterion()
    assert criterion.__class__.__name__ == "PicoCompactPoseLoss"
    loss, items = criterion(preds, _batch())
    assert loss.shape == (9,)
    assert torch.isfinite(loss).all()
    assert float(items["box_loss"]) > 0 and float(items["dfl_loss"]) > 0
    assert float(items["geom_loss"]) == 0.0

    model.args.geom = 1.0
    _, items_on = criterion(preds, _batch())
    assert float(items_on["geom_loss"]) > 0
    assert list(items_on) == [
        "box_loss",
        "pose_loss",
        "kobj_loss",
        "cls_loss",
        "dfl_loss",
        "color_loss",
        "number_loss",
        "quality_loss",
        "geom_loss",
    ]


def test_fuse_keeps_branches():
    model = _model()
    head = model.model[-1]
    model.fuse()
    assert head.cv2 is not None and head.cv_joint is not None
    with torch.no_grad():
        model.eval()
        y = model(torch.randn(1, 3, 480, 640))[0]
    assert y.shape[1] == 4 + NC + NK


def test_checkpoint_module_path(tmp_path):
    head = _model().model[-1]
    path = tmp_path / "head.pt"
    torch.save(head, path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    assert type(loaded).__module__ == "ext.pico_pose.compact_head"


def test_geometry_loss_is_scale_invariant():
    from ext.pico_pose.loss import pose_geometry_loss

    pts = torch.tensor([[[0.0, 0.0], [100.0, 5.0], [98.0, 60.0], [-2.0, 55.0]]])
    assert torch.allclose(pose_geometry_loss(pts), pose_geometry_loss(pts * 3.0), atol=1e-6)


def test_invalid_shapes_raise():
    from ext.pico_pose.compact_head import PicoPoseCompactAuxBox

    with pytest.raises(ValueError, match="kpt_shape"):
        PicoPoseCompactAuxBox(nc=14, kpt_shape=(4, 3), ch=(16, 32, 64))
    with pytest.raises(ValueError, match="nc"):
        PicoPoseCompactAuxBox(nc=21, kpt_shape=(4, 2), ch=(16, 32, 64))
