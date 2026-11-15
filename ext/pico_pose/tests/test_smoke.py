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
    """Axis-aligned plates in LT RT RB LB order so the role mask has weight 1."""
    kpts = torch.tensor(
        [
            [[0.40, 0.40], [0.60, 0.40], [0.60, 0.60], [0.40, 0.60]],
            [[0.25, 0.25], [0.35, 0.25], [0.35, 0.35], [0.25, 0.35]],
        ]
    )[:n]
    return {
        "batch_idx": torch.zeros(n, dtype=torch.long),
        "cls": torch.tensor([[0.0], [13.0]])[:n],
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.3, 0.3, 0.1, 0.1]])[:n],
        "keypoints": kpts,
    }


def _args(**overrides):
    args = dict(box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0, color=0.1, geom=0.0)
    args.update(overrides)
    return SimpleNamespace(**args)


def test_head_builds_and_training_forward():
    model = _model()
    head = model.model[-1]
    assert head.__class__.__name__ == "PicoPoseCompactAuxBox"
    assert head.kpt_shape == (NKPT, NDIM)
    assert head.compact_no == 9
    assert head.no_export == 16

    model.train()
    out = model(torch.randn(1, 3, 480, 640))
    assert set(out) == {"kpts", "color", "number", "scores", "feats", "boxes"}
    assert out["kpts"].shape == (1, NK, A)
    assert out["color"].shape == (1, 1, A)
    assert out["number"].shape == (1, 7, A)
    assert out["scores"].shape == (1, 7, A)
    assert torch.equal(out["scores"], out["number"]), "TAL and the export share the role logits"
    assert out["boxes"].shape == (1, 4 * head.reg_max, A)


def test_eval_shape():
    model = _model().eval()
    with torch.no_grad():
        y, preds = model(torch.randn(1, 3, 480, 640))
    assert y.shape == (1, 4 + NC + NK, A)
    assert "boxes" in preds and "raw16" not in preds


def test_head_structure():
    model = _model()
    head = model.model[-1]
    assert head.no == NC + 4 * 16
    assert head.compact_no == 9
    assert head.no_export == 16
    assert not hasattr(head, "p3_extra")
    assert not hasattr(head, "cv_joint")

    for i, c in enumerate((64, 128, 256)):
        cv2 = head.cv2[i]
        assert len(cv2) == 3
        assert cv2[0].conv.kernel_size == (3, 3) and cv2[0].conv.in_channels == c
        assert cv2[1].conv.kernel_size == (3, 3) and cv2[1].conv.in_channels == 64
        assert cv2[2].out_channels == 64 and cv2[2].kernel_size == (1, 1)

        shared = head.cv_shared[i]
        assert shared[0].conv.kernel_size == (3, 3) and shared[0].conv.groups == c
        assert shared[1].conv.out_channels == 32
        assert head.cv_compact[i].in_channels == 32 and head.cv_compact[i].out_channels == 9

        cv3 = head.cv3[i]
        assert [type(m).__name__ for m in cv3] == ["DWConv", "DWConv", "Conv2d"]
        assert cv3[0].conv.in_channels == 32 and cv3[0].conv.groups == 32
        assert cv3[1].conv.in_channels == 32 and cv3[2].out_channels == 7
        assert head.dfl.c1 == 16


def test_export_is_raw16_row_major():
    model = _model().eval()
    head = model.model[-1]
    head.export = True
    try:
        with torch.no_grad():
            y = model(torch.randn(1, 3, 480, 640))
    finally:
        head.export = False
    assert y.shape == (1, A, 16)


def test_export_skips_box_branch_keeps_role_head():
    model = _model().eval()
    head = model.model[-1]
    calls, role = [], []
    handles = [m.register_forward_hook(lambda *_: calls.append(1)) for m in list(head.cv2) + [head.dfl]]
    handles += [m.register_forward_hook(lambda *_: role.append(1)) for m in list(head.cv3)]
    head.export = True
    try:
        with torch.no_grad():
            model(torch.randn(1, 3, 480, 640))
    finally:
        head.export = False
        for h in handles:
            h.remove()
    assert not calls
    assert role, "cv3 is part of the exported graph"


def test_loss_vector_and_geometry_switch():
    model = _model()
    model.args = _args()
    model.train()
    preds = model(torch.randn(1, 3, 480, 640))
    criterion = model.init_criterion()
    assert criterion.__class__.__name__ == "PicoCompactPoseLoss"
    loss, items = criterion(preds, _batch())
    assert items.shape == (8,)
    assert torch.isfinite(loss).all()
    assert float(items[0]) > 0 and float(items[4]) > 0
    assert float(items[5]) > 0, "colour BCE must fire on proper quads"
    assert float(items[6]) == 0.0
    assert float(items[7]) == 0.0  # consistency disabled by default

    model.args.geom = 1.0
    _, items_on = criterion(preds, _batch())
    assert float(items_on[6]) > 0


def test_fuse_keeps_branches():
    model = _model()
    head = model.model[-1]
    model.fuse()
    assert head.cv2 is not None and head.cv3 is not None
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


def test_deploy_boxes_use_corner_aabb():
    """Validation boxes must match the 4-corner extent, the same box the C++ decoder builds."""
    model = _model()
    head = model.model[-1]
    model.eval()
    with torch.no_grad():
        head.deploy_boxes = True
        try:
            y, preds = model(torch.randn(1, 3, 480, 640))
        finally:
            head.deploy_boxes = False
    k = head.kpts_decode(preds["kpts"]).view(1, head.nkpt, head.ndim, -1)
    xs, ys = k[:, :, 0], k[:, :, 1]
    x0, x1 = xs.min(1).values, xs.max(1).values
    y0, y1 = ys.min(1).values, ys.max(1).values
    want = torch.cat(
        (((x0 + x1) * 0.5)[:, None], ((y0 + y1) * 0.5)[:, None], (x1 - x0)[:, None], (y1 - y0)[:, None]), 1
    )
    assert torch.allclose(y[:, :4], want, atol=1e-4)


def test_colour_is_one_complementary_logit():
    model = _model().eval()
    head = model.model[-1]
    z = torch.tensor([[-6.0], [0.0], [6.0]])
    p = head.color_prob(z)
    assert p.shape == (3, 2)
    assert torch.allclose(p.sum(1), torch.ones(3), atol=1e-6)
    assert torch.allclose(p[:, 1], torch.sigmoid(z[:, 0])) and torch.allclose(p[:, 0], 1 - torch.sigmoid(z[:, 0]))


def test_role_logits_are_the_exported_confidence():
    """Val scores must be colour x role with no objectness factor anywhere."""
    model = _model().eval()
    head = model.model[-1]
    with torch.no_grad():
        y, preds = model(torch.randn(1, 3, 480, 640))
    color_p = head.color_prob(preds["color"])
    role_p = preds["number"].sigmoid()
    scores = head._class_scores(color_p, role_p)
    assert torch.allclose(y[:, 4 : 4 + NC], scores, atol=1e-6)
    # the product is separable: max over classes = max(colour) * max(role)
    assert torch.allclose(scores.max(1).values, color_p.max(1).values * role_p.max(1).values, atol=1e-6)
    assert not hasattr(head, "quality") and head.compact_no == 9


def test_empty_batch_has_zero_extra_terms():
    model = _model()
    model.args = _args()
    model.train()
    preds = model(torch.randn(1, 3, 480, 640))
    criterion = model.init_criterion()
    empty = {
        "batch_idx": torch.zeros(0, dtype=torch.long),
        "cls": torch.zeros(0, 1),
        "bboxes": torch.zeros(0, 4),
        "keypoints": torch.zeros(0, NKPT, NDIM),
    }
    loss, items = criterion(preds, empty)
    assert items.shape == (8,)
    assert torch.isfinite(loss).all()
    assert items[5:].abs().sum() == 0
