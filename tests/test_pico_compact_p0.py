"""Contract tests for ``PicoPoseCompactAuxBox`` deployment and training.

The exported tensor is ``[B, A, 18]``:

* columns 0:8   keypoint offsets, decoded as ``(raw + anchor) * stride``;
* column  8     objectness logit (historic name: quality);
* columns 9:11  colour logits;
* columns 11:18 number logits;
* score         ``sigmoid(objectness) * max(softmax(colour)) * max(softmax(number))``;
* class id      ``argmax(colour) * 7 + argmax(number)``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

CFG = Path(__file__).resolve().parents[1] / "ext" / "pico_pose" / "cfg" / "yolo11-pose-pico-compact.yaml"
NC, NKPT, NDIM = 14, 4, 2
NCH = 4 + NC + NKPT * NDIM
A = (480 // 8) * (640 // 8) + (480 // 16) * (640 // 16) + (480 // 32) * (640 // 32)
NUM_ROLE = 7


@pytest.fixture(scope="module")
def model():
    from ultralytics.nn.tasks import PoseModel

    m = PoseModel(str(CFG), nc=NC, data_kpt_shape=(NKPT, NDIM), verbose=False)
    m.eval()
    return m


@pytest.fixture(autouse=True)
def _eval_mode(model):
    model.eval()
    yield
    model.eval()


@pytest.fixture(scope="module")
def eval_outputs(model):
    x = torch.randn(1, 3, 480, 640)
    with torch.no_grad():
        y, preds = model(x)
    return x, y, preds


@pytest.fixture(scope="module")
def export_raw(model, eval_outputs):
    x = eval_outputs[0]
    head = model.model[-1]
    head.export = True
    try:
        with torch.no_grad():
            raw = model(x)
    finally:
        head.export = False
    return raw


def test_eval_shape_is_4_14_8(eval_outputs):
    _, y, _ = eval_outputs
    assert y.shape == (1, NCH, A)
    assert torch.isfinite(y).all()


def test_val_scores_are_objectness_x_colour_x_number(model, eval_outputs):
    _, y, preds = eval_outputs
    color_p = preds["color"].softmax(1)
    number_p = preds["number"].softmax(1)
    obj = preds["quality"].sigmoid()
    expected = (color_p[:, :, None] * number_p[:, None, :]).reshape(1, NC, -1) * obj
    assert torch.allclose(y[:, 4 : 4 + NC], expected, atol=1e-6, rtol=1e-5)
    # each softmax is normalised before the product, so the 14 classes sum to obj
    assert torch.allclose(y[:, 4 : 4 + NC].sum(1), obj[:, 0], atol=1e-6, rtol=1e-5)


def test_class_product_alignment_and_normalisation(model):
    """The (colour, role) product must be a normalised distribution over exactly nc classes."""
    from ext.pico_pose.attrs import COLOR_BY_CLASS, ROLE_BY_CLASS

    # exported class id = argmax(colour) * 7 + argmax(role) must match attrs.py
    assert all(int(c) * NUM_ROLE + int(r) == i for i, (c, r) in enumerate(zip(COLOR_BY_CLASS, ROLE_BY_CLASS)))

    head = model.model[-1]
    torch.manual_seed(0)
    color_p = torch.rand(1, 2, 5)
    color_p /= color_p.sum(1, keepdim=True)
    role_p = torch.rand(1, 7, 5)
    role_p /= role_p.sum(1, keepdim=True)
    scores = head._class_scores(torch.ones(1, 1, 5), color_p, role_p)
    assert torch.allclose(scores.sum(1), torch.ones(1, 5), atol=1e-6), "product must sum to 1"
    assert torch.equal(scores.argmax(1), color_p.argmax(1) * NUM_ROLE + role_p.argmax(1))


def test_class_scores_keep_objectness_ordering(model):
    head = model.model[-1]
    old_filter, old_thr = head.deploy_filter, head.obj_threshold
    try:
        head.deploy_filter = False
        obj = torch.tensor([[[0.9, 0.4]]])  # second anchor below the deployment gate
        color_p = torch.tensor([[[0.8, 0.3], [0.2, 0.7]]])
        number_p = torch.full((1, 7, 2), 1 / 7)
        scores = head._class_scores(obj, color_p, number_p)
        # soft: the objectness ordering survives in the score itself
        assert torch.allclose(scores[0, :, 0].sum(), torch.tensor(0.9), atol=1e-6)
        assert torch.allclose(scores[0, :, 1].sum(), torch.tensor(0.4), atol=1e-6)

        # deployment gate only removes membership, surviving scores are unchanged
        head.deploy_filter, head.obj_threshold = True, 0.5
        filtered = head._class_scores(obj, color_p, number_p)
        assert torch.allclose(filtered[0, :, 0], scores[0, :, 0], atol=1e-7)
        assert filtered[0, :, 1].abs().sum() == 0
    finally:
        head.deploy_filter, head.obj_threshold = old_filter, old_thr


def test_export_raw18_layout_and_logits(eval_outputs, export_raw):
    _, _, preds = eval_outputs
    raw = export_raw
    assert raw.shape == (1, A, 18)
    assert torch.allclose(raw[..., 0:8], preds["kpts"].permute(0, 2, 1), atol=1e-6)
    assert torch.allclose(raw[..., 8], preds["quality"][:, 0], atol=1e-6)
    assert torch.allclose(raw[..., 9:11], preds["color"].permute(0, 2, 1), atol=1e-6)
    assert torch.allclose(raw[..., 11:18], preds["number"].permute(0, 2, 1), atol=1e-6)


def _cpp_decode(raw, anchors, strides, obj_threshold=0.5):
    obj = raw[:, 8].sigmoid()
    color_p = raw[:, 9:11].softmax(-1)
    number_p = raw[:, 11:18].softmax(-1)
    color_max, color_id = color_p.max(-1)
    number_max, number_id = number_p.max(-1)
    keep = obj >= obj_threshold
    score = obj * color_max * number_max  # soft score for NMS ordering
    score = score * keep.to(score.dtype)  # gate: membership only
    class_id = color_id * NUM_ROLE + number_id
    kpts_px = (raw[:, :8].reshape(-1, NKPT, NDIM) + anchors.T[:, None, :]) * strides.T[:, None, :]
    return score, class_id, kpts_px, keep


def test_cpp_raw18_formula_matches_validation(model, eval_outputs, export_raw):
    _, y, _ = eval_outputs
    head = model.model[-1]
    score, class_id, kpts_px, keep = _cpp_decode(export_raw[0], head.anchors, head.strides, head.obj_threshold)
    assert torch.allclose(score[keep], y[0, 4 : 4 + NC].max(0).values[keep], atol=1e-6)
    assert score[~keep].abs().sum() == 0
    color_id = export_raw[0, :, 9:11].softmax(-1).argmax(-1)
    number_id = export_raw[0, :, 11:18].softmax(-1).argmax(-1)
    assert torch.equal(class_id, color_id * NUM_ROLE + number_id)
    assert torch.allclose(kpts_px, y[0, 4 + NC :].T.reshape(A, NKPT, NDIM), atol=1e-5)


def test_kpts_decode_matches_anchor_plus_raw(model, eval_outputs):
    _, _, preds = eval_outputs
    head = model.model[-1]
    decoded = head.kpts_decode(preds["kpts"])
    manual = (preds["kpts"].view(1, NKPT, NDIM, A) + head.anchors.view(1, 1, NDIM, A)) * head.strides.view(
        1, 1, 1, A
    )
    assert torch.allclose(decoded, manual.reshape(1, NKPT * NDIM, A), atol=1e-6)


def test_val_boxes_are_cv2_dfl_not_kpt_hull(model, eval_outputs):
    _, y, preds = eval_outputs
    head = model.model[-1]
    expected = head.decode_bboxes(head.dfl(preds["boxes"]), head.anchors.unsqueeze(0)) * head.strides
    assert torch.allclose(y[:, :4], expected, atol=1e-6, rtol=1e-5)


def test_extra_loss_names_are_logged(model):
    head = model.model[-1]
    assert head.extra_loss_names == ("color_loss", "number_loss", "quality_loss", "geom_loss")
    assert 5 + len(head.extra_loss_names) == 9


def test_head_structure_without_p3_extra(model):
    import torch.nn as nn

    head = model.model[-1]
    assert head.no == 14 + 4 * 16
    assert head.compact_no == 18
    assert head.num_colors == 2 and head.num_roles == 7
    assert not hasattr(head, "p3_extra")

    for i, c in enumerate((64, 128, 256)):
        cv2 = head.cv2[i]
        assert isinstance(cv2, nn.Sequential) and len(cv2) == 3
        assert cv2[0].conv.in_channels == c and cv2[0].conv.kernel_size == (3, 3)
        assert cv2[1].conv.in_channels == 64 and cv2[1].conv.kernel_size == (3, 3)
        assert cv2[2].out_channels == 64 and len(list(cv2[2].children())) == 0

        shared = head.cv_shared[i]
        assert shared[0].conv.kernel_size == (3, 3) and shared[0].conv.groups == c
        assert shared[1].conv.out_channels == 32
        assert head.cv_compact[i].in_channels == 32 and head.cv_compact[i].out_channels == 18
        joint = head.cv_joint[i]
        assert [type(m).__name__ for m in joint] == ["Conv", "Conv", "Conv2d"]
        assert joint[0].conv.in_channels == 32 and joint[0].conv.out_channels == 64
        assert joint[1].conv.in_channels == 64 and joint[1].conv.out_channels == 64
        assert joint[2].in_channels == 64 and joint[2].out_channels == 14
        assert head.dfl.c1 == 16


def test_tal_receives_joint_scores_and_box_branch(model):
    model.args = SimpleNamespace(
        box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0,
        color=0.1, number=0.1, quality=0.05, geom=0.0,
    )
    model.train()
    preds = model(torch.randn(1, 3, 480, 640))
    criterion = model.init_criterion()
    captured = {}
    orig = criterion.assigner.forward

    def wrap(*args, **kwargs):
        captured["scores"] = tuple(args[0].shape)
        captured["boxes"] = tuple(args[1].shape)
        return orig(*args, **kwargs)

    criterion.assigner.forward = wrap
    batch = {
        "batch_idx": torch.tensor([0, 0]),
        "cls": torch.tensor([[0.0], [13.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.3, 0.3, 0.1, 0.1]]),
        "keypoints": torch.rand(2, NKPT, NDIM),
    }
    loss, items = criterion(preds, batch)
    assert captured["scores"] == (1, A, NC)
    assert captured["boxes"] == (1, A, 4)
    assert items.shape == (9,)
    assert torch.isfinite(loss).all()
    assert float(items[5]) > 0
    assert float(items[6]) > 0
    assert float(items[7]) > 0
    assert float(items[8]) == 0.0

    model.args.geom = 1.0
    _, items_on = criterion(preds, batch)
    assert float(items_on[8]) > 0


def test_objectness_target_is_binary(model, monkeypatch):
    """Channel 8 must be trained as objectness (1 on positives, 0 elsewhere), not OKS."""
    model.args = SimpleNamespace(
        box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0,
        color=0.1, number=0.1, quality=0.05, geom=0.0,
    )
    model.train()
    preds = model(torch.randn(1, 3, 480, 640))
    criterion = model.init_criterion()
    batch = {
        "batch_idx": torch.tensor([0, 0]),
        "cls": torch.tensor([[0.0], [13.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.3, 0.3, 0.1, 0.1]]),
        "keypoints": torch.rand(2, NKPT, NDIM),
    }
    captured = []
    orig = torch.nn.functional.binary_cross_entropy_with_logits

    def wrap(*args, **kwargs):
        if len(args) >= 2:
            captured.append(args[1].detach().clone())
        return orig(*args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "binary_cross_entropy_with_logits", wrap)
    criterion(preds, batch)
    targets = [t for t in captured if t.numel() == A]
    assert targets, "no objectness BCE with one value per anchor"
    target = targets[-1]
    assert target.max().item() == 1.0, "positive anchors must target 1 (objectness)"
    assert target.min().item() == 0.0, "background anchors must target 0"
    assert 0 < int((target > 0).sum()) < target.numel()


def test_objectness_pos_weight_exponent(model, monkeypatch):
    """obj_pw_pow=1 is class-balanced; 0.5 gives the square root (background-heavy)."""
    model.args = SimpleNamespace(
        box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0,
        color=0.1, number=0.1, quality=0.05, geom=0.0, obj_pw_pow=1.0,
    )
    model.train()
    preds = model(torch.randn(1, 3, 480, 640))
    criterion = model.init_criterion()
    batch = {
        "batch_idx": torch.tensor([0, 0]),
        "cls": torch.tensor([[0.0], [13.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.3, 0.3, 0.1, 0.1]]),
        "keypoints": torch.rand(2, NKPT, NDIM),
    }
    captured = []
    orig = torch.nn.functional.binary_cross_entropy_with_logits

    def wrap(*args, **kwargs):
        pw = kwargs.get("pos_weight")
        if pw is not None:
            captured.append(float(pw))
        return orig(*args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "binary_cross_entropy_with_logits", wrap)
    criterion(preds, batch)
    balanced = captured[-1]
    model.args.obj_pw_pow = 0.5
    criterion(preds, batch)
    half = captured[-1]
    assert balanced > 1.0
    assert abs(half**2 - balanced) <= 1e-3 * balanced, f"{half**2} != {balanced}"


def test_loss_monitors_are_populated(model):
    model.args = SimpleNamespace(
        box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0,
        color=0.1, number=0.1, quality=0.05, geom=0.0, obj_pw_pow=0.5,
    )
    model.train()
    preds = model(torch.randn(1, 3, 480, 640))
    criterion = model.init_criterion()
    batch = {
        "batch_idx": torch.tensor([0, 0]),
        "cls": torch.tensor([[0.0], [13.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.3, 0.3, 0.1, 0.1]]),
        "keypoints": torch.rand(2, NKPT, NDIM),
    }
    criterion(preds, batch)
    mon = criterion.pop_monitors()
    assert {"mon/s_auc", "mon/s_gap", "mon/attr_agree"} <= set(mon)
    assert any(k.startswith("assign/") for k in mon)
    assert 0.0 <= mon["mon/s_auc"] <= 1.0
    assert 0.0 <= mon["mon/attr_agree"] <= 1.0
    assert criterion.pop_monitors() == {}


def test_assignment_stats_are_reported(model):
    model.args = SimpleNamespace(
        box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0,
        color=0.1, number=0.1, quality=0.05, geom=0.0, obj_pw_pow=0.5,
    )
    model.train()
    preds = model(torch.randn(1, 3, 480, 640))
    criterion = model.init_criterion()
    batch = {
        "batch_idx": torch.tensor([0, 0]),
        "cls": torch.tensor([[0.0], [13.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.3, 0.3, 0.1, 0.1]]),
        "keypoints": torch.rand(2, NKPT, NDIM),
    }
    criterion(preds, batch)
    mon = criterion.pop_monitors()
    n = sum(mon[f"assign/{b}_n"] for b in ("tiny", "small", "medium", "large"))
    assert n == 2, f"one count per GT expected, got {n}"
    assert 0.0 <= mon["assign/zero_frac"] <= 1.0
    assert 0.0 <= mon["assign/mean_ratio"] <= 1.0


def test_empty_batch_does_not_crash(model):
    """A whole batch without GT (possible after mosaic) makes TAL return a float fg_mask."""
    model.args = SimpleNamespace(
        box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0,
        color=0.1, number=0.1, quality=0.05, geom=0.0, obj_pw_pow=0.5,
    )
    model.train()
    preds = model(torch.randn(1, 3, 480, 640))
    criterion = model.init_criterion()
    empty = {
        "batch_idx": torch.zeros(0, dtype=torch.long),
        "cls": torch.zeros((0, 1)),
        "bboxes": torch.zeros((0, 4)),
        "keypoints": torch.zeros((0, NKPT, NDIM)),
    }
    loss, items = criterion(preds, empty)
    assert items.shape == (9,)
    assert torch.isfinite(loss).all()
    mon = criterion.pop_monitors()
    assert set(mon) == {"mon/s_auc", "mon/s_gap", "mon/attr_agree"}


def test_export_strips_box_joint_and_dfl(model, eval_outputs, export_raw):
    head = model.model[-1]
    calls = []
    handles = []
    for module in list(head.cv2) + list(head.cv_joint) + [head.dfl]:
        handles.append(module.register_forward_hook(lambda *_: calls.append(1)))
    head.export = True
    try:
        with torch.no_grad():
            out = model(eval_outputs[0])
    finally:
        head.export = False
        for h in handles:
            h.remove()
    assert not calls
    assert out.shape == (1, A, 18)


def test_onnx_raw18_matches_pytorch_export(tmp_path, model, eval_outputs):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")

    head = model.model[-1]
    x = eval_outputs[0]
    head.export = True
    try:
        with torch.no_grad():
            raw_pt = model(x)
            torch.onnx.export(
                model,
                x,
                str(tmp_path / "compact_raw18.onnx"),
                opset_version=12,
                input_names=["images"],
                output_names=["raw18"],
                dynamo=False,
            )
    finally:
        head.export = False

    sess = ort.InferenceSession(str(tmp_path / "compact_raw18.onnx"), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"images": x.numpy()})[0]
    assert out.shape == (1, A, 18)
    assert torch.allclose(torch.from_numpy(out), raw_pt, atol=1e-5, rtol=1e-4)
