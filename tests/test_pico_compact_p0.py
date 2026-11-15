"""Contract tests for ``PicoPoseCompactAuxBox`` deployment and training.

The exported tensor is ``[B, A, 16]``:

* columns 0:8   keypoint offsets, decoded as ``(raw + anchor) * stride``;
* column  8     one colour logit, ``p_red = sigmoid(z)``;
* columns 9:16  seven role logits, ``sigmoid`` (the YOLOv8 cls head);
* score         ``color_p[c] * sigmoid(role_logit[r])`` with ``color_p = [1-p_red, p_red]``;
* class id      ``argmax(color_p) * 7 + argmax(role)``.

There is no objectness channel: TAL and the exported graph both use the seven
role logits as the anchor confidence.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

CFG = Path(__file__).resolve().parents[1] / "ext" / "pico_pose" / "cfg" / "yolo11-pose-pico-compact.yaml"
NC, NKPT, NDIM = 14, 4, 2
NCH = 4 + NC + NKPT * NDIM
NK = NKPT * NDIM
A = (480 // 8) * (640 // 8) + (480 // 16) * (640 // 16) + (480 // 32) * (640 // 32)
NUM_ROLE = 7
N_EXTRA = 3


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


def _args(**overrides):
    args = dict(box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0, color=0.1, geom=0.0)
    args.update(overrides)
    return SimpleNamespace(**args)


def _rect_kpts():
    """Two axis-aligned plates in label order LT RT RB LB (role mask weight 1)."""
    return torch.tensor(
        [
            [[0.40, 0.40], [0.60, 0.40], [0.60, 0.60], [0.40, 0.60]],
            [[0.25, 0.25], [0.35, 0.25], [0.35, 0.35], [0.25, 0.35]],
        ]
    )


def _batch():
    return {
        "batch_idx": torch.tensor([0, 0]),
        "cls": torch.tensor([[0.0], [13.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.3, 0.3, 0.1, 0.1]]),
        "keypoints": _rect_kpts(),
    }


def test_eval_shape_is_4_14_8(eval_outputs):
    _, y, _ = eval_outputs
    assert y.shape == (1, NCH, A)
    assert torch.isfinite(y).all()


def test_val_scores_are_colour_x_role(model, eval_outputs):
    _, y, preds = eval_outputs
    p_red = preds["color"].sigmoid()
    color_p = torch.cat((1.0 - p_red, p_red), 1)
    role_p = preds["number"].sigmoid()
    expected = (color_p[:, :, None] * role_p[:, None, :]).reshape(1, NC, -1)
    assert torch.allclose(y[:, 4 : 4 + NC], expected, atol=1e-6, rtol=1e-5)
    # colour is complementary, so the class scores sum to the role probability mass
    assert torch.allclose(y[:, 4 : 4 + NC].sum(1), role_p.sum(1), atol=1e-6, rtol=1e-5)


def test_class_product_alignment(model):
    """The (colour, role) product must follow attrs.py and argmax must separate."""
    from ext.pico_pose.attrs import COLOR_BY_CLASS, ROLE_BY_CLASS

    # exported class id = argmax(colour) * 7 + argmax(role) must match attrs.py
    assert all(int(c) * NUM_ROLE + int(r) == i for i, (c, r) in enumerate(zip(COLOR_BY_CLASS, ROLE_BY_CLASS)))

    head = model.model[-1]
    torch.manual_seed(0)
    logit = torch.randn(1, 1, 64)
    color_p = head.color_prob(logit)
    assert torch.allclose(color_p.sum(1), torch.ones(1, 64), atol=1e-6)
    assert torch.allclose(color_p[:, 1], logit[:, 0].sigmoid(), atol=1e-7)
    role_p = torch.rand(1, NUM_ROLE, 64)
    scores = head._class_scores(color_p, role_p)
    assert scores.shape == (1, NC, 64)
    assert torch.equal(scores.argmax(1), color_p.argmax(1) * NUM_ROLE + role_p.argmax(1))


def test_export_raw16_layout_and_logits(eval_outputs, export_raw):
    _, _, preds = eval_outputs
    raw = export_raw
    assert raw.shape == (1, A, 16)
    assert torch.allclose(raw[..., 0:8], preds["kpts"].permute(0, 2, 1), atol=1e-6)
    assert torch.allclose(raw[..., 8], preds["color"][:, 0], atol=1e-6)
    assert torch.allclose(raw[..., 9:16], preds["number"].permute(0, 2, 1), atol=1e-6)


def _cpp_decode(raw, anchors, strides):
    p_red = raw[:, 8].sigmoid()
    color_p = torch.stack((1.0 - p_red, p_red), -1)
    role_p = raw[:, 9:16].sigmoid()
    # p_red >= 0.5 is the numerically stable form of argmax(color_p)
    color_id = (raw[:, 8] >= 0).long()
    role_id = role_p.argmax(-1)
    scores = (color_p[:, :, None] * role_p[:, None, :]).reshape(-1, NC).T
    kpts_px = (raw[:, :8].reshape(-1, NKPT, NDIM) + anchors.T[:, None, :]) * strides.T[:, None, :]
    return scores, color_id * NUM_ROLE + role_id, kpts_px


def test_cpp_raw16_formula_matches_validation(model, eval_outputs, export_raw):
    _, y, _ = eval_outputs
    head = model.model[-1]
    scores, class_id, kpts_px = _cpp_decode(export_raw[0], head.anchors, head.strides)
    assert torch.allclose(scores, y[0, 4 : 4 + NC], atol=1e-6)
    assert torch.allclose(scores.max(0).values, y[0, 4 : 4 + NC].max(0).values, atol=1e-7)
    color_id = export_raw[0, :, 8].ge(0).long()
    role_id = export_raw[0, :, 9:16].argmax(-1)
    assert torch.equal(class_id, color_id * NUM_ROLE + role_id)
    assert torch.allclose(kpts_px, y[0, 4 + NC :].T.reshape(A, NKPT, NDIM), atol=1e-5)


def test_kpts_decode_matches_anchor_plus_raw(model, eval_outputs):
    _, _, preds = eval_outputs
    head = model.model[-1]
    decoded = head.kpts_decode(preds["kpts"])
    manual = (preds["kpts"].view(1, NKPT, NDIM, A) + head.anchors.view(1, 1, NDIM, A)) * head.strides.view(
        1, 1, 1, A
    )
    assert torch.allclose(decoded, manual.reshape(1, NK, A), atol=1e-6)


def test_val_boxes_are_cv2_dfl_by_default(model, eval_outputs):
    _, y, preds = eval_outputs
    head = model.model[-1]
    assert head.deploy_boxes is False
    expected = head.decode_bboxes(head.dfl(preds["boxes"]), head.anchors.unsqueeze(0)) * head.strides
    assert torch.allclose(y[:, :4], expected, atol=1e-6, rtol=1e-5)


def test_deploy_boxes_use_corner_aabb(model):
    """deploy parity: validation builds the NMS box from the four decoded corners."""
    head = model.model[-1]
    head.deploy_boxes = True
    try:
        with torch.no_grad():
            y, preds = model(torch.randn(1, 3, 480, 640))
    finally:
        head.deploy_boxes = False
    k = head.kpts_decode(preds["kpts"]).view(1, NKPT, NDIM, -1)
    xs, ys = k[:, :, 0], k[:, :, 1]
    x0, x1 = xs.min(1).values, xs.max(1).values
    y0, y1 = ys.min(1).values, ys.max(1).values
    want = torch.cat(
        (((x0 + x1) * 0.5)[:, None], ((y0 + y1) * 0.5)[:, None], (x1 - x0)[:, None], (y1 - y0)[:, None]), 1
    )
    assert torch.allclose(y[:, :4], want, atol=1e-4)
    scores = head._class_scores(head.color_prob(preds["color"]), preds["number"].sigmoid())
    assert torch.allclose(y[:, 4 : 4 + NC], scores, atol=1e-6)
    assert torch.allclose(y[:, 4 + NC :], k.reshape(1, NK, -1), atol=1e-6)


def test_extra_loss_names_are_logged(model):
    head = model.model[-1]
    assert head.extra_loss_names == ("color_loss", "geom_loss", "consistency_loss")
    assert 5 + len(head.extra_loss_names) == 8


def test_head_structure_without_train_only_branches(model):
    import torch.nn as nn

    head = model.model[-1]
    assert head.no == 14 + 4 * 16
    assert head.compact_no == 9
    assert head.no_export == 16
    assert head.num_colors == 2 and head.num_roles == 7
    assert not hasattr(head, "p3_extra")
    assert not hasattr(head, "cv_joint")

    for i, c in enumerate((64, 128, 256)):
        cv2 = head.cv2[i]
        assert isinstance(cv2, nn.Sequential) and len(cv2) == 3
        assert cv2[0].conv.in_channels == c and cv2[0].conv.kernel_size == (3, 3)
        assert cv2[1].conv.in_channels == 64 and cv2[1].conv.kernel_size == (3, 3)
        assert cv2[2].out_channels == 64 and len(list(cv2[2].children())) == 0

        shared = head.cv_shared[i]
        assert shared[0].conv.kernel_size == (3, 3) and shared[0].conv.groups == c
        assert shared[1].conv.out_channels == 32
        assert head.cv_compact[i].in_channels == 32 and head.cv_compact[i].out_channels == 9

        cv3 = head.cv3[i]
        assert [type(m).__name__ for m in cv3] == ["DWConv", "DWConv", "Conv2d"]
        assert cv3[0].conv.groups == 32 and cv3[0].conv.kernel_size == (3, 3)
        assert cv3[1].conv.groups == 32 and cv3[1].conv.out_channels == 32
        assert cv3[2].in_channels == 32 and cv3[2].out_channels == NUM_ROLE
        assert head.dfl.c1 == 16


def test_tal_receives_role_scores_and_box_branch(model):
    model.args = _args()
    model.train()
    preds = model(torch.randn(1, 3, 480, 640))
    criterion = model.init_criterion()
    assert criterion.assigner.num_classes == NUM_ROLE
    captured = {}
    orig = criterion.assigner.forward

    def wrap(*args, **kwargs):
        captured["scores"] = tuple(args[0].shape)
        captured["boxes"] = tuple(args[1].shape)
        return orig(*args, **kwargs)

    criterion.assigner.forward = wrap
    loss, items = criterion(preds, _batch())
    assert captured["scores"] == (1, A, NUM_ROLE)
    assert captured["boxes"] == (1, A, 4)
    assert items.shape == (5 + N_EXTRA,)
    assert torch.isfinite(loss).all()
    assert float(items[0]) > 0, "box loss must fire"
    assert float(items[3]) > 0, "role (cls) loss must fire"
    assert float(items[4]) > 0, "dfl loss must fire"
    assert float(items[5]) > 0, "colour loss must fire on proper quads"
    assert float(items[6]) == 0.0
    assert float(items[7]) == 0.0

    model.args.geom = 1.0
    _, items_on = criterion(preds, _batch())
    assert float(items_on[6]) > 0


def test_label_smoothing_only_touches_colour_targets(model, monkeypatch):
    """Colour BCE targets are 1-D per-positive values; smoothing pulls them to 0.5."""
    model.args = _args(role_label_smoothing=0.1)
    model.train()
    preds = model(torch.randn(1, 3, 480, 640))
    criterion = model.init_criterion()
    captured = []
    orig = torch.nn.functional.binary_cross_entropy_with_logits

    def wrap(*args, **kwargs):
        if len(args) >= 2 and torch.is_tensor(args[1]) and args[1].dim() == 1:
            captured.append(args[1].detach().clone())
        return orig(*args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "binary_cross_entropy_with_logits", wrap)
    criterion(preds, _batch())
    assert captured, "colour BCE target not seen"
    target = captured[-1]
    values = {round(float(v), 6) for v in target.unique()}
    assert values <= {0.05, 0.95}, f"unexpected smoothed colour targets {values}"
    assert len(values) == 2, "batch contains one blue and one red plate"


def test_loss_monitors_are_populated(model):
    model.args = _args()
    model.train()
    preds = model(torch.randn(1, 3, 480, 640))
    criterion = model.init_criterion()
    criterion(preds, _batch())
    mon = criterion.pop_monitors()
    assert {"mon/s_auc", "mon/s_gap", "mon/role_mask"} <= set(mon)
    assert any(k.startswith("assign/") for k in mon)
    assert 0.0 <= mon["mon/s_auc"] <= 1.0
    assert 0.0 <= mon["mon/role_mask"] <= 1.0
    assert criterion.pop_monitors() == {}


def test_assignment_stats_are_reported(model):
    model.args = _args()
    model.train()
    preds = model(torch.randn(1, 3, 480, 640))
    criterion = model.init_criterion()
    criterion(preds, _batch())
    mon = criterion.pop_monitors()
    n = sum(mon[f"assign/{b}_n"] for b in ("tiny", "small", "medium", "large"))
    assert n == 2, f"one count per GT expected, got {n}"
    assert 0.0 <= mon["assign/zero_frac"] <= 1.0
    assert 0.0 <= mon["assign/mean_ratio"] <= 1.0


def test_empty_batch_does_not_crash(model):
    """A whole batch without GT (possible after mosaic) makes TAL return a float fg_mask."""
    model.args = _args()
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
    assert items.shape == (5 + N_EXTRA,)
    assert torch.isfinite(loss).all()
    assert items[5:].abs().sum() == 0
    mon = criterion.pop_monitors()
    assert set(mon) == {"mon/s_auc", "mon/s_gap", "mon/role_mask"}


def test_export_strips_box_and_dfl_keeps_role(model, eval_outputs, export_raw):
    head = model.model[-1]
    stripped, kept = [], []
    handles = [m.register_forward_hook(lambda *_: stripped.append(1)) for m in list(head.cv2) + [head.dfl]]
    handles += [m.register_forward_hook(lambda *_: kept.append(1)) for m in list(head.cv3)]
    head.export = True
    try:
        with torch.no_grad():
            out = model(eval_outputs[0])
    finally:
        head.export = False
        for h in handles:
            h.remove()
    assert not stripped
    assert kept, "cv3 is the exported confidence branch and must run"
    assert out.shape == (1, A, 16)


def test_onnx_raw16_matches_pytorch_export(tmp_path, model, eval_outputs):
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
                str(tmp_path / "compact_raw16.onnx"),
                opset_version=12,
                input_names=["images"],
                output_names=["raw16"],
                dynamo=False,
            )
    finally:
        head.export = False

    sess = ort.InferenceSession(str(tmp_path / "compact_raw16.onnx"), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"images": x.numpy()})[0]
    assert out.shape == (1, A, 16)
    assert torch.allclose(torch.from_numpy(out), raw_pt, atol=1e-5, rtol=1e-4)
