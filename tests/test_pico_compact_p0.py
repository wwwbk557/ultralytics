"""Contract tests for ``PicoPoseCompactAuxBox`` deployment and training.

The exported tensor is ``[B, A, 18]``:

* columns 0:8   keypoint offsets, decoded as ``(raw + anchor) * stride``;
* column  8     quality logit;
* columns 9:11  colour logits;
* columns 11:18 number logits;
* score         ``sigmoid(quality) * max(softmax(colour)) * max(softmax(number))``;
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
NUM_NUMBER = 7


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


def test_val_scores_are_colour_x_number_x_quality(eval_outputs):
    _, y, preds = eval_outputs
    color_p = preds["color"].softmax(1)
    number_p = preds["number"].softmax(1)
    expected = (color_p[:, :, None] * number_p[:, None, :]).reshape(1, NC, -1)
    expected = expected * preds["quality"].sigmoid()
    assert torch.allclose(y[:, 4 : 4 + NC], expected, atol=1e-6, rtol=1e-5)
    assert torch.allclose(y[:, 4 : 4 + NC].sum(1), preds["quality"].sigmoid()[:, 0], atol=1e-6, rtol=1e-5)


def test_export_raw18_layout_and_logits(eval_outputs, export_raw):
    _, _, preds = eval_outputs
    raw = export_raw
    assert raw.shape == (1, A, 18)
    assert torch.allclose(raw[..., 0:8], preds["kpts"].permute(0, 2, 1), atol=1e-6)
    assert torch.allclose(raw[..., 8], preds["quality"][:, 0], atol=1e-6)
    assert torch.allclose(raw[..., 9:11], preds["color"].permute(0, 2, 1), atol=1e-6)
    assert torch.allclose(raw[..., 11:18], preds["number"].permute(0, 2, 1), atol=1e-6)


def _cpp_decode(raw, anchors, strides):
    quality = raw[:, 8].sigmoid()
    color_p = raw[:, 9:11].softmax(-1)
    number_p = raw[:, 11:18].softmax(-1)
    color_max, color_id = color_p.max(-1)
    number_max, number_id = number_p.max(-1)
    score = quality * color_max * number_max
    class_id = color_id * NUM_NUMBER + number_id
    kpts_px = (raw[:, :8].reshape(-1, NKPT, NDIM) + anchors.T[:, None, :]) * strides.T[:, None, :]
    return score, class_id, kpts_px


def test_cpp_raw18_formula_matches_validation(model, eval_outputs, export_raw):
    _, y, _ = eval_outputs
    head = model.model[-1]
    score, class_id, kpts_px = _cpp_decode(export_raw[0], head.anchors, head.strides)
    assert torch.allclose(score, y[0, 4 : 4 + NC].max(0).values, atol=1e-6)
    color_id = export_raw[0, :, 9:11].softmax(-1).argmax(-1)
    number_id = export_raw[0, :, 11:18].softmax(-1).argmax(-1)
    assert torch.equal(class_id, color_id * NUM_NUMBER + number_id)
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


def test_extra_loss_names_are_logged():
    from ext.pico_pose.loss import LOSS_NAMES

    assert LOSS_NAMES[:5] == ("box_loss", "pose_loss", "kobj_loss", "cls_loss", "dfl_loss")
    assert LOSS_NAMES[5:] == ("color_loss", "number_loss", "quality_loss", "geom_loss")
    assert len(LOSS_NAMES) == 9


def test_head_structure_without_p3_extra(model):
    import torch.nn as nn

    head = model.model[-1]
    assert head.no == 14 + 4 * 16
    assert head.compact_no == 18
    assert head.num_colors == 2 and head.num_numbers == 7
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
        assert head.cv_joint[i].in_channels == 32 and head.cv_joint[i].out_channels == 14
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
    assert loss.shape == (9,)
    assert torch.isfinite(loss).all()
    assert float(items["color_loss"]) > 0
    assert float(items["number_loss"]) > 0
    assert float(items["quality_loss"]) > 0
    assert float(items["geom_loss"]) == 0.0

    model.args.geom = 1.0
    _, items_on = criterion(preds, batch)
    assert float(items_on["geom_loss"]) > 0


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
