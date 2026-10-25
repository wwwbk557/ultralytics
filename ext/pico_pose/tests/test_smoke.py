"""PicoPoseNoBox 的冒烟测试：结构解析、前向、导出通道、checkpoint 类路径。"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

CFG = Path(__file__).resolve().parents[1] / "cfg" / "yolo12-pose-pico-nobox.yaml"
Y11_CFG = CFG.with_name("yolo11-pose-pico-nobox.yaml")
AUX_CFG = CFG.with_name("yolo12-pose-pico-auxbox.yaml")
AUX11_CFG = CFG.with_name("yolo11-pose-pico-auxbox.yaml")
NC, NKPT, NDIM = 1, 4, 3


def _model(cfg: Path = CFG):
    """按 4 点单类构建 PoseModel。"""
    from ultralytics.nn.tasks import PoseModel

    return PoseModel(str(cfg), nc=NC, data_kpt_shape=(NKPT, NDIM), verbose=False)


def test_head_built_and_forward_shapes():
    """parse_model 能解析 PicoPoseNoBox，训练态返回 scores/kpts/feats。"""
    model = _model()
    head = model.model[-1]
    assert head.__class__.__name__ == "PicoPoseNoBox"
    assert head.kpt_shape == (NKPT, NDIM)

    model.train()
    out = model(torch.randn(1, 3, 480, 640))
    assert set(out) == {"scores", "kpts", "feats"}
    assert out["scores"].shape == (1, NC, 6300)
    assert out["kpts"].shape == (1, NKPT * NDIM, 6300)


def test_yolo11_config_builds():
    """同一个头可直接挂到 YOLO11-pose 的 P3/P4/P5 颈部。"""
    model = _model(Y11_CFG)
    head = model.model[-1]
    assert head.__class__.__name__ == "PicoPoseNoBox"
    assert head.kpt_shape == (NKPT, NDIM)


def test_validate_output_shape():
    """验证态仍是 4 框 + nc + nk，保证 PoseValidator/NMS 可复用。"""
    model = _model().eval()
    with torch.no_grad():
        preds, _ = model(torch.randn(1, 3, 480, 640))
    assert preds.shape[1] == 4 + NC + NKPT * NDIM


def test_validation_pseudo_box_is_xywh():
    """验证态前 4 列必须是 xywh；NMS 的 xywh->xyxy 之后应等于关键点 hull。"""
    from ultralytics.utils.ops import xywh2xyxy

    model = _model().eval()
    with torch.no_grad():
        preds, _ = model(torch.randn(1, 3, 480, 640))
    box = preds[0, :4]  # (4, A), xywh
    kpts = preds[0, 4 + NC :].reshape(NKPT, NDIM, -1)
    xs, ys = kpts[:, 0], kpts[:, 1]
    xywh = torch.stack(
        (
            (xs.min(0).values + xs.max(0).values) / 2,
            (ys.min(0).values + ys.max(0).values) / 2,
            xs.max(0).values - xs.min(0).values,
            ys.max(0).values - ys.min(0).values,
        ),
        0,
    )
    assert torch.allclose(box, xywh, atol=1e-4)

    xyxy = xywh2xyxy(box.T).T  # 与 non_max_suppression 内部的处理一致
    assert torch.allclose(xyxy[0], xs.min(0).values, atol=1e-4)
    assert torch.allclose(xyxy[1], ys.min(0).values, atol=1e-4)
    assert torch.allclose(xyxy[2], xs.max(0).values, atol=1e-4)
    assert torch.allclose(xyxy[3], ys.max(0).values, atol=1e-4)


def test_export_channel_contract():
    """默认 export_xy_only=True：输出 nc + 2*nkpt，不含框和可见性。"""
    model = _model().eval()
    head = model.model[-1]
    head.export = True
    try:
        with torch.no_grad():
            out = model(torch.randn(1, 3, 480, 640))
    finally:
        head.export = False
    assert out.shape == (1, NC + 2 * NKPT, 6300)


def test_export_row_major_and_decode():
    """导出可选 [A, C] 行优先 + 图内 kpt 解码，C++ 不再需要 anchor/stride。"""
    model = _model().eval()
    head = model.model[-1]
    head.export = True
    head.export_row_major = True
    head.export_decode = True
    try:
        with torch.no_grad():
            out = model(torch.randn(1, 3, 480, 640))
    finally:
        head.export = False
        head.export_row_major = False
        head.export_decode = False
    assert out.shape == (1, 6300, NC + 2 * NKPT)
    # 解码后 xy 在像素尺度，不应再是 0 附近的 raw 值
    assert out[..., NC:].abs().max() > 1.0


def test_pred_heads_use_3x3_dw():
    """默认输出块是 DW 3x3(+BN+HS) -> 1x1；空间尺寸和输出通道都不变。"""
    from ext.pico_pose import PicoHeadBlock

    head = _model().model[-1]
    assert head.head_k == 3
    assert head.head_dw is True
    for blocks in (head.cls_pred, head.reg_pred):
        for blk in blocks:
            assert isinstance(blk, PicoHeadBlock)
            assert blk.spatial.conv.kernel_size == (3, 3)
            assert blk.spatial.conv.padding == (1, 1)
            assert blk.spatial.conv.groups == head.feat_out  # 深度可分离
            assert blk.pred.kernel_size == (1, 1)


def test_head_k1_restores_1x1_ablation():
    """head_k=1 退回旧的单层 1x1 预测，便于 A/B；参数量差值 = 6 个 DW3x3 块。"""
    from ext.pico_pose import PicoPoseNoBox

    h1 = PicoPoseNoBox(nc=NC, kpt_shape=(NKPT, NDIM), ch=(64, 128, 256), head_k=1)
    h3 = PicoPoseNoBox(nc=NC, kpt_shape=(NKPT, NDIM), ch=(64, 128, 256), head_k=3)
    for blk in list(h1.cls_pred) + list(h1.reg_pred):
        assert isinstance(blk.spatial, torch.nn.Identity)
        assert blk.pred.kernel_size == (1, 1)

    # 每层 cls/reg 各一个 DW3x3：64*3*3 权重 + 64*2 BN
    diff = sum(p.numel() for p in h3.parameters()) - sum(p.numel() for p in h1.parameters())
    assert diff == 3 * 2 * (64 * 3 * 3 + 64 * 2)


def test_fuse_folds_output_block_bn():
    """BaseModel.fuse 应把输出块里的 BN 折进卷积，导出图里不残留 BN。"""
    model = _model()
    head = model.model[-1]
    assert any(isinstance(m, torch.nn.BatchNorm2d) for m in head.cls_pred.modules())

    model.fuse()
    assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in head.modules())
    with torch.no_grad():
        model.eval()
        y = model(torch.randn(1, 3, 480, 640))[0]
    assert y.shape[1] == 4 + NC + NKPT * NDIM


def test_aux_box_is_train_and_val_only():
    """aux box 在训练和验证时存在（验证要算 val loss），导出路径不执行 cv2。"""
    model = _model(AUX_CFG)
    head = model.model[-1]
    assert head.aux_box is True
    assert hasattr(head, "cv2") and hasattr(head, "dfl")

    model.train()
    out = model(torch.randn(1, 3, 480, 640))
    assert "boxes" in out
    assert out["boxes"].shape == (1, 4 * head.reg_max, 6300)

    calls = []
    handles = [m.register_forward_hook(lambda *_: calls.append(1)) for m in head.cv2]
    model.eval()
    with torch.no_grad():
        _, preds = model(torch.randn(1, 3, 480, 640))
    assert "boxes" in preds  # validator 的 val loss 需要

    calls.clear()
    head.export = True
    try:
        with torch.no_grad():
            y = model(torch.randn(1, 3, 480, 640))
    finally:
        head.export = False
        for h in handles:
            h.remove()

    assert not calls, "导出路径不应执行辅助框分支"
    assert y.shape == (1, NC + 2 * NKPT, 6300)


def test_aux_loss_runs_tal_box_dfl():
    """aux 模式走官方 v8PoseLoss：box/dfl 槽位应非零。"""
    from types import SimpleNamespace

    model = _model(AUX_CFG)
    model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0)
    model.train()
    preds = model(torch.randn(1, 3, 480, 640))
    batch = {
        "batch_idx": torch.tensor([0, 0]),
        "cls": torch.tensor([[0.0], [0.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.3, 0.3, 0.1, 0.1]]),
        "keypoints": torch.rand(2, NKPT, NDIM),
    }
    loss, items = model.init_criterion()(preds, batch)
    assert items.shape == (5,)
    assert torch.isfinite(loss).all()
    assert float(items[0]) > 0, "box loss 应被激活"
    assert float(items[4]) > 0, "dfl loss 应被激活"


def test_aux_yolo11_config_builds():
    """aux 头同样可挂 YOLO11 颈部。"""
    model = _model(AUX11_CFG)
    assert model.model[-1].aux_box is True


def test_checkpoint_uses_ext_module_path(tmp_path):
    """新 checkpoint 直接引用 ext.pico_pose.head，不再依赖上游 shim。"""
    from ext.pico_pose import PicoPoseNoBox

    assert PicoPoseNoBox.__module__ == "ext.pico_pose.head"

    head = _model().model[-1]
    path = tmp_path / "head.pt"
    torch.save(head, path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    assert type(loaded).__module__ == "ext.pico_pose.head"
