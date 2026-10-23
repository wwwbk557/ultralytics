"""PicoPoseNoBox 的冒烟测试：结构解析、前向、导出通道、旧 checkpoint 兼容。

这些测试刻意写得很小，只锁住“迁移代码位置后不能坏”的契约。
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

CFG = Path(__file__).resolve().parents[1] / "cfg" / "yolo12-pose-pico-nobox.yaml"
REPO = CFG.parents[3]
NC, NKPT, NDIM = 1, 4, 3


def _model():
    """按 4 点单类构建 PoseModel。"""
    from ultralytics.nn.tasks import PoseModel

    return PoseModel(str(CFG), nc=NC, data_kpt_shape=(NKPT, NDIM), verbose=False)


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


def test_validate_output_shape():
    """验证态仍是 4 框 + nc + nk，保证 PoseValidator/NMS 可复用。"""
    model = _model().eval()
    with torch.no_grad():
        preds, _ = model(torch.randn(1, 3, 480, 640))
    assert preds.shape[1] == 4 + NC + NKPT * NDIM


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


def test_legacy_checkpoint_loads():
    """旧 checkpoint 的 pickle 路径应通过上游 shim 解析成功。"""
    ckpt = REPO / "runs" / "0913poseNoBox.pt"
    if not ckpt.exists():
        pytest.skip(f"缺少旧 checkpoint: {ckpt}")
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert type(ck["model"].model[-1]).__name__ == "PicoPoseNoBox"
