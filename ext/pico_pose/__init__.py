"""PicoPoseNoBox：轻量无框 Pose 头与 ATSS/VFL/Wing 损失。

本模块**完全惰性**导入子模块，目的是避免和上游 shim 形成循环导入：
``ultralytics`` 初始化时可能经 shim 反向导入这里，此时若急切导入 head 会拿到
一个只初始化了一半的模块。显式访问符号时再加载即可。
"""

from __future__ import annotations

__all__ = (
    "PICO_HEAD_CFG",
    "PicoDWBlock",
    "PicoPoseNoBox",
    "PicoSE",
    "PicoPoseNoBoxLoss",
    "atss_assign",
    "pairwise_iou",
    "varifocal_loss",
    "wing_loss",
)

_HEAD_SYMBOLS = {"PICO_HEAD_CFG", "PicoDWBlock", "PicoPoseNoBox", "PicoSE"}
_LOSS_SYMBOLS = {"PicoPoseNoBoxLoss", "atss_assign", "pairwise_iou", "varifocal_loss", "wing_loss"}


def __getattr__(name: str):
    """按需导入 head / loss，并在加载 head 前确保 ultralytics.nn 已初始化完。"""
    if name in _HEAD_SYMBOLS:
        import ultralytics.nn.tasks  # noqa: F401  先完成 nn 初始化，避免 shim 循环导入

        from . import head

        return getattr(head, name)
    if name in _LOSS_SYMBOLS:
        from . import loss

        return getattr(loss, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
