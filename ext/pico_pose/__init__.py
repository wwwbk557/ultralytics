from __future__ import annotations

__all__ = ("PicoPoseCompactAuxBox", "PicoCompactPoseLoss", "pose_geometry_loss")

_HEAD_SYMBOLS = {"PicoPoseCompactAuxBox"}
_LOSS_SYMBOLS = {"PicoCompactPoseLoss", "pose_geometry_loss"}


def __getattr__(name: str):
    if name in _HEAD_SYMBOLS:
        import ultralytics.nn.tasks

        from . import compact_head

        return getattr(compact_head, name)
    if name in _LOSS_SYMBOLS:
        from . import loss

        return getattr(loss, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
