#!/usr/bin/env python3
"""Pico 头训练入口：默认用带训练期辅助框的 yolo12 配置。

用法::

    python -m ext.pico_pose.train data=/path/to/pose.yaml epochs=300 imgsz=480,640 device=0
    python -m ext.pico_pose.train model=ext/pico_pose/cfg/yolo11-pose-pico-auxbox.yaml data=...

所有参数都是 Ultralytics 的 ``key=value`` 形式；数字/布尔/逗号分隔的尺寸会被
转换成 Python 值后再交给 ``PoseTrainer``。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from ultralytics.models.yolo.pose import PoseTrainer

CFG = Path(__file__).resolve().parent / "cfg" / "yolo12-pose-pico-auxbox.yaml"


def _parse_value(raw: str) -> Any:
    """把命令行字符串转成 int/float/bool/list，其它保持字符串。"""
    if "," in raw:
        return [_parse_value(part) for part in raw.split(",")]
    low = raw.lower()
    if low in {"true", "yes", "on"}:
        return True
    if low in {"false", "no", "off"}:
        return False
    if low in {"none", "null"}:
        return None
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            pass
    return raw


def main() -> None:
    """解析 ``key=value`` 参数并启动训练。"""
    overrides: dict[str, Any] = {"model": str(CFG)}
    for arg in sys.argv[1:]:
        key, sep, value = arg.partition("=")
        if not sep or not key:
            raise SystemExit(f"参数必须是 key=value 形式，收到: {arg!r}")
        overrides[key.strip()] = _parse_value(value.strip())
    PoseTrainer(overrides=overrides).train()


if __name__ == "__main__":
    main()
