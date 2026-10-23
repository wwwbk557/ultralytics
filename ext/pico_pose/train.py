#!/usr/bin/env python3
"""PicoPoseNoBox 训练入口：默认模型 yaml 指向本扩展目录。

用法::

    python -m ext.pico_pose.train data=/path/to/pose.yaml epochs=300 imgsz=480,640 device=0

所有参数都是 Ultralytics 的 ``key=value`` 形式，会原样传给 ``PoseTrainer``。
"""

from __future__ import annotations

import sys
from pathlib import Path

from ultralytics.models.yolo.pose import PoseTrainer

CFG = Path(__file__).resolve().parent / "cfg" / "yolo12-pose-pico-nobox.yaml"


def main() -> None:
    """解析 ``key=value`` 参数并启动训练。"""
    overrides: dict[str, str] = {"model": str(CFG)}
    for arg in sys.argv[1:]:
        key, sep, value = arg.partition("=")
        if not sep or not key:
            raise SystemExit(f"参数必须是 key=value 形式，收到: {arg!r}")
        overrides[key.strip()] = value.strip()
    PoseTrainer(overrides=overrides).train()


if __name__ == "__main__":
    main()
