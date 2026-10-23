# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""EXT shim：真正的实现在 ``ext/pico_pose/head.py``。

保留这个文件有两个硬性原因，二者都不能靠“把代码全搬走”解决：

1. 旧 checkpoint（如 ``runs/0913poseNoBox.pt``）里的 pickle 路径是
   ``ultralytics.nn.modules.picopose.PicoPoseNoBox``；
2. ``ultralytics.nn.tasks.parse_model`` 通过模块 globals 按类名解析 yaml 里的
   ``PicoPoseNoBox``。

因此这里只做 re-export，不含任何实现。
"""

from ext.pico_pose.head import PICO_HEAD_CFG, PicoDWBlock, PicoPoseNoBox, PicoSE

__all__ = ("PicoPoseNoBox", "PicoDWBlock", "PicoSE", "PICO_HEAD_CFG")
