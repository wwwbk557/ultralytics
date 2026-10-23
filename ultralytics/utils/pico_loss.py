# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""EXT shim：真正的实现在 ``ext/pico_pose/loss.py``。

``PoseModel.init_criterion`` 在运行时按这个路径导入无框损失，保留 shim 可让
上游那处接入代码保持不变。
"""

from ext.pico_pose.loss import (
    PicoPoseNoBoxLoss,
    atss_assign,
    pairwise_iou,
    varifocal_loss,
    wing_loss,
)

__all__ = ("PicoPoseNoBoxLoss", "atss_assign", "wing_loss", "varifocal_loss", "pairwise_iou")
