# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""PicoPoseNoBox: 轻量无框 Pose 头(PicoDet-S 风格), 用于替换 yolo12-pose 的头部.

设计要点
--------
1. 每个 FPN 层只有一个 **共享** 的深度可分离特征块(DW kxk + PW 1x1 + BN + Hardswish),
   后面可以接一个 SE 通道门控(PicoDet 的 PicoSE);
2. 特征块之后只有两个 1x1 卷积: 类别头 ``nc`` 与四点回归头 ``nk = nkpt * ndim``;
3. **没有 DFL, 也没有框回归分支** —— 网络参数里根本不存在 "框" 这个量;
4. 解码约定与部署端 C++ (``fourpoints_detector.cpp::disPred2Bbox``) 完全一致::

       p_px = (anchor_grid + raw) * stride      # anchor_grid 已含 +0.5

   而 Ultralytics 原生 ``Pose`` 是 ``(raw * 2 + anchor - 0.5) * stride`` —— 这里改成
   PicoDet 的写法, 才能让导出的 ONNX 直接喂给现有 C++ 解码器;
5. ``export=True`` 时输出 ``cls.sigmoid() + 原始回归值``(不解码), 并默认丢掉可见性
   通道, 于是通道数 = ``nc + 2 * nkpt`` —— 与 C++ 里 ``cls_layer`` / ``dis_layer``
   两张输出张量的读法一一对应。

可调参数(默认值即 "轻量 F=64")
-----------------------------
``PICO_HEAD_CFG`` 是模块级默认配置, 也可以在训练脚本里先改再建模型::

    from ultralytics.nn.modules import picopose
    picopose.PICO_HEAD_CFG.update(feat_out=96, use_se=True)

命令行训练时可以用环境变量, 例如::

    ULTRALYTICS_PICO_HEAD="feat_out=96,use_se=1" python -m ultralytics yolo pose train ...

导入注意
--------
不要把 ``import ext.pico_pose.head`` 作为进程里的第一个相关导入: 它需要先导入
上游的 ``Pose``/``Conv``, 会和 ``ultralytics.nn`` 初始化期间的 shim 形成循环。
请使用包级导入 ``from ext.pico_pose import PicoPoseNoBox``(会先完成 nn 初始化),
或者先 ``import ultralytics.nn.tasks``。
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.head import Pose
from ultralytics.utils.tal import make_anchors

__all__ = ("PicoPoseNoBox", "PicoDWBlock", "PicoSE")


def _parse_cfg_from_env(cfg: dict) -> dict:
    """从 ``ULTRALYTICS_PICO_HEAD`` 环境变量读取 ``k=v,k=v`` 形式的覆盖项."""
    raw = os.environ.get("ULTRALYTICS_PICO_HEAD", "").strip()
    if not raw:
        return cfg
    out = dict(cfg)
    for item in raw.split(","):
        k, _, v = item.partition("=")
        k, v = k.strip(), v.strip()
        if not k or not v:
            continue
        for cast in (int, float):
            try:
                out[k] = cast(v)
                break
            except ValueError:
                pass
        else:
            out[k] = v.lower() in {"1", "true", "yes", "y", "on"}
    return out


# 默认: 与骨干同尺度(3 级共用的中间通道数), 2 层, 开 SE
PICO_HEAD_CFG = _parse_cfg_from_env(
    dict(feat_out=64, num_convs=2, use_se=True, k=5, export_xy_only=True, prior_prob=0.01)
)


class PicoDWBlock(nn.Module):
    """PicoFeat 的一层: kxk 深度卷积 + 1x1 逐点卷积 + BN + Hardswish.

    用 Ultralytics 的 ``Conv``(conv+bn+act) 组织, 这样 ``model.fuse()`` 能把 BN 折进
    卷积, 导出 ONNX 后没有额外 BN 算子。
    """

    def __init__(self, cin: int, cout: int, k: int = 5):
        """``groups=math.gcd(cin, cout)`` 保证任意通道组合都能做深度可分离卷积."""
        super().__init__()
        self.dw = Conv(cin, cout, k, 1, g=math.gcd(cin, cout), act=nn.Hardswish())
        self.pw = Conv(cout, cout, 1, 1, act=nn.Hardswish())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向: dw(conv+bn+hs) -> pw(conv+bn+hs)."""
        return self.pw(self.dw(x))


class PicoSE(nn.Module):
    """PicoDet 的通道门控: ``out = bn(conv(x * sigmoid(fc(avgpool(x)))))``."""

    def __init__(self, c: int):
        """初始化通道门控."""
        super().__init__()
        self.fc = nn.Conv2d(c, c, 1)
        self.conv = Conv(c, c, 1, 1, act=nn.Hardswish())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向: 通道注意力加权后再做一次 1x1 卷积."""
        w = self.fc(x.mean((2, 3), keepdim=True)).sigmoid()
        return self.conv(x * w)


class PicoPoseNoBox(Pose):
    """无框轻量 Pose 头(继承 ``Pose`` 以复用 validator/predictor 的 isinstance 分支).

    与 ``Pose`` 的区别:

    - 不创建 ``cv2``(框回归)与 ``dfl``(DFL 卷积), 因此没有框分支;
    - 用 PicoFeat 风格的共享深度可分离特征块 + 两个 1x1 输出;
    - 训练时前向返回 ``{"scores", "kpts", "feats"}``, 交给 ``PicoPoseNoBoxLoss``;
    - 验证时用关键点 min/max 合成伪框, 使现有 ``PoseValidator``/NMS 可原样复用;
    - 导出时只输出 ``nc + (2*nkpt | nk)`` 通道, 对齐部署端 C++ 解码。
    """

    def __init__(
        self,
        nc: int = 80,
        kpt_shape: tuple = (17, 3),
        reg_max: int = 16,
        end2end: bool = False,
        ch: tuple = (),
        feat_out: int | None = None,
        num_convs: int | None = None,
        use_se: bool | None = None,
        k: int | None = None,
        export_xy_only: bool | None = None,
        prior_prob: float | None = None,
    ):
        """构造无框头. ``feat_out`` 等参数缺省时取 ``PICO_HEAD_CFG``."""
        nn.Module.__init__(self)
        cfg = PICO_HEAD_CFG
        self.feat_out = int(cfg["feat_out"] if feat_out is None else feat_out)
        self.num_convs = int(cfg["num_convs"] if num_convs is None else num_convs)
        self.use_se = bool(cfg["use_se"] if use_se is None else use_se)
        self.dw_k = int(cfg["k"] if k is None else k)
        self.export_xy_only = bool(cfg["export_xy_only"] if export_xy_only is None else export_xy_only)
        self.prior_prob = float(cfg["prior_prob"] if prior_prob is None else prior_prob)

        self.nc = nc
        self.nl = len(ch)
        self.reg_max = reg_max  # 仅保留字段以兼容 parse_model / 其它工具
        self.no = nc + int(kpt_shape[0]) * int(kpt_shape[1])
        self.stride = torch.zeros(self.nl)
        self.kpt_shape = tuple(int(v) for v in kpt_shape)
        self.nkpt, self.ndim = self.kpt_shape
        self.nk = self.nkpt * self.ndim
        self._end2end = False  # 不提供 one2one 分支

        stems, ses, cls_head, reg_head = [], [], [], []
        for c in ch:
            blocks = []
            for i in range(self.num_convs):
                cin = c if i == 0 else self.feat_out
                blocks.append(PicoDWBlock(cin, self.feat_out, self.dw_k))
            stems.append(nn.Sequential(*blocks))
            ses.append(PicoSE(self.feat_out) if self.use_se else nn.Identity())
            cls_head.append(nn.Conv2d(self.feat_out, nc, 1))
            reg_head.append(nn.Conv2d(self.feat_out, self.nk, 1))
        self.stem = nn.ModuleList(stems)
        self.se = nn.ModuleList(ses)
        self.cls_pred = nn.ModuleList(cls_head)
        self.reg_pred = nn.ModuleList(reg_head)

    # ------------------------------------------------------------------ 前向
    def forward(
        self, x: list[torch.Tensor]
    ) -> dict[str, torch.Tensor] | torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """训练: ``dict(scores, kpts, feats)``; 验证/导出: 拼接张量."""
        bs = x[0].shape[0]
        scores, kpts = [], []
        for i in range(self.nl):
            f = self.se[i](self.stem[i](x[i]))  # 共享特征(每层只算一次)
            scores.append(self.cls_pred[i](f).view(bs, self.nc, -1))
            kpts.append(self.reg_pred[i](f).view(bs, self.nk, -1))
        preds = {"scores": torch.cat(scores, -1), "kpts": torch.cat(kpts, -1), "feats": x}
        if self.training:
            return preds
        y = self._inference(preds)
        return y if self.export else (y, preds)

    def kpts_decode(self, kpts: torch.Tensor) -> torch.Tensor:
        """``(bs, nk, A)`` 原始输出 -> 像素坐标, 约定 ``p = (anchor + raw) * stride``."""
        bs = kpts.shape[0]
        y = kpts.view(bs, self.nkpt, self.ndim, -1).clone()
        a, st = self.anchors, self.strides  # (2, A), (1, A)
        y[:, :, 0] = (y[:, :, 0] + a[0]) * st
        y[:, :, 1] = (y[:, :, 1] + a[1]) * st
        if self.ndim == 3:
            y[:, :, 2] = y[:, :, 2].sigmoid()
        return y.reshape(bs, self.nk, -1)

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """导出: ``nc + 2*nkpt`` 原始值; 验证: 伪框 + 分数 + 解码关键点."""
        shape = x["feats"][0].shape
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (t.transpose(0, 1) for t in make_anchors(x["feats"], self.stride, 0.5))
            self.shape = shape
        bs = x["scores"].shape[0]

        if self.export:  # 交给 C++ 的 ct + raw * stride
            raw = x["kpts"]
            if self.export_xy_only:
                raw = raw.view(bs, self.nkpt, self.ndim, -1)[:, :, :2].reshape(bs, self.nkpt * 2, -1)
            return torch.cat((x["scores"].sigmoid(), raw), 1)

        # 验证: 关键点 min/max 合成伪框, 让现有 NMS / PoseValidator 不用改
        kpts = self.kpts_decode(x["kpts"]).view(bs, self.nkpt, self.ndim, -1)
        xs, ys = kpts[:, :, 0], kpts[:, :, 1]
        dbox = torch.cat(
            (
                xs.min(1).values.unsqueeze(1),
                ys.min(1).values.unsqueeze(1),
                xs.max(1).values.unsqueeze(1),
                ys.max(1).values.unsqueeze(1),
            ),
            1,
        )
        return torch.cat((dbox, x["scores"].sigmoid(), kpts.reshape(bs, self.nk, -1)), 1)

    def bias_init(self):
        """类别头偏置按 ``prior_prob`` 初始化(框头不存在, 无需 2.0 初始化)."""
        for i, head in enumerate(self.cls_pred):
            head.bias.data[: self.nc] = math.log(self.prior_prob / (1 - self.prior_prob))
        for head in self.reg_pred:
            if head.bias is not None:
                head.bias.data.zero_()

    def fuse(self) -> None:
        """无框头没有需要融合的 one2many 分支, 保留空实现以兼容 ``BaseModel.fuse``."""
        return


# EXT: torch.save 会把类按 __module__ / __qualname__ 写进 checkpoint。把三个自定义类
# 的模块路径固定到上游 shim，保证代码搬到 ext/ 之后，新旧 checkpoint 都仍能从
# ultralytics.nn.modules.picopose 解析到同一个类。
for _cls in (PicoDWBlock, PicoSE, PicoPoseNoBox):
    _cls.__module__ = "ultralytics.nn.modules.picopose"
del _cls
