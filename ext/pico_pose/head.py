# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""PicoPoseNoBox: 轻量无框 Pose 头(PicoDet-S 风格), 用于替换 yolo12-pose 的头部.

设计要点
--------
1. 每个 FPN 层只有一个 **共享** 的深度可分离特征块(DW kxk + PW 1x1 + BN + Hardswish),
   后面可以接一个 SE 通道门控(PicoDet 的 PicoSE);
2. 特征块之后是 **解耦输出块**: 每个分支各做一次 ``head_k x head_k`` 空间卷积
   (默认 3x3 深度可分离 + BN + Hardswish), 再 1x1 投影到 ``nc`` / ``nk = nkpt * ndim``。
   3x3 让相邻网格参与决策(空间感知), 也让类别数变多时分类头有足够容量;
   ``head_k=1`` 可退回旧的单层 1x1 预测做 A/B;
3. **没有 DFL, 也没有框回归分支** —— 网络参数里根本不存在 "框" 这个量;
4. 解码约定与部署端 C++ (``fourpoints_detector.cpp::disPred2Bbox``) 完全一致::

       p_px = (anchor_grid + raw) * stride      # anchor_grid 已含 +0.5

   而 Ultralytics 原生 ``Pose`` 是 ``(raw * 2 + anchor - 0.5) * stride`` —— 这里改成
   PicoDet 的写法, 才能让导出的 ONNX 直接喂给现有 C++ 解码器;
5. ``export=True`` 时输出 ``cls.sigmoid() + 原始回归值``(不解码), 并默认丢掉可见性
   通道, 于是通道数 = ``nc + 2 * nkpt`` —— 与 C++ 里 ``cls_layer`` / ``dis_layer``
   两张输出张量的读法一一对应。

可调参数(默认值即 "轻量 F=64, 3x3 输出块")
------------------------------------------
``PICO_HEAD_CFG`` 是模块级默认配置, 也可以在训练脚本里先改再建模型::

    from ext.pico_pose import PICO_HEAD_CFG
    PICO_HEAD_CFG.update(feat_out=96, use_se=True, head_k=3)

命令行训练时可以用环境变量, 例如::

    ULTRALYTICS_PICO_HEAD="feat_out=96,use_se=1,head_k=3" python -m ultralytics yolo pose train ...

``head_k=1`` 或 ``head_dw=0``(完整 3x3) 都可用于消融; 导出通道数只由 ``nc``/``nkpt``
决定, 与 ``head_k`` 无关。

变体
----
``PicoPoseAuxBox`` 在训练/验证时额外输出 ``boxes``(DFL 分布)，配合
``PicoAuxPoseLoss`` 复用上游 ``v8PoseLoss``(TAL + box/DFL + KeypointLoss)；
``export=True`` 时不执行辅助框分支，ONNX 输出仍只有 cls + kpts。

导入注意
--------
不要把 ``import ext.pico_pose.head`` 作为进程里的第一个相关导入: 它需要先导入
上游的 ``Pose``/``Conv``, 会和 ``ultralytics.nn`` 初始化期间对 ``ext`` 的反向
导入形成循环。请使用包级导入 ``from ext.pico_pose import PicoPoseNoBox``(会先
完成 nn 初始化), 或者先 ``import ultralytics.nn.tasks``。
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn

from ultralytics.nn.modules.block import DFL
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.head import Pose
from ultralytics.utils.tal import make_anchors

__all__ = ("PicoPoseNoBox", "PicoPoseAuxBox", "PicoDWBlock", "PicoSE", "PicoHeadBlock")


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


# 默认: 与骨干同尺度(3 级共用的中间通道数), 2 层, 开 SE, 输出前接 3x3 深度可分离空间块
PICO_HEAD_CFG = _parse_cfg_from_env(
    dict(
        feat_out=64,
        num_convs=2,
        use_se=True,
        k=5,
        head_k=3,
        head_dw=True,
        export_xy_only=True,
        # 部署导向的导出选项(默认关闭, 保持旧图 [B, C, A] + raw xy 不变):
        #   export_row_major: 输出 [B, A, C], 每行一个候选, C++ 可直接顺序读
        #   export_decode:    在图内把 kpt xy 解码成像素 ((anchor + raw) * stride)
        #                     开启后 C++ 不再需要 anchor/stride 表
        export_row_major=False,
        export_decode=False,
        prior_prob=0.01,
        aux_box=False,
    )
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


class PicoHeadBlock(nn.Module):
    """解耦输出块: ``k x k`` 空间卷积(+BN+Hardswish) -> 1x1 预测.

    ``3x3`` 让每个网格的预测参考邻域, 而不是只看当前点; 类别数 ``nc`` 较大时, 先做一次
    非线性空间/通道混合再接 1x1, 分类边界更稳。``k=1`` 时退化为单层 1x1 预测(旧行为),
    便于 A/B 消融。

    Args:
        cin: 输入通道数(即 ``feat_out``)。
        cout: 输出通道数(``nc`` 或 ``nk``)。
        k: 空间卷积核大小, 建议奇数; ``k=1`` 表示不加空间卷积。
        dw: 是否用深度可分离卷积(``groups=cin``)。``False`` 时用完整 ``kxk`` 卷积。
    """

    def __init__(self, cin: int, cout: int, k: int = 3, dw: bool = True):
        """按 ``k``/``dw`` 组装空间卷积与 1x1 输出; ``k=1`` 时空间卷积为 ``Identity``."""
        super().__init__()
        self.spatial = Conv(cin, cin, k, 1, g=cin if dw else 1, act=nn.Hardswish()) if k > 1 else nn.Identity()
        self.pred = nn.Conv2d(cin, cout, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向: 空间卷积 -> 1x1 输出."""
        return self.pred(self.spatial(x))


class PicoPoseNoBox(Pose):
    """无框轻量 Pose 头(继承 ``Pose`` 以复用 validator/predictor 的 isinstance 分支).

    与 ``Pose`` 的区别:

    - 默认不创建 ``cv2``/``dfl``；``PicoPoseAuxBox`` 子类会加一个仅训练期使用的辅助框分支;
    - 用 PicoFeat 风格的共享深度可分离特征块 + 解耦的 ``3x3 -> 1x1`` 输出块;
    - 训练时前向返回 ``{"scores", "kpts", "feats"}``(aux 子类多一个 ``boxes``), 交给
      ``PicoPoseNoBoxLoss`` / ``PicoAuxPoseLoss``;
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
        head_k: int | None = None,
        head_dw: bool | None = None,
        export_xy_only: bool | None = None,
        export_row_major: bool | None = None,
        export_decode: bool | None = None,
        prior_prob: float | None = None,
        aux_box: bool | None = None,
    ):
        """构造无框头. ``feat_out`` 等参数缺省时取 ``PICO_HEAD_CFG``."""
        nn.Module.__init__(self)
        cfg = PICO_HEAD_CFG
        self.feat_out = int(cfg["feat_out"] if feat_out is None else feat_out)
        self.num_convs = int(cfg["num_convs"] if num_convs is None else num_convs)
        self.use_se = bool(cfg["use_se"] if use_se is None else use_se)
        self.dw_k = int(cfg["k"] if k is None else k)
        self.head_k = int(cfg["head_k"] if head_k is None else head_k)
        self.head_dw = bool(cfg["head_dw"] if head_dw is None else head_dw)
        self.export_xy_only = bool(cfg["export_xy_only"] if export_xy_only is None else export_xy_only)
        self.export_row_major = bool(cfg["export_row_major"] if export_row_major is None else export_row_major)
        self.export_decode = bool(cfg["export_decode"] if export_decode is None else export_decode)
        self.prior_prob = float(cfg["prior_prob"] if prior_prob is None else prior_prob)
        self.aux_box = bool(cfg["aux_box"] if aux_box is None else aux_box)

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
            cls_head.append(PicoHeadBlock(self.feat_out, nc, self.head_k, self.head_dw))
            reg_head.append(PicoHeadBlock(self.feat_out, self.nk, self.head_k, self.head_dw))
        self.stem = nn.ModuleList(stems)
        self.se = nn.ModuleList(ses)
        self.cls_pred = nn.ModuleList(cls_head)
        self.reg_pred = nn.ModuleList(reg_head)
        if self.aux_box:
            # 仅训练时使用的辅助框分支：复用 Detect 的 DFL 输出格式，交给 v8PoseLoss/TAL。
            self.cv2 = nn.ModuleList(
                nn.Sequential(
                    Conv(self.feat_out, self.feat_out, 3),
                    nn.Conv2d(self.feat_out, 4 * self.reg_max, 1),
                )
                for _ in ch
            )
            self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    # ------------------------------------------------------------------ 前向
    def forward(
        self, x: list[torch.Tensor]
    ) -> dict[str, torch.Tensor] | torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """训练: ``dict(scores, kpts, feats)``; 验证/导出: 拼接张量."""
        bs = x[0].shape[0]
        scores, kpts, boxes = [], [], []
        aux_active = self.aux_box and (self.training or not self.export)  # 验证需要它算 val loss；导出不要
        for i in range(self.nl):
            f = self.se[i](self.stem[i](x[i]))  # 共享特征(每层只算一次)
            scores.append(self.cls_pred[i](f).view(bs, self.nc, -1))
            kpts.append(self.reg_pred[i](f).view(bs, self.nk, -1))
            if aux_active:
                boxes.append(self.cv2[i](f).view(bs, 4 * self.reg_max, -1))
        preds = {"scores": torch.cat(scores, -1), "kpts": torch.cat(kpts, -1), "feats": x}
        if aux_active:
            preds["boxes"] = torch.cat(boxes, -1)  # 给 v8PoseLoss 用；export=True 时不生成
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

    def _export_flag(self, name: str) -> bool:
        """导出开关: 环境变量显式设置优先, 其次模型属性(旧 checkpoint 可能没有该属性)."""
        raw = os.environ.get("ULTRALYTICS_PICO_HEAD", "")
        for item in raw.split(","):
            key, _, _ = item.partition("=")
            if key.strip() == name:
                return bool(PICO_HEAD_CFG.get(name, False))
        return bool(getattr(self, name, False))

    def _decode_export(self, y: torch.Tensor) -> torch.Tensor:
        """导出图内解码: ``[B, nc + nk, A]`` 的 raw kpt -> 像素坐标.

        ``y`` 前 ``nc`` 个通道保持 ``sigmoid`` 后的分类分数; 后面是 ``raw xy``
        (``export_xy_only=True``) 或 ``raw xy + vis logits``。解码约定与
        :meth:`kpts_decode` 一致: ``p = (anchor + raw) * stride``。
        """
        bs = y.shape[0]
        scores, raw = y[:, : self.nc], y[:, self.nc:]
        nxy = 2 if self.export_xy_only else self.ndim
        out_nk = self.nkpt * 2 if self.export_xy_only else self.nk
        xy = raw.view(bs, self.nkpt, nxy, -1)
        a, st = self.anchors, self.strides  # (2, A), (1, A)
        kx = (xy[:, :, 0] + a[0]) * st
        ky = (xy[:, :, 1] + a[1]) * st
        if not self.export_xy_only and self.ndim == 3:
            kp = torch.stack((kx, ky, xy[:, :, 2].sigmoid()), 2)
        else:
            kp = torch.stack((kx, ky), 2)
        return torch.cat((scores, kp.reshape(bs, out_nk, -1)), 1)

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """导出: ``nc + 2*nkpt`` 原始值; 验证: 伪框(xywh) + 分数 + 解码关键点."""
        shape = x["feats"][0].shape
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (t.transpose(0, 1) for t in make_anchors(x["feats"], self.stride, 0.5))
            self.shape = shape
        bs = x["scores"].shape[0]

        if self.export:
            raw = x["kpts"]
            if self.export_xy_only:
                raw = raw.view(bs, self.nkpt, self.ndim, -1)[:, :, :2].reshape(bs, self.nkpt * 2, -1)
            y = torch.cat((x["scores"].sigmoid(), raw), 1)  # [B, C, A]
            if self._export_flag("export_decode"):
                y = self._decode_export(y)
            if self._export_flag("export_row_major"):
                y = y.permute(0, 2, 1).contiguous()  # [B, A, C]
            return y

        # 验证: 关键点 min/max 合成伪框, 让现有 NMS / PoseValidator 不用改。
        # 注意 NMS 约定前 4 列是 **xywh**(中心+宽高), 并会再做一次 xywh->xyxy;
        # 这里若直接输出 xyxy, 框会被二次转换放大数倍, box 指标恒为 0。
        kpts = self.kpts_decode(x["kpts"]).view(bs, self.nkpt, self.ndim, -1)
        xs, ys = kpts[:, :, 0], kpts[:, :, 1]
        x1, y1 = xs.min(1).values, ys.min(1).values
        x2, y2 = xs.max(1).values, ys.max(1).values
        dbox = torch.stack(((x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1), 1)  # xywh
        return torch.cat((dbox, x["scores"].sigmoid(), kpts.reshape(bs, self.nk, -1)), 1)

    def bias_init(self):
        """类别头初始化；aux 模式对齐 Detect 的 stride 相关 bias，避免 nc=1 时 cls loss 偏大."""
        for i, head in enumerate(self.cls_pred):
            bias = head.pred.bias
            if self.aux_box:
                bias.data[: self.nc] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)
            else:
                bias.data[: self.nc] = math.log(self.prior_prob / (1 - self.prior_prob))
        for head in self.reg_pred:
            head.pred.bias.data.zero_()
        if self.aux_box:
            for head in self.cv2:
                head[-1].bias.data.fill_(2.0)

    def fuse(self) -> None:
        """输出块的 Conv+BN 由 ``BaseModel.fuse`` 统一折叠; 这里没有 one2many 分支需要重参数化."""
        return


class PicoPoseAuxBox(PicoPoseNoBox):
    """训练期带辅助框分支、验证/导出仍只输出 cls+kpts 的 Pico 头。

    训练时 ``forward`` 额外返回 ``boxes``(4*reg_max 的 DFL 分布)，配合
    ``PicoAuxPoseLoss``(复用 ``v8PoseLoss``) 使用 TAL + box/DFL + KeypointLoss；
    ``export=True`` 或 eval 时 ``_inference`` 不会引用框分支，ONNX 图里也没有 cv2/dfl。
    """

    def __init__(self, *args, **kwargs):
        """强制打开 aux_box，其余参数与父类一致。"""
        kwargs["aux_box"] = True
        super().__init__(*args, **kwargs)
