"""YOLO-ULM blocks (CVPR 2026): D3C2f and RepDown.

Independent re-implementation following the YOLO-ULM paper (Han et al.,
CVPR 2026, Sec. 3.1/3.2), adapted from the local armor-ulm reference.
D3C2f replaces C3k2 in the trunk; RepDown replaces stride-2 Conv.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv

__all__ = ("DSConv", "D3Block", "D3C2f", "RepDown")


class DSConv(nn.Module):
    """1x1 pointwise followed by a kxk depthwise convolution."""

    def __init__(self, c1: int, c2: int, k: int = 3):
        super().__init__()
        self.pw = Conv(c1, c2, 1, 1)
        self.dw = nn.Conv2d(c2, c2, k, 1, k // 2, groups=c2, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.dw(self.pw(x))))


class D3Block(nn.Module):
    """DSConv(3) -> DWConv(7) -> DSConv(3) with a residual."""

    def __init__(self, c1: int, c2: int, e: float = 1.0):
        super().__init__()
        c_ = int(c2 * e)
        self.ds1 = DSConv(c1, c_)
        self.dw7 = nn.Conv2d(c_, c_, 7, 1, 3, groups=c_, bias=False)
        self.bn7 = nn.BatchNorm2d(c_)
        self.act7 = nn.SiLU()
        self.ds2 = DSConv(c_, c2)
        self.short = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()

    def forward(self, x):
        y = self.ds2(self.act7(self.bn7(self.dw7(self.ds1(x)))))
        return y + self.short(x)


class D3C2f(nn.Module):
    """C2f-style aggregation with D3Block instead of Bottleneck/C3k.

    ``shortcut`` and ``g`` are accepted for cfg compatibility; the D3Block
    always carries its own residual, so they do not change the graph.
    """

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, e: float = 0.5, g: int = 1):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(D3Block(self.c, self.c) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class RepDown(nn.Module):
    """Parallel 7x7/3x3 depthwise stride-2 downsampling.

    Both branches stay in the inference graph; ``fuse`` only folds the
    pointwise Conv+BN pairs and does not merge the depthwise branches.
    """

    def __init__(self, c1: int, c2: int):
        super().__init__()
        c_ = max(c2 // 2, 1)
        g = math.gcd(c1, c_)
        self.dw7 = nn.Conv2d(c1, c_, 7, 2, 3, groups=g, bias=False)
        self.bn7 = nn.BatchNorm2d(c_)
        self.pw7 = Conv(c_, c_, 1, 1)
        self.dw3 = nn.Conv2d(c1, c_, 3, 2, 1, groups=g, bias=False)
        self.bn3 = nn.BatchNorm2d(c_)
        self.pw3 = Conv(c_, c_, 1, 1)
        self.fused = False

    def forward(self, x):
        a = self.pw7(self.bn7(self.dw7(x)))
        b = self.pw3(self.bn3(self.dw3(x)))
        return torch.cat((a, b), 1)

    def fuse(self, verbose: bool = False):
        """Fold the pointwise Conv+BN pairs; idempotent and safe after ``BaseModel.fuse``."""
        if self.fused:
            return self
        for pw in (self.pw7, self.pw3):
            bn = getattr(pw, "bn", None)
            if isinstance(bn, nn.BatchNorm2d):
                w = bn.weight / (bn.running_var + bn.eps) ** 0.5
                b = bn.bias - bn.running_mean * w
                pw.conv.weight.data *= w.view(-1, 1, 1, 1)
                pw.conv.bias = nn.Parameter(b)
                pw.bn = nn.Identity()
                pw.forward = pw.forward_fuse
        self.fused = True
        return self
