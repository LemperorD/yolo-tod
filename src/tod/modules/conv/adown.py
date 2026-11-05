"""ADown —— 双分支自适应下采样（YOLOv9 提出，SPAE-YOLOv8 用作 ADown 模块）。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tod.registry import register


@register(
    ep="EP1",
    paper="YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information "
          "(ADown)；SPAE-YOLOv8 §3.3 将其用于小目标检测",
    url="https://arxiv.org/abs/2402.13616",
    year=2024,
    license="原实现 GPL-3.0；本文件为按论文描述独立重写",
    cost="相同输出通道下 FLOPs 低于 stride-2 3x3 卷积；参数量略增（多一个 1x1 分支）",
    notes="替换主干中普通 stride-2 下采样，减少小目标在下采样中的细节丢失；"
          "输入通道必须为偶数",
    aliases=("Adown", "adown"),
)
class ADown(nn.Module):
    """自适应下采样：avg-pool → 双分支（Conv3x3/s2 与 MaxPool+Conv1x1）→ concat。"""

    def __init__(self, c1: int, c2: int):
        super().__init__()
        if c1 % 2 != 0:
            raise ValueError(
                f"ADown 要求输入通道为偶数（需沿通道一分为二），实际 c1={c1}。"
            )
        self.c = c2 // 2
        conv = _base_conv()
        self.cv1 = conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = self.cv2(F.max_pool2d(x2, 3, 2, 1))
        return torch.cat((x1, x2), 1)


def _base_conv():
    """延迟取框架的 Conv，避免在 import 阶段硬依赖 ultralytics。"""
    from tod.compat import base_conv

    return base_conv()
