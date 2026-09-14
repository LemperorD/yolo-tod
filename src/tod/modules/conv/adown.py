"""ADown —— 双分支自适应下采样（YOLOv9 提出，SPAE-YOLOv8 用作 ADown 模块）。

为什么对**小目标**有用（SPAE-YOLOv8 §3.3 原文）：
    "repeated downsampling may weaken the boundary and texture cues of tiny targets...
     Instead of relying on a single stride-2 convolution, ADown adopts a dual-branch
     structure to preserve complementary feature responses while reducing computational cost."

结构（与原文 §3.3 描述逐条对应）::

    输入 x
      ├─ avg_pool2d(k=2, s=1, p=0)          # 先做平均池化，保留局部上下文
      ├─ chunk(2, dim=1) → x1, x2
      ├─ x1 ─ Conv3x3(s=2) ──┐              # 分支一：卷积下采样 + 特征提取
      └─ x2 ─ max_pool2d(3,2,1) ─ Conv1x1 ──┘  # 分支二：最大池化突出显著响应 + 通道精炼
      → cat([x1, x2], dim=1)

两个分支空间分辨率一致（均为输入的 1/2），通道各占 c2/2，拼接后正好 c2。
相对单个 stride-2 卷积：平滑（avg）+ 显著性增强（max）+ 可学习表示（conv）三者互补，
并且在同等输出通道下更省算力。

YAML 用法（ultralytics 风格）::

    [-1, 1, ADown, [256]]      # c1 由上一层的输出通道自动推断，c2=256

注意：``c1``（输入通道）必须为偶数，因为要沿通道一分为二。
"""

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
        self.cv1 = conv(c1 // 2, self.c, 3, 2, 1)   # 分支一：3x3, stride 2
        self.cv2 = conv(c1 // 2, self.c, 1, 1, 0)   # 分支二：1x1 通道精炼

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # k=2, s=1, p=0 的 avg_pool 使输出恰好为 H/2、W/2（原文：先平均池化保留上下文）
        x = F.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)                            # 3x3/s2：卷积下采样
        x2 = self.cv2(F.max_pool2d(x2, 3, 2, 1))     # max-pool 突出显著响应后 1x1 精炼
        return torch.cat((x1, x2), 1)


def _base_conv():
    """延迟取框架的 Conv，避免在 import 阶段硬依赖 ultralytics。"""
    from tod.compat import base_conv

    return base_conv()
