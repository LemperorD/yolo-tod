"""DualAttention —— 通道 ⊗ 空间 双注意力（SDD-YOLO §4.5，式 (4)）。

论文：Pengyu Chen, Haotian Sa, Yiwei Hu, Yuhan Cheng, Junbo Wang,
"SDD-YOLO: A Small-Target Detection Framework for Ground-to-Air Anti-UAV
Surveillance with Edge-Efficient Deployment", arXiv:2603.25218（2026-08-10）。

原文（§4.5 Dual Attention Mechanism）：

    "To suppress false positives from aerial clutter (birds, clouds, building edges),
     we embed dual attention modules between the backbone and detection heads:
      * Spatial attention highlights high-probability motion regions within the wide
        aerial field of view, suppressing static background activation.
      * Channel attention re-weights feature channels to amplify UAV-discriminative
        frequency responses and dampen background-heavy channels."

并给出唯一的公式（式 (4)）::

    A = σ(W_c · GAP(F)) ⊗ σ(Conv_{7×7}([AvgPool; MaxPool](F)))

    A ∈ R^{C×H×W}，σ 为 sigmoid，GAP 为全局平均池化，⊗ 为逐元素相乘。

【与论文正文不一致之处（已记录，见 variants/SDD-YOLO26n/paper-notes.md）】
    正文说空间分支用于"运动区域（motion regions）"，但式 (4) 的空间分支是
    ``[AvgPool; MaxPool] + Conv7×7`` —— 这是 CBAM 式的**通道池化对比**，
    与运动/帧差无关（单帧推理也没有运动信息可用）。
    本实现严格按**公式**落地；"运动"只能理解为对高对比度/显著区域的定性描述。
    另外论文未给出 W_c 的结构与压缩比 r，本实现按 CBAM 惯例取 r=16（可配置）。

小目标意义：地面-空中场景的假阳性主要来自天空纹理、云层边缘、飞鸟；把注意力放在
"通道响应 + 空间显著位置"的联合权重上，可在不增大特征图分辨率的前提下压低背景激活。

用法（两种，均为**通道不变**的原位增强）::

    # 1) 模型图手术（本库 SDD 变体走这条路，见 tod/engine/surgery.py）
    #    把 neck 输出接到检测头之前包一层：x -> x * A(x)
    # 2) YAML 直接插节点（c2 省略或等于 c1；注意力不改变通道数）
    [-1, 1, DualAttention, [256]]
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tod.registry import register


@register(
    name="DualAttention",
    ep="EP4",
    paper="SDD-YOLO: A Small-Target Detection Framework for Ground-to-Air Anti-UAV "
          "Surveillance with Edge-Efficient Deployment (arXiv:2603.25218) §4.5 式 (4)",
    url="https://arxiv.org/abs/2603.25218",
    year=2026,
    license="论文为 arXiv 预印本（未声明代码许可证）；本文件为按式 (4) 独立实现",
    cost="额外参数约 2C²/r + 98（r=16 时 C=64 → 约 0.6k），几乎不影响延迟；"
         "空间分支需一次 7×7 卷积",
    notes="通道注意力（GAP→W_c→σ）与空间注意力（[AvgMax]→Conv7×7→σ）逐元素相乘后原位加权；"
          "通道数不变，可插入任意 neck→head 连接处",
    aliases=("DAM", "DualAttn", "SDD_Attention", "dual_attention"),
)
class DualAttention(nn.Module):
    """式 (4) 的落地实现：``x * σ(W_c·GAP(x)) * σ(Conv7x7([Avg;Max](x)))``。

    Args:
        c1: 输入通道数（也是输出通道数；注意力模块不改变通道）。
        c2: 仅用于兼容 ultralytics YAML 的 ``(c1, c2)`` 调用约定，
            必须为 ``None`` 或等于 ``c1``。
        reduction: 通道分支的压缩比 r（论文未给出，默认 16，与 CBAM 一致）。
        kernel_size: 空间分支卷积核 k（论文为 7）。
    """

    def __init__(self, c1: int, c2: int | None = None, reduction: int = 16,
                 kernel_size: int = 7):
        super().__init__()
        if c2 is not None and int(c2) != int(c1):
            raise ValueError(
                f"DualAttention 不改变通道数，要求 c2 == c1，实际 c1={c1}, c2={c2}。"
            )
        if c1 < 2:
            raise ValueError(f"DualAttention 需要至少 2 个通道，实际 c1={c1}。")
        hidden = max(1, int(c1) // int(reduction))
        # 通道分支：σ(W_c · GAP(F)) —— W_c 用两层 1x1 卷积（bottleneck）实现
        self.channel = nn.Sequential(
            nn.Conv2d(c1, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, c1, 1, bias=True),
        )
        # 空间分支：σ(Conv_{k×k}([AvgPool; MaxPool](F)))
        self.spatial = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        # --- 通道注意力（GAP → W_c → sigmoid）---
        ch = self.channel(x.mean(dim=(2, 3), keepdim=True)).sigmoid()
        # --- 空间注意力（通道维 avg/max 拼接 → k×k 卷积 → sigmoid）---
        pooled = torch.cat(
            (x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)), dim=1
        )
        sp = self.spatial(pooled).sigmoid()
        return x * ch * sp


__all__ = ["DualAttention"]
