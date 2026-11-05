"""DualAttention —— 通道 ⊗ 空间 双注意力（SDD-YOLO §4.5，式 (4)）。"""

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
    """式 (4) 的落地实现：``x * σ(W_c·GAP(x)) * σ(Conv7x7([Avg;Max](x)))``。"""

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
        self.channel = nn.Sequential(
            nn.Conv2d(c1, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, c1, 1, bias=True),
        )
        self.spatial = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        ch = self.channel(x.mean(dim=(2, 3), keepdim=True)).sigmoid()
        pooled = torch.cat(
            (x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)), dim=1
        )
        sp = self.spatial(pooled).sigmoid()
        return x * ch * sp


__all__ = ["DualAttention"]
