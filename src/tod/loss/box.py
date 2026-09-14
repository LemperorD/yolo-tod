"""边界框回归损失集合（EP7）。

本文件提供与框架 ``bbox_iou`` **同语义**的相似度函数：返回值越大越好，
训练损失 = ``1 - similarity``。这样可以直接替换框架默认的 CIoU 项，
而不必改动 DFL 与分类分支。

已实现：
  * ``siou`` —— SPAE-YOLOv8 采用的回归损失（角度 + 距离 + 形状三部分代价）
"""

from __future__ import annotations

import math
from typing import Callable

import torch

from tod.registry import register


@register(
    name="siou",
    ep="EP7",
    paper="SIoU Loss: More Powerful Learning for Bounding Box Regression",
    url="https://arxiv.org/abs/2205.12740",
    year=2022,
    license="MIT（参考实现）；本文件为按论文公式独立重写",
    cost="计算量与 CIoU 同量级；无额外参数",
    notes="SPAE-YOLOv8 §3.1 用其替换 YOLOv8 默认 CIoU，提升小目标定位精度",
    aliases=("SIoU", "siou_loss"),
)
def siou(
    box1: torch.Tensor,
    box2: torch.Tensor,
    xywh: bool = True,
    theta: float = 4.0,
    eps: float = 1e-7,
) -> torch.Tensor:
    """SIoU 相似度（越大越好）。

    与 SPAE-YOLOv8 §3.1 的公式（1)–(7) 逐项对应：

    ==================  ==========================================================
    角度代价 Λ          ``Λ = 1 - 2·sin²(arcsin(sinα) - π/4)``，等价于 ``cos(2·arcsin(sinα))``；
                        sinα 取中心连线与 x/y 轴夹角中较小者的正弦
    距离代价 Δ          ``Δ = Σ_t (1 - exp(-(2-Λ)·P_t))``，P_x=(Δcx/Cw)²，P_y=(Δcy/Ch)²
    形状代价 Ω          ``Ω = Σ_t (1 - exp(-|w-w_gt|/max(w,w_gt)))^θ``，θ 默认 4
    最终                ``SIoU = IoU - (Δ + Ω) / 2``
    ==================  ==========================================================

    Args:
        box1/box2: ``(..., 4)``；``xywh=True`` 时为 (cx, cy, w, h)，否则为 (x1, y1, x2, y2)。
        theta: 形状代价的注意力权重 θ。
    """
    # ---- 解包为角点坐标 ----
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # ---- IoU ----
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * (
        torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)
    ).clamp(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    # ---- 最小外接矩形 Cw / Ch（公式 3、4 的分母）----
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1) + eps
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1) + eps

    # ---- 中心偏移与 σ ----
    s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5
    s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5
    sigma = torch.pow(s_cw**2 + s_ch**2, 0.5) + eps

    # ---- 角度代价 Λ ----
    sin_alpha_1 = torch.abs(s_cw) / sigma
    sin_alpha_2 = torch.abs(s_ch) / sigma
    threshold = math.sqrt(2) / 2          # 45°，取与坐标轴夹角更小的那一支
    sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
    angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)

    # ---- 距离代价 Δ ----
    rho_x = (s_cw / cw) ** 2
    rho_y = (s_ch / ch) ** 2
    gamma = angle_cost - 2
    distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)

    # ---- 形状代价 Ω ----
    omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
    omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
    shape_cost = torch.pow(1 - torch.exp(-omiga_w), theta) + torch.pow(
        1 - torch.exp(-omiga_h), theta
    )

    return iou - 0.5 * (distance_cost + shape_cost) + eps


#: 名字 → 相似度函数。新增损失请同时在这里登记，训练准则按名查找。
BOX_LOSSES: dict[str, Callable[..., torch.Tensor]] = {
    "siou": siou,
}


def box_loss(name: str) -> Callable[..., torch.Tensor]:
    """按名字取回归损失函数。"""
    key = name.lower()
    if key not in BOX_LOSSES:
        raise KeyError(f"未实现的回归损失 {name!r}；已有：{sorted(BOX_LOSSES)}")
    return BOX_LOSSES[key]
