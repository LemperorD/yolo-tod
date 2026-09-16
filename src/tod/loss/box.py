"""边界框回归损失集合（EP7）。

本文件提供与框架 ``bbox_iou`` **同语义**的相似度函数：返回值越大越好，
训练损失 = ``1 - similarity``。这样可以直接替换框架默认的 CIoU 项，
而不必改动 DFL 与分类分支。

已实现：
  * ``siou``  —— SPAE-YOLOv8 采用的回归损失（角度 + 距离 + 形状三部分代价）
  * ``wiou``  —— SDD-YOLO §4.3 式 (3) 采用的 Wise-IoU v3（动态非单调聚焦）
"""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn

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


@register(
    name="wiou",
    ep="EP7",
    paper="Wise-IoU: Bounding Box Regression Loss with Dynamic Focusing Mechanism "
          "(SDD-YOLO §4.3 式 (3) 用它替换 DFL 分支的回归项)",
    url="https://arxiv.org/abs/2301.10051",
    year=2023,
    license="官方实现 MIT；本文件为按论文公式独立重写",
    cost="与 IoU 同量级（多一次 exp 与跨 batch 标量均值）；无额外参数、"
         "只多 1 个不参与梯度的滑动均值 buffer",
    notes="v3 的非单调聚焦系数 r 会同时压低极易/极难样本的权重；"
          "对小目标（IoU 抖动大）比 CIoU 稳",
    aliases=("WiseIoU", "WIoU", "wiou_v3", "WiseIoUv3"),
)
class WiseIoU(nn.Module):
    """Wise-IoU v1/v2/v3 的**相似度**形式（返回值越大越好，loss = 1 - 返回值）。

    论文（arXiv 2301.10051）把 WIoU 写成损失，这里按本库约定取 ``1 - L``：

    ==========  ====================================================================
    v1           ``L = R_WIoU · (1 - IoU)``，``R_WIoU = exp(Δ² / (Wg² + Hg²))``
                 距离注意力系数，**detach**（不参与反传）；Δ 为中心点欧氏距离，
                 Wg/Hg 为最小外接框宽高
    v2           在 v1 上乘离群度 ``β = L*_IoU / L̄_IoU``（均 detach），
                 抑制低质量样本的梯度
    v3           ``r = β / (δ·α^(β-δ))``，非单调聚焦系数（α=1.9、δ=3.0）；
                 β=δ 时 r 最大，极易与极难样本都被压低
    ==========  ====================================================================

    ``L̄_IoU``（``iou_mean``）是跨 batch 的滑动均值（momentum=0.99，官方实现同款），
    首个 batch 用 1.0 初始化 —— 因此**前若干步的 β 不可信**，属于该方法固有行为。

    Args:
        variant: 1 / 2 / 3。
        alpha, delta: v3 的非单调聚焦超参（论文取 1.9 / 3.0）。
        momentum: ``iou_mean`` 的滑动平均系数。
    """

    def __init__(self, variant: int = 3, alpha: float = 1.9, delta: float = 3.0,
                 momentum: float = 0.99, eps: float = 1e-7):
        super().__init__()
        if variant not in (1, 2, 3):
            raise ValueError(f"WiseIoU 只支持 variant ∈ {{1,2,3}}，实际 {variant}。")
        self.variant = int(variant)
        self.alpha = float(alpha)
        self.delta = float(delta)
        self.momentum = float(momentum)
        self.eps = float(eps)
        # 注册成 buffer：换设备/存 checkpoint 时一起走，且不参与梯度
        self.register_buffer("iou_mean", torch.tensor(1.0))

    def forward(self, box1: torch.Tensor, box2: torch.Tensor, xywh: bool = True,
                theta: float | None = None, **_: object) -> torch.Tensor:
        """返回 ``1 - L_WIoU``（theta 仅为与 siou 等损失统一调用签名而存在，此处忽略）。"""
        if xywh:
            (cx1, cy1, w1, h1), (cx2, cy2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
            b1x1, b1x2 = cx1 - w1 / 2, cx1 + w1 / 2
            b1y1, b1y2 = cy1 - h1 / 2, cy1 + h1 / 2
            b2x1, b2x2 = cx2 - w2 / 2, cx2 + w2 / 2
            b2y1, b2y2 = cy2 - h2 / 2, cy2 + h2 / 2
        else:
            b1x1, b1y1, b1x2, b1y2 = box1.chunk(4, -1)
            b2x1, b2y1, b2x2, b2y2 = box2.chunk(4, -1)
            cx1, cy1 = (b1x1 + b1x2) / 2, (b1y1 + b1y2) / 2
            cx2, cy2 = (b2x1 + b2x2) / 2, (b2y1 + b2y2) / 2

        inter = (torch.min(b1x2, b2x2) - torch.max(b1x1, b2x1)).clamp(0) * (
            torch.min(b1y2, b2y2) - torch.max(b1y1, b2y1)
        ).clamp(0)
        union = ((b1x2 - b1x1) * (b1y2 - b1y1) + (b2x2 - b2x1) * (b2y2 - b2y1)
                 - inter + self.eps)
        loss = 1.0 - inter / union                        # L_IoU

        # --- R_WIoU：距离注意力（detach，只调权重不改梯度方向）---
        wg = torch.max(b1x2, b2x2) - torch.min(b1x1, b2x1)
        hg = torch.max(b1y2, b2y2) - torch.min(b1y1, b2y1)
        dist2 = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2
        r_w = torch.exp(dist2 / (wg.pow(2) + hg.pow(2) + self.eps)).detach()
        loss = r_w * loss                                  # L_WIoUv1

        if self.variant >= 2:
            mean = self._update_mean(loss)
            beta = (loss.detach() / mean)                   # 离群度 β
            if self.variant == 2:
                loss = beta * loss
            else:
                r = beta / (self.delta * self.alpha ** (beta - self.delta))
                loss = r.detach() * loss
        return 1.0 - loss

    @torch.no_grad()
    def _update_mean(self, loss: torch.Tensor) -> torch.Tensor:
        """用当前 batch 的均值更新滑动均值 ``L̄_IoU``，返回更新后的值。"""
        self.iou_mean.mul_(self.momentum).add_(
            loss.detach().mean().to(self.iou_mean) * (1.0 - self.momentum)
        )
        return self.iou_mean.clamp(min=self.eps)


#: 名字 → 相似度函数/可调用对象（类会被实例化，见 ``box_loss``）。
BOX_LOSSES: dict[str, Callable[..., torch.Tensor] | type[nn.Module]] = {
    "siou": siou,
    "wiou": WiseIoU,
}


def box_loss(name: str, **kwargs: object):
    """按名字取回归损失。

    Returns:
        可调用对象：``fn(box1, box2, xywh=False, **kw) -> similarity``。
        注册为类的损失（如 ``WiseIoU``）会在这里实例化，以便持有跨 batch 状态。
    """
    key = name.lower()
    if key not in BOX_LOSSES:
        raise KeyError(f"未实现的回归损失 {name!r}；已有：{sorted(BOX_LOSSES)}")
    entry = BOX_LOSSES[key]
    return entry(**kwargs) if isinstance(entry, type) else entry

