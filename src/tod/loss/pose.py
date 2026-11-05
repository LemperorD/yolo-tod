"""EP7 姿态关键点损失（Pose / Keypoint）—— 小目标口径的 OKS 项。"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from tod.compat import CompatError
from tod.registry import register

COCO_SIGMA_17 = (
    0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
    0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089,
)
COCO_SIGMA_MEAN = sum(COCO_SIGMA_17) / len(COCO_SIGMA_17)
KPT_MASK_RULE = "gt_kpt[..., 2] != 0（与框架一致：vis=0 视为未标注）"


def sigma_tensor(kpt_shape: Any, strategy: str = "auto",
                 device: Any = None) -> tuple[torch.Tensor, str]:
    """按策略生成 OKS sigma，返回 ``(sigma, 说明)``。"""
    nkpt = int(kpt_shape[0])
    if strategy == "uniform":
        sigma = torch.full((max(nkpt, 1),), COCO_SIGMA_MEAN, dtype=torch.float32)
        note = (f"uniform（所有 {nkpt} 点同 σ={COCO_SIGMA_MEAN:.4f}；"
                "无「哪个点更难」先验时的诚实选择）")
    elif strategy in ("person", "framework") and nkpt == len(COCO_SIGMA_17):
        sigma = torch.tensor(COCO_SIGMA_17, dtype=torch.float32)
        note = "person（COCO 17 点标准 sigma，未缩放 → 与框架逐元素一致）"
    elif strategy == "balanced":
        sigma = _normalized(_auto_sigma(nkpt) ** 0.5)
        note = (f"balanced（auto sigma 压平后归一到均值 1，n={nkpt}）"
                "[!] 改变了 OKS 项的绝对尺度，与框架/COCO 数值不可直接比较")
    elif nkpt == len(COCO_SIGMA_17):
        sigma = torch.tensor(COCO_SIGMA_17, dtype=torch.float32)
        note = "auto（17 点：沿用 COCO 标准 sigma，未缩放）"
    else:
        sigma = _auto_sigma(nkpt)
        note = (f"auto（n={nkpt}：本库几何自适应 sigma，缩放到 COCO sigma 均值 "
                f"{COCO_SIGMA_MEAN:.4f} 量级；非论文参数）")
    if device is not None:
        sigma = sigma.to(device)
    return sigma, note


def _auto_sigma(nkpt: int) -> torch.Tensor:
    """为任意关键点数生成形状自适应的 sigma（**本库的推断，不是论文参数**）。"""
    if nkpt <= 1:
        return torch.ones(max(nkpt, 1), dtype=torch.float32) * COCO_SIGMA_MEAN
    idx = torch.linspace(-1.0, 1.0, nkpt, dtype=torch.float32)
    weights = 0.05 + 0.95 * (1.0 - torch.cos(math.pi * idx)) / 2.0
    return _scaled(weights)


def _scaled(values: torch.Tensor) -> torch.Tensor:
    """把一组相对权重缩放到 COCO sigma 的**均值**量级（保持 OKS 损失尺度可比）。"""
    mean = values.mean()
    if not torch.isfinite(mean) or mean <= 0:
        return values * COCO_SIGMA_MEAN
    return values * (COCO_SIGMA_MEAN / mean)


def _normalized(values: torch.Tensor) -> torch.Tensor:
    """把一组权重归一到均值 1（``balanced`` 策略用：只改相对权重，尺度另行说明）。"""
    mean = values.mean()
    if not torch.isfinite(mean) or mean <= 0:
        return values
    return values / mean


@register(
    ep="EP7",
    paper="OKS（Object Keypoint Similarity，COCO）+ 本库的 sigma 策略显式化",
    url="https://cocodataset.org/#keypoints-eval",
    year=2020,
    license="OKS 公式为公开评测口径；本文件为独立实现",
    cost="与框架 KeypointLoss 同量级（逐元素 exp 运算，参数量 0）；"
         "不改变关键点分支的显存占用",
    notes="替换框架 KeypointLoss：sigma 策略 person/auto/balanced + σ 下限 + 可见性开关；"
          "默认参数下与框架逐元素一致（回归对照见 tests/test_pose.py）",
    aliases=("tiny_pose", "TinyPose", "PoseLoss"),
)
class TinyPoseLoss(nn.Module):
    """关键点 OKS 损失：可配置 sigma 策略与可见性加权（替换框架 ``KeypointLoss``）。"""

    def __init__(self, sigmas: torch.Tensor, strategy: str = "auto",
                 min_sigma: float = 1e-3, eps: float = 1e-9):
        super().__init__()
        self.strategy = strategy
        self.min_sigma = float(min_sigma)
        self.eps = float(eps)
        sigma = torch.as_tensor(sigmas, dtype=torch.float32).flatten().clamp_min(self.min_sigma)
        self.register_buffer("sigmas", sigma)

    def forward(self, pred_kpts: torch.Tensor, gt_kpts: torch.Tensor,
                kpt_mask: torch.Tensor, area: torch.Tensor) -> torch.Tensor:
        """与框架同式的 OKS 损失。"""
        d = ((pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2)
             + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2))
        factor = kpt_mask.shape[1] / (kpt_mask.sum(dim=1) + self.eps)
        sigma = self.sigmas.to(d.device).view(1, -1)
        e = d / ((2 * sigma).pow(2) * (area + self.eps) * 2)
        mask = kpt_mask.to(d.dtype) if kpt_mask.dtype != d.dtype else kpt_mask
        return (factor.view(-1, 1) * ((1.0 - torch.exp(-e)) * mask)).mean()

    def describe(self) -> str:
        sig = ", ".join(f"{v:.3f}" for v in self.sigmas.tolist()[:6])
        return (f"TinyPoseLoss(strategy={self.strategy}, n_kpt={self.sigmas.numel()}, "
                f"min_sigma={self.min_sigma:g}, σ[:6]=[{sig}…])")


def pose_criteria(criterion: Any) -> list[Any]:
    """列出准则里所有持有 ``keypoint_loss`` 的子准则（姿态准则判别器）。"""
    if hasattr(criterion, "keypoint_loss"):
        return [criterion]
    # 关键点损失也要覆盖 O2M/O2O，避免只对一个分支生效。
    subs = [c for name in ("one2many", "one2one")
            if hasattr(criterion, name) and hasattr(getattr(criterion, name), "keypoint_loss")
            for c in [getattr(criterion, name)]]
    if not subs:
        raise CompatError(
            "准则里找不到 keypoint_loss —— 模型可能不是姿态模型（检测头不是 Pose/Pose26）。"
            "请确认变体 spec 的 task 是 'pose' 且模型图以 *-pose.yaml 为底座。"
        )
    return subs


def build_pose_loss(criterion: Any, *, enabled: bool = True, strategy: str = "auto",
                    min_sigma: float = 1e-3, **_: Any) -> list[str]:
    """把 ``TinyPoseLoss`` 接进框架姿态准则；返回可读改动描述列表。"""
    subs = pose_criteria(criterion)
    if not enabled:
        return ["pose: 保留框架原生 KeypointLoss（EP7.pose=False，消融列）"]

    report: list[str] = []
    shared: TinyPoseLoss | None = None
    for sub in subs:
        base = sub.keypoint_loss
        shape = getattr(sub, "kpt_shape", None) or [len(getattr(base, "sigmas", [])) or 17, 3]
        if shared is None:
            device = getattr(getattr(base, "sigmas", None), "device", None)
            sigma, note = sigma_tensor(shape, strategy=strategy, device=device)
            shared = TinyPoseLoss(sigma, strategy=strategy, min_sigma=min_sigma)
            report.append(f"pose: KeypointLoss -> TinyPoseLoss[{note}]")
            report.append(f"       可见性掩码规则：{KPT_MASK_RULE}")
        sub.keypoint_loss = shared
    return report


__all__ = ["COCO_SIGMA_17", "KPT_MASK_RULE", "TinyPoseLoss", "build_pose_loss",
           "pose_criteria", "sigma_tensor"]
