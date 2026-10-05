"""EP7 姿态关键点损失（Pose / Keypoint）—— 小目标口径的 OKS 项。

**这个文件解决什么问题**：ultralytics 的 ``KeypointLoss``（框架内部，见
``ultralytics/utils/loss.py``）用的就是 COCO 的 OKS 形式

    e   = d / (2 · σ² · area · 2)          d = 预测点与 GT 点的平方像素距离
    L   = (1/N_vis) · Σ (1 − exp(−e)) · mask

其中 ``area`` 是**归一化**（imgsz 尺度）的 GT 框面积，``σ`` 是 COCO 的 17 点人体
sigma（鼻子 0.026 … 髋 0.107）。这套参数有一个对 tiny 目标很要命的内禀性质：

    σ² · area 越小 → 同样的**像素**误差造成的 e 越大 → 损失越陡。

极端例子：一个 6×6 px 的小人，area ≈ (6/640)² ≈ 8.8e-5；鼻子处 ``2·σ²·area·2``
≈ 2.4e-7，也就是说 lr 稍有波动，1 px 的偏差就会把该点的损失推到饱和区（e ≈ 4e6）。
换句话说，**框架的 OKS 项在 tiny 目标上几乎是"全或无"的门**，梯度信息很弱。
这是本库必须在 EP7 显式暴露的东西，而不是让变体默默继承。

本模块提供三件事：

1. ``TinyPoseLoss`` —— 与框架 ``KeypointLoss`` **同签名**的替换件，可配置
   sigma 策略与数值下限，默认行为与框架逐元素一致（可回归对照，见 tests）；
2. ``build_pose_loss`` —— 把替换件接进框架的准则（单分支 ``v8PoseLoss``，以及
   YOLO26 ``end2end`` 下 ``E2ELoss`` 持有的 O2M/O2O **两套** ``PoseLoss26``）；
3. 可见性（``kobj``）与定位（``pose``）两项增益的**显式化**：框架把它们放在
   ``model.args`` 里当超参，本库在变体 spec 里也写一遍，便于消融与自检。

【刻意不做的事】不改动框的 IoU/DFL 项：那是 EP7 的 ``tod.loss.box`` 与
``tod.loss.criterion`` 负责的，姿态只加一条**并列**的关键点支路。因此
``box="wiou"`` + ``pose="oks"`` 可以同时生效，消融也互不干扰。
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from tod.compat import CompatError
from tod.registry import register

#: COCO 17 点人体姿态的 OKS sigma（顺序与 COCO keypoints 一致）。
COCO_SIGMA_17 = (
    0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
    0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089,
)
#: COCO 17 点 sigma 的均值。非 17 点数据集没有公认 sigma，用它把自研 sigma
#: 缩放到**同一量级**，否则 OKS 项与框架/COCO 的数值不可比（实测：把 COCO sigma
#: 均值归一到 1 会让损失从 0.369 掉到 0.005，即 73 倍）。
COCO_SIGMA_MEAN = sum(COCO_SIGMA_17) / len(COCO_SIGMA_17)
#: COCO 标注里 vis=0 表示"该点未标注"；框架用 ``!= 0`` 当有效掩码，本库沿用。
KPT_MASK_RULE = "gt_kpt[..., 2] != 0（与框架一致：vis=0 视为未标注）"


def sigma_tensor(kpt_shape: Any, strategy: str = "auto",
                 device: Any = None) -> tuple[torch.Tensor, str]:
    """按策略生成 OKS sigma，返回 ``(sigma, 说明)``。

    **sigma 的绝对尺度非常重要**：OKS 损失是 ``1 − exp(−e)``，``e`` 正比于
    ``1/σ²``。把一组 sigma 整体缩放 k 倍，损失曲线就整体平移（不是等价的
    "相对权重调整"）。因此本函数只在**确有理由**时改变尺度，并在返回值里写明。

    Args:
        kpt_shape: ``(n_kpts, n_dims)``，来自模型头的 ``kpt_shape``。
        strategy:
            * ``"person"`` / ``"framework"`` —— 17 点人体直接用 **COCO 原始 sigma**
              （不做任何缩放，与框架 ``KeypointLoss`` 逐元素一致，本库有回归对照）；
            * ``"auto"`` —— 17 点同 ``person``；其它点数用**形状自适应的几何衰减**
              sigma（见 ``_auto_sigma``），并缩放到 COCO sigma 的均值量级；
            * ``"uniform"`` —— 所有点同 σ（见下）。
            * ``"balanced"`` —— 在 auto 的基础上把 sigma 压平（推向均值 1）并归一化，
              减小"末端点权重 5 倍于髋部"的悬殊。**这是刻意改变损失尺度的激进选项**，
              只在明确要"让离群点不再主导 tiny 目标的关键点损失"时使用；
            * ``"uniform"`` —— 所有点用同一个 sigma。**没有"某个点更难"的先验时用它**：
              机体关键点（四个电机/机臂）通常是同质的，几何曲线会把首尾点无依据地
              放松/收紧；uniform 的诚实表述是"我们不知道哪个点更难，所以一视同仁"。
              尺度同样对齐到 COCO sigma 均值，便于与其它策略比较。
        device: 目标设备。

    Returns:
        ``(sigma, 可读说明)``。
    """
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
                "⚠️ 改变了 OKS 项的绝对尺度，与框架/COCO 数值不可直接比较")
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
    """为任意关键点数生成形状自适应的 sigma（**本库的推断，不是论文参数**）。

    构造原则（只在 n ≠ 17 时使用，因为 17 点有 COCO 标准值可依）：
      * 用一条"中心小、末端大"的对称曲线：关键点序号越靠两端，允许的定位误差越大
        （与 COCO 的 17 点分布同形：鼻子/眼/耳最小，腕/踝/髋最大）；
      * 均值缩放到 ``COCO_SIGMA_MEAN``，保证 OKS 项与该框架默认值同量级。

    **明确标注为本库推断**：非 17 点的姿态数据集（车辆角点、机械臂关节、细胞极性…）
    没有公认 sigma，本函数只是避免"σ=1/nkpt 全等权"这种把大误差点淹没的退化做法。
    变体卡片里必须写清用的是哪个策略。
    """
    if nkpt <= 1:
        return torch.ones(max(nkpt, 1), dtype=torch.float32) * COCO_SIGMA_MEAN
    # 0..1 的对称位置权重：中点小、两端大（cos 形）
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
    """关键点 OKS 损失：可配置 sigma 策略与可见性加权（替换框架 ``KeypointLoss``）。

    与框架 ``KeypointLoss.forward(pred_kpts, gt_kpts, kpt_mask, area)`` **签名完全一致**，
    因此可以直接把手里的框架实例换掉，而不用重写 ``v8PoseLoss.loss`` 的上百行逻辑。

    Args:
        sigmas: 每点 sigma（1D tensor，长度 = 关键点数）。
        strategy: 策略名，仅用于日志/自检（实际 sigma 由 ``sigmas`` 决定）。
        min_sigma: sigma 下限（默认 1e-3）。防止某个点的 sigma 被压到 0 后
            ``(2σ)²·area`` 下溢，让该点的损失恒为 1（饱和）而丧失梯度。
        eps: 分母保护项，与框架的 1e-9 保持一致以便对照。
    """

    def __init__(self, sigmas: torch.Tensor, strategy: str = "auto",
                 min_sigma: float = 1e-3, eps: float = 1e-9):
        super().__init__()
        self.strategy = strategy
        self.min_sigma = float(min_sigma)
        self.eps = float(eps)
        sigma = torch.as_tensor(sigmas, dtype=torch.float32).flatten().clamp_min(self.min_sigma)
        self.register_buffer("sigmas", sigma)

    # ------------------------------------------------------------------ 前向
    def forward(self, pred_kpts: torch.Tensor, gt_kpts: torch.Tensor,
                kpt_mask: torch.Tensor, area: torch.Tensor) -> torch.Tensor:
        """与框架同式的 OKS 损失。

        Args:
            pred_kpts: ``(N, K, d)``，d ≥ 2（像素/stride 单位的归一化坐标）。
            gt_kpts: 同形状 GT。
            kpt_mask: ``(N, K)`` bool/int，1 表示该点有效。
            area: ``(N, 1)`` 归一化框面积（= (w/s)·(h/s)，见框架 ``calculate_keypoints_loss``）。

        Returns:
            标量损失（与框架一致：先按可见点数放大，再对 batch 取均值）。
        """
        d = ((pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2)
             + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2))
        # 可见点越少，单点权重越大（框架原式；保持行为一致，避免"点少的样本被稀释"）
        factor = kpt_mask.shape[1] / (kpt_mask.sum(dim=1) + self.eps)
        sigma = self.sigmas.to(d.device).view(1, -1)
        e = d / ((2 * sigma).pow(2) * (area + self.eps) * 2)
        mask = kpt_mask.to(d.dtype) if kpt_mask.dtype != d.dtype else kpt_mask
        return (factor.view(-1, 1) * ((1.0 - torch.exp(-e)) * mask)).mean()

    # ---------------------------------------------------------------- 自检
    def describe(self) -> str:
        sig = ", ".join(f"{v:.3f}" for v in self.sigmas.tolist()[:6])
        return (f"TinyPoseLoss(strategy={self.strategy}, n_kpt={self.sigmas.numel()}, "
                f"min_sigma={self.min_sigma:g}, σ[:6]=[{sig}…])")


# --------------------------------------------------------------- 接线（被 Trainer 调用）


def pose_criteria(criterion: Any) -> list[Any]:
    """列出准则里所有持有 ``keypoint_loss`` 的子准则（姿态准则判别器）。

    单分支（``v8PoseLoss`` / ``PoseLoss26``）返回 1 个；YOLO26 ``end2end`` 下
    ``E2ELoss`` 持有 ``one2many`` + ``one2one`` **两套** ``PoseLoss26``，返回 2 个
    （与 ``tod.loss.criterion.bbox_criteria`` 同思路 —— 只替换一半会静默漏改）。
    """
    if hasattr(criterion, "keypoint_loss"):
        return [criterion]
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
    """把 ``TinyPoseLoss`` 接进框架姿态准则；返回可读改动描述列表。

    Args:
        criterion: ``model.init_criterion()`` 的结果（可能已被 ``build_detection_loss``
            包过 —— 两者可以叠加，互不干扰：一个换框损失，一个换关键点损失）。
        enabled: ``False`` 表示**保留框架原生 KeypointLoss**（消融用）。
        strategy: ``person`` / ``auto`` / ``balanced``，见 ``sigma_tensor``。
        min_sigma: sigma 下限。

    Returns:
        改动描述列表（``enabled=False`` 时返回"未替换"的说明，便于日志自检）。
    """
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
