"""特征对齐知识蒸馏（SDD-YOLO §4.7，EP9）。

论文原文（式 (6)(7)）::

    L_total = (1 - λ)·L_task + λ·L_KD                              (6)
    L_KD    = Σ_{l ∈ {P2,P3,P4,P5}} T² · KL( σ(z_s^l / T) ‖ σ(z_t^l / T) )   (7)

    "we utilize a pre-trained YOLO26x (or a fine-tuned heavy YOLO-X) teacher model
     to guide the training of the SDD-YOLO-n student ... Following empirical
     validation, we set the distillation weight λ = 0.5 and temperature T = 3.0"

实现要点与本库的判断：
  * ``z^l`` 论文写的是 "logits from the student and teacher at feature level l"，
    式 (7) 又是逐层 softmax + KL —— 因此本实现取**检测头每一层的分类 logits**
    （形状 ``(b, A_l, nc)``），而不是特征图本身。"feature-aligned" 指的是
    **跨尺度对齐**（P2–P5 逐层），不是特征图回归；
  * 逐层按 **stride 对齐**（而非列表位置）：教师若没有 P2 头，就只在 P3–P5 上蒸馏，
    并在日志里说明；
  * 框架的 ``L_task`` 已经乘了 batch size（见 ultralytics 的
    ``v8DetectionLoss.loss``），所以 KD 项同样乘 batch size，
    保证式 (6) 的 λ 是"同量纲"的加权，而不是被 batch 大小悄悄改变；
  * teacher 以 ``eval()`` 跑（不动 BN 统计），但把**检测头单独置为 train()**，
    因为框架的检测头只有训练模式才返回逐层原始 logits（推理模式返回的是解码结果）。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from tod.compat import CompatError
from tod.registry import register


def stride_to_level(stride: int) -> int:
    """stride → 金字塔层号（4→2、8→3、16→4、32→5）。"""
    return int(round(math.log2(int(stride))))


def unwrap_preds(preds: Any) -> dict:
    """把检测头的输出规整成含 ``boxes/scores/feats`` 的 dict。

    YOLO26 (``end2end=True``) 的输出是 ``{"one2many": {...}, "one2one": {...}}``；
    蒸馏用 **one2many**（训练主分支，与论文 §4.4 的 O2M 训练分支一致）。
    """
    if isinstance(preds, tuple):
        preds = preds[1]
    if isinstance(preds, dict) and "one2many" in preds:
        preds = preds["one2many"]
    if not isinstance(preds, dict) or "scores" not in preds or "feats" not in preds:
        raise CompatError(
            f"无法从检测头输出里取逐层 logits（拿到 {type(preds).__name__}）。"
            "请确认模型处于 train() 模式且框架版本受支持。"
        )
    return preds


def level_logits(preds: Any, strides: Iterable[int]) -> dict[int, torch.Tensor]:
    """返回每一层的分类 logits，**以 P 层号（2/3/4/5）为键**：``{2: (b,A_l,nc), ...}``。

    键用层号而不是 stride，才能和 ``FeatureAlignKD.levels``（默认 P2–P5）直接对齐。

    框架 8.4 的检测头给出 ``scores`` 形状为 ``(b, nc, A)``（A 为各层锚点数之和），
    这里自动识别**锚点维度**并拆层，统一整理成 ``(b, A_l, nc)``，
    因此对 ``(b, A, nc)`` 的历史布局同样适用。
    """
    preds = unwrap_preds(preds)
    scores, feats = preds["scores"], list(preds["feats"])
    strides = [int(s) for s in strides]
    if len(strides) != len(feats):
        raise CompatError(
            f"stride 数（{len(strides)}）与特征图层数（{len(feats)}）不一致，"
            "无法按尺度对齐 logits。"
        )
    sizes = [f.shape[-2] * f.shape[-1] for f in feats]
    total = sum(sizes)
    anchor_dim = next((i for i, s in enumerate(scores.shape) if int(s) == total), None)
    if anchor_dim is None:
        raise CompatError(
            f"scores 形状 {tuple(scores.shape)} 里没有等于锚点总数 {total} 的维度，"
            "无法拆层。请检查框架版本的检测头输出约定。"
        )
    parts = list(torch.split(scores, sizes, dim=anchor_dim))
    if anchor_dim != 1:                                 # (b, nc, A) → (b, A, nc)
        parts = [p.transpose(anchor_dim, 1) for p in parts]
    return {stride_to_level(strides[i]): parts[i] for i in range(len(feats))}


@register(
    name="FeatureAlignKD",
    ep="EP9",
    paper="SDD-YOLO §4.7 式 (6)(7)：多尺度特征对齐知识蒸馏（Hinton et al. 2015 的 KL 蒸馏）",
    url="https://arxiv.org/abs/2603.25218",
    year=2026,
    license="本文件为独立实现（KL 蒸馏为公开方法）",
    cost="每个 batch 多一次 teacher 前向（训练时间约 ×1.3–2.0，取决于教师规模）；"
         "推理零成本（教师只在训练时存在）",
    notes="P2–P5 逐层分类 logits 的 KL；λ=0.5、T=3.0（论文 §4.7 的经验取值）",
    aliases=("KD", "FeatureAlignmentKD", "feature_align_kd"),
)
class FeatureAlignKD(nn.Module):
    """式 (6)(7) 的蒸馏项：逐层 KL（按 stride 对齐）。

    Args:
        lambda_: 式 (6) 的蒸馏权重 λ（论文取 0.5）。
        temperature: 蒸馏温度 T（论文取 3.0）。
        levels: 参与蒸馏的金字塔层号（默认 P2–P5）。
        anchor_reduction: KL 在**锚点维**上取均值还是求和。论文式 (7) 只写了
            "Σ_{l}"，没有规定锚点维的归一化方式：若按锚点求和，KD 会随输入分辨率
            （P2 在 1024 输入下有 65 536 个锚点）线性放大，λ=0.5 会直接压垮 L_task。
            因此默认 ``"mean"``（每个锚点、每张图取平均），``"sum"`` 保留另一种读法。
    """

    def __init__(self, lambda_: float = 0.5, temperature: float = 3.0,
                 levels: Iterable[int] = (2, 3, 4, 5), anchor_reduction: str = "mean"):
        super().__init__()
        if anchor_reduction not in ("mean", "sum"):
            raise ValueError(f"anchor_reduction 只能是 mean/sum，实际 {anchor_reduction!r}。")
        self.lambda_ = float(lambda_)
        self.temperature = float(temperature)
        self.levels = tuple(int(x) for x in levels)
        self.anchor_reduction = anchor_reduction

    # ---------------------------------------------------------------- 前向
    def forward(self, student: dict[int, torch.Tensor],
                teacher: dict[int, torch.Tensor]) -> torch.Tensor:
        """返回 ``L_KD``（式 (7)，再乘 batch size 以与框架 L_task 同量纲）。"""
        shared = [lv for lv in self.levels if lv in student and lv in teacher]
        if not shared:
            raise CompatError(
                f"学生/教师没有共同的蒸馏层：学生 {sorted(student)} vs 教师 {sorted(teacher)}。"
                "请检查教师模型是否包含对应尺度的检测头。"
            )
        batch = next(iter(student.values())).shape[0]
        total = student[shared[0]].new_zeros(())
        t = self.temperature
        for lv in shared:
            zs, zt = student[lv], teacher[lv]
            if zs.shape != zt.shape:
                raise CompatError(
                    f"第 P{lv} 层 logits 形状不一致：学生 {tuple(zs.shape)} vs "
                    f"教师 {tuple(zt.shape)}（类别数或输入分辨率不同？）"
                )
            # 逐锚点的 KL(σ(z_s/T) ‖ σ(z_t/T))：先在类别维求和，再按 anchor_reduction 归一化
            kl = F.kl_div(
                F.log_softmax(zs / t, dim=-1),
                F.softmax(zt / t, dim=-1),
                reduction="none",
            ).sum(-1)
            per_level = kl.mean() if self.anchor_reduction == "mean" else kl.sum()
            total = total + per_level * (t * t)         # 式 (7) 的 T²
        return total * batch                            # 与框架 L_task 同量纲

    def combine(self, task_loss: torch.Tensor, kd_loss: torch.Tensor) -> torch.Tensor:
        """式 (6)：``(1-λ)·L_task + λ·L_KD``（两个入参都应当是**标量**）。

        为什么强调标量：框架的准则返回的是 ``(box, cls, dfl)`` 三分量向量，
        若把 KD 标量直接加到该向量上，它会同时进入三个分量，
        等效权重变成 3λ（实测：λ=0.5 时 KD 对总损失的贡献是 1.5 倍 KD）。
        框架训练器自己会对返回的 loss 求 ``sum()``，所以这里先把 task 分量求和。
        """
        return (1.0 - self.lambda_) * task_loss + self.lambda_ * kd_loss


class KDCriterion:
    """把蒸馏项接到框架准则外面（不 fork 训练循环）。

    用法：``model.criterion = build_kd(model, teacher, ...)``。
    框架只在训练时调用 ``criterion(preds, batch)``，因此这里可以顺手跑一次 teacher。

    为什么是**普通类**而不是 ``nn.Module``：框架自己的准则（``v8DetectionLoss`` /
    ``E2ELoss``）就是普通对象。若换成 ``nn.Module``，``model.criterion = ...`` 会把它
    注册成 ``DetectionModel`` 的子模块，于是 teacher 也会被卷进 ``model.parameters()``、
    ``ModelEMA`` 与 checkpoint（教师必须完全在训练回路之外）；而且之后想把
    ``model.criterion`` 换回框架准则时，PyTorch 会拒绝把普通对象赋给已存在的子模块槽位。
    KD 项本身没有可训练参数，用普通类零代价。
    """

    def __init__(self, base: Any, teacher: nn.Module, kd: FeatureAlignKD,
                 student_strides: Iterable[int], teacher_strides: Iterable[int]):
        self.base = base
        self.teacher = teacher.eval()
        self.kd = kd
        self.student_strides = [int(s) for s in student_strides]
        self.teacher_strides = [int(s) for s in teacher_strides]

    def __call__(self, preds: Any, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward(preds, batch)

    # ---- 与框架 E2ELoss 的 ProgLoss 调度接口对齐（trainer 会在每个 epoch 调 update）----
    def update(self) -> None:
        inner = getattr(self.base, "update", None)
        if callable(inner):
            if hasattr(self, "updates"):
                self.base.updates = self.updates
            inner()

    def forward(self, preds: Any, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        task_components, items = self.base(preds, batch)
        teacher_preds = self._teacher_forward(batch["img"])
        kd_loss = self.kd(level_logits(preds, self.student_strides),
                          level_logits(teacher_preds, self.teacher_strides))
        # 先对三分量求和再与 KD 加权（原因见 FeatureAlignKD.combine）
        total = self.kd.combine(task_components.sum(), kd_loss)
        return total, items

    @torch.no_grad()
    def _teacher_forward(self, img: torch.Tensor) -> Any:
        """跑一次 teacher：整体 eval（不动 BN 统计），但检测头临时置 train 以拿原始 logits。"""
        teacher = self.teacher
        head = teacher.model[-1] if hasattr(teacher, "model") else None
        was_training = teacher.training
        head_training = getattr(head, "training", None)
        try:
            teacher.eval()
            if head is not None:
                head.train()
            return teacher(img)
        finally:
            teacher.train(was_training)
            if head is not None and head_training is not None:
                head.train(head_training)


def head_strides(model: nn.Module) -> list[int]:
    """取检测头的 stride 列表（用于按尺度对齐 teacher/student）。"""
    inner = getattr(model, "model", None)
    if inner is None or not hasattr(inner, "__getitem__") or not len(inner):
        raise CompatError("模型没有 .model Sequential，无法读取检测头 stride。")
    head = inner[-1]
    if hasattr(head, "module"):                      # DataParallel 包装
        head = head.module
    strides = getattr(head, "stride", None)
    if strides is None:
        raise CompatError("模型没有检测头 stride，无法做逐层对齐。")
    return [int(s) for s in strides]


def load_teacher(src: Any) -> nn.Module:
    """加载教师模型。

    * ``nn.Module``：直接用（测试里常把学生深拷贝当教师，避免下载权重）；
    * ``str/Path``：只接受**本地** ``.pt``（本库不自动下载权重，训练机通常离线）。

    Returns:
        教师 ``DetectionModel``（de-parallel）。
    """
    if isinstance(src, nn.Module):
        model = src
        return model.module if hasattr(model, "module") else model

    path = Path(str(src))
    if not path.is_file():
        raise CompatError(
            f"教师权重不存在：{path}。请显式提供本地 .pt 路径"
            "（避免在离线训练机上触发自动下载）。"
        )
    from tod.compat import ensure_runtime_env

    ensure_runtime_env()
    from ultralytics import YOLO

    model = YOLO(str(path)).model
    return model.module if hasattr(model, "module") else model


def build_kd(base_criterion: Any, student_model: nn.Module, teacher: Any,
             *, lambda_: float = 0.5, temperature: float = 3.0,
             levels: Iterable[int] = (2, 3, 4, 5)) -> KDCriterion:
    """构造带蒸馏的准则（式 (6)）。"""
    teacher_model = load_teacher(teacher).eval()
    teacher_model.to(next(student_model.parameters()).device)
    for p in teacher_model.parameters():
        p.requires_grad_(False)
    kd = FeatureAlignKD(lambda_=lambda_, temperature=temperature, levels=levels)
    return KDCriterion(
        base=base_criterion,
        teacher=teacher_model,
        kd=kd,
        student_strides=head_strides(student_model),
        teacher_strides=head_strides(teacher_model),
    )


__all__ = ["FeatureAlignKD", "KDCriterion", "build_kd", "head_strides", "level_logits",
           "load_teacher", "stride_to_level", "unwrap_preds"]
