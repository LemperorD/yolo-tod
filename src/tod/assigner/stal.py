"""STAL —— 小目标感知的标签分配（SDD-YOLO §4.6，EP6）。

论文原文（§4.6，全文关于 STAL 只有这两处）：

    摘要："native small-target-aware label assignment (STAL)"
    正文："This is coupled with STAL (Small-Target-Aware Label Assignment), which
           assigns adaptive higher weights to micro-target anchors."

    ↳ **论文没有给出 STAL 的公式、伪代码或超参**，也没有说明"higher weights"加在哪里。

框架侧的事实（ultralytics 8.4，``TaskAlignedAssigner.select_candidates_in_gts``）::

    gt_bboxes_xywh = xyxy2xywh(gt_bboxes)
    wh_mask = gt_bboxes_xywh[..., 2:] < self.stride[0]      # 比最小 stride 还小的目标
    gt_bboxes_xywh[..., 2:] = torch.where(
        (wh_mask * mask_gt).bool(),
        torch.tensor(self.stride_val, ...),                 # stride[1]（次小 stride）
        gt_bboxes_xywh[..., 2:])
    gt_bboxes = xywh2xyxy(gt_bboxes_xywh)
    ... 之后按"锚点中心是否落在 GT 框内"筛选正样本

    也就是说：**当 GT 的宽或高小于最小 stride（P3 = 8px）时，把中心采样区域从
    实际 GT 框放宽到次小 stride（16px）**，让小目标也能分到足够的正样本锚点。
    这就是论文所说的 "small-target-aware"。

【本库的理解与落地（标注为推断）】
    论文说的 "assigns adaptive higher weights to micro-target anchors"，
    本库理解为**让极小目标匹配到更多正样本锚点**（先验放宽），而不是额外的权重重标定：
      * ``small_target_aware=True``（默认）：显式复刻上面这条规则。在 8.4 上与框架
        原生实现**数值完全一致**（tests/test_modules.py 有逐元素对照），旧版框架没有
        该规则时则由本类补上；
      * ``small_target_aware=False``：退回经典 YOLOv8 TAL（不做放宽），
        从而得到论文 Table 3 里 STAL 那一列的**可消融开关**。

    如果论文作者后来公布了 STAL 的具体形式（例如对 align_metric 做面积相关的重标定），
    只需替换本文件，不影响其它 EP。
"""

from __future__ import annotations

from typing import Any

import torch

from tod.compat import tal_assigner
from tod.registry import register

#: 每个开关值只动态建一次子类（框架基类导入较慢）。
_CLASSES: dict[bool, type] = {}


def _xywh2xyxy(x: torch.Tensor) -> torch.Tensor:
    """``(..., 4)`` 的 xywh → xyxy（不依赖框架内部工具，避免版本漂移）。"""
    cx, cy, w, h = x.unbind(-1)
    return torch.stack((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), dim=-1)


def select_candidates_in_gts(self, xy_centers, gt_bboxes, mask_gt, eps=1e-9):
    """锚点中心是否落在 GT 框内；``small_target_aware=True`` 时对极小目标放宽。

    与框架实现同序、同语义（见文件头），返回 ``(b, n_boxes, h*w)`` 的 bool 掩码。
    """
    xywh = torch.stack(
        ((gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2,
         (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2,
         gt_bboxes[..., 2] - gt_bboxes[..., 0],
         gt_bboxes[..., 3] - gt_bboxes[..., 1]),
        dim=-1,
    )
    if getattr(self, "small_target_aware", True):
        smallest = self.stride[0]
        widened = getattr(self, "stride_val",
                          self.stride[1] if len(self.stride) > 1 else self.stride[0])
        wh_mask = xywh[..., 2:] < smallest
        xywh[..., 2:] = torch.where(
            (wh_mask * mask_gt).bool(),
            torch.tensor(widened, dtype=xywh.dtype, device=xywh.device),
            xywh[..., 2:],
        )
        gt_bboxes = _xywh2xyxy(xywh)

    bs, n_boxes, _ = gt_bboxes.shape
    n_anchors = xy_centers.shape[0]
    lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)
    deltas = torch.cat(
        (xy_centers[None] - lt, rb - xy_centers[None]), dim=2
    ).view(bs, n_boxes, n_anchors, -1)
    return deltas.amin(3).gt_(eps)


def small_target_assigner_class(small_target_aware: bool = True) -> type:
    """构造（并缓存）STAL 分配器类：框架 ``TaskAlignedAssigner`` 的子类。"""
    key = bool(small_target_aware)
    if key not in _CLASSES:
        base = tal_assigner()
        name = "SmallTargetAssigner" if key else "ClassicTALAssigner"
        _CLASSES[key] = type(
            name,
            (base,),
            {"small_target_aware": key, "select_candidates_in_gts": select_candidates_in_gts},
        )
    return _CLASSES[key]


@register(
    name="STAL",
    ep="EP6",
    paper="SDD-YOLO §4.6 的 STAL（Small-Target-Aware Label Assignment）；"
          "论文未给公式，本库按 ultralytics 8.4 的 TAL 小目标先验落地并标注为推断",
    url="https://arxiv.org/abs/2603.25218",
    year=2026,
    license="本文件为独立实现；基类 TaskAlignedAssigner 来自 ultralytics（AGPL-3.0）",
    cost="与 TAL 同量级；只改中心采样区域，无额外参数与显存",
    notes="GT 宽/高 < 最小 stride（P3=8px）时把匹配区域放宽到次小 stride（16px），"
          "让小目标获得足够正样本；设 small_target_aware=False 即退回经典 TAL（消融）",
    aliases=("STALAssigner", "SmallTargetAssigner", "small_target_assigner"),
)
def build_stal(**kwargs: Any):
    """按论文语义构造一个 STAL 分配器实例（工厂；注册表里登记的就是它）。"""
    aware = bool(kwargs.pop("small_target_aware", True))
    return small_target_assigner_class(aware)(**kwargs)


def assigner_criteria(criterion: Any) -> list[Any]:
    """列出准则里所有持有 ``assigner`` 的子准则（E2ELoss 会有两套）。"""
    if hasattr(criterion, "assigner"):
        return [criterion]
    subs = [getattr(criterion, n) for n in ("one2many", "one2one")
            if hasattr(getattr(criterion, n, None), "assigner")]
    if not subs:
        raise RuntimeError("准则里找不到 assigner（既不是单分支也不是 E2ELoss 双分支）。")
    return subs


def install_assigner(criterion: Any, small_target_aware: bool = True) -> list[str]:
    """把准则里的 TAL 换成 STAL（或显式退回经典 TAL），返回可读描述。"""
    report: list[str] = []
    cls = small_target_assigner_class(small_target_aware)
    for sub in assigner_criteria(criterion):
        old = sub.assigner
        kwargs = {
            "topk": getattr(old, "topk", 10),
            "num_classes": getattr(old, "num_classes", 80),
            "alpha": getattr(old, "alpha", 0.5),
            "beta": getattr(old, "beta", 6.0),
            "stride": list(getattr(old, "stride", [8, 16, 32])),
            "eps": getattr(old, "eps", 1e-9),
        }
        if getattr(old, "topk2", None) is not None:
            kwargs["topk2"] = old.topk2
        sub.assigner = cls(**kwargs)
        report.append(f"assigner -> {cls.__name__}(small_target_aware={small_target_aware})")
    return report


__all__ = ["build_stal", "install_assigner", "select_candidates_in_gts",
           "small_target_assigner_class"]
