"""STAL —— 小目标感知的标签分配（SDD-YOLO §4.6，EP6）。"""

from __future__ import annotations

from typing import Any

import torch

from tod.compat import tal_assigner
from tod.registry import register

_CLASSES: dict[bool, type] = {}


def _xywh2xyxy(x: torch.Tensor) -> torch.Tensor:
    """``(..., 4)`` 的 xywh → xyxy（不依赖框架内部工具，避免版本漂移）。"""
    cx, cy, w, h = x.unbind(-1)
    return torch.stack((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), dim=-1)


def select_candidates_in_gts(self, xy_centers, gt_bboxes, mask_gt, eps=1e-9):
    """锚点中心是否落在 GT 框内；``small_target_aware=True`` 时对极小目标放宽。"""
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
        cls = type(
            name,
            (base,),
            {"small_target_aware": key, "select_candidates_in_gts": select_candidates_in_gts},
        )
        cls.__module__ = __name__
        globals()[name] = cls
        _CLASSES[key] = cls
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
