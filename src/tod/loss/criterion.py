"""把 EP7 的回归损失接进框架的训练准则（含 SDD-YOLO §4.3 的"无 DFL"回归项）。"""

from __future__ import annotations

import inspect
from typing import Any

import torch
import torch.nn as nn

from tod.compat import CompatError
from tod.loss.box import box_loss

BBOX_FORWARD_ARGS = (
    "pred_dist",
    "pred_bboxes",
    "anchor_points",
    "target_bboxes",
    "target_scores",
    "target_scores_sum",
    "fg_mask",
)
BBOX_FORWARD_OPTIONAL_ARGS = ("imgsz", "stride")


class TODBboxLoss(nn.Module):
    """包装框架的 ``BboxLoss``，仅替换 IoU 项。"""

    def __init__(self, base: nn.Module, kind: str = "siou", theta: float = 4.0,
                 use_dfl: bool | None = None, **kind_kwargs: Any):
        super().__init__()
        self.base = base
        self.kind = kind
        self.theta = theta
        self.use_dfl = use_dfl
        self.fn = box_loss(kind, **kind_kwargs)
        self._assert_compatible()
        self._pass_theta = "theta" in inspect.signature(self.fn.forward).parameters \
            if isinstance(self.fn, nn.Module) else "theta" in inspect.signature(self.fn).parameters

    def _assert_compatible(self) -> None:
        try:
            params = set(inspect.signature(self.base.forward).parameters)
        except (TypeError, ValueError) as exc:  # pragma: no cover
            raise CompatError(f"无法检查框架 BboxLoss 签名：{exc}") from exc
        missing = [a for a in BBOX_FORWARD_ARGS if a not in params]
        if missing:
            raise CompatError(
                f"框架 BboxLoss.forward 缺少参数 {missing}，签名可能已变更。"
                "请更新 src/tod/compat.py 与 tod/loss/criterion.py 后重试。"
            )
        self._base_extra = tuple(a for a in BBOX_FORWARD_OPTIONAL_ARGS if a in params)

    def forward(
        self,
        pred_dist,
        pred_bboxes,
        anchor_points,
        target_bboxes,
        target_scores,
        target_scores_sum,
        fg_mask,
        imgsz=None,
        stride=None,
    ):
        if self.use_dfl is False:
            loss_dfl = torch.zeros((), device=pred_bboxes.device, dtype=pred_bboxes.dtype)
        else:
            extra = {"imgsz": imgsz, "stride": stride}
            kwargs = {k: extra[k] for k in self._base_extra if extra.get(k) is not None}
            _, loss_dfl = self.base(
                pred_dist, pred_bboxes, anchor_points,
                target_bboxes, target_scores, target_scores_sum, fg_mask, **kwargs,
            )

        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        call_kwargs = {"xywh": False}
        if self._pass_theta:
            call_kwargs["theta"] = self.theta
        sim = self.fn(pred_bboxes[fg_mask], target_bboxes[fg_mask], **call_kwargs)
        loss_iou = ((1.0 - sim) * weight).sum() / target_scores_sum
        return loss_iou, loss_dfl


def bbox_criteria(criterion: Any) -> list[Any]:
    """列出准则里所有持有 ``bbox_loss`` 的子准则。"""
    if hasattr(criterion, "bbox_loss"):
        return [criterion]
    # 端到端头持有 O2M/O2O 两套准则，必须同时替换。
    subs = [c for name in ("one2many", "one2one")
            if hasattr(criterion, name) and hasattr(getattr(criterion, name), "bbox_loss")
            for c in [getattr(criterion, name)]]
    if not subs:
        raise CompatError(
            "准则里找不到 bbox_loss（既不是单分支也不是 E2ELoss 双分支）。"
            "框架结构可能已变更，请检查 src/tod/loss/criterion.py。"
        )
    return subs


def build_detection_loss(model, kind: str | None = "siou", theta: float = 4.0,
                         use_dfl: bool | None = None, stal: bool | None = None,
                         **kind_kwargs: Any):
    """构造检测训练准则：框架原生准则 + 我们的回归损失（+ 可选的 EP6 分配器/无 DFL）。"""
    init = getattr(model, "init_criterion", None)
    if not callable(init):
        raise CompatError(
            "模型没有 init_criterion()，无法构造训练准则（框架结构可能已变更）。"
        )
    criterion = init()
    report: list[str] = []
    if kind is not None:
        for sub in bbox_criteria(criterion):
            sub.bbox_loss = TODBboxLoss(sub.bbox_loss, kind=kind, theta=theta,
                                        use_dfl=use_dfl, **kind_kwargs)
            report.append(f"bbox_loss -> {kind}"
                          + ("" if use_dfl is None else f" (use_dfl={use_dfl})"))
    if stal is not None:
        from tod.assigner.stal import install_assigner

        report += install_assigner(criterion, small_target_aware=stal)
    criterion._tod_patched = report
    return criterion


def resolve_box_kind(ep7: dict | None) -> tuple[str | None, float]:
    """从 EP7 配置里解析"是否替换框损失"与形状代价权重。"""
    ep7 = ep7 or {}
    kind = ep7.get("box")
    theta = float(ep7.get("theta", 4.0) or 4.0)
    return kind, theta


def build_pose_criterion(model, ep7: dict | None = None, stal: bool | None = None,
                         use_dfl: bool | None = None) -> tuple[Any, list[str]]:
    """构造**姿态**训练准则：框架 PoseLoss 基线 + EP7 的框损失与关键点损失替换。"""
    from tod.loss.pose import build_pose_loss

    ep7 = ep7 or {}
    kind, theta = resolve_box_kind(ep7)
    criterion = build_detection_loss(
        model, kind=kind, theta=theta, use_dfl=use_dfl, stal=stal,
        **(ep7.get("kind_kwargs") or {}),
    )

    pose_key = ep7.get("pose")
    if pose_key is None:
        return criterion, list(getattr(criterion, "_tod_patched", []))
    enabled = not (pose_key is False or str(pose_key).lower() in {"native", "off", "none"})
    report = list(getattr(criterion, "_tod_patched", []))
    report += build_pose_loss(
        criterion,
        enabled=enabled,
        strategy=str(ep7.get("sigma_strategy", ep7.get("sigma", "auto"))),
        min_sigma=float(ep7.get("min_sigma", 1e-3) or 1e-3),
    )
    criterion._tod_patched = report
    return criterion, report


def build_reproj_criterion(criterion: Any, spec: dict, *,
                           data_cfg: dict | None = None) -> Any:
    """按变体 spec 决定是否给姿态准则加上论文的 ``L_rep``（几何重投影项）。"""
    from tod.loss.reproj import build_reproj_loss

    ep7 = ((spec.get("eps") or {}).get("EP7") or {})
    flag = ep7.get("reproj", False)
    enabled = not (flag is False or flag is None
                   or str(flag).lower() in {"", "off", "none", "false"})
    if not enabled:
        return criterion
    cfg = data_cfg if data_cfg is not None else (spec.get("dataset_cfg") or {})
    tod = cfg.get("tod") or {}
    wrapper = build_reproj_loss(
        criterion, {"data_cfg": cfg},
        enabled=True,
        weight=float(ep7.get("reproj_weight", 1.0) or 1.0),
        warmup_epochs=int(tod.get("reproj_warmup_epochs",
                                  ep7.get("reproj_warmup_epochs", 15)) or 0),
    )
    base_report = list(getattr(criterion, "_tod_patched", []))
    wrapper._tod_patched = base_report + [
        f"reproj: L_rep 已启用（论文式 (8)，权重 {wrapper.weight}，"
        f"前 {wrapper.warmup_epochs} epoch 只用 2D 项；"
        f"{wrapper.reproj.n_kpt}/{len(tod.get('object_points') or [])} 个关键点参与）"
    ]
    return wrapper
