"""把 EP7 的回归损失接进框架的训练准则（含 SDD-YOLO §4.3 的"无 DFL"回归项）。

做法（刻意保持最小侵入）：
  * **完全复用**框架原生准则的标签分配、DFL/L1 回归分支、分类损失与权重归一化；
  * 只把其中的 IoU 项换成我们注册的相似度函数（``siou`` / ``wiou``）。

为什么要用 ``model.init_criterion()`` 而不是自己 import ``v8DetectionLoss``：
    ultralytics 8.4 的 YOLO26 走 ``end2end=True``，准则被换成 ``E2ELoss``，
    内部持有 ``one2many`` / ``one2one`` **两套** ``v8DetectionLoss``（分别对应论文 §4.4
    的 O2M 训练分支与 O2O 推理分支）。我们要替换的是**两套**里的 IoU 项，
    自己 import 会漏掉一半。由框架决定准则类型最稳。

关于"无 DFL"（SDD-YOLO §4.3）：
    ``use_dfl=False`` 时本包装器直接丢弃框架回归分支的第二个返回值（DFL 或
    reg_max=1 时的 L1 归一化项）。论文的原文做法是 "setting dfl=0.0"，
    也就是把该项的**增益**置 0（等价效果，且不动框架）；两种方式本库都支持：
    训练配置里写 ``dfl=0.0`` 即走框架增益，``use_dfl=False`` 走本包装器。
"""

from __future__ import annotations

import inspect
from typing import Any

import torch
import torch.nn as nn

from tod.compat import CompatError
from tod.loss.box import box_loss

#: 框架 ``BboxLoss.forward`` 的**必需**参数名（8.2 → 8.4 稳定）。
BBOX_FORWARD_ARGS = (
    "pred_dist",
    "pred_bboxes",
    "anchor_points",
    "target_bboxes",
    "target_scores",
    "target_scores_sum",
    "fg_mask",
)
#: 8.4 起新增的可选参数（用于 reg_max=1 时的 L1 归一化）。
BBOX_FORWARD_OPTIONAL_ARGS = ("imgsz", "stride")


class TODBboxLoss(nn.Module):
    """包装框架的 ``BboxLoss``，仅替换 IoU 项。

    Args:
        base: 框架原生 ``BboxLoss`` 实例（提供回归分支与设备/参数管理）。
        kind: 回归损失名（见 ``tod.loss.box.BOX_LOSSES``），如 ``"wiou"``。
        theta: 形状代价的注意力权重，仅部分损失（SIoU）使用；其余损失忽略。
        use_dfl: ``None`` = 原样复用框架回归分支；``False`` = 丢弃该项
            （SDD-YOLO §4.3 的 DFL-free）。
        kind_kwargs: 传给损失构造函数的额外参数（如 ``variant=3``）。
    """

    def __init__(self, base: nn.Module, kind: str = "siou", theta: float = 4.0,
                 use_dfl: bool | None = None, **kind_kwargs: Any):
        super().__init__()
        self.base = base
        self.kind = kind
        self.theta = theta
        self.use_dfl = use_dfl
        self.fn = box_loss(kind, **kind_kwargs)
        self._assert_compatible()
        # 只有声明了 theta 的损失才收到该参数（WiseIoU 等显式忽略它）
        self._pass_theta = "theta" in inspect.signature(self.fn.forward).parameters \
            if isinstance(self.fn, nn.Module) else "theta" in inspect.signature(self.fn).parameters

    # ------------------------------------------------------------------ 兼容
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

    # ---------------------------------------------------------------- 前向
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
        # --- 回归分支：原样复用框架实现（DFL 或 reg_max=1 时的 L1 归一化）---
        if self.use_dfl is False:
            loss_dfl = torch.zeros((), device=pred_bboxes.device, dtype=pred_bboxes.dtype)
        else:
            extra = {"imgsz": imgsz, "stride": stride}
            kwargs = {k: extra[k] for k in self._base_extra if extra.get(k) is not None}
            _, loss_dfl = self.base(
                pred_dist, pred_bboxes, anchor_points,
                target_bboxes, target_scores, target_scores_sum, fg_mask, **kwargs,
            )

        # --- IoU 项：换成我们注册的相似度函数 ---
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        call_kwargs = {"xywh": False}
        if self._pass_theta:
            call_kwargs["theta"] = self.theta
        sim = self.fn(pred_bboxes[fg_mask], target_bboxes[fg_mask], **call_kwargs)
        loss_iou = ((1.0 - sim) * weight).sum() / target_scores_sum
        return loss_iou, loss_dfl


def bbox_criteria(criterion: Any) -> list[Any]:
    """列出准则里所有持有 ``bbox_loss`` 的子准则。

    单分支（``v8DetectionLoss``）返回 1 个；YOLO26 的 ``E2ELoss`` 返回 2 个
    （``one2many`` + ``one2one``，见论文 §4.4）。
    """
    if hasattr(criterion, "bbox_loss"):
        return [criterion]
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
    """构造检测训练准则：框架原生准则 + 我们的回归损失（+ 可选的 EP6 分配器/无 DFL）。

    Args:
        model: 已构建的 ``DetectionModel``（必须是 de-parallel 的）。
        kind: 回归损失名；``None`` 表示**保留框架默认的 IoU 项**，只做其它改动。
        use_dfl: 见 ``TODBboxLoss``。
        stal: ``None`` = 保留框架自带分配器；``True`` = 换成 STAL（小目标感知，
            框架 < 8.4 时由本库补上）；``False`` = 显式退回经典 TAL（消融）。
        kind_kwargs: 传给损失构造函数的额外参数。

    Returns:
        打好补丁的准则对象；``criterion._tod_patched`` 记录本次改动的可读描述，
        供 ``--dry-run`` 与测试断言使用。
    """
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
    """从 EP7 配置里解析"是否替换框损失"与形状代价权重。

    抽出来是为了让**检测**与**姿态**两条训练路径共用同一套判断
    （``tools/train.py --dry-run`` 与 ``TODDetectionTrainer`` 已有两处重复，
    姿态训练器再加一处就是三处，迟早漂移）。

    Returns:
        ``(kind, theta)``；``kind=None`` 表示保留框架默认的 IoU 项。
    """
    ep7 = ep7 or {}
    # 只认 "box" 键：姿态变体常只写 ``pose=...``（不换框损失），此时必须返回 None
    # 而不是把 "pose" 当损失名传下去（``box_loss("oks")`` 会直接报未注册）。
    kind = ep7.get("box")
    theta = float(ep7.get("theta", 4.0) or 4.0)
    return kind, theta


def build_pose_criterion(model, ep7: dict | None = None, stal: bool | None = None,
                         use_dfl: bool | None = None) -> tuple[Any, list[str]]:
    """构造**姿态**训练准则：框架 PoseLoss 基线 + EP7 的框损失与关键点损失替换。

    姿态准则（``v8PoseLoss`` / YOLO26 的 ``E2ELoss(PoseLoss26)``）继承自
    ``v8DetectionLoss``，所以"换框损失"这条路**完全复用** ``build_detection_loss``；
    关键点项则由 ``tod.loss.pose.build_pose_loss`` 单独替换。两件事互不干扰，
    因此 ``box="wiou"`` 与 ``pose="oks"`` 可以同时生效，消融也各自独立。

    Args:
        ep7: EP7 覆盖段，识别两个键：
            * ``box``   —— 框回归损失名（``siou``/``wiou``…；缺省保留框架默认）；
            * ``pose``  —— 关键点损失：``"oks"``/``True`` 启用 ``TinyPoseLoss``，
              ``False``/``"native"`` 显式保留框架原生 ``KeypointLoss``（消融列），
              缺省 = 不替换（与框架完全一致）。
            其余键（``sigma_strategy``/``min_sigma``）透传给 ``TinyPoseLoss``。

    Returns:
        ``(criterion, report)``；``criterion._tod_patched`` 同样记录描述，便于
        ``--dry-run`` 与测试断言。
    """
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
