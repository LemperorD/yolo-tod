"""把 EP7 的回归损失接进框架的训练准则。

做法（刻意保持最小侵入）：
  * **完全复用**框架原生 ``BboxLoss`` 的 DFL 部分与 ``v8DetectionLoss`` 的
    标签分配、分类损失、权重归一化；
  * 只把其中的 IoU 项换成我们注册的相似度函数。

这样升级 ultralytics 时最不容易碎；也避免重写一整个 criterion 带来的偏差。
"""

from __future__ import annotations

import inspect

import torch.nn as nn

from tod.compat import CompatError
from tod.loss.box import box_loss

#: 框架 BboxLoss.forward 的参数名（8.0 → 8.3 稳定）。
#: 若某天对不上，这里会立刻报错并指向 compat.py，而不是静默算错损失。
BBOX_FORWARD_ARGS = (
    "pred_dist",
    "pred_bboxes",
    "anchor_points",
    "target_bboxes",
    "target_scores",
    "target_scores_sum",
    "fg_mask",
)


class TODBboxLoss(nn.Module):
    """包装框架的 ``BboxLoss``，仅替换 IoU 项。

    Args:
        base: 框架原生 ``BboxLoss`` 实例（提供 DFL 与设备/参数管理）。
        kind: 回归损失名（见 ``tod.loss.box.BOX_LOSSES``），如 ``"siou"``。
        theta: 形状代价的注意力权重，仅部分损失使用。
    """

    def __init__(self, base: nn.Module, kind: str = "siou", theta: float = 4.0):
        super().__init__()
        self.base = base
        self.kind = kind
        self.theta = theta
        self.fn = box_loss(kind)
        self._assert_compatible()

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

    def forward(
        self,
        pred_dist,
        pred_bboxes,
        anchor_points,
        target_bboxes,
        target_scores,
        target_scores_sum,
        fg_mask,
    ):
        # DFL 分支原样复用框架实现
        _, loss_dfl = self.base(
            pred_dist, pred_bboxes, anchor_points,
            target_bboxes, target_scores, target_scores_sum, fg_mask,
        )
        # IoU 分支换成我们注册的相似度函数
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = self.fn(pred_bboxes[fg_mask], target_bboxes[fg_mask],
                      xywh=False, theta=self.theta)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
        return loss_iou, loss_dfl


def build_detection_loss(model, kind: str = "siou", theta: float = 4.0):
    """构造检测训练准则：框架原生 ``v8DetectionLoss`` + 我们的回归损失。

    框架版本敏感的导入集中在这里，失败时给出可操作的报错。
    """
    try:
        from ultralytics.utils.loss import v8DetectionLoss
    except ImportError as exc:  # pragma: no cover
        raise CompatError(
            "未能导入 ultralytics.utils.loss.v8DetectionLoss，"
            "框架结构可能已变更，请检查 src/tod/compat.py。"
        ) from exc

    criterion = v8DetectionLoss(model)
    base = getattr(criterion, "bbox_loss", None)
    if base is None:
        raise CompatError("v8DetectionLoss 没有 bbox_loss 属性，无法替换回归损失。")
    criterion.bbox_loss = TODBboxLoss(base, kind=kind, theta=theta)
    return criterion
