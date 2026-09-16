"""EP7 损失函数：边界框回归损失（按名字取用，供训练准则与 YAML 引用）。"""

from tod.loss.box import BOX_LOSSES, WiseIoU, box_loss, siou  # noqa: F401

__all__ = ["siou", "WiseIoU", "box_loss", "BOX_LOSSES"]
