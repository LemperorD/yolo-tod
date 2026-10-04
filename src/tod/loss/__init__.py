"""EP7 损失函数：边界框回归损失与姿态关键点损失（按名字取用，供训练准则与 YAML 引用）。"""

from tod.loss.box import BOX_LOSSES, WiseIoU, box_loss, siou  # noqa: F401
from tod.loss.pose import (COCO_SIGMA_17, TinyPoseLoss,  # noqa: F401
                           build_pose_loss, pose_criteria, sigma_tensor)

__all__ = ["siou", "WiseIoU", "box_loss", "BOX_LOSSES",
           "TinyPoseLoss", "build_pose_loss", "pose_criteria", "sigma_tensor",
           "COCO_SIGMA_17"]
