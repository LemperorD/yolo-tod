"""数据侧模块（EP0）。

目前包含一件事：把 **6D 位姿**读进来喂给 EP7 的几何重投影损失 ``L_rep``。

为什么需要它：论文式 (8) 的 ``L_rep`` 要比较"预测 2D 关键点对应的相机射线"与
"真值 3D 关键点"，后者必须由 ``p^c = R·p^o + t`` 得到 —— 也就是**逐目标的 6D 位姿**。
标准的 YOLO-pose 标签行里没有这个量（也不该有，否则会破坏与 ultralytics 的格式兼容），
所以本库用旁挂的 ``labels_6d/<split>.csv``，格式见 ``docs/KEYPOINTS.md`` §5.2。
"""

from tod.data.pose6d import (  # noqa: F401
    Pose6DError,
    load_pose_csv,
    parse_pose_rows,
    pose6d_dataset,
    quat_to_rot,
)

__all__ = ["Pose6DError", "load_pose_csv", "parse_pose_rows", "pose6d_dataset",
           "quat_to_rot"]
