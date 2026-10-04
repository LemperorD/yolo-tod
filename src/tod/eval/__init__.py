"""评测模块（PLAN §7）：尺度分层指标、按面积上报、显著性检验的落脚点。

两条并行口径（任务不同，不可混用）：
    * ``detect`` —— ``tod.eval.scales``：整体 AP + ``AP_small`` / ``AP_tiny``（按框边长分层）；
    * ``pose`` —— ``tod.eval.pose``：整体 **OKS-AP** + 按**目标边长**分层 + 按**关键点尺度**
      的逐点诊断（≤1px 命中率 / 平均误差 / 平均 OKS）。

姿态为什么必须单独一套：OKS 的分母含框面积，同一个像素误差在小目标上的 OKS 远低于
大目标 —— 只看整体 OKS-AP 会把"小目标关键点全废"看成"基本没变"（见 pose.py 的模块说明）。
"""

from tod.eval.pose import (  # noqa: F401
    evaluate_pose,
    evaluate_pose_ap,
    extract_pose_preds,
    format_pose_table,
    load_pose_labels,
    match_pose_image,
    oks,
)
from tod.eval.scales import (  # noqa: F401
    Bin,
    average_precision,
    bins_from_edges,
    dataset_bins,
    evaluate,
    format_table,
    match_image,
    resolve_split,
)

__all__ = ["Bin", "average_precision", "bins_from_edges", "dataset_bins", "evaluate",
           "format_table", "match_image", "resolve_split",
           "evaluate_pose", "evaluate_pose_ap", "extract_pose_preds",
           "format_pose_table", "load_pose_labels", "match_pose_image", "oks"]
