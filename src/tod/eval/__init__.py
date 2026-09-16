"""评测模块（PLAN §7）：尺度分层指标、按面积上报、显著性检验的落脚点。"""

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
           "format_table", "match_image", "resolve_split"]
