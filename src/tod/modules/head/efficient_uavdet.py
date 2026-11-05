"""Efficient_UAVDet —— 基于分组卷积的轻量检测头（SPAE-YOLOv8 §3.4）。"""

from __future__ import annotations

import torch
import torch.nn as nn

from tod.compat import CompatError
from tod.registry import register


def _valid_groups(c_in: int, c_out: int, per_group: int, *, allow_degenerate: bool = True) -> int:
    """求满足"每组约 per_group 通道"且能整除输入/输出通道的最大分组数。"""
    g = max(1, int(c_in) // int(per_group))
    while g > 1 and (c_in % g or c_out % g):
        g -= 1
    if g == 1 and int(per_group) < int(c_in) and not allow_degenerate:
        raise ValueError(
            f"无法在 in={c_in} / out={c_out} 上实现 ~{per_group} 通道/组的分组卷积"
            f"（只能退化为 g=1，即普通卷积）。请改 per_group，或显式允许退化。"
        )
    return g


@register(
    ep="EP5",
    paper="Efficient_UAVDet（SPAE-YOLOv8 §3.4，式 10–11 / Figure 4 / Table 3）",
    url="https://doi.org/10.3390/s26113424",
    year=2026,
    license="论文 CC BY 4.0；本文件为按论文描述独立实现",
    cost="检测头参数量约减半；论文报告整机 3.0M→2.4M（−20%）、FPS 197.2→252.9；"
         "代价是 mAP@0.5 −0.4pp（论文明确承认是压缩/加速手段而非涨点手段）",
    notes="两层连续 3x3 分组卷积替换分支 stem，g=x//16（每组 16 通道），"
          "末端 1x1 保持普通卷积；不含 channel shuffle",
    aliases=("EfficientUAVDet", "EfficientUAVDetHead", "Efficient_UAVDet"),
)
def _best_groups(c_in: int, c_mid: int, prefer: int) -> int:
    """取不超过首选值、且能同时整除 ``c_in`` 与 ``c_mid`` 的最大分组数。"""
    g = min(int(prefer), int(c_in), int(c_mid))
    while g > 1 and (int(c_in) % g or int(c_mid) % g):
        g -= 1
    return max(1, g)


def choose_stem(c_in: int, c_mid: int, requested: int | None) -> tuple[int, str]:
    """决定分支 stem 的**实际**分组数与理由（供日志与断言）。"""
    if requested is None:
        return 1, "kept"
    prefer = max(1, int(c_in) // int(requested))
    best = _best_groups(c_in, c_mid, prefer)
    if best <= 1:
        return 1, "infeasible"
    return (best, "ok") if best == prefer else (best, "auto_degraded")


class GroupedStem(nn.Module):
    """检测头分支 stem：两层连续的 3x3 分组卷积（k=3, s=1, p=1，组内约 16 通道）。"""

    def __init__(self, c_in: int, c_mid: int, per_group: int = 16, groups: int | None = None):
        super().__init__()
        from tod.compat import base_conv

        conv = base_conv()
        self.groups = int(groups) if groups else _valid_groups(c_in, c_mid, per_group)
        self.per_group = int(per_group)
        self.cv1 = conv(c_in, c_mid, 3, 1, 1, g=self.groups)
        self.cv2 = conv(c_mid, c_mid, 3, 1, 1, g=self.groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv2(self.cv1(x))


class KeptStem(nn.Module):
    """**不压缩**的分支 stem：两层连续的标准 3×3 卷积，接口与 ``GroupedStem`` 一致。"""

    def __init__(self, c_in: int, c_mid: int, per_group: int = 16):
        super().__init__()
        from tod.compat import base_conv

        conv = base_conv()
        self.groups = 1
        self.per_group = per_group
        self.cv1 = conv(c_in, c_mid, 3, 1, 1)
        self.cv2 = conv(c_mid, c_mid, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv2(self.cv1(x))


class Efficient_UAVDet(nn.Module):
    """完整轻量检测头的**独立实现**（可选；主路径见 ``swap_detect_head``）。"""

    def __init__(self, nc: int = 80, ch: tuple = (), per_group: int = 16,
                 channels: str = "native"):
        super().__init__()
        if channels not in ("native", "input"):
            raise ValueError(f"channels 只能是 'native' 或 'input'，实际 {channels!r}")
        from tod.compat import base_conv

        conv = base_conv()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.channels = channels
        self.per_group = per_group
        self.stride = torch.zeros(self.nl)

        c2 = max((16, ch[0] // 4, self.reg_max * 4)) if ch else self.reg_max * 4
        c3 = max(ch[0], nc) if ch else nc
        self.cv2 = nn.ModuleList()
        self.cv3 = nn.ModuleList()
        for x in ch:
            box_mid = x if channels == "input" else c2
            cls_mid = x if channels == "input" else c3
            self.cv2.append(
                nn.Sequential(GroupedStem(x, box_mid, per_group),
                              nn.Conv2d(box_mid, 4 * self.reg_max, 1))
            )
            self.cv3.append(
                nn.Sequential(GroupedStem(x, cls_mid, per_group),
                              nn.Conv2d(cls_mid, nc, 1))
            )
        self._conv = conv

    def forward(self, x):
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        shape = x[0].shape
        return torch.cat(
            [xi.view(shape[0], self.no, -1) for xi in x], 2
        )


def swap_detect_head(head: nn.Module, per_group: int = 16, channels: str = "native",
                     keypoint_branches: bool = True,
                     keypoint_per_group: int | None = None) -> nn.Module:
    """就地替换一个**已构建**的检测头的分支卷积（本库采用的主路径）。"""
    if not (hasattr(head, "cv2") and hasattr(head, "cv3")):
        raise TypeError("传入的对象不是检测头（缺少 cv2/cv3 分支）。")

    kpt_attrs = [f"cv{name}" for name in ("4", "4_kpts", "4_sigma")
                 if hasattr(head, f"cv{name}")]
    # 仅替换分支 stem，保留框架预测层与关键点解码；关键点压缩需显式配置。
    attrs = ["cv2", "cv3"] + (kpt_attrs if keypoint_branches else [])
    stem_plan = {a: per_group for a in ("cv2", "cv3")}
    if keypoint_branches:
        stem_plan.update({a: keypoint_per_group for a in kpt_attrs})

    stem_convs = {}
    for attr in attrs:
        branches = getattr(head, attr, None)
        if branches and isinstance(branches[0], nn.Sequential):
            stem_convs[attr] = sum(count_stem_convs(m) for m in branches[0][:-1])
    head._tod_ep5_stem_convs = stem_convs

    for attr in attrs:
        branches = getattr(head, attr, None)
        if not branches:
            continue
        if not isinstance(branches[0], nn.Sequential):
            continue
        granularity = stem_plan.get(attr)
        new_branches = nn.ModuleList()
        for branch in branches:
            stem = branch[0]
            out_conv = branch[-1]
            c_in = _in_channels(stem)
            c_out_conv = _in_channels(out_conv)
            c_final = _out_channels(out_conv)

            mid = c_in if channels == "input" else c_out_conv
            new_stem = _build_stem(c_in, mid, granularity)
            if mid == c_out_conv:
                new_branch = nn.Sequential(new_stem, out_conv)
            else:
                new_branch = nn.Sequential(new_stem, nn.Conv2d(mid, c_final, 1))
            new_branches.append(new_branch)
        setattr(head, attr, new_branches)
    return head


def _build_stem(c_in: int, c_mid: int, per_group: int | None) -> nn.Module:
    """按粒度构造分支 stem：``None`` = 不压缩（``KeptStem``），否则 ``GroupedStem``。"""
    groups, reason = choose_stem(c_in, c_mid, per_group)
    if reason == "kept":
        return KeptStem(c_in, c_mid)
    if reason == "infeasible":
        raise ValueError(
            f"无法在 in={c_in} / mid={c_mid} 上实现约 {per_group} 通道/组的分组卷积"
            f"（两者没有 >1 的公约数，只能 g=1 = 普通卷积）。"
            "请改 per_group（例如 3），或把 keypoint_per_group 设为 None 表示不压缩。"
        )
    stem = GroupedStem(c_in, c_mid, int(per_group), groups=groups)
    stem._tod_group_reason = reason
    return stem


def _leaf(module: nn.Module, first: bool) -> nn.Module:
    """钻到嵌套 ``Sequential`` 的最前/最后一个卷积（8.4 的分类分支是嵌套结构）。"""
    for _ in range(8):
        if hasattr(module, "conv") or not isinstance(module, nn.Sequential) or not len(module):
            return module
        module = module[0] if first else module[-1]
    return module


def _in_channels(module: nn.Module) -> int:
    conv = getattr(_leaf(module, True), "conv", _leaf(module, True))
    if not hasattr(conv, "in_channels"):
        raise CompatError(
            f"无法从 {type(module).__name__} 推断输入通道："
            "框架的检测头分支结构可能已变化，请在 tod/modules/head/efficient_uavdet.py 适配。"
        )
    return int(conv.in_channels)


def _out_channels(module: nn.Module) -> int:
    conv = getattr(_leaf(module, False), "conv", _leaf(module, False))
    if not hasattr(conv, "out_channels"):
        raise CompatError(
            f"无法从 {type(module).__name__} 推断输出通道："
            "框架的检测头分支结构可能已变化。"
        )
    return int(conv.out_channels)


def count_stem_convs(module: nn.Module) -> int:
    """统计模块里的卷积层数（用于暴露框架/主干之间的头结构差异）。"""
    if isinstance(module, nn.Sequential):
        return sum(count_stem_convs(m) for m in module)
    return 1 if hasattr(module, "conv") or hasattr(module, "weight") else 0


def describe(head: nn.Module) -> str:
    """返回人类可读的头结构摘要（自检/卡片用）。"""
    lines = []
    roles = {"cv2": "回归", "cv3": "分类", "cv4": "关键点",
             "cv4_kpts": "关键点(均值)", "cv4_sigma": "关键点(sigma)"}
    for attr, role in roles.items():
        branches = getattr(head, attr, None)
        if branches is None:
            continue
        for i, branch in enumerate(branches):
            stem = branch[0]
            if not isinstance(stem, nn.Sequential) or not hasattr(stem, "cv1"):
                continue
            g = getattr(stem, "groups", "?")
            c_in = _in_channels(stem.cv1)
            c_mid = _out_channels(stem.cv2)
            per_group = (c_in / g) if isinstance(g, int) and g else 0
            kind = "分组" if isinstance(g, int) and g > 1 else "未压缩"
            lines.append(f"{attr}[{i}] {role}: in={c_in} -> mid={c_mid} (g={g}, "
                         f"{per_group:.1f} ch/group, {kind})")
    stem_convs = getattr(head, "_tod_ep5_stem_convs", {})
    lines.append("本库把分支 stem 换成 2 层分组卷积（论文 §3.4）；替换前 stem 的卷积层数："
                 + "、".join(f"{k}={v}" for k, v in stem_convs.items())
                 + "（YOLOv8 系在 8.4 上仍是 2/2；而 YOLO26 系（非 legacy）的分类分支是"
                 " DWConv+Conv 嵌套两块 = 4 层，被压成 2 层分组卷积属更激进压缩）")
    lines.append("注：关键点分支默认**不压缩**（论文的 g=x/16 只针对检测头，关键点分支是"
                 "姿态扩展；且其中间通道常不被 16 的约数整除，强行分组会退化成 g=1 的"
                 "静默失效）。要压请显式给 keypoint_per_group。")
    return "\n".join(lines)
