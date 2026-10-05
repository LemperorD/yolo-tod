"""Efficient_UAVDet —— 基于分组卷积的轻量检测头（SPAE-YOLOv8 §3.4）。

论文：Rushang Zhang, Xiaogang Fu, "SPAE-YOLOv8 for Onboard Real-Time Perception:
Lightweight Small UAV Detection from Air-to-Air Perspectives", Sensors 2026, 26(11):3424.
DOI: 10.3390/s26113424（CC BY 4.0）

原文机制（§3.4，式 10–11，Figure 4，Table 3）：
    "The standard convolutions located before the regression and classification
     branches of the original detection head are substituted with channel-adaptive
     grouped convolution."

    * 只替换**分类/回归分支之前的两层标准 3x3 卷积**（stem），末端 1x1 输出卷积保持普通卷积；
    * 两层连续 3x3 分组卷积，卷积核 k=3、stride=1、padding=1，两层规格相同；
    * 分组数 **g = x / 16**（x 为该分支输入通道数），即**每组固定 16 通道**；
    * 分类分支与回归分支**不共享**卷积（Figure 4 为两条并行对称分支）；
    * **不是**深度可分离卷积（论文中 "depthwise separable" 只出现在 Table 1 对 MobileNet 的综述里）。

论文 Table 3 对 YOLOv8n（含 P2）给出的分组配置：

    ======  ====  =============  ==========
    尺度     x     g = x/16       每组通道
    ======  ====  =============  ==========
    P2       32    2              16
    P3       64    4              16
    P4      128    8              16
    P5      256   16              16
    ======  ====  =============  ==========

    ↳ 当 P2 融合块输出 128 通道（width=0.25 缩放后为 32）时，
      YOLOv8n 检测头输入通道恰为 32/64/128/256，与 Table 3 完全吻合。
      这也是本库 SPAE 配方取 ``p2_channels=128`` 的依据。

【必须知道的代价】论文 §4.5 明确承认此头"降本不涨价"：
    "Efficient_UAVDet is mainly designed for model compression and inference
     acceleration rather than direct accuracy improvement."
    消融：单独换头 mAP@0.5 0.850 → 0.846（−0.4pp），但 Params 3.0M → 2.4M、
    FPS 197.2 → 252.9。论文把精度略降归因于分组卷积限制了组间信息交互，
    并提出未来用 channel shuffle 补偿 —— **当前版本不含 channel shuffle**。

【通道策略的一处不确定】原文只说"替换分支前的标准卷积"，未明说替换后通道数
是否改变。本实现给出两种策略，默认取"就地替换"：

    * ``channels="native"``（默认）：保持框架原生检测头的分支通道（c2/c3）不变，
      只把卷积换成分组卷积 —— 与"Butt 替换"的字面表述、以及论文报告的参数量
      降幅最吻合；
    * ``channels="input"``：分支内 in=out=x（严格保持输入通道），对应 Figure 4
      "两层规格相同"的另一种读法。

两种策略在 Table 3 的 x 与 g 上都能对上，无法从公开材料进一步区分。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tod.compat import CompatError
from tod.registry import register


def _valid_groups(c_in: int, c_out: int, per_group: int, *, allow_degenerate: bool = True) -> int:
    """求满足"每组约 per_group 通道"且能整除输入/输出通道的最大分组数。

    论文给的是 ``g = x / 16``；当 x 不是 16 的整数倍时严格公式不可用，
    这里向下取到能整除的 g，并保证 g >= 1（g=1 即退化为普通卷积）。

    ⚠️ **g=1 是"静默失效"**：调用方以为压了，其实一个通道都没压（参数、FLOPs、
    分组语义全都没变）。``allow_degenerate=False`` 时宁可报错，也不要假装压过 ——
    关键点分支就踩过这个坑（见 ``swap_detect_head`` 的 ``keypoint_per_group``）。
    """
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
    """取不超过首选值、且能同时整除 ``c_in`` 与 ``c_mid`` 的最大分组数。

    首选值来自论文的 ``g = x / per_group``（例如 x=32、per_group=16 → 2）。
    """
    g = min(int(prefer), int(c_in), int(c_mid))
    while g > 1 and (int(c_in) % g or int(c_mid) % g):
        g -= 1
    return max(1, g)


def choose_stem(c_in: int, c_mid: int, requested: int | None) -> tuple[int, str]:
    """决定分支 stem 的**实际**分组数与理由（供日志与断言）。

    论文的规则是 ``g = x / per_group``（每组约 per_group 通道）。这个规则在关键点
    分支上常常**不可行**：``c4 = max(ch[0]//4, nk)`` 与输入通道可能没有足够大的
    公约数（9 点：in=32 / mid=27 → 最大公约数 1）。

    因此：
      * ``requested is None`` → 不压缩（``groups=1``，理由 ``"kept"``）；
      * 否则先按论文算首选值 ``g = c_in // requested``，再降到可行的最大 g：
          - 恰好等于首选值 → 理由 ``"ok"``；
          - 被降级（仍 > 1）→ 理由 ``"auto_degraded"``，**必须在日志里点出来**；
          - 只能 g=1 → 理由 ``"infeasible"``，调用方**报错**而不是静默失效。

    Returns:
        ``(groups, reason)``，``reason ∈ {"kept", "ok", "auto_degraded", "infeasible"}``。
    """
    if requested is None:
        return 1, "kept"
    prefer = max(1, int(c_in) // int(requested))
    best = _best_groups(c_in, c_mid, prefer)
    if best <= 1:
        return 1, "infeasible"
    return (best, "ok") if best == prefer else (best, "auto_degraded")


class GroupedStem(nn.Module):
    """检测头分支 stem：两层连续的 3x3 分组卷积（k=3, s=1, p=1，组内约 16 通道）。

    Args:
        c_in: 分支输入通道（用于确定分组数 g = c_in // per_group）。
        c_mid: stem 输出通道（``channels="native"`` 时为框架原生的 c2/c3）。
        per_group: 每组通道数，论文取 16。
        groups: 显式指定分组数；``None`` 时按 ``c_in // per_group`` 推算。
    """

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
    """**不压缩**的分支 stem：两层连续的标准 3×3 卷积，接口与 ``GroupedStem`` 一致。

    为什么需要它（实测踩到的坑）：论文的 ``g = x/16`` 只对**检测头**成立。姿态头的
    关键点分支中间通道是 ``c4 = max(ch[0]//4, nk)``；当关键点数让 c4 不被 2 整除时
    （例如 9 点 → c4=27），``_valid_groups`` 只能退化成 **g=1 = 普通卷积**，
    此时再挂 ``GroupedStem`` 就是"看起来压了、其实一个通道都没压"。
    与其留一个静默失效的开关，不如显式提供一个"明确不压"的实现：
    参数、FLOPs 与语义都是原生的，写进日志也不会骗人。
    """

    def __init__(self, c_in: int, c_mid: int, per_group: int = 16):
        super().__init__()
        from tod.compat import base_conv

        conv = base_conv()
        self.groups = 1
        self.per_group = per_group                 # 仅记录调用方意图，供日志核对
        self.cv1 = conv(c_in, c_mid, 3, 1, 1)
        self.cv2 = conv(c_mid, c_mid, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv2(self.cv1(x))


class Efficient_UAVDet(nn.Module):
    """完整轻量检测头的**独立实现**（可选；主路径见 ``swap_detect_head``）。

    为什么主路径不是"在 YAML 里把 Detect 换成这个类"：
    ultralytics 的 ``parse_model`` 用**精确类成员判断**（``m in {Detect, ...}``）
    来给检测头追加输入通道列表 ``ch``，子类不会被识别，直接写进 YAML 会缺少 ``ch``。
    因此本库改为在模型构建完成后**就地替换**分支卷积（EP5 surgery），
    既不改框架源码，也不依赖该判断的实现细节。

    本类保留完整实现，供"框架会把 ch 传给子类"的场景（如自行 fork 的
    tasks.py）或单元测试直接使用。参数与原版 ``Detect`` 保持同构：
    ``cv2`` = 回归分支，``cv3`` = 分类分支，末端均为普通 1x1 卷积。
    """

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
        self._conv = conv  # 供外部（如导出/初始化）参考

    def forward(self, x):
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        shape = x[0].shape
        return torch.cat(
            [xi.view(shape[0], self.no, -1) for xi in x], 2
        )


# --------------------------------------------------------------------- 主路径


def swap_detect_head(head: nn.Module, per_group: int = 16, channels: str = "native",
                     keypoint_branches: bool = True,
                     keypoint_per_group: int | None = None) -> nn.Module:
    """就地替换一个**已构建**的检测头的分支卷积（本库采用的主路径）。

    只替换各分支里**输出层之前的卷积**，末端 1x1 输出卷积**原样复用**
    （因此预训练权重不受影响）。

    Args:
        head: 框架的 ``Detect``（或结构同构的）实例；姿态头（``Pose``/``Pose26``）
            也支持 —— 见 ``keypoint_branches`` / ``keypoint_per_group``。
        per_group: 论文的每组通道数 16（**框/分类分支**用）。
        channels: ``"native"`` 保持原生分支通道；``"input"`` 令分支内 in=out=x。
        keypoint_branches: 姿态头是否**处理关键点分支**（``cv4`` 等）。``False`` 时
            关键点分支原样不动。
        keypoint_per_group: 关键点分支的每组通道数。
            * ``None``（**默认**）—— 关键点分支**不压缩**（换成 ``KeptStem``，
              等价于原生两层标准卷积，但走本库的 stem 接口以便统一描述/统计）。
              为什么默认不压：论文的 ``g = x/16`` 是针对**检测头**给出的，
              关键点分支是姿态任务的扩展，论文对它没有定义；而且关键点分支的中间
              通道 ``c4 = max(ch[0]//4, nk)`` 往往与输入通道没有 >1 的公约数
              （9 点 → c4=27，与 32 的公约数只有 1），强行分组只会退化成 g=1，
              是**静默失效**。
            * 正整数 —— 把它当作"每组通道数**上限**"：实际取 ≤ 该值且能整除
              in/mid 的最大 g，并在日志里标注是否发生了退化（``auto_degraded``）；
              完全不可行（只能 g=1）时**报错**，绝不悄悄放行。
              例：9 点、in=32/mid=27 时 ``keypoint_per_group=3`` → g=3。

    Returns:
        同一个 head 对象（就地修改）。
    """
    if not (hasattr(head, "cv2") and hasattr(head, "cv3")):
        raise TypeError("传入的对象不是检测头（缺少 cv2/cv3 分支）。")

    # 关键点分支：默认不压；显式给了 per_group 才压（且不允许静默退化为 g=1）
    kpt_attrs = [f"cv{name}" for name in ("4", "4_kpts", "4_sigma")
                 if hasattr(head, f"cv{name}")]
    attrs = ["cv2", "cv3"] + (kpt_attrs if keypoint_branches else [])
    stem_plan = {a: per_group for a in ("cv2", "cv3")}
    if keypoint_branches:
        stem_plan.update({a: keypoint_per_group for a in kpt_attrs})

    # 记录替换前各分支 stem（输出 1x1 之前的全部卷积）的卷积层数：
    # 8.2/8.3 的 cv2/cv3 都是 2；8.4 的 cv3 变成 "DWConv+Conv" 嵌套两块 = 4 层。
    # 注意有些分支**根本不是 Sequential**（Pose26 的 ``cv4_kpts``/``cv4_sigma`` 是
    # 单层 ``nn.Conv2d``），那种分支没有 stem 可换，直接跳过 —— 否则会
    # ``TypeError: 'Conv2d' object is not subscriptable``。
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
            continue               # 单层卷积分支（Pose26 的 kpts/sigma）没有 stem
        granularity = stem_plan.get(attr)
        new_branches = nn.ModuleList()
        for branch in branches:
            stem = branch[0]
            out_conv = branch[-1]
            c_in = _in_channels(stem)
            c_out_conv = _in_channels(out_conv)      # 原生分支的中间通道
            c_final = _out_channels(out_conv)

            mid = c_in if channels == "input" else c_out_conv
            new_stem = _build_stem(c_in, mid, granularity)
            if mid == c_out_conv:
                # 通道不变：直接复用原 1x1 输出卷积（保留预训练权重）
                new_branch = nn.Sequential(new_stem, out_conv)
            else:
                new_branch = nn.Sequential(new_stem, nn.Conv2d(mid, c_final, 1))
            new_branches.append(new_branch)
        setattr(head, attr, new_branches)
    return head


def _build_stem(c_in: int, c_mid: int, per_group: int | None) -> nn.Module:
    """按粒度构造分支 stem：``None`` = 不压缩（``KeptStem``），否则 ``GroupedStem``。

    语义（与 ``choose_stem`` 一致）：
      * 粒度 ``None`` → 不压（如实报告 groups=1）；
      * 粒度 = 目标每组通道数 → 取 ≤ 该粒度的最大可行 g；若只能 g=1，则**报错**
        （避免"配了压缩、其实没压"的假消融）。
    """
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
    stem._tod_group_reason = reason          # auto_degraded 会在日志里被点出来
    return stem


def _leaf(module: nn.Module, first: bool) -> nn.Module:
    """钻到嵌套 ``Sequential`` 的最前/最后一个卷积（8.4 的分类分支是嵌套结构）。"""
    for _ in range(8):                                # 防御性上限，避免意外深递归
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
    """统计模块里的卷积层数（用于暴露框架/主干之间的头结构差异）。

    YOLOv8 系（8.2–8.4）的分类分支 stem 是 2 层 3×3 卷积 —— 与论文 Efficient_UAVDet
    的"两层"完全对应；YOLO26 系（非 legacy）的分类分支改成 `DWConv+Conv` 嵌套两块 = 4 层，
    本库的换头仍然只放两层分组卷积，因此在新主干上属于"比论文更激进的压缩"。
    这个计数会被记录到 `head._tod_ep5_stem_convs` 并出现在 `describe()` 与日志里。
    """
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
                continue                       # 单层卷积分支（Pose26 的 kpts/sigma）
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
