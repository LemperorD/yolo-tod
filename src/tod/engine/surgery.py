"""EP5 检测头 / EP4 注意力的"建模后手术"：把变体改动应用到已构建的模型上。

为什么需要它（重要设计决定）：
    ultralytics 的 ``parse_model`` 通过**精确类成员判断**来给检测头追加输入
    通道列表 ``ch``（形如 ``if m in {Detect, Segment, Pose, OBB, ...}``）。
    自定义检测头即使继承自 ``Detect`` 也不在该集合里，直接写进 YAML 会因为
    构造时缺少 ``ch`` 而失败。

    社区常见做法是 fork 一份 ``tasks.py`` 把自己的类加进那个集合——这正是本库
    极力避免的。改为：YAML 里仍是框架原生的 ``Detect``，模型**构建完成后**
    就地做模块对象级手术。好处：
      * 不依赖框架内部的类判断实现细节，升级更不容易碎；
      * 末端 1×1 输出卷积可以**原样复用**，预训练权重不丢；
      * 能被 ``tools/train.py --dry-run`` 单独验证。

两类手术：
    * **EP5 检测头**：``swap_detect_head`` 替换分支 stem 卷积（见 tod/modules/head/）；
    * **EP4 注意力**：``attach_head_attention`` 在每个检测头输入前**插入**一个注意力
      节点（SDD-YOLO §4.5："embed dual attention modules between the backbone and
      detection heads"）。

EP4 为什么是"插入节点"而不是"包在既有节点外面"：
    脖子里的 P3/P4 输出同时被检测头和自底向上路径消费（如 YOLO26 的节点 16 既进
    Detect 又进下一级 Conv）。若把注意力包在被共享的节点外面，注意力就会**顺带**
    改动颈部聚合——那不是论文的做法，也会让消融归因变脏。所以这里在 ``Detect``
    之前插入新节点，并把检测头的输入索引指向它们：颈部其余部分**一个张量都不变**。

副作用：手术后的模型若被保存为 checkpoint，加载时进程里必须有
``tod.modules`` 可导入（``tod.bootstrap()`` 会做这件事）。
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch.nn as nn

from tod.compat import CompatError


def find_head(model: Any) -> nn.Module:
    """从 DetectionModel / nn.Sequential 中找到检测头模块。"""
    if isinstance(model, nn.Module) and hasattr(model, "cv2") and hasattr(model, "cv3"):
        return model
    inner = getattr(model, "model", None)
    if isinstance(inner, nn.Sequential) and len(inner):
        head = inner[-1]
        if hasattr(head, "cv2") and hasattr(head, "cv3"):
            return head
    raise CompatError(
        "未能定位检测头（需要具备 cv2/cv3 分支的模块）。"
        "框架结构可能已变更，请检查 src/tod/engine/surgery.py。"
    )


# ------------------------------------------------------------------ EP4 注意力


def stride_to_level(stride: int) -> int:
    """stride → 金字塔层号：4→P2、8→P3、16→P4、32→P5。"""
    return int(round(math.log2(int(stride))))


def head_input_channels(head: nn.Module) -> list[int]:
    """读取检测头每一层的输入通道数。

    框架的 ``Detect``（8.4）**不再保存** ``self.ch``，所以从分支首个卷积反推：
    ``cv2[i][0].conv.in_channels``；退化情况下用权重形状 × groups。
    """
    branches = getattr(head, "cv2", None)
    if not branches:
        raise CompatError("检测头没有 cv2 分支，无法推断输入通道。")
    chans: list[int] = []
    for branch in branches:
        first = branch[0]
        conv = getattr(first, "conv", None)
        if conv is not None and hasattr(conv, "in_channels"):
            chans.append(int(conv.in_channels))
            continue
        weight = next(first.parameters(), None)
        if weight is None:
            raise CompatError(f"无法从 {type(first).__name__} 推断输入通道。")
        groups = int(getattr(first, "groups", 1) or 1)
        chans.append(int(weight.shape[1]) * groups)
    return chans


def _iter_attention(seq: nn.Sequential) -> list[nn.Module]:
    return [m for m in seq if getattr(m, "_tod_ep4", None)]


def attach_head_attention(model: Any, *, module: str = "DualAttention",
                          levels: Sequence[int] | None = None,
                          **kwargs: Any) -> list[str]:
    """在检测头每个输入前插入注意力节点；返回可读的改动描述。

    实现步骤（保持颈部其余节点不变）::

        ... neck 节点 ...            # 索引 0..k-1，原样不动
        [f_i] -> Attn_i              # 新增，索引 k..k+n-1
        Detect(输入索引改为 k..k+n-1)  # 原先是 f_i

    Args:
        model: DetectionModel（有 ``.model`` 这个 Sequential）。
        module: 注册表中的注意力模块名（如 ``"DualAttention"``）。
        levels: 只对这些层加注意力（4→P2、8→P3、16→P4、32→P5）；``None`` = 所有头输入。
        kwargs: 传给注意力模块构造函数的额外参数（如 ``reduction=16``）。

    Returns:
        改动描述列表（空列表表示未改动）。
    """
    from tod.registry import get as get_spec

    seq = getattr(model, "model", None)
    if not isinstance(seq, nn.Sequential) or len(seq) < 2:
        raise CompatError("attach_head_attention 需要 DetectionModel（含 .model Sequential）。")
    head = find_head(model)

    targets = getattr(head, "f", None)
    if not isinstance(targets, (list, tuple)) or not targets:
        raise CompatError(f"检测头的输入索引不可用（f={targets!r}）。")
    strides = [int(s) for s in getattr(head, "stride", [])]
    if len(strides) != len(targets):
        raise CompatError(
            f"检测头 stride 数（{len(strides)}）与输入数（{len(targets)}）不一致。"
        )

    spec = get_spec(module)
    ctor = spec.obj
    chans = head_input_channels(head)

    # 选择要加注意力的输入位置（按层号过滤）
    picked = [i for i, s in enumerate(strides)
              if levels is None or stride_to_level(s) in set(levels)]
    if not picked:
        return []

    attn = [ctor(chans[i], **kwargs) for i in picked]

    modules = list(seq)
    k = len(modules) - 1                      # Detect 当前所在索引（约定：头在最后）
    if modules[k] is not head:
        raise CompatError("检测头不是 Sequential 的最后一个模块，无法安全插入注意力。")

    n = len(attn)
    # 新索引：第 j 个注意力节点占据 k+j
    new_head_f = list(targets)
    for j, i in enumerate(picked):
        new_head_f[i] = k + j

    for j, (i, m) in enumerate(zip(picked, attn)):
        m.i = k + j                            # 图索引（框架的 _predict_once 会用到）
        m.f = targets[i]                       # 输入仍来自原颈部节点
        m.type = f"{spec.name}({strides[i]})"
        m.np = sum(p.numel() for p in m.parameters())
        m._tod_ep4 = spec.name

    new_seq = nn.Sequential(*modules[:k], *attn, *modules[k:])
    model.model = new_seq
    head.f = new_head_f
    head.i = k + n

    save = getattr(model, "save", None)
    if isinstance(save, list):
        shifted = {(i + n if i >= k else i) for i in save}
        model.save = sorted(shifted | {k + j for j in range(n)})

    desc = [f"EP4: {spec.name} × {n}"
            f"（层 {'/'.join(f'P{stride_to_level(strides[i])}' for i in picked)}，"
            f"通道 {'/'.join(str(chans[i]) for i in picked)}）"]
    return desc


# ------------------------------------------------------------------ 总入口


def apply_spec(model: Any, spec: dict[str, Any]) -> list[str]:
    """按变体 spec 里的 EP4/EP5 配置对模型做手术，返回已应用项的描述。"""
    applied: list[str] = []
    eps = spec.get("eps") or {}

    # --- EP4 先做：此时检测头分支还是原生卷积，反推输入通道最可靠 ---
    ep4 = eps.get("EP4") or {}
    attention = ep4.get("attention")
    if isinstance(attention, str) and attention and attention != "None":
        levels = ep4.get("levels")
        kwargs = {k: v for k, v in ep4.items() if k not in ("attention", "levels")}
        if _iter_attention(model.model if hasattr(model, "model") else model):
            applied.append(f"EP4: {attention} 已存在，跳过（幂等）")
        else:
            applied += attach_head_attention(model, module=attention, levels=levels, **kwargs)

    # --- EP5 检测头手术 ---
    head_name = (eps.get("EP5") or {}).get("head")
    if isinstance(head_name, str) and head_name and head_name != "Detect":
        applied += _apply_head(model, head_name, eps.get("EP5") or {})
    return applied


def _apply_head(model: Any, head_name: str, ep5: dict[str, Any]) -> list[str]:
    if head_name not in ("Efficient_UAVDet", "EfficientUAVDet", "EfficientUAVDetHead"):
        raise CompatError(
            f"EP5 头 {head_name!r} 还没有建模后手术的实现。"
            "请在 tod/modules/head/ 中补充换头函数，并在 tod/engine/surgery.py 里接线。"
        )

    from tod.modules.head.efficient_uavdet import describe, swap_detect_head

    per_group = int(ep5.get("per_group", 16))
    channels = str(ep5.get("channels", "native"))
    head = swap_detect_head(find_head(model), per_group=per_group, channels=channels)
    return [f"EP5: {head_name}(per_group={per_group}, channels={channels})", describe(head)]
