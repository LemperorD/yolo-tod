"""EP5 检测头 / EP4 注意力的"建模后手术"：把变体改动应用到已构建的模型上。"""

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


def stride_to_level(stride: int) -> int:
    """stride → 金字塔层号：4→P2、8→P3、16→P4、32→P5。"""
    return int(round(math.log2(int(stride))))


def head_input_channels(head: nn.Module) -> list[int]:
    """读取检测头每一层的输入通道数。"""
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
    """在检测头每个输入前插入注意力节点；返回可读的改动描述。"""
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

    picked = [i for i, s in enumerate(strides)
              if levels is None or stride_to_level(s) in set(levels)]
    if not picked:
        return []

    attn = [ctor(chans[i], **kwargs) for i in picked]

    modules = list(seq)
    k = len(modules) - 1
    if modules[k] is not head:
        raise CompatError("检测头不是 Sequential 的最后一个模块，无法安全插入注意力。")

    n = len(attn)
    new_head_f = list(targets)
    for j, i in enumerate(picked):
        new_head_f[i] = k + j

    for j, (i, m) in enumerate(zip(picked, attn)):
        m.i = k + j
        m.f = targets[i]
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


def apply_spec(model: Any, spec: dict[str, Any]) -> list[str]:
    """按变体 spec 里的 EP4/EP5 配置对模型做手术，返回已应用项的描述。"""
    applied: list[str] = []
    eps = spec.get("eps") or {}

    ep4 = eps.get("EP4") or {}
    attention = ep4.get("attention")
    if isinstance(attention, str) and attention and attention != "None":
        levels = ep4.get("levels")
        kwargs = {k: v for k, v in ep4.items() if k not in ("attention", "levels")}
        if _iter_attention(model.model if hasattr(model, "model") else model):
            applied.append(f"EP4: {attention} 已存在，跳过（幂等）")
        else:
            applied += attach_head_attention(model, module=attention, levels=levels, **kwargs)

    head_name = (eps.get("EP5") or {}).get("head")
    if isinstance(head_name, str) and head_name and head_name != "Detect":
        applied += _apply_head(model, head_name, eps.get("EP5") or {})
    return applied


def _apply_head(model: Any, head_name: str, ep5: dict[str, Any]) -> list[str]:
    head = find_head(model)
    if getattr(head, "_tod_ep5", None):
        return [f"EP5: {head._tod_ep5} 已存在，跳过（幂等）"]
    if head_name not in ("Efficient_UAVDet", "EfficientUAVDet", "EfficientUAVDetHead"):
        raise CompatError(
            f"EP5 头 {head_name!r} 还没有建模后手术的实现。"
            "请在 tod/modules/head/ 中补充换头函数，并在 tod/engine/surgery.py 里接线。"
        )

    from tod.modules.head.efficient_uavdet import describe, swap_detect_head

    per_group = int(ep5.get("per_group", 16))
    channels = str(ep5.get("channels", "native"))
    kpt_branches = bool(ep5.get("keypoint_branches", True))
    kpt_per_group = ep5.get("keypoint_per_group")
    kpt_per_group = None if kpt_per_group is None else int(kpt_per_group)
    head = swap_detect_head(head, per_group=per_group, channels=channels,
                            keypoint_branches=kpt_branches,
                            keypoint_per_group=kpt_per_group)
    head._tod_ep5 = head_name
    return [f"EP5: {head_name}(per_group={per_group}, channels={channels}, "
            f"keypoint_branches={kpt_branches}, keypoint_per_group={kpt_per_group})",
            describe(head)]
