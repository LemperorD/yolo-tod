"""EP5 检测头的"建模后手术"：把变体里的头部改动应用到已构建的模型上。

为什么需要它（重要设计决定）：
    ultralytics 的 ``parse_model`` 通过**精确类成员判断**来给检测头追加输入
    通道列表 ``ch``（形如 ``if m in {Detect, Segment, Pose, OBB, ...}``）。
    自定义检测头即使继承自 ``Detect`` 也不在该集合里，直接写进 YAML 会因为
    构造时缺少 ``ch`` 而失败。

    社区常见做法是 fork 一份 ``tasks.py`` 把自己的类加进那个集合——这正是本库
    极力避免的。改为：YAML 里仍是框架原生的 ``Detect``，模型**构建完成后**
    就地替换分支卷积（模块对象级手术）。好处：
      * 不依赖框架内部的类判断实现细节，升级更不容易碎；
      * 末端 1×1 输出卷积可以**原样复用**，预训练权重不丢；
      * 能被 ``tools/train.py --dry-run`` 单独验证。

副作用：手术后的模型若被保存为 checkpoint，加载时进程里必须有
``tod.modules`` 可导入（``tod.bootstrap()`` 会做这件事）。
"""

from __future__ import annotations

from typing import Any

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


def apply_spec(model: Any, spec: dict[str, Any]) -> list[str]:
    """按变体 spec 里的 EP5 配置对模型做手术，返回已应用项的描述。"""
    head_name = ((spec.get("eps") or {}).get("EP5") or {}).get("head")
    if not head_name or head_name == "Detect":
        return []

    if head_name not in ("Efficient_UAVDet", "EfficientUAVDet", "EfficientUAVDetHead"):
        raise CompatError(
            f"EP5 头 {head_name!r} 还没有建模后手术的实现。"
            "请在 tod/engine/surgery.py 中补充，或改用框架原生 Detect。"
        )

    from tod.modules.head.efficient_uavdet import describe, swap_detect_head

    per_group = int(((spec.get("eps") or {}).get("EP5") or {}).get("per_group", 16))
    channels = str(((spec.get("eps") or {}).get("EP5") or {}).get("channels", "native"))
    head = swap_detect_head(find_head(model), per_group=per_group, channels=channels)
    return [f"EP5: {head_name}(per_group={per_group}, channels={channels})",
            describe(head)]
