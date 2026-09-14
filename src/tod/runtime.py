"""运行时上下文：保存"当前正在跑的变体 spec"。

为什么需要它：ultralytics 会严格校验训练参数（未知键会被拒），
所以本库的配置不能塞进 ``model.train(**kwargs)`` 里。
改为由 ``tools/train.py`` 先 ``set_active(spec)``，
自定义 Trainer / 损失再从 ``tod.runtime`` 读——两边解耦，互不污染。
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

_ACTIVE: dict[str, Any] | None = None


def set_active(spec: dict[str, Any] | None) -> None:
    global _ACTIVE
    _ACTIVE = spec


def active() -> dict[str, Any]:
    """当前变体 spec（未设置时返回空 dict）。"""
    return _ACTIVE or {}


def get(path: str, default: Any = None) -> Any:
    """按点号路径取值，例如 ``get("eps.EP7.box", "ciou")``。"""
    node: Any = active()
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


@contextmanager
def use(spec: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """临时激活一个 spec（测试用）。"""
    previous = _ACTIVE
    set_active(spec)
    try:
        yield spec
    finally:
        set_active(previous)
