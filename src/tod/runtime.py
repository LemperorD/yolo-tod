"""运行时上下文：保存"当前正在跑的变体 spec"。"""

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
    # 离开上下文时恢复先前变体，避免嵌套调用串用配置。
    finally:
        set_active(previous)
