"""训练引擎扩展：把本库的 EP7/EP9 配置接进框架训练循环。"""

from __future__ import annotations

from typing import Any

__all__ = ["TODDetectionTrainer"]


def __getattr__(name: str) -> Any:  # pragma: no cover
    if name == "TODDetectionTrainer":
        from tod.engine.trainer import TODDetectionTrainer

        return TODDetectionTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
