"""训练引擎扩展：把本库的 EP7/EP9 配置接进框架训练循环。

``TODDetectionTrainer`` 采用**延迟导入**（PEP 562）：它需要 ultralytics，
而 ``tod.engine.distill`` / ``tod.engine.surgery`` 只需要 torch。
若在这里直接 import，未安装框架的环境连蒸馏/手术模块都拿不到，
tools/catalog.py 的总表也会缺条目。
"""

from __future__ import annotations

from typing import Any

__all__ = ["TODDetectionTrainer"]


def __getattr__(name: str) -> Any:  # pragma: no cover - 延迟导入分支
    if name == "TODDetectionTrainer":
        from tod.engine.trainer import TODDetectionTrainer

        return TODDetectionTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
