"""TOD —— 小目标检测魔改 YOLO 库。

设计原则见 PLAN.md：一切魔改通过 registry 注册，通过 compose 组合，
主干框架（ultralytics）只作为加载器与训练引擎，绝不 fork。
"""

from tod.registry import catalog, get, install, list_specs, register

__all__ = ["register", "install", "catalog", "get", "list_specs"]
__version__ = "0.0.1"
