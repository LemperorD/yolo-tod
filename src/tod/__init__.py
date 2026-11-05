"""TOD —— 小目标检测魔改 YOLO 库。"""

import sys

from tod.registry import catalog, get, has, install, list_specs, register

__all__ = ["register", "install", "bootstrap", "import_libraries", "catalog", "get",
           "has", "list_specs", "configure_stdio"]
__version__ = "0.0.1"


def configure_stdio() -> None:
    """把 stdout / stderr 切到 UTF-8，免得 Windows 默认代码页（GBK/cp936）编不出字符。"""
    import os

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, OSError, ValueError):
            pass
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")


def import_libraries() -> None:
    """导入所有注册库（触发 ``@register``），但**不**注入框架命名空间。"""
    import tod.assigner  # noqa: F401
    import tod.engine.distill  # noqa: F401
    import tod.loss  # noqa: F401
    import tod.modules  # noqa: F401
    import tod.optim  # noqa: F401


def bootstrap() -> None:
    """导入模块库并把所有已注册的魔改注入框架命名空间。"""
    configure_stdio()
    # 先完成所有模块注册，再注入框架解析命名空间。
    import_libraries()
    install()
