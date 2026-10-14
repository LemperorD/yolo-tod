"""TOD —— 小目标检测魔改 YOLO 库。

设计原则见 PLAN.md：一切魔改通过 registry 注册，通过 compose 组合，
主干框架（ultralytics）只作为加载器与训练引擎，绝不 fork。
"""

import sys

from tod.registry import catalog, get, has, install, list_specs, register

__all__ = ["register", "install", "bootstrap", "import_libraries", "catalog", "get",
           "has", "list_specs", "configure_stdio"]
__version__ = "0.0.1"


def configure_stdio() -> None:
    """把 stdout / stderr 切到 UTF-8，免得 Windows 默认代码页（GBK/cp936）编不出字符。

    为什么需要：Windows 上 ``sys.stdout.encoding`` 常是 ``gbk``，一旦输出被重定向到
    文件、被管道接走或被 CI 捕获，``print()`` 碰到 GBK 之外的字符就会
    ``UnicodeEncodeError`` 让整条命令失败；同样的代码在 Linux（UTF-8）上却正常，
    属于"只在某个平台炸"的隐蔽故障。

    做法：优先 ``reconfigure``；失败就静默忽略（这只是输出层面的改善，
    不该让业务逻辑挂掉）。``errors="replace"`` 保证最坏情况显示成 ``?`` 而不是中断。
    """
    import os

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, OSError, ValueError):
            pass
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")


def import_libraries() -> None:
    """导入所有注册库（触发 ``@register``），但**不**注入框架命名空间。

    需要"拿到完整注册表"但不该依赖 ultralytics 的场景（变体卡片、模块总表）
    用这个；训练入口用 ``bootstrap()``。
    """
    import tod.assigner  # noqa: F401
    import tod.engine.distill  # noqa: F401
    import tod.loss  # noqa: F401
    import tod.modules  # noqa: F401
    import tod.optim  # noqa: F401


def bootstrap() -> None:
    """导入模块库并把所有已注册的魔改注入框架命名空间。

    训练入口（tools/train.py）必须先调用它，否则 YAML 里的魔改模块名
    无法被框架解析。刻意分成"导入模块"与"注入"两步：
    单测可以直接 import registry 而不触发 torch 依赖。

    注意：必须**先**把各子库都导入（触发 ``@register``），**再** ``install()``。
    早期版本只导入了 ``tod.modules``，结果 EP6/EP9 的模块（STAL、MuSGD、
    FeatureAlignKD）虽然注册了，却不在 parse_model 命名空间里。
    """
    configure_stdio()
    import_libraries()
    install()
