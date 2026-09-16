"""TOD —— 小目标检测魔改 YOLO 库。

设计原则见 PLAN.md：一切魔改通过 registry 注册，通过 compose 组合，
主干框架（ultralytics）只作为加载器与训练引擎，绝不 fork。
"""

from tod.registry import catalog, get, has, install, list_specs, register

__all__ = ["register", "install", "bootstrap", "catalog", "get", "has", "list_specs"]
__version__ = "0.0.1"


def bootstrap() -> None:
    """导入模块库并把所有已注册的魔改注入框架命名空间。

    训练入口（tools/train.py）必须先调用它，否则 YAML 里的魔改模块名
    无法被框架解析。刻意分成"导入模块"与"注入"两步：
    单测可以直接 import registry 而不触发 torch 依赖。

    注意：必须**先**把各子库都导入（触发 ``@register``），**再** ``install()``。
    早期版本只导入了 ``tod.modules``，结果 EP6/EP9 的模块（STAL、MuSGD、
    FeatureAlignKD）虽然注册了，却不在 parse_model 命名空间里。
    """
    import tod.assigner  # noqa: F401  （导入即触发 @register）
    import tod.engine.distill  # noqa: F401
    import tod.loss  # noqa: F401
    import tod.modules  # noqa: F401
    import tod.optim  # noqa: F401

    install()
