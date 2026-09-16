"""魔改模块库（零件库）。

约定（PLAN.md §2 / §4.1）：
  * 一个魔改 = 一个文件 + 一个 ``@register`` 装饰器 + 完整元数据；
  * 模块只做"算子/块/损失"本身，不关心它被放在哪张图的哪个位置；
  * 目录按扩展点分层：conv/ block/ attention/ neck/ upsample/ head/ ...

新增模块时请同时：
  1. 在 ``tests/test_modules.py`` 增加形状/数值测试；
  2. 跑 ``python tools/catalog.py --write`` 刷新 docs/VARIANTS.md。
"""

# 导入即注册。集中列在这里，保证 ``import tod.modules`` 能拿到全部零件。
from tod.modules.attention import dual_attention  # noqa: F401
from tod.modules.conv import adown  # noqa: F401
from tod.modules.head import efficient_uavdet  # noqa: F401
