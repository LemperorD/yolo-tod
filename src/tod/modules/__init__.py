"""魔改模块库（零件库）。

约定（PLAN.md §2 / §4.1）：
  * 一个魔改 = 一个文件 + 一个 ``@register`` 装饰器 + 完整元数据；
  * 模块只做"算子/块/损失"本身，不关心它被放在哪张图的哪个位置；
  * 目录按扩展点分层：conv/ block/ attention/ neck/ upsample/ head/ ...

M0 阶段本目录为空：先打通算力与环境，再按 PLAN.md §5 的 P0 清单逐项落地。
加入第一个模块时，请同时：
  1. 在 ``tests/smoke.py`` 增加该模块的形状/数值测试；
  2. 跑 ``python -m tod.registry > docs/VARIANTS.md`` 刷新总表。
"""
