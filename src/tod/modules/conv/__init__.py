"""EP1 Backbone 相关魔改：下采样算子、卷积块。"""

from tod.modules.conv import adown  # noqa: F401

__all__ = ["adown"]
