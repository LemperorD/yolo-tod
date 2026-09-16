"""EP9 训练策略（优化器 / 蒸馏 / 损失调度）。

本目录的模块不是网络结构，不进 ``parse_model`` 命名空间，而是由
``tod.engine.trainer`` 在训练时读取变体配置后装配。
"""

from tod.optim import musgd  # noqa: F401
from tod.optim.musgd import MuSGD, build, resolve  # noqa: F401

__all__ = ["MuSGD", "build", "resolve"]
