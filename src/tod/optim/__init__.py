"""EP9 训练策略（优化器 / 蒸馏 / 损失调度）。"""

from tod.optim import musgd  # noqa: F401
from tod.optim.musgd import MuSGD, build, resolve  # noqa: F401

__all__ = ["MuSGD", "build", "resolve"]
