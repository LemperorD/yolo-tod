"""EP4 注意力模块（零件库）。

约定：注意力模块**不改变通道数与分辨率**，因此既可在 YAML 里插节点，
也可由 ``tod.engine.surgery`` 原位包一层（见 AttentionWrapper）。
"""

from tod.modules.attention import dual_attention  # noqa: F401
