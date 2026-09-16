"""MuSGD —— Muon（Newton–Schulz 正交化）+ SGD 的混合优化器（SDD-YOLO §4.6，式 (5)）。

论文：SDD-YOLO §4.6 "YOLO26's MuSGD Hybrid Training Strategy"：

    "MuSGD applies gradient orthogonalization via Newton-Schulz iteration to
     high-dimensional weight matrices in the backbone. This ensures that the update
     direction remains orthonormal, effectively preserving the feature expressiveness
     even under sparse supervision. Formally, for a backbone weight matrix W, the
     update rule is defined as  G' = NS(G),  W <- W - eta G'   (式 (5))
     ... To maintain training stability for non-matrix parameters, standard SGD with
     momentum is retained for one-dimensional tensors (e.g., biases and
     normalization layers)."

    出处：Muon 由 Moonshot AI 提出（arXiv:2502.16982）；YOLO26 把它做成
    "分解式更新"（decomposed update）用于实时检测训练。

【本库与框架的分工】ultralytics 8.4 起自带 ``ultralytics.optim.MuSGD``
（``optim/muon.py``），实现了同一条式 (5)（NS 正交化 + SGD 分量，``muon``/``sgd`` 两个
系数混合）。因此：

    * **首选**框架原生实现（``resolve()`` 会返回它），训练器只负责把参数分组交过去；
    * 本文件提供**独立实现**，用于 (a) 框架 < 8.4、(b) 需要与框架做数值对照、
      (c) 想在 YAML/配置里显式指名优化器。两者更新公式一致，见 tests/test_modules.py 的对照测试。

实现细节（与框架实现逐条对齐，便于对照）：
    * NS 迭代取 5 步、系数 (3.4445, -4.7750, 2.0315)；行数 > 列数时先转置再算；
    * ``momentum``（β）同时用作 Muon 分量的 EMA 系数与 SGD 分量的动量；
      默认 ``nesterov=True``，与框架 ``build_optimizer`` 里对 MuSGD 的取值一致；
    * 4D 卷积核按 ``(out, in*k*k)`` 展平成 2D 再正交化，随后按
      ``sqrt(max(1, rows/cols))`` 缩放（RMS 匹配）；
    * 混合模式只对 SGD 分量施加 weight decay（框架同样如此）。

⚠️ 与框架原生路径的一处差异（已知，不影响主流程）：框架 ``build_optimizer`` 在
MuSGD 下会把 YOLO26 辅助头（``proto.semseg`` / ``SegmentationHead`` 等）的参数学习率
放大 3 倍；本文件的 ``param_groups()`` 不做这件事（本库不训练分割辅助头）。
"""

from __future__ import annotations

from typing import Any, Iterable

import torch
import torch.nn as nn

from tod.registry import register

#: NS 迭代的超参（Muon 原论文的 "quintic" 系数）。
NS_COEFFS = (3.4445, -4.7750, 2.0315)
NS_STEPS = 5


def zeropower_newton_schulz(g: torch.Tensor, steps: int = NS_STEPS,
                            eps: float = 1e-7) -> torch.Tensor:
    """用 Newton–Schulz 迭代求矩阵的近似正交化（近似 ``U V^T``）。"""
    if g.ndim != 2:
        raise ValueError(f"Newton–Schulz 只接受 2D 张量，实际 {tuple(g.shape)}。")
    x = g.to(torch.float32)
    x = x / (x.norm() + eps)
    transposed = g.size(0) > g.size(1)
    if transposed:
        x = x.T
    a, b, c = NS_COEFFS
    for _ in range(steps):
        aa = x @ x.T
        bb = b * aa + c * (aa @ aa)
        x = a * x + bb @ x
    return x.T if transposed else x


def muon_update(grad: torch.Tensor, momentum_buffer: torch.Tensor, beta: float = 0.95,
                nesterov: bool = True) -> torch.Tensor:
    """式 (5) 的 ``G' = NS(G)``：先聚动量，再多步正交化，最后做 RMS 缩放。"""
    momentum_buffer.lerp_(grad, 1 - beta)                      # buf = β·buf + (1-β)·g
    update = grad.lerp(momentum_buffer, beta) if nesterov else momentum_buffer
    if update.ndim == 4:                                        # 卷积核：展平后正交化
        update = update.view(len(update), -1)
    update = zeropower_newton_schulz(update)
    update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    return update


@register(
    name="MuSGD",
    ep="EP9",
    paper="SDD-YOLO §4.6 式 (5)（MuSGD，源自 Moonshot AI 的 Muon, arXiv:2502.16982；"
          "YOLO26 将其用于实时检测训练）",
    url="https://arxiv.org/abs/2603.25218",
    year=2026,
    license="本文件为独立实现；框架 ultralytics 8.4+ 亦自带同名实现（AGPL-3.0），优先用框架版",
    cost="每步多 5 次 Newton–Schulz 矩阵迭代（只作用于 ndim>=2 的参数）；"
         "显存多一份 momentum buffer；1D 参数走普通 SGD 分量",
    notes="backbone 高维权重用正交化更新、1D 参数（bias/BatchNorm）用 SGD 动量；"
          "对小目标稀疏监督下的梯度震荡有抑制作用",
    aliases=("MuSGDFallback", "musgd"),
)
class MuSGD(torch.optim.Optimizer):
    """混合优化器：``use_muon`` 组 = Muon 更新 + SGD 更新，其余组 = 纯 SGD。

    Args:
        params: 参数或参数组（组内可含 ``use_muon`` 键）。
        lr, momentum, weight_decay, nesterov: 与框架 ``build_optimizer`` 的语义一致。
        muon, sgd: 两个分量的步长系数（YOLO26 取 0.2 / 1.0）。
    """

    def __init__(self, params: Iterable, lr: float = 1e-3, momentum: float = 0.9,
                 weight_decay: float = 0.0, nesterov: bool = True,
                 use_muon: bool = False, muon: float = 0.2, sgd: float = 1.0):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        nesterov=nesterov, use_muon=use_muon)
        super().__init__(params, defaults)
        self.muon = float(muon)
        self.sgd = float(sgd)

    @torch.no_grad()
    def step(self, closure=None):  # noqa: D102 - 与 torch.optim 接口一致
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta = group["momentum"]
            wd = group["weight_decay"]
            nesterov = group["nesterov"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if group["use_muon"]:
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                        state["momentum_buffer_sgd"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=beta,
                                         nesterov=nesterov)
                    p.add_(update.reshape(p.shape).to(p.dtype), alpha=-(lr * self.muon))
                    grad = p.grad
                    if wd != 0:
                        grad = grad.add(p, alpha=wd)
                    buf = state["momentum_buffer_sgd"]
                    buf.mul_(beta).add_(grad)
                    sgd_update = grad.add(buf, alpha=beta) if nesterov else buf
                    p.add_(sgd_update, alpha=-(lr * self.sgd))
                else:
                    grad = p.grad
                    if wd != 0:
                        grad = grad.add(p, alpha=wd)
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    buf = state["momentum_buffer"]
                    buf.mul_(beta).add_(grad)
                    sgd_update = grad.add(buf, alpha=beta) if nesterov else buf
                    p.add_(sgd_update, alpha=-lr)
        return loss


def native_musgd() -> type | None:
    """框架自带的 MuSGD（ultralytics ≥ 8.4）；没有则返回 ``None``。"""
    from tod.compat import ensure_runtime_env, installed

    if not installed():
        return None
    ensure_runtime_env()
    try:
        from ultralytics.optim import MuSGD as _Native
    except ImportError:  # pragma: no cover - 旧版框架
        return None
    return _Native


def resolve() -> tuple[type, str]:
    """返回 (优化器类, 来源描述)：优先框架原生，其次本库实现。"""
    native = native_musgd()
    if native is not None:
        return native, "ultralytics.optim.MuSGD（框架原生）"
    return MuSGD, "tod.optim.musgd.MuSGD（本库实现，NS 正交化 + SGD 分量）"


def param_groups(model: nn.Module, *, lr: float, momentum: float = 0.9,
                 weight_decay: float = 0.0) -> list[dict[str, Any]]:
    """按 MuSGD 的语义把参数分组：muon（ndim≥2）/ bias（不衰减）/ norm（不衰减）/ weight。

    分组规则与框架 ``BaseTrainer.build_optimizer`` 对齐，便于两条路径互换。
    """
    norm_types = tuple(v for k, v in vars(nn).items() if "Norm" in k)
    muon, bias, norm, weight = [], [], [], []
    for name, module in model.named_modules():
        for pname, param in module.named_parameters(recurse=False):
            full = f"{name}.{pname}" if name else pname
            if param.ndim >= 2:                       # 高维权重矩阵 → Muon 分量
                muon.append(param)
            elif "bias" in full:
                bias.append(param)
            elif isinstance(module, norm_types):
                norm.append(param)
            else:
                weight.append(param)

    def group(params: list, *, decay: float, use_muon: bool, tag: str) -> dict[str, Any]:
        return {"params": params, "lr": lr, "momentum": momentum, "nesterov": True,
                "weight_decay": decay, "use_muon": use_muon, "param_group": tag}

    return [
        group(muon, decay=weight_decay, use_muon=True, tag="muon"),
        group(norm, decay=0.0, use_muon=False, tag="bn"),
        group(bias, decay=0.0, use_muon=False, tag="bias"),
        group(weight, decay=weight_decay, use_muon=False, tag="weight"),
    ]


def build(model: nn.Module, *, lr: float, momentum: float = 0.9,
          weight_decay: float = 0.0, iterations: float = 0.0,
          prefer_native: bool = True) -> tuple[torch.optim.Optimizer, str]:
    """构造 MuSGD 优化器。

    与框架训练器的取舍一致：有原生实现就交给框架的优化器类（含 YOLO26 辅助头
    3× lr 的细节由训练器负责），否则用本库实现的参数分组。

    Returns:
        ``(optimizer, 来源描述)``。
    """
    groups = param_groups(model, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if prefer_native:
        native = native_musgd()
        if native is not None:
            return native(params=groups, muon=0.2, sgd=1.0), \
                "ultralytics.optim.MuSGD（框架原生，参数分组由本库给出）"
    return MuSGD(params=groups, muon=0.2, sgd=1.0), \
        "tod.optim.musgd.MuSGD（本库实现）"


__all__ = ["MuSGD", "build", "muon_update", "native_musgd", "param_groups", "resolve",
           "zeropower_newton_schulz"]
