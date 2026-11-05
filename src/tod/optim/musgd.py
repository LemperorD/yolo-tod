"""MuSGD —— Muon（Newton–Schulz 正交化）+ SGD 的混合优化器（SDD-YOLO §4.6，式 (5)）。"""

from __future__ import annotations

from typing import Any, Iterable

import torch
import torch.nn as nn

from tod.registry import register

NS_COEFFS = (3.4445, -4.7750, 2.0315)
NS_STEPS = 5


def zeropower_newton_schulz(g: torch.Tensor, steps: int = NS_STEPS,
                            eps: float = 1e-7) -> torch.Tensor:
    """用 Newton–Schulz 迭代求矩阵的近似正交化（近似 ``U V^T``）。"""
    if g.ndim != 2:
        raise ValueError(f"Newton–Schulz 只接受 2D 张量，实际 {tuple(g.shape)}。")
    x = g.to(torch.float32)
    x = x / (x.norm() + eps)
    # 把迭代放在较小的矩阵维度上，最终恢复原方向。
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
    momentum_buffer.lerp_(grad, 1 - beta)
    update = grad.lerp(momentum_buffer, beta) if nesterov else momentum_buffer
    if update.ndim == 4:
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
    """混合优化器：``use_muon`` 组 = Muon 更新 + SGD 更新，其余组 = 纯 SGD。"""

    def __init__(self, params: Iterable, lr: float = 1e-3, momentum: float = 0.9,
                 weight_decay: float = 0.0, nesterov: bool = True,
                 use_muon: bool = False, muon: float = 0.2, sgd: float = 1.0):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        nesterov=nesterov, use_muon=use_muon)
        super().__init__(params, defaults)
        self.muon = float(muon)
        self.sgd = float(sgd)

    @torch.no_grad()
    def step(self, closure=None):  # noqa: D102
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
    except ImportError:  # pragma: no cover
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
    """按 MuSGD 的语义把参数分组：muon（ndim≥2）/ bias（不衰减）/ norm（不衰减）/ weight。"""
    norm_types = tuple(v for k, v in vars(nn).items() if "Norm" in k)
    muon, bias, norm, weight = [], [], [], []
    for name, module in model.named_modules():
        for pname, param in module.named_parameters(recurse=False):
            full = f"{name}.{pname}" if name else pname
            if param.ndim >= 2:
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
    """构造 MuSGD 优化器。"""
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
