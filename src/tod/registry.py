"""注册表：所有魔改进入本库的唯一入口。

一个魔改 = 一个 ``@register`` 装饰的类/函数 + 一份元数据。
元数据不是可选项：没有来源、许可证、扩展点归属的模块不允许入库（见 PLAN.md §2）。

用法::

    from tod.registry import register

    @register(ep="EP3",
              paper="DySample: Learning to Upsample by Learning to Sample (ICCV 2023)",
              url="https://arxiv.org/abs/2308.15085",
              year=2023, license="Apache-2.0",
              cost="参数量几乎不变；P2 特征图上有轻微显存增量",
              notes="替代最近邻上采样；小目标对 P2 上采样质量敏感")
    class DySample(nn.Module):
        ...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

# 合法扩展点（PLAN.md §4.3）。新增 EP 需先改文档，再改这里。
EXTENSION_POINTS: dict[str, str] = {
    "EP0": "数据 / 输入",
    "EP1": "Backbone",
    "EP2": "Neck",
    "EP3": "上采样",
    "EP4": "注意力",
    "EP5": "Head",
    "EP6": "标签分配",
    "EP7": "损失",
    "EP8": "推理 / 后处理",
    "EP9": "训练策略",
}

# 变体状态机（PLAN.md §10.2）
STATUSES = ("planned", "reproducing", "reproduced", "promoted", "dropped", "failed")


class RegistryError(RuntimeError):
    """注册表使用错误（重复注册、非法 EP 等）。"""


@dataclass(frozen=True)
class ModuleSpec:
    """一个可复用魔改模块的完整身份信息。"""

    name: str                       # YAML / 配置中引用的名字，如 "DySample"
    obj: Any                        # 实际类或函数
    ep: str                         # 扩展点，EP0..EP9
    paper: str = ""                 # 论文标题
    url: str = ""                   # 论文 / 官方仓库链接
    year: int = 0
    license: str = ""
    cost: str = ""                  # 参数量 / FLOPs / 显存 / 延迟的定性或定量提示
    notes: str = ""
    aliases: tuple[str, ...] = field(default_factory=tuple)
    source_file: str = ""           # 自动填充，便于回溯

    @property
    def ep_name(self) -> str:
        return EXTENSION_POINTS.get(self.ep, self.ep)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ep": self.ep,
            "ep_name": self.ep_name,
            "paper": self.paper,
            "url": self.url,
            "year": self.year,
            "license": self.license,
            "cost": self.cost,
            "notes": self.notes,
            "aliases": list(self.aliases),
            "source_file": self.source_file,
        }


_REGISTRY: dict[str, ModuleSpec] = {}
_CANONICAL: list[str] = []          # 保持注册顺序，保证文档/表格输出稳定


def register(
    name: str | None = None,
    *,
    ep: str,
    paper: str = "",
    url: str = "",
    year: int = 0,
    license: str = "",
    cost: str = "",
    notes: str = "",
    aliases: Iterable[str] = (),
    override: bool = False,
) -> Callable[[Any], Any]:
    """把一个模块登记进注册表。

    Args:
        name: 注册名；缺省用类/函数名。
        ep: 扩展点，必须是 ``EXTENSION_POINTS`` 的键。
        override: 同名模块默认报错（防止无意覆盖）；显式覆盖时置 True。
    """
    if ep not in EXTENSION_POINTS:
        raise RegistryError(
            f"非法扩展点 {ep!r}；可选：{sorted(EXTENSION_POINTS)}。"
            "若确有新扩展点，请先更新 PLAN.md §4.3 与 EXTENSION_POINTS。"
        )

    def deco(obj: Any) -> Any:
        reg_name = name or getattr(obj, "__name__", None)
        if not reg_name:
            raise RegistryError(f"无法推断注册名：{obj!r}")
        if reg_name in _REGISTRY and not override:
            raise RegistryError(
                f"注册名 {reg_name!r} 已存在（来源：{_REGISTRY[reg_name].source_file}）。"
                "换一个名字，或显式 override=True。"
            )
        spec = ModuleSpec(
            name=reg_name,
            obj=obj,
            ep=ep,
            paper=paper,
            url=url,
            year=year,
            license=license,
            cost=cost,
            notes=notes,
            aliases=tuple(aliases),
            source_file=getattr(obj, "__module__", "") or "",
        )
        _REGISTRY[reg_name] = spec
        if reg_name not in _CANONICAL:
            _CANONICAL.append(reg_name)
        for alias in spec.aliases:
            if alias in _REGISTRY and not override:
                raise RegistryError(f"别名 {alias!r} 与已有模块冲突。")
            _REGISTRY[alias] = spec
        return obj

    return deco


def get(name: str) -> ModuleSpec:
    """按注册名或别名取回 spec。"""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise RegistryError(f"未注册的模块 {name!r}。已注册：{sorted(set(_CANONICAL))}") from None


def has(name: str) -> bool:
    return name in _REGISTRY


def list_specs(ep: str | None = None) -> list[ModuleSpec]:
    """按注册顺序返回规范条目（已去重、不含别名）。"""
    specs = [_REGISTRY[n] for n in _CANONICAL]
    if ep is not None:
        specs = [s for s in specs if s.ep == ep]
    return specs


def install() -> None:
    """把注册的模块注入框架的模型解析命名空间，使其可在 YAML 中按名引用。

    这是本库**唯一**需要触碰框架内部的动作，全部实现收敛在 ``tod.compat``。
    """
    from tod import compat

    ns = compat.model_globals()
    for spec in list_specs():
        ns[spec.name] = spec.obj


def catalog() -> str:
    """生成 ``docs/VARIANTS.md`` 的模块总表（按 EP 分组）。

    文档一律由代码生成，禁止手工维护（PLAN.md §9.10）。
    """
    lines: list[str] = ["# 模块总表（自动生成，请勿手工编辑）", "",
                        "运行 `python tools/catalog.py --write` 重新生成。", ""]
    for ep, ep_name in EXTENSION_POINTS.items():
        specs = list_specs(ep)
        if not specs:
            continue
        lines.append(f"## {ep} {ep_name}（{len(specs)} 项）")
        lines.append("")
        lines.append("| 模块 | 年份 | 论文 / 来源 | 许可证 | 成本提示 | 备注 |")
        lines.append("|---|---|---|---|---|---|")
        for s in sorted(specs, key=lambda x: (x.year, x.name)):
            paper = f"[{s.paper}]({s.url})" if s.url else (s.paper or "-")
            lines.append(
                f"| `{s.name}` | {s.year or '-'} | {paper or '-'} | "
                f"{s.license or '-'} | {s.cost or '-'} | {s.notes or '-'} |"
            )
        lines.append("")
    if len(lines) <= 4:
        lines.append("_（尚未注册任何模块）_")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - 文档生成入口
    print(catalog())
