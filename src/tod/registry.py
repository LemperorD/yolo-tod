"""注册表：所有魔改进入本库的唯一入口。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

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

STATUSES = ("planned", "reproducing", "reproduced", "promoted", "dropped", "failed")


class RegistryError(RuntimeError):
    """注册表使用错误（重复注册、非法 EP 等）。"""


@dataclass(frozen=True)
class ModuleSpec:
    """一个可复用魔改模块的完整身份信息。"""

    name: str
    obj: Any
    ep: str
    paper: str = ""
    url: str = ""
    year: int = 0
    license: str = ""
    cost: str = ""
    notes: str = ""
    aliases: tuple[str, ...] = field(default_factory=tuple)
    source_file: str = ""

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
# 保留规范名的注册顺序，使清单与生成文档稳定。
_CANONICAL: list[str] = []


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
    """把一个模块登记进注册表。"""
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
    """把注册的模块注入框架的模型解析命名空间，使其可在 YAML 中按名引用。"""
    from tod import compat

    ns = compat.model_globals()
    for spec in list_specs():
        ns[spec.name] = spec.obj
        for alias in spec.aliases:
            ns[alias] = spec.obj


def catalog() -> str:
    """生成 ``docs/VARIANTS.md`` 的模块总表（按 EP 分组）。"""
    lines: list[str] = ["# 模块总表（自动生成，请勿手工编辑）", "",
                        "运行 `python3 tools/catalog.py --write` 重新生成。", ""]
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


if __name__ == "__main__":  # pragma: no cover
    print(catalog())
