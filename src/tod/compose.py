"""变体构造 DSL：骨架 + 扩展点覆盖 → 变体配置 / 模型 YAML / 变体卡片。

设计意图（PLAN.md §4.2）：变体不由人手写 YAML 维护，而是由代码生成。
消融只需 ``v.without("DySample")``，配置永远与代码同步，不会漂移。

用法::

    from tod.compose import Variant

    v = (Variant("YOLOv8n_P2_DySample_NWD", base="yolov8n")
         .data("visdrone2019-det", imgsz=1280)
         .upsample("DySample")
         .head("Detect", levels=[2, 3, 4, 5], box_loss="NWD")
         .train(epochs=150, optimizer="AdamW", batch=4, amp=True))

    v.dump("configs/variants/visdrone/YOLOv8n_P2_DySample_NWD.yaml")
    v.model_yaml("variants/YOLOv8n_P2_DySample_NWD/model.yaml")   # 需已安装 ultralytics
    v.card("variants/YOLOv8n_P2_DySample_NWD/card.md")
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from tod.registry import EXTENSION_POINTS, get, has

#: 本库自有的变体配置 schema 版本，便于将来迁移。
SPEC_VERSION = 1


def _merge(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _merge(dst[k], v)
        else:
            dst[k] = v
    return dst


@dataclass
class Variant:
    """一个可复现的魔改变体 = 骨架 + 各 EP 的覆盖 + 数据 + 训练配置。"""

    name: str
    base: str = "yolov8n"
    dataset: str = ""
    status: str = "planned"
    eps: dict[str, dict[str, Any]] = field(default_factory=dict)
    data_cfg: dict[str, Any] = field(default_factory=dict)
    train_cfg: dict[str, Any] = field(default_factory=dict)
    model_cfg: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    notes: str = ""

    # ---------------------------------------------------------------- 构造

    def patch(self, ep: str, **kwargs: Any) -> "Variant":
        """在任意扩展点上打覆盖。EP 必须在 ``EXTENSION_POINTS`` 内。"""
        if ep not in EXTENSION_POINTS:
            raise KeyError(f"未知扩展点 {ep!r}；可选 {sorted(EXTENSION_POINTS)}")
        _merge(self.eps.setdefault(ep, {}), kwargs)
        return self

    def without(self, target: str) -> "Variant":
        """剥离某个模块名或配置键（消融 / leave-one-out 用）。

        同时匹配"键名"与"值"：``v.without("DySample")`` 与 ``v.without("upsample")``
        都能让 EP3 的 DySample 覆盖失效。
        """
        for cfg in self.eps.values():
            for key in [k for k in cfg if k == target]:
                cfg.pop(key)
            for key, value in list(cfg.items()):
                if value == target:
                    cfg.pop(key)
                elif isinstance(value, (list, tuple)) and target in value:
                    cfg[key] = [x for x in value if x != target]
        return self

    def data(self, dataset: str, **kwargs: Any) -> "Variant":
        self.dataset = dataset
        _merge(self.data_cfg, kwargs)
        return self

    def train(self, **kwargs: Any) -> "Variant":
        _merge(self.train_cfg, kwargs)
        return self

    def model(self, **kwargs: Any) -> "Variant":
        """模型图级别的参数（如 imgsz 无关的 nc、scale、P2 注入参数）。"""
        _merge(self.model_cfg, kwargs)
        return self

    # 语义化快捷方式（等价于 patch）
    def backbone(self, **kw: Any) -> "Variant":
        return self.patch("EP1", **kw)

    def neck(self, **kw: Any) -> "Variant":
        return self.patch("EP2", **kw)

    def upsample(self, name: str | None = None, **kw: Any) -> "Variant":
        return self.patch("EP3", upsample=name, **kw)

    def attention(self, name: str | None = None, **kw: Any) -> "Variant":
        return self.patch("EP4", attention=name, **kw)

    def head(self, name: str | None = None, **kw: Any) -> "Variant":
        return self.patch("EP5", head=name, **kw)

    def assigner(self, name: str | None = None, **kw: Any) -> "Variant":
        return self.patch("EP6", assigner=name, **kw)

    def loss(self, **kw: Any) -> "Variant":
        return self.patch("EP7", **kw)

    def inference(self, **kw: Any) -> "Variant":
        return self.patch("EP8", **kw)

    def strategy(self, **kw: Any) -> "Variant":
        return self.patch("EP9", **kw)

    # ---------------------------------------------------------------- 导出

    def used_modules(self) -> list[str]:
        """收集本变体引用到的、已在注册表中登记的模块名。"""
        found: list[str] = []
        for cfg in self.eps.values():
            for value in cfg.values():
                for item in (value if isinstance(value, (list, tuple)) else [value]):
                    if isinstance(item, str) and has(item) and item not in found:
                        found.append(item)
        return found

    def spec(self) -> dict[str, Any]:
        return {
            "spec_version": SPEC_VERSION,
            "id": self.name,
            "base": self.base,
            "status": self.status,
            "dataset": self.dataset,
            "tags": list(self.tags),
            "notes": self.notes,
            "eps": {k: dict(v) for k, v in sorted(self.eps.items()) if v},
            "data": dict(self.data_cfg),
            "model": dict(self.model_cfg),
            "train": dict(self.train_cfg),
            "modules": self.used_modules(),
        }

    def dump(self, path: str | Path) -> Path:
        from tod.compat import dump_yaml

        return dump_yaml(self.spec(), path)

    # ------------------------------------------------------------ 模型 YAML

    def _load_base_cfg(self) -> tuple[dict, str | None]:
        """读取 base 模型图；支持 ``yolov8`` / ``yolov8n`` 两种写法。

        官方只提供 ``yolov8.yaml`` + ``scales``，规模由文件名尾字母决定，
        因此这里把 ``yolov8n`` 拆成 ``yolov8`` + ``scale='n'``。
        """
        from tod import compat

        name, scale = self.base, self.model_cfg.get("scale")
        try:
            path = compat.base_model_yaml(name)
        except compat.CompatError:
            if len(name) > 1 and name[-1] in "nsmlx":
                scale, name = name[-1], name[:-1]
                path = compat.base_model_yaml(name)
            else:
                raise
        cfg = deepcopy(compat.load_yaml(path))
        cfg.pop("scale", None)      # 避免与 base 名冲突
        return cfg, scale

    def model_yaml(self, path: str | Path | None = None, *, write: bool = True) -> dict:
        """由 ``base`` 的内置模型 YAML 生成变体模型图。

        当前 M0 支持两类结构性改造：
          * 类型替换（``type_map``，如 ``Conv -> SPDConv``、``nn.Upsample -> DySample``）
          * P2 检测头注入（``add_p2=True``，见 ``inject_p2_head``）

        更复杂的颈部重拓扑（BiFPN / AFPN / HS-FPN）属于 M1 工作。
        """
        cfg, scale = self._load_base_cfg()
        type_map: dict[str, str] = dict(self.model_cfg.get("type_map", {}))

        # EP 覆盖 → 节点类型替换。只映射语义等价的节点类型：
        # 注意力类模块不是 Conv 的等价替换，必须写在 type_map 或 neck/head 配置里。
        ep_targets = (("EP3", "upsample", "nn.Upsample"),
                      ("EP1", "conv", "Conv"),
                      ("EP1", "block", "C2f"),
                      ("EP5", "head", "Detect"))
        for ep, key, target in ep_targets:
            new = self.eps.get(ep, {}).get(key)
            if isinstance(new, str) and new and new != target:
                type_map[target] = new

        if self.model_cfg.get("nc") is not None:
            cfg["nc"] = self.model_cfg["nc"]
        if scale:
            cfg["scale"] = scale

        if type_map:
            apply_type_map(cfg, type_map)

        if self.model_cfg.get("add_p2"):
            inject_p2_head(
                cfg,
                p2_idx=int(self.model_cfg.get("p2_idx", 2)),
                p2_channels=int(self.model_cfg.get("p2_channels", 64)),
                fuse_block=str(self.model_cfg.get("p2_fuse_block", "C2f")),
                upsample=str(self.eps.get("EP3", {}).get("upsample") or "nn.Upsample"),
            )

        if write:
            if path is None:
                raise ValueError("write=True 时必须给出 path")
            from tod.compat import dump_yaml

            dump_yaml(cfg, path)
        return cfg

    # ---------------------------------------------------------------- 卡片

    def card(self, path: str | Path) -> Path:
        """生成变体卡片（PLAN.md §4.4）。"""
        lines = [
            f"# {self.name}",
            "",
            f"- **id**: `{self.name}`",
            f"- **base**: `{self.base}`",
            f"- **dataset**: `{self.dataset or '未指定'}`",
            f"- **status**: `{self.status}`",
            f"- **tags**: {', '.join(self.tags) if self.tags else '-'}",
            "",
            "## EP 改动",
            "",
            "| EP | 名称 | 覆盖 |",
            "|---|---|---|",
        ]
        if self.eps:
            for ep in sorted(self.eps):
                if not self.eps[ep]:
                    continue
                kv = "; ".join(f"`{k}={v}`" for k, v in self.eps[ep].items())
                lines.append(f"| {ep} | {EXTENSION_POINTS.get(ep, '?')} | {kv} |")
        else:
            lines.append("| - | - | 未做任何魔改（纯基线） |")

        lines += ["", "## 引用模块来源", ""]
        used = self.used_modules()
        if used:
            lines += ["| 模块 | 年份 | 论文 / 来源 | 许可证 | 成本提示 |",
                      "|---|---|---|---|---|"]
            for name in used:
                s = get(name)
                paper = f"[{s.paper}]({s.url})" if s.url else (s.paper or "-")
                lines.append(f"| `{s.name}` | {s.year or '-'} | {paper} | "
                             f"{s.license or '-'} | {s.cost or '-'} |")
        else:
            lines.append("_（无，或引用的模块尚未注册）_")

        lines += [
            "",
            "## 相对基线",
            "",
            "| 指标 | 基线 | 本变体 | Δ |",
            "|---|---|---|---|",
            "| AP50:95 | | | |",
            "| **AP_small** | | | |",
            "| AP_tiny (<16px) | | | |",
            "| Params / FLOPs | | | |",
            "| 延迟 (ms, batch=1) | | | |",
            "| 峰值显存 (GB) | | | |",
            "",
            "## 已知坑 / 冲突 / 结论",
            "",
            self.notes or "_待填_",
            "",
        ]
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines), encoding="utf-8")
        return path


# -------------------------------------------------------------------- 图改造


def _node_lists(cfg: dict) -> tuple[list, list, int]:
    """返回 (节点列表, 该列表在全局的起始索引, 是否为 head 段)。

    兼容两种 YAML 形式：``backbone:`` + ``head:``（官方 v8/v11 风格）
    与单一 ``model:`` 列表。
    """
    if "backbone" in cfg and "head" in cfg:
        return cfg["head"], len(cfg["backbone"]), True
    if "model" in cfg:
        return cfg["model"], 0, False
    raise ValueError("模型 YAML 既没有 backbone/head，也没有 model 字段。")


def _find_detect(nodes: Sequence) -> int:
    for i in range(len(nodes) - 1, -1, -1):
        if str(nodes[i][2]).endswith("Detect"):
            return i
    raise ValueError("未在模型图中找到 Detect 头节点。")


def apply_type_map(cfg: dict, type_map: dict[str, str]) -> dict:
    """按 ``{旧类型: 新类型}`` 替换节点类型；就地修改并返回 cfg。"""
    if "backbone" in cfg and "head" in cfg:
        lists = [cfg["backbone"], cfg["head"]]
    else:
        lists = [cfg["model"]]
    hit = {k: 0 for k in type_map}
    for nodes in lists:
        for node in nodes:
            t = str(node[2])
            if t in type_map:
                node[2] = type_map[t]
                hit[t] += 1
    cfg["_type_map_applied"] = {k: v for k, v in hit.items() if v}
    return cfg


def inject_p2_head(
    cfg: dict,
    *,
    p2_idx: int = 2,
    p2_channels: int = 64,
    fuse_block: str = "C2f",
    upsample: str = "nn.Upsample",
    repeats: int = 3,
) -> dict:
    """把 P2（stride=4）分支接进检测头 —— PLAN.md 中最高 ROI 的改动。

    生成的结构（以 YOLOv8 为例，P3 头节点为 15、P2 源为 2）::

        [-1, 1, nn.Upsample, [None, 2, "nearest"]]   # 由 P3 上采样
        [[-1, 2], 1, Concat, [1]]                    # 与 backbone P2 拼接
        [-1, 3, C2f, [64]]                           # 融合
        [[24, 15, 18, 21], 1, Detect, [nc]]          # Detect(P2, P3, P4, P5)

    这是"直连式 P2 头"（M0 版本）：只做一次自顶向下融合，不做完整双向重拓扑。
    完整 P2 双向融合 / BiFPN 属于 M1。

    Args:
        p2_idx: backbone 中 P2（stride=4）特征源的全局节点索引。
            YOLOv8 系列为 2；换主干后请对照模型 YAML 注释确认。
        p2_channels: 融合块输出通道（会被 width 缩放）。
        upsample: 上采样模块名，可填注册表中的名字（如 ``DySample``）。
    """
    nodes, base, _ = _node_lists(cfg)
    det_i = _find_detect(nodes)
    det_node = nodes[det_i]
    src = list(det_node[0])

    if len(src) == 4:
        cfg["_p2_status"] = "already_present"
        return cfg
    if len(src) != 3:
        raise ValueError(f"预期 Detect 有 3 个输入（P3,P4,P5），实际 {src}。")

    p3_idx, p4_idx, p5_idx = src
    det_global = base + det_i

    new_nodes = [
        [p3_idx, 1, upsample, [None, 2, "nearest"]],
        [[-1, p2_idx], 1, "Concat", [1]],
        [-1, repeats, fuse_block, [p2_channels]],
    ]
    for offset, node in enumerate(new_nodes):
        nodes.insert(det_i + offset, node)

    fuse_idx = det_global + 2
    nodes[det_i + 3][0] = [fuse_idx, p3_idx, p4_idx, p5_idx]
    cfg["_p2_status"] = "injected"
    return cfg
