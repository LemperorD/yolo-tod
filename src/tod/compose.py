"""变体构造 DSL：骨架 + 扩展点覆盖 → 变体配置 / 模型 YAML / 变体卡片。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from tod.registry import EXTENSION_POINTS, get, has

SPEC_VERSION = 2
TASKS = ("detect", "pose", "segment", "obb")
TASK_MODEL_SUFFIX = {"detect": "", "pose": "-pose", "segment": "-seg", "obb": "-obb"}


def _merge(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _merge(dst[k], v)
        else:
            dst[k] = v
    return dst


def _as_list(value: Any) -> list:
    """把 tuple / list / 逗号字符串统一成 list（``(17,3)`` 与 ``"17,3"`` 都能吃）。"""
    if isinstance(value, str):
        out: list = []
        for chunk in value.replace(";", ",").split(","):
            text = chunk.strip()
            if not text:
                continue
            try:
                out.append(int(text))
            except ValueError:
                out.append(text)
        return out
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


@dataclass
class Variant:
    """一个可复现的魔改变体 = 骨架 + 各 EP 的覆盖 + 数据 + 训练配置。"""

    name: str
    base: str = "yolov8n"
    dataset: str = ""
    status: str = "planned"
    task: str = "detect"
    eps: dict[str, dict[str, Any]] = field(default_factory=dict)
    data_cfg: dict[str, Any] = field(default_factory=dict)
    train_cfg: dict[str, Any] = field(default_factory=dict)
    model_cfg: dict[str, Any] = field(default_factory=dict)
    dataset_cfg: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    notes: str = ""

    def __post_init__(self) -> None:
        if self.task not in TASKS:
            raise ValueError(f"未知任务 {self.task!r}；可选 {sorted(TASKS)}")


    def patch(self, ep: str, **kwargs: Any) -> "Variant":
        """在任意扩展点上打覆盖。EP 必须在 ``EXTENSION_POINTS`` 内。"""
        if ep not in EXTENSION_POINTS:
            raise KeyError(f"未知扩展点 {ep!r}；可选 {sorted(EXTENSION_POINTS)}")
        _merge(self.eps.setdefault(ep, {}), kwargs)
        return self

    def without(self, target: str) -> "Variant":
        """剥离某个模块名或配置键（消融 / leave-one-out 用）。"""
        for cfg in (*self.eps.values(), self.model_cfg, self.train_cfg):
            for key in [k for k in cfg if k == target]:
                cfg.pop(key)
            for key, value in list(cfg.items()):
                if value == target:
                    cfg.pop(key)
                elif isinstance(value, (list, tuple)) and target in value:
                    cfg[key] = [x for x in value if x != target]
                elif isinstance(value, dict) and target in value:
                    cfg[key] = {k: v for k, v in value.items() if k != target}
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

    def as_task(self, task: str) -> "Variant":
        """设定任务类型（``detect`` / ``pose`` / …）。"""
        if task not in TASKS:
            raise ValueError(f"未知任务 {task!r}；可选 {sorted(TASKS)}")
        self.task = task
        return self

    def pose(self, **kwargs: Any) -> "Variant":
        """姿态任务的快捷方式：``v.pose(kpt_shape=(17, 3))``。"""
        self.task = "pose"
        if kwargs:
            _merge(self.model_cfg, kwargs)
        return self

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


    def used_modules(self) -> list[str]:
        """收集本变体引用到的、已在注册表中登记的模块名。"""
        found: list[str] = []

        def scan(cfg: dict) -> None:
            for value in cfg.values():
                for item in (value if isinstance(value, (list, tuple)) else [value]):
                    if isinstance(item, str) and has(item) and item not in found:
                        found.append(item)

        for cfg in self.eps.values():
            scan(cfg)
        scan(self.model_cfg)
        return found

    def spec(self) -> dict[str, Any]:
        return {
            "spec_version": SPEC_VERSION,
            "id": self.name,
            "base": self.base,
            "task": self.task,
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

    @classmethod
    def from_spec(cls, data: dict[str, Any]) -> "Variant":
        """从 ``dump()`` 写出的 spec 还原变体对象。"""
        v = cls(
            name=str(data.get("id") or data.get("name") or "unnamed"),
            base=str(data.get("base", "yolov8n")),
            dataset=str(data.get("dataset", "")),
            status=str(data.get("status", "planned")),
            task=str(data.get("task", "detect")),
            tags=list(data.get("tags") or []),
            notes=str(data.get("notes", "")),
        )
        v.eps = {k: dict(x) for k, x in (data.get("eps") or {}).items()}
        v.data_cfg = dict(data.get("data") or {})
        v.train_cfg = dict(data.get("train") or {})
        v.model_cfg = dict(data.get("model") or {})
        return v


    def _load_base_cfg(self) -> tuple[dict, str | None]:
        """读取 base 模型图；支持 ``yolov8`` / ``yolov8n`` / ``yolo26n`` 三种写法。"""
        from tod import compat

        name, scale = self.base, self.model_cfg.get("scale")
        suffix = TASK_MODEL_SUFFIX.get(self.task, "") if self.task != "detect" else ""
        candidates = []
        if suffix:
            candidates.append(name if name.endswith(suffix) else f"{name}{suffix}")
        candidates.append(name)

        last_error: Exception | None = None
        for candidate in candidates:
            try:
                path = compat.base_model_yaml(candidate)
            except compat.CompatError as exc:
                last_error = exc
                if len(candidate) > 1 and candidate[-1] in "nsmlx":
                    scale, base_name = candidate[-1], candidate[:-1]
                    try:
                        path = compat.base_model_yaml(f"{base_name}{suffix}")
                    except compat.CompatError as exc2:
                        last_error = exc2
                        continue
                else:
                    continue
            cfg = deepcopy(compat.load_yaml(path))
            cfg.pop("scale", None)
            return cfg, scale
        raise last_error if last_error else compat.CompatError(f"找不到 {name!r} 的模型图")

    def model_yaml(self, path: str | Path | None = None, *, write: bool = True) -> dict:
        """由 ``base`` 的内置模型 YAML 生成变体模型图。"""
        cfg, scale = self._load_base_cfg()

        if self.model_cfg.get("nc") is not None:
            cfg["nc"] = self.model_cfg["nc"]
        if self.model_cfg.get("kpt_shape") is not None:
            cfg["kpt_shape"] = _as_list(self.model_cfg["kpt_shape"])
        if scale:
            cfg["scale"] = scale
            scales = cfg.get("scales")
            if isinstance(scales, dict) and scale in scales:
                cfg["scales"] = {
                    scale: scales[scale],
                    **{k: v for k, v in scales.items() if k != scale},
                }

        ds_module = self.model_cfg.get("downsample") or self.eps.get("EP1", {}).get("downsample")
        if ds_module:
            replace_downsample(
                cfg,
                module=str(ds_module),
                indices=self.model_cfg.get("downsample_indices")
                or self.eps.get("EP1", {}).get("downsample_indices"),
            )

        if self.model_cfg.get("add_p2"):
            p2_pre = self.model_cfg.get("p2_pre")
            if isinstance(p2_pre, dict):
                p2_pre = [p2_pre["type"], p2_pre.get("args", [])]
            inject_p2_head(
                cfg,
                p2_idx=int(self.model_cfg.get("p2_idx", 2)),
                p2_channels=int(self.model_cfg.get("p2_channels", 64)),
                fuse_block=str(self.model_cfg.get("p2_fuse_block", "C2f")),
                upsample=str(self.eps.get("EP3", {}).get("upsample") or "nn.Upsample"),
                p2_pre=p2_pre,
            )

        type_map: dict[str, str] = dict(self.model_cfg.get("type_map", {}))
        ep_targets = (("EP3", "upsample", "nn.Upsample"),
                      ("EP1", "conv", "Conv"),
                      ("EP1", "block", "C2f"))
        for ep, key, target in ep_targets:
            new = self.eps.get(ep, {}).get(key)
            if isinstance(new, str) and new and new != target:
                type_map[target] = new

        if type_map:
            apply_type_map(cfg, type_map)

        head_name = self.eps.get("EP5", {}).get("head")
        if isinstance(head_name, str) and head_name not in ("", "Detect"):
            cfg["_head_surgery"] = head_name

        if write:
            if path is None:
                raise ValueError("write=True 时必须给出 path")
            from tod.compat import dump_yaml

            dump_yaml(cfg, path)
        return cfg


    def card(self, path: str | Path) -> Path:
        """生成变体卡片（PLAN.md §4.4）。"""
        lines = [
            f"# {self.name}",
            "",
            f"- **id**: `{self.name}`",
            f"- **base**: `{self.base}`",
            f"- **task**: `{self.task}`"
            + ("" if self.task == "detect" else "（非检测任务：数据加载/准则/评测都与检测不同）"),
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

        if self.model_cfg:
            lines += ["", "## 模型图改造（model 段）", "", "| 键 | 值 |", "|---|---|"]
            for key, value in self.model_cfg.items():
                lines.append(f"| `{key}` | `{value}` |")
            derived = []
            if self.model_cfg.get("add_p2"):
                head_kind = "Pose/Detect" if self.task == "pose" else "Detect"
                derived.append(
                    "**P2 检测头（EP5/EP2）**：`inject_p2_head` 注入 stride=4 分支 —— "
                    f"上采样(P3) ⊕ backbone 节点 {self.model_cfg.get('p2_idx', 2)} → "
                    f"`{self.model_cfg.get('p2_fuse_block', 'C2f')}` 融合 → "
                    f"{head_kind}(P2,P3,P4,P5)")
                if self.task == "pose":
                    derived.append(
                        "**姿态分支跟着走**：注入 P2 后关键点分支（``cv4`` / Pose26 的 "
                        "``cv4_kpts``+``cv4_sigma``）**自动多一层** —— 关键点分辨率随之翻倍，"
                        "这是小目标关键点最直接的收益来源，不需要额外手术。")
            if self.model_cfg.get("kpt_shape"):
                derived.append(f"**关键点形状**：`kpt_shape={_as_list(self.model_cfg['kpt_shape'])}`"
                               "（数据集 YAML 里的同名键优先级更高）")
            if self.model_cfg.get("downsample"):
                derived.append(f"**主干下采样替换（EP1）**：`{self.model_cfg['downsample']}`"
                               f"（索引 {self.model_cfg.get('downsample_indices', '默认 1/3/5/7')}）")
            if self.model_cfg.get("type_map"):
                derived.append(f"**节点类型替换**：`{self.model_cfg['type_map']}`")
            if derived:
                lines += [""] + [f"- {d}" for d in derived]

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

        metric_rows = [
            "| AP50:95 | | | |",
            "| **AP_small** | | | |",
            "| AP_tiny (<16px) | | | |",
        ]
        if self.task == "pose":
            metric_rows += [
                "| **OKS-AP**（本库口径，框 IoU 匹配） | | | |",
                "| OKS-AP50 | | | |",
                "| **OKS-AP_small / OKS-AP_tiny** | | | |",
                "| 关键点平均误差 (px, ≤1px 命中率) | | | |",
            ]
        metric_rows += [
            "| Params / FLOPs | | | |",
            "| 延迟 (ms, batch=1) | | | |",
            "| 峰值显存 (GB) | | | |",
        ]
        lines += [
            "",
            "## 相对基线",
            "",
            "| 指标 | 基线 | 本变体 | Δ |",
            "|---|---|---|---|",
            *metric_rows,
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


def _node_lists(cfg: dict) -> tuple[list, list, int]:
    """返回 (节点列表, 该列表在全局的起始索引, 是否为 head 段)。"""
    if "backbone" in cfg and "head" in cfg:
        return cfg["head"], len(cfg["backbone"]), True
    if "model" in cfg:
        return cfg["model"], 0, False
    raise ValueError("模型 YAML 既没有 backbone/head，也没有 model 字段。")


def _find_detect(nodes: Sequence) -> int:
    """定位检测头节点（Detect / Pose / Pose26 / Segment / OBB 等任意头）。"""
    families = ("detect", "pose", "segment", "obb")
    for i in range(len(nodes) - 1, -1, -1):
        name = str(nodes[i][2]).lower()
        if any(f in name for f in families):
            return i
    raise ValueError(
        "未在模型图中找到检测头节点（类型名应含 detect/pose/segment/obb 之一）。"
    )


def _head_inputs(node: Sequence) -> list:
    """读出头节点的输入索引列表（兼容 ``[in, n, Type, args]`` 与省略 repeats 的写法）。"""
    if len(node) == 4:
        return list(node[0])
    return [node[0]]


def _rebuild_head_node(node: Sequence, inputs: list) -> list:
    """把新的输入列表写回节点**而不丢其它参数**。"""
    updated = list(node)
    updated[0] = list(inputs)
    return updated


def apply_type_map(cfg: dict, type_map: dict[str, str]) -> dict:
    """按 ``{旧类型: 新类型}`` 替换节点类型；就地修改 cfg（节点本身替换为新列表）。"""
    if "backbone" in cfg and "head" in cfg:
        lists = [cfg["backbone"], cfg["head"]]
    else:
        lists = [cfg["model"]]
    hit = {k: 0 for k in type_map}
    for nodes in lists:
        for i, node in enumerate(nodes):
            t = str(node[2])
            if t in type_map:
                new_node = list(node)
                new_node[2] = type_map[t]
                nodes[i] = new_node
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
    p2_pre: Sequence | None = None,
) -> dict:
    """把 P2（stride=4）分支接进检测头 —— PLAN.md 中最高 ROI 的改动。"""
    nodes, base, _ = _node_lists(cfg)
    det_i = _find_detect(nodes)
    src = _head_inputs(nodes[det_i])

    if len(src) == 4:
        cfg["_p2_status"] = "already_present"
        return cfg
    if len(src) != 3:
        raise ValueError(f"预期检测头有 3 个输入（P3,P4,P5），实际 {src}。")

    p3_idx, p4_idx, p5_idx = src
    # 新分支插在原输出头之前，保留 P3–P5 既有特征节点的索引。
    det_global = base + det_i
    offset = 0

    if p2_pre is not None:
        pre_type, pre_args = p2_pre[0], list(p2_pre[1])
        nodes.insert(det_i + offset, [p2_idx, 1, pre_type, pre_args])
        p2_src = det_global + offset
        offset += 1
    else:
        p2_src = p2_idx

    nodes.insert(det_i + offset, [p3_idx, 1, upsample, [None, 2, "nearest"]])
    offset += 1

    nodes.insert(det_i + offset, [[-1, p2_src], 1, "Concat", [1]])
    offset += 1

    fuse_idx = det_global + offset
    nodes.insert(det_i + offset, [-1, repeats, fuse_block, [p2_channels]])
    offset += 1

    nodes[det_i + offset] = _rebuild_head_node(
        nodes[det_i + offset], [fuse_idx, p3_idx, p4_idx, p5_idx]
    )
    cfg["_p2_status"] = "injected"
    cfg["_p2_channels"] = p2_channels
    return cfg


def _module_arity(name: str) -> int | None:
    """模块 ``__init__`` 的**必需**位置参数个数（不含 self）；拿不到返回 None。"""
    import inspect

    obj = None
    try:
        from tod.compat import model_globals

        obj = model_globals().get(name)
    except Exception:  # noqa: BLE001
        obj = None
    if obj is None:
        try:
            from tod.registry import get as _get

            obj = _get(name).obj
        except Exception:  # noqa: BLE001
            return None
    try:
        params = list(inspect.signature(obj.__init__).parameters.values())[1:]
    except (TypeError, ValueError):
        return None
    required = [p for p in params
                if p.default is p.empty
                and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    return len(required)


def replace_downsample(
    cfg: dict,
    *,
    module: str = "ADown",
    indices: Sequence[int] | None = None,
    segments: Sequence[str] = ("backbone",),
    trim_args: bool = True,
) -> dict:
    """把主干中的 stride-2 下采样卷积换成给定模块（如 ADown）。"""
    indices = tuple(indices) if indices is not None else (1, 3, 5, 7)
    replaced: dict[int, str] = {}
    arity = _module_arity(module) if trim_args else None

    for segment in segments:
        if segment not in cfg:
            continue
        seg_nodes = cfg[segment]
        seg_base = len(cfg["backbone"]) if segment == "head" and "backbone" in cfg else 0
        for local_i, node in enumerate(seg_nodes):
            global_i = seg_base + local_i
            if global_i not in indices:
                continue
            old = str(node[2])
            if old == module:
                continue
            new_node = list(node)
            new_node[2] = module
            if arity is not None and len(new_node) > 3 and new_node[3]:
                keep = max(1, arity - 1)
                new_node[3] = list(new_node[3])[:keep]
            seg_nodes[local_i] = new_node
            replaced[global_i] = f"{old} -> {module}"

    cfg["_downsample_replaced"] = replaced
    return cfg
