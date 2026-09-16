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
        """收集本变体引用到的、已在注册表中登记的模块名。

        同时扫描各 EP 覆盖与 ``model`` 段（P2 预处理、融合块等也常填模块名），
        用于在变体卡片里自动列出论文来源与许可证。
        """
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
        """从 ``dump()`` 写出的 spec 还原变体对象。

        这样训练脚本可以直接读变体配置，并**现场重新生成**模型 YAML，
        保证 model.yaml 与变体配置永不脱节（不需要把生成物手工 commit 后再维护）。
        """
        v = cls(
            name=str(data.get("id") or data.get("name") or "unnamed"),
            base=str(data.get("base", "yolov8n")),
            dataset=str(data.get("dataset", "")),
            status=str(data.get("status", "planned")),
            tags=list(data.get("tags") or []),
            notes=str(data.get("notes", "")),
        )
        v.eps = {k: dict(x) for k, x in (data.get("eps") or {}).items()}
        v.data_cfg = dict(data.get("data") or {})
        v.train_cfg = dict(data.get("train") or {})
        v.model_cfg = dict(data.get("model") or {})
        return v

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

        改造按"先结构、后改名"的顺序执行，顺序不可颠倒：
          1. ``downsample`` —— 按节点索引把主干下采样换成 ADown 等模块
          2. ``add_p2``      —— 注入 P2（stride=4）检测分支
          3. ``type_map``    —— 节点类型批量改名

        **检测头（EP5）刻意不做 YAML 改名**：ultralytics 的 ``parse_model`` 用
        精确类成员判断（``m in {Detect, ...}``）来给检测头追加输入通道列表 ``ch``，
        子类不会被识别，写进 YAML 会因为缺少 ``ch`` 而构造失败。因此 EP5 的改动
        记录到 ``_head_surgery``，由 ``tod.engine.surgery`` 在模型构建完成后
        就地替换分支卷积——不改框架源码，也不依赖该判断的实现细节。

        更复杂的颈部重拓扑（BiFPN / AFPN / HS-FPN）属于 M1 工作。
        """
        cfg, scale = self._load_base_cfg()

        if self.model_cfg.get("nc") is not None:
            cfg["nc"] = self.model_cfg["nc"]
        if scale:
            cfg["scale"] = scale
            # 为什么要把所选规模排到 scales 的**第一位**：
            # ultralytics 的 ``yaml_model_load`` 会用**文件名**猜规模
            # （``guess_model_scale`` 只认 yolo<数字><nsmlx> 这种名字），我们生成的
            # 文件叫 model.yaml，猜不出来 → ``d["scale"] = ""``，YAML 里写的 scale
            # 被覆盖；``parse_model`` 于是退化为 ``next(iter(scales.keys()))``。
            # 对 yolov8n/yolo26n 恰好等于 'n' 所以一直没暴露，换成 s/m/l/x 就会
            # **静默建错规模的模型**。把所选规模放首位即可彻底消除这个隐患。
            scales = cfg.get("scales")
            if isinstance(scales, dict) and scale in scales:
                cfg["scales"] = {
                    scale: scales[scale],
                    **{k: v for k, v in scales.items() if k != scale},
                }

        # ---- 1) 结构：主干下采样替换 ----
        ds_module = self.model_cfg.get("downsample") or self.eps.get("EP1", {}).get("downsample")
        if ds_module:
            replace_downsample(
                cfg,
                module=str(ds_module),
                indices=self.model_cfg.get("downsample_indices")
                or self.eps.get("EP1", {}).get("downsample_indices"),
            )

        # ---- 2) 结构：P2 检测分支 ----
        if self.model_cfg.get("add_p2"):
            p2_pre = self.model_cfg.get("p2_pre")
            if isinstance(p2_pre, dict):        # {type: Conv, args: [64,1,1]}
                p2_pre = [p2_pre["type"], p2_pre.get("args", [])]
            inject_p2_head(
                cfg,
                p2_idx=int(self.model_cfg.get("p2_idx", 2)),
                p2_channels=int(self.model_cfg.get("p2_channels", 64)),
                fuse_block=str(self.model_cfg.get("p2_fuse_block", "C2f")),
                upsample=str(self.eps.get("EP3", {}).get("upsample") or "nn.Upsample"),
                p2_pre=p2_pre,
            )

        # ---- 3) 改名：EP 覆盖 → 节点类型替换 ----
        type_map: dict[str, str] = dict(self.model_cfg.get("type_map", {}))
        # 只映射语义等价的节点类型；注意力类模块不是 Conv 的等价替换，
        # 必须显式写在 type_map 或颈部配置里。
        ep_targets = (("EP3", "upsample", "nn.Upsample"),
                      ("EP1", "conv", "Conv"),
                      ("EP1", "block", "C2f"))
        for ep, key, target in ep_targets:
            new = self.eps.get(ep, {}).get(key)
            if isinstance(new, str) and new and new != target:
                type_map[target] = new

        if type_map:
            apply_type_map(cfg, type_map)

        # ---- 4) 检测头：记录为建模后手术，见上面的说明 ----
        head_name = self.eps.get("EP5", {}).get("head")
        if isinstance(head_name, str) and head_name not in ("", "Detect"):
            cfg["_head_surgery"] = head_name

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

        # 模型图级改造（P2 注入 / 下采样替换 / 类型替换）不走 EP 覆盖，单列一节，
        # 否则卡片上会看不到"加了 P2 头"这种最关键的改动。
        if self.model_cfg:
            lines += ["", "## 模型图改造（model 段）", "", "| 键 | 值 |", "|---|---|"]
            for key, value in self.model_cfg.items():
                lines.append(f"| `{key}` | `{value}` |")
            derived = []
            if self.model_cfg.get("add_p2"):
                derived.append(
                    "**P2 检测头（EP5/EP2）**：`inject_p2_head` 注入 stride=4 分支 —— "
                    f"上采样(P3) ⊕ backbone 节点 {self.model_cfg.get('p2_idx', 2)} → "
                    f"`{self.model_cfg.get('p2_fuse_block', 'C2f')}` 融合 → Detect(P2,P3,P4,P5)")
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
    """定位检测头节点（兼容 Detect / DetectP2 / Efficient_UAVDet 等命名）。"""
    for i in range(len(nodes) - 1, -1, -1):
        if "detect" in str(nodes[i][2]).lower():
            return i
    raise ValueError("未在模型图中找到检测头节点（类型名含 'detect'）。")


def apply_type_map(cfg: dict, type_map: dict[str, str]) -> dict:
    """按 ``{旧类型: 新类型}`` 替换节点类型；就地修改 cfg（节点本身替换为新列表）。

    节点采用"替换而非改写"的方式，避免污染调用方浅拷贝共享的节点对象。
    """
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
    """把 P2（stride=4）分支接进检测头 —— PLAN.md 中最高 ROI 的改动。

    生成的结构（以 YOLOv8 为例，P3 头节点为 15、P2 源为 2）::

        [2, 1, Conv, [64, 1, 1]]                     # 可选：P2 侧 1x1 降维/校准
        [-1, 1, nn.Upsample, [None, 2, "nearest"]]   # 由 P3 上采样
        [[-1, 2], 1, Concat, [1]]                    # 与 backbone P2 拼接
        [-1, 3, C2f, [64]]                           # 融合
        [[24, 15, 18, 21], 1, Detect, [nc]]          # Detect(P2, P3, P4, P5)

    这是"直连式 P2 头"：只做一次自顶向下融合，不做完整双向重拓扑。

    Args:
        p2_idx: backbone 中 P2（stride=4）特征源的全局节点索引。
            YOLOv8 系列为 2；换主干后请对照模型 YAML 注释确认。
        p2_channels: 融合块输出通道（会被 width 缩放）。
        upsample: 上采样模块名，可填注册表中的名字（如 ``DySample``）。
        p2_pre: 可选的 P2 侧预处理节点，形如 ``[类型, [参数...]]``。
            SPAE-YOLOv8 §3.2 在此处用 1x1 卷积降维并做特征校准，
            对应 ``["Conv", [64, 1, 1]]``。
    """
    nodes, base, _ = _node_lists(cfg)
    det_i = _find_detect(nodes)
    src = list(nodes[det_i][0])

    if len(src) == 4:
        cfg["_p2_status"] = "already_present"
        return cfg
    if len(src) != 3:
        raise ValueError(f"预期 Detect 有 3 个输入（P3,P4,P5），实际 {src}。")

    p3_idx, p4_idx, p5_idx = src
    det_global = base + det_i          # 插入后 Detect 自身将占用的全局索引
    offset = 0

    # 1) 可选的 P2 侧降维/校准
    if p2_pre is not None:
        pre_type, pre_args = p2_pre[0], list(p2_pre[1])
        nodes.insert(det_i + offset, [p2_idx, 1, pre_type, pre_args])
        p2_src = det_global + offset
        offset += 1
    else:
        p2_src = p2_idx

    # 2) P3 上采样
    nodes.insert(det_i + offset, [p3_idx, 1, upsample, [None, 2, "nearest"]])
    offset += 1

    # 3) 与 P2 拼接
    nodes.insert(det_i + offset, [[-1, p2_src], 1, "Concat", [1]])
    offset += 1

    # 4) 融合
    fuse_idx = det_global + offset
    nodes.insert(det_i + offset, [-1, repeats, fuse_block, [p2_channels]])
    offset += 1

    # 5) Detect 输入扩展为 P2-P5
    #    写回的是**新列表**而非就地修改：调用方可能传入浅拷贝的节点列表
    #    （例如 {"model": backbone + head}），就地改写会污染其原图。
    updated = list(nodes[det_i + offset])
    updated[0] = [fuse_idx, p3_idx, p4_idx, p5_idx]
    nodes[det_i + offset] = updated
    cfg["_p2_status"] = "injected"
    cfg["_p2_channels"] = p2_channels
    return cfg


def _module_arity(name: str) -> int | None:
    """模块 ``__init__`` 的**必需**位置参数个数（不含 self）；拿不到返回 None。

    用于把 Conv 节点改型成别的模块时裁剪参数：YAML 里 stride-2 Conv 的
    ``[c2, k, s]`` 对 ``ADown(c1, c2)`` 来说是超编的（8.4 会直接 TypeError）。
    """
    import inspect

    obj = None
    try:
        from tod.compat import model_globals

        obj = model_globals().get(name)
    except Exception:  # noqa: BLE001 - 未安装框架时退回注册表查找
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
    """把主干中的 stride-2 下采样卷积换成给定模块（如 ADown）。

    为什么不能直接用 ``apply_type_map``：ADown 只在**下采样位置**语义等价，
    把全图 Conv 都换掉会破坏 neck 与 head。所以这里按**节点索引**精确替换。

    Args:
        module: 替换后的模块名。
        indices: 全局节点索引列表，默认 ``(1, 3, 5, 7)``
            —— YOLOv8 主干的 P2/P3/P4/P5 下采样点（索引 0 是 stem，
            输入通道为 3（奇数），ADown 这类需要通道一分为二的模块不能放）。
        segments: 在哪些段里替换，默认只改 backbone。
        trim_args: 是否把节点参数裁剪到目标模块的必需参数个数（默认 True）。
            原节点的参数是 Conv 的 ``[c2, k, s]``，而 ``ADown(c1, c2)`` 只吃两个位置参数
            （c1 由 ``parse_model`` 自动补），不裁剪会在建图时报
            ``ADown.__init__() takes 3 positional arguments but 5 were given``。

    Returns:
        就地修改后的 cfg，并写入 ``_downsample_replaced`` 供核查。
    """
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
            if old == module:                     # 幂等
                continue
            new_node = list(node)
            new_node[2] = module
            if arity is not None and len(new_node) > 3 and new_node[3]:
                keep = max(1, arity - 1)          # parse_model 会补 c1
                new_node[3] = list(new_node[3])[:keep]
            seg_nodes[local_i] = new_node
            replaced[global_i] = f"{old} -> {module}"

    cfg["_downsample_replaced"] = replaced
    return cfg
