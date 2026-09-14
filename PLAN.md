# 小目标检测（Tiny/Small Object Detection, TOD）YOLO 魔改收集库 —— 架构与选型方案

> 文档版本：v0.1 ｜ 更新日期：2026-09-14 ｜ 状态：待评审
> 本文档回答两件事：**(A) 用什么代码架构来"收集大量魔改 YOLO"才不失控；(B) 先收哪些魔改、优先级如何。**

---

## 0. TL;DR

1. **不要为每个魔改 fork 一份 YOLO 代码。** 用 `注册表（registry）+ 单一配置文件 + 扩展点（EP）` 的插件化架构：一个魔改 = 一个模块文件 + 一个 YAML 变体，主干代码永不复制。
2. **把小目标检测拆成 10 个扩展点 EP0–EP9**（数据/输入 → 主干 → 颈部 → 上采样 → 注意力 → 检测头 → 标签分配 → 损失 → 推理后处理 → 训练策略）。每个魔改都能落到唯一一个 EP 上，天然可消融、可组合。
3. **最高性价比的四件事**（先做这四件，再谈花哨模块）：
   - **EP5：加 P2 检测头（stride=4）** —— 小目标检测收益最大、成本最透明的一步。
   - **EP0：高分辨率输入（1536/2048）+ SAHI 切片推理** —— 推理侧零训练成本。
   - **EP7：把 IoU 系损失换成 NWD（归一化高斯 Wasserstein 距离）** —— 极小子目标（<16px）几乎是必需品。
   - **EP1/EP2：SPD-Conv 无损下采样 + 带 P2 节点的 BiFPN/ASFF 融合**。
4. **每个变体必须带"变体卡片"**（论文、来源、许可证、改动点、复现配置、结果、耗时），否则半年后这个库会变成一堆无法解释的散装代码。
5. **评测必须先立基线**：同一数据集、同分辨率、同 epoch、同增强、同 seed 跑 3 次，报 `AP_small` 而不只是 `AP50`。

---

## 1. 范围与问题定义

### 1.1 "小目标"的三种口径（先统一，否则结果不可比）

| 口径 | 定义 | 典型数据集 |
|---|---|---|
| COCO 小目标 | `area < 32²`（约 32×32 像素） | COCO, VisDrone |
| 极小目标 / tiny | `area < 16²` 甚至 `< 8²`；AI-TOD 平均目标仅 ~12.8 px | AI-TOD, xView, TinyPerson |
| 场景口径 | 无人机航拍、遥感、远处行人/车辆、工业缺陷、细胞/人脸 | VisDrone, UAVDT, DOTA, NEU-DET |

**结论：库内所有实验必须显式声明用哪套口径，并同时上报整体 AP 与 `AP_small`。** 只报 AP50 会掩盖小目标退化。

### 1.2 三条主赛道（魔改的有效性高度依赖赛道，不要混着刷榜）

- **A. 无人机 / 航拍**：目标小且密集、类别不均衡、图像分辨率高、算力受限 → 代表作 TPH-YOLOv5、CEASC、UAV-DETR。
- **B. 遥感 / 卫星**：超大图、极端密集、方向性（旋转框）、背景复杂 → 代表作 LSKNet、PKINet、DOTA 系。
- **C. 通用 / 工业 / 生物医学**：单图小目标少但对比度低 → ASF-YOLO（细胞）、YOLO-Face（人脸）、缺陷检测。

---

## 2. 设计原则

1. **配置驱动，不 fork 主干。** 主干（ultralytics / MMYOLO）只做"加载器 + 训练引擎"，所有魔改以插件形式挂载。
2. **一个魔改 = 一个模块 + 一条注册元数据 + 一个变体配置。** 不允许出现"复制一份 yolov8.yaml 改两行"这种不可追溯的操作（这类改动应表达为 YAML 片段覆盖）。
3. **每个改动可单独开关、可单独消融。** 变体的价值不在"涨了几个点"，而在"这个点是谁贡献的、代价多少"。
4. **所有变体共享同一套训练/评测/日志/导出流水线。** 否则组合爆炸后无法批量跑实验。
5. **基线优先（baseline-first）。** 任何新模块进来，先在固定基线上复现论文增益，再进入组合池。
6. **许可证合规。** ultralytics 为 AGPL-3.0、YOLOv5 为 GPL-3.0；引入第三方模块代码时记录来源与许可证，商业用途需提前甄别。
7. **版本可复现。** 固定框架版本 + 记录 commit hash + 固定随机种子；框架升级走 `compat.py` 兼容层。

---

## 3. 推荐目录结构

```
yolo-tod/
├─ PLAN.md                     # 本文档
├─ README.md                   # 快速开始
├─ pyproject.toml              # 依赖（含 ultralytics / mmyolo 引脚）
├─ requirements-lock.txt       # 完整锁定版本，用于复现实验
│
├─ configs/                    # 只放配置，不放代码
│  ├─ _base_/
│  │  ├─ datasets/             # visdrone.yaml / ai-tod.yaml / dota.yaml ...
│  │  ├─ schedules/            # 100e_adamw_cosine.yaml ...
│  │  └─ runtime/              # 单卡/多卡/AMP/梯度检查点
│  ├─ variants/                # ★ 变体 = 组合后的完整配置
│  │  ├─ visdrone/YOLOv8n_P2_DySample_NWD.yaml
│  │  └─ ai-tod/...
│  └─ ablations/               # 消融用的最小覆盖片段（只有 diff）
│
├─ src/tod/
│  ├─ registry.py              # ★ 注册表 + 元数据 + 命名空间注入
│  ├─ compat.py                # 框架版本兼容层（唯一允许碰内部的文件）
│  ├─ compose.py               # ★ 变体构造 DSL（骨架 + EP 覆盖 → 导出 YAML）
│  ├─ modules/
│  │  ├─ conv/                 # SPD-Conv, DCNv4, AKConv, DSConv, RepConv...
│  │  ├─ block/                # C2f/C3k2/GELAN/Rep/FasterNet/PKI 变体
│  │  ├─ attention/            # CA, ECA, SimAM, EMA, LSK, MLCA...
│  │  ├─ neck/                 # BiFPN, AFPN, HS-FPN, Gold-YOLO GD, ASFF, CCFM...
│  │  ├─ upsample/             # CARAFE, DySample, SAPT, WaveletUp
│  │  └─ head/                 # DetectP2, DyHead, QueryDet, DualHead(NMS-free)
│  ├─ assigner/                # TAL, SimOTA, ATSS, RFLA, NWD-assigner
│  ├─ loss/                    # NWD, WiseIoU, InnerIoU, Focaler-IoU, SlideLoss
│  ├─ data/                    # slicing(SAHI), copy-paste, mosaic9, oversample
│  ├─ engine/                  # trainer / validator / predictor 包装
│  ├─ eval/                    # 尺度分层指标、按面积上报、显著性检验
│  └─ utils/                   # 可视化、日志、成本统计(FLOPs/延迟)
│
├─ variants/                   # ★ 论文级完整复现（变体卡片）
│  └─ <name>/
│     ├─ card.md               # 论文/来源/改动点/结论/坑
│     ├─ model.yaml            # 网络结构
│     ├─ train.yaml            # 训练配置
│     └─ results.json          # 指标（machine-readable）
│
├─ third_party/                # 只能整仓引入的魔改（git submodule）
│  └─ README.md                # 来源、commit、许可证、适配说明
│
├─ tools/                      # train.py val.py export.py bench.py
│  └─ ablation.py              # 批量跑消融网格
├─ docs/
│  ├─ VARIANTS.md              # ★ 由 registry 自动生成的清单总表
│  ├─ PROTOCOL.md              # 评测协议
│  └─ COST.md                  # 参数/FLOPs/延迟/显存记录表
└─ results/                    # 实验输出（gitignore，仅保留 summary）
```

**关键点：`modules/` 是"零件库"，`variants/` 是"成品车"。** 两者分离，才能既做组合实验又保留论文可追溯性。

---

## 4. 核心机制

### 4.1 注册表（一切魔改的唯一入口）

```python
# src/tod/registry.py
from dataclasses import dataclass, field

@dataclass(frozen=True)
class ModuleSpec:
    name: str            # 在 YAML 中使用的名字，如 "DySample"
    obj: object          # 实际类/函数
    ep: str              # EP0..EP9 扩展点
    paper: str = ""      # 论文标题
    url: str = ""        # 论文/官方仓库链接
    year: int = 0
    license: str = ""
    cost: str = ""       # 成本提示：参数量/FLOPs/显存/延迟的定性或定量说明
    notes: str = ""
    aliases: tuple[str, ...] = field(default_factory=tuple)

_REGISTRY: dict[str, ModuleSpec] = {}

def register(name=None, *, ep, paper="", url="", year=0, license="",
             cost="", notes="", aliases=()):
    def deco(obj):
        spec = ModuleSpec(name or obj.__name__, obj, ep, paper, url, year,
                          license, cost, notes, tuple(aliases))
        _REGISTRY[spec.name] = spec
        for a in aliases:
            _REGISTRY[a] = spec
        return obj
    return deco

def install(*extra_ns):
    """把注册的模块注入框架的 parse_model 命名空间，使其可在 YAML 中按名引用。"""
    from tod.compat import model_globals
    ns = model_globals()                       # 例如 ultralytics.nn.tasks.__dict__
    for spec in set(_REGISTRY.values()):
        ns[spec.name] = spec.obj
    for m in extra_ns:
        ns.update(m)

def catalog():
    """生成 docs/VARIANTS.md 的模块总表（按 EP 分组）。"""
    ...
```

模块文件示例：

```python
# src/tod/modules/upsample/dysample.py
import torch.nn as nn
from tod.registry import register

@register(ep="EP3",
          paper="DySample: Learning to Upsample by Learning to Sample (ICCV 2023)",
          url="https://arxiv.org/abs/2308.15085",
          year=2023, license="Apache-2.0",
          cost="参数量几乎不变，+~0.1ms/层；对 P2 特征图有轻微显存增量",
          notes="动态采样点，替代最近邻上采样；小目标对 P2 上采样质量敏感")
class DySample(nn.Module):
    def __init__(self, c1, c2, scale=2, style="lp"):
        ...
```

统一入口：

```python
# src/tod/__init__.py
from tod.registry import install
install()
import tod.modules          # 触发所有 @register
```

### 4.2 变体构造 DSL（骨架 + EP 覆盖）

直接手写 ultralytics 风格 YAML 可行但组合爆炸后难维护。推荐用一层薄 DSL 生成 YAML：

```python
# 用法示意
from tod.compose import Variant

v = (Variant("YOLOv8n_P2_DySample_NWD", base="yolov8n")
     .data("visdrone2019-det", imgsz=1536, slicing=dict(patch=1024, overlap=0.2))
     .backbone(conv="SPDConv", block="C2f")
     .neck(fusion="BiFPN", levels="P2-P5", upsample="DySample")
     .head("DetectP2", levels="P2-P5", assigner="TAL", box_loss="NWD")
     .train(epochs=150, optimizer="AdamW", close_mosaic=20, seed=0))

v.dump("configs/variants/visdrone/YOLOv8n_P2_DySample_NWD.yaml")
v.card("variants/YOLOv8n_P2_DySample_NWD/card.md")   # 自动写来源/许可证/改动点
```

**好处**：消融只需 `v.without("DySample")` / `v.patch(EP3=None)`；配置永远由代码生成，不会出现手工 YAML 漂移。

### 4.3 扩展点定义（EP0–EP9）

| EP | 名称 | 允许改动 | 主要接口 |
|---|---|---|---|
| EP0 | 数据 / 输入 | 分辨率、切片、Copy-Paste、过采样、超分 | `Dataset`, `Augment`, `Slicer` |
| EP1 | Backbone | 下采样算子、基础块、轻量化、高分辨率分支 | `Conv`, `Block`, `Stage` |
| EP2 | Neck | 融合拓扑、P2 节点、加权融合、跨层交互 | `Fusion`, `levels` |
| EP3 | 上采样 | 最近邻 → 内容感知/动态采样 | `Upsample` |
| EP4 | 注意力 | 通道/空间/多尺度/大核/无参注意力 | `Attention` |
| EP5 | Head | 输出层数、动态头、查询式头、NMS-free 双头 | `Head`, `DetectP2` |
| EP6 | 标签分配 | TAL / SimOTA / ATSS / RFLA / NWD-assigner | `Assigner` |
| EP7 | 损失 | 框回归损失、分类损失、尺度加权 | `BoxLoss`, `ClsLoss` |
| EP8 | 推理 / 后处理 | SAHI、WBF、Soft-NMS、多尺度 TTA | `Predictor`, `PostProcess` |
| EP9 | 训练策略 | 课程学习、蒸馏、EMA、优化器、微调分辨率 | `Trainer` |

**规则：一个新魔改如果不能明确归到某一个 EP，就不要合进库**（说明它要么是组合，要么还没想清楚）。

### 4.4 变体卡片字段（强制）

```markdown
# <变体名>
- id / base / dataset / imgsz
- EP 改动：EP2: [PAN-P2] | EP3: [DySample] | EP5: [DetectP2] | EP7: [NWD]
- 来源：论文标题、链接、年份、官方仓库、许可证
- 相对基线：ΔAP / ΔAP_small / Δparams / ΔFLOPs / Δlatency / Δ显存
- 复现状态：planned / reproducing / reproduced / failed（失败也要留档！）
- 已知坑 / 与哪些模块冲突 / 超参敏感度
```

### 4.5 三条流水线

- **训练**：`tools/train.py --variant <yaml>`，统一输出到 `results/<variant>/<timestamp>/`。
- **评测**：`tools/val.py` 同时输出整体指标、`AP_small`、按尺度分层召回、切片/非切片两套结果。
- **消融**：`tools/ablation.py` 读一个 EP 模块池，自动生成 `baseline + 单模块 + 累积组合` 的笛卡尔实验，串行/并行调度，汇总成一张表。
- **自动文档**：CI 中跑 `tod.registry.catalog()` 重新生成 `docs/VARIANTS.md`，保证文档与代码不脱节。

### 4.6 第三方整仓魔改的接入规范

有些魔改（如官方 TPH-YOLOv5、CEASC）拆不出干净模块，采用 `third_party/` 子模块 + 适配器：

1. `third_party/<name>/` 作为 git submodule，**记录 commit hash**；
2. `src/tod/adapters/<name>.py` 只做两件事：把它的模块 `@register` 进来 / 把它的配置翻译成我们的变体 YAML；
3. 在 `third_party/README.md` 记录许可证与是否可商用；
4. **不允许修改 submodule 内部代码**（改不动就写 patch 文件，可追溯）。

### 4.7 兼容层是唯一允许碰框架内部的文件

ultralytics 内部 API 变动频繁（`parse_model`、`DetectionModel`、`v8DetectionLoss` 都改过）。全部适配集中到 `compat.py`，其他文件只依赖我们自己的接口。升级框架 = 只改 `compat.py` + 重跑冒烟测试。

---

## 5. 值得放进来的魔改清单

> 优先级：**P0 = 必收**（收益大/成本低/影响面广）；**P1 = 强烈建议**；**P2 = 有价值但边际或成本高**。
> 「预期收益」为文献与社区经验值，**必须以库内自测为准**，不要当成承诺。

### EP0 数据 / 输入（低成本高收益区）

| 优先级 | 魔改 | 关键思想 | 为什么对小目标有用 | 成本/风险 |
|---|---|---|---|---|
| P0 | **SAHI 切片推理 + 切片微调** ([Akyön et al., ICIP 2022](https://arxiv.org/abs/2202.06934)) | 大图切 patch 分别推理再合并 | 小目标在整图缩放后像素太少；切片等效放大 | 推理耗时随切片数线性增长 |
| P0 | **高分辨率输入（1536/2048）** | 提高输入长边 | 最直接地增加小目标像素数 | 显存/耗时平方增长，需配 P2 头 |
| P0 | **小目标 Copy-Paste** ([Kisantal et al., CVPRW 2019](https://arxiv.org/abs/1904.00853)) | 复制小目标粘贴到随机位置 | 直接增加小目标样本数与位置多样性 | 粘贴尺度/上下文不匹配会引入噪声 |
| P1 | **Stitcher** ([Chen et al. 2021](https://arxiv.org/abs/2101.06186)) | 拼图式重排，保持尺度 | 让模型见到更多小尺寸目标 | 破坏原图上下文 |
| P1 | **Mosaic-9 / 多图拼接** | 9 图拼接 | 小目标绝对数量上升 | 与 Copy-Paste 叠加易过增强 |
| P1 | **多尺度训练 + 小目标类别过采样** | 随机缩放 / 类别重采样 | 缓解长尾与尺度失衡 | 需调 `close_mosaic` |
| P1 | **超分辅助（SR + Det 联合/级联）** | 检测前先超分 | 直接提升可辨识度 | 额外算力，可能放大噪声 |
| P2 | **合成小目标（GAN / 3D 渲染）** | 生成稀缺小目标 | 数据层面补长尾 | 域偏移风险 |

### EP1 Backbone

| 优先级 | 魔改 | 来源 | 作用 | 成本 |
|---|---|---|---|---|
| P0 | **SPD-Conv（Space-to-Depth 无损下采样）** | [arXiv 2208.03641](https://arxiv.org/abs/2208.03641) | 替换 stride-2 卷积，避免小目标信息在早期池化中丢失 | 参数量小增 |
| P1 | **DCNv2 / v3 / v4 可变形卷积** | DCNv4 (CVPR 2024) | 采样点自适应，贴合不规则小目标 | v3/v4 需编译算子 |
| P1 | **RepConv / RepVGG 重参数化** | RepVGG (CVPR 2021) | 训练多分支、推理单分支，涨点不涨延迟 | 训练显存略增 |
| P1 | **FasterNet (PConv) / MobileNetV4 / StarNet** | 各原始论文 | 轻量化，为高分辨率输入腾算力 | 绝对精度可能下降 |
| P1 | **PKINet（Poly Kernel Inception）** | [CVPR 2024](https://openaccess.thecvf.com//content/CVPR2024/html/Cai_Poly_Kernel_Inception_Network_for_Remote_Sensing_Detection_CVPR_2024_paper.html) | 大核多尺度感受野，遥感小目标友好 | 大核算子显存开销 |
| P1 | **LSK 大选择性核（LSKNet）** | [ICCV 2023](https://arxiv.org/abs/2303.09030) | 动态调整感受野，适应遥感目标尺度差异 | 需调 kernel 系列超参 |
| P1 | **GELAN / C3k2 / R-ELAN+Area Attention** | YOLOv9 / v11 / v12 | 新主干基础块，作为"现代基线" | 与老模块兼容性 |
| P2 | **AKConv / DSConv / PKI+** | 各原始论文 | 任意形状核、可变核 | 算子实现碎片化 |
| P2 | **HRNet / 高分辨率分支保留** | HRNet (CVPR 2019) | 全程保持高分辨率特征 | 算力大，需裁剪 |

### EP2 Neck（小目标的第二战场）

| 优先级 | 魔改 | 来源 | 作用 |
|---|---|---|---|
| P0 | **加入 P2 融合节点** | YOLOv5/v8-P2 系 | 让 stride-4 特征参与自顶向下/自底向上融合，配 P2 头才有效 |
| P0 | **BiFPN 加权双向融合** | EfficientDet (CVPR 2020) | 可学习权重，抑制小目标路径上的噪声 |
| P0 | **ASFF 自适应空间特征融合** | [arXiv 1911.09516](https://arxiv.org/abs/1911.09516) | 不同尺度特征空间加权，缓解尺度冲突 |
| P1 | **AFPN 渐近特征金字塔** | [CVPR 2023](https://arxiv.org/abs/2306.15988) | 非相邻层渐进融合，减少语义鸿沟 |
| P1 | **HS-FPN 层次化尺度融合** | [arXiv 2402.19298](https://arxiv.org/abs/2402.19298) | 通道+尺度双路筛选，突出显著小目标 |
| P1 | **Gold-YOLO 的 GD 机制** | [NeurIPS 2023](https://arxiv.org/abs/2309.11331) | 汇聚-分发式全局信息注入，低延迟涨点 |
| P1 | **CCFM / SSFF（ASF-YOLO）** | [Image & Vision Computing 2024](https://www.sciencedirect.com/science/article/pii/S0262885624001616) | 通道与尺度序列融合，细胞/小实例分割用它起家 |
| P2 | **SDI 选择性密集尺度交互 / BiFormer 动态稀疏注意力** | 各原始论文 | 更激进的融合/注意力拓扑 |
| P2 | **小波 / 频域融合** | WTConv 等 | 高频细节对小目标敏感 |

### EP3 上采样

| 优先级 | 魔改 | 来源 | 作用 / 结论 |
|---|---|---|---|
| P1 | **DySample** | [ICCV 2023](https://arxiv.org/abs/2308.15085) | 动态采样点，几乎零额外参数，通常优于最近邻与 CARAFE |
| P1 | **CARAFE 内容感知重组** | ICCV 2019 | 经典内容感知上采样，社区实现成熟 |
| P2 | **SAPT 超分式上采样** | TPH-YOLOv5 系 | 上采样即做超分，显存敏感 |
| P2 | **小波上采样 / PixelShuffle 变体** | 各原始论文 | 保留高频，适合纹理型小目标 |

> 经验：**P2 分支的上采样质量对小目标影响远大于主干**，值得单独消融。

### EP4 注意力（注意别无限堆）

| 优先级 | 魔改 | 来源 | 特点 |
|---|---|---|---|
| P0 | **Coordinate Attention (CA)** | [CVPR 2021](https://arxiv.org/abs/2103.02907) | 坐标信息嵌入，方向敏感，几乎无延迟 |
| P1 | **ECA** | [CVPR 2020](https://arxiv.org/abs/1910.03151) | 一维卷积通道注意力，极轻 |
| P1 | **SimAM** | ICML 2021 | **无参数**，即插即用，作为默认兜底 |
| P1 | **EMA** | ICASSP 2023 | 多尺度并行子网 + 跨空间学习，小目标场景常有效 |
| P1 | **LSK（大选择性核）** | ICCV 2023 | 大核感受野 + 空间选择，遥感类常用 |
| P1 | **MLCA 混合局部通道注意力** | 2023 | 兼顾通道与局部空间，轻量 |
| P2 | **CBAM / Triplet / Criss-Cross / Shuffle / SGE / GAM** | 各原始论文 | 作为标准对照组，不建议全堆 |
| P2 | **Deformable Attention / Focal Modulation** | 各原始论文 | 需编译算子或改写量大 |

> 反模式：同一 backbone 里塞 3 种以上注意力模块，收益常常互相抵消、延迟叠加。

### EP5 Head（最高 ROI 区）

| 优先级 | 魔改 | 来源 | 作用 |
|---|---|---|---|
| **P0** | **P2 检测头（stride=4）** | YOLOv5/v8-P2 | 小目标检测的"第一性"改动：让网络直接在 1/4 分辨率上预测 |
| P1 | **DyHead 动态头** | [CVPR 2021](https://arxiv.org/abs/2106.08322) | 尺度/空间/任务三重感知注意力，可叠在 P2–P5 上 |
| P1 | **QueryDet 由粗到细 + 高分辨稀疏卷积** | [CVPR 2022](https://arxiv.org/abs/2109.11844) | 先在低分辨定位候选，再只对高分辨关键位置算卷积，省算力的高分辨方案 |
| P1 | **解耦头 + DFL** | YOLOv8 | 现代基线配置，小目标回归更稳 |
| P1 | **NMS-free 双头（一对多 + 一对一）** | [YOLOv10](https://arxiv.org/abs/2405.14458) | 免 NMS，密集小目标场景下 NMS 误抑制明显减少 |
| P1 | **Transformer 头：RT-DETR / RT-DETRv2 / D-FINE / DEIM** | 各原始论文 | 端到端、免 NMS，DEIM 系在航拍尺度变化场景报告有增益 |
| P2 | **附加 P6 层** | YOLOv5-P6 | **对小目标无用**，除非同时做超大图（保留但标注为"非小目标项"） |

### EP6 标签分配

| 优先级 | 魔改 | 来源 | 作用 |
|---|---|---|---|
| P0 | **RFLA 感受野标签分配** | [TGRS 2022 / arXiv 2204.13317](https://arxiv.org/abs/2204.13317) | 用高斯感受野而非 IoU 匹配，配合 NWD 是 tiny 目标标配 |
| P0 | **TAL（Task-Aligned Assigner）** | TOOD | 现代基线 |
| P1 | **SimOTA / OTA** | YOLOX / OTA | 动态分配，正样本更多对小目标有利 |
| P1 | **ATSS** | CVPR 2020 | 自适应统计分配，稳定对照 |
| P1 | **Dot Distance 分配** | [DOTA 系, arXiv 2103.04551](https://arxiv.org/abs/2103.04551) | 用中心点距离替代 IoU，密集小目标友好 |
| P2 | **软标签 / 模糊正样本** | 各原始论文 | 进一步缓解小目标正样本过少 |

### EP7 损失函数

| 优先级 | 魔改 | 来源 | 作用 |
|---|---|---|---|
| **P0** | **NWD 归一化高斯 Wasserstein 距离** | [arXiv 2110.13389](https://arxiv.org/abs/2110.13389) | IoU 对微小框极其敏感（1px 偏移即可让 IoU 从 0.8 掉到 0.3）；NWD 把框建成高斯分布，尺度平滑，tiny 目标几乎必备 |
| P1 | **Wise-IoU v1/v2/v3** | [arXiv 2301.10051](https://arxiv.org/abs/2301.10051) | 动态非单调聚焦，抑制低质量样本 |
| P1 | **Inner-IoU** | [arXiv 2311.02877](https://arxiv.org/abs/2311.02877) | 用辅助框计算 IoU，加速收敛，便于调尺度 |
| P1 | **Focaler-IoU** | [arXiv 2401.10525](https://arxiv.org/abs/2401.10525) | 聚焦难/易样本区间 |
| P1 | **Slide Loss** | YOLO-Face | 强调难样本，人脸/小目标类常用 |
| P2 | **MPDIoU / PIoU / SIoU / EIoU** | 各原始论文 | 作为对照组合 |
| P2 | **VFL / QFL / GFL** | VarifocalNet 等 | 分类-定位联合质量建模 |

### EP8 推理 / 后处理

| 优先级 | 魔改 | 作用 |
|---|---|---|
| **P0** | **SAHI 推理（切片 + 重叠 + 合并）** | 训练不动、直接涨小目标召回，几乎是免费午餐 |
| P1 | **WBF / Soft-NMS** | 密集小目标下比硬 NMS 更稳 |
| P1 | **多尺度 TTA** | 稳妥但慢，作为上限参考 |
| P1 | **小目标专用后处理** | 低置信度保留 + 按面积调整 NMS IoU 阈值 |
| P1 | **切片训练 + 整图推理（或反之）** | 跨域一致性，常用于航拍 |

### EP9 训练策略

| 优先级 | 魔改 | 作用 |
|---|---|---|
| P0 | **高分辨率微调（低分辨率预训练 → 高分辨微调）** | 省算力且效果接近全程高分辨 |
| P1 | **课程学习 / 渐进式分辨率** | 稳定收敛 |
| P1 | **超参工程：优化器、EMA、warmup、`close_mosaic`** | 常常比换模块涨得更多 |
| P2 | **知识蒸馏（大模型/高分辨率教师 → 小模型）** | 轻量部署路线的关键 |
| P2 | **自训练 / 伪标签 / 半监督** | 小目标标注漏标严重时收益大 |

---

### 5.10 论文级"整篇魔改"（值得作为完整变体收录）

| 名称 | 赛道 | 核心卖点 | 收录方式 |
|---|---|---|---|
| **TPH-YOLOv5 / TPH-YOLOv5++** | 航拍 VisDrone | Transformer 预测头 + 额外小目标头 + 复制粘贴增强 | 建议整仓 submodule + 适配器 |
| **CEASC** ([CVPR 2023](https://openaccess.thecvf.com//content/CVPR2023/html/Du_Adaptive_Sparse_Convolutional_Networks_With_Global_Context_Enhancement_for_Faster_CVPR_2023_paper.html)) | 航拍 | 自适应稀疏卷积 + 全局上下文增强，加速密集小目标推理 | 模块化（EP1/EP4） |
| **Gold-YOLO** (NeurIPS 2023) | 通用 | GD 汇聚-分发颈部，低延迟涨点 | 模块化（EP2） |
| **LSKNet / LSKNet-RS** (ICCV 2023) | 遥感 | 大选择性核动态感受野 | 模块化（EP1/EP4） |
| **PKINet** (CVPR 2024) | 遥感 | 多尺度大核 Inception 主干 | 模块化（EP1） |
| **ASF-YOLO** (IVC 2024) | 细胞/实例 | 注意力尺度序列融合 + SSFF | 模块化（EP2） |
| **QueryDet** (CVPR 2022) | 通用高分辨 | 由粗到细的稀疏高分辨检测 | 模块化（EP5） |
| **YOLOv9 / v10 / v11 / v12 / v13** | 通用 | GELAN / NMS-free / C3k2+PSA / Area Attention+R-ELAN / 超图增强 | 作为"现代基线骨架"整代收录 |
| **RT-DETR / RT-DETRv2 / D-FINE / DEIM** | 通用/航拍 | 端到端 Transformer 检测，免 NMS | 作为对照家族 |
| **YOLO-Face / YOLOv5-Face** | 人脸小目标 | Slide Loss、五点回归 | 变体（EP7） |
| **ClusDet / DMNet / CRENet** | 密集人群小目标 | 密度图/聚类裁剪 | 作为"两阶段裁剪"对照（EP0/EP8） |

---

## 6. 推荐的三条基线组合（先跑通这三条）

### 路线 1：最小改动基线（1 天可跑通）
```
YOLOv8n + P2 头 + imgsz=1536 + 原损失
```
用途：确定"加 P2 + 提分辨率"到底值多少，作为所有后续改动的对照锚点。

### 路线 2：性价比组合（推荐作为默认基线）
```
YOLOv8s
 ├ EP0: imgsz=1536, Copy-Paste, close_mosaic
 ├ EP1: SPD-Conv 下采样
 ├ EP2: PAN + P2 节点 + BiFPN
 ├ EP3: DySample
 ├ EP4: CA（仅 P2/P3 分支）
 ├ EP5: DetectP2（P2–P5 四层头）
 ├ EP6: TAL
 ├ EP7: NWD + Wise-IoU 组合
 └ EP8: SAHI 推理
```
用途：库的"招牌基线"，所有新模块都在它上面测增量。

### 路线 3：冲榜组合（算力换点）
```
路线 2 + QueryDet/DyHead + DCNv4 + AFPN/HS-FPN + 蒸馏 + 多尺度 TTA
```
用途：探上限，同时记录 FPS/显存，避免"刷点不可部署"。

**禁忌**：不要在没跑通路线 1 的情况下直接上路线 3，否则无法归因。

---

## 7. 数据集与评测协议

### 7.1 数据集建议

| 数据集 | 赛道 | 特点 | 用途 |
|---|---|---|---|
| **VisDrone2019-DET** | 航拍 | 10 类，目标小且密集 | 主战场 A |
| **UAVDT** | 航拍车辆 | 3 类，运动模糊 | 补充 A |
| **AI-TOD** | 极小目标 | 平均 ~12.8px，8 类 | **tiny 口径主战场**，配 NWD/RFLA |
| **TinyPerson** | 极小行人 | 平均 ~20px，海面/远景 | 极端 tiny 下限测试 |
| **SODA-A / SODA-D** | 航拍 | 大规模 | 规模验证 |
| **DOTA-v1.5 / v2.0** | 遥感 | 超大图、旋转框、密集 | 主战场 B（可先做水平框子集） |
| **xView** | 遥感 | 极端小 + 长尾 | 压力测试 |
| **NEU-DET / 工业缺陷集** | 工业 | 低对比度小缺陷 | 主战场 C |

### 7.2 指标（必须全报）

- 常规：`AP50`、`AP50:95`、`Precision`、`Recall`、每类 AP。
- **小目标专属**：`AP_small`（area<32²）、`AP_tiny`、**按尺度分层的 Recall**（<8px / 8–16 / 16–32 / 32–96）。
- 成本：参数量、FLOPs、**实测延迟（目标设备，batch=1）**、峰值显存、训练 GPU 小时。
- 部署视角：切片推理的端到端延迟、导出格式（ONNX/TensorRT）后的精度损失。

### 7.3 公平性规则

1. 同一数据集、同 `imgsz`、同 epoch、同增强、同优化器；**seed ≥ 3 次，报 mean±std**。
2. **两种对齐口径都要报**：参数量对齐、FLOPs/延迟对齐（否则大模型涨点毫无意义）。
3. 消融必须包含：`baseline`、`+单模块`、`+累积组合`、`去掉单模块（leave-one-out）`。
4. 每次实验落 `results.json`，不进 git 的巨大权重另存。
5. **失败的复现也要写进卡片**（"这个模块在我们的基线上没涨"是高价值信息）。

---

## 8. 落地路线图

| 阶段 | 目标 | 交付物 |
|---|---|---|
| **M0 骨架**（~2 天）✅ 架构层完成 | registry + compose DSL + train/val 打通 | 见 §12 落地进展 |
| **M1 基线**（~1 周） | 三条路线的路线 1、2 跑通并入库 | `docs/VARIANTS.md` 初版 + 基线数字 |
| **M2 批量收录**（~2–3 周） | 按 P0 清单把 EP0–EP8 的必收项全部实现 | 模块库覆盖 P0 全项 + 每项单模块消融结果 |
| **M3 组合与筛选**（~2 周） | 用 `ablation.py` 跑组合网格 | 一张"模块 × ΔAP_small × Δ延迟"的决策表 |
| **M4 扩展**（持续） | P1/P2 模块、第三赛道数据集、蒸馏与部署 | 变体卡片 + 部署基线 |

---

## 9. 已知风险与坑

1. **框架版本漂移**：ultralytics 内部 API 变动频繁 → 全部适配收敛到 `compat.py`，并锁定版本。
2. **许可证**：ultralytics 是 **AGPL-3.0**，YOLOv5 是 **GPL-3.0**。收录第三方代码必须登记许可证；若有商业部署需求，需从架构上预留"可替换主干"的抽象。
3. **P2 头显存爆炸**：1/4 分辨率特征图面积是 1/8 的 4 倍。对策：AMP、梯度检查点、减小 batch、只在 P2 用轻量块、QueryDet 式稀疏计算。
4. **高分辨率 + 切片 = 延迟失控**：必须同时记录延迟，并为部署提供"关闭 SAHI / 降分辨率"的降级路径。
5. **过增强**：Mosaic + Copy-Paste + Stitcher 叠加会让分布严重偏离真实数据，务必做增强强度的消融。
6. **只看 AP50 掩盖退化**：密集小目标场景 AP50 涨、`AP_small` 掉是常见现象，两个都要看。
7. **标注漏标**：tiny 数据集漏标极普遍，会导致损失震荡；考虑忽略区域（ignore region）或软标签。
8. **模块堆叠负交互**：注意力/DCN/大核常互相抵消。**默认单模块进入，组合必须实测**。
9. **复现口径不一致**：论文的 epoch、增强、输入分辨率往往与我们的基线不同，直接对比论文数字是无效的，必须在本库基线内复现。
10. **文档腐化**：禁止手工维护清单，`docs/VARIANTS.md` 一律由 registry 生成。

---

## 10. 附录

### 10.1 命名规范

- 模块类名：与论文一致（`DySample`、`SPDConv`、`C2f_EMA`），别名注册常见写法（`Dysample`、`dysample`）。
- 变体 id：`<数据集>-<base>-<EP 关键改动串联>`，如 `visdrone-yolov8s-p2-bifpn-dysample-nwd`。
- 配置文件：与变体 id 同名 `.yaml`，避免歧义。

### 10.2 变体状态机

```
planned → reproducing → reproduced → (promoted | dropped)
                     ↘ failed  （保留失败记录与原因）
```

### 10.3 参考链接（本方案引用的主要来源）

- 小目标检测综述：[Small object detection: A comprehensive survey (2025)](https://www.sciencedirect.com/science/article/pii/S2667305325000870)
- 航拍实时检测综述：[Recent Real-Time Aerial Object Detection Approaches (Sensors 2025)](https://www.mdpi.com/1424-8220/25/24/7563)
- NWD / RFLA：[A Normalized Gaussian Wasserstein Distance for Tiny Object Detection](https://arxiv.org/abs/2110.13389)、[RFLA](https://arxiv.org/abs/2204.13317)
- SAHI：[Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection](https://arxiv.org/abs/2202.06934)
- 稀疏卷积加速（航拍）：[CEASC, CVPR 2023](https://openaccess.thecvf.com//content/CVPR2023/html/Du_Adaptive_Sparse_Convolutional_Networks_With_Global_Context_Enhancement_for_Faster_CVPR_2023_paper.html)
- 遥感大核主干：[PKINet, CVPR 2024](https://openaccess.thecvf.com//content/CVPR2024/html/Cai_Poly_Kernel_Inception_Network_for_Remote_Sensing_Detection_CVPR_2024_paper.html)
- 尺度序列融合：[ASF-YOLO (Image and Vision Computing 2024)](https://www.sciencedirect.com/science/article/pii/S0262885624001616)
- 上采样：[DySample, ICCV 2023](https://arxiv.org/abs/2308.15085)
- 颈部与分配：[AFPN, CVPR 2023](https://arxiv.org/abs/2306.15988)、[Gold-YOLO, NeurIPS 2023](https://arxiv.org/abs/2309.11331)、[HS-FPN](https://arxiv.org/abs/2402.19298)、[SpotNet / Dot Distance](https://arxiv.org/abs/2103.04551)
- 头与端到端：[QueryDet, CVPR 2022](https://arxiv.org/abs/2109.11844)、[DyHead, CVPR 2021](https://arxiv.org/abs/2106.08322)、[YOLOv10](https://arxiv.org/abs/2405.14458)
- 损失：[Wise-IoU](https://arxiv.org/abs/2301.10051)、[Inner-IoU](https://arxiv.org/abs/2311.02877)、[Focaler-IoU](https://arxiv.org/abs/2401.10525)

---

## 11. 已确认的决策（2026-09-14）

| # | 决策项 | 结论 |
|---|---|---|
| 1 | 主干框架 | **ultralytics**（AGPL-3.0；registry/compose 层保持框架解耦，为将来换 MMYOLO 留口子） |
| 2 | 第一主战场 | **航拍 VisDrone2019-DET**；AI-TOD（tiny 口径）与 DOTA（遥感）后置 |
| 3 | 旋转框 OBB | 暂不引入；EP5/EP6/EP7 预留角度分支接口，但 M0–M3 不做 |

### 11.1 本机环境实况（重要约束）

| 项 | 实测值 | 影响 |
|---|---|---|
| GPU | **NVIDIA RTX 5060 Laptop, 8 GB** | 显存是硬约束，`imgsz=1536` + P2 头会非常吃力 |
| Python | 3.14.5 | torch/ultralytics 对新版 Python 的轮子支持滞后，**建议另建 Python 3.11/3.12 虚拟环境** |
| 已装 | 仅 numpy 2.4.6 | torch / ultralytics 均未安装，需先装环境 |

**由此调整的三条工程约束：**

1. **分辨率策略改为渐进式**：`1024 → 1280 → 1536` 分阶段验证，不要一上来就 1536。
2. **8 GB 显存下的参考配置**：`imgsz=1280, batch=4~8, AMP=True`；开 P2 头时优先 `batch=2~4` + 梯度检查点；
   若 OOM，优先降 batch 而不是降分辨率（降分辨率会直接毁掉小目标收益）。
3. **延迟必须实测**：本机 GPU 是笔记本级，FPS 数字仅作相对比较，不代表部署平台。

### 11.2 仍然开放的问题

1. **是否需商用**：决定第三方 GPL/AGPL 代码的收录边界（ultralytics 本体为 AGPL-3.0）。
2. **可用 GPU 小时**：决定 M2/M3 阶段消融网格的规模（全组合 vs 贪心筛选）。

---

## 12. 落地进展

### M0 架构层（已完成，`python tests/smoke.py` 35 项检查全绿）

| 文件 | 作用 | 状态 |
|---|---|---|
| `src/tod/registry.py` | 注册表：强制登记 EP 归属 / 论文 / 许可证 / 成本；重复注册与非法 EP 直接报错；`catalog()` 自动生成模块总表 | ✅ |
| `src/tod/compat.py` | 唯一触碰 ultralytics 内部的兼容层：命名空间注入、内置模型 YAML 定位、版本区间校验 | ✅ |
| `src/tod/compose.py` | 变体 DSL：`patch/without/data/train/model` → 变体配置、模型 YAML、变体卡片；含 **P2 头注入**与节点类型替换 | ✅ |
| `tests/smoke.py` | 无 torch 依赖的架构冒烟测试（35 项），含 P2 注入的索引正确性验证 | ✅ |
| `tools/catalog.py` | 生成 `docs/VARIANTS.md` | ✅ |
| `configs/_base_/datasets/visdrone2019-det.yaml` | VisDrone 数据配置 + 尺度分层/长尾类别标注 | ✅ |
| `pyproject.toml` / `.gitignore` / `README.md` | 工程外壳 | ✅ |

**已验证的关键逻辑**（对官方 yolov8 同构图做单元验证）：

- `inject_p2_head` 生成的 `Detect` 输入为 `[24, 15, 18, 21]`，上采样取自 P3 节点、Concat 接 backbone P2、
  融合块通道经 width 缩放，**backbone 不被改动**，且**重复调用幂等**；
- `apply_type_map` 可把 `Conv→SPDConv`、`nn.Upsample→DySample` 全图替换并统计命中数；
- 变体配置/卡片可落盘，卡片自动带出来源与许可证表格。

### 尚未完成（阻塞项）

1. **训练环境未装**：本机只有 numpy + pyyaml，缺 torch / ultralytics。
   且 `pip` 在 workspace-write 沙箱下无法写用户级 site-packages（需放宽权限或改用 venv 后申请一次）。
2. **`inject_p2_head` 只覆盖直连式 P2 头**：完整 P2 双向融合 / BiFPN / AFPN 重拓扑属于 M1。
3. **模块库为空**：`src/tod/modules/` 尚无模块，需按 §5 的 P0 清单逐项落地。

### M0 剩余任务（下一步）

- [ ] 用 Python 3.12 建 venv，安装 `torch`（CUDA 版）+ `ultralytics`，跑通真实模型构建（`tests/smoke.py` 的最后一项会从 SKIP 变为真实校验）
- [ ] `tools/train.py` / `tools/val.py` / `tools/ablation.py`
- [ ] 尺度分层评测（`src/tod/eval/`）：在上报中固定输出 `AP_small` / `AP_tiny` / 分层召回
- [ ] 路线 1 基线：`YOLOv8n + P2 头 + imgsz=1280`，记录 AP/显存/延迟作为锚点
