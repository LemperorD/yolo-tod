# yolo-tod —— 小目标检测魔改 YOLO 收集库

收集、复现并**可归因地**对比针对小目标检测（Tiny/Small Object Detection）的 YOLO 魔改。
完整方案见 [`PLAN.md`](PLAN.md)：架构设计、值得收录的魔改清单、优先级、评测协议。

**当前状态：M0 骨架已完成（51 项冒烟检查全绿）；首个变体 SPAE-YOLOv8n 已实现，待环境就绪后验证。**

- 主干框架：**ultralytics**（AGPL-3.0）
- 第一主战场：**VisDrone2019-DET**（航拍小目标）
- 本机：RTX 5060 Laptop **8 GB** ／ Python 3.14.5 → 建议另建 **Python 3.12** 虚拟环境

## 已实现的变体

### SPAE-YOLOv8n —— 空对空微小型无人机检测（Sensors 2026）

论文 [SPAE-YOLOv8](https://doi.org/10.3390/s26113424) 的四个组件全部落地：

| 组件 | 本库位置 | 要点 |
|---|---|---|
| **P2 浅层** | `compose.inject_p2_head` | 160×160 分支，P2-P3-P4-P5 四层头；1×1 降维 |
| **ADown** | `modules/conv/adown.py` | 双分支自适应下采样（avg-pool → Conv3×3/s2 与 MaxPool+Conv1×1） |
| **SIoU** | `loss/box.py` + `loss/criterion.py` | 角度+距离+形状三项代价，只替换 IoU 项、复用框架 DFL |
| **Efficient_UAVDet** | `modules/head/efficient_uavdet.py` | 分支 stem 换成两层 3×3 分组卷积，**g = x/16**（每组 16 通道） |

```powershell
# 物化变体（生成 variant.yaml / model.yaml / card.md）
python tools\make_variant.py variants\SPAE-YOLOv8n\recipe.py

# 结构自检：建模型 + 一次前向，不需要数据集
python tools\train.py --variant variants\SPAE-YOLOv8n\variant.yaml --dry-run

# 真训练（需先准备 VisDrone）
python tools\train.py --variant variants\SPAE-YOLOv8n\variant.yaml --data configs\_base_\datasets\visdrone2019-det.yaml
```

取证笔记（含论文原文摘录、消融表、**两处论文内部数据冲突**、以及"哪些是原文、哪些是我们的推断"）
见 `variants/SPAE-YOLOv8n/paper-notes.md`。

> ⚠️ 两个必须知道的结论：① 论文消融显示 **P2 贡献 +7.5pp 是绝对主力**，而 Efficient_UAVDet 是
> **−0.4pp 的"降本不涨点"**（参数 −20%、FPS +28%），论文自己也承认；② 论文用 Det-Fly
> 空对空数据集，本库换成 VisDrone 空对地，**论文的 mAP 数字不可直接引用**。

## 核心思想

**一个魔改 = 一个注册模块 + 一份元数据 + 一个变体配置。永不 fork 主干代码。**

- `src/tod/registry.py` —— 所有魔改的唯一入口，强制登记论文/许可证/扩展点归属
- `src/tod/compose.py` —— 变体 DSL：`骨架 + 扩展点覆盖 → 模型 YAML / 变体卡片`
- `src/tod/compat.py` —— **唯一**允许触碰 ultralytics 内部结构的文件
- 扩展点 `EP0–EP9` —— 数据 / 主干 / 颈部 / 上采样 / 注意力 / 头 / 分配 / 损失 / 推理 / 训练策略

## 快速开始

```powershell
# 1) 建环境（强烈建议 3.12，torch 轮子最稳）
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]

# 2) 架构层冒烟测试（不需要 GPU / torch）
python tests\smoke.py

# 3) 查看已登记的魔改模块总表
python tools\catalog.py
```

## 最小示例：构造一个「P2 头 + DySample + NWD」变体

```python
from tod.compose import Variant

v = (Variant("visdrone-yolov8n-p2-dysample-nwd", base="yolov8n")
     .data("visdrone2019-det", imgsz=1280, slicing={"patch": 1024, "overlap": 0.2})
     .upsample("DySample")          # EP3
     .loss(box="NWD")               # EP7
     .head(levels=[2, 3, 4, 5])     # EP5：加 P2 检测头
     .model(add_p2=True, nc=10)
     .train(epochs=150, batch=4, amp=True))

v.dump("configs/variants/visdrone/visdrone-yolov8n-p2-dysample-nwd.yaml")
v.model_yaml("variants/visdrone-yolov8n-p2-dysample-nwd/model.yaml")
v.card("variants/visdrone-yolov8n-p2-dysample-nwd/card.md")
```

## 目录

```
src/tod/
├─ registry.py      魔改注册表（强制元数据）
├─ compose.py       变体 DSL + 模型图改造（P2 注入 / 下采样替换 / 类型替换）
├─ compat.py        唯一触碰 ultralytics 内部的地方
├─ runtime.py       变体 spec 的运行时上下文
├─ modules/
│  ├─ conv/adown.py            ADown（EP1）
│  └─ head/efficient_uavdet.py Efficient_UAVDet（EP5）
├─ loss/box.py + criterion.py  SIoU 与训练准则接入（EP7）
└─ engine/trainer.py + surgery.py  自定义 Trainer 与检测头"建模后手术"
configs/_base_/     数据集等基础配置
variants/<name>/    recipe.py（代码定义）+ variant.yaml + model.yaml + card.md + paper-notes.md
tools/              make_variant.py / train.py / catalog.py
tests/              smoke.py（无 torch 依赖，51 项）+ test_modules.py（形状/换头/端到端）
```

## 下一步（M1）

1. 装好环境（conda + Python 3.12 + torch/ultralytics），跑
   `python tools\train.py --variant variants\SPAE-YOLOv8n\variant.yaml --dry-run`，
   确认 P2 头、ADown、Efficient_UAVDet 三处改造全部生效并记录参数量；
2. 准备 VisDrone，跑通「YOLOv8n + P2 + imgsz=640」干净基线作为锚点；
3. 把 SPAE 的四个组件做单模块消融（`v.without(...)`），复现论文的贡献排序；
4. 再按 `PLAN.md §5` 的 P0 清单补模块（SAHI、Copy-Paste、SPD-Conv、BiFPN/ASFF、NWD…）。

> 注意：`PLAN.md §11.1` 记录了 8 GB 显存下的分辨率/batch 约束，不要一上来就用 1536。
