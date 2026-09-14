# yolo-tod —— 小目标检测魔改 YOLO 收集库

收集、复现并**可归因地**对比针对小目标检测（Tiny/Small Object Detection）的 YOLO 魔改。
完整方案见 [`PLAN.md`](PLAN.md)：架构设计、值得收录的魔改清单、优先级、评测协议。

**当前状态：M0（骨架）已完成并通过冒烟测试；尚未安装训练环境。**

- 主干框架：**ultralytics**（AGPL-3.0）
- 第一主战场：**VisDrone2019-DET**（航拍小目标）
- 本机：RTX 5060 Laptop **8 GB** ／ Python 3.14.5 → 建议另建 **Python 3.12** 虚拟环境

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
src/tod/          registry / compose / compat + 模块库(modules/)
configs/          _base_ 基础配置、variants 变体、ablations 消融片段
variants/         论文级复现：card.md + model.yaml + results.json
third_party/      只能整仓引入的魔改（git submodule + 适配器）
tools/            catalog.py（文档生成）等
tests/smoke.py    架构层冒烟测试（无 torch 依赖）
```

## 下一步（M1）

1. 装好 torch/ultralytics 环境，跑通「YOLOv8n + P2 头 + imgsz=1280」最小基线；
2. 按 `PLAN.md §5` 的 P0 清单落地模块（SAHI、Copy-Paste、SPD-Conv、BiFPN/ASFF、NWD…）；
3. 每加一个模块，同步补 `tests/smoke.py` 用例与 `python tools/catalog.py --write`。

> 注意：`PLAN.md §11.1` 记录了 8 GB 显存下的分辨率/batch 约束，不要一上来就用 1536。
