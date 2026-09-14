"""SPAE-YOLOv8n 变体配方（论文架构的严格复现）。

论文：Rushang Zhang, Xiaogang Fu, "SPAE-YOLOv8 for Onboard Real-Time Perception:
Lightweight Small UAV Detection from Air-to-Air Perspectives", Sensors 2026, 26(11):3424.
DOI: 10.3390/s26113424 ｜ PMC: PMC13258917 ｜ 许可证：CC BY 4.0

SPAE 的四个组成部分（论文摘要原文）：“SIoU loss, P2 shallow feature layer,
ADown adaptive downsampling, and Efficient_UAVDet lightweight detection head.”

    组件              本库落地位置                              对应论文
    ----------------  --------------------------------------  -----------------------------
    SIoU              EP7 box=siou + 训练准则替换（criterion）  §3.1（式 1–7）
    P2 浅层           model.add_p2 + p2_pre 1x1 降维/校准      §3.2（160×160，P2-P3-P4-P5）
    ADown             EP1 downsample=ADown（主干 4 处下采样）   §3.3（双分支自适应下采样）
    Efficient_UAVDet  EP5 head=Efficient_UAVDet（建模后手术）   §3.4（式 10–11，Fig.4）

举证要点（详见同目录 paper-notes.md 与 evidence/）：
  * Efficient_UAVDet = 分类/回归分支前**两层连续 3×3 分组卷积**，k=3/s=1/p=1，
    分组数 **g = x/16**（每组 16 通道），末端 1×1 保持普通卷积；
  * 论文 §4.5 明确承认该头是"压缩 + 加速"而非涨点手段（单独换头 mAP@0.5 −0.4pp，
    但 Params 3.0M→2.4M、FPS 197.2→252.9），且当前版本**不含 channel shuffle**。

────────────────────────────────────────────────────────────────────────────
与论文的**刻意差异**（必须记录，否则结论会被误读）
────────────────────────────────────────────────────────────────────────────
1. **数据集**：论文用 Det-Fly（空对空微小型无人机，13,271 张，目标平均占 0.12%），
   本库第一主战场是 VisDrone2019-DET（空对地）。这是域迁移，**论文的 mAP 数字
   不能直接引用**，必须在本库基线内重测（PLAN.md §7.3）。
2. **batch**：论文 batch=32（RTX 4090 24GB）。本机 RTX 5060 Laptop 仅 8GB，
   降到 8。注意论文的 lr0=0.01 是按 batch=32 调的，小 batch 下需要重新调 lr
   （或改用 `nbs` 归一化），否则不是公平复现。
3. **p2_channels=128**：论文 Table 3 给出 P2 的 x=32，而 width=0.25 缩放后
   128→32，故 P2 融合块原始通道取 128。这样检测头输入通道恰为
   32/64/128/256，与 Table 3 的 g=2/4/8/16 完全吻合（这是反向验证）。
4. **未标注项**：论文未说明是否使用预训练权重（全文检索无命中）、
   §3.2 的 "feature calibration" 无任何定义、P2 拼接后接什么模块未写明。
   本实现按 YOLOv8 惯例处理并已在 paper-notes.md 中标注为推断。

用法::

    python tools/make_variant.py variants/SPAE-YOLOv8n/recipe.py
    python tools/train.py --variant variants/SPAE-YOLOv8n/variant.yaml --dry-run
"""

from __future__ import annotations

from tod.compose import Variant

#: §3.2：P2 侧先用 1x1 卷积降维并做特征校准（原文未定义 calibration 的具体算子）
P2_PRE = ["Conv", [128, 1, 1]]

variant = (
    Variant(
        "SPAE-YOLOv8n",
        base="yolov8n",
        tags=["paper-repro", "small-object", "lightweight", "uav"],
        notes=(
            "论文原设定：Det-Fly 数据集、batch=32、SGD、200 epochs；"
            "本变体迁移到 VisDrone2019-DET 且 batch 降到 8（8GB 显存），"
            "属于域迁移 + 超参改动，论文 mAP 数字不可直接引用。\n\n"
            "论文消融（对 baseline 0.850）：P2 +7.5pp（主贡献）、ADown +0.4pp、"
            "SIoU +0.1pp、Efficient_UAVDet −0.4pp（论文自认是压缩/加速手段）。\n\n"
            "引用论文数字时注意其表间冲突：Table 6 baseline 记 0.850，"
            "而 Table 5/7 记 0.922；同一配置 FPS 在 Table 6 为 203.0、Table 5 为 161.5。"
        ),
    )
    # ---- EP1：主干下采样换成 ADown（论文 §3.3）----
    # 索引 0 是 stem（输入 3 通道，奇数），ADown 要把通道一分为二，故不含 0
    .backbone(downsample="ADown", downsample_indices=[1, 3, 5, 7])
    # ---- EP5：轻量检测头（论文 §3.4）----
    # channels="native"：保持框架原生分支通道，只把两层 stem 卷积换成分组卷积
    .head("Efficient_UAVDet", per_group=16, channels="native", levels=[2, 3, 4, 5])
    # ---- EP7：SIoU 回归损失（论文 §3.1）----
    .loss(box="siou", theta=4.0)
    # ---- 数据 ----
    .data("visdrone2019-det", imgsz=640)
    # ---- 模型图：P2 浅层（论文 §3.2）+ VisDrone 10 类 ----
    .model(
        nc=10,
        add_p2=True,
        p2_idx=2,              # YOLOv8 backbone 第 3 层（索引 2）C2f 输出，stride=4
        p2_channels=128,       # width=0.25 → 32，与论文 Table 3 的 x=32 一致
        p2_pre=P2_PRE,         # 1x1 卷积降维 + 特征校准
        p2_fuse_block="C2f",
    )
    # ---- 训练：论文 Table 4 的超参，仅 batch 因显存下调 ----
    .train(
        optimizer="SGD",
        epochs=200,
        batch=8,               # 论文为 32（24GB 卡）；8GB 卡下调
        imgsz=640,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        close_mosaic=10,
        workers=4,
        seed=0,
        deterministic=True,
        amp=True,
        project="results",
    )
)
