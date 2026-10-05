"""SPAE-YOLOv8n-pose 配方 —— 在 SPAE-YOLOv8n 基础上**增加关键点（姿态）检测**。

## 这个变体是什么

上游变体 `variants/SPAE-YOLOv8n`（Sensors 2026, DOI 10.3390/s26113424）复现的是
**只出框**的轻量检测器：SIoU + P2 浅层 + ADown + Efficient_UAVDet。
本变体在**完全保留这四个组件**的前提下，把检测头换成姿态头、再加一路关键点回归，
用于在 **RflySim 录制的仿真数据**上训练"检测框 + 机体关键点"的多任务模型。

    组件                    检测变体 SPAE-YOLOv8n        本变体（+ 关键点）
    ----------------------  ----------------------------  ------------------------------
    SIoU（EP7）              box=siou                      box=siou（原样保留）
    P2 浅层（EP2/EP5）        Detect(P2,P3,P4,P5)           **Pose(P2,P3,P4,P5)**
    ADown（EP1）             主干 4 处下采样                原样保留（索引 1/3/5/7）
    Efficient_UAVDet（EP5）  cv2/cv3 分组卷积              cv2/cv3 分组 + **cv4 关键点分支**
    关键点（EP7/新增）        —                             OKS 损失（TinyPoseLoss）

## 关键点口径（9 点，来自论文，**顺序为推断**）

关键点数与语义按 You Zheng et al., *Keypoint-Guided Efficient Pose Estimation and
Domain Adaptation for Micro Aerial Vehicles*, IEEE T-RO 40 (2024) 2967–2983
（DOI 10.1109/TRO.2024.3400938，作者仓库 WindyLab/MAV6D，MIT）的 **9 点机体关键点**。

⚠️ 该论文在付费墙后、作者仓库只有 README，**9 个点的确切顺序无法核对**：
本库的编号表在 `docs/KEYPOINTS.md`，状态标为 `inferred`，并已做成"改数据集 YAML 两行
（`kpt_names` / `flip_idx`）即可校正"。拿到论文 Figure 11 后请校正并把
`tod.kpt_definition_status` 改为 `paper-verified`。

## 与论文/SAPD 的三个刻意差异（必须记录）

1. **不是论文复现，是"SPAE + 关键点"**：SPAE 论文（原变体）本身**没有关键点**，
   本变体是**新组合**，因此**不能**宣称"复现了 Keypoint-Guided 论文的精度"，
   也不能把 SPAE 的 mAP 数字当成本变体的框基线（多了关键点支路 + 换头）。
2. **数据集**：SPAE 论文用 Det-Fly（空对空真实航拍），本变体用 **RflySim 仿真**。
   仿真→真实的域差异是**已知且显著**的（这正是 Keypoint-Guided 论文专门做无监督域适应的原因），
   所以仿真上的数字**不能**直接当作真实场景指标。
3. **关键点损失用 OKS（EP7 `pose="oks"`）而不是论文的质心引导回归网络**：
   论文的贡献之一是"centroid point-guided keypoint localization network"（自定义网络结构）。
   本库的做法是**复用 YOLO 的姿态头**（不改主干/不引入新网络），因为它能在同一套
   EP 消融流水线下工作。**两者的结构不等价**，对比时不要混为一谈。

## 用法

    # 1) 物化 + 结构自检（不需要数据）
    python tools/make_variant.py variants/spae-yolov8n-pose/recipe.py
    python tools/train.py --variant variants/spae-yolov8n-pose/variant.yaml --dry-run

    # 2) 开训前校验你的 RflySim 数据（kpt_shape / flip_idx / 标签列数 / 可见率）
    python tools/check_pose_dataset.py --data configs/_base_/datasets/rflysim-pose.yaml

    # 3) 真训练
    python tools/train.py --variant variants/spae-yolov8n-pose/variant.yaml \\
        --data configs/_base_/datasets/rflysim-pose.yaml --epochs 200 --batch 4 --imgsz 640

    # 4) 评测（整体 + 按目标边长分层的 OKS-AP + 逐关键点尺度诊断）
    python tools/val_pose.py --variant variants/spae-yolov8n-pose/variant.yaml \\
        --weights results/spae-yolov8n-pose/weights/best.pt \\
        --data configs/_base_/datasets/rflysim-pose.yaml --sigma-strategy uniform
"""

from __future__ import annotations

from tod.compose import Variant

#: §3.2：P2 侧先用 1x1 卷积降维并做特征校准（原文未定义 calibration 的具体算子）
P2_PRE = ["Conv", [128, 1, 1]]

#: 9 点机体关键点（顺序见 docs/KEYPOINTS.md，状态 inferred）
KPT_SHAPE = (9, 3)

variant = (
    Variant(
        "spae-yolov8n-pose",
        base="yolov8n",
        task="pose",
        tags=["self-designed", "keypoint", "pose", "small-object", "uav",
              "air-to-air", "spae", "p2-head", "adown", "rflysim", "sim2real"],
        notes=(
            "= SPAE-YOLOv8n（SIoU + P2 浅层 + ADown + Efficient_UAVDet）+ 9 点机体关键点。\n\n"
            "⚠️ 这不是 SPAE 论文的复现（该论文只出框，没有关键点），也不能宣称复现了 "
            "Keypoint-Guided（T-RO 2024）的精度：本库用的是 YOLO 姿态头 + OKS 损失，"
            "而后者用的是自定义的质心引导关键点定位网络，结构不等价。\n\n"
            "⚠️ 9 个关键点的**顺序是推断**（论文付费墙 + 作者仓库只有 README），"
            "定义在 docs/KEYPOINTS.md，状态 tod.kpt_definition_status=inferred；"
            "校正只需改数据集 YAML 的 kpt_names / flip_idx 两行，无需改代码。\n\n"
            "⚠️ 数据集是 RflySim **仿真**录制（域：simulation），与 SPAE 论文的 Det-Fly "
            "真实空对空数据不同域；仿真数字不能直接当真实场景指标 "
            "（Keypoint-Guided 论文专门做无监督域适应正是因为这个 gap）。\n\n"
            "关键点分支的压缩策略：**不压缩**（保持原生两层 3×3）。原因是论文的 g=x/16 只针对"
            "检测头，且关键点分支的中间通道 c4=max(ch[0]//4, nk)=max(8,27)=27 不被 16 的约数整除，"
            "强行分组会退化成 g=1 的**静默失效**（本库已把这种情况改成报错）。"
            "要压请显式给 EP5.keypoint_per_group（例如 9 → g=3）。\n\n"
            "消融列（都只需改配置）：① EP7.pose=False（退回框架原生 KeypointLoss）；"
            "② EP7.sigma_strategy=auto/uniform/balanced；③ EP5.keypoint_branches=False；"
            "④ 去掉 P2（v.without('add_p2')）看框与关键点各自的代价；"
            "⑤ EP7.box=ciou 对照 SIoU 的贡献。"
        ),
    )
    # ---- EP1：主干下采样换成 ADown（论文 §3.3，与检测变体完全一致）----
    .backbone(downsample="ADown", downsample_indices=[1, 3, 5, 7])
    # ---- EP5：轻量检测头（论文 §3.4）+ 关键点分支不压缩（见 notes）----
    .head("Efficient_UAVDet", per_group=16, channels="native", levels=[2, 3, 4, 5],
          keypoint_branches=True, keypoint_per_group=None)
    # ---- EP7：SIoU 框损失（论文 §3.1）+ OKS 关键点损失（本变体新增）----
    #      sigma_strategy=uniform：9 点机体关键点（四个电机/机臂）是同质的，
    #      没有"哪个点更难"的先验 —— 几何曲线会无依据地放松/收紧首尾点。
    #      这是**可消融**的选择，换 auto/balanced 只需改这一个字段。
    .loss(box="siou", theta=4.0, pose="oks", sigma_strategy="uniform", min_sigma=0.001)
    # ---- 数据：RflySim 仿真录制（9 点机体关键点）----
    .data("rflysim-pose", imgsz=640)
    # ---- 模型图：P2 浅层（论文 §3.2）+ 9 点关键点 ----
    #      与检测变体唯一的差别是 kpt_shape 与 task="pose"（→ 用 yolov8n-pose 底座）
    .model(
        nc=1,                  # 单一机型；多机型请改这里并在数据集 YAML 里逐类给 kpt_names
        kpt_shape=KPT_SHAPE,
        add_p2=True,
        p2_idx=2,              # YOLOv8 backbone 第 3 层（索引 2）C2f 输出，stride=4
        p2_channels=128,       # width=0.25 → 32，与论文 Table 3 的 x=32 一致
        p2_pre=P2_PRE,         # 1x1 卷积降维 + 特征校准
        p2_fuse_block="C2f",
    )
    # ---- 训练：SPAE 论文 Table 4 的超参；batch 因多一路关键点 + 8GB 显存下调 ----
    .train(
        optimizer="SGD",
        epochs=200,
        batch=4,               # 论文 32（24GB）；纯检测的 SPAE 在本机用 8，加关键点后取 4
        imgsz=640,
        lr0=0.01,              # 注意：论文的 lr 是按 batch=32 调的，小 batch 需要重调
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        close_mosaic=10,
        pose=12.0,             # 关键点定位损失增益（框架默认 12.0，显式写出便于消融）
        kobj=1.0,              # 关键点可见性损失增益（框架默认 1.0）
        workers=4,
        seed=0,
        deterministic=True,
        amp=True,
        project="results",
    )
)
