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

## 关键点口径（9 点，**按预印本**：1 质心 + 8 角点）

关键点来自 Ye Zheng et al., *Keypoint-Guided Efficient Pose Estimation and Domain
Adaptation for Micro Aerial Vehicles*, IEEE T-RO 40 (2024) 2967–2983
（DOI 10.1109/TRO.2024.3400938，作者仓库 WindyLab/MAV6D MIT）。
**预印本 PDF 已存于 `docs/`**，本变体按它落地了三件事：

1. **点结构**（§III-A-1 原文）："nine keypoints ... one centroid point (numbered p0)
   and eight corner points (numbered p1 ∼ p8)" → `kpt_names` 的首项是 `centroid`，
   其余 8 个是角点；质心**不可自遮挡**，是引导其余 8 点的锚点；
2. **3D 几何损失 `L_rep`**（式 6/8）：罚"预测相机射线"与"真值 3D 关键点"的**横向**偏差
   —— 沿视线的深度误差无法从单目 2D 观测判别，故不罚。论文消融显示它是
   KDA / Lrep / MFMF 三个组件里增益最大的（**10°10cm 提升 > 10%**）；
3. **训练策略**（§IV-A-2 原文）："we only use the 2-D regression loss L_loc ... in the
   **first 15 training epochs**" → 由 `tod.reproj_warmup_epochs: 15` 驱动。

⚠️ **p1~p8 的逐一编号语义**在论文 Fig. 2(a)（图片）里，PDF 文本层给不出，
因此按"前/后 × 左/右 × 上/下"推定，定义与状态写在 `docs/KEYPOINTS.md`。
校正只改数据集 YAML 的 `kpt_names` / `flip_idx` 两行，**不用改代码**。

## 与论文的刻意差异（必须记录）

1. **不是复现**：论文用 **CPKL-Net**（1×1 卷积 + EfficientRep + Rep-PAN + MFMF +
   网格三分支回归），本库用 **YOLOv8 Pose 头**承载关键点，结构不等价；
   论文的 2D 损失是像素 MSE，本库用的是 OKS。**不能宣称复现其精度**。
2. **数据集**：论文用自建的真实 MAV6D（室内 VICON，57 075 张），本变体用 **RflySim 仿真**。
   仿真→真实的域差异显著（论文专门做了无监督域适应），仿真数字不能当真实指标。
3. **6D 位姿输出与 PnP 未接**：论文的 10°10cm 与位姿层面 ADD 依赖 PnP，本库只出 2D 关键点。
   本库的 ADD 用真值深度（关键点层面的 3D 误差），**与论文含 PnP 的 ADD 不可直接比较**。
4. **未实现**：域适应（论文贡献之三）、KDA / MFMF 增强、PnP → 10°10cm 指标。

## 用法

    # 1) 物化 + 结构自检（不需要数据）
    python tools/make_variant.py variants/spae-yolov8n-pose/recipe.py
    python tools/train.py --variant variants/spae-yolov8n-pose/variant.yaml --dry-run

    # 2) 数据到位后先校验（flip_idx / 6D 位姿行序 / 相机内参 / 3D 点个数）
    python tools/check_pose_dataset.py --data configs/_base_/datasets/rflysim-pose.yaml

    # 3) 真训练（前 15 epoch 只用 2D 项，之后自动加入 L_rep）
    python tools/train.py --variant variants/spae-yolov8n-pose/variant.yaml \\
        --data configs/_base_/datasets/rflysim-pose.yaml --epochs 200 --batch 4 --imgsz 640

    # 4) 评测（OKS 分层 + 逐点诊断 + ADD/ADD(0.5d)）
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
            "= SPAE-YOLOv8n（SIoU + P2 浅层 + ADown + Efficient_UAVDet）+ 论文口径的 9 点机体关键点。\n\n"
            "按 Keypoint-Guided（T-RO 2024，预印本存于 docs/）落地三件事："
            "① 点结构 = 1 个质心点 p0 + 8 个角点 p1~p8（§III-A-1 原文）；"
            "② 3D 几何损失 L_rep（式 6/8）——只罚垂直于相机射线的横向偏差，"
            "论文消融里增益最大（10°10cm 提升 >10%）；"
            "③ 训练策略：前 15 epoch 只用 2D 项，之后加入 L_rep（§IV-A-2）。\n\n"
            "本库对论文唯一一处**明确修正**：式 (8) 的文字把 v_i 说成「预测的关键点」，"
            "照字面取 v=预测像素坐标会得到与预测无关、梯度恒 0 的量；"
            "本库按几何唯一说得通的读法取「穿过预测点的相机射线方向」，详见 paper-notes §3.2。\n\n"
            "⚠️ 这不是论文复现：论文用 CPKL-Net（EfficientRep + Rep-PAN + MFMF + 网格三分支），"
            "本库用 YOLOv8 Pose 头承载关键点；2D 损失用 OKS 而非论文的像素 MSE。"
            "不能宣称复现其精度。\n"
            "⚠️ 数据集是 RflySim **仿真**（论文是真实 MAV6D），域差异显著。\n"
            "⚠️ p1~p8 的逐一编号语义在 Fig. 2(a)（图片）里，按「前后×左右×上下」推定，"
            "状态见 docs/KEYPOINTS.md；校正只改数据集 YAML 两行。\n"
            "⚠️ tod.object_points 与 tod.camera_matrix 必须换成你机型的真实几何"
            "（论文没给这两张表，配置里是占位值）。\n\n"
            "关键点分支的压缩策略：**不压缩**（论文的 g=x/16 只针对检测头，"
            "且 c4=max(ch[0]//4, nk)=27 与输入 32 的公约数为 1，强行分组只会静默退化成 g=1）。\n\n"
            "消融列（都只需改配置）：① EP7.reproj=False / reproj_weight=0（复现论文 baseline 列）；"
            "② tod.reproj_warmup_epochs=0（关 warm-up）；③ EP7.pose=False（退回框架 KeypointLoss）；"
            "④ EP7.sigma_strategy=auto/balanced；⑤ EP5.keypoint_per_group；"
            "⑥ v.without('add_p2') 看 P2 对框与关键点各自的代价。"
        ),
    )
    # ---- EP1：主干下采样换成 ADown（论文 §3.3，与检测变体完全一致）----
    .backbone(downsample="ADown", downsample_indices=[1, 3, 5, 7])
    # ---- EP5：轻量检测头（论文 §3.4）+ 关键点分支不压缩（见 notes）----
    .head("Efficient_UAVDet", per_group=16, channels="native", levels=[2, 3, 4, 5],
          keypoint_branches=True, keypoint_per_group=None)
    # ---- EP7：SIoU 框损失（论文 SPAE §3.1）+ OKS 关键点损失 + **L_rep 几何项** ----
    #      sigma_strategy=uniform：9 点机体关键点（质心 + 8 角点）无"哪个点更难"的先验。
    #      reproj=True：启用 Keypoint-Guided 论文式 (8) 的 3D 几何重投影损失
    #        （罚"预测相机射线"与"真值 3D 关键点"的横向偏差；沿视线的深度误差不罚）。
    #        论文消融里它是三个组件中增益最大的（10°10cm 提升 >10%），
    #        且规定前 15 epoch 只用 2D 项（warm-up 由 tod.loss.reproj 实现）。
    #        需要数据集提供 tod.camera_matrix / tod.object_points 与 labels_6d 位姿，
    #        缺任何一个都会**明确报错**（不会静默跳过）。
    .loss(box="siou", theta=4.0, pose="oks", sigma_strategy="uniform",
          min_sigma=0.001, reproj=True, reproj_weight=1.0)
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
