"""visdrone-yolo26n-pose-p2-p16 配方 —— 航拍小目标**姿态关键点**检测（本库自研，非论文复现）。

## 这个变体要回答的问题

小目标检测（本库主线）与关键点检测的交集有一个**被 OKS 口径掩盖的坑**：

    OKS 的分母含框面积（``2·σ²·area``）。同一个 3 px 的定位误差，
    在 120 px 的人体上 OKS ≈ 0.95，在 8 px 的行人上 OKS ≈ 0.2。

于是"整体 OKS-AP 涨了"完全可能对应"小目标关键点全废"。本变体把这个问题做成
**可归因的配置**，而不是靠感觉调：

| 改动 | EP | 期望 | 怎么验证 |
|---|---|---|---|
| **P2 关键点头（stride=4）** | EP5/EP2 | 关键点分辨率 ×2（P2 分支的关键点预测在 1/4 分辨率） | `tools/val_pose.py` 的 `lt8`/`8-16` 两层 OKS-AP |
| **OKS 损失（EP7 显式化）** | EP7 | 把 sigma 策略与可见性开关从框架默认变成可消融项 | `tests/test_pose.py` 的数值对照 + `pose=native` 消融列 |
| **关键点分支是否压缩** | EP5 | 分组卷积压扁关键点分支**可能**伤 tiny 关键点精度 | `EP5.keypoint_branches` 开关（默认 False = 不压关键点） |
| 框损失 Wise-IoU v3 | EP7 | 与检测变体一致，抑制低质量样本 | 整体 AP 不退化 |

## 与检测变体的关系（不是同一件事）

* ``visdrone-yolo26n-pose-p2-p16`` 的**框**部分与 SDD-YOLO26n 同源（YOLO26 + P2 + Wise-IoU + STAL），
  但关键点是**新增的一路输出**，参数量与显存都会涨（关键点分支 = 每个尺度一条卷积 + Pose26 的归一化流）；
* 因此本变体**不能**与 SDD 的 AP 直接比较"谁更强"—— 任务不同（一个出框，一个出框+17 点）。
  可比的是 **框的 AP**（同样输入分辨率、同样 P2），以及"加关键点后框 AP 掉多少"。

## ⚠️ 数据集状态（引用任何数字前必读）

VisDrone2019-DET **没有关键点标注**。本变体声明的目标数据集 ``visdrone2019-pose``
目前是**接口占位**（``configs/_base_/datasets/visdrone2019-pose.yaml`` 里
``tod.kpt_annotation: none``），标注来源与协议写在该文件的注释里。

因此在拿到真实标注之前：

* 本变体只能跑 **结构自检**（``--dry-run``）与 **合成数据回路自检**
  （``tools/make_dummy_dataset.py --task pose`` + ``tests/train_pose_smoke.py``）；
* 变体卡片的精度栏**必须留空**，不能写任何来自合成数据的数字（合成数据的"人"
  是我画的火柴人，分布与真实航拍行人毫无关系）。

## 与"参考实现"的关系

本变体的做法与 **FlyPose**（WACV 2026，航拍人体姿态，https://arxiv.org/abs/2601.05747）
的问题设定一致（航拍 + 小尺度人体 + 高分辨率输入），但**不是它的复现**：
FlyPose 的网络设计、训练策略与数据集我们都没有采用，只是问题设定重叠。
把它列出来是为了避免"我们以为自己在复现它"这种误读。

用法::

    python tools/make_variant.py variants/visdrone-yolo26n-pose-p2-p16/recipe.py
    python tools/train.py --variant variants/visdrone-yolo26n-pose-p2-p16/variant.yaml --dry-run
"""

from __future__ import annotations

from tod.compose import Variant

variant = (
    Variant(
        "visdrone-yolo26n-pose-p2-p16",
        base="yolo26n",
        task="pose",
        tags=["self-designed", "keypoint", "pose", "small-object", "uav",
              "p2-head", "oks", "tiny-keypoint", "dataset-pending"],
        notes=(
            "本库自研变体（非论文复现）：把航拍小目标检测的能力延伸到**关键点**。\n\n"
            "解决的核心问题：OKS 的分母含框面积，同一个像素误差在 8 px 行人上的 OKS 远低于 "
            "120 px 人体 —— 只报整体 OKS-AP 会把'小目标关键点全废'看成'基本没变'。"
            "因此本变体固定配套 tools/val_pose.py：整体 + 按目标边长分层 + **逐关键点尺度诊断**"
            "（≤1px 命中率 / 平均误差 / 平均 OKS）。\n\n"
            "⚠️ 数据集状态：VisDrone2019-DET 没有关键点标注，目标数据集 visdrone2019-pose "
            "当前是**接口占位**（tod.kpt_annotation: none）。在真实标注到位前，"
            "本变体只能做结构自检与合成数据回路自检，**卡片精度栏必须留空**。\n\n"
            "与检测变体的可比性：框部分与 SDD-YOLO26n 同源（YOLO26 + P2 + Wise-IoU + STAL），"
            "但多了一路关键点输出，任务不同，AP 不可直接横向比较；可比的是框 AP 与"
            "'加关键点后框 AP 掉多少'。\n\n"
            "与 FlyPose（WACV 2026, arXiv:2601.05747，航拍人体姿态）的关系：问题设定重叠，"
            "**非复现**（未采用其网络与训练策略，仅作为数据集/问题设定的参考）。\n\n"
            "已知待验证：① P2 关键点头对 lt8 层 OKS-AP 的增益需要真实标注才能测；"
            "② 本变体**不用 EP5 换头**（保持原生 Pose26 头）—— 分组卷积压缩对 tiny 关键点"
            "精度的影响未知，若要用，把 ``.head(\"Efficient_UAVDet\")`` 打开并用 "
            "``keypoint_branches`` 做开关消融；③ Pose26 的归一化流（RealNVP + RLE 损失）在"
            "极小目标上是否稳定未知 —— rle 增益保留框架默认 1.0。"
        ),
    )
    # ---- EP7：框损失沿用 SDD 的 Wise-IoU v3；关键点损失显式换成 TinyPoseLoss ----
    #      keypoint branch 的 sigma 取 person（COCO 17 点标准），与数据集 kpt_shape 对应
    .loss(
        box="wiou",
        kind_kwargs={"variant": 3, "alpha": 1.9, "delta": 3.0},
        pose="oks",                 # EP7 显式化：可写成 False 做"框架原生 KeypointLoss"消融列
        sigma_strategy="person",    # person / auto / balanced（见 tod/loss/pose.py）
        min_sigma=0.001,            # σ 下限：防止某个点的 (2σ)²·area 下溢导致饱和
    )
    # ---- EP6：STAL 小目标感知分配（与 SDD 一致；小目标正样本本来就少）----
    .assigner("STAL", small_target_aware=True)
    # ---- 数据：VisDrone-Pose 接口（标注待准备）；1024 输入（PLAN §11.1 的显存约束）----
    .data("visdrone2019-pose", imgsz=1024)
    # ---- 模型图：P2 分支 = 上采样(P3) ⊕ backbone P2 → C3 瓶颈；关键点形状 17×3 ----
    .model(
        nc=10,                      # VisDrone 10 类（pedestrian/people 才是有关键点的类）
        kpt_shape=(17, 3),          # 数据集 YAML 里的同名键优先级更高
        add_p2=True,
        p2_idx=2,                   # yolo26 backbone 索引 2（C3k2, stride=4）
        p2_channels=128,            # width=0.25 → 32 通道，与 P3/P4/P5 成体系
        p2_fuse_block="C3",         # 与 SDD 一致，便于"加关键点前后"的框 AP 对照
    )
    # ---- EP9：姿态超参显式化（框架把它们放在 model.args 里，容易看漏）----
    #      键名必须是**框架的真实参数名**（pose/kobj/rle），写错会被框架直接拒绝。
    #      这三个增益最终进 train_args（见 tools/train.py），并出现在 dry-run 自检日志里。
    .strategy(
        prog_loss="framework",      # YOLO26 的 E2ELoss.update 即 ProgLoss
        distill=None,               # 蒸馏默认关闭（教师在本机 8 GB 放不下）
    )
    # ---- 训练：与 SDD 的 8 GB 约束一致；姿态多一条关键点支路，显存更紧 ----
    .train(
        dfl=0.0,                    # YOLO26 底座 reg_max=1，此项本就是 0 增益
        pose=12.0,                  # 关键点定位损失增益（框架默认 12.0）
        kobj=1.0,                   # 关键点可见性损失增益（框架默认 1.0）
        rle=1.0,                    # Pose26 归一化流的 RLE 增益（框架默认 1.0）
        epochs=100,                 # 数据集未标注，epoch 待真实数据到位后重定
        batch=4,                    # 1024 输入 + P2 关键点分支，8 GB 的保守值
        imgsz=1024,
        lr0=0.01,
        lrf=0.01,
        momentum=0.9,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        close_mosaic=10,
        mosaic=1.0,
        mixup=0.05,
        multi_scale=0.25,
        amp=True,
        workers=4,
        seed=0,
        deterministic=True,
        project="results",
    )
)
