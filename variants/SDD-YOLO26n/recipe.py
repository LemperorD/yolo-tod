"""SDD-YOLO26n 变体配方（SDD-YOLO 的忠实落地）。

论文：Pengyu Chen, Haotian Sa, Yiwei Hu, Yuhan Cheng, Junbo Wang,
"SDD-YOLO: A Small-Target Detection Framework for Ground-to-Air Anti-UAV
Surveillance with Edge-Efficient Deployment", arXiv:2603.25218（2026-08-10，
东南大学吴健雄学院 / 信息科学与工程学院）。

论文摘要自述的四大贡献（逐条对应本变体的落点）：

    贡献                                          本库落点
    --------------------------------------------  ----------------------------------------
    (i)  DroneSOD-30K 数据集（约 30K 张 G2A 图）   不可得 → 换 VisDrone2019-DET（域迁移）
    (ii) P2 高分辨率检测头（4× 下采样）+ 双注意力  EP5/EP2（compose 注入 P2，C3 融合）
                                                  + EP4（DualAttention）
    (iii) 采纳 YOLO26 的三大创新：                 EP7（wiou + dfl=0.0）/ EP8+EP5（end2end）
          DFL-free、NMS-free、MuSGD+ProgLoss+STAL  / EP9（MuSGD）+ 框架 E2ELoss.update（ProgLoss）
                                                  + EP6（STAL）
    (iv) 推理效率验证（226 FPS @5090 / 35 FPS @Xeon）  本机为 8 GB 笔记本 GPU，仅作相对比较

【四件事不是我们发明的，而是底座自带的 —— 但都必须显式验证】
    ultralytics 8.4 的 ``yolo26.yaml`` 自带 ``end2end: True`` 与 ``reg_max: 1``，
    框架的 ``E2ELoss.update()`` 就是论文所说的 ProgLoss（O2M 权重 0.8→0.1 衰减），
    ``TaskAlignedAssigner`` 也已内置小目标先验（STAL）。本变体的价值在于把它们
    **接进本库的可消融配置**并逐项验证，而不是重新发明：
      * 一条命令即可 ``v.without("wiou")`` / ``stal=False`` / ``use_dfl`` 做消融，
        对应论文 Table 3 的 ¬DFL / NMS-free / MuSGD / STAL 四列；
      * 每一项在 tests/test_modules.py 里有数值或结构断言。

【与论文的刻意差异（必须记录，否则结论会被误读）】
 1. **数据集**：DroneSOD-30K 未公开（论文只给了统计量：训练 30 655 / 验证 14 010 /
    测试 3 085 张 —— 三者相加 47 750，与摘要"约 30K"自相矛盾）。
    本库用 VisDrone2019-DET（空对地航拍），属域迁移，**论文的 86.0 mAP@0.5 不可直接引用**。
 2. **输入分辨率**：论文 §4.2 明写 1024×1024，这里保留 1024。
 3. **batch**：论文未给 batch（RTX 5090 32 GB）；本机 RTX 5060 Laptop 8 GB，
    按 PLAN §11.1 把 batch 压到 4（P2 头 + 1024 输入）。
 4. **教师模型**：论文用 YOLO26x（58.81 M）做蒸馏。本机 8 GB 装不下
    （x 在 1024 输入下的激活远超 8 GB），因此配方里蒸馏默认**不启用**，
    只把接线方式写在 EP9 里（``distill`` + ``teacher`` 指向本地 .pt 即可打开）。
 5. **未标注项**：论文没给 epoch 数、优化器 lr 调度、是否用预训练权重、
    STAL 的具体形式、式 (7) 的锚点归一化方式 —— 均在 paper-notes.md 里逐条标注为推断。

用法::

    python tools/make_variant.py variants/SDD-YOLO26n/recipe.py
    python tools/train.py --variant variants/SDD-YOLO26n/variant.yaml --dry-run
"""

from __future__ import annotations

from tod.compose import Variant

#: §4.6 式 (5) 的 MuSGD：框架原生实现优先（含 YOLO26 辅助头 3× lr 细节）
MUSGD = {"optimizer": "MuSGD", "prefer_native": True}

variant = (
    Variant(
        "SDD-YOLO26n",
        base="yolo26n",
        tags=["paper-repro", "small-object", "uav", "ground-to-air",
              "nms-free", "dfl-free", "distillation"],
        notes=(
            "论文原设定：DroneSOD-30K（未公开，约 30K 张空对地无人机图）、1024×1024 输入、"
            "MuSGD + ProgLoss + STAL、DFL-free、NMS-free、YOLO26x 教师蒸馏（λ=0.5, T=3.0）。\n\n"
            "本变体差异：① 数据集换成本库主战场 VisDrone2019-DET（域迁移，86.0 mAP@0.5 不可引用）；"
            "② batch 因 8 GB 显存降到 4（论文用 32 GB 的 5090，未给 batch）；"
            "③ 蒸馏默认关闭（YOLO26x 教师在本机放不下），键位已备好，给本地 .pt 路径即可启用。\n\n"
            "论文自相矛盾之处（引用数字前必读）：\n"
            "  · 消融表 Table 3 的基线 0.8341 与主表 Table 2 的 YOLO26n 原生 0.786 对不上；"
            "同一配置 '+P2' 在 Table 3 是 0.8419、Table 2 是 0.849。\n"
            "  · Table 2/4 里 YOLO26n、'+P2'、'Final' 三行的 Params(2.50M) 与 FLOPs(5.77G) "
            "完全相同 —— 加了 P2 头与注意力不可能一个参数都不变。\n"
            "  · 数据集三个子集相加 47 750 张 ≠ 摘要'约 30K'；Table 2 的 CPU FPS 列与 "
            "Table 4 的同一模型也有出入（30.4 vs 30.4、35.0 vs 35.0 一致，但 "
            "YOLOv5n 34.9 与'低于 YOLO26n 的 30.4'叙述矛盾）。\n"
            "  · 式 (4) 的空间分支公式（[AvgPool;MaxPool]+Conv7×7）与正文'捕捉运动区域'的描述不符；"
            "式 (7) 没说 KL 在锚点维求和还是求均值（本库取均值：求和时 KD 会压垮任务损失）。\n"
        ),
    )
    # ---- EP4：双注意力（§4.5 式 4），插在 neck→head 之间，颈部零改动 ----
    .attention("DualAttention", levels=[2, 3, 4, 5], reduction=16)
    # ---- EP6：STAL 小目标感知分配（§4.6）----
    .assigner("STAL", small_target_aware=True)
    # ---- EP7：Wise-IoU v3（§4.3 式 3）+ 无 DFL（train 里的 dfl=0.0）----
    .loss(box="wiou", kind_kwargs={"variant": 3, "alpha": 1.9, "delta": 3.0})
    # ---- EP9：MuSGD（§4.6 式 5）；ProgLoss 由框架 E2ELoss.update 提供 ----
    #        蒸馏默认关闭：把 distill/teacher 打开即可（见文件头差异 ④）
    .strategy(**MUSGD, prog_loss="framework", distill=None,
              kd_lambda=0.5, temperature=3.0, kd_levels=[2, 3, 4, 5])
    # ---- 数据：论文 1024×1024；数据集换 VisDrone（域迁移）----
    .data("visdrone2019-det", imgsz=1024)
    # ---- 模型图（§4.2）：P2 分支 = 上采样(P3) ⊕ backbone P2 → C3 瓶颈 ----
    .model(
        nc=10,                  # VisDrone 10 类
        add_p2=True,
        p2_idx=2,               # yolo26 backbone 索引 2（C3k2，stride=4）即 P2 源
        p2_channels=128,        # width=0.25 → 32 通道，与 P3/P4/P5 的 64/128/256 成体系
        p2_fuse_block="C3",     # §4.2 原文：fusing ... via a C3 bottleneck
    )
    # ---- 训练：论文 §4.6 的 MuSGD + dfl=0.0；batch 受 8 GB 显存约束 ----
    .train(
        dfl=0.0,                # §4.3：DFL 分支增益置 0（连计算一起省掉，见 trainer）
        epochs=100,             # 论文未标注，取框架默认
        batch=4,                # 论文未标注；本机 8 GB + 1024 输入的保守取值
        imgsz=1024,
        lr0=0.01,               # 框架 build_optimizer 对 MuSGD 的推荐值
        lrf=0.01,
        momentum=0.9,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        close_mosaic=10,
        mosaic=1.0,
        mixup=0.05,             # 论文 §5.1：Mosaic / Mixup / 多尺度训练
        multi_scale=0.25,
        amp=True,
        workers=4,
        seed=0,
        deterministic=True,
        project="results",
    )
)
