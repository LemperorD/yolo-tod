# SDD-YOLO26n

- **id**: `SDD-YOLO26n`
- **base**: `yolo26n`
- **dataset**: `visdrone2019-det`
- **status**: `planned`
- **tags**: paper-repro, small-object, uav, ground-to-air, nms-free, dfl-free, distillation

## EP 改动

| EP | 名称 | 覆盖 |
|---|---|---|
| EP4 | 注意力 | `attention=DualAttention`; `levels=[2, 3, 4, 5]`; `reduction=16` |
| EP6 | 标签分配 | `assigner=STAL`; `small_target_aware=True` |
| EP7 | 损失 | `box=wiou`; `kind_kwargs={'variant': 3, 'alpha': 1.9, 'delta': 3.0}` |
| EP9 | 训练策略 | `optimizer=MuSGD`; `prefer_native=True`; `prog_loss=framework`; `distill=None`; `kd_lambda=0.5`; `temperature=3.0`; `kd_levels=[2, 3, 4, 5]` |

## 模型图改造（model 段）

| 键 | 值 |
|---|---|
| `nc` | `10` |
| `add_p2` | `True` |
| `p2_idx` | `2` |
| `p2_channels` | `128` |
| `p2_fuse_block` | `C3` |

- **P2 检测头（EP5/EP2）**：`inject_p2_head` 注入 stride=4 分支 —— 上采样(P3) ⊕ backbone 节点 2 → `C3` 融合 → Detect(P2,P3,P4,P5)

## 引用模块来源

| 模块 | 年份 | 论文 / 来源 | 许可证 | 成本提示 |
|---|---|---|---|---|
| `DualAttention` | 2026 | [SDD-YOLO: A Small-Target Detection Framework for Ground-to-Air Anti-UAV Surveillance with Edge-Efficient Deployment (arXiv:2603.25218) §4.5 式 (4)](https://arxiv.org/abs/2603.25218) | 论文为 arXiv 预印本（未声明代码许可证）；本文件为按式 (4) 独立实现 | 额外参数约 2C²/r + 98（r=16 时 C=64 → 约 0.6k），几乎不影响延迟；空间分支需一次 7×7 卷积 |
| `STAL` | 2026 | [SDD-YOLO §4.6 的 STAL（Small-Target-Aware Label Assignment）；论文未给公式，本库按 ultralytics 8.4 的 TAL 小目标先验落地并标注为推断](https://arxiv.org/abs/2603.25218) | 本文件为独立实现；基类 TaskAlignedAssigner 来自 ultralytics（AGPL-3.0） | 与 TAL 同量级；只改中心采样区域，无额外参数与显存 |
| `wiou` | 2023 | [Wise-IoU: Bounding Box Regression Loss with Dynamic Focusing Mechanism (SDD-YOLO §4.3 式 (3) 用它替换 DFL 分支的回归项)](https://arxiv.org/abs/2301.10051) | 官方实现 MIT；本文件为按论文公式独立重写 | 与 IoU 同量级（多一次 exp 与跨 batch 标量均值）；无额外参数、只多 1 个不参与梯度的滑动均值 buffer |
| `MuSGD` | 2026 | [SDD-YOLO §4.6 式 (5)（MuSGD，源自 Moonshot AI 的 Muon, arXiv:2502.16982；YOLO26 将其用于实时检测训练）](https://arxiv.org/abs/2603.25218) | 本文件为独立实现；框架 ultralytics 8.4+ 亦自带同名实现（AGPL-3.0），优先用框架版 | 每步多 5 次 Newton–Schulz 矩阵迭代（只作用于 ndim>=2 的参数）；显存多一份 momentum buffer；1D 参数走普通 SGD 分量 |

## 相对基线

| 指标 | 基线 | 本变体 | Δ |
|---|---|---|---|
| AP50:95 | | | |
| **AP_small** | | | |
| AP_tiny (<16px) | | | |
| Params / FLOPs | | | |
| 延迟 (ms, batch=1) | | | |
| 峰值显存 (GB) | | | |

## 已知坑 / 冲突 / 结论

论文原设定：DroneSOD-30K（未公开，约 30K 张空对地无人机图）、1024×1024 输入、MuSGD + ProgLoss + STAL、DFL-free、NMS-free、YOLO26x 教师蒸馏（λ=0.5, T=3.0）。

本变体差异：① 数据集换成本库主战场 VisDrone2019-DET（域迁移，86.0 mAP@0.5 不可引用）；② batch 因 8 GB 显存降到 4（论文用 32 GB 的 5090，未给 batch）；③ 蒸馏默认关闭（YOLO26x 教师在本机放不下），键位已备好，给本地 .pt 路径即可启用。

论文自相矛盾之处（引用数字前必读）：
  · 消融表 Table 3 的基线 0.8341 与主表 Table 2 的 YOLO26n 原生 0.786 对不上；同一配置 '+P2' 在 Table 3 是 0.8419、Table 2 是 0.849。
  · Table 2/4 里 YOLO26n、'+P2'、'Final' 三行的 Params(2.50M) 与 FLOPs(5.77G) 完全相同 —— 加了 P2 头与注意力不可能一个参数都不变。
  · 数据集三个子集相加 47 750 张 ≠ 摘要'约 30K'；Table 2 的 CPU FPS 列与 Table 4 的同一模型也有出入（30.4 vs 30.4、35.0 vs 35.0 一致，但 YOLOv5n 34.9 与'低于 YOLO26n 的 30.4'叙述矛盾）。
  · 式 (4) 的空间分支公式（[AvgPool;MaxPool]+Conv7×7）与正文'捕捉运动区域'的描述不符；式 (7) 没说 KL 在锚点维求和还是求均值（本库取均值：求和时 KD 会压垮任务损失）。

