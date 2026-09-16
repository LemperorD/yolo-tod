# 模块总表（自动生成，请勿手工编辑）

运行 `python tools/catalog.py --write` 重新生成。

## EP1 Backbone（1 项）

| 模块 | 年份 | 论文 / 来源 | 许可证 | 成本提示 | 备注 |
|---|---|---|---|---|---|
| `ADown` | 2024 | [YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information (ADown)；SPAE-YOLOv8 §3.3 将其用于小目标检测](https://arxiv.org/abs/2402.13616) | 原实现 GPL-3.0；本文件为按论文描述独立重写 | 相同输出通道下 FLOPs 低于 stride-2 3x3 卷积；参数量略增（多一个 1x1 分支） | 替换主干中普通 stride-2 下采样，减少小目标在下采样中的细节丢失；输入通道必须为偶数 |

## EP4 注意力（1 项）

| 模块 | 年份 | 论文 / 来源 | 许可证 | 成本提示 | 备注 |
|---|---|---|---|---|---|
| `DualAttention` | 2026 | [SDD-YOLO: A Small-Target Detection Framework for Ground-to-Air Anti-UAV Surveillance with Edge-Efficient Deployment (arXiv:2603.25218) §4.5 式 (4)](https://arxiv.org/abs/2603.25218) | 论文为 arXiv 预印本（未声明代码许可证）；本文件为按式 (4) 独立实现 | 额外参数约 2C²/r + 98（r=16 时 C=64 → 约 0.6k），几乎不影响延迟；空间分支需一次 7×7 卷积 | 通道注意力（GAP→W_c→σ）与空间注意力（[AvgMax]→Conv7×7→σ）逐元素相乘后原位加权；通道数不变，可插入任意 neck→head 连接处 |

## EP5 Head（1 项）

| 模块 | 年份 | 论文 / 来源 | 许可证 | 成本提示 | 备注 |
|---|---|---|---|---|---|
| `GroupedStem` | 2026 | [Efficient_UAVDet（SPAE-YOLOv8 §3.4，式 10–11 / Figure 4 / Table 3）](https://doi.org/10.3390/s26113424) | 论文 CC BY 4.0；本文件为按论文描述独立实现 | 检测头参数量约减半；论文报告整机 3.0M→2.4M（−20%）、FPS 197.2→252.9；代价是 mAP@0.5 −0.4pp（论文明确承认是压缩/加速手段而非涨点手段） | 两层连续 3x3 分组卷积替换分支 stem，g=x//16（每组 16 通道），末端 1x1 保持普通卷积；不含 channel shuffle |

## EP6 标签分配（1 项）

| 模块 | 年份 | 论文 / 来源 | 许可证 | 成本提示 | 备注 |
|---|---|---|---|---|---|
| `STAL` | 2026 | [SDD-YOLO §4.6 的 STAL（Small-Target-Aware Label Assignment）；论文未给公式，本库按 ultralytics 8.4 的 TAL 小目标先验落地并标注为推断](https://arxiv.org/abs/2603.25218) | 本文件为独立实现；基类 TaskAlignedAssigner 来自 ultralytics（AGPL-3.0） | 与 TAL 同量级；只改中心采样区域，无额外参数与显存 | GT 宽/高 < 最小 stride（P3=8px）时把匹配区域放宽到次小 stride（16px），让小目标获得足够正样本；设 small_target_aware=False 即退回经典 TAL（消融） |

## EP7 损失（2 项）

| 模块 | 年份 | 论文 / 来源 | 许可证 | 成本提示 | 备注 |
|---|---|---|---|---|---|
| `siou` | 2022 | [SIoU Loss: More Powerful Learning for Bounding Box Regression](https://arxiv.org/abs/2205.12740) | MIT（参考实现）；本文件为按论文公式独立重写 | 计算量与 CIoU 同量级；无额外参数 | SPAE-YOLOv8 §3.1 用其替换 YOLOv8 默认 CIoU，提升小目标定位精度 |
| `wiou` | 2023 | [Wise-IoU: Bounding Box Regression Loss with Dynamic Focusing Mechanism (SDD-YOLO §4.3 式 (3) 用它替换 DFL 分支的回归项)](https://arxiv.org/abs/2301.10051) | 官方实现 MIT；本文件为按论文公式独立重写 | 与 IoU 同量级（多一次 exp 与跨 batch 标量均值）；无额外参数、只多 1 个不参与梯度的滑动均值 buffer | v3 的非单调聚焦系数 r 会同时压低极易/极难样本的权重；对小目标（IoU 抖动大）比 CIoU 稳 |

## EP9 训练策略（2 项）

| 模块 | 年份 | 论文 / 来源 | 许可证 | 成本提示 | 备注 |
|---|---|---|---|---|---|
| `FeatureAlignKD` | 2026 | [SDD-YOLO §4.7 式 (6)(7)：多尺度特征对齐知识蒸馏（Hinton et al. 2015 的 KL 蒸馏）](https://arxiv.org/abs/2603.25218) | 本文件为独立实现（KL 蒸馏为公开方法） | 每个 batch 多一次 teacher 前向（训练时间约 ×1.3–2.0，取决于教师规模）；推理零成本（教师只在训练时存在） | P2–P5 逐层分类 logits 的 KL；λ=0.5、T=3.0（论文 §4.7 的经验取值） |
| `MuSGD` | 2026 | [SDD-YOLO §4.6 式 (5)（MuSGD，源自 Moonshot AI 的 Muon, arXiv:2502.16982；YOLO26 将其用于实时检测训练）](https://arxiv.org/abs/2603.25218) | 本文件为独立实现；框架 ultralytics 8.4+ 亦自带同名实现（AGPL-3.0），优先用框架版 | 每步多 5 次 Newton–Schulz 矩阵迭代（只作用于 ndim>=2 的参数）；显存多一份 momentum buffer；1D 参数走普通 SGD 分量 | backbone 高维权重用正交化更新、1D 参数（bias/BatchNorm）用 SGD 动量；对小目标稀疏监督下的梯度震荡有抑制作用 |
