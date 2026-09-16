# SPAE-YOLOv8n

- **id**: `SPAE-YOLOv8n`
- **base**: `yolov8n`
- **dataset**: `visdrone2019-det`
- **status**: `planned`
- **tags**: paper-repro, small-object, lightweight, uav

## EP 改动

| EP | 名称 | 覆盖 |
|---|---|---|
| EP1 | Backbone | `downsample=ADown`; `downsample_indices=[1, 3, 5, 7]` |
| EP5 | Head | `head=Efficient_UAVDet`; `per_group=16`; `channels=native`; `levels=[2, 3, 4, 5]` |
| EP7 | 损失 | `box=siou`; `theta=4.0` |

## 模型图改造（model 段）

| 键 | 值 |
|---|---|
| `nc` | `10` |
| `add_p2` | `True` |
| `p2_idx` | `2` |
| `p2_channels` | `128` |
| `p2_pre` | `['Conv', [128, 1, 1]]` |
| `p2_fuse_block` | `C2f` |

- **P2 检测头（EP5/EP2）**：`inject_p2_head` 注入 stride=4 分支 —— 上采样(P3) ⊕ backbone 节点 2 → `C2f` 融合 → Detect(P2,P3,P4,P5)

## 引用模块来源

| 模块 | 年份 | 论文 / 来源 | 许可证 | 成本提示 |
|---|---|---|---|---|
| `ADown` | 2024 | [YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information (ADown)；SPAE-YOLOv8 §3.3 将其用于小目标检测](https://arxiv.org/abs/2402.13616) | 原实现 GPL-3.0；本文件为按论文描述独立重写 | 相同输出通道下 FLOPs 低于 stride-2 3x3 卷积；参数量略增（多一个 1x1 分支） |
| `GroupedStem` | 2026 | [Efficient_UAVDet（SPAE-YOLOv8 §3.4，式 10–11 / Figure 4 / Table 3）](https://doi.org/10.3390/s26113424) | 论文 CC BY 4.0；本文件为按论文描述独立实现 | 检测头参数量约减半；论文报告整机 3.0M→2.4M（−20%）、FPS 197.2→252.9；代价是 mAP@0.5 −0.4pp（论文明确承认是压缩/加速手段而非涨点手段） |
| `siou` | 2022 | [SIoU Loss: More Powerful Learning for Bounding Box Regression](https://arxiv.org/abs/2205.12740) | MIT（参考实现）；本文件为按论文公式独立重写 | 计算量与 CIoU 同量级；无额外参数 |

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

论文原设定：Det-Fly 数据集、batch=32、SGD、200 epochs；本变体迁移到 VisDrone2019-DET 且 batch 降到 8（8GB 显存），属于域迁移 + 超参改动，论文 mAP 数字不可直接引用。

论文消融（对 baseline 0.850）：P2 +7.5pp（主贡献）、ADown +0.4pp、SIoU +0.1pp、Efficient_UAVDet −0.4pp（论文自认是压缩/加速手段）。

引用论文数字时注意其表间冲突：Table 6 baseline 记 0.850，而 Table 5/7 记 0.922；同一配置 FPS 在 Table 6 为 203.0、Table 5 为 161.5。
