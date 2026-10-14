# spae-yolov8n-pose

- **id**: `spae-yolov8n-pose`
- **base**: `yolov8n`
- **task**: `pose`（非检测任务：数据加载/准则/评测都与检测不同）
- **dataset**: `rflysim-pose`
- **status**: `planned`
- **tags**: self-designed, keypoint, pose, small-object, uav, air-to-air, spae, p2-head, adown, rflysim, sim2real

## EP 改动

| EP | 名称 | 覆盖 |
|---|---|---|
| EP1 | Backbone | `downsample=ADown`; `downsample_indices=[1, 3, 5, 7]` |
| EP5 | Head | `head=Efficient_UAVDet`; `per_group=16`; `channels=native`; `levels=[2, 3, 4, 5]`; `keypoint_branches=True`; `keypoint_per_group=None` |
| EP7 | 损失 | `box=siou`; `theta=4.0`; `pose=oks`; `sigma_strategy=uniform`; `min_sigma=0.001`; `reproj=True`; `reproj_weight=1.0` |

## 模型图改造（model 段）

| 键 | 值 |
|---|---|
| `nc` | `1` |
| `kpt_shape` | `(9, 3)` |
| `add_p2` | `True` |
| `p2_idx` | `2` |
| `p2_channels` | `128` |
| `p2_pre` | `['Conv', [128, 1, 1]]` |
| `p2_fuse_block` | `C2f` |

- **P2 检测头（EP5/EP2）**：`inject_p2_head` 注入 stride=4 分支 —— 上采样(P3) ⊕ backbone 节点 2 → `C2f` 融合 → Pose/Detect(P2,P3,P4,P5)
- **姿态分支跟着走**：注入 P2 后关键点分支（``cv4`` / Pose26 的 ``cv4_kpts``+``cv4_sigma``）**自动多一层** —— 关键点分辨率随之翻倍，这是小目标关键点最直接的收益来源，不需要额外手术。
- **关键点形状**：`kpt_shape=[9, 3]`（数据集 YAML 里的同名键优先级更高）

## 引用模块来源

| 模块 | 年份 | 论文 / 来源 | 许可证 | 成本提示 |
|---|---|---|---|---|
| `ADown` | 2024 | [YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information (ADown)；SPAE-YOLOv8 §3.3 将其用于小目标检测](https://arxiv.org/abs/2402.13616) | 原实现 GPL-3.0；本文件为按论文描述独立重写 | 相同输出通道下 FLOPs 低于 stride-2 3x3 卷积；参数量略增（多一个 1x1 分支） |
| `_best_groups` | 2026 | [Efficient_UAVDet（SPAE-YOLOv8 §3.4，式 10–11 / Figure 4 / Table 3）](https://doi.org/10.3390/s26113424) | 论文 CC BY 4.0；本文件为按论文描述独立实现 | 检测头参数量约减半；论文报告整机 3.0M→2.4M（−20%）、FPS 197.2→252.9；代价是 mAP@0.5 −0.4pp（论文明确承认是压缩/加速手段而非涨点手段） |
| `siou` | 2022 | [SIoU Loss: More Powerful Learning for Bounding Box Regression](https://arxiv.org/abs/2205.12740) | MIT（参考实现）；本文件为按论文公式独立重写 | 计算量与 CIoU 同量级；无额外参数 |

## 相对基线

| 指标 | 基线 | 本变体 | Δ |
|---|---|---|---|
| AP50:95 | | | |
| **AP_small** | | | |
| AP_tiny (<16px) | | | |
| **OKS-AP**（本库口径，框 IoU 匹配） | | | |
| OKS-AP50 | | | |
| **OKS-AP_small / OKS-AP_tiny** | | | |
| 关键点平均误差 (px, ≤1px 命中率) | | | |
| Params / FLOPs | | | |
| 延迟 (ms, batch=1) | | | |
| 峰值显存 (GB) | | | |

## 已知坑 / 冲突 / 结论

= SPAE-YOLOv8n（SIoU + P2 浅层 + ADown + Efficient_UAVDet）+ 论文口径的 9 点机体关键点。

按 Keypoint-Guided（T-RO 2024，预印本存于 docs/）落地三件事：① 点结构 = 1 个质心点 p0 + 8 个角点 p1~p8（§III-A-1 原文）；② 3D 几何损失 L_rep（式 6/8）——只罚垂直于相机射线的横向偏差，论文消融里增益最大（10°10cm 提升 >10%）；③ 训练策略：前 15 epoch 只用 2D 项，之后加入 L_rep（§IV-A-2）。

本库对论文唯一一处**明确修正**：式 (8) 的文字把 v_i 说成「预测的关键点」，照字面取 v=预测像素坐标会得到与预测无关、梯度恒 0 的量；本库按几何唯一说得通的读法取「穿过预测点的相机射线方向」，详见 paper-notes §3.2。

⚠️ 这不是论文复现：论文用 CPKL-Net（EfficientRep + Rep-PAN + MFMF + 网格三分支），本库用 YOLOv8 Pose 头承载关键点；2D 损失用 OKS 而非论文的像素 MSE。不能宣称复现其精度。
⚠️ 数据集是 RflySim **仿真**（论文是真实 MAV6D），域差异显著。
⚠️ p1~p8 的逐一编号语义在 Fig. 2(a)（图片）里，按「前后×左右×上下」推定，状态见 docs/KEYPOINTS.md；校正只改数据集 YAML 两行。
⚠️ tod.object_points 与 tod.camera_matrix 必须换成你机型的真实几何（论文没给这两张表，配置里是占位值）。

关键点分支的压缩策略：**不压缩**（论文的 g=x/16 只针对检测头，且 c4=max(ch[0]//4, nk)=27 与输入 32 的公约数为 1，强行分组只会静默退化成 g=1）。

消融列（都只需改配置）：① EP7.reproj=False / reproj_weight=0（复现论文 baseline 列）；② tod.reproj_warmup_epochs=0（关 warm-up）；③ EP7.pose=False（退回框架 KeypointLoss）；④ EP7.sigma_strategy=auto/balanced；⑤ EP5.keypoint_per_group；⑥ v.without('add_p2') 看 P2 对框与关键点各自的代价。
