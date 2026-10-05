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
| EP7 | 损失 | `box=siou`; `theta=4.0`; `pose=oks`; `sigma_strategy=uniform`; `min_sigma=0.001` |

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
| `GroupedStem` | 2026 | [Efficient_UAVDet（SPAE-YOLOv8 §3.4，式 10–11 / Figure 4 / Table 3）](https://doi.org/10.3390/s26113424) | 论文 CC BY 4.0；本文件为按论文描述独立实现 | 检测头参数量约减半；论文报告整机 3.0M→2.4M（−20%）、FPS 197.2→252.9；代价是 mAP@0.5 −0.4pp（论文明确承认是压缩/加速手段而非涨点手段） |
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

= SPAE-YOLOv8n（SIoU + P2 浅层 + ADown + Efficient_UAVDet）+ 9 点机体关键点。

⚠️ 这不是 SPAE 论文的复现（该论文只出框，没有关键点），也不能宣称复现了 Keypoint-Guided（T-RO 2024）的精度：本库用的是 YOLO 姿态头 + OKS 损失，而后者用的是自定义的质心引导关键点定位网络，结构不等价。

⚠️ 9 个关键点的**顺序是推断**（论文付费墙 + 作者仓库只有 README），定义在 docs/KEYPOINTS.md，状态 tod.kpt_definition_status=inferred；校正只需改数据集 YAML 的 kpt_names / flip_idx 两行，无需改代码。

⚠️ 数据集是 RflySim **仿真**录制（域：simulation），与 SPAE 论文的 Det-Fly 真实空对空数据不同域；仿真数字不能直接当真实场景指标 （Keypoint-Guided 论文专门做无监督域适应正是因为这个 gap）。

关键点分支的压缩策略：**不压缩**（保持原生两层 3×3）。原因是论文的 g=x/16 只针对检测头，且关键点分支的中间通道 c4=max(ch[0]//4, nk)=max(8,27)=27 不被 16 的约数整除，强行分组会退化成 g=1 的**静默失效**（本库已把这种情况改成报错）。要压请显式给 EP5.keypoint_per_group（例如 9 → g=3）。

消融列（都只需改配置）：① EP7.pose=False（退回框架原生 KeypointLoss）；② EP7.sigma_strategy=auto/uniform/balanced；③ EP5.keypoint_branches=False；④ 去掉 P2（v.without('add_p2')）看框与关键点各自的代价；⑤ EP7.box=ciou 对照 SIoU 的贡献。
