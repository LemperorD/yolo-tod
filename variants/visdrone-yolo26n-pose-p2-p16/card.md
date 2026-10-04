# visdrone-yolo26n-pose-p2-p16

- **id**: `visdrone-yolo26n-pose-p2-p16`
- **base**: `yolo26n`
- **task**: `pose`（非检测任务：数据加载/准则/评测都与检测不同）
- **dataset**: `visdrone2019-pose`
- **status**: `planned`
- **tags**: self-designed, keypoint, pose, small-object, uav, p2-head, oks, tiny-keypoint, dataset-pending

## EP 改动

| EP | 名称 | 覆盖 |
|---|---|---|
| EP6 | 标签分配 | `assigner=STAL`; `small_target_aware=True` |
| EP7 | 损失 | `box=wiou`; `kind_kwargs={'variant': 3, 'alpha': 1.9, 'delta': 3.0}`; `pose=oks`; `sigma_strategy=person`; `min_sigma=0.001` |
| EP9 | 训练策略 | `prog_loss=framework`; `distill=None` |

## 模型图改造（model 段）

| 键 | 值 |
|---|---|
| `nc` | `10` |
| `kpt_shape` | `(17, 3)` |
| `add_p2` | `True` |
| `p2_idx` | `2` |
| `p2_channels` | `128` |
| `p2_fuse_block` | `C3` |

- **P2 检测头（EP5/EP2）**：`inject_p2_head` 注入 stride=4 分支 —— 上采样(P3) ⊕ backbone 节点 2 → `C3` 融合 → Pose/Detect(P2,P3,P4,P5)
- **姿态分支跟着走**：注入 P2 后关键点分支（``cv4`` / Pose26 的 ``cv4_kpts``+``cv4_sigma``）**自动多一层** —— 关键点分辨率随之翻倍，这是小目标关键点最直接的收益来源，不需要额外手术。
- **关键点形状**：`kpt_shape=[17, 3]`（数据集 YAML 里的同名键优先级更高）

## 引用模块来源

| 模块 | 年份 | 论文 / 来源 | 许可证 | 成本提示 |
|---|---|---|---|---|
| `wiou` | 2023 | [Wise-IoU: Bounding Box Regression Loss with Dynamic Focusing Mechanism (SDD-YOLO §4.3 式 (3) 用它替换 DFL 分支的回归项)](https://arxiv.org/abs/2301.10051) | 官方实现 MIT；本文件为按论文公式独立重写 | 与 IoU 同量级（多一次 exp 与跨 batch 标量均值）；无额外参数、只多 1 个不参与梯度的滑动均值 buffer |
| `STAL` | 2026 | [SDD-YOLO §4.6 的 STAL（Small-Target-Aware Label Assignment）；论文未给公式，本库按 ultralytics 8.4 的 TAL 小目标先验落地并标注为推断](https://arxiv.org/abs/2603.25218) | 本文件为独立实现；基类 TaskAlignedAssigner 来自 ultralytics（AGPL-3.0） | 与 TAL 同量级；只改中心采样区域，无额外参数与显存 |

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

本库自研变体（非论文复现）：把航拍小目标检测的能力延伸到**关键点**。

解决的核心问题：OKS 的分母含框面积，同一个像素误差在 8 px 行人上的 OKS 远低于 120 px 人体 —— 只报整体 OKS-AP 会把'小目标关键点全废'看成'基本没变'。因此本变体固定配套 tools/val_pose.py：整体 + 按目标边长分层 + **逐关键点尺度诊断**（≤1px 命中率 / 平均误差 / 平均 OKS）。

⚠️ 数据集状态：VisDrone2019-DET 没有关键点标注，目标数据集 visdrone2019-pose 当前是**接口占位**（tod.kpt_annotation: none）。在真实标注到位前，本变体只能做结构自检与合成数据回路自检，**卡片精度栏必须留空**。

与检测变体的可比性：框部分与 SDD-YOLO26n 同源（YOLO26 + P2 + Wise-IoU + STAL），但多了一路关键点输出，任务不同，AP 不可直接横向比较；可比的是框 AP 与'加关键点后框 AP 掉多少'。

与 FlyPose（WACV 2026, arXiv:2601.05747，航拍人体姿态）的关系：问题设定重叠，**非复现**（未采用其网络与训练策略，仅作为数据集/问题设定的参考）。

已知待验证：① P2 关键点头对 lt8 层 OKS-AP 的增益需要真实标注才能测；② 本变体**不用 EP5 换头**（保持原生 Pose26 头）—— 分组卷积压缩对 tiny 关键点精度的影响未知，若要用，把 ``.head("Efficient_UAVDet")`` 打开并用 ``keypoint_branches`` 做开关消融；③ Pose26 的归一化流（RealNVP + RLE 损失）在极小目标上是否稳定未知 —— rle 增益保留框架默认 1.0。
