# SPAE-YOLOv8 —— 论文取证笔记

> 用途：本变体的**来源与证据留档**。任何引用本实现的结论都必须能追溯到本文件。
> 取证日期：2026-09-14 ｜ 取证方式：Europe PMC 全文 XML + Figure 4 图像文字

**论文**：Rushang Zhang, Xiaogang Fu, *SPAE-YOLOv8 for Onboard Real-Time Perception:
Lightweight Small UAV Detection from Air-to-Air Perspectives*, Sensors 2026, 26(11):3424.
DOI: [10.3390/s26113424](https://doi.org/10.3390/s26113424) ｜ PMC13258917 ｜ 许可证 CC BY 4.0

**取证来源（可复核）**

| 来源 | 说明 |
|---|---|
| [Europe PMC 全文 XML](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC13258917/fullTextXML) | **唯一拿到完整正文的入口**（228 KB）。PMC HTML 与 MDPI 官网分别被截断与 403 |
| [PMC 文章页](https://pmc.ncbi.nlm.nih.gov/articles/PMC13258917/) | 摘要、图表标题 |
| `evidence/g004.jpg` | Figure 4：轻量检测头结构（**图内文字**是 §3.4 卷积规格的关键证据） |
| `evidence/g001.jpg` | Figure 1：整体架构（P2 支路） |
| `evidence/paper.txt` / `sec34.txt` / `math.txt` | 去标签全文 / §3.4 原文 / 公式文本 |

---

## 1. Efficient_UAVDet 检测头（§3.4，式 10–11，Figure 4，Table 3）

**原文机制**：

> "The standard convolutions located before the regression and classification branches of
> the original detection head are substituted with channel-adaptive grouped convolution."

- 只替换**分类/回归分支之前的两层标准 3×3 卷积**（stem），末端 1×1 输出卷积保持普通卷积；
- 两层连续 3×3 分组卷积：k=3, s=1, p=1，两层规格相同；
- **分组数 g = x / 16**（x = 该分支输入通道数），即**每组固定 16 通道**；
- 分类与回归是**两条并行分支，不共享卷积**（Figure 4 图内文字）；
- **不是 depthwise**（g=x/16 ≠ g=x），**不是深度可分离**（该词全文只出现在 Table 1 对 MobileNet 的综述中）。

**Figure 4 逐层（图内文字）**

| 阶段 | 回归分支 | 分类分支 |
|---|---|---|
| Stem-1 | Conv 3×3, s=1, p=1, GroupConv g=x/16 | 同左 |
| Stem-2 | Conv 3×3, s=1, p=1, GroupConv g=x/16 | 同左 |
| 输出层 | Conv2d 1×1, s=1, p=0 → `4 × reg_max` | Conv2d 1×1, s=1, p=0 → `nc` |

**Table 3（论文给出的分组配置）**

| 尺度 | x | g = x/16 | 每组通道 |
|---|---|---|---|
| P2 | 32 | 2 | 16 |
| P3 | 64 | 4 | 16 |
| P4 | 128 | 8 | 16 |
| P5 | 256 | 16 | 16 |

> ✅ **交叉验证**：当 P2 融合块原始通道取 128（width=0.25 → 32）时，YOLOv8n 检测头
> 输入通道恰为 32/64/128/256，与 Table 3 完全一致 —— 这也是本库取 `p2_channels=128` 的依据。

**⚠️ 代价（论文 §4.5 原文承认）**

> "Therefore, Efficient_UAVDet is mainly designed for model compression and inference
> acceleration rather than direct accuracy improvement. The slight accuracy degradation may
> be attributed to grouped convolutions, which reduce computational cost but also limit
> information interaction among different channel groups."

论文提出未来用 **channel shuffle** 补偿 —— **当前版本不含 channel shuffle**。

---

## 2. 训练超参（§4.2 Table 4）

| 参数 | 值 | 参数 | 值 |
|---|---|---|---|
| optimizer | **SGD** | warmup_momentum | 0.8 |
| epochs | **200** | warmup_epochs | 3 |
| batch | **32** | momentum | **0.937** |
| workers | 4 | weight_decay | **0.0005** |
| imgsz | **640** | lr0 | **0.01** |
| close_mosaic | **10** | lrf | **0.01** |

环境：Windows 10 / i9-13900K / **RTX 4090 24 GB** / PyTorch 1.13.0 / CUDA 11.7。
数据集 Det-Fly：13,271 张，3840×2160，**目标平均占画面 0.12%**，train/val/test = 70/20/10。
部署：ONNX + OpenVINO，Intel NUC11TNHi7，640×640 下 43.9 FPS / 22.8 ms。

**⚠️ 未找到**：是否加载预训练权重（对 "pretrain / pre-train / .pt / ImageNet / from scratch /
transfer" 全文检索**零命中**）。这是复现上的实质空缺，本库默认按框架惯例使用预训练权重，
但必须标注为**与原论文可能不一致**。

---

## 3. 消融（§4.5 Table 6，Det-Fly 数据集）

| SIoU | P2 | ADown | Efficient_UAVDet | mAP@0.5 | mAP@0.5:0.95 | FPS | Params(M) | Size(MB) |
|:-:|:-:|:-:|:-:|---|---|---|---|---|
| – | – | – | – | **0.850** | 0.529 | 197.2 | 3.0 | 5.9 |
| ✓ | – | – | – | 0.851 | 0.517 | 198.0 | 3.0 | 5.9 |
| – | ✓ | – | – | **0.925** | 0.576 | 168.9 | 2.9 | 5.9 |
| – | – | ✓ | – | 0.854 | 0.524 | 187.2 | 2.6 | 5.4 |
| – | – | – | ✓ | 0.846 | 0.521 | 252.9 | 2.4 | 4.8 |
| ✓ | ✓ | – | – | 0.929 | 0.589 | 170.8 | 2.9 | 5.9 |
| – | ✓ | ✓ | – | 0.851 | 0.525 | 185.9 | 2.6 | 5.2 |
| – | ✓ | – | ✓ | 0.850 | 0.519 | 237.3 | 2.0 | 4.0 |
| – | – | ✓ | ✓ | 0.922 | 0.586 | 162.6 | 2.5 | 5.1 |
| ✓ | ✓ | ✓ | – | 0.920 | 0.577 | 217.0 | 2.5 | 5.0 |
| – | ✓ | ✓ | ✓ | 0.851 | 0.523 | 256.1 | 2.4 | 4.8 |
| ✓ | – | ✓ | ✓ | 0.849 | 0.520 | 236.6 | 2.0 | 4.0 |
| ✓ | – | – | ✓ | 0.918 | 0.573 | 218.3 | 2.5 | 5.0 |
| ✓ | ✓ | – | ✓ | 0.929 | 0.593 | 161.5 | 2.5 | 5.1 |
| ✓ | ✓ | ✓ | ✓ | **0.922** | 0.582 | 203.0 | 2.1 | 4.3 |

**单模块相对 baseline(0.850) 的增量**

| 模块 | ΔmAP@0.5 | 备注 |
|---|---|---|
| **P2** | **+7.5 pp** | 精度主贡献者，Recall 0.780 → 0.886 |
| ADown | +0.4 pp | Params 3.0 → 2.6 M |
| SIoU | +0.1 pp | Recall +1.2 pp，但 mAP@0.5:0.95 −1.2 pp |
| Efficient_UAVDet | **−0.4 pp** | Params 3.0 → 2.4 M，FPS 197.2 → 252.9（压缩/加速手段） |

**Table 7 分组数敏感性（每组 16 通道的依据）**

| x/g | g (P2/P3/P4/P5) | mAP@0.5 | mAP@0.5:0.95 | FPS | Params(M) | GFLOPs |
|---|---|---|---|---|---|---|
| 8 | 4/8/16/32 | 0.922 | 0.580 | 201.1 | 1.99 | 6.0 |
| **16** | **2/4/8/16** | **0.922** | **0.582** | **203.0** | 2.06 | 6.4 |
| 32 | 1/2/4/8 | 0.918 | 0.579 | 196.4 | 2.19 | 7.3 |

---

## 4. ⚠️ 论文内部数据冲突（引用时必须说明用的是哪张表）

1. **baseline 不一致**：Table 6 把 YOLOv8n 记为 mAP@0.5 = **0.850**（据此"全模块 0.922"才等于
   摘要宣称的 +7.2 pp），但 Table 5 中 YOLOv8n 记为 **0.922**，Table 7 各配置也都在 0.918–0.922。
2. **FPS 不一致**：同一"全模块"配置，Table 6 记 203.0 FPS，Table 5 记 161.5 FPS；
   Table 6 的 "+P2+ADown" 行 GFLOPs 也与 Table 7 的 x/g=32 行相同。

**→ 本库不引用论文的任何绝对 mAP，只引用其架构与超参**；所有数字在本库基线上重测。

---

## 5. 未找到 / 只能推断的部分（不得当作原文结论）

| 项 | 状态 |
|---|---|
| §3.2 的 "feature calibration" 具体算子 | **完全未定义**（无公式、无图示、未在 Figure/Table 中出现）。本实现按"1×1 卷积降维 + 通道对齐"处理 |
| P2 与上采样特征 Concat 之后接什么模块 | 正文未写。本实现按 YOLOv8 惯例用 `C2f`（**推断**） |
| 检测头 P2–P5 四个尺度间是否共享权重 | 原文未说明。本实现按框架惯例**不共享**（**推断**） |
| 是否使用预训练权重 | **未找到** |
| head stem 各层的确切通道数值 | 原文只写"等于该分支输入通道数"，未给具体数值 |
| Table 3 的 x（32/64/128/256）口径 | 与 neck Concat 后通道的常见取值不符；本实现按"检测头每层实际输入通道"理解，并用 `p2_channels=128` 反向对齐 |
| 官方代码仓库 | 全文无数据/代码可用性声明 |

---

## 6. 本库实现与原论文的差异（复现清单）

| 项 | 论文 | 本库 | 影响 |
|---|---|---|---|
| 数据集 | Det-Fly（空对空） | VisDrone2019-DET（空对地） | **域迁移**，结论不可直接引用 |
| batch | 32 | 8（8GB 显存） | lr0=0.01 未按 batch 重调，非严格公平复现 |
| 通道策略 | 未明说 | `channels="native"`（就地替换） | 另一种读法 `"input"` 也已实现，可切换消融 |
| 预训练 | 未说明 | 默认启用 | 与原文可能不一致 |
