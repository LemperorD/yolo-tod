# SDD-YOLO 取证笔记（arXiv:2603.25218）

> 本文件回答三件事：**论文到底说了什么**、**本库据此实现了什么**、**哪些地方论文自相矛盾或根本没写**。
> 结论先行：**SDD-YOLO 的可复现内核 =「P2 头 + 双注意力」两项自研改动 + YOLO26 底座自带的
> 「DFL-free / NMS-free / MuSGD / ProgLoss / STAL」四项能力**；其中后四项在本次实测所用的
> ultralytics 8.4.60 里**已经内建**，本库的贡献是把它们接进可消融配置并逐项验证，
> 而不是重新发明。论文给出的所有精度/速度数字都**不可直接引用**（见 §5、§7）。

---

## 1. 论文元信息与可获取性

| 项 | 内容 |
|---|---|
| 标题 | SDD-YOLO: A Small-Target Detection Framework for Ground-to-Air Anti-UAV Surveillance with Edge-Efficient Deployment |
| 作者 | Pengyu Chen, Haotian Sa, Yiwei Hu, Yuhan Cheng（东南大学吴健雄学院）；Junbo Wang（东南大学信息科学与工程学院，通讯作者） |
| 出处 | arXiv:2603.25218（全文正文取自 ar5iv: `https://ar5iv.labs.arxiv.org/html/2603.25218`） |
| 日期 | 文首标注 **August 10, 2026**；但 arXiv 编号 `2603.*` 对应 **2026 年 3 月** —— 二者矛盾（详见 §5-8） |
| 数据集 | DroneSOD-30K（论文自称构建）。**论文未给出下载地址、许可或构建脚本** |
| 代码/权重 | 论文**未声明**开源仓库或权重发布 |
| 许可证 | arXiv 预印本，未见代码许可证声明 → 本库按"公式独立重写"处理，不搬运任何未公开代码 |
| 致谢 | 东南大学 SRTP 项目（202661033）+ SparkLab 实验室资源 |

---

## 2. 原文关键摘录（按组件）

### 2.1 摘要（总目标）

> "SDD-YOLO introduces a P2 high-resolution detection head operating at 4× downsampling.
> Furthermore, we integrate the recent architectural advancements from YOLO26, including a
> **DFL-free, NMS-free** architecture for streamlined inference, and the **MuSGD hybrid training
> strategy with ProgLoss and STAL**, which substantially mitigates gradient oscillation on sparse
> small-target signals."

> "SDD-YOLO-n achieves a mAP@0.5 of **86.0%** on DroneSOD-30K, surpassing the YOLOv5n baseline by
> **7.8 percentage points**." → 与 Table 2 的 0.782 自洽（0.860 − 0.782 = 0.078）。

> "our model attains **226 FPS on an NVIDIA RTX 5090** and **35 FPS on an Intel Xeon CPU**"

### 2.2 §4.2 P2 高分辨率检测头（本库 EP5 + EP2）

> "Standard YOLO architectures use P3 (8× downsampling) as the highest-resolution feature map.
> For an input of 640×640, a target occupying 8×8 pixels is reduced to a 1×1 feature response at P3."

> "**The P2 feature map is obtained by fusing the shallow backbone output with upsampled P3 features
> via a C3 bottleneck**, preserving high-frequency spatial details."

式 (1)：`f_P2 = s/4, f_P3 = s/8`；"Under our **1024×1024** input resolution"。
→ 8 像素目标在 P2 是 `2×2`、在 P3 塌成 `1×1`。

**本库落地**：`Variant.model(add_p2=True, p2_idx=2, p2_channels=128, p2_fuse_block="C3")`
→ `compose.inject_p2_head()` 生成
`上采样(P3) → Concat(backbone P2) → C3 → Detect(P2,P3,P4,P5)`。
`p2_fuse_block="C3"` 是**照抄原文**（不是 C2f/C3k2）。

### 2.3 §4.3 DFL-free（本库 EP7 + train 的 `dfl=0.0`）

> "DFL models bounding-box regression as a discrete probability distribution: `b̂ = Σ p_i · i`,
> `p = Softmax(z)` (2). The Softmax operator … is known to cause accuracy collapse under INT8
> symmetric quantization on edge NPUs."

> "**By setting dfl=0.0**, we replace the DFL branch with a direct IoU-optimized regression loss:
> `L_box = 1 − WIoU(b̂, b*)` (3), where b* is the ground-truth box and **WIoU denotes Wise-IoU v3**."

**本库落地**：EP7 `box="wiou"`（`WiseIoU(variant=3, alpha=1.9, delta=3.0)`）+ 训练配置 `dfl=0.0`；
`tod/engine/trainer.py` 看到 `dfl` 增益为 0 时同时把回归分支的**计算**省掉（`use_dfl=False`）。

### 2.4 §4.4 NMS-free 端到端（本库 EP8/EP5，由底座提供）

> "SDD-YOLO utilizes a NMS-free dual label assignment mechanism. It employs: One-to-many (O2M)
> assignment during training for rich gradient signal, **One-to-one (O2O) assignment for inference**,
> ensuring each target yields exactly one prediction without NMS. This reduces end-to-end inference
> latency by approximately **20–30%** on resource-constrained SoCs."

**落地方式**：`yolo26.yaml` 自带 `end2end: True`，ultralytics 的 `E2ELoss` 内部就是
`one2many`（tal_topk=10）+ `one2one`（tal_topk=7, topk2=1）两套 `v8DetectionLoss`。
本库做的是：**把两套准则的 IoU 项都替换成我们的损失**（`criterion.bbox_criteria()` 会返回两个子准则，
只 patch 一个会漏掉一半 —— 这是实现期的真实坑）。

### 2.5 §4.5 双注意力（本库 EP4）

> "we embed dual attention modules **between the backbone and detection heads**. Spatial attention
> highlights high-probability motion regions within the wide aerial field of view … Channel attention
> re-weights feature channels to amplify UAV-discriminative frequency responses."

式 (4)：`A = σ(W_c · GAP(F)) ⊗ σ(Conv_{7×7}([AvgPool; MaxPool](F)))`

**本库落地**：`tod/modules/attention/dual_attention.py` 严格按式 (4)；
插入方式见 §4「框架依赖」——**插在检测头每个输入前**（`Detect.f` 重指向新节点），
颈部其余节点一个张量都不变，保证消融归因干净。

### 2.6 §4.6 MuSGD + ProgLoss + STAL（本库 EP9 + EP6）

> "MuSGD applies gradient orthogonalization via **Newton-Schulz iteration** to high-dimensional
> weight matrices in the backbone … `G' = NS(G), W ← W − ηG'` (5) … standard **SGD with momentum is
> retained for one-dimensional tensors** (e.g., biases and normalization layers)."
> 出处：Muon（Moonshot AI, arXiv:2502.16982）+ YOLO26 的"分解式更新"。

> "we utilize YOLO26's **ProgLoss (Progressive Loss)** to dynamically re-weight loss components
> across training epochs. This is coupled with **STAL (Small-Target-Aware Label Assignment), which
> assigns adaptive higher weights to micro-target anchors**."

**落地方式**：
- MuSGD：`tod/optim/musgd.py`（本库独立实现，NS 5 步 + SGD 分量）；实测 ultralytics 8.4.60
  **自带** `ultralytics.optim.MuSGD`，训练器优先用它，二者数值对照见 §8。
- ProgLoss：框架 `E2ELoss.update()`（O2M 权重 0.8 → 0.1 线性衰减，每个 epoch 由训练器调用）——
  **不是我们实现的**，但本库会打印它的来源以保证透明。
- STAL：框架 `TaskAlignedAssigner.select_candidates_in_gts()` 已内置小目标先验
  （`wh_mask = gt_bboxes_xywh[..., 2:] < self.stride[0]` → 放宽到 `stride_val`）。
  本库把它**显式化**为可消融开关（`EP6 assigner=STAL/TAL`），并做了与框架的逐元素对照。

### 2.7 §4.7 特征对齐知识蒸馏（本库 EP9）

> `L_total = (1 − λ)·L_task + λ·L_KD` (6)
> `L_KD = Σ_{l∈{P2,P3,P4,P5}} T² · KL(σ(z_s^l/T) ‖ σ(z_t^l/T))` (7)
> "Following empirical validation, we set the distillation weight **λ = 0.5** and temperature **T = 3.0**."

**落地方式**：`tod/engine/distill.py`（`FeatureAlignKD` + `KDCriterion`），教师默认**关闭**
（论文用 YOLO26x 58.81 M，本机 8 GB 装不下），键位已备好：`EP9 teacher=<本地 .pt>`。

### 2.8 §5 实验设置

> "Models are trained on an NVIDIA RTX 5090 (32GB) using PyTorch … Data augmentation includes
> **Mosaic, Mixup, and multi-scale training**."

**未给**：batch size、epoch 数、优化器超参（除 MuSGD 名字）、是否用预训练权重、随机种子、验证频率。

---

## 3. 组件逐条落地表

| 论文组件 | 论文出处 | 本库实现 | 证据等级 |
|---|---|---|---|
| P2 头（4× 下采样） | §4.2 式 (1) | `compose.inject_p2_head` + `model` 段（**EP5/EP2**） | **原文**（C3 融合块是原文用词） |
| P2 融合用 C3 瓶颈 | §4.2 原文 | `p2_fuse_block="C3"` | **原文** |
| 1024×1024 输入 | §4.2 | `data(imgsz=1024)` / `train(imgsz=1024)` | **原文** |
| DFL-free | §4.3 "setting dfl=0.0" | `train(dfl=0.0)` + `use_dfl=False` | **原文**（做法；本库额外省掉计算） |
| WIoU v3 回归 | §4.3 式 (3) | `tod/loss/box.py::WiseIoU(variant=3)` | **原文指定 v3**；α/δ 取论文默认 1.9/3.0 |
| NMS-free O2M/O2O | §4.4 | 框架 `end2end=True` + `E2ELoss`；两套准则都打补丁 | **原文**（本库负责接线与验证） |
| 双注意力（式 4） | §4.5 式 (4) | `tod/modules/attention/dual_attention.py` + `surgery.attach_head_attention` | **公式原文**；`W_c` 结构/r 为**推断**（取 CBAM 惯例 r=16） |
| 插在 backbone 与 head 之间 | §4.5 | 插在 Detect 每个输入前（颈部零改动） | 原文语义；实现细节为**本库设计** |
| MuSGD | §4.6 式 (5) | `tod/optim/musgd.py`；优先框架原生 | **原文**（NS 步数 5 与系数来自 Muon 原论文） |
| ProgLoss | §4.6 | 框架 `E2ELoss.update()` | **原文**（底座能力，非本库实现） |
| STAL | §4.6 | `tod/assigner/stal.py`（显式化 + 消融开关） | **推断**：论文只有一句话、无公式，本库按框架 TAL 的小目标先验理解 |
| KD（λ=0.5, T=3.0） | §4.7 式 (6)(7) | `tod/engine/distill.py`（默认关闭） | **原文**；锚点维归一化为**推断**（见 §6-3） |
| DroneSOD-30K | §3 | **不用**：换 VisDrone2019-DET | 数据集不可得（论文未给地址） |

---

## 4. 框架依赖（ultralytics 8.4.60 实测）

论文的"YOLO26 创新"在框架里已内建，本库必须先验证底座是否具备，才能谈复现：

| 论文主张 | 框架对应物 | 本库验证方式 |
|---|---|---|
| DFL-free | `yolo26.yaml` 的 `reg_max: 1`；`BboxLoss` 在 `reg_max==1` 时改为 L1 归一化分支 | `tests/smoke.py`：`reg_max == 1`；`tests/.tmp` 探针显示 `loss_items[2] == 0`（`dfl=0.0`） |
| NMS-free | `yolo26.yaml` 的 `end2end: True` → `Detect.end2end`、`E2ELoss` | `tests/smoke.py`：`end2end is True`；`test_modules`：两套子准则都换成 `wiou` |
| MuSGD | `ultralytics.optim.muon.MuSGD`（`muon`/`sgd` 两系数混合） | `test_modules.test_musgd`：与框架原生 5 步后参数最大偏差 < 1e-3 |
| ProgLoss | `E2ELoss.update()`（O2M 0.8→0.1） | 训练器打印其来源；`KDCriterion` 转发 `update()` |
| STAL | `TaskAlignedAssigner.select_candidates_in_gts` 的 `stride_val` 放宽 | `test_modules.test_stal`：与框架逐元素一致 |

**由此产生的一个实现决定**：EP4 注意力用"**在 Detect 前插入节点**"而不是"包在既有节点外面"。
理由：颈部的 P3/P4 输出**同时**被检测头和自底向上路径消费（yolo26 节点 16 既进 Detect 又进下一级
Conv），包在外面会顺带改动颈部聚合 —— 那既不是论文说的"between the backbone and detection heads"，
也会让 `v.without("DualAttention")` 的消融不再干净。插入后 `Detect.f` 从 `[25,16,19,22]`
变为 `[26,27,28,29]`，`model.save` 同步维护（否则 `_predict_once` 取不到张量）。

---

## 5. 论文内部矛盾与可疑之处（引用数字前必读）

1. **同一配置三组不同数字**：消融 Table 3 的基线记 `0.8341`，主表 Table 2 的原生 YOLO26n 记
   `0.786`；同一"只加 P2 头"配置在 Table 3 是 `0.8419`、在 Table 2 是 `0.849`。
2. **加了模块却"零成本"**：Table 2/4 中 `YOLO26n (Native)`、`SDD-YOLO-n (+ P2 Head)`、
   `SDD-YOLO-n (Final)` 三行的 **Params(2.50 M) 与 FLOPs(5.77 G) 完全相同**。
   本库实测（ultralytics `get_flops`，imgsz=1024，nc=10）：

   | 配置 | Params | GFLOPs |
   |---|---|---|
   | yolo26n（nc=10） | 2,507,700 | 14.83 |
   | + P2 头 | 2,486,800（**−20,900**） | 18.43（**+3.60**） |
   | + P2 头 + DualAttention | 2,498,586（+11,786） | 18.47（+0.04） |

   → 参数**不可能不变**：P2 头会改变 `Detect` 的 `ch[0]`（64 → 32），而分类分支宽度取
   `c3 = max(ch[0], min(nc,100))`，于是**分支变窄**（净参数反而减少）；FLOPs 则明确 +24%。
3. **FLOPs 口径与输入分辨率不符**：论文正文声明 1024×1024，但 5.77 GFLOPs 与
   `yolo26.yaml` 自带的 **640** 输入统计（6.1 GFLOPs）同量级；本库在 1024 下测得 14.83 GFLOPs（基线）。
4. **数据集规模自相矛盾**：摘要"approximately **30,000** annotated images"，而 §3.2 给的三个子集
   是训练 30 655 + 验证 14 010 + 测试 3 085 = **47 750** 张。
5. **同一模型两个 CPU FPS**：Table 2 的 `YOLO26n (Native)` 记 34.7 FPS，Table 4 的同一模型记 30.4；
   正文还用"35.0 FPS 略高于 YOLOv5n 的 34.9"论证效率，而 Table 2 里 YOLOv5n 是 34.9、YOLO26n 是 34.7
   （即按 Table 2 的列，YOLO26n 本来就慢于 YOLOv5n，"略高"的叙述依赖 Table 4 的数字）。
6. **教师表内部重复**：`YOLO26l-v2` 与 `YOLO26x-final` 的 Params/FLOPs 完全相同（58.81 M / 208.51 G）；
   两个不同模型共享同一个 GPU FPS（117.4）。
7. **式 (4) 与正文描述不符**：正文说空间分支负责"motion regions"（运动区域），
   但式 (4) 的空间分支是 `[AvgPool;MaxPool] + Conv7×7` —— 这是 CBAM 式的通道池化对比，
   与运动/帧差无关（单帧推理本来也没有运动信息）。本库**按公式**实现。
8. **日期与编号矛盾**：文首 "August 10, 2026"，arXiv 编号却是 `2603.*`（2026 年 3 月）。
9. **术语/笔误**：引言出现 "**INT9** quantization"（应为 INT8）；"domestic NPU platforms" 等表述
   缺少可核查的平台与工具链版本；"spot-level targets (fewer than 20 pixels)" 与 §4.2 的
   "sub-16-pixel" 口径不统一。
10. **消融不完整**：Table 3 只有 3 行，`¬DFL / NMS-free / MuSGD / STAL` 四列被**捆绑成一列**
    （第 3 行四个 ✓ 同时出现），因此**无法读出四项各自的贡献**；而且缺"只有 (¬DFL,NMS-free,MuSGD,STAL)
    而没有 P2"的行。本库的 `v.without(...)` 正是为了补上这种可归因性。

---

## 6. 论文未标注项（本库的推断清单）

1. **`W_c` 的结构与压缩比 r**：式 (4) 只写了 `W_c`，未给层数/是否有 ReLU/r 取值。
   本库取 CBAM 惯例：两层 1×1 卷积瓶颈 + ReLU，`r=16`（可配置）。
2. **STAL 的具体形式**：全文只有一句话（"assigns adaptive higher weights to micro-target anchors"），
   无公式、无超参、也无"higher weights 加在哪里"的说明。本库按框架 TAL 的小目标先验
   （GT 宽/高 < 最小 stride 时放宽中心采样区域到次小 stride）实现，**标注为推断**。
3. **式 (7) 的锚点维归一化**：`Σ_l T²·KL(...)` 没说 KL 在锚点维是求和还是求均值。
   本库默认**按锚点求均值**：实测若按锚点求和，KD 项 = **2017.8** 而 `L_task` 仅 **22.8**，
   λ=0.5 会直接把任务损失压垮；取均值后 KD ≈ **0.44**（约 `L_task` 的 2%），量级合理。
   两种读法都保留（`anchor_reduction="mean"|"sum"`）以便复现论文口径。
4. **`σ` 到底是 softmax 还是 sigmoid**：式 (7) 注明 σ 是 softmax，但 YOLO 检测头是
   **sigmoid 多标签**分类。本库按公式对类别维做 softmax（会让类别互相耦合），
   并把这一点记为论文的处理不明之处。
5. **蒸馏用哪条分支的 logits**：论文只说 "logits from the student and teacher at feature level l"。
   本库取 **O2M（one2many）训练分支**的逐层分类 logits —— 与 §4.4 "O2M 用于训练"一致。
6. **ProgLoss 的具体调度**：论文只说"across training epochs"，无曲线。本库直接用框架
   `E2ELoss.update()` 的 0.8→0.1 线性衰减（视为 YOLO26 官方实现）。
7. **是否使用预训练权重**：全文未提；本库默认框架行为（`pretrained=True`），并在卡片里标注。
8. **epoch / batch / 种子 / 验证协议**：均未给。本库取 epochs=100（框架默认）、batch=4（8 GB 显存）。
9. **`target of s pixels` 的 s 定义**：式 (1) 的 s 未定义是边长还是面积（按上下文应为边长）。

---

## 7. 与论文的刻意差异（复现时必须声明）

| 项 | 论文 | 本库 | 影响 |
|---|---|---|---|
| 数据集 | DroneSOD-30K（未公开，G2A 无人机） | VisDrone2019-DET（G2A 航拍，10 类） | **域迁移**：86.0 mAP@0.5 / 0.480 mAP@.5:.95 不可引用 |
| batch | 未给（RTX 5090 32 GB） | 4（RTX 5060 Laptop 8 GB） | 小 batch 下 lr 需重调，非公平复现 |
| 教师 | YOLO26x（58.81 M） | 默认**不启用**（本机放不下） | 蒸馏增益本库暂不可验证 |
| 底座版本 | YOLO26（论文引用的 Zenodo/GitHub 版本未锁定） | ultralytics **8.4.60** | 版本差异会改变 STAL/ProgLoss/MuSGD 细节 |
| 评测 | COCO 式 mAP，未分尺度上报 | 强制同时上报 `AP_small`（PLAN §7） | 论文的 mAP 无法与"小目标收益"直接对话 |

---

## 8. 本库实测数据（可复现）

| 指标 | 值 | 复现方式 |
|---|---|---|
| Params（SDD 骨架，nc=10） | 2,498,586（2.50 M） | `python tools/train.py --variant variants/SDD-YOLO26n/variant.yaml --dry-run` |
| 检测层数 / strides | `nl=4`，`[4, 8, 16, 32]` | 同上 |
| Detect 输入索引（含注意力插入） | `[26, 27, 28, 29]` | 同上 |
| GFLOPs（imgsz=1024, nc=10） | 基线 14.83 → +P2 18.43 → +注意力 18.47 | `tests/.tmp/probe_flops.py`（临时脚本） |
| STAL 对 4×4 目标的收益 | 经典 TAL 正样本 **0** 个 → STAL **4** 个 | `tests/test_modules.py::test_stal` |
| MuSGD vs 框架原生 | 5 步后参数最大偏差 **1.5e-4**（框架 NS 用 bfloat16） | `test_modules.py::test_musgd` |
| KD 项量级（λ=0.5, T=3.0, 均值口径） | KD ≈ 0.44 vs `L_task` ≈ 22.8 | `test_modules.py::test_kd` |
| 式 (6) 一致性 | criterion 11.617682 vs `0.5·L_task+0.5·L_KD` 11.617683 | 同上 |
| 双注意力开销 | +11,786 参数、+0.04 GFLOPs | §5-2 表 |

> 精度类数字（mAP / AP_small / 延迟）**尚未产生**：本机训练环境刚刚可用，
> 首次训练与消融属于 M1 任务（见 PLAN.md §12）。

---

## 9. 引用注意事项（写作/汇报时必须带上）

1. 引用 **86.0 mAP@0.5 / 226 FPS / 35 FPS** 时，必须同时说明：数据集是未公开的 DroneSOD-30K、
   底座是 YOLO26、硬件是 5090/Xeon 8470Q；且该论文的消融表与主表数字互相冲突（§5-1）。
2. 不要引用 "Params/FLOPs 不变" 这一结论 —— 已被本库实测证伪（§5-2、§5-3）。
3. 本库只复现**结构与方法**，不复现论文的精度数字；任何"复现成功/失败"的判断都必须用
   本库自己的基线（VisDrone + 统一协议）给出。
4. STAL、式 (7) 的归一化、`W_c` 结构都是**推断**，不要把它们当作论文的结论引用。

---

## 10. 复现步骤

```powershell
# 1) 物化变体（生成 variant.yaml / model.yaml / card.md）
python tools\make_variant.py variants\SDD-YOLO26n\recipe.py

# 2) 结构自检：建图 + 前向 + EP4/EP6/EP7/EP9 生效情况（不需要数据集）
python tools\train.py --variant variants\SDD-YOLO26n\variant.yaml --dry-run

# 3) 真训练（需先准备 VisDrone2019-DET）
python tools\train.py --variant variants\SDD-YOLO26n\variant.yaml `
    --data configs\_base_\datasets\visdrone2019-det.yaml

# 4) 单模块与数值验证
python tests\smoke.py            # 无 torch 依赖的架构检查
python tests\test_modules.py     # 形状 / 数值 / 手术 / 蒸馏 / 优化器
```

**消融怎么开**（对应论文 Table 3 的列）：

```python
v.without("wiou")             # 回到框架默认 IoU 项 → 论文的 ¬DFL 列近似
v.assigner("TAL")             # 关掉 STAL 的小目标放宽
v.strategy(optimizer="SGD")   # 关掉 MuSGD
v.without("DualAttention")    # 去掉双注意力
v.without("C3")               # 换掉 P2 的融合块
```
