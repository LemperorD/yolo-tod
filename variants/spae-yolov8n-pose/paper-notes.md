# spae-yolov8n-pose —— 方法学笔记（SPAE + 论文口径的 9 点机体关键点）

> 本变体是**新组合**：`SPAE-YOLOv8n（只出框）` + `9 点机体关键点` + `论文的 3D 几何损失 L_rep`，
> 数据集换成 **RflySim 仿真录制**。本文件写清"哪些来自谁、哪些是论文原文、哪些是本库推断"，
> 避免把两篇不同的工作混成一个精度数字。

---

## 0. 一句话定位与三层来源

```
┌─ 来自 SPAE-YOLOv8（Sensors 2026, DOI 10.3390/s26113424, CC BY 4.0）───────────┐
│  SIoU 框回归损失（EP7）      P2 浅层检测头（EP2/EP5）                          │
│  ADown 自适应下采样（EP1）   Efficient_UAVDet 轻量头（EP5）                    │
└──────────────────────────────────────────────────────────────────────────────┘
┌─ 来自 Keypoint-Guided（IEEE T-RO 40 (2024) 2967-2983）—— 预印本已存 docs/ ──┐
│  ① 关键点结构：**1 个质心点 p0 + 8 个角点 p1~p8**（§III-A-1，Fig. 2(a)）      │
│  ② **3D 几何损失 L_rep**（式 6/8）：横向偏差惩罚，论文消融里增益最大            │
│  ③ **训练策略**：前 15 epoch 只用 2D 的 L_loc，之后再加入 L_rep（§IV-A-2）    │
│  ④ 评测指标：ADD / ADD(0.5d)（式 13）、10°10cm（式 14/15）                     │
│  ⚠️ **没有**采用：CPKL-Net 的网络结构（三分支 + 网格回归）、无监督域适应        │
└──────────────────────────────────────────────────────────────────────────────┘
┌─ 本库自有 ───────────────────────────────────────────────────────────────────┐
│  用 YOLOv8 的 Pose 头承载关键点（不改主干、不引入新网络）+ OKS 损失            │
│  OKS 分层评测 + 逐点诊断（EP8）+ 6D 位姿旁挂数据与校验器（EP0）                │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 0.1 论文原文（已逐条核对，预印本 PDF 在 `docs/`）

| 项 | 原文 | 本库落点 |
|---|---|---|
| 关键点定义 | "nine keypoints ... **one centroid point (numbered p0) and eight corner points (numbered p1 ∼ p8)**" | `docs/KEYPOINTS.md` §2、`kpt_names` |
| 质心的作用 | "the center point remains visible as it cannot be self-occluded"、"the relative geometry of the corner keypoints remains invariant with respect to the centroid point" | 质心编号 0、在镜像轴上自配对（`flip_idx[0]=0`） |
| 2D 损失 | 式 (7)：`L_loc = Σ ½‖x^pr − x^gt‖²`（像素 MSE） | 由 EP7 的 OKS 项承担同类角色（**不是论文的 MSE**，见 §3） |
| 3D 损失 | 式 (8)：`L_rep = Σ smoothL1((I − v̂v̂ᵀ)p^c)` | **`src/tod/loss/reproj.py`**（新增） |
| 训练策略 | "we only use the 2-D regression loss L_loc ... in the **first 15 training epochs**" | `tod.reproj_warmup_epochs: 15` |
| 权重 | `λ_reg = 1`；`λ_conf = 5`（格内有目标）/ 1（否则）；单机型时 `λ_cls = 0` | `EP7.reproj_weight = 1.0` |
| 指标 | ADD(0.5d)（式 13，阈值 = 0.5×机型直径）、10°10cm（式 14/15） | `tod/eval/pose.py` 的 `add_metric` / `add_0.5d_ratio` |
| 实现规模 | 输入 640×480、40×30 网格、卷积核 1×1、EfficientRep + Rep-PAN + MFMF、C/C′/C″=448/128/128 | **未采用**（那是 CPKL-Net 的结构） |
| 消融结论 | 加 `L_rep` 后 **10°10cm 提升 > 10%**；KDA、MFMF 各提升 ADD(0.5d) > 7% | 本库可复现的只有 `L_rep` 这一项（KDA/MFMF 未实现，见 §5） |

---

## 1. ⚠️ 三条不能含糊的边界

### 1.1 不能宣称"复现了 Keypoint-Guided 论文"

| 维度 | Keypoint-Guided（T-RO 2024） | 本变体 |
|---|---|---|
| 关键点定位网络 | **CPKL-Net**：1×1 卷积 + EfficientRep 主干 + Rep-PAN + MFMF，网格三分支（回归/分类/置信度），质心引导角点 | YOLOv8 **Pose 头**（一组解耦卷积 + `kpts_decode`），**结构不等价** |
| 2D 损失 | 像素 MSE（式 7） | OKS（`TinyPoseLoss`）—— 对尺度更鲁棒，但**不是论文的式子** |
| 3D 损失 | 式 (8) | ✅ 已实现（本库对 `v̂` 的读法见 §3.2） |
| 输出 | 6D 位姿（PnP 求出 R, t） | 2D 关键点 + 框（6D 需另接 PnP，**未实现**） |
| 域适应 | 自训练式无监督域适应（论文贡献之三） | **未实现** |
| 数据集 | 自建真实 MAV6D（室内 VICON 采集，57 075 张） | RflySim **仿真** |

**可比的东西只有**：同样的 9 点定义与同样的 `L_rep` 下，"YOLO 姿态头"能到什么水平。
论文数字**不能**作为本变体的基线。

### 1.2 关键点编号：结构与顺序的证据强度不同

* **结构（1 质心 + 8 角点）= 论文明确**（§III-A-1 原文，已引在 §0.1）；
* **p1…p8 的逐一语义 = 读图推断**：论文 Fig. 2(a) 是图片，PDF 文本层给不出编号对应关系。
  本库按"前/后 × 左/右 × 上/下"排列，标注为 `paper-structured`（不是 `paper-verified`）。
  校正只需改数据集 YAML 的 `kpt_names` / `flip_idx` 两行，**不用改代码**。

### 1.3 数据集是仿真，且当前还没到位

`configs/_base_/datasets/rflysim-pose.yaml` 指向 `./data/rflysim-pose`。
在数据到位前，本变体只有结构自检 + 合成 9 点数据回路自检，**没有任何精度数字**。
合成数据的"无人机"是按 `_project`（论文式 12）投影生成的，**几何自洽但不代表真实分布**。

---

## 2. SPAE 四个组件在姿态头上的落点（实测）

用 `python tools/train.py --variant variants/spae-yolov8n-pose/variant.yaml --dry-run --imgsz 640` 可复算：

| SPAE 组件 | 结果 | 判定 |
|---|---|---|
| **P2 浅层** | `Pose(P2,P3,P4,P5)`；`nl=4`、`stride=[4,8,16,32]`、头输入 `f=[25,15,18,21]` | ✅ **关键点分支自动多一层**，关键点分辨率 ×2 |
| **ADown** | 主干索引 1/3/5/7 全部替换 | ✅ 姿态不改变主干 |
| **SIoU** | `bbox_loss -> siou`（`v8PoseLoss` 继承 `v8DetectionLoss`，替换路径完全复用） | ✅ |
| **Efficient_UAVDet** | `cv2/cv3` 分组 `g=[2,4,8,16]`（16 ch/组，与论文 Table 3 一致）；`cv4` 关键点分支**默认不压缩** | ⚠️ 见 §2.1 |

**参数（imgsz=640、nc=1、9 点、P2-P5）**：换头前 2 750 856 → 换头后 **2 250 312**（−500 544）。

### 2.1 关键点分支不能照搬 `g = x/16`（实测发现）

论文的 `g = x/16` 针对**检测头**。关键点分支的中间通道是框架固定的
`c4 = max(ch[0]//4, nk)`：9 点时 `c4 = max(8, 27) = 27`，与输入 32 的**最大公约数是 1**
→ 照搬只能退化成 `g=1`（普通卷积），即"配了压缩、其实一个通道都没压"的**静默失效**。
本库：默认不压（`KeptStem`，日志如实写"未压缩"）；要压必须显式给 `keypoint_per_group`，
且**不可行时直接报错**。

---

## 3. `L_rep`：论文式 (6)/(8) 的落地（本变体最有价值的部分）

### 3.1 论文为什么加这一项（§III-A-3 原文）

> "Since the distance between two keypoints in the image is not able to reflect their
> distance in the 3-D space, it could be adverse to keypoint localization if we only use
> the image distance to define the keypoint regression loss."

即：**像素距离不能反映 3D 距离**。式 (8) 把误差放到 3D 空间，并用 `(I − v̂v̂ᵀ)` 投影到
**垂直于相机射线**的平面 —— 等价于"点到射线的垂距"。
沿视线的**深度误差无法从单目 2D 观测判别**，所以不该被惩罚；只有横向偏差是真错。
论文消融显示它是三个组件里增益最大的（**10°10cm 提升 > 10%**）。

### 3.2 本库对 `v̂` 的一处**明确修正**（必读）

论文把 `v_i` 的文字说明写成 "`v_i` is the predicted keypoint"。照字面取
`v_i = x_i^pr`（预测的 2D 像素坐标）会得到 `‖(I − v̂v̂ᵀ)c‖`（`c` 为相机中心），
这个量与预测**无关**、梯度恒为 0，等于没有这一项。

本库按几何上唯一说得通的读法实现：

```
v̂_i = normalize( K^{-1} · [u_i^pr, v_i^pr, 1]^T )      # 穿过预测点的相机射线方向
L_rep = Σ_i smoothL1( (I − v̂_i v̂_iᵀ) p_i^c ),
        p_i^c = R·p_i^o + t                             # 真值 3D 关键点（相机系）
```

这与论文自己对 `L_rep` 的解释（"reprojection error mapped into 3-D space"）一致。
**这是本库的推断并已标注**；若拿到作者代码发现定义不同，只需改
`tod/loss/reproj.py::camera_ray_dirs()` 一个函数。

### 3.3 几何正确性（有测试钉住）

| 断言 | 实测 |
|---|---|
| 完美预测（=真值投影）→ `L_rep` | **2.7e-14**（≈0） |
| 横向 3 px 偏移 → `L_rep` > 0 | 1.75e-4 |
| 对预测像素可导 | ✅ |
| `smoothL1` 与式 (9) 逐元素一致 | ✅（`[1.5, 0.5, 0.125, 0, …]`） |
| `kpt_indices` 子集同时切预测点与 3D 点 | ✅（早期只切 3D 点 → 维度不匹配） |
| warm-up：前 N epoch 不计入、之后生效 | ✅ |

### 3.4 怎么接进框架（不改框架源码）

框架的 `v8PoseLoss.loss()` 内部有 `fg_mask` / `target_gt_idx` / `anchor_points`，
但**不往外传**。本库给每个姿态子准则的 `get_assigned_targets_and_loss` 打一个
**记录 hook**（首次调用时安装），于是拿到的锚点信息与框架本次损失**严格同源**
（`E2ELoss` 下 O2M/O2O 两套各记一次，各算各的）。warm-up 由框架每 epoch 末的
`criterion.update()` 驱动，**不需要改训练循环**。

### 3.5 需要的输入（必须由你提供）

| 量 | 位置 | 论文是否给了 |
|---|---|---|
| `tod.object_points` | 数据集 YAML：9 个关键点在机体系 Σo 的 3D 坐标 | ❌ **没给**（正文只有式 (12) 的记号 `p_i^o`） |
| `tod.camera_matrix` | 数据集 YAML：3×3 内参 K | ❌ 没给（在作者数据包的转换脚本里） |
| `tod.model_diameter` | 数据集 YAML：机型直径，用于 ADD(0.5d) | ❌ 没给 |
| 逐目标 6D 位姿 | `labels_6d/<split>.csv`（旁挂，格式见 `docs/KEYPOINTS.md` §5.2） | 作者数据集的 `label/` 就是这个 |

`configs/_base_/datasets/rflysim-pose.yaml` 里的值是**占位**，必须换成你 RflySim 机模的真实几何。

---

## 4. 评测口径

| 指标 | 论文 | 本库 | 可比性 |
|---|---|---|---|
| AP（框） | 由关键点组的外接框算 | `tools/val.py` 的尺度分层 AP | ✅ 同口径可自比 |
| **OKS-AP** | 论文**没有**（用 ADD/10°10cm） | `tools/val_pose.py` 的整体 + 分层 OKS-AP + 逐点诊断 | 本库新增；**与 COCO keypoints AP 不可比**（本库用框 IoU 匹配） |
| **ADD / ADD(0.5d)** | 式 (13)，含 PnP 估计位姿 | `tod/eval/pose.py::add_metric`，**用真值深度、不含 PnP** | ⚠️ **不可直接比较**（本库是"关键点层面"的 3D 误差） |
| 10°10cm | 式 (14)/(15)，需 PnP 出 R/t | **未实现**（需要 PnP） | ❌ |

**关于 OKS 的分母含框面积**（沿用上一轮结论）：微小型无人机在画面里很小，
同一个像素误差在小目标上的 OKS 远低于大目标（合成数据实测：0.5 px 偏移使
8–16 px 层 OKS-AP 掉 **0.83**，32–96 px 层只掉 **0.22**）。所以**只看整体 OKS-AP 没有意义**，
必须看 `lt8` / `8-16` 两层与逐点诊断。

---

## 5. 已知待办与未验证项

1. **p1~p8 的编号语义待校正**（§1.2）—— 有论文 PDF 就可以对 Fig. 2(a) 定下来。
2. **3D 关键点坐标、相机内参必须换成你的机型**（§3.5）—— 论文没给，占位值不可用。
3. **PnP 与 6D 位姿输出未实现**：论文的 10°10cm 与"位姿层面 ADD"都依赖它。
   已知 2D-3D 对应关系（`object_points` + 预测关键点）时用 `cv2.solvePnP` 即可，
   属 EP8 的后处理范畴，**登记在此**。
4. **域适应未实现**：论文贡献之一是仿真→真实的无监督自训练域适应。本库只有接口。
5. **KDA / MFMF 未实现**：论文消融里各值 >7% ADD(0.5d)。
   * MFMF（多尺度特征图融合）在本库等价于"P2 注入 + 颈部多尺度"，已部分覆盖；
   * KDA（亮度/对比度/色相/饱和度抖动 + 旋转 ±30° + 平移 ±10 px + 缩放 0.8–1.2）
     大部分可用框架内置增强表达（`hsv_h/hsv_s/hsv_v/degrees/translate/scale`），
     若要严格复现应逐项对齐后做消融。
6. **σ 策略**：本变体默认 `uniform`（角点无"哪个更难"的先验）。
   论文用 MSE 不用 OKS，两者不可比；想对齐论文应把 `EP7.pose=False` 并接 MSE（未实现）。
7. **`nc` 与多机型**：当前 `nc=1`。多机型混训时不同机型点数不同**不能**塞进同一个
   `kpt_shape`（校验器会提醒）；论文用"微调"处理跨机型（Table IV 显示零样本跨机型很差）。

---

## 6. 复现命令

```powershell
# 0) 结构自检（不需要数据）
python tools\make_variant.py variants\spae-yolov8n-pose\recipe.py
python tools\train.py --variant variants\spae-yolov8n-pose\variant.yaml --dry-run

# 1) 数据到位后**先校验**（会抓 flip_idx / 6D 位姿行序 / 相机内参 / 3D 点个数）
python tools\check_pose_dataset.py --data configs\_base_\datasets\rflysim-pose.yaml --split train
python tools\check_pose_dataset.py --data configs\_base_\datasets\rflysim-pose.yaml --split val

# 2) 合成 9 点数据跑通回路（含 L_rep 与 ADD 指标）
python tests\train_pose_smoke.py --layout uav --warmup 0 --epochs 1 --device 0

# 3) 真训练（前 15 epoch 自动只用 2D 项，之后加入 L_rep）
python tools\train.py --variant variants\spae-yolov8n-pose\variant.yaml `
    --data configs\_base_\datasets\rflysim-pose.yaml --epochs 200 --batch 4 --imgsz 640

# 4) 评测（OKS 分层 + 逐点诊断 + 论文的 ADD/ADD(0.5d)）
python tools\val_pose.py --variant variants\spae-yolov8n-pose\variant.yaml `
    --weights results\spae-yolov8n-pose\weights\best.pt `
    --data configs\_base_\datasets\rflysim-pose.yaml --sigma-strategy uniform `
    --json variants\spae-yolov8n-pose\results.json

# 5) 消融（每项只改一个字段）
#    EP7.reproj=False / reproj_weight=0   → 关掉 L_rep（复现论文的 baseline 列）
#    tod.reproj_warmup_epochs=0           → 关掉 warm-up
#    EP7.pose=False                       → 退回框架原生 KeypointLoss
#    EP7.sigma_strategy=auto/balanced     → σ 策略
#    EP5.keypoint_per_group=...           → 关键点分支压缩与否
```
