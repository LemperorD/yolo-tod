# spae-yolov8n-pose —— 方法学笔记（SPAE + 9 点机体关键点）

> 本变体是**新组合**，不是任何单篇论文的复现。本文件写清"哪些来自谁、哪些是我们的选择、
> 哪些还没有证据"，避免把三篇不同的工作混成一个精度数字。

---

## 0. 一句话定位

`SPAE-YOLOv8n（只出框）` **+** `9 点机体关键点（出框 + 关键点）`，数据集换成 **RflySim 仿真录制**。

```
┌─ 来自 SPAE-YOLOv8（Sensors 2026, DOI 10.3390/s26113424, CC BY 4.0）───────────┐
│  SIoU 框回归损失（EP7）      P2 浅层检测头（EP2/EP5）                          │
│  ADown 自适应下采样（EP1）   Efficient_UAVDet 轻量头（EP5）                    │
└──────────────────────────────────────────────────────────────────────────────┘
┌─ 来自 Keypoint-Guided（IEEE T-RO 2024, DOI 10.1109/TRO.2024.3400938）───────┐
│  任务设定：微小型无人机的 **9 点机体关键点**（点数与语义）                      │
│  ⚠️ 仅此而已：**没有**采用它的网络结构（质心引导定位网络）、损失与域适应方法      │
└──────────────────────────────────────────────────────────────────────────────┘
┌─ 本库自有（新增）────────────────────────────────────────────────────────────┐
│  姿态头（YOLOv8 Pose 系）作为关键点支路 + OKS 损失（EP7 TinyPoseLoss）          │
│  OKS 分层评测（EP8 tod.eval.pose）+ RflySim 数据校验器（tools/check_pose_dataset）│
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. ⚠️ 三条不能含糊的边界

### 1.1 不能宣称"复现了 Keypoint-Guided 论文"

| 维度 | Keypoint-Guided（T-RO 2024） | 本变体 |
|---|---|---|
| 关键点定位 | **自定义的质心引导关键点定位网络**（论文三大贡献之一） | YOLOv8 `Pose` 头 + OKS 损失，**结构不等价** |
| 输出 | 6D 位姿（3D 位置 + 3D 姿态四元数） | 2D 关键点（框 + 9 点像素坐标）；6D 需另接 PnP |
| 域适应 | **自训练式无监督域适应**（仿真→真实，论文贡献之三） | **未实现**（见 §4 待办） |
| 数据集 | 作者自建的真实 MAV 6D 数据集 | RflySim **仿真** |
| 关键点损失 | 论文自有设计 | OKS（COCO 口径的推广） |

**可比的东西只有一个**：同样的 9 点定义下，"YOLO 姿态头"能否达到可用的关键点精度。
论文数字**不能**作为本变体的基线。

### 1.2 9 个关键点的顺序是**推断**

论文在 IEEE 付费墙后，作者仓库 [`WindyLab/MAV6D`](https://github.com/WindyLab/MAV6D) 只有
一个 README（提供 6D 位姿标注 + 相机/机体参数 + 转格式脚本，**没有关键点定义图或代码**），
因此**无法核对**那 9 个点的确切语义与编号。

* 本库的定义在 [`docs/KEYPOINTS.md`](../../docs/KEYPOINTS.md)，状态 `inferred`；
* 数据集配置里 `tod.kpt_definition_status: inferred`，`tools/check_pose_dataset.py` 会**每次都警告**；
* 校正只需改数据集 YAML 的 `kpt_names` 与 `flip_idx` 两行（长度必须等于 `kpt_shape[0]`），
  改完把状态改成 `paper-verified`。**不需要改任何代码。**

### 1.3 数据集是仿真，且当前**还没到位**

`configs/_base_/datasets/rflysim-pose.yaml` 指向 `./data/rflysim-pose`，
那是**你的 RflySim 录制数据要放的位置**。在数据到位前，本变体只有：

* 结构自检（`--dry-run`）：建图 / 前向 / 准则接线 / 换头；
* 合成 9 点数据回路自检（`tests/train_pose_smoke.py --layout uav`）。

**没有任何精度数字**（合成数据的"无人机"是程序画的多边形 + 圆点）。

---

## 2. SPAE 四个组件在姿态头上的落点（逐条核实结果）

用 `python tools/train.py --variant variants/spae-yolov8n-pose/variant.yaml --dry-run --imgsz 640`
可复算下表：

| SPAE 组件 | 在姿态底座上的实测结果 | 判定 |
|---|---|---|
| **P2 浅层** | `inject_p2_head` → `Pose(P2,P3,P4,P5)`；`nl=4`、`stride=[4,8,16,32]`、头输入 `f=[25,15,18,21]` | ✅ **关键点分支自动多一层**（关键点分辨率 ×2），不需额外手术 |
| **ADown** | 主干索引 1/3/5/7 全部替换（`{1:'Conv->ADown', …}`），与检测变体完全一致 | ✅ 姿态不改变主干，替换点不变 |
| **SIoU** | `bbox_loss -> siou`（`v8PoseLoss` 继承 `v8DetectionLoss`，替换路径完全复用） | ✅ 组件的"框"语义未被姿态影响 |
| **Efficient_UAVDet** | `cv2/cv3` 分组 `g=[2,4,8,16]`（16 ch/组，与论文 Table 3 一致）；**`cv4` 关键点分支默认不压缩** | ⚠️ 见下 |

### 2.1 关键点分支**不能**照搬论文的 `g = x/16`（实测发现）

论文规定每组 16 通道 → `g = x/16`。但姿态头关键点分支的中间通道是框架固定的
`c4 = max(ch[0] // 4, nk)`：

| 关键点数 | `nk` | `ch[0]`（P2，width=0.25） | `c4 = max(32//4, nk)` | `g = c4/16` 能整除吗 |
|---|---|---|---|---|
| 17（人体） | 51 | 32 | **51** | ✗（51 = 3×17） |
| 9（机体） | 27 | 32 | **27** | ✗（27 = 3³） |
| 4 | 12 | 32 | 12 | ✗ |
| 5 | 15 | 32 | 15 | ✗ |

`_valid_groups` 在这种情况下只能退化为 **g=1 = 普通卷积** —— 也就是
**"配了压缩、实际一个通道都没压"的静默失效**。本库的处理：

1. **默认不压关键点分支**（用显式的 `KeptStem`，日志里写"未压缩"，不骗人）；
2. 想压必须显式给 `EP5.keypoint_per_group`，且**退化时直接报错**而不是静默放行
   （`_valid_groups(..., allow_degenerate=False)`）；
3. `tests/test_pose.py` 用 9 点 / 17 点两种 nk 钉住上述行为。

**实测参数（imgsz=640、nc=1、9 点、P2-P5）**：

| 阶段 | 参数量 |
|---|---|
| 换头前（原生 Pose 头 + P2） | 2 750 856 |
| 换头后（SPAE 头，关键点分支不压） | **2 250 312**（−500 544） |

---

## 3. OKS 口径在"机体关键点"上的适配（本变体的关键设计选择）

### 3.1 为什么不能用人体 σ

COCO 的 17 个 σ 编码的是**人体关节的定位难度**（鼻子/眼睛最严，髋/腕最松）。
机体关键点（四个电机、机臂中点）**没有这个层级**：四个电机在几何上是对称的，
定位难度主要由视角与遮挡决定，而不是"这个部位天生更难"。

因此本变体默认 **`sigma_strategy="uniform"`**：所有 9 点同一个 σ（= `COCO_SIGMA_MEAN`，对齐量级）。

| 策略 | σ 形状（9 点） | 适用与风险 |
|---|---|---|
| `uniform`（**本变体默认**） | 全 0.0669 | 无"哪个点更难"先验时的**诚实选择**；四个电机同权 |
| `auto` | 0.0058 … 0.1159（几何曲线），均值对齐 | 兜底；**但会把首尾点无依据地放松/收紧** |
| `balanced` | 压平后归一到均值 1 | 改变损失绝对尺度，**不可与 COCO 口径比较** |

三档都是**一个字段**（`EP7.sigma_strategy`）就能切换的消融列。

### 3.2 OKS 在微小目标上的塌缩（从上一轮继承的结论）

OKS 的分母含框面积，微小型无人机在画面里很小（Det-Fly 论文口径：目标平均约占画面 0.12%），
所以**同一个像素误差在小目标上的 OKS 远低于大目标**。上一轮在合成数据上量化为：

```
0.5 px 偏移造成的 OKS-AP 降幅：8–16 px 层 −0.78，32–96 px 层 −0.057（相差 13.6 倍）
```

**结论：本变体的评测必须看 `tools/val_pose.py` 的 `lt8` / `8-16` 两层与"逐关键点尺度诊断"，
只看整体 OKS-AP 没有意义。** 这与检测侧的 `AP_small` 是同一个道理。

---

## 4. 已知待办与未验证项

1. **关键点顺序待论文校正**（§1.2）—— 最高优先级。
2. **域适应未实现**：论文的核心贡献之一是仿真→真实的无监督域适应。
   本库目前只有仿真训练 + 真实评测的接口，**没有**自训练/伪标签/对抗对齐。
   要复现该贡献需要另开模块（EP9 训练策略），不属于本变体范围。
3. **6D 位姿未接**：RflySim 与 MAV6D 都有 6D 标注，理论上可以
   "2D 关键点 + 已知机体 3D 点 → PnP → 6D 位姿"，并用重投影误差当额外监督
   （几何自校验）。**未实现**，登记在此。
4. **σ 的实测标定**：`docs/KEYPOINTS.md §4` 提到的 `joint_adaptive`（用实测误差分布反推 σ）
   未实现；等 RflySim 数据到位后，用 `tools/check_pose_dataset.py` 的统计 +
   训练后的逐点误差来定，比人体先验更合理。
5. **`nc` 与多机型**：当前 `nc=1`。多机型混训时不同机型点数可能不同，
   **不能**塞进同一个 `kpt_shape`，必须拆数据集（`tools/check_pose_dataset.py` 会提醒）。
6. **batch 与 lr 的公平性**：SPAE 论文 `batch=32 / lr0=0.01`；本机 8 GB 只能 `batch=4`。
   小 batch 下沿用原 lr **不是公平复现**（原变体的注释里也写了同一问题）。

---

## 5. 复现命令

```powershell
# 0) 结构自检（不需要数据）
python tools\make_variant.py variants\spae-yolov8n-pose\recipe.py
python tools\train.py --variant variants\spae-yolov8n-pose\variant.yaml --dry-run

# 1) 把你的 RflySim 数据放到 configs 里 path 指向的位置，然后**先校验**
python tools\check_pose_dataset.py --data configs\_base_\datasets\rflysim-pose.yaml --split train
python tools\check_pose_dataset.py --data configs\_base_\datasets\rflysim-pose.yaml --split val

# 2) 合成 9 点数据跑通回路（先确认代码没问题，再上真数据）
python tests\train_pose_smoke.py --layout uav --epochs 1 --device 0

# 3) 真训练
python tools\train.py --variant variants\spae-yolov8n-pose\variant.yaml `
    --data configs\_base_\datasets\rflysim-pose.yaml --epochs 200 --batch 4 --imgsz 640

# 4) 评测（整体 + 分层 OKS-AP + 逐点诊断）
python tools\val_pose.py --variant variants\spae-yolov8n-pose\variant.yaml `
    --weights results\spae-yolov8n-pose\weights\best.pt `
    --data configs\_base_\datasets\rflysim-pose.yaml --sigma-strategy uniform `
    --json variants\spae-yolov8n-pose\results.json

# 5) 消融（每项只改一个字段）
#    EP7.pose=False            → 退回框架原生 KeypointLoss
#    EP7.sigma_strategy=auto   → σ 策略对比
#    EP5.keypoint_branches=False / keypoint_per_group=9 → 关键点分支压缩与否
#    v.without("add_p2")       → P2 对框与关键点各自的贡献
```
