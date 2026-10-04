# visdrone-yolo26n-pose-p2-p16 —— 取证与方法学笔记

> 本变体**不是论文复现**，因此本文件不写"论文说了什么"，而写**口径、推断与坑**：
> 哪些数字是我们算出来的、哪些参数是框架默认、哪些结论还没有证据。
> 引用卡片上的任何数字前，请先读完"§0 数据集状态"。

---

## 0. 数据集状态（**最高优先级的免责说明**）

| 项 | 状态 |
|---|---|
| 目标数据集 | `visdrone2019-pose`（配置见 `configs/_base_/datasets/visdrone2019-pose.yaml`） |
| VisDrone2019-DET 是否有关键点标注 | **没有**。官方发布只有检测框 |
| 当前标注状态 | `tod.kpt_annotation: none`（**接口占位，数据集不存在**） |
| 现阶段能做什么 | 结构自检（`--dry-run`）、合成关键点数据回路自检（`tests/train_pose_smoke.py`） |
| 现阶段**不能**做什么 | 报任何精度数字（合成数据是程序画的火柴人，与真实航拍行人分布无关） |

候选标注来源（按落地难度，均需自行核对许可证）：

1. **FlyPose** — WACV 2026，*Towards Robust Human Pose Estimation From Aerial Views*，
   <https://arxiv.org/abs/2601.05747>。航拍人体姿态的数据/模型工作，问题设定与本变体重叠，
   但**本变体不是它的复现**（既没采用它的网络，也没用它的训练策略）。
2. **UAV-Human** — CVPR 2021，无人机视角人体行为，含 17 点骨骼标注。
3. 自建：在 VisDrone-DET 的 `pedestrian`/`people` 框内标注 COCO 17 点。
   **建议先做 500–1000 框的子集**，用它先量两件事：跨尺度标注成本、以及下节的
   OKS 口径在真实数据上的塌缩幅度；再决定是否全量标。

---

## 1. 本变体真正要解决的问题：OKS 的分母含框面积

COCO 的 OKS 定义（本库 `tod/eval/pose.py::oks`）：

```
OKS = Σ_k exp( −d_k² / (2 · s² · σ_k²) ) · δ(v_k>0)  /  Σ_k δ(v_k>0)
```

- `d_k` 是第 k 个关键点的预测误差（像素）；
- `s²` 是**目标框面积**（像素²）；
- `σ_k` 是每个关键点的容忍度（COCO 17 点从鼻子的 0.026 到髋部的 0.107）。

**关键性质：同样的像素误差，`s²` 越小，OKS 越低。** 这不是"调参能解决"的口径问题，
而是 OKS 的定义本身。本库用 `tests/train_pose_smoke.py` 把它变成了可复核的断言：

```
半像素偏移（0.5 px）造成的 OKS-AP 降幅（合成数据，完全相同的预测器）：
    目标边长  8–16 px 层： −0.7806
    目标边长 32–96 px 层： −0.0574      ← 相差 13.6 倍
```

也就是说：**同一个模型、同一个误差，小目标层的 OKS-AP 会掉到接近 0，而大目标层几乎不动。**
只报一个总体 OKS-AP（框架 `PoseValidator` 的 `metrics.pose` 就是只给总体），
会把"小目标关键点全废"看成"整体还行"。这就是本库必须自带 `tools/val_pose.py` 的原因。

---

## 2. 训练侧的 OKS 项：为什么在 tiny 目标上会"饱和"

框架的 `KeypointLoss`（`ultralytics/utils/loss.py`，本库 `TinyPoseLoss` 与它同签名）：

```
e = d² / ( (2σ)² · area · 2 )        loss = 1 − exp(−e)
```

注意 `area` 是**归一化**面积（`(w/s)·(h/s)`，`s` 为 stride），而 `d` 是**归一化坐标**下的
平方距离。于是

```
e ∝ 1 / (σ² · area)
```

对一个 6×6 px 的小人（imgsz=640 时 `area ≈ 8.8e-5`）、鼻子处 `σ=0.026`：
`2·σ²·area·2 ≈ 2.4e-7`。只要 `d²` 超过 ~1e-6（也就是约 0.02 归一化像素），`e` 就 > 4，
`1−exp(−e)` 已经 ≈ 0.98；再大一点就恒等于 1。

**结论：框架的 OKS 项在 tiny 目标上近似"全或无"的门，梯度信息很弱。**
本库不假装解决它，而是把它暴露成可消融的三个旋钮：

| 旋钮 | 位置 | 本库默认 | 作用 |
|---|---|---|---|
| `sigma_strategy` | `EP7.sigma_strategy` | `person`（COCO 原始 σ） | `person`/`framework` = 与框架逐元素一致；`auto` = 非 17 点用自研几何 σ；`balanced` = 压平 σ（**会改变损失绝对尺度**） |
| `min_sigma` | `EP7.min_sigma` | `1e-3` | σ 下限，防止 `(2σ)²·area` 下溢把该项永久钉在 1 |
| `pose` | `EP7.pose` | `"oks"` | 写成 `False` 即"保留框架原生 KeypointLoss"——**这是本变体的核心消融列** |

**关于 σ 缩放的一个实测教训（已写进代码注释与测试）**：σ 的**绝对尺度**不能乱动。
早期实现把 COCO σ 均值归一到 1，结果同一个 batch 的损失从 `0.369` 掉到 `0.005`（73 倍），
"相对权重调整"变成了"损失曲线整体平移"。现在：

- `person` / `framework` / `auto`（17 点）→ **原始 COCO σ，未缩放**，与框架数值**逐元素一致**
  （`tests/test_pose.py` 有回归对照，容差 1e-9）；
- `auto`（非 17 点）→ 自研 σ，但缩放到 COCO σ 的**均值** `0.0669` 量级，保证与框架可比；
- `balanced` → 明确标注"改变了绝对尺度，不可与 COCO 数字比较"。

---

## 3. 评测口径：本库的 OKS-AP ≠ COCO keypoints AP

| | COCO keypoints | 本库 `tod/eval/pose.py` |
|---|---|---|
| 匹配准则 | **OKS ≥ 阈值**（0.5:0.05:0.95 的 OKS 阈值） | **框 IoU ≥ `match_iou`**（默认 0.5） |
| 每个匹配对算分 | 命中即 1（OKS 已经是匹配准则） | 匹配后按 **OKS 当置信度**做 AP（阈值 0.50:0.05:0.95） |
| 报出的量 | AP / AP50 / AP75 / AP_M / AP_L | OKS-AP / OKS-AP50 / OKS≥0.5 命中率 + 尺度分层 + 逐点诊断 |

**为什么本库换匹配准则**：

1. 框的 AP 与关键点的 AP 可以直接对比 —— 差值就是"关键点带来的额外损失"。
   用 OKS 当匹配准则时，"框对但点偏"会被判成 FP，与框指标不可比；
2. 小目标上 OKS 匹配阈值本身极难标定（见 §1），拿它当匹配准则会让 AP 对 σ 的选择
   高度敏感 —— 而那正是我们要**测量**的量，不该混进匹配里。

**代价（必须写进卡片）**：本表的 OKS-AP **不能**与论文里的 COCO keypoints AP 横向比较，
**只能在库内变体之间比较**。

分层口径与检测侧完全一致（`tod/eval/scales.py`）：按 GT 框边长 `<8 / 8–16 / 16–32 / 32–96 / ≥96`，
命中本层之外 GT 的预测按 COCO 的 **ignore 语义**忽略，
**有 GT 但一个预测都没有的类别计 0 分**（不能悄悄跳过）。

### 逐关键点尺度诊断（本库新增，框架没有）

对"匹配上的目标对"逐点统计：`n_kpt`、`≤1px 命中率`、`平均误差 (px)`、`平均 OKS`，
按目标边长分层。它的用途是回答"关键点定位到底差在哪一层"——
OKS-AP 是**排序指标**，小样本下噪声很大，而平均误差是**度量指标**，两者互补。

---

## 4. 框架默认值 vs 本库改动（逐项）

| 项 | 框架默认（ultralytics 8.4.60） | 本变体 | 备注 |
|---|---|---|---|
| 底座 | `yolo26-pose.yaml`（`end2end=True`、`reg_max=1`） | 同（`base="yolo26n"` + `task="pose"` → 自动找 `yolo26n-pose`） | `kpt_shape: [17,3]` 由底座自带 |
| 关键点分支 | `Pose26`：`cv4` 两层 3×3 + **单层 1×1** `cv4_kpts`/`cv4_sigma` + RealNVP 流 | 同 | 加 P2 后关键点分支自动多一层 |
| `pose` 增益 | 12.0 | 12.0（显式写进 `train`） | 框架参数名就是 `pose` |
| `kobj` 增益 | 1.0 | 1.0（显式） | 关键点可见性损失 |
| `rle` 增益 | 1.0 | 1.0（显式） | Pose26 归一化流的 RLE 损失 |
| `dfl` 增益 | 1.5 | 0.0 | 底座 `reg_max=1`，框架实际不产生 DFL 项（`use_dfl=False` 只在换了框损失时才生效） |
| 框损失 | CIoU | Wise-IoU v3 | 与 SDD-YOLO26n 同源，便于"加关键点前后"的框 AP 对照 |
| 关键点损失 | `KeypointLoss`（COCO σ） | `TinyPoseLoss`（`person`） | 数值上**逐元素一致**；改的是"可配置性"，不是数值 |
| 分配器 | TAL（8.4 已含小目标先验） | STAL 显式开启 | 与 SDD 一致 |
| 优化器 | auto（实测选中 AdamW） | auto | 未改；MuSGD 留给消融 |
| `imgsz` / `batch` | — | 1024 / 4 | 本机 8 GB 显存约束（PLAN §11.1） |
| `epochs` | 100 | 100 | **数据集未标注，等真实数据到位后重定** |

---

## 4.5 本库实测的结构数据（确定性，可复算；**不是精度**）

下面这些数字来自本机实跑（ultralytics 8.4.60 / torch 2.11.0+cu128 / RTX 5060 Laptop 8 GB），
与数据集无关，可以用 `python tools\train.py --variant ... --dry-run --imgsz 640` 复算
（除标注"训练时"的那一行）：

| 项 | 实测值 | 复算方式 |
|---|---|---|
| 参数量（nc=10, 17×3, P2-P5） | **3 856 384（3.86 M）** | `--dry-run --imgsz 640` 的 `参数量=` 行 |
| 检测层数 `nl` | 4 | 同上 |
| `stride` | `[4, 8, 16, 32]`（含 P2） | 同上 |
| 头部输入特征图索引 `f` | `[25, 16, 19, 22]` | 同上 |
| `kpt_shape` / `nk` | `[17, 3]` / 51 | 同上 |
| 关键点分支类型 | `cv4`（两层 3×3）+ `cv4_kpts`/`cv4_sigma`（单层 1×1）+ RealNVP 流 | 同上 |
| 融合后模型（训练时，imgsz=320、nc=2 合成数据） | 3 016 052 参数 / 13.2 GFLOPs | 框架训练日志的 `model summary (fused)` |

**注意 `f=[25,16,19,22]` 与 SDD 的差异**：SDD 用的是普通 `Detect`（args 只有 `[nc]`），
本变体是 `Pose26`（args 是 `[nc, kpt_shape]`），两者在 P2 注入后的全局索引会差 1
（SDD 的 P2 融合块是 24，这里是 25）—— 核对模型图时不要拿 SDD 的索引套本变体。

**精度类数字全部留空**（原因见 §0）；`card.md` 的「相对基线」表在真实标注到位前不应填写。

---

## 5. 已知待验证 / 未定义项（引用前必读）
1. **P2 关键点头的真实增益未知**。合成数据无法回答这个问题（火柴人的关键点分布是程序生成的）。
   需要真实标注才能测，并必须用 `tools/val_pose.py` 的 `lt8`/`8-16` 两层说话。
2. **EP5 换头是否伤 tiny 关键点精度未知**。本变体**不用换头**。
   若要用：`.head("Efficient_UAVDet")`，并用 `EP5.keypoint_branches` 做开关消融
   （Pose26 上这个开关只影响 `cv4` 特征块，`cv4_kpts`/`cv4_sigma` 是单层 1×1、没有 stem 可换）。
3. **Pose26 的归一化流在极小目标上是否稳定未知**。合成数据 1–3 epoch 内 `rle_loss` 一直为 0
   —— 这不是 bug：`fg_mask.sum()==0`（一个正样本都没有）时框架整段跳过关键点损失，
   RLE 也跟着跳过。真实数据上需要专门盯这一列。
4. **`balanced` σ 策略没有任何证据**。它只是"把 σ 压平以削弱末端点权重"的假设，
   且改变了损失绝对尺度。想用必须先做消融，并且**不能用 COCO 口径的数字对比它**。
5. **`kpt_shape` 的双写**：模型图与数据集 YAML 里都写了，框架以**数据集**为准
   （`PoseModel.__init__` 覆盖并打印 INFO）。两处不一致时不要惊讶。
6. **类别与关键点的关系**：`kpt_names` 是按**类别**给的。VisDrone 的 10 类里只有
   `pedestrian`/`people` 真正有关键点，其余类别（car/van/…）在当前配置下也套用了同一份
   17 点名称 —— 这是配置上的将就，真实标注落地时应改成"只给有人体关键点的类别"取名。
7. **漏标与遮挡**：航拍行人遮挡普遍，`v=0`（未标注）与 `v=1`（遮挡但存在）的区分
   直接影响 `kobj` 损失与 OKS 掩码。标注协议（写在数据集配置的 `tod.kpt_protocol` 里）
   一旦改动，历史结果不可合并。

---

## 6. 本变体带来的三条工程结论（可复用到其它任务）

1. **任务类型必须显式传递**。ultralytics 的 `guess_model_task` **先看文件名**；
   本库生成的模型图叫 `model.yaml`，判不出 pose。因此 `Variant` 增加了 `task` 字段，
   `tools/train.py` 用 `YOLO(path, task=spec["task"])` 建模型 —— 不写明会**静默**按 detect
   建图/建加载器（关键点列被当成多余的列丢掉）。
2. **P2 注入必须对姿态头正确**。`Pose26` 的节点 args 是 `[nc, kpt_shape]`（比 `Detect` 多一个），
   早期实现只改节点第 0 项（输入列表）对 Detect 恰好正确，换成 Pose 就会参数错位
   （`Pose(nc, reg_max=17)`）。现在 `_rebuild_head_node` 显式区分两种布局，并有单元测试。
3. **姿态准则要替换"两套"**。YOLO26 的 `end2end` 走 `E2ELoss`，里面持有一对
   `PoseLoss26`（O2M/O2O）。只替换一套会得到"训练能跑、一半损失还是框架默认"的静默错误 ——
   和当初 `bbox_criteria` 面对的是同一个坑，因此 `pose_criteria` 复用了同一套识别逻辑。

---

## 7. 复现命令

```powershell
# 结构自检（不需要数据集）：建图 + 前向 + 准则接线
python tools\make_variant.py variants\visdrone-yolo26n-pose-p2-p16\recipe.py
python tools\train.py --variant variants\visdrone-yolo26n-pose-p2-p16\variant.yaml --dry-run

# 合成关键点数据端到端自检（真训练 + 载权重 + OKS 分层断言）
python tests\train_pose_smoke.py --epochs 1 --device 0

# 模块级测试（损失数值 / σ 策略 / OKS 口径 / 建图 / 换头手术）
python tests\test_pose.py

# 真实数据到位后：
python tools\train.py --variant variants\visdrone-yolo26n-pose-p2-p16\variant.yaml `
    --data configs\_base_\datasets\visdrone2019-pose.yaml --epochs 100 --imgsz 1024 --batch 4
python tools\val_pose.py `
    --variant variants\visdrone-yolo26n-pose-p2-p16\variant.yaml `
    --weights results\visdrone-yolo26n-pose-p2-p16\weights\best.pt `
    --data configs\_base_\datasets\visdrone2019-pose.yaml --imgsz 1024 `
    --sigma-strategy person --json variants\visdrone-yolo26n-pose-p2-p16\results.json
```
