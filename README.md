# yolo-tod —— 小目标检测魔改 YOLO 收集库

收集、复现并**可归因地**对比针对小目标检测（Tiny/Small Object Detection）的 YOLO 魔改。
完整方案见 [`PLAN.md`](PLAN.md)：架构设计、值得收录的魔改清单、优先级、评测协议。

**当前状态：M0 骨架已完成（`tests/smoke.py` 77 项全绿，`tests/test_modules.py` 105 项全绿）；
两个论文级变体已实现 —— SPAE-YOLOv8n 与 SDD-YOLO26n（结构自检全通）；
**姿态式关键点检测（pose）已打通完整回路**（`tests/test_pose.py` 220 项全绿 +
`tests/train_pose_smoke.py` 合成关键点数据真训练 32 项），已有两个姿态变体：
`spae-yolov8n-pose`（**SPAE + 论文口径的 9 点机体关键点 + 3D 几何损失 L_rep**，
面向 RflySim 仿真数据；预印本已存 `docs/`）与
`visdrone-yolo26n-pose-p2-p16`（航拍行人 17 点，**数据集标注待准备**）；
训练回路已用**合成数据**端到端跑通（`tests/train_smoke.py`：训练/验证/EMA/存权重/载权重/推理），
并补齐了**尺度分层评测**（`tools/val.py`：整体 AP + AP_small/AP_tiny）、
**姿态 OKS + ADD 分层评测**（`tools/val_pose.py`）、**关键点数据开训前校验**（`tools/check_pose_dataset.py`）与
**消融流水线**（`tools/ablation.py`：leave-one-out + 逐字段 diff + 汇总表）。
真实数据集上的精度数字待 M1。推理端（C++/TensorRT）工厂化骨架已落地：
`src/todrt/`，CPU 自检 107 项全绿。**

- 主干框架：**ultralytics**（AGPL-3.0；本库实测区间 8.2 → **8.4**，8.4 才自带
  `end2end`/`MuSGD`/TAL 小目标先验，SDD-YOLO 依赖这三项）
- 第一主战场：**VisDrone2019-DET**（航拍小目标）
- 本机：RTX 5060 Laptop **8 GB** ／ Python 3.14.5 → 建议另建 **Python 3.12** 虚拟环境
  （3.14 也能用：torch 2.11 有 cp314 的 cu128 轮子；唯一要避开的是 **3.14.1**，
  因为 torchvision 明确排除了它。见 `requirement.txt` / `tools/setup_venv.ps1`）
- 部署端：**Linux**，四个可选后端 —— TensorRT（NVIDIA / Jetson DLA）、**RKNN（RK3588 NPU）**、
  **ONNX Runtime（通用 CPU，AMD x86 主力）**、**OpenVINO（x86 CPU / Intel iGPU）**；
  见 [`docs/DEPLOY.md`](docs/DEPLOY.md)

## 已实现的变体

### spae-yolov8n-pose —— SPAE **加关键点**（按预印本口径，RflySim 仿真训练）

在 **既有 SPAE-YOLOv8n 上新增一条关键点输出支路**：四个 SPAE 组件全部保留，
检测头换成姿态头，用于"检测框 + 9 点机体关键点"的多任务模型。
预印本（IEEE T-RO 2024，*Keypoint-Guided Efficient Pose Estimation and Domain
Adaptation for Micro Aerial Vehicles*）**已存于 `docs/`**，本变体按它落地了三件事：

| 预印本内容 | 原文依据 | 本库落点 |
|---|---|---|
| **点结构**：1 个质心点 p0 + 8 个角点 p1~p8 | §III-A-1："one centroid point (numbered p0) and eight corner points (numbered p1 ∼ p8)" | `docs/KEYPOINTS.md`、数据集 `kpt_names`/`flip_idx` |
| **3D 几何损失 `L_rep`**（只罚垂直于相机射线的横向偏差） | 式 (6)/(8)；消融显示 **10°10cm 提升 >10%**，是三个组件里增益最大的 | `src/tod/loss/reproj.py`（新增） |
| **训练策略**：前 15 epoch 只用 2D 项，之后加入 `L_rep` | §IV-A-2 原文 | `tod.reproj_warmup_epochs: 15` |
| **指标**：ADD / ADD(0.5d)（阈值 = 0.5×机型直径） | 式 (13) | `tod/eval/pose.py::add_metric`、`tools/val_pose.py` |

| SPAE 组件 | 在姿态头上的落点（实测） |
|---|---|
| **P2 浅层** | `Pose(P2,P3,P4,P5)`，`nl=4`、`stride=[4,8,16,32]` → **关键点分辨率 ×2**，不需额外手术 |
| **ADown** | 主干索引 1/3/5/7 替换，与检测变体完全一致 |
| **SIoU** | `v8PoseLoss` 继承 `v8DetectionLoss`，框损失替换路径完全复用 |
| **Efficient_UAVDet** | `cv2/cv3` 分组 `g=[2,4,8,16]`（16 ch/组）；**`cv4` 关键点分支默认不压缩**（见下） |

```powershell
python tools\make_variant.py variants\spae-yolov8n-pose\recipe.py
python tools\train.py --variant variants\spae-yolov8n-pose\variant.yaml --dry-run
python tools\check_pose_dataset.py --data configs\_base_\datasets\rflysim-pose.yaml   # 开训前校验
python tests\train_pose_smoke.py --layout uav --warmup 0   # 合成数据端到端（含 L_rep 与 ADD）
```

> ⚠️ 四条必须知道的边界：
> ① **不是论文复现**：论文用 **CPKL-Net**（1×1 卷积 + EfficientRep + Rep-PAN + MFMF +
> 网格三分支回归），本库用 **YOLOv8 Pose 头**承载关键点；2D 损失用 OKS 而非论文的像素 MSE。
> **结构不等价，不能宣称复现其精度。**
> ② **p1~p8 的逐一编号语义仍是推断**（论文 Fig. 2(a) 是图片，PDF 文本层给不出）——
> 结构与质心/角点之分是论文明确，编号顺序按"前后×左右×上下"推定，状态
> `paper-structured`；校正只改数据集 YAML 两行。
> ③ **`tod.object_points` 与 `tod.camera_matrix` 必须换成你机型的真实几何**：
> 论文**没有**给出这两张表（正文只有式 (12) 的记号），配置里是占位值。
> ④ 数据集是 **RflySim 仿真**（论文是真实 MAV6D），域差异显著；
> 本库的 **ADD 用真值深度、不含 PnP**，与论文含 PnP 的 ADD **不可直接比较**；
> 域适应与 10°10cm 指标**未实现**。

#### 关键点分支为什么默认"不压缩"

论文的 `g = x/16` 是针对**检测头**给出的。关键点分支的中间通道是框架固定的
`c4 = max(ch[0]//4, nk)`：9 点时 `c4=27`，与输入 32 **最大公约数为 1**，
照搬 `g=x/16` 只会退化成 `g=1`（= 普通卷积）——**"配了压缩、其实一个通道都没压"的静默失效**。
本库因此：默认不压（`KeptStem`，日志里如实写"未压缩"）；要压必须显式给
`EP5.keypoint_per_group`，且**不可行时直接报错**而不是放行。



### visdrone-yolo26n-pose-p2-p16 —— 航拍小目标**姿态关键点**（本库自研）

把"小目标检测"延伸到**关键点**，并把 OKS 口径的坑做成可归因配置：

| 组件 | 本库落点 | 要点 |
|---|---|---|
| **P2 关键点头** | `model(add_p2=True)`（EP5/EP2） | 关键点分支跟着多一层，关键点预测分辨率 ×2 |
| **OKS 关键点损失** | `modules/loss/pose.py`（EP7） | `TinyPoseLoss` 与框架同签名；sigma 策略 `person`/`auto`/`balanced`；σ 下限；可见性开关 |
| **姿态训练器** | `engine/pose_trainer.py`（EP9） | 继承检测侧全部 EP 接线，改用 `PoseModel` + 姿态准则（O2M/O2O **两套**都替换） |
| **OKS 分层评测** | `eval/pose.py`（EP8） | OKS-AP 50:95 + 按目标边长分层 + **逐关键点尺度诊断**（≤1px 命中率/平均误差/平均 OKS） |
| **合成关键点数据** | `tools/make_dummy_dataset.py --task pose`（EP0） | 火柴人 + COCO 17 点，tiny(12–24px) 与日常(48–112px) 双尺度 |

```powershell
python tools\make_variant.py variants\visdrone-yolo26n-pose-p2-p16\recipe.py
python tools\train.py --variant variants\visdrone-yolo26n-pose-p2-p16\variant.yaml --dry-run
python tests\train_pose_smoke.py      # 合成关键点数据真训练（回路自检）
python tools\val_pose.py --weights results\...\best.pt `
    --data configs\_base_\datasets\visdrone2019-pose.yaml --imgsz 1024 --sigma-strategy person
```

> ⚠️ **数据集状态**：VisDrone2019-DET **没有关键点标注**，`configs/_base_/datasets/visdrone2019-pose.yaml`
> 目前是**接口占位**（`tod.kpt_annotation: none`，候选标注来源写在该文件注释里）。
> 在真实标注到位前，本变体只能跑结构自检与合成数据回路自检，**卡片精度栏必须留空**。
>
> ⚠️ 三条必须知道的结论：① **OKS 的分母含框面积** —— 同一个 3 px 误差在 120 px 人体上 OKS≈0.95、
> 在 8 px 行人上 OKS≈0.2，因此**只看整体 OKS-AP 会掩盖小目标关键点退化**，这也是本库必须自带
> `val_pose.py` 分层口径的原因；② 框架 OKS 项在 tiny 目标上会**饱和**（`1−exp(−e)`，e 随 1/area
> 爆炸），本库的 σ 下限与 `balanced` 策略就是为这个现象准备的可消融开关；
> ③ 本变体**不是** FlyPose（WACV 2026）的复现，只是问题设定重叠。

### MAV6D —— 真实数据的小目标检测 + 9 点关键点（训练/测试脚本）

把**作者原始 MAV6D 数据集**（`D:/MAV6D`，IEEE T-RO 2024 的公开数据）直接接进
上面的 `spae-yolov8n-pose` 变体。三件套：

| 脚本 | 作用 |
|---|---|
| `tools/mav6d_calibrate.py` | 从数据反解**相机内参与机体 3D 关键点**；`--check-util` 用作者公开标定核对投影链（实测 **0.0004 px / 300 帧**逐值吻合） |
| `tools/mav6d_to_yolo.py` | 转成 YOLO-pose（检测 + 9 点）+ `labels_6d` 6D 位姿 + `dataset.yaml`（含全部 `tod.*`） |
| `tools/train_mav6d.py` / `tools/test_mav6d.py` | 训练与测试（测试报**论文全部指标**：ADD / ADD(0.5d) / 10°10cm + OKS 分层 + 框 AP 分层） |

```powershell
python tools\mav6d_calibrate.py --mav6d D:\MAV6D --model phantom4 --check-util   # 核对投影链
python tools\mav6d_to_yolo.py --mav6d D:\MAV6D --model phantom4 --out data\mav6d  # 转换
python tools\check_pose_dataset.py --data data\mav6d\phantom4\dataset.yaml       # 开训前校验
python tools\train_mav6d.py --model phantom4 --epochs 100 --batch 4 --imgsz 1280  # 训练
python tools\test_mav6d.py --model phantom4       # 测试（--model both 两机型一起报）
```

**本库从数据里实证核对过的四件事**（都写进了代码注释，别改）：

1. `labels_yolo6d` 是 **21 列** = `cls` + 9 个点 ×2 + **(w,h)**。9 点 = 1 个 3D 框中心 +
   8 个角点（与作者 `util.py` 的 `corners` 及论文 p0~p8 一致）。作者自己的
   `unzip_tools.py` 报告里写成"10 个归一化点"**是错的**（本库用作者投影链逐值反解确认）。
2. 归一化基准固定 **1920×1080**（转换时按各图实际尺寸重新归一化）。
3. `labels` 每行 **16 个数**，只有**相机位姿**（`[9:12]` + `[12:16]`）可靠；目标位姿块是
   `[1:4]`（平移）+ `[4:8]`（四元数），而 `[0]`/`[8]` 是纳秒时间戳 —— 偏移写错会差 670–1900 px。
4. **标注含镜头畸变**（作者用了 5 个畸变系数，画幅边缘达几十像素）→ 转换时把 2D 关键点
   **去畸变**，让标签与 `L_rep` 的针孔几何自洽。

已转换工作量（可复算）：`phantom4` **11 674 帧**（10 622 训练 / 1 052 验证，按**序列**切分），
**全部带有效 6D 位姿**；框长边归一化 0.099–0.273（1920 px 下约 190–520 px）。

### SPAE-YOLOv8n —— 空对空微小型无人机检测（Sensors 2026）

论文 [SPAE-YOLOv8](https://doi.org/10.3390/s26113424) 的四个组件全部落地：

| 组件 | 本库位置 | 要点 |
|---|---|---|
| **P2 浅层** | `compose.inject_p2_head` | 160×160 分支，P2-P3-P4-P5 四层头；1×1 降维 |
| **ADown** | `modules/conv/adown.py` | 双分支自适应下采样（avg-pool → Conv3×3/s2 与 MaxPool+Conv1×1） |
| **SIoU** | `loss/box.py` + `loss/criterion.py` | 角度+距离+形状三项代价，只替换 IoU 项、复用框架 DFL |
| **Efficient_UAVDet** | `modules/head/efficient_uavdet.py` | 分支 stem 换成两层 3×3 分组卷积，**g = x/16**（每组 16 通道） |

```powershell
# 物化变体（生成 variant.yaml / model.yaml / card.md）
python tools\make_variant.py variants\SPAE-YOLOv8n\recipe.py

# 结构自检：建模型 + 一次前向，不需要数据集
python tools\train.py --variant variants\SPAE-YOLOv8n\variant.yaml --dry-run

# 真训练（需先准备 VisDrone）
python tools\train.py --variant variants\SPAE-YOLOv8n\variant.yaml --data configs\_base_\datasets\visdrone2019-det.yaml
```

取证笔记（含论文原文摘录、消融表、**两处论文内部数据冲突**、以及"哪些是原文、哪些是我们的推断"）
见 `variants/SPAE-YOLOv8n/paper-notes.md`。

> ⚠️ 两个必须知道的结论：① 论文消融显示 **P2 贡献 +7.5pp 是绝对主力**，而 Efficient_UAVDet 是
> **−0.4pp 的"降本不涨点"**（参数 −20%、FPS +28%），论文自己也承认；② 论文用 Det-Fly
> 空对空数据集，本库换成 VisDrone 空对地，**论文的 mAP 数字不可直接引用**。

### SDD-YOLO26n —— 空对地反无人机小目标检测（arXiv:2603.25218）

论文 [SDD-YOLO](https://arxiv.org/abs/2603.25218) 的四大贡献中，可复现部分全部落地：

| 组件 | 论文 | 本库落点 | 说明 |
|---|---|---|---|
| **P2 高分辨率头** | §4.2 式 (1) | `model(add_p2=True, p2_fuse_block="C3")`（EP5/EP2） | 4× 下采样；**C3 瓶颈融合是原文用词** |
| **双注意力** | §4.5 式 (4) | `modules/attention/dual_attention.py`（EP4） | `σ(W_c·GAP) ⊗ σ(Conv7×7([Avg;Max]))`，通道不变、原位插入 |
| **DFL-free** | §4.3 式 (3) | `train(dfl=0.0)` + `loss(box="wiou")`（EP7） | 底座已是 `reg_max=1`；增益为 0 时连计算都省掉 |
| **NMS-free** | §4.4 | 底座 `end2end=True` + `E2ELoss` | 本库把**两套**准则（O2M/O2O）的 IoU 项都替换掉 |
| **MuSGD** | §4.6 式 (5) | `optim/musgd.py`（EP9） | 优先用框架原生；二者数值对照 <1e-3 |
| **ProgLoss / STAL** | §4.6 | 底座 `E2ELoss.update` / `TaskAlignedAssigner` | 论文未给 STAL 公式，本库显式化为可消融开关 |
| **特征对齐 KD** | §4.7 式 (6)(7) | `engine/distill.py`（EP9） | λ=0.5、T=3.0；默认关闭（YOLO26x 教师 8 GB 放不下） |

```powershell
python tools\make_variant.py variants\SDD-YOLO26n\recipe.py
python tools\train.py --variant variants\SDD-YOLO26n\variant.yaml --dry-run
```

> ⚠️ 引用前必读 `variants/SDD-YOLO26n/paper-notes.md`：论文用**未公开**的 DroneSOD-30K
> （且三个子集相加 47 750 ≠ 摘要"约 30K"），本库换 VisDrone，**86.0 mAP@0.5 不可引用**；
> 论文 Table 3 与 Table 2 的同一配置给了三组不同数字；论文声称"加 P2 后 Params/FLOPs 完全不变"，
> 本库实测（`get_flops`，imgsz=1024）为 **+3.60 GFLOPs（18.43 vs 基线 14.83）**、参数方向取决于
> `ch[0]` 与 `nc`（nc=10 时反而 −20,900）；式 (4) 的空间分支与正文"运动区域"描述不符；
> 式 (7) 未定义锚点维归一化（按锚点求和时 KD=2017.8 会压垮 L_task=22.8，本库默认取均值）。

## 核心思想

**一个魔改 = 一个注册模块 + 一份元数据 + 一个变体配置。永不 fork 主干代码。**

- `src/tod/registry.py` —— 所有魔改的唯一入口，强制登记论文/许可证/扩展点归属
- `src/tod/compose.py` —— 变体 DSL：`骨架 + 扩展点覆盖 → 模型 YAML / 变体卡片`
- `src/tod/compat.py` —— **唯一**允许触碰 ultralytics 内部结构的文件
- 扩展点 `EP0–EP9` —— 数据 / 主干 / 颈部 / 上采样 / 注意力 / 头 / 分配 / 损失 / 推理 / 训练策略

## 快速开始

**完整全流程（拉取 → 建环境 → 数据 → 训练 → 测试）见 [`WORKFLOW.md`](WORKFLOW.md)**，
下面只是最短路径。用到的文件：`requirement.txt`（必需依赖）、`tools/setup_venv.ps1`（一键建 `.venv`）。

```powershell
# 1) 建环境 + 装依赖 + 自检（一键）
powershell -ExecutionPolicy Bypass -File tools\setup_venv.ps1

# 等价的手工步骤（脚本做的就是这些）
python -m venv .venv                      # 3.10~3.14 均可（唯一要避开 3.14.1）
.\.venv\Scripts\Activate.ps1
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128  # 按你的 CUDA 换 cuXXX
python -m pip install -r requirement.txt
python -m pip install -e . --no-deps      # 本仓库可编辑安装

# 2) 架构层冒烟测试（不需要 GPU / torch）
python tests\smoke.py

# 3) 查看已登记的魔改模块总表
python tools\catalog.py
```

> `torch` / `torchvision` **不能**从 PyPI 装：PyPI 上是 CPU-only 构建，GPU 版只在
> `https://download.pytorch.org/whl/cuXXX` 上。所以清单里**不含 torch**，由你自己按显卡与驱动
> 选索引（RTX 5060 这类 Blackwell 卡需 cu128 及以上；`nvidia-smi` 可看驱动支持的 CUDA 版本）。
> 顺序也别反：**先 torch 后 `requirement.txt`**，否则 pip 会在装 `ultralytics` 时把 CPU 版 torch 盖上去。
>
> `tools/setup_venv.ps1` 刻意写成**纯 ASCII**：Windows PowerShell 5.1 会用 ANSI 代码页
> （本机 936/GBK）解析无 BOM 的 UTF-8 脚本，中文注释会被解坏并导致语法错误。


## 最小示例：构造一个「P2 头 + DySample + NWD」变体

```python
from tod.compose import Variant

v = (Variant("visdrone-yolov8n-p2-dysample-nwd", base="yolov8n")
     .data("visdrone2019-det", imgsz=1280, slicing={"patch": 1024, "overlap": 0.2})
     .upsample("DySample")          # EP3
     .loss(box="NWD")               # EP7
     .head(levels=[2, 3, 4, 5])     # EP5：加 P2 检测头
     .model(add_p2=True, nc=10)
     .train(epochs=150, batch=4, amp=True))

v.dump("configs/variants/visdrone/visdrone-yolov8n-p2-dysample-nwd.yaml")
v.model_yaml("variants/visdrone-yolov8n-p2-dysample-nwd/model.yaml")
v.card("variants/visdrone-yolov8n-p2-dysample-nwd/card.md")
```

## 目录

```
src/tod/            训练侧（Python）
├─ registry.py      魔改注册表（强制元数据；规范名与别名都会注入框架命名空间）
├─ compose.py       变体 DSL + 模型图改造（P2 注入 / 下采样替换 / 类型替换 / **task 与 kpt_shape**）
├─ compat.py        唯一触碰 ultralytics 内部的地方（含只读配置目录回退、PoseModel 获取）
├─ runtime.py       变体 spec 的运行时上下文
├─ modules/
│  ├─ conv/adown.py            ADown（EP1）
│  ├─ attention/dual_attention.py DualAttention（EP4，SDD-YOLO §4.5）
│  └─ head/efficient_uavdet.py Efficient_UAVDet（EP5，检测/关键点分支均可换）
├─ assigner/stal.py  STAL 小目标感知分配（EP6，可消融开关）
├─ optim/musgd.py    MuSGD（EP9，Muon 式 NS 正交化 + SGD 分量）
├─ loss/box.py + criterion.py  SIoU / Wise-IoU v3 与训练准则接入（EP7）
├─ loss/pose.py      **TinyPoseLoss**：OKS 关键点损失（EP7，sigma 策略 person/uniform/auto/balanced + σ 下限 + 可见性开关）
├─ loss/reproj.py    **L_rep 几何重投影损失**（EP7，论文式 6/8/9：只罚垂直于相机射线的横向偏差 + warm-up 调度）
├─ data/pose6d.py    6D 位姿旁挂数据（EP0，labels_6d CSV → batch 的 gt_pose / gt_image_size）
├─ eval/scales.py    尺度分层评测（整体 AP + AP_small/AP_tiny，COCO 式 101 点插值）
├─ eval/pose.py      **姿态 OKS + ADD 评测**（EP8：OKS-AP 50:95 + 按目标/关键点尺度分层 + 论文 ADD(0.5d)）
└─ engine/           trainer.py + **pose_trainer.py**（EP 接线）/ surgery.py（EP4·EP5 建模后手术）
                    / distill.py（EP9 特征对齐蒸馏）

src/todrt/          部署侧（C++17，四个后端可选）—— 与 src/tod 并列，可独立复用
├─ include/todrt/   core（注册表）/ json / modules / factory / backend
├─ src/models/      ★ 变体配方（一个变体一个文件，两行宏）
├─ src/             factory / preprocess（float 或 uint8）/ decode / nms / config_io
├─ src/backend/     四后端：engine_trt / engine_rknn / engine_ort / engine_openvino
│                   + backend_dispatch（唯一分派处）+ simple_detector（共用外壳）
├─ apps/todrt_cli   部署工具（list/info/dryrun/probe/bench/run）
└─ tests/cpp_smoke  架构自检（**不需要任何后端**，148 项）

configs/_base_/     数据集等基础配置（visdrone2019-det / **visdrone2019-pose** / **rflysim-pose**）
configs/deploy/     部署配置（由 tools/export_onnx.py 生成，Python 与 C++ 的唯一契约）
variants/<name>/    recipe.py（代码定义）+ variant.yaml + model.yaml + card.md + paper-notes.md
tools/              make_variant.py / train.py / val.py / **val_pose.py** / ablation.py / catalog.py
                    / **check_pose_dataset.py**（关键点数据开训前校验）
                    / make_dummy_dataset.py（--task detect|pose --layout person|uav --kpt-shape N,D）
                    / **train_mav6d.py + test_mav6d.py**（MAV6D 训练与全指标测试）
                    / **setup_venv.ps1**（一键建 .venv，纯 ASCII）
                    / export_onnx.py / convert_rknn.py
WORKFLOW.md         **全流程命令**：拉取 → 建 venv → 数据 → 训练 → 测试 → 消融 → 导出
requirement.txt     venv 依赖清单（**不含 torch**：GPU 轮子按自己的 CUDA 版本单独装）
docs/DEPLOY.md      部署流程与四个后端（TRT / RKNN / ORT / OpenVINO）的实操与坑
docs/KEYPOINTS.md   **9 点机体关键点定义（唯一编号来源，含 flip_idx 与推断状态）**
docs/VARIANTS.md    模块总表（由 tools/catalog.py 自动生成）
tests/              smoke.py（无 torch 依赖，77 项）+ test_modules.py（105 项）
                    + test_pose.py（**姿态/关键点/6D 位姿/L_rep，220 项**）
                    + **test_mav6d.py**（MAV6D 工具链，49 项；`--skip-real` 可跳过真实数据）
                    + train_smoke.py（检测回路真训练，opt-in）
                    + train_pose_smoke.py（**姿态回路真训练 + L_rep + ADD 断言，32 项**）
```

## 评测与消融（可归因对比）

```powershell
# 0) 合成数据集：几秒钟造一份"训练回路自检"数据（离线、确定性）
python tools\make_dummy_dataset.py --out tests\.tmp\tiny-detect

# 1) 训练回路自检：真训练 + 验证 + 存/载权重 + 推理（1–2 分钟）
python tests\train_smoke.py --epochs 2 --device 0

# 2) 尺度分层评测：整体 AP **与** AP_small / AP_tiny 一起报（只看整体会掩盖小目标退化）
python tools\val.py --variant variants\SDD-YOLO26n\variant.yaml `
    --weights results\SDD-YOLO26n\weights\best.pt `
    --data configs\_base_\datasets\visdrone2019-det.yaml --imgsz 1024 `
    --json variants\SDD-YOLO26n\results.json

# 3) 消融流水线：先看网格与逐字段 diff（秒级、不训练），确认无误再真跑
python tools\ablation.py --recipe variants\SDD-YOLO26n\recipe.py --plan
python tools\ablation.py --recipe variants\SDD-YOLO26n\recipe.py `
    --data configs\_base_\datasets\visdrone2019-det.yaml --epochs 100 --imgsz 1024 --batch 4
```

> 消融的两条硬规矩（都写进了代码）：① 组件的"关掉"动作不能一律用 `without` ——
> 由**框架继承**来的能力（STAL / MuSGD）删键只会退回框架默认（8.4 的默认 TAL 本身就带
> 小目标先验），必须显式切到经典替代物；② `v.without()` 覆盖 EP、model、train 三段，
> 并会自动识别"剔掉的键本来就是默认值"这种假消融。

## 部署（推理端）

```powershell
# 导出 ONNX + 部署配置（需要 torch/ultralytics）
python tools\export_onnx.py --variant variants\SPAE-YOLOv8n\variant.yaml `
    --weights results\SPAE-YOLOv8n\weights\best.pt --imgsz 640 `
    --deploy-config configs\deploy\spae-yolov8n.json
```

```bash
# 按目标硬件选一个后端构建（Linux）
cmake -S src/todrt -B build/x -DTODRT_WITH_TENSORRT=ON      # Jetson Orin / dGPU
cmake -S src/todrt -B build/x -DTODRT_WITH_RKNN=ON          # RK3588 NPU（librknnrt.so）
cmake -S src/todrt -B build/x -DTODRT_WITH_ORT=ON           # 通用 CPU（AMD x86）
cmake -S src/todrt -B build/x -DTODRT_WITH_OPENVINO=ON      # x86 CPU / Intel iGPU
cmake --build build/x -j$(nproc)

./build/x/todrt_cli probe                              # 逐后端报可用性
./build/x/todrt_cli dryrun configs/deploy/xxx.json     # 配置装配自检（不需要模型/设备）
./build/x/todrt_cli bench  configs/deploy/xxx.json sample.ppm 200
```

RK3588 的 `.rknn` 在 **x86 主机**上转换（板上不转换）：

```bash
pip install rknn-toolkit2
python tools/convert_rknn.py --onnx exports/SPAE-YOLOv8n.onnx --target rk3588 \
    --out exports/spae.rk3588.rknn --dataset exports/calib.txt \
    --deploy-config configs/deploy/spae-yolov8n.json   # 自动切 uint8 NHWC 前处理
```

调用方只有两行（换后端不改业务代码）：

```cpp
auto det = todrt::Detector::CreateFromFile("configs/deploy/xxx.json");
auto results = det->Run(bgr_image);          // 坐标已是原图像素
```

详见 [`docs/DEPLOY.md`](docs/DEPLOY.md)（各后端的构建/转换/坑）与
[`src/todrt/README.md`](src/todrt/README.md)（工厂模式设计说明）。

## 下一步（M1）

1. 准备 VisDrone，跑通「YOLOv8n + P2 + imgsz=640」与「YOLO26n + P2 + imgsz=1024」两条**干净基线**
   作为锚点（后者的结构自检已通过：2.50 M 参数、nl=4、stride=[4,8,16,32]；训练回路已用合成数据验证）；
2. 用 `tools/ablation.py` 做单模块消融：SPAE 复现论文贡献排序；
   SDD 补齐论文 Table 3 **捆绑在一起**的 `¬DFL / NMS-free / MuSGD / STAL` 四列各自贡献；
3. 用 `tools/val.py` 统一上报整体 AP 与 `AP_small`/`AP_tiny`（含 3 个 seed 的重复性检查）；
4. 再按 `PLAN.md §5` 的 P0 清单补模块（SAHI、Copy-Paste、SPD-Conv、BiFPN/ASFF、NWD…）。

部署侧（`src/todrt/`）待实机验证的清单：

- [ ] 在 Orin 上跑 `todrt_cli probe`，记录 TensorRT 版本与 DLA core 数；
- [ ] `dryrun` 一份真实部署配置，确认 anchor 数、strides、layout 与导出图一致；
- [ ] 构建 FP16 engine，`bench` 出「推理 / 后处理 / 端到端」三段延迟，
      判断瓶颈在哪一段（小目标变体经常卡在后处理，因为 P2 头有 25600 个 anchor）；
- [ ] 开 DLA（`--preset=orin`）后对比层分布：DLA 层数、GPU 回退层是哪几层；
- [ ] 复测 ONNX / FP16 / INT8 / DLA 四档的 `AP_small`（**不能只看 FPS**）。

> 注意：`PLAN.md §11.1` 记录了 8 GB 显存下的分辨率/batch 约束，不要一上来就用 1536。
> 部署端只能验证延迟与吞吐，**精度必须回到训练侧评测协议**（`PLAN.md §7`）。
