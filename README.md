# yolo-tod —— 小目标检测魔改 YOLO 收集库

收集、复现并**可归因地**对比针对小目标检测（Tiny/Small Object Detection）的 YOLO 魔改。
当前代码结构见 [架构文档](docs/ARCHITECTURE.md)，逐文件职责见 [代码文件索引](#代码文件索引)。
研究选型、优先级与评测协议见 [PLAN.md](PLAN.md)，全流程命令见 [WORKFLOW.md](WORKFLOW.md)。
面向无人机本体的新变体选型见 [RGB 目标无人机检测文献与复刻路线](docs/RGB_UAV_DETECTION_SURVEY.md)：55 项论文/方法，含抗干扰、小目标、视频和机体姿态。

当前已实现检测变体 **SPAE-YOLOv8n / SDD-YOLO26n**，以及姿态变体
**spae-yolov8n-pose / visdrone-yolo26n-pose-p2-p16**。训练端包含 9 点 UAV 几何监督、
MAV6D 转换与训练入口、尺度分层 AP/OKS、ADD/PnP、开训前校验和消融流水线；
另有合成数据的实际训练自检。航拍人体 17 点的真实标注仍待准备，真实数据集精度需独立验证。
推理端 `src/todrt/` 提供工厂装配与四种可选后端。

2026-10-29 注释与文档整理时，架构自检通过 77 项。当前本机环境使用 ultralytics 8.4.152，
模块与姿态测试中的分配器 dtype / 损失返回值兼容失败在整理前源码上同样存在；
本次整理保留原有执行逻辑，验证记录见 [架构文档](docs/ARCHITECTURE.md#6-验证与维护)。

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

```text
python3 tools/make_variant.py variants/spae-yolov8n-pose/recipe.py
python3 tools/train.py --variant variants/spae-yolov8n-pose/variant.yaml --dry-run
python3 tools/check_pose_dataset.py --data configs/_base_/datasets/rflysim-pose.yaml   # 开训前校验
python3 tests/train_pose_smoke.py --layout uav --warmup 0   # 合成数据端到端（含 L_rep 与 ADD）
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
| **OKS 关键点损失** | `src/tod/loss/pose.py`（EP7） | `TinyPoseLoss` 与框架同签名；sigma 策略 `person`/`auto`/`balanced`；σ 下限；可见性开关 |
| **姿态训练器** | `engine/pose_trainer.py`（EP9） | 继承检测侧全部 EP 接线，改用 `PoseModel` + 姿态准则（O2M/O2O **两套**都替换） |
| **OKS 分层评测** | `eval/pose.py`（EP8） | OKS-AP 50:95 + 按目标边长分层 + **逐关键点尺度诊断**（≤1px 命中率/平均误差/平均 OKS） |
| **合成关键点数据** | `tools/make_dummy_dataset.py --task pose`（EP0） | 火柴人 + COCO 17 点，tiny(12–24px) 与日常(48–112px) 双尺度 |

```text
python3 tools/make_variant.py variants/visdrone-yolo26n-pose-p2-p16/recipe.py
python3 tools/train.py --variant variants/visdrone-yolo26n-pose-p2-p16/variant.yaml --dry-run
python3 tests/train_pose_smoke.py      # 合成关键点数据真训练（回路自检）
python3 tools/val_pose.py --weights results/.../best.pt `
    --data configs/_base_/datasets/visdrone2019-pose.yaml --imgsz 1024 --sigma-strategy person
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

```text
python3 tools/mav6d_calibrate.py --mav6d D:/MAV6D --model phantom4 --check-util   # 核对投影链
python3 tools/mav6d_to_yolo.py --mav6d D:/MAV6D --model phantom4 --out data/mav6d  # 转换
python3 tools/check_pose_dataset.py --data data/mav6d/phantom4/dataset.yaml       # 开训前校验
python3 tools/train_mav6d.py --model phantom4 --no-lrep --epochs 100 --batch 4 --imgsz 1280  # 2D baseline
python3 tools/train_mav6d.py --model phantom4 --epochs 100 --batch 4 --imgsz 1280            # + L_rep
python3 tools/test_mav6d.py --model phantom4       # 测试（--model both 两机型一起报）
```

> **数据架构/格式/坐标约定见 [`docs/MAV6D.md`](docs/MAV6D.md)。**
> 原始 labels 不在盘上时，用 `--pose-from-pnp` 由 2D 标签反解 `labels_6d`；
> Mavic 2 的机体宽高深比例与 Phantom 4 不同，必须再加 `--fit-pnp-shape`：
> `python3 tools/mav6d_to_yolo.py --model mavic2 --out data/mav6d --pose-from-pnp --fit-pnp-shape`。
> L_rep 的 6D 位姿块/单位/内参/翻转问题与修复记录见 [`docs/TRAINING_REVIEW.md`](docs/TRAINING_REVIEW.md)；
> 建议先跑 `--no-lrep` 的 2D baseline，再开几何项。

> 训练集位置统一由 `configs/datasets.yaml` 决定（换盘/换机器只改它的 `data_root`，
> 脚本里没有任何绝对路径）；`python3 tools/train.py --list-datasets` 可先确认解析结果。
> `dataset.yaml` 是转换产物（含标定得到的 `tod.camera_matrix` / `tod.object_points`，**不能手写**）；
> 图/标签已在盘上、只缺它时用 `--yaml-only` 补写：
> `python3 tools/mav6d_to_yolo.py --model phantom4 --out data/mav6d --yaml-only`。

**MAV6D 的四项格式与几何约定**（详见 [数据文档](docs/MAV6D.md) 与 [架构契约](docs/ARCHITECTURE.md#3-mav6d-与几何监督)）：

1. `labels_yolo6d` 是 **21 列** = `cls` + 9 个点 ×2 + **(w,h)**。9 点 = 1 个 3D 框中心 +
   8 个角点（与作者 `util.py` 的 `corners` 及论文 p0~p8 一致）。作者自己的
   `unzip_tools.py` 报告里写成"10 个归一化点"**是错的**（本库用作者投影链逐值反解确认）。
2. 归一化基准固定 **1920×1080**（转换时按各图实际尺寸重新归一化）。
3. `labels` 每行 **16 个数**：`[0]` 相机时间戳、`[1:8]` 相机位姿、`[8]` 机体时间戳、
   **`[9:16]` 机体位姿**。2026-10-26 修正：旧版把 `[1:8]` 当机体、`[9:16]` 当相机，
   正好反了（差 ~2400 px / 169°）。现在 `labels_6d` 由 2D 关键点 PnP 重建。
4. **标注含镜头畸变**（作者用了 5 个畸变系数，画幅边缘达几十像素）→ 转换时把 2D 关键点
   **去畸变**，让标签与 `L_rep` 的针孔几何自洽。

已转换工作量（可复算）：`phantom4` **11 674 帧**（10 622 训练 / 1 052 验证，按**序列**切分），
其中 **95.4%** 有 PnP 重建的 6D 位姿（其余帧 `L_rep` 跳过）；框长边归一化
min 0.0835 / p50 0.172 / max 0.526。`mavic2` 11 788 帧（11 122 / 666），
6D 覆盖率 **95.7%**，并用 `--fit-pnp-shape` 拟合了与 Phantom 4 不同的宽高深比例。

### SPAE-YOLOv8n —— 空对空微小型无人机检测（Sensors 2026）

论文 [SPAE-YOLOv8](https://doi.org/10.3390/s26113424) 的四个组件全部落地：

| 组件 | 本库位置 | 要点 |
|---|---|---|
| **P2 浅层** | `compose.inject_p2_head` | 160×160 分支，P2-P3-P4-P5 四层头；1×1 降维 |
| **ADown** | `modules/conv/adown.py` | 双分支自适应下采样（avg-pool → Conv3×3/s2 与 MaxPool+Conv1×1） |
| **SIoU** | `loss/box.py` + `loss/criterion.py` | 角度+距离+形状三项代价，只替换 IoU 项、复用框架 DFL |
| **Efficient_UAVDet** | `modules/head/efficient_uavdet.py` | 分支 stem 换成两层 3×3 分组卷积，**g = x/16**（每组 16 通道） |

```text
# 物化变体（生成 variant.yaml / model.yaml / card.md）
python3 tools/make_variant.py variants/SPAE-YOLOv8n/recipe.py

# 结构自检：建模型 + 一次前向，不需要数据集
python3 tools/train.py --variant variants/SPAE-YOLOv8n/variant.yaml --dry-run

# 真训练（需先准备 VisDrone；数据集位置由 configs/datasets.yaml 决定，见 WORKFLOW §4.0）
python3 tools/train.py --variant variants/SPAE-YOLOv8n/variant.yaml --data visdrone2019-det
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

```text
python3 tools/make_variant.py variants/SDD-YOLO26n/recipe.py
python3 tools/train.py --variant variants/SDD-YOLO26n/variant.yaml --dry-run
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
- `src/tod/compat.py` —— 集中管理框架版本、命名空间和运行环境兼容；训练/数据适配在各自职责内调用框架接口
- 扩展点 `EP0–EP9` —— 数据 / 主干 / 颈部 / 上采样 / 注意力 / 头 / 分配 / 损失 / 推理 / 训练策略

## 快速开始

**完整全流程（拉取 → 建环境 → 数据 → 训练 → 测试）见 [`WORKFLOW.md`](WORKFLOW.md)**，
下面只是最短路径，**Windows / Linux 通用**。用到的文件：`requirement.txt`（必需依赖）、
`tools/setup_venv.ps1`（Windows 一键建 `.venv`）、`tools/setup_venv.sh`（Linux/macOS 对等脚本）。

```bash
# 1) 建环境 + 装依赖 + 自检（一键）
bash tools/setup_venv.sh                       # Linux / macOS
powershell -ExecutionPolicy Bypass -File tools/setup_venv.ps1   # Windows

# 等价的手工步骤（脚本做的就是这些）
python3 -m venv .venv                      # 3.10~3.14 均可（唯一要避开 3.14.1）
source .venv/bin/activate                  # Linux/macOS；Windows: .\.venv\Scripts\Activate.ps1
python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128  # 按你的 CUDA 换 cuXXX
python3 -m pip install -r requirement.txt
python3 -m pip install -e . --no-deps      # 本仓库可编辑安装

# 2) 架构层冒烟测试（完整检查需已安装训练依赖）
python3 tests/smoke.py

# 3) 查看已登记的魔改模块总表
python3 tools/catalog.py
```

> **命令写法**：统一用 `python3` 与正斜杠 `/`（Windows 的 Python 与 PowerShell 都接受正斜杠）。
> Windows 上若 `python3` 不可用（可能是 Microsoft Store 的 0 字节占位假壳），
> 把它换成 `python` 即可，或按 `WORKFLOW.md` §0.0 的两行配置让它真正可用。
>
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

```text
src/tod/       Python 变体、训练、损失与评测
src/todrt/     C++17 推理接口、后端、CLI 与 CMake
tools/         环境、数据、训练、评测、消融与导出入口
variants/      每个实验变体的配方、配置、卡片和证据
configs/       数据集注册表、模板和标定预设
tests/         Python 自检与可选实际训练验证
scripts/       Git 辅助脚本
docs/          架构、关键点、MAV6D、训练分析与部署文档
results/       训练与评测输出（不入库）
build/、.cache/ 构建目录与运行缓存（不入库）
```

完整目录、依赖关系、调用链和数据契约见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。C++ 实机部署面向 Linux；本次整理在 Windows 上只做静态核对。

## 代码文件索引

以下按职责逐个列出当前代码与构建文件。文件内保留简短接口说明和关键不变量，长篇解释统一放在本 README 与专题文档。

### Python 基础与配置装配

| 文件 | 用途与调用关系 |
|---|---|
| [src/tod/__init__.py](src/tod/__init__.py) | 包入口；提供 UTF-8 输出配置、注册库加载与 bootstrap，训练前完成模块注册和框架注入。 |
| [src/tod/registry.py](src/tod/registry.py) | 定义 ModuleSpec、扩展点与状态；校验模块注册、管理别名，并生成带来源/许可证/成本的清单。 |
| [src/tod/compose.py](src/tod/compose.py) | 实现 Variant DSL、配置合并、P2 注入、类型和下采样替换；将配方物化为 variant.yaml、model.yaml 与卡片。 |
| [src/tod/compat.py](src/tod/compat.py) | 集中适配框架版本、模型命名空间、基础 YAML、运行配置目录和受限环境；向调用层提供兼容性错误。 |
| [src/tod/paths.py](src/tod/paths.py) | 读取数据集注册表与本机覆盖，统一解析名字/目录/YAML，检查图像标签并生成绝对路径 YAML 缓存。 |
| [src/tod/runtime.py](src/tod/runtime.py) | 维护当前变体 spec；用上下文管理器设置和恢复配置，供 Trainer 与损失装配读取。 |

### Python 数据、组件与优化器

| 文件 | 用途与调用关系 |
|---|---|
| [src/tod/data/__init__.py](src/tod/data/__init__.py) | 数据扩展包入口；导出 6D CSV 读取、行校验、姿态数据集构造、四元数转换及异常类型。 |
| [src/tod/data/pose6d.py](src/tod/data/pose6d.py) | 校验 labels_6d CSV 的目标行序与四元数，将位姿、有效性掩码和原图尺寸附加到样本，并临时替换框架数据集。 |
| [src/tod/modules/__init__.py](src/tod/modules/__init__.py) | 网络组件集合入口；导入卷积、注意力和头组件以触发注册，不保存实验配方。 |
| [src/tod/modules/conv/__init__.py](src/tod/modules/conv/__init__.py) | 主干卷积包入口；导入并对外提供 ADown。 |
| [src/tod/modules/conv/adown.py](src/tod/modules/conv/adown.py) | 实现 ADown 双分支下采样；平均池化后拆分卷积与池化支路，再拼接输出，要求输入通道可均分。 |
| [src/tod/modules/attention/__init__.py](src/tod/modules/attention/__init__.py) | 注意力包入口；导入并对外提供 DualAttention。 |
| [src/tod/modules/attention/dual_attention.py](src/tod/modules/attention/dual_attention.py) | 实现通道与空间双注意力；保持张量通道和空间尺寸，作为 EP4 特征增强模块。 |
| [src/tod/modules/head/__init__.py](src/tod/modules/head/__init__.py) | 输出头包入口；加载 Efficient_UAVDet 的注册实现。 |
| [src/tod/modules/head/efficient_uavdet.py](src/tod/modules/head/efficient_uavdet.py) | 实现分组卷积 stem、分组可行性检查和原生头分支替换；保留预测输出层、DFL 与姿态解码，并报告实际压缩情况。 |
| [src/tod/assigner/__init__.py](src/tod/assigner/__init__.py) | 标签分配包入口；导出 STAL 构造与安装接口。 |
| [src/tod/assigner/stal.py](src/tod/assigner/stal.py) | 构造带小目标先验开关的 TaskAlignedAssigner；识别单/双分支准则并安装分配器。 |
| [src/tod/optim/__init__.py](src/tod/optim/__init__.py) | 优化器包入口；导出 MuSGD 与原生实现选择接口。 |
| [src/tod/optim/musgd.py](src/tod/optim/musgd.py) | 实现 Newton–Schulz 正交化与 MuSGD 回退优化器；按参数维度分组，优先选择框架原生 MuSGD。 |

### Python 损失、训练与评测

| 文件 | 用途与调用关系 |
|---|---|
| [src/tod/loss/__init__.py](src/tod/loss/__init__.py) | 损失包入口；导出 SIoU/WIoU、TinyPoseLoss、sigma 与关键点准则替换接口；完整准则由 criterion.py 装配。 |
| [src/tod/loss/box.py](src/tod/loss/box.py) | 实现 SIoU 与有状态 WiseIoU，以及按名字选择回归损失的接口；不承担完整训练循环。 |
| [src/tod/loss/criterion.py](src/tod/loss/criterion.py) | 将框损失、可选 DFL、STAL、关键点损失和 L_rep 装配进框架准则，同时覆盖端到端 O2M/O2O 子准则。 |
| [src/tod/loss/pose.py](src/tod/loss/pose.py) | 实现 TinyPoseLoss、OKS sigma 策略及可见性配置；按关键点数量生成或校验 sigma，并替换姿态子准则。 |
| [src/tod/loss/reproj.py](src/tod/loss/reproj.py) | 实现相机射线与三维机体点的横向几何损失、warm-up 包装、像素解码和逐目标位姿查找；依赖相机内参与 labels_6d。 |
| [src/tod/engine/__init__.py](src/tod/engine/__init__.py) | 训练引擎包入口；延迟提供检测/姿态 Trainer，避免仅使用基础模块时提前加载完整训练框架。 |
| [src/tod/engine/trainer.py](src/tod/engine/trainer.py) | 扩展 DetectionTrainer；接入建模后修改、框损失、STAL、MuSGD 与 KD，并在断点恢复后重装训练准则。 |
| [src/tod/engine/pose_trainer.py](src/tod/engine/pose_trainer.py) | 复用检测扩展与框架 PoseTrainer；按需绑定 6D loader，装配关键点和几何损失，报告 warm-up 与实际生效状态。 |
| [src/tod/engine/surgery.py](src/tod/engine/surgery.py) | 定位预测头和金字塔层；按变体配置挂接注意力、替换头分支，并记录实际执行的模型修改。 |
| [src/tod/engine/distill.py](src/tod/engine/distill.py) | 加载冻结教师，按特征层对齐类别分布；实现温度缩放 KL 蒸馏及包装准则，支持锚点归约方式配置。 |
| [src/tod/eval/__init__.py](src/tod/eval/__init__.py) | 评测包入口；对外组织框尺度分层与姿态评测接口，避免将训练逻辑混入指标实现。 |
| [src/tod/eval/scales.py](src/tod/eval/scales.py) | 解析数据划分和框标签，完成 IoU 匹配、AP 积分与按目标尺寸分桶，输出整体和 tiny/small 指标。 |
| [src/tod/eval/pose.py](src/tod/eval/pose.py) | 实现 OKS 匹配/AP、关键点像素误差、ADD 与 PnP 位姿指标；包含真值深度诊断和完整 R/t 指标两种口径。 |

### 命令行工具

| 文件 | 用途与调用关系 |
|---|---|
| [tools/setup_venv.ps1](tools/setup_venv.ps1) | Windows 环境安装入口；支持 -PythonVersion、-TorchIndex、-SkipChecks，创建 .venv、安装依赖与可编辑包并执行自检；脚本保持 ASCII。 |
| [tools/setup_venv.sh](tools/setup_venv.sh) | Bash 环境安装入口；支持 --python、--torch-index、--skip-checks、--help，并兼容 POSIX 与 Windows Python 的 venv 布局。 |
| [tools/catalog.py](tools/catalog.py) | 加载注册组件，将模块来源、许可证、成本和扩展点输出为清单；可更新 docs/VARIANTS.md。 |
| [tools/make_variant.py](tools/make_variant.py) | 加载 recipe.py 中的变体与消融配方，调用 DSL 生成可审阅的模型 YAML、实验配置和卡片。 |
| [tools/make_dummy_dataset.py](tools/make_dummy_dataset.py) | 生成固定随机种子的极小检测/姿态数据；支持人体和 UAV 布局，并可生成几何监督所需的 6D 位姿。 |
| [tools/check_pose_dataset.py](tools/check_pose_dataset.py) | 开训前检查关键点定义、镜像索引、标签列数/范围/可见性、图像配对、尺度分布及 6D 配置；可写 JSON 报告。 |
| [tools/mav6d_calibrate.py](tools/mav6d_calibrate.py) | 读取作者原始相机/机体位姿和 9 点标签，执行标定、三角化、几何求解，并用 --check-util 核对作者投影链。 |
| [tools/mav6d_to_yolo.py](tools/mav6d_to_yolo.py) | 按机型和序列转换 MAV6D；处理去畸变、YOLO 标签、PnP 位姿重建、机体比例拟合与转换清单，支持只补 dataset.yaml。 |
| [tools/train.py](tools/train.py) | 通用训练入口；解析变体、数据路径及超参覆盖，选择检测/姿态 Trainer，支持 --dry-run 与 --list-datasets。 |
| [tools/train_mav6d.py](tools/train_mav6d.py) | MAV6D 专用训练入口；选择机型与数据集，控制 L_rep 开关、warm-up 和参数覆盖，再调用相应 Trainer。 |
| [tools/val.py](tools/val.py) | 加载检测权重与数据划分，调用尺度分层评测，打印整体/tiny/small AP，并保存带实验信息的 JSON。 |
| [tools/val_pose.py](tools/val_pose.py) | 姿态评测入口；配置 sigma、数据划分和尺度桶，输出 OKS-AP、逐关键点诊断及可选 ADD。 |
| [tools/test_mav6d.py](tools/test_mav6d.py) | 定位单/双机型权重并评测；结合检测 AP、OKS 分层、PnP ADD/ADD(0.5d)、旋转/平移与 10°10cm 指标生成报告。 |
| [tools/ablation.py](tools/ablation.py) | 加载完整变体和 leave-one-out 配方，比较逐字段差异，批量训练/评测并汇总 CSV 与 Markdown；--plan 仅预览。 |
| [tools/export_onnx.py](tools/export_onnx.py) | 加载变体/权重并导出 ONNX；推断实际输出布局、stride 和类别数，生成 C++ 部署配置，可选构建 TensorRT engine。 |
| [tools/convert_rknn.py](tools/convert_rknn.py) | 使用 RKNN-Toolkit2 将 ONNX 转为 .rknn，设置输入归一化和量化参数，并更新部署配置中的模型路径与输入契约。 |

### 变体配方

| 文件 | 用途与调用关系 |
|---|---|
| [variants/SPAE-YOLOv8n/recipe.py](variants/SPAE-YOLOv8n/recipe.py) | 组合 P2、ADown、SIoU 与 Efficient_UAVDet，提供各组件的消融入口；物化为同目录配置和卡片。 |
| [variants/SDD-YOLO26n/recipe.py](variants/SDD-YOLO26n/recipe.py) | 组合 YOLO26 的 P2/C3、双注意力、WIoU、STAL 与 MuSGD；保留 NMS/DFL-free 底座及可选教师蒸馏配置。 |
| [variants/spae-yolov8n-pose/recipe.py](variants/spae-yolov8n-pose/recipe.py) | 将 SPAE 组件接入 YOLOv8 Pose，定义 9 点 UAV 任务、OKS 与 L_rep；关键点分支默认保留原生通道。 |
| [variants/visdrone-yolo26n-pose-p2-p16/recipe.py](variants/visdrone-yolo26n-pose-p2-p16/recipe.py) | 定义 P2 航拍人体 17 点姿态变体与 OKS 策略；真实 VisDrone 关键点标签仍需准备。 |

### Python 验证入口

| 文件 | 用途与调用关系 |
|---|---|
| [tests/smoke.py](tests/smoke.py) | 检查注册表、DSL 往返、图变换、消融和模型配置；部分实际模块签名/建图检查需要 torch 与框架。 |
| [tests/test_dataset_paths.py](tests/test_dataset_paths.py) | 建立临时数据树，检查注册表深合并、三种路径入口、环境/命令行优先级、错误信息与数据统计。 |
| [tests/test_modules.py](tests/test_modules.py) | 验证 ADown、SIoU/WIoU、双注意力、STAL、MuSGD、分组头、蒸馏和完整检测建图的数值及梯度。 |
| [tests/test_pose.py](tests/test_pose.py) | 验证关键点 sigma/OKS、标签格式、姿态建图、头替换、双准则、L_rep 单位与 6D 数据绑定。 |
| [tests/test_pose6d_swap.py](tests/test_pose6d_swap.py) | 回归验证两个框架命名空间中的数据集替换、样本位姿字段及替换上下文恢复，防止几何监督静默缺失。 |
| [tests/test_mav6d.py](tests/test_mav6d.py) | 使用合成数据检查原始标签解析、位姿、畸变、序列切分与转换；真实数据检查可用 --skip-real 跳过。 |
| [tests/test_repo_hygiene.py](tests/test_repo_hygiene.py) | 检查非锚定忽略规则是否误伤源码/配置、关键源文件是否已跟踪、临时脚本是否误入版本库。 |
| [tests/train_smoke.py](tests/train_smoke.py) | 可选的实际检测训练自检；合成数据训练后核对权重、日志、参数更新、载入与推理。 |
| [tests/train_pose_smoke.py](tests/train_pose_smoke.py) | 可选的实际姿态训练自检；检查二维关键点、UAV 6D/L_rep、warm-up、ADD 和权重推理证据。 |

### C++ 公共接口与后端契约

| 文件 | 用途与调用关系 |
|---|---|
| [src/todrt/include/todrt/core.hpp](src/todrt/include/todrt/core.hpp) | 声明统一异常、设备/精度、日志和注册元数据接口，是其他运行时头文件的基础。 |
| [src/todrt/include/todrt/json.hpp](src/todrt/include/todrt/json.hpp) | 声明轻量 JSON Value、解析与序列化接口，支持配置文件所需的宽松语法。 |
| [src/todrt/include/todrt/modules.hpp](src/todrt/include/todrt/modules.hpp) | 定义框、图像/张量视图、前处理结果、输出布局、DFL/NMS 配置与前后处理接口，统一坐标和数据类型契约。 |
| [src/todrt/include/todrt/detector_options.hpp](src/todrt/include/todrt/detector_options.hpp) | 组合一次部署的模型、设备、构建、输入形状和前后处理选项，并声明配置读写接口。 |
| [src/todrt/include/todrt/factory.hpp](src/todrt/include/todrt/factory.hpp) | 声明 builder、模型配方、注册宏与 Detector API，提供同步/异步检测、装配计划和性能测试接口。 |
| [src/todrt/include/todrt/backend/factories.hpp](src/todrt/include/todrt/backend/factories.hpp) | 声明 EngineInputSpec、后端查找/构造与配置合成辅助接口，供公共工厂和 SDK 实现共享。 |
| [src/todrt/include/todrt/backend/simple_detector.hpp](src/todrt/include/todrt/backend/simple_detector.hpp) | 定义 IEngineRunner、AsyncDetectorBase 与 RunnerDetector；统一队列、像素所有权、推理缓冲和后处理流水线。 |
| [src/todrt/include/todrt/backend/engine_trt.hpp](src/todrt/include/todrt/backend/engine_trt.hpp) | 以 PIMPL 暴露 TensorRT 引擎构建/加载、执行、序列化和 DLA 诊断接口，公共头不引入 TRT SDK 类型。 |
| [src/todrt/include/todrt/backend/trt_factory.hpp](src/todrt/include/todrt/backend/trt_factory.hpp) | 声明 TensorRT Detector 构造入口，将引擎与公共前后处理外壳组装为可用检测器。 |
| [src/todrt/include/todrt/backend/rknn_factory.hpp](src/todrt/include/todrt/backend/rknn_factory.hpp) | 声明 RKNN Detector 构造入口，隔离 Rockchip SDK 实现与上层工厂。 |
| [src/todrt/include/todrt/backend/ort_factory.hpp](src/todrt/include/todrt/backend/ort_factory.hpp) | 声明 ONNX Runtime Detector 构造入口，上层无需包含 ORT SDK 头。 |
| [src/todrt/include/todrt/backend/openvino_factory.hpp](src/todrt/include/todrt/backend/openvino_factory.hpp) | 声明 OpenVINO Detector 构造入口，隔离 SDK 与公共调用接口。 |

### C++ 实现、应用与构建

| 文件 | 用途与调用关系 |
|---|---|
| [src/todrt/src/core.cpp](src/todrt/src/core.cpp) | 实现日志回调、枚举解析、注册元数据/别名查询与依赖检查，保持规范名清单的稳定顺序。 |
| [src/todrt/src/json.cpp](src/todrt/src/json.cpp) | 实现 JSON 递归解析、类型访问和序列化，并处理注释、尾随逗号与错误位置。 |
| [src/todrt/src/config_io.cpp](src/todrt/src/config_io.cpp) | 读写部署与运行时配置，按配置段和硬件 preset 合成 DetectorOptions，处理类型和参数校验。 |
| [src/todrt/src/factory.cpp](src/todrt/src/factory.cpp) | 维护组件构造函数表，合并配方默认值、解析工厂链，创建 Detector 并生成 PlanAssembly 装配诊断。 |
| [src/todrt/src/preprocess.cpp](src/todrt/src/preprocess.cpp) | 无 OpenCV 的 letterbox/缩放实现；输出 float32 NCHW 或 uint8 NHWC，并记录源图坐标反变换参数。 |
| [src/todrt/src/decode.cpp](src/todrt/src/decode.cpp) | 按四种输出布局完成预测解码，包括 DFL 与插件 NMS 结果；校验锚点数量、通道和配置。 |
| [src/todrt/src/nms.cpp](src/todrt/src/nms.cpp) | 实现硬 NMS、soft-NMS 和后处理器工厂，将解码结果按置信度、类别和保留数量筛选。 |
| [src/todrt/src/models/yolov8_baseline.cpp](src/todrt/src/models/yolov8_baseline.cpp) | 注册原生 YOLOv8 检测配方、别名与元数据，作为工厂装配和部署对照基线。 |
| [src/todrt/src/models/spae_yolov8n.cpp](src/todrt/src/models/spae_yolov8n.cpp) | 注册 SPAE 模型配方和硬件相关默认项；实际网络结构由导出模型及部署配置描述。 |
| [src/todrt/src/backend/backend_dispatch.cpp](src/todrt/src/backend/backend_dispatch.cpp) | 集中将后端名称/别名映射到具体构造函数，并对未编译的 SDK 后端返回明确诊断。 |
| [src/todrt/src/backend/backend_support.cpp](src/todrt/src/backend/backend_support.cpp) | 根据引擎真实输入合成前处理选项，根据导出配置合成后处理选项，处理布局和数据类型冲突。 |
| [src/todrt/src/backend/simple_detector.cpp](src/todrt/src/backend/simple_detector.cpp) | 执行 runner、组织主机输出 TensorView、调用解码/NMS 并还原源图坐标，同时输出检测器描述。 |
| [src/todrt/src/backend/engine_trt.cpp](src/todrt/src/backend/engine_trt.cpp) | 实现 TRT 模型构建/反序列化、动态形状、DLA/精度设置、CUDA 缓冲与 Graph 执行，并转换输出供公共解码。 |
| [src/todrt/src/backend/engine_rknn.cpp](src/todrt/src/backend/engine_rknn.cpp) | 加载 RKNN 模型，处理 NPU 核选择和上下文、uint8 NHWC 输入及 float32 反量化输出。 |
| [src/todrt/src/backend/engine_ort.cpp](src/todrt/src/backend/engine_ort.cpp) | 创建 ORT session，选择执行设备、解析动态输入输出形状，管理 CPU 张量缓冲并执行推理。 |
| [src/todrt/src/backend/engine_openvino.cpp](src/todrt/src/backend/engine_openvino.cpp) | 加载/编译 OpenVINO 模型，配置设备与编译缓存，创建推理请求并提供统一 runner 接口。 |
| [src/todrt/src/backend/engine_stub.cpp](src/todrt/src/backend/engine_stub.cpp) | 在没有 SDK 后端时提供占位 builder 与诊断，使公共装配和自检仍可独立使用。 |
| [src/todrt/apps/todrt_cli.cpp](src/todrt/apps/todrt_cli.cpp) | 命令行应用；支持目录清单、设备探测、配置/装配检查、推理描述和性能测试。 |
| [src/todrt/tests/cpp_smoke.cpp](src/todrt/tests/cpp_smoke.cpp) | 验证注册、配置优先级、工厂装配、输入契约、前处理、解码与 NMS；公共检查不依赖 SDK。 |
| [src/todrt/CMakeLists.txt](src/todrt/CMakeLists.txt) | 定义 C++17 库、CLI 和自检目标，独立探测四种 SDK，处理架构/指令集、链接和安装导出。 |
| [src/todrt/cmake/aarch64-linux-gnu.cmake](src/todrt/cmake/aarch64-linux-gnu.cmake) | 配置 Linux aarch64 交叉编译器、sysroot 和依赖查找范围，用于 Jetson/RK3588 等目标。 |
| [src/todrt/cmake/todrtConfig.cmake.in](src/todrt/cmake/todrtConfig.cmake.in) | 安装包配置模板；使下游 CMake 项目可通过 find_package(todrt) 导入目标。 |
| [src/todrt/cmake/todrt.pc.in](src/todrt/cmake/todrt.pc.in) | pkg-config 元数据模板，提供安装后的头文件、库路径、链接选项及已编译后端信息。 |

### 仓库辅助脚本

| 文件 | 用途与调用关系 |
|---|---|
| [scripts/sh/gitpush.sh](scripts/sh/gitpush.sh) | Bash 推送助手；检测当前分支/远端，首次推送设置 upstream。执行会推送远端，不属于训练流水线。 |
| [scripts/bat/gitpush.bat](scripts/bat/gitpush.bat) | Windows 批处理推送助手；设置脚本内代理后检查分支/远端并推送，首次推送设置 upstream。 |

### 配置、变体产物与文档

| 文件 | 用途 |
|---|---|
| [pyproject.toml](pyproject.toml) | Python 包元数据、依赖入口、源码发现和 Ruff 规则；完整训练环境另按 requirement.txt 安装。 |
| [requirement.txt](requirement.txt) | 训练/数据处理/验证依赖清单；torch 与 torchvision 按所需 CUDA/CPU 索引单独安装。 |
| [configs/datasets.yaml](configs/datasets.yaml) | 入库数据集注册表，统一公共数据根目录、机型、源数据位置与训练数据位置；本机覆盖放 configs/datasets.local.yaml。 |
| [visdrone2019-det.yaml](configs/_base_/datasets/visdrone2019-det.yaml) | VisDrone 检测类别、数据划分与尺寸评测口径模板。 |
| [visdrone2019-pose.yaml](configs/_base_/datasets/visdrone2019-pose.yaml) | 航拍人体 17 点模板，包含关键点顺序和镜像索引；真实标注尚待准备。 |
| [rflysim-pose.yaml](configs/_base_/datasets/rflysim-pose.yaml) | UAV 9 点模板、镜像索引与几何监督字段；相机内参与机体点需换成真实配置。 |
| [_mav6d_mavic2_calib.json](configs/_base_/datasets/_mav6d_mavic2_calib.json) | Mavic 2 标定/机体几何预设，供转换器按机型解析。 |
| [_mav6d_phantom4_calib.json](configs/_base_/datasets/_mav6d_phantom4_calib.json) | Phantom 4 标定/机体几何预设，供转换器生成 dataset.yaml 与位姿数据。 |
| 各 variants/&lt;name&gt;/variant.yaml | 配方物化后的数据、EP 和训练配置；训练入口直接读取。 |
| 各 variants/&lt;name&gt;/model.yaml | 配方物化后的网络结构，仅负责建图。 |
| 各 variants/&lt;name&gt;/card.md、paper-notes.md | 分别记录变体身份/验证状态与论文证据/实现差异；evidence/ 保存参考材料。 |
| [WORKFLOW.md](WORKFLOW.md) | 建环境、准备数据、训练、评测、消融和导出的操作步骤。 |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 当前实际目录、模块边界、训练/部署调用链与关键数据契约。 |
| [docs/RGB_UAV_DETECTION_SURVEY.md](docs/RGB_UAV_DETECTION_SURVEY.md) | RGB 目标无人机检测、鸟机辨识、视频与机体姿态文献，附创新点、来源和 TOD 复刻路线。 |
| [PLAN.md](PLAN.md) | 研究选型、扩展点设计、评测协议与后续路线；当前实现以架构文档和文件索引为准。 |
| [docs/VARIANTS.md](docs/VARIANTS.md) | 由 catalog 工具生成的模块清单及元数据。 |
| [docs/KEYPOINTS.md](docs/KEYPOINTS.md) | 9 点编号、镜像关系、几何监督字段和指标定义。 |
| [docs/MAV6D.md](docs/MAV6D.md) | 作者原始数据到训练产物的格式、坐标与转换约定。 |
| [docs/MAV6D_TRAINING_FLOW.md](docs/MAV6D_TRAINING_FLOW.md) | MAV6D 数据流、监督和损失接线，以及当前训练假设与边界。 |
| [docs/TRAINING_REVIEW.md](docs/TRAINING_REVIEW.md) | 训练、位姿、单位与内参问题的核对记录和限制。 |
| [docs/DEPLOY.md](docs/DEPLOY.md) | Linux 部署步骤、SDK/CMake 选项和设备配置。 |
| [src/todrt/README.md](src/todrt/README.md) | C++ API、工厂用法、阅读顺序、后端与构建参考。 |

## 评测与消融（可归因对比）

```text
# 0) 合成数据集：几秒钟造一份"训练回路自检"数据（离线、确定性）
python3 tools/make_dummy_dataset.py --out tests/.tmp/tiny-detect

# 1) 训练回路自检：真训练 + 验证 + 存/载权重 + 推理（1–2 分钟）
python3 tests/train_smoke.py --epochs 2 --device 0

# 2) 尺度分层评测：整体 AP **与** AP_small / AP_tiny 一起报（只看整体会掩盖小目标退化）
python3 tools/val.py --variant variants/SDD-YOLO26n/variant.yaml `
    --weights results/SDD-YOLO26n/weights/best.pt `
    --data visdrone2019-det --imgsz 1024 `
    --json variants/SDD-YOLO26n/results.json

# 3) 消融流水线：先看网格与逐字段 diff（秒级、不训练），确认无误再真跑
python3 tools/ablation.py --recipe variants/SDD-YOLO26n/recipe.py --plan
python3 tools/ablation.py --recipe variants/SDD-YOLO26n/recipe.py `
    --data configs/_base_/datasets/visdrone2019-det.yaml --epochs 100 --imgsz 1024 --batch 4
```

> `tools/ablation.py` 的 `--data` 仍按**路径**收（它直接读 YAML 内容）；
> 换盘后写 `<data_root>/...` 的路径即可，`configs/_base_/datasets/*.yaml` 里的相对
> `path` 不会被框架正确解析（详见 WORKFLOW §4.0）。

> 消融的两条硬规矩（都写进了代码）：① 组件的"关掉"动作不能一律用 `without` ——
> 由**框架继承**来的能力（STAL / MuSGD）删键只会退回框架默认（8.4 的默认 TAL 本身就带
> 小目标先验），必须显式切到经典替代物；② `v.without()` 覆盖 EP、model、train 三段，
> 并会自动识别"剔掉的键本来就是默认值"这种假消融。

## 部署（推理端）

```text
# 导出 ONNX + 部署配置（需要 torch/ultralytics）
python3 tools/export_onnx.py --variant variants/SPAE-YOLOv8n/variant.yaml `
    --weights results/SPAE-YOLOv8n/weights/best.pt --imgsz 640 `
    --deploy-config configs/deploy/spae-yolov8n.json
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
python3 tools/convert_rknn.py --onnx exports/SPAE-YOLOv8n.onnx --target rk3588 \
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
