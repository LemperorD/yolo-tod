# yolo-tod —— 小目标检测魔改 YOLO 收集库

收集、复现并**可归因地**对比针对小目标检测（Tiny/Small Object Detection）的 YOLO 魔改。
完整方案见 [`PLAN.md`](PLAN.md)：架构设计、值得收录的魔改清单、优先级、评测协议。

**当前状态：M0 骨架已完成（`tests/smoke.py` 66 项全绿，`tests/test_modules.py` 105 项全绿）；
两个论文级变体已实现 —— SPAE-YOLOv8n 与 SDD-YOLO26n（结构自检全通）；
训练回路已用**合成数据**端到端跑通（`tests/train_smoke.py`：训练/验证/EMA/存权重/载权重/推理），
并补齐了**尺度分层评测**（`tools/val.py`：整体 AP + AP_small/AP_tiny）与
**消融流水线**（`tools/ablation.py`：leave-one-out + 逐字段 diff + 汇总表）。
真实数据集上的精度数字待 M1。推理端（C++/TensorRT）工厂化骨架已落地：
`src/todrt/`，CPU 自检 107 项全绿。**

- 主干框架：**ultralytics**（AGPL-3.0；本库实测区间 8.2 → **8.4**，8.4 才自带
  `end2end`/`MuSGD`/TAL 小目标先验，SDD-YOLO 依赖这三项）
- 第一主战场：**VisDrone2019-DET**（航拍小目标）
- 本机：RTX 5060 Laptop **8 GB** ／ Python 3.14.5 → 建议另建 **Python 3.12** 虚拟环境
- 部署端：**Linux**，四个可选后端 —— TensorRT（NVIDIA / Jetson DLA）、**RKNN（RK3588 NPU）**、
  **ONNX Runtime（通用 CPU，AMD x86 主力）**、**OpenVINO（x86 CPU / Intel iGPU）**；
  见 [`docs/DEPLOY.md`](docs/DEPLOY.md)

## 已实现的变体

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

```powershell
# 1) 建环境（强烈建议 3.12，torch 轮子最稳）
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]

# 2) 架构层冒烟测试（不需要 GPU / torch）
python tests\smoke.py

# 3) 查看已登记的魔改模块总表
python tools\catalog.py
```

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
├─ compose.py       变体 DSL + 模型图改造（P2 注入 / 下采样替换 / 类型替换）
├─ compat.py        唯一触碰 ultralytics 内部的地方（含只读配置目录回退）
├─ runtime.py       变体 spec 的运行时上下文
├─ modules/
│  ├─ conv/adown.py            ADown（EP1）
│  ├─ attention/dual_attention.py DualAttention（EP4，SDD-YOLO §4.5）
│  └─ head/efficient_uavdet.py Efficient_UAVDet（EP5）
├─ assigner/stal.py  STAL 小目标感知分配（EP6，可消融开关）
├─ optim/musgd.py    MuSGD（EP9，Muon 式 NS 正交化 + SGD 分量）
├─ loss/box.py + criterion.py  SIoU / Wise-IoU v3 与训练准则接入（EP7）
├─ eval/scales.py    尺度分层评测（整体 AP + AP_small/AP_tiny，COCO 式 101 点插值）
└─ engine/           trainer.py（EP 接线）/ surgery.py（EP4·EP5 建模后手术）
                    / distill.py（EP9 特征对齐蒸馏）

src/todrt/          部署侧（C++17，四个后端可选）—— 与 src/tod 并列，可独立复用
├─ include/todrt/   core（注册表）/ json / modules / factory / backend
├─ src/models/      ★ 变体配方（一个变体一个文件，两行宏）
├─ src/             factory / preprocess（float 或 uint8）/ decode / nms / config_io
├─ src/backend/     四后端：engine_trt / engine_rknn / engine_ort / engine_openvino
│                   + backend_dispatch（唯一分派处）+ simple_detector（共用外壳）
├─ apps/todrt_cli   部署工具（list/info/dryrun/probe/bench/run）
└─ tests/cpp_smoke  架构自检（**不需要任何后端**，148 项）

configs/_base_/     数据集等基础配置
configs/deploy/     部署配置（由 tools/export_onnx.py 生成，Python 与 C++ 的唯一契约）
variants/<name>/    recipe.py（代码定义）+ variant.yaml + model.yaml + card.md + paper-notes.md
tools/              make_variant.py / train.py / val.py / ablation.py / catalog.py
                    / make_dummy_dataset.py / export_onnx.py / convert_rknn.py
docs/DEPLOY.md      部署流程与四个后端（TRT / RKNN / ORT / OpenVINO）的实操与坑
docs/VARIANTS.md    模块总表（由 tools/catalog.py 自动生成）
tests/              smoke.py（无 torch 依赖，66 项）+ test_modules.py（形状/数值/手术/蒸馏/
                    优化器/评测/端到端，105 项）+ train_smoke.py（合成数据跑真训练，opt-in）
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
