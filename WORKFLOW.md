# yolo-tod 全流程命令

从**拉取仓库**到**配置 venv**、**准备数据**、**训练**、**测试**的完整命令序列。
每条命令都写明「参数从哪来 / 会产出什么文件」，**Windows 与 Linux 通用，可直接复制执行**。

> 全部命令在**仓库根目录**执行。别用 conda：本仓库走 `python3 -m venv`。

## 0.0 两个平台通用的写法约定

**解释器写 `python3`。** 两个平台都按这个口径写命令，Windows 上要额外做一步：

| 平台 | 情况 | 怎么办 |
|---|---|---|
| Linux / macOS | `python3` 一般就是系统 Python | 直接用 |
| Windows | 只有 `python`，**没有 `python3`**；如果你在终端敲 `python3` 被跳转到微软商店或什么都不输出，那是 Microsoft Store 的**占位假壳（0 字节，exit 9009）** | 让 `python3` 指向真解释器，见下 |

Windows 上让 `python3` 可用的两个办法（任选其一，一次性配置）：

```text
# 办法 A（推荐）：安装 Python Launcher for Unix，它提供真正的 python3 命令
winget install BrettCannon.PythonLauncher
# 装完新开一个终端，python3 --version 应能输出版本号

# 办法 B：自己做一个 python3.cmd 转发到 python（放到 PATH 里靠前的目录即可）
'@python "%~dp0..\python.exe" %*' | Set-Content -Encoding ASCII "$env:USERPROFILE\bin\python3.cmd"
# 之后把 $env:USERPROFILE\bin 加到 PATH（或用 tools/setup_venv.ps1 的检查确认）
```

> 验证：`python3 --version` 能打印 3.x 版本号即可；打印不出来或弹出商店页面 = 还是假壳。
> 不想配置也行：本文档命令里的 `python3` 在 Windows 上**一律替换成 `python`**，其余不变。

**路径一律用正斜杠 `/`。** Windows 的 Python 与 PowerShell 都接受正斜杠（本仓库
`tools/` 下所有脚本已实测跑通），因此 `/` 是唯一能两边通用的分隔符：

```text
python3 tools/train_mav6d.py --model phantom4          # 两个平台都对
python3 tools/train_mav6d.py --model phantom4          # 只有 Windows 对，Linux 会当成一个奇怪的文件名
```

**跨平台的两处写法差异**（只有这两处）：激活虚拟环境、以及带空格路径的引用。

```text
# Windows PowerShell
.\.venv\Scripts\Activate.ps1        # 激活
python3 -c "print('hi')"            # 双引号包 -c 的代码
```

```bash
# Linux / macOS
source .venv/bin/activate           # 激活
python3 -c 'print("hi")'            # 单引号包 -c 的代码
```

**不激活也能用**（本文档所有命令都适用，也是 CI 里的推荐做法）：

```
.venv/Scripts/python3.exe           # Windows（若用 python 建的 venv 就是 python.exe）
.venv/bin/python3                   # Linux / macOS
```

---

## 0. 前置检查（1 分钟内）

```text
git --version                 # 需要 git
python3 --version             # 需要 3.10 ~ 3.14（注意 3.14.1 不可用，见下）
nvidia-smi                    # 需要 GPU 时；右上角 "CUDA Version" 决定 torch 索引
```

| 检查项 | 要求 | 说明 |
|---|---|---|
| Python | **3.10 ~ 3.14** | `pyproject.toml` 的 `requires-python = ">=3.10"`。**唯一要避开 3.14.1**：torchvision 的 `requires-python` 里明确写了 `!=3.14.1`。本机实测 3.13.13 与 3.12.14 都可用，3.14.5 也有 cu128 轮子。 |
| 显卡 | 可选 | 没有 GPU 也能跑通全部测试与结构自检，只是训练极慢（`--device cpu`）。 |
| 磁盘 | ≈ 6 GB | torch+torchvision（cu128）约 4.5 GB，依赖约 1 GB。数据另计（MAV6D 转换产物约 20 GB）。 |
| 网络 | 必需 | torch 的 GPU 轮子要访问 `download.pytorch.org`；国内可用镜像（见「常见问题」）。 |
| 系统 | Windows / Linux 均可 | 文档命令两边通用；两个平台各有一个建环境脚本（见 2.4）。 |

---

## 1. 拉取仓库

```text
git clone <你的远端地址> yolo-tod
cd yolo-tod
```

如果仓库里已有本地改动，先确认状态：

```text
git status --short
```

---

## 2. 配置 venv（不使用 conda）

### 2.1 建虚拟环境

```text
python3 -m venv .venv
.\.venv\Scripts\Activate.ps1          # Linux/macOS: source .venv/bin/activate
python3 -m pip install --upgrade pip
```

> 用别的解释器建：Windows 上 `py -3.12 -m venv .venv`（`py -0p` 列出本机所有版本）；
> Linux 上 `python3.12 -m venv .venv`。
> Windows 上激活脚本被执行策略拦住时：`Set-ExecutionPolicy -Scope Process RemoteSigned`，
> 或者不激活、全程用 `.venv/Scripts/python3.exe`（本文档的命令都适用）。

### 2.2 先装 torch（你自己选 CUDA 版本）

**PyPI 上的 torch 是 CPU-only 构建**，GPU 版只在 PyTorch 自己的索引上，所以这一步必须指定索引：

```text
# 把 cu128 换成你需要的版本；纯 CPU 就换成 cpu
python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

选哪个版本：

| 你的情况 | 索引 |
|---|---|
| RTX 50 系（Blackwell，如本机 RTX 5060） | `cu128` 及以上（**不能**用 cu118/cu126） |
| RTX 30/40 系 | `cu126` / `cu128` 都行 |
| 老卡 / 驱动旧 | `cu118` |
| 没有 NVIDIA 显卡 | `.../whl/cpu`，或直接 `pip install torch torchvision` |
| 不确定 | `nvidia-smi` 看 "CUDA Version"，或 <https://pytorch.org/get-started/locally/> |

装完立刻自检（**这步必做**，装成 CPU 版是最常见的坑）：

```text
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')"
```

> Linux/macOS 上把双引号换成单引号：`python3 -c 'import torch; print(torch.cuda.is_available())'`

期望：版本号带 `+cu128` 后缀，`cuda.is_available()` 为 `True`。
若为 `False` 或版本号没有 `+cuXXX`：`python3 -m pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128`

### 2.3 装其余依赖 + 本仓库

```text
python3 -m pip install -r requirement.txt          # 必需内容（见该文件的注释）
python3 -m pip install -e . --no-deps              # 本仓库可编辑安装
```

> 顺序有意义：**先 torch 后 requirement.txt**。反过来 pip 会在装 `ultralytics`
> 时去 PyPI 拉一个 CPU 版 torch 把 GPU 版盖掉。
> `--no-deps` 是因为依赖已由前两步装齐，避免 pip 再去解析一遍索引。

### 2.4 一键脚本（等价于 2.1~2.3 + 自检）

两个平台各有一个脚本，行为一致：

```text
# Windows
powershell -ExecutionPolicy Bypass -File tools/setup_venv.ps1
# 换 CUDA 版本 / 换解释器：
powershell -ExecutionPolicy Bypass -File tools/setup_venv.ps1 -TorchIndex https://download.pytorch.org/whl/cu126
powershell -ExecutionPolicy Bypass -File tools/setup_venv.ps1 -PythonVersion 3.12
```

```bash
# Linux / macOS
bash tools/setup_venv.sh
# 换 CUDA 版本 / 换解释器：
bash tools/setup_venv.sh --torch-index https://download.pytorch.org/whl/cu126
bash tools/setup_venv.sh --python python3.12
```

脚本依次做：建 `.venv` → 引导 pip → 从 `-TorchIndex` 装 torch/torchvision →
装 `requirement.txt` → 可编辑安装本仓库 → 打印 torch 版本、CUDA 可用性、跑 `tests/smoke.py`。

> 脚本刻意写成**纯 ASCII**：Windows PowerShell 5.1 用 ANSI 代码页（中文机器上是 936/GBK）
> 解析无 BOM 的 UTF-8 脚本，中文注释会被解坏并导致语法错误。

---

## 3. 验证环境（不需要数据集，1–3 分钟）

按「从快到慢 / 从浅到深」的顺序跑：

```text
# 1) 架构层自检：不需要 torch，纯配置/注册表/构图检查（秒级，77 项）
python3 tests/smoke.py

# 2) 模块与损失层：需要 torch，105 项
python3 tests/test_modules.py

# 3) 关键点/6D 位姿/L_rep：220 项
python3 tests/test_pose.py

# 4) 变体物化（生成 variant.yaml / model.yaml / card.md）
python3 tools/make_variant.py variants/spae-yolov8n-pose/recipe.py

# 5) 结构自检：建模型 + 前向一次，不训练（几秒）
python3 tools/train.py --variant variants/spae-yolov8n-pose/variant.yaml --dry-run
```

期望：`全部通过：77 项检查` / `通过 105 项，跳过 0 项，失败 0 项` / `通过 220 项，跳过 0 项，失败 0 项`。

> 真实数据端到端自检（可选，需要 `D:/MAV6D`）：
> `python3 tests/test_mav6d.py`（默认 `--mav6d D:/MAV6D`，49 项；没有数据就 `--skip-real`）

**训练回路真训练自检**（可选，含 6D 位姿与 ADD 断言，需 GPU 或 `--device cpu`）：

```text
python3 tests/train_pose_smoke.py --device 0      # 1 epoch，几分钟
python3 tests/train_smoke.py   --epochs 2 --device 0   # 检测回路
```

---

## 4. 准备数据

### 4.1 路线 A：MAV6D（本仓库已跑通，推荐）

原始数据放在 `D:/MAV6D`（作者公开数据集；`phantom4` 与 `mavic2` 两个机型）。

```text
# 0) 先核对投影链：用作者公开标定逐值比对（实测 0.0004 px / 300 帧）
python3 tools/mav6d_calibrate.py --mav6d D:/MAV6D --model phantom4 --check-util

# 1) 转换：→ YOLO-pose（检测框 + 9 点关键点）+ labels_6d 6D 位姿 + dataset.yaml
python3 tools/mav6d_to_yolo.py --mav6d D:/MAV6D --model phantom4 --out data/mav6d
python3 tools/mav6d_to_yolo.py --mav6d D:/MAV6D --model mavic2   --out data/mav6d

# 2) 开训前校验（必做；有问题会在开训前报出来而不是训到一半才炸）
python3 tools/check_pose_dataset.py --data data/mav6d/phantom4/dataset.yaml
python3 tools/check_pose_dataset.py --data data/mav6d/mavic2/dataset.yaml
```

产物（实测规模）：

```
data/mav6d/phantom4/
├─ images\train|val\        # 10 622 / 1 052 帧（按序列切分，硬链接原图，不占额外空间）
├─ labels\train|val\        # YOLO-pose：cls + 框 + 9×(x,y,v)
├─ labels_6d\train|val.csv  # 6D 位姿旁挂数据（L_rep 用）
├─ dataset.yaml             # 含 kpt_shape / flip_idx / tod.object_points / tod.camera_matrix
└─ manifest.json            # 转换清册（可复算）
```

`mavic2` 为 11 122 / 666 帧。两个机型都 100% 带有效 6D 位姿。

> 图像是**硬链接**（`--link`，默认）：删产物时注意别删到原始 `D:/MAV6D` 的图；
> 需要真复制用 `--copy`。想先小规模试跑可加 `--max-frames 500`。

### 4.2 路线 B：RflySim / 自己的数据（论文口径的 9 点）

```text
# 生成模板数据集与配置（UAV 版式 = 1 个质心 + 8 个角点的 9 点定义）
python3 tools/make_dummy_dataset.py --task pose --layout uav --out tests/.tmp/pose-dummy

# 换成你自己的数据：按 configs/_base_/datasets/rflysim-pose.yaml 的格式准备
python3 tools/check_pose_dataset.py --data configs/_base_/datasets/rflysim-pose.yaml
```

> `configs/_base_/datasets/rflysim-pose.yaml` 里的 `tod.object_points` 与
> `tod.camera_matrix` 是**占位值**，必须换成你自己机型的真实几何（论文没给这两张表）。
> 关键点定义与 `flip_idx` 见 `docs/KEYPOINTS.md`。

### 4.3 路线 C：VisDrone2019-DET（纯检测）

数据集需自行下载；配置已备好：

```text
python3 tools/train.py --variant variants/SPAE-YOLOv8n/variant.yaml `
    --data configs/_base_/datasets/visdrone2019-det.yaml --dry-run   # 先看结构
```

> `visdrone2019-pose.yaml` 目前是**接口占位**（VisDrone 没有关键点标注），
> 在标注到位前只能跑结构自检与合成数据回路自检。

---

## 5. 训练

### 5.1 MAV6D（小目标检测 + 9 点关键点）

```text
# 1) 冒烟：2 epoch / 640 / batch 2，只确认数据、损失、显存都没问题
python3 tools/train_mav6d.py --model phantom4 --smoke

# 2) 正式训练
python3 tools/train_mav6d.py --model phantom4 --epochs 100 --batch 4 --imgsz 1280

# 3) 另一机型（论文 Table IV 显示跨机型零样本很差，建议分别训）
python3 tools/train_mav6d.py --model mavic2 --epochs 100 --batch 4 --imgsz 1280

# 4) 显存不够时的两个旋钮
python3 tools/train_mav6d.py --model phantom4 --batch 2 --imgsz 960
python3 tools/train_mav6d.py --model phantom4 --set workers=0            # Windows 上更稳

# 5) 断点续训
python3 tools/train_mav6d.py --model phantom4 --resume results/mav6d-phantom4/weights/last.pt
```

默认超参（都可覆盖）：`imgsz=1280 / batch=4 / epochs=100 / mosaic=0 / degrees=10 / scale=0.4 /
fliplr=0.5 / pose=12 / kobj=1 / amp=True / workers=4 / seed=0`。

为什么 1280 而不是 640：MAV6D 目标只占画面长边 10–17%（1920 下约 190–320 px），
640 会把 190 px 的目标缩到 63 px，关键点定位精度直接损失。
`mosaic=0` 是因为拼接会改变尺度分布，与关键点回归的尺度一致性冲突。

产物：

```
results/mav6d-phantom4/
├─ weights\best.pt / last.pt
├─ results.csv               # 逐 epoch 的损失与指标
└─ args.yaml                 # 本次训练的全部参数（复现依据）
```

### 5.2 其他变体（检测）

```text
# 物化变体（生成 model.yaml / variant.yaml / card.md）
python3 tools/make_variant.py variants/SDD-YOLO26n/recipe.py

# 结构自检 → 真训练
python3 tools/train.py --variant variants/SDD-YOLO26n/variant.yaml --dry-run
python3 tools/train.py --variant variants/SDD-YOLO26n/variant.yaml `
    --data configs/_base_/datasets/visdrone2019-det.yaml `
    --epochs 100 --batch 4 --imgsz 1024
```

产物在 `results/<变体 id>\weights\best.pt`。
`tools/train.py` 每次会用 recipe 现场重生成 `model.yaml`，保证模型图与变体配置永不脱节。

### 5.3 训练期注意

- 中断后想接着跑用 `--resume`（MAV6D 脚本）或 `--set resume=True`。
- `--no-pretrained` 从零训练，便于消融「预训练带来的增益」。
- 关掉不该动的增强：`--set mosaic=0 fliplr=0`（`flip_idx` 缺失时框架会静默关掉 fliplr，
  本仓库的数据集 YAML 已显式给出 `flip_idx`，所以它真的生效）。

---

## 6. 测试与评测

### 6.1 MAV6D 一键全指标（推荐）

```text
# 自动找 results\ 下该机型最新的 best.pt
python3 tools/test_mav6d.py --model phantom4

# 两个机型一起报（论文 Table I/IV 就是分机型报的）
python3 tools/test_mav6d.py --model both

# 指定权重 / 划分 / 输出 JSON（JSON 里带 git commit 与全部口径说明，便于归档）
python3 tools/test_mav6d.py --model phantom4 --weights results/mav6d-phantom4/weights/best.pt `
    --split val --json results/mav6d-phantom4/test.json

# 快速冒烟：只测前 200 张
python3 tools/test_mav6d.py --model phantom4 --max-images 200
```

打印的汇总表列：`机型 | OKS-AP | OKS-AP50 | AP50:95(框) | ADD(m) | ADD(.5d) | 10°10cm`。
其中 **ADD / ADD(0.5d) / 10°10cm 走 PnP，与论文同口径**（判定阈值 10° 且 10 cm）。

### 6.2 通用评测

```text
# 检测：整体 AP + AP_small / AP_tiny 分层
python3 tools/val.py --weights results/SDD-YOLO26n/weights/best.pt `
    --data configs/_base_/datasets/visdrone2019-det.yaml --imgsz 1024 `
    --json variants/SDD-YOLO26n/results.json

# 姿态：OKS-AP 50:95 + 尺度分层 + 逐关键点诊断
python3 tools/val_pose.py --weights results/mav6d-phantom4/weights/best.pt `
    --data data/mav6d/phantom4/dataset.yaml --imgsz 1280 --sigma-strategy uniform

# 只测前 N 张，先确认口径通了
python3 tools/val_pose.py --weights results/mav6d-phantom4/weights/best.pt `
    --data data/mav6d/phantom4/dataset.yaml --max-images 100
```

> **只看整体 OKS-AP 会掩盖小目标退化**：OKS 的分母含框面积，3 px 误差在 120 px 人体上
> OKS≈0.95、在 8 px 行人上 OKS≈0.2。所以评测一律带尺度分层。
> σ 策略可选 `uniform / auto / person / framework / balanced`（MAV6D 的 9 点用 `uniform`）。

### 6.3 消融（可归因对比）

```text
# 先看网格与逐字段 diff（秒级，不训练），确认无误再真跑
python3 tools/ablation.py --recipe variants/SDD-YOLO26n/recipe.py --plan

python3 tools/ablation.py --recipe variants/SDD-YOLO26n/recipe.py `
    --data configs/_base_/datasets/visdrone2019-det.yaml --epochs 100 --imgsz 1024 --batch 4
```

产物：`results/ablations/<变体>/summary.md` + `summary.csv`。

---

## 7. 导出部署（可选）

```text
# ONNX + 部署配置（Python 与 C++ 侧的唯一契约）
python3 tools/export_onnx.py --variant variants/SPAE-YOLOv8n/variant.yaml `
    --weights results/SPAE-YOLOv8n/weights/best.pt --imgsz 640 `
    --deploy-config configs/deploy/spae-yolov8n.json
```

C++/TensorRT 侧（Linux）见 `docs/DEPLOY.md` 与 `src/todrt/README.md`。

---

## 8. 命令速查表

| 目的 | 命令 |
|---|---|
| 建环境（Linux/macOS） | `bash tools/setup_venv.sh` |
| 建环境（Windows） | `powershell -ExecutionPolicy Bypass -File tools/setup_venv.ps1` |
| 装 torch（自己选 CUDA） | `python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128` |
| 装其余依赖 | `python3 -m pip install -r requirement.txt` |
| 本仓库安装 | `python3 -m pip install -e . --no-deps` |
| GPU 自检 | `python3 -c "import torch; print(torch.cuda.is_available())"` |
| 架构自检 | `python3 tests/smoke.py` |
| 模块/姿态自检 | `python3 tests/test_modules.py` / `python3 tests/test_pose.py` |
| 数据转换 | `python3 tools/mav6d_to_yolo.py --mav6d /path/to/MAV6D --model phantom4 --out data/mav6d` |
| 数据校验 | `python3 tools/check_pose_dataset.py --data data/mav6d/phantom4/dataset.yaml` |
| 训练 | `python3 tools/train_mav6d.py --model phantom4 --epochs 100 --batch 4 --imgsz 1280` |
| 测试 | `python3 tools/test_mav6d.py --model both` |
| 消融计划 | `python3 tools/ablation.py --recipe variants/SDD-YOLO26n/recipe.py --plan` |

---

## 9. 常见问题

**pip 装得很慢 / 超时。** torch 的 GPU 轮子只有官方索引有，PyPI 镜像上没有 GPU 版。
国内可用 `--index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu128`
（实测可解析到 `torch 2.11.0+cu128`）；其余的 PyPI 依赖可加
`-i https://pypi.tuna.tsinghua.edu.cn/simple`。注意两者不要混在一条命令里。

**`torch.cuda.is_available()` 是 False。** 99% 是装了 CPU 版：检查 `torch.__version__`
有没有 `+cuXXX`。修复：`python3 -m pip install --force-reinstall torch torchvision --index-url <你的索引>`。

**`python3 -m venv .venv` 报 ensurepip 失败。** 受限/被管控的终端会拦住 pip 往临时目录写文件。
脚本会自动改走 `--without-pip` 并尝试用系统 pip 引导；仍失败时请换一个普通终端（不要用受限沙箱会话）。

**Windows 上标签缓存报 `PermissionError [WinError 5]`。** ultralytics 的标签缓存用命名管道
（`multiprocessing.Pool`），受限的终端/沙箱会话里会失败。`tools/train_mav6d.py` 默认已改用
顺序缓存；通用脚本可加 `--set workers=0`（多数情况够用），或给 `tools/train.py` 加
`--allow-threaded-cache` 的反面——即**不要**传这个开关（默认就是顺序缓存）。

**显存不够（8 GB 卡）。** 依次降：`--batch 2` → `--imgsz 960` → `--model` 换小骨架。
1280 + P2 头是显存大头。

**`git status` 里 `results/` 一大堆权重。** `.gitignore` 本意是 `results/*` 只保留 `.gitkeep`，
但 `git add -A --force`（本仓库的自动提交脚本用的就是 force）会无视忽略规则把权重一起提交。
清理：`git rm -r --cached results` 后确认 `.gitignore` 生效。
