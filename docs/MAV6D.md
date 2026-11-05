# MAV6D 数据集架构（源数据 → 本库训练产物）

> 本文只讲**数据长什么样、字段怎么解释、怎么进训练**。训练代码的 P0 修复与审查见
> [`TRAINING_REVIEW.md`](TRAINING_REVIEW.md)；关键点定义见 [`KEYPOINTS.md`](KEYPOINTS.md)。

## 1. 一句话架构

MAV6D 是**真实室内 VICON** 采集的无人机位姿/关键点数据：每帧一张 1920×1080 图，
标注给 9 个机体 3D 点（1 个 3D 框中心 + 8 个角点）的 2D 投影，以及相机/机体的 6D 位姿。
本库把它转成 **YOLO-pose（检测框 + 9 点）+ 旁挂 6D 位姿 CSV**，挂到
`spae-yolov8n-pose` 变体上训练。

```text
原始 MAV6D                                  本库转换产物
<MAV6D>/<model>/                            <data_root>/mav6d/<model>/
├─ JPEGImages/<组>/<序列>/*.jpg      ──►     ├─ images/{train,val}/*.jpg     （软链/复制）
├─ labels/<组>/<序列>/*.txt         ──►     ├─ labels/{train,val}/*.txt     （YOLO-pose 32 列）
├─ labels_yolo6d/<组>/<序列>/*.txt  ──►     ├─ labels_6d/{train,val}.csv    （逐目标 6D 位姿）
├─ train.txt / test.txt                    ├─ dataset.yaml                  （kpt_shape / tod.*）
└─ （VICON 原始数据）                       └─ manifest.json                 （划分/过滤/标定统计）
```

## 2. 原始数据布局

```text
<MAV6D>/
└─ phantom4/  或  mavic2/
   ├─ JPEGImages/
   │   └─ 01/0101/1.jpg、1.jpg ...          # 组/序列/帧
   ├─ labels/
   │   └─ 01/0101/1.txt                     # 每行 16 个数（相机 8 + 机体 8）
   ├─ labels_yolo6d/
   │   └─ 01/0101/1.txt                     # 每行 21 列（cls + 9×2 + w,h）
   ├─ train.txt
   └─ test.txt                              # 作者机器上的绝对路径清单
```

两机型必须**分别训练**：论文 Table IV 显示跨机型零样本很差；本库也按机型分目录。

## 3. 四种文本格式（最容易错的地方）

### 3.1 原始 `labels/<组>/<序列>/<stem>.txt`：16 个数

```text
[0]       相机时间戳（纳秒，1.6e18 量级）
[1:4]     相机位置 (x, y, z)，VICON 系
[4:8]     相机四元数 (qx, qy, qz, qw)，相机→VICON
[8]       机体时间戳
[9:12]    机体位置 (x, y, z)，VICON 系
[12:16]   机体四元数 (qx, qy, qz, qw)，机体→VICON
```

> **这是本库在 2026-10-26 修正后的口径。** 旧代码把 `[1:8]` 当机体位姿、`[9:16]` 当相机位姿，
> 正好反了；用旧口径生成的 `labels_6d` 与 2D 关键点差 ~2400 px、旋转差 ~169°。
> 判断依据：`mav6d_calibrate.py::check_against_util()` 用 `[9:16]` 把机体 3D 点变换到 VICON、
> 再套 `UTIL_CAM2VICON`，在 300 帧上得到 **0.000376 px** 重投影误差。
> 因此 `[9:16]` 是**机体（MAV）位姿**，`[1:8]` 是**相机位姿**。

### 3.2 原始 `labels_yolo6d/<组>/<序列>/<stem>.txt`：21 列

```text
cls  x0 y0  x1 y1  ...  x8 y8  w h
```

- 9 个点 = 1 个 3D 框中心 p0 + 8 个角点 p1~p8（与论文 §III-A-1、作者 `util.py` 的 `corners` 一致）；
- `x,y` 的归一化基准固定 **1920×1080**（不是各图实际尺寸）；
- 末尾 `w,h` 是 9 点包围盒宽高；
- 作者 `unzip_tools.py` 报告里写“10 个归一化点”是错的，别按 10 点解析。

### 3.3 转换后 `labels/<split>/<stem>.txt`：32 个数

```text
cls  cx cy w h  x0 y0 v0  x1 y1 v1 ... x8 y8 v8
1  +  4       +  9 × 3 = 32
```

- `cx cy w h`：由**可见关键点**的包围盒得到，归一化到各图实际尺寸；
- `v=2`：该点在图内；`v=0`：越界/占位（框架的 `kpt_mask = v != 0`）；
- 2D 关键点已**去畸变**（与 `L_rep` 的针孔模型自洽），图像本身保持原始畸变图。

### 3.4 转换后 `labels_6d/<split>.csv`：9 列

```csv
image,obj_index,tx,ty,tz,qx,qy,qz,qw
01_0101_1,0,-2.09,-5.32,3.42,-0.72,0.067,-0.04,0.69
```

- `image`：图像文件名（不含扩展名）；
- `obj_index`：该图内目标序号（0 起），**必须与标签文件的行序一致**；
- `t`：机体→相机的平移（米）；`q`：机体→相机的旋转四元数 `(qx,qy,qz,qw)`；
- 这是 `L_rep` 的监督信号，不是可选附件。`pose6d.py` 会强校验
  “CSV 行数 == 标签行数”，`tools/check_pose_dataset.py` 开训前再查一遍。

## 4. 坐标与几何约定

| 量 | 约定 |
|---|---|
| 相机内参 `K` | 针孔，`fx=1979.4, fy=1979.1, cx=976.8, cy=534.0`（1920×1080） |
| 机体 3D 点 `object_points` | 第 0 行是 3D 框中心，1..8 是 8 个角点；单位米；顺序与 `kpt_names` 一致 |
| 位姿 `R,t` | `p_cam = R @ p_body + t`，即**机体→相机** |
| 四元数 | `(qx,qy,qz,qw)`，右手系，已归一化 |
| 2D 关键点 | 已去畸变；按各图实际尺寸归一化 |
| 翻转 | `flip_idx = [0,5,6,7,8,1,2,3,4]`；但启用 `L_rep` 时**不能直接开 `fliplr`**（位姿没有同步翻） |
| 模型直径 | `model_diameter`，用于 ADD(0.5d) 阈值 = `0.5 × diameter` |

## 5. 划分与统计（当前盘上）

| 机型 | train 图 | val 图 | train 序列 | val 序列 | 备注 |
|---|---:|---:|---:|---:|---|
| phantom4 | 10 622 | 1 052 | 53 | 6 | 按**序列**切分，`val_ratio≈0.1` |
| mavic2 | 11 122 | 666 | 47 | 5 | 同上 |

- **按序列切**，不是按帧随机切：同一段飞行的相邻帧高度相似，按帧切会泄漏；
- 两机型都全部带 `labels_6d`；phantom4 的框长边归一化约 `p50=0.172`、`p90=0.248`；
- 位置由 `configs/datasets.yaml` 的 `data_root` + `datasets.mav6d-<model>.root` 决定，
  换盘只改注册表，不改脚本。

## 6. 转换与校验命令

```powershell
# 1) 完整转换（有原始 MAV6D 时；会先清空 <out>/<model>）
D:\miniconda3\python.exe tools/mav6d_to_yolo.py `
  --mav6d D:/MAV6D --model phantom4 --out data/mav6d --val-ratio 0.1

# 2) 只补 dataset.yaml（图/标签已在盘上，标定取仓库预置 JSON）
D:\miniconda3\python.exe tools/mav6d_to_yolo.py `
  --model phantom4 --out data/mav6d --yaml-only

# 3) 原始 labels 不可用 / 旧 labels_6d 已损坏：由 2D 关键点 PnP 反解位姿
D:\miniconda3\python.exe tools/mav6d_to_yolo.py `
  --model phantom4 --out data/mav6d --pose-from-pnp

# 3b) Mavic 2 的机体宽高深比例与 Phantom 4 不同：先拟合各向异性尺度再反解
D:\miniconda3\python.exe tools/mav6d_to_yolo.py `
  --model mavic2 --out data/mav6d --pose-from-pnp --fit-pnp-shape

# 4) 开训前校验
D:\miniconda3\python.exe tools/check_pose_dataset.py `
  --data data/mav6d/phantom4/dataset.yaml
```

`--pose-from-pnp` 会用每帧的 9 点 2D 标签 + `K` + `object_points` 做
`cv2.solvePnP`，把重投影误差 >5 px 的帧跳过；跳过的帧在 `dataset.yaml` 里
用 `tod.pose_strict: false` 标出来，训练时 `L_rep` 会跳过它们，而不是报错。
旧 CSV 会备份为 `labels_6d/<split>.csv.bak`。

`--fit-pnp-shape` 只用 2D 标签拟合机体的 x/y/z 三轴尺度（不覆盖标定 JSON，
只更新 `dataset.yaml` 的 `tod.object_points` / `tod.model_diameter`）。
Mavic 2 实测 `scale_xyz≈[1.062, 1.246, 0.664]`，PnP 重投影从 ~28 px 降到 **0.06 px**；
不做这一步时 `L_rep` 会因 PnP 失败而几乎不生效。

当前重建结果：`phantom4` 覆盖率 **95.4%**（平均 0.11 px），`mavic2` **95.7%**
（平均 0.06 px）；`pose_strict: false` 是预期值，不是错误。

## 7. 训练入口与推荐超参

```powershell
# 2D baseline（先关 L_rep；几何增强可以开）
D:\miniconda3\python.exe tools/train_mav6d.py `
  --model phantom4 --no-lrep `
  --imgsz 1280 --batch 4 --epochs 100 --name mav6d-p4-2d

# 修好 labels_6d + L_rep 后再开几何项
D:\miniconda3\python.exe tools/train_mav6d.py `
  --model phantom4 `
  --imgsz 1280 --batch 4 --epochs 100 --name mav6d-p4-lrep

# 测试（论文指标）
D:\miniconda3\python.exe tools/test_mav6d.py `
  --model phantom4 --weights results/mav6d-p4-lrep/weights/best.pt --imgsz 1280
```

推荐值：`imgsz=1280`、`batch=4`、`epochs=100`（从零训练可到 200）、`mosaic=0`、
`close_mosaic=0`、`pose=12`、`kobj=1`、`amp=true`；`L_rep` 的 warm-up 保持 15。
显存实测：1280×4 峰值 2.86 GiB，1280×8 峰值 5.69 GiB（8 GB 卡可跑）。

> 在受限沙箱 / Windows 命名管道被禁的环境里，`workers>0` 会报
> `PermissionError [WinError 5]`；加 `--set workers=0` 即可（本库的标签缓存也会自动切顺序模式）。
> 普通本机终端可以保留 `workers=4`。

## 8. 数据侧验收清单

- [ ] `labels_6d` 与 `labels` 行数逐图一致（或 `pose_strict: false` 且缺的帧是 PnP 失败帧）；
- [ ] 任取 20 帧，用 CSV 位姿把 `object_points` 投影到图像，平均误差 <1 px；
- [ ] `check_pose_dataset.py` 通过；
- [ ] `dataset.yaml` 的 `tod.camera_matrix` / `tod.object_points` / `kpt_shape` 三者自洽；
- [ ] `flip_idx` 是长度 9 的对换置换；
- [ ] `mavic2` 与 `phantom4` 不混训（除非明确做跨机型实验）。
