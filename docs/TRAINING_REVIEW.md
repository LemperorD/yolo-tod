# 训练代码审查与训练方案（MAV6D / SPAE-YOLOv8n-pose）

> 审查时间：2026-10-26。对象：`variants/spae-yolov8n-pose` + `tools/train_mav6d.py`
> + `tools/train.py` + `src/tod/engine/pose_trainer.py` + `src/tod/loss/{pose,criterion,reproj}.py`。
>
> **修复状态（2026-10-26 实施）**：
> - P0-1 6D 位姿块/矩阵顺序：已修；`phantom4`/`mavic2` 的 `labels_6d` 已用 PnP 重建
>   （phantom4 覆盖率 95.4%、mavic2 95.7%）。
> - P0-2/3/4 `L_rep` 单位、锚点匹配、内参缩放、`fliplr`：已修；GT 点上 `L_rep≈0` 的回归测试已加。
> - P1 默认值覆盖、`--no-lrep`、smoke 触发 `L_rep`、L_rep 日志：已修。
> - `mavic2` 的 3D 框与 Phantom 4 比例不同，已加 `--fit-pnp-shape` 拟合各向异性尺度
>   （scale_xyz≈[1.062, 1.246, 0.664]，PnP 残差 0.06 px）。
>
> 下面是**审查时的原始分析**（保留作为修复理由与取证记录）。现在训练入口请用
> `docs/MAV6D.md` §6/§7 的命令；`EP7.reproj=true` 已可用，但仍建议先跑 2D baseline。
>
> **审查结论（当时）**：6D 位姿标签是错的，`L_rep` 的坐标单位、锚点匹配、内参缩放也都有错。
> 前 15 个 epoch 因为 warm-up 不启用 `L_rep`，所以 2 epoch 的 smoke 看起来“正常”；
> 一旦 warm-up 结束，几何项会把训练带向错误方向。先跑 2D baseline，修完 P0 再开 `L_rep`。

---

## 0. 先看三个事实

1. **仓库里唯一一次 recorded 训练是 2 epoch 的 smoke**
   （`results/mav6d-smoke/args.yaml`：`epochs=2`、`imgsz=640`、`data=tests/.tmp/mav6d-out/...`）。
   它的 `mAP50=0`、`mAP50-95=0` 是**正常现象**，不是训练失败的证据。
   这个 run 也不能验证 `L_rep`：`tod.reproj_warmup_epochs=15`，2 个 epoch 根本没走到 `L_rep`。

2. **6D 位姿 CSV 与 2D 关键点严重不一致**（见 P0-1 的实测表）：
   用 9 点 + 标定内参做 PnP，平均重投影 **0.216 px**；
   用 `labels_6d` 里的位姿投影同一批关键点，平均 **2435 px**、旋转差 **169°**。
   这不是小误差，是位姿块取错/矩阵乘反了。

3. **`L_rep` 的坐标单位也错了**：`criterion.kpts_decode()` 返回的是 **stride/网格单位**，
   不是 [0,1] 归一化坐标；`_one_branch()` 把它当归一化坐标再乘 `imgsz`，
   等于把像素放大了几十到一百多倍。锚点匹配同样拿 stride 单位去和归一化框比。

---

## 1. 训练链路：从命令到权重

```text
tools/train_mav6d.py  （或 tools/train.py）
│
├─ 1) 数据集定位
│    configs/datasets.yaml
│      → tod.paths.resolve()
│      → .cache/tod/datasets/<name>-<hash>.yaml（path 改成绝对路径的副本）
│      → 校验 train/val 图片数与标签数
│
├─ 2) 变体 spec / 模型图
│    variants/spae-yolov8n-pose/variant.yaml
│      → Variant.from_spec()
│      → variants/spae-yolov8n-pose/model.yaml（P2 / ADown / Pose 头）
│
├─ 3) 建模型
│    YOLO(model.yaml, task="pose")  → PoseModel
│
├─ 4) model.train(trainer=TODPoseTrainer, **train_args)
│    ├─ get_model()                 → apply_spec()：EP4/EP5 建模后手术
│    ├─ get_dataloader()            → 启用 L_rep 时把 YOLODataset 换成 Pose6DDataset
│    ├─ set_model_attributes()      → _install_criterion()
│    │    ├─ build_pose_criterion() → SIoU + TinyPoseLoss(OKS)
│    │    └─ build_reproj_criterion() → ReprojCriterion（L_rep 包装）
│    └─ build_optimizer()           → SGD / MuSGD
│
└─ 5) 训练循环（ultralytics BaseTrainer）
     train epoch → val epoch → EMA → save best/last → results.csv
```

每个环节的“唯一负责文件”：

| 环节 | 文件 | 说明 |
|---|---|---|
| 入口 / 超参组装 | `tools/train_mav6d.py`、`tools/train.py` | 目前两者用的不是同一套默认值 |
| 数据集路径 | `configs/datasets.yaml`、`src/tod/paths.py` | 只改注册表，不改脚本 |
| 变体 DSL / 模型图 | `src/tod/compose.py` | `model.yaml` 是生成物，不是手写配置 |
| 建模后手术 | `src/tod/engine/surgery.py` | Efficient_UAVDet / DualAttention |
| 训练器钩子 | `src/tod/engine/trainer.py`、`pose_trainer.py` | 损失必须在 `set_model_attributes()` 之后重建 |
| 姿态/框损失 | `src/tod/loss/criterion.py`、`pose.py`、`box.py` | `TinyPoseLoss` 是 OKS 的替换件 |
| 几何损失 | `src/tod/loss/reproj.py` | `L_rep` 的坐标/内参/匹配都在这里，P0 集中地 |
| 6D 数据旁挂 | `src/tod/data/pose6d.py` | `labels_6d/<split>.csv` → batch 的 `gt_pose` |
| 评测 | `tools/test_mav6d.py`、`src/tod/eval/pose.py` | ADD/ADD(0.5d)/10°10cm 用 PnP 单独算 |

**记住三个“不是”**：

- `model.yaml` 不是训练配置，它只描述网络结构；训练超参在 `variant.yaml` 的 `train:` 段和命令行。
- `labels_6d` 不是“可选附加信息”；只要 `EP7.reproj=true`，它就是 `L_rep` 的监督信号，错一点就是错的梯度。
- 2 epoch 的 smoke 不是 baseline；它只验证“回路能跑”，不验证“学得对”。

---

## 2. 当前实际生效的超参 vs 文档写的

`tools/train_mav6d.py` 第 245–248 行：

```python
train_args = dict(spec.get("train") or {})
for key, value in DEFAULT_TRAIN.items():
    train_args.setdefault(key, value)   # ← 变体里已有的键不会被 MAV6D 默认值覆盖
```

`variant.yaml` 的 `train:` 里已经有 `imgsz: 640`、`epochs: 200` 等，所以 `setdefault` 全部保留变体值。
结果是：

| 参数 | 文档/注释说 | 实际不传 CLI 时 | 原因 |
|---|---|---|---|
| `imgsz` | 1280 | **640** | 变体 `train.imgsz=640` 赢了 `setdefault` |
| `epochs` | 100 | **200** | 变体 `train.epochs=200` 赢了 |
| `batch` | 4 | 4 | 两边相同，看不出问题 |
| `optimizer` | auto | **SGD** | 变体写了 `SGD`，`DEFAULT_TRAIN` 没有这个键 |
| 预训练 | 从 `yolov8n-pose.pt` 起步 | **从零训练** | `weights` 变量算出来后没有传给 `YOLO` |
| `degrees/scale/translate` | 10/0.4/0.1 | **0/0/0** | `L_rep` 启用后 `pose_safe_augment_overrides()` 强制关闭 |
| `fliplr` | 0.5 | **0.5** | 安全覆盖里没关它；但 `L_rep` 下它是不安全的（P0-4） |

因此“按文档跑默认值”和“实际跑的值”不是一回事。
**在修好之前，所有命令都显式写 `--imgsz 1280 --batch 4 --epochs 100`，不要依赖默认值。**

---

## 3. P0：会让训练“看起来正常、第 16 epoch 开始崩”的问题

### P0-1　6D 位姿标签整体错误（最严重）

涉及代码：

- `tools/mav6d_calibrate.py::read_pose_file()`（第 121–163 行）
- `tools/mav6d_to_yolo.py::_pose_line()`（第 265–321 行）
- `tools/mav6d_to_yolo.py::convert_frame()`（第 261 行）

`read_pose_file()` 返回 `(target_raw, camera_pose)`，其中：

- `target_raw = values[0:8]`，`_pose_line()` 用它的 `[1:4]`/`[4:8]` 当 **MAV→VICON**；
- `camera_pose = values[9:16]`，`_pose_line()` 完全没用。

但同一仓库的 `mav6d_calibrate.py::check_against_util()`（第 533–574 行）**用的是 `camera_pose` 去变换机体 3D 点**，
再套 `UTIL_CAM2VICON`，得到 **0.0004 px** 的重投影误差（`tests/.tmp/c0.txt` 也记录了 0.000376 px）。
这说明：

- **`[9:16]` 才是 MAV→VICON（目标位姿）**；
- **`[1:8]` 是相机位姿**，不是 MAV 位姿；
- 当前 `_pose_line()` 把相机位姿当成了 MAV 位姿；
- 而且矩阵乘顺序也反了：正确是  
  `cam2mav = inv(cam2vicon) @ mav2vicon`，  
  当前是 `mav2vicon @ inv(cam2vicon)`。

**实测证据**（`.cache/diag_pose_pnp.py`，22 帧真实 MAV6D 标签）：

| 指标 | PnP 位姿（正确） | `labels_6d` 当前位姿 |
|---|---|---|
| 重投影平均误差 | **0.216 px** | **2435.2 px** |
| 重投影最大误差 | 4.736 px | 2513.2 px |
| 与 PnP 旋转差（均值） | 0° | **168.89°** |
| 与 PnP 旋转差（最大） | 0° | 173.2° |
| 平移模长（均值） | 3.014 m | 6.658 m |

**为什么测试没抓到**：
`tests/test_mav6d.py::test_pose_line_roundtrip()` 自己构造了一个 `target`，
并断言 `_pose_line()` 等于 `mav2vicon @ inv(cam2vicon)`。
它验证的是“函数和它自己的错误公式一致”，不是“和真实 2D 标签一致”。
`tests/train_pose_smoke.py` 用的又是合成数据，`labels_6d` 是测试自己按同一套几何造的，
所以也不会暴露真实数据的问题。

**修复方向**：

```python
# read_pose_file(): 语义改成
camera_pose = np.asarray(values[1:8], dtype=float)   # [x,y,z,qx,qy,qz,qw]
mav_pose    = np.asarray(values[9:16], dtype=float)  # [x,y,z,qx,qy,qz,qw]
return camera_pose, mav_pose

# _pose_line(): 用 mav_pose；矩阵顺序
cam2vicon = np.asarray(calib.UTIL_CAM2VICON, dtype=float)
mav2vicon = np.eye(4)
mav2vicon[:3, :3] = calib.quat_to_rot(q_mav / np.linalg.norm(q_mav))
mav2vicon[:3, 3] = t_mav
cam2mav = np.linalg.inv(cam2vicon) @ mav2vicon
```

并加一道**开训前/转换时**的自检：用 `K` 把 3D 点投影到图像，和 `labels_yolo6d` 的 9 点比，
平均误差 > 1–2 px 就拒绝该帧/报错。这个检查比“距离在 0.3–15 m”强得多。

> 现在 `D:/MAV6D` 原始标签已经不在盘上，只有转换产物。
> 两条路：
> 1. 有原始数据时：修 `_pose_line()` 后重新转换；
> 2. 没有原始数据时：用现有的 YOLO-pose 标签 + `K` + `object_points` 做 **PnP 反解**，
>    重建 `labels_6d`（PnP 重投影 0.2 px，足够作为几何监督的位姿来源）。
>    这属于“由 2D 标签反解伪 6D 标签”，要在 dataset.yaml 里如实标注 `pose_source: pnp`。

### P0-2　`L_rep` 的坐标单位与锚点匹配错误

涉及代码：`src/tod/loss/reproj.py::ReprojCriterion._one_branch()`（第 405–465 行）、
`_nearest_anchor_per_box()`（第 483–501 行）。

`v8PoseLoss.kpts_decode()` / `PoseLoss26.kpts_decode()` 的输出是**网格单位**：

```python
y[..., :2] *= 2.0
y[..., 0] += anchor_points[:, 0] - 0.5   # anchor_points 是 0..feat_w 的网格坐标
```

在 640 输入、P2 头下，`decoded` 的范围实测是 **[-4.7, 160.97]**，不是 [0,1]。
`_one_branch()` 却把它当归一化坐标：

```python
sel_px = picked * [img_w, img_h]      # 160 * 640 = 102400
sel_norm = sel_px / [img_w, img_h]    # 又变回 picked
value = self.reproj(sel_norm * [img_w, img_h], R, t, K=K)  # 实际是 picked * 640
```

真正的像素应该是 `decoded * stride`（P2 的 160 网格 × stride 4 = 640 像素）。
当前相当于把像素放大了 `stride` 倍，P2 上约 160×，P5 上约 20×。

**匹配也错了**：
`_nearest_anchor_per_box()` 用 `decoded_i[:, 0, :2]`（0..160 的网格单位）和
GT 框中心（归一化 0..1）比最近邻，`argmin` 几乎总是选到左上角附近的锚点，
而不是目标对应的锚点。于是 `L_rep` 的“每个 GT 取一个最负责的预测”实际是“每个 GT 取一个随机角落的预测”。

**实测**：在 640、随机初始化模型、一个真实 batch 上，
当前 `_one_branch()` 返回 **27.92**；而用正确单位 + 正确匹配 + 正确内参，
在 GT 关键点上的 `L_rep` 应接近 **0**（见 P0-3 表）。

**修复方向**：

```python
anchor_points, stride_tensor = make_anchors(feats, stride, 0.5)
decoded = criterion.kpts_decode(anchor_points, raw)[..., :2]  # 网格单位
scale = torch.tensor([img_w, img_h], ...)
decoded_px = decoded * stride_tensor.view(1, -1, 1, 1)         # 增强后图像像素
decoded_norm = decoded_px / scale                              # [0,1]
# 匹配用 decoded_norm；取点用 decoded_px
idxs = nearest_anchor_indices(decoded_norm[i, :, 0, :2], boxes[:n_obj])
sel_px = decoded_px[i, idxs]                                    # (n_obj, K, 2)
value = self.reproj(sel_px, R, t, K=K_scaled)
```

### P0-3　`L_rep` 的内参缩放用错（max 而不是 min）

`_scaled_intrinsics()`（第 504–532 行）：

```python
ratio = max(aug_w / w0, aug_h / h0)   # ← 反了
K2[0, 2] = aug_w / 2.0                # ← 没考虑原主点和 letterbox 补边
K2[1, 2] = aug_h / 2.0
```

letterbox 是**等比缩放**，比例是 `min(aug_w/w0, aug_h/h0)`。
MAV6D 原图 1920×1080、输入 640 时：

| 量 | 正确值 | 当前代码 |
|---|---|---|
| `fx` | 1979.4 × 1/3 = **659.8** | 1979.4 × 0.5926 = **1173.0**（+78%） |
| `cx` | 976.8/3 + 0 = **325.6** | 320.0 |
| `cy` | 534.0/3 + 140 = **318.0** | 320.0 |

**实测**（`.cache/diag_lrep_geom.py`，8 帧，GT 关键点 + PnP 真值位姿）：

| 配置 | `L_rep` |
|---|---|
| 正确 K（min + 主点映射） | **0.0000** |
| 当前 K（max + 中心） | 0.0629–0.0734 |
| 水平翻转、位姿不翻转 | 1.1389–1.2771 |
| 水平翻转、位姿同步翻转 | **0.0000** |

正确 K 下 GT 的 `L_rep` 应该是 0；当前 K 有 0.06–0.07 的系统性偏差。
这不是“正则化”，是每步都在推一个错误的尺度。

**修复方向**：用 batch 里的 `ori_shape` / `resized_shape` / `img.shape` 直接算：

```python
gain = min(img_w / w0, img_h / h0)          # 若 LetterBox 有 scaleup=False，再 min(..., 1.0)
pad_x = (img_w - w0 * gain) / 2.0
pad_y = (img_h - h0 * gain) / 2.0
K2[0, 0] *= gain; K2[1, 1] *= gain
K2[0, 2] = K[0, 2] * gain + pad_x
K2[1, 2] = K[1, 2] * gain + pad_y
```

（不要直接用 `ratio_pad[0][0]` 前先确认它的语义；这个版本里它和 `resized_shape/ori_shape` 的顺序容易混。
直接用 `ori_shape`/`img.shape` 最稳。）

### P0-4　`fliplr` 与 `L_rep` 不自洽

`src/tod/data/pose6d.py::pose_safe_augment_overrides()` 只关了
`degrees/scale/translate/shear/perspective/mosaic/copy_paste/mixup`，
**没有关 `fliplr`**，而 `DEFAULT_TRAIN` 里 `fliplr=0.5`。

框架的 `fliplr` 会把图像和关键点水平翻转（并按 `flip_idx` 重排关键点），
但 `gt_pose` 的 `R,t` 不会跟着翻。`L_rep` 用未翻转的位姿去对翻转后的预测点，
50% 的 batch 是系统性错配。

**实测**（同上，GT 关键点）：

| 情况 | `L_rep` |
|---|---|
| 不翻转 | 0.0000 |
| 翻转图像、不翻位姿 | 1.14–1.28 |
| 翻转图像 + `R'=MR, t'=Mt`，K 主点镜像 | 0.0000 |

**修复方向（短期）**：`L_rep` 启用时把 `fliplr` 也关掉：

```python
return {
    "degrees": 0.0, "scale": 0.0, "translate": 0.0,
    "shear": 0.0, "perspective": 0.0,
    "mosaic": 0.0, "close_mosaic": 0,
    "copy_paste": 0.0, "mixup": 0.0,
    "fliplr": 0.0,        # ← 新增：位姿没有同步翻，开了就是错的
}
```

**长期**：在 `Pose6DDataset.__getitem__` 里自己做翻转，
同时翻 `img`、`bboxes`、`keypoints`、`gt_pose`，并把 `K` 的主点镜像；
或者干脆不做水平翻转，用颜色增强 + 更多数据/epoch 补。

### P0-5（可观测性）　`L_rep` 在日志里不可见

`ReprojCriterion.forward()` 把 `L_rep` 加到 `total`，但返回的 `items` 还是
`v8PoseLoss` 的 5 项（box/pose/kobj/cls/dfl）。`results.csv` 里没有 `lrep` 列。
如果第 16 epoch 几何项爆炸，你只能看到总 loss 变大，不知道是 `L_rep`。

**修复方向**：在 trainer 里加一个 epoch 末日志/callback，读 `criterion.last_value` 和
`criterion.reproj_terms()`；或者在 `label_loss_items()` 里加第 6 项。
至少要在 warm-up 结束后的前几个 epoch 打印：

```text
[L_rep] epoch=16 value=... terms=...
```

---

## 4. P1：不会立刻让 loss 爆炸，但会让结论不可信

### P1-1　预训练权重从未加载

`tools/train_mav6d.py` 第 232–239 行算出了 `weights`，第 288 行却是：

```python
model = YOLO(str(model_yaml), task=variant.task)
```

`weights` 只用于 print。`--no-pretrained` 也没有任何实际作用。
`tools/train.py` 同样是从 `model.yaml` 随机初始化。
所以现在所有 MAV6D 训练都是**从零训练**；如果按“预训练 100 epoch”的预期去调参，会误判。

**修复方向**（二选一）：

1. 明确“从零训练”，删掉 `weights` 和 `--no-pretrained` 的假日志；
2. 真做迁移：先建 `YOLO(weights)` 的模型，再把兼容层 `load_state_dict(..., strict=False)`
   到自定义结构（ADown/P2/头分支会丢形状，只迁移骨干/颈部同名层），并在日志里报告匹配率。

在没做 2 之前，不要写“从 yolov8n-pose.pt 起步”。

### P1-2　`DEFAULT_TRAIN` 实际不生效

见 §2。`setdefault` 的方向反了。建议只对“数据集相关键”强制覆盖：

```python
MAV6D_OVERRIDES = ("imgsz", "batch", "epochs", "mosaic", "close_mosaic",
                   "copy_paste", "mixup", "fliplr", "flipud",
                   "project", "plots", "val", "save_period")
for key in MAV6D_OVERRIDES:
    if key in DEFAULT_TRAIN:
        train_args[key] = DEFAULT_TRAIN[key]
```

优化器/lr 等训练策略仍由变体决定，CLI 最后覆盖。

### P1-3　smoke 永远验证不到 `L_rep`

`tests/train_pose_smoke.py` 的合成数据 + `--warmup 15` 默认值，
2 epoch 的 smoke 不会激活 `L_rep`；`tools/train_mav6d.py --smoke` 也一样。
**建议**：

- `--smoke` 强制 `tod.reproj_warmup_epochs=0`（直接在一个真实 batch 上跑 `L_rep`）；
- 或者加 `--probe-lrep`：建一个真实 batch，`criterion.set_epoch(warmup+1)`，
  断言 `reproj_terms()>0`、`L_rep` 有限、GT 上 `L_rep<1e-4`。

### P1-4　`pose_safe_augment_overrides()` 把几何增强全关了

`L_rep` 启用时 `degrees/scale/translate/shear/perspective/mosaic/copy_paste/mixup` 全为 0，
只剩 `fliplr`（且它本身不安全，见 P0-4）和 HSV。
对 10k 帧的 MAV6D，这可能限制泛化。建议分阶段：

1. **2D baseline**：全几何增强（`degrees=10, scale=0.4, translate=0.1, fliplr=0.5, mosaic=0`）；
2. **+L_rep**：先按安全覆盖（无几何增强）验证几何项真的有用；
3. **pose-aware aug**（可选）：把 2D 仿射同步到 `R,t/K`，再恢复旋转/缩放/平移。

不要在 `L_rep` 和几何增强同时开的情况下直接比较精度——那会把两个变量混在一起。

### P1-5　测试把错误公式固化了

`tests/test_mav6d.py::test_pose_line_roundtrip()` 断言的是
`mav2vicon @ inv(cam2vicon)`，而正确是 `inv(cam2vicon) @ mav2vicon`。
修 P0-1 时必须同步改这个测试，并加一条“用 2D 关键点重投影校验”的真实数据测试，
否则测试越绿，错误越稳。

---

## 5. 正确的训练顺序（建议照这个顺序做）

### Step 0　固定环境

当前 shell 的 `python` 指向 `C:\Python314\python.exe`，**没有 torch**；
能跑训练的是：

```text
D:\miniconda3\python.exe                          # torch 2.11.0+cu128, ultralytics 8.4.60
D:\miniconda3\envs\YOLO-TOD\python.exe            # python 3.12, torch 2.10.0+cu128, ultralytics 8.4.152
```

先固定一个，不要混用。下面命令用 `D:\miniconda3\python.exe`（与最后一次 smoke 的 8.4 系列一致）。

### Step 1　先跑 2D baseline（不要开 L_rep）

目的：拿到“只有 2D OKS + 框损失”的干净上限，确认数据/模型/增强/学习率没问题。

```powershell
# 复制一份 2D 变体，只把 EP7.reproj 关掉（不要改原变体）
Copy-Item variants/spae-yolov8n-pose/variant.yaml variants/spae-yolov8n-pose/variant-2d.yaml
# 编辑 variant-2d.yaml：EP7.reproj: false
D:\miniconda3\python.exe tools/train_mav6d.py `
  --model phantom4 --variant variants/spae-yolov8n-pose/variant-2d.yaml `
  --imgsz 1280 --batch 4 --epochs 100 --name mav6d-p4-2d
```

此时 `pose_safe_augment_overrides()` 不会触发，几何增强按 `DEFAULT_TRAIN`/CLI 生效。

### Step 2　修 P0-1：6D 标签

- 有原始 `D:/MAV6D`：修 `read_pose_file()` + `_pose_line()`，重新转换；
- 没有原始数据：加 `--pose-from-pnp`，用现有 `labels/train,val` + `K` + `object_points`
  做 PnP，重建 `labels_6d/*.csv`。

无论哪条路，都要跑：

```powershell
D:\miniconda3\python.exe .cache/diag_pose_pnp.py
# 期望：CSV 重投影 < 1 px，dR ~ 0°
```

### Step 3　修 P0-2/3/4：`L_rep`

- 单位：`decoded * stride` → 像素；匹配用归一化中心；
- 内参：`min` + 主点映射；
- `fliplr`：短期设为 0，长期做 pose-aware flip；
- 加测试：GT 关键点 + PnP 位姿 → `L_rep ≈ 0`；1 px 平移 → 小正数；
  翻转不翻位姿 → 大数（证明检测得到）。

```powershell
D:\miniconda3\python.exe .cache/diag_lrep_geom.py
# 期望：correctK=0.0000，flip_noPoseFix 大，flip_poseFix=0.0000
```

### Step 4　开 `L_rep` 的 short run

```powershell
D:\miniconda3\python.exe tools/train_mav6d.py `
  --model phantom4 --imgsz 1280 --batch 4 --epochs 100 --name mav6d-p4-lrep
```

先在 `tod.reproj_warmup_epochs=15` 下跑，重点看：

- 第 16 epoch 起 `lrep` 是否有日志、是否有限；
- `train/pose_loss` 是否突然跳变；
- `results.csv` 的 `mAP50(P)` / `mAP50-95(P)` 是否比 2D baseline 更差。

如果第 16 epoch 后变差，先别调 `L_rep` 权重，回到 Step 2/3 核对标签和单位。

### Step 5　评测

```powershell
D:\miniconda3\python.exe tools/test_mav6d.py `
  --model phantom4 --weights results/mav6d-p4-lrep/weights/best.pt --imgsz 1280
```

必须同时看：

- PnP 口径 `ADD / ADD(0.5d) / 10°10cm`（与论文同口径）；
- 关键点 `OKS-AP 50:95` + 按目标边长分层；
- 框 `AP` + `AP_small/AP_tiny`（`tools/val.py`）。

只看整体 mAP 会掩盖小目标退化；只看 OKS-AP 会掩盖 3D 位姿退化。

---

## 6. 超参与显存建议

实测（RTX 5060 Laptop 8 GB，AMP，合成前向+反向）：

| imgsz | batch | peak alloc |
|---|---|---|
| 1280 | 4 | 2.86 GiB |
| 1280 | 8 | 5.69 GiB |
| 1024 | 4 | 1.84 GiB |
| 1536 | 2 | 2.11 GiB |

所以 **1280 + batch 4 是安全的**；batch 8 也能放下模型，但要留出数据/验证/Windows 开销，
建议先用 4，稳定后再试 8。OOM 不是当前训练问题的主因。

建议 baseline 超参：

```yaml
imgsz: 1280
batch: 4
epochs: 100        # 从零训练可到 200；有预训练再降
optimizer: SGD     # 或 AdamW；固定一种，不要中途换
lr0: 0.01
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3.0
mosaic: 0.0        # 关键点任务不建议 mosaic
close_mosaic: 0
degrees: 10.0      # 仅 2D baseline
scale: 0.4         # 仅 2D baseline
translate: 0.1     # 仅 2D baseline
fliplr: 0.5        # 仅 2D baseline；L_rep 开启时设 0
pose: 12.0
kobj: 1.0
amp: true
workers: 4
seed: 0
deterministic: true
```

---

## 7. 验收清单（修完必须全过）

- [ ] `read_pose_file()` / `_pose_line()` 用 `[9:16]` 作 MAV 位姿，`inv(cam2vicon) @ mav2vicon`；
- [ ] 真实数据 PnP 重投影平均 < 1 px（当前 2435 px）；
- [ ] 重建后的 `labels_6d` 与 `labels` 行数一一对应，且 `check_pose_dataset.py` 通过；
- [ ] `L_rep` 在 GT 关键点上 < 1e-4（正确 K）；
- [ ] `L_rep` 对 1 px 平移有正响应，对 10 px 平移明显更大；
- [ ] `fliplr` 与位姿同步（或 `L_rep` 下关闭 `fliplr`）；
- [ ] 真实 batch 上 `reproj_terms() > 0`，`last_value` 有限；
- [ ] `results.csv` 有 `lrep` 列或 epoch 日志；
- [ ] 第 16 epoch 前后 `pose_loss` 无跳变、无 NaN；
- [ ] 2D baseline vs +L_rep 的 `ADD(0.5d)` / `10°10cm` / `AP_small` 同表对比；
- [ ] 至少 3 个 seed 的重复性检查。

---

## 8. 这次审查用到的诊断脚本（临时，未入库）

| 脚本 | 作用 | 关键输出 |
|---|---|---|
| `.cache/diag_pose_pnp.py` | 用 PnP 反解真实 2D 标签，和 `labels_6d` 对比 | CSV 2435 px / 169°；PnP 0.216 px |
| `.cache/diag_lrep_geom.py` | GT 点上比较正确 K / 当前 K / 翻转 | correctK=0，codeK=0.063–0.073，flip=1.14–1.28 |
| `.cache/diag_reproj.py` | 看 `kpts_decode` 的实际单位、当前 `_one_branch` 的 L_rep | decoded 范围 [-4.7, 161]，L_rep 27.9 |
| `.cache/diag_mem.py` | 1280/1024/1536 的显存 | 1280×4=2.86 GiB，1280×8=5.69 GiB |

运行示例：

```powershell
D:\miniconda3\python.exe .cache/diag_pose_pnp.py
D:\miniconda3\python.exe .cache/diag_lrep_geom.py
D:\miniconda3\python.exe .cache/diag_mem.py
```

---

## 9. 一句话总结

**先把 `EP7.reproj` 关掉，跑一个 1280/4 的 2D baseline；
修好 6D 位姿标签和 `L_rep` 的单位/内参/翻转，再加回几何项。
在 `labels_6d` 修好之前，任何“加 L_rep 提升精度”的结论都不可信。**
