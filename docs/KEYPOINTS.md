# 9 点 UAV 机体关键点定义（SPAE-YOLOv8n-pose / RflySim）

本文件是本库对**"微小型无人机 9 个机体关键点"**的唯一定义来源。
所有配置、脚本、评测都引用这里的编号与名字，不得各处自行编号。

---

## 1. 来源与核对状态

| 项 | 内容 |
|---|---|
| 方法出处 | Ye Zheng, Canlun Zheng, Jiahao Shen, Peidong Liu, Shiyu Zhao, *Keypoint-Guided Efficient Pose Estimation and Domain Adaptation for Micro Aerial Vehicles*, **IEEE T-RO** 40 (2024) 2967–2983，DOI [10.1109/TRO.2024.3400938](https://doi.org/10.1109/TRO.2024.3400938) |
| 预印本 | 已存档于本仓库：`docs/Zheng 等 - 2024 - Keypoint-Guided Efficient Pose Estimation and Domain Adaptation for Micro Aerial Vehicles.pdf` |
| 作者仓库 | <https://github.com/WindyLab/MAV6D>（**MIT**）；论文正文另注 <https://github.com/WestlakeAerialRobotics/MAV6D>（该组织路径当前 404） |
| **点结构** | ✅ **论文明确**：1 个质心点 + 8 个角点（见 §2） |
| **编号顺序** | ⚠️ `FIGURE_INFERRED`：论文 Fig. 2(a) 是图片，文本未给 p1…p8 的逐一语义，按"绕着质心的 8 个角"常规读图顺序推定 |
| **3D 坐标** | ❌ **论文未给**：正文只说 "the 3-D coordinate of keypoint i in Σo"（式 12），**没有给 p0…p8 在机体坐标系的数值表**。必须由你自己的机型几何提供（见 §5） |

### 1.1 为什么仍有推断成分

论文文字把**结构**说得很清楚（1 质心 + 8 角点、质心引导角点、刚体几何不变），
但**逐一编号的语义**在 Fig. 2(a) 这张图里，PDF 文本层给不出来。
本库因此把两件事分开标注：

* **结构（质心 + 8 角点）= 论文明确**，配置里体现为 `kpt_names[0] == "centroid"`；
* **p1…p8 的对应关系 = 读图推断**，写作 `FIGURE_INFERRED`。

校正只改数据集 YAML 的 `kpt_names` 与 `flip_idx` 两行，**不需要改任何代码**。

---

## 2. 9 点表

论文原文（§III-A-1，Fig. 2(a) 图注）：

> "There are nine keypoints of an MAV, including one centroid point (numbered p0)
> and eight corner points (numbered p1 ∼ p8)."

> "First, the centroid point corresponds to the distinctive appearance features of the
> MAV, making it a reliable feature point for MAV localization in the image. While the
> other eight corner points may be subject to occlusion by the MAV's body, the center
> point remains visible as it cannot be self-occluded. Second, since the MAV is a rigid
> body, the relative geometry of the corner keypoints remains invariant with respect to
> the centroid point."

这两条正是**质心引导**的两个理由，也是本库设计的关键：

1. **质心不会被自遮挡** → 它是 9 点里最可靠的点，所以让它当"锚点"；
2. **刚体 ⇒ 角点相对质心的几何不变** → 定位到质心后，8 个角点只在附近搜索即可。

机体固定坐标系 Σo 约定：**x 向前（机头）、y 向左、z 向上**。

| # | `kpt_names` | 语义 | `flip_idx` | 依据 |
|---|---|---|---|---|
| **0** | `centroid` | **质心点** —— 机体几何中心，**不可自遮挡** | 0（在镜像轴上） | 论文明确（p0） |
| 1 | `corner_1` | 角点 1（推定：左前上） | 2 | 读图推断 |
| 2 | `corner_2` | 角点 2（推定：右前上） | 1 | 读图推断 |
| 3 | `corner_3` | 角点 3（推定：左前下） | 4 | 读图推断 |
| 4 | `corner_4` | 角点 4（推定：右前下） | 3 | 读图推断 |
| 5 | `corner_5` | 角点 5（推定：左后上） | 6 | 读图推断 |
| 6 | `corner_6` | 角点 6（推定：右后上） | 5 | 读图推断 |
| 7 | `corner_7` | 角点 7（推定：左后下） | 8 | 读图推断 |
| 8 | `corner_8` | 角点 8（推定：右后下） | 7 | 读图推断 |

`kpt_shape = [9, 3]`（x, y, visibility）。
`flip_idx = [0, 2, 1, 4, 3, 6, 5, 8, 7]`：沿机体 y 轴（左右）镜像，
**质心在镜像轴上自配对**，其余 4 对左右角点互换。

> **如果你的机型不是四旋翼**（六旋翼、带起落架…）：改 `kpt_names` 与 `flip_idx`
> 两行 + `kpt_shape`；`tools/check_pose_dataset.py` 会校验一致性。

---

## 3. `flip_idx` 为什么必须显式给出（框架硬行为，已核实）

ultralytics 在 `ultralytics/data/augment.py::build_transforms` 里：

```python
flip_idx = dataset.data.get("flip_idx", [])
if dataset.use_keypoints:
    if len(flip_idx) == 0 and (hyp.fliplr > 0.0 or hyp.flipud > 0.0):
        hyp.fliplr = hyp.flipud = 0.0     # 直接关掉翻转增强！
        LOGGER.warning("No 'flip_idx' array defined in data.yaml, disabling ...")
    elif flip_idx and (len(flip_idx) != kpt_shape[0]):
        raise ValueError("flip_idx length must be equal to kpt_shape[0]")
```

翻转时执行的是 `instances.keypoints[:, flip_idx, :]`，即

> **翻转后的第 i 号点 = 原来的第 `flip_idx[i]` 号点**（等价于把标注点按索引重排）。

因此 `flip_idx` 必须是一个**对换置换**（`flip_idx[flip_idx[i]] == i`）。
写错的后果是**静默的**——增强会把关键点配到错误的角点上，训练仍能跑、损失仍会降，
只是学出一个错的机体结构。本库因此：

* `flip_idx` 一律显式写在数据集 YAML 里（绝不依赖框架默认）；
* `tools/check_pose_dataset.py` 校验它是合法对换置换、且长度等于 `kpt_shape[0]`；
* `tests/test_pose.py` 断言配对表与配置一致。

---

## 4. σ（OKS 容忍度）怎么取

COCO 的 17 个 σ 是**人体**关节专用的，机体角点不适用。
本库 EP7 提供（`src/tod/loss/pose.py`，`EP7.sigma_strategy`）：

| `sigma_strategy` | 行为 | 适用 |
|---|---|---|
| `uniform`（**本变体默认**） | 所有点同 σ（对齐 COCO σ 均值量级） | 无"哪个点更难"先验时的诚实选择 |
| `auto` | 形状自适应几何曲线（中点小、两端大），缩放到 COCO σ 均值量级 | 通用兜底；**但角点没有"末端更难"的依据**，别默认用它 |
| `balanced` | 压平后归一到均值 1 | 明确接受**损失尺度改变**时（测试里有断言钉住） |
| `person` / `framework` | **仅 17 点**（COCO 标准 σ，未缩放） | 人体姿态；9 点用它会回退 `auto` 并打日志 |

论文用的是 **MSE（式 7）** 而不是 OKS，且额外加了 3D 几何项（§5）。
本库保留 OKS 作为默认（对尺度更鲁棒、能直接接 COCO 口径的评测），
但**这不是论文的做法**，变体卡片里必须写清。

---

## 5. 3D 几何损失 `Lrep` 需要的机体 3D 关键点（**本变体的核心增强**）

论文式 (6)/(8)：

```
L_reg = L_loc + L_rep
L_loc = Σ_i ½‖x_i^pr − x_i^gt‖²                    （2D 像素 MSE）
L_rep = Σ_i smoothL1( (I − v̂_i v̂_iᵀ) p_i^c )       （3D 几何一致性）
```

* `p_i^c`：第 i 个关键点在**相机坐标系**下的真值位置 = `R · p_i^o + t`；
* `p_i^o`：第 i 个关键点在**机体坐标系** Σo 下的坐标 —— **这就是本节要提供的东西**；
* `v̂_i`：由**预测**的 2D 关键点得到的相机射线方向（归一化）；
* `(I − v̂ v̂ᵀ)`：把误差投影到**垂直于该射线**的平面 —— 即"点到射线的垂距"。
  这正是它有效的原因：**沿视线的深度误差无法从 2D 观测判别，所以不该被惩罚**；
  只有横向偏差会被算进损失。

论文消融（Table III）：加 `Lrep` 后 **10°10cm 提升 > 10%**，是三个组件里最显著的。
论文还规定：**前 15 个 epoch 只用 `L_loc`**（此时网络定位能力还弱，
过早启用几何项反而有害），之后再加入 `L_rep`。

### 5.1 必须由你提供的量

| 量 | 说明 | 放哪 |
|---|---|---|
| `tod.object_points` | 9 个关键点在 Σo 的 3D 坐标（米），**顺序与 §2 的表一致** | 数据集 YAML（本库新增字段） |
| `tod.model_diameter` | 机型外接直径（米），用于 ADD 阈值 `ADD(0.5d)` | 数据集 YAML |
| `tod.camera_matrix` | 3×3 相机内参 K（`fx, fy, cx, cy`） | 数据集 YAML |
| 每帧 6D 位姿 | `R, t`（机体→相机），逐目标一行 | `labels_6d/<split>.csv` 旁挂文件（见 §5.2） |

论文**没有**给出 Phantom 4 / Mavic 2 的具体 3D 坐标表，所以本库不假装知道；
`configs/_base_/datasets/rflysim-pose.yaml` 里给的是一份**占位**（四旋翼对称布局 +
尺寸待填），必须换成你 RflySim 机模的真实几何。

### 5.2 6D 位姿旁挂文件格式（本库约定）

```
labels_6d/<split>.csv     # 逗号分隔，带表头
image,obj_index,tx,ty,tz,qx,qy,qz,qw
train_0000,0,0.12,-0.03,1.85,0.02,-0.01,0.71,0.70
```

* `image`：图像文件名（不含扩展名），与 `images/<split>/` 对应；
* `obj_index`：该图内目标序号（0 起），**必须与标签文件里的行序一致** ——
  校验器会检查行数与索引连续性，不一致直接报错；
* `t*`：机体→相机的平移（米）；`q*`：旋转四元数（`qx, qy, qz, qw`，机体→相机）。

> 为什么不直接塞进 YOLO 标签行：那会破坏与 `ultralytics` 标准姿态格式的兼容，
> `labels/*.txt` 的列数也不再是 `5 + 3·n_kpt`（校验器与框架都会失配）。
> 旁挂文件让标准标签保持干净，代价是"行序必须对齐"——所以校验器强制检查它。

### 5.3 如何从 6D 位姿生成 2D 关键点真值（RflySim 录制参考流程）

```
p_cam = R_cam_uav @ p_o + t_cam_uav        # 机体点 → 相机系
u, v  = (K @ p_cam) / p_cam.z              # → 像素
x, y  = u / W, v / H                       # → 归一化（YOLO 格式）
vis   = 2 可见 / 1 存在但被遮挡 / 0 未标注（越界）
```

论文强调 `T_c^o`（机体相对相机的位姿）**必须标定准确**，否则反投影出的 2D 真值
根本不可用（论文 Fig. 5 专门演示了这一点）。RflySim 是仿真，这一步天然精确；
但**相机内参与外参必须用你仿真里实际配置的那一组**，不能随手填。
