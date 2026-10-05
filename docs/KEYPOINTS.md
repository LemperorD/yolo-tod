# 9 点 UAV 机体关键点定义（SPAE-YOLOv8n-pose / RflySim）

本文件是本库对**"微小型无人机 9 个机体关键点"**的唯一定义来源。
所有配置、脚本、评测都引用这里的编号与名字，不得各处自行编号。

---

## 1. 来源与状态

| 项 | 内容 |
|---|---|
| 方法出处 | Ye Zheng, Canlun Zheng, Jiahao Shen, Peidong Liu, Shiyu Zhao, *Keypoint-Guided Efficient Pose Estimation and Domain Adaptation for Micro Aerial Vehicles*, **IEEE T-RO** 40 (2024) 2967–2983，DOI [10.1109/TRO.2024.3400938](https://doi.org/10.1109/TRO.2024.3400938) |
| 作者官方仓库 | <https://github.com/WindyLab/MAV6D>（**MIT**） |
| 作者数据集 | MAV6D（6D 位姿，Westlake 分享链接见仓库 README）：`label/` 每行 `timestamp t_x t_y t_z r_x r_y r_z r_w`（相机与 UAV 各一组），另附 3D 关键点坐标与相机内参 + 一个转格式脚本 |
| 本文件状态 | ⚠️ **9 个点的顺序为 `INFERRED`（本库推断），等待论文 Figure 11 校正** |

### 为什么是"推断"

论文全文在 IEEE Xplore **付费墙**后，作者仓库只有 README（无图、无关键点定义、无代码），
因此**无法核对**论文里那 9 个点的确切语义与编号顺序。本库按以下两条硬规矩处理：

1. **不假装知道**：所有引用这份定义的地方都带 `INFERRED` 标记；
   `configs/_base_/datasets/rflysim-pose.yaml` 里的 `tod.kpt_definition_status: inferred`。
2. **可一键校正**：`kpt_names` 与 `flip_idx` 都在数据集 YAML 里，改这两行即可，
   不需要动任何代码（`tools/check_pose_dataset.py` 会校验它们与标签的一致性）。

> **请把论文 PDF 放到本目录（`docs/`）**，或提供 Figure 11 的截图。
> 拿到之后本文件与数据集配置会同步校正，并把状态改为 `paper-verified`。

---

## 2. 推断的 9 点表（`INFERRED`）

机体固定坐标系约定：**x 向前（机头）、y 向左、z 向上**；"左/右"是**飞行器自身**的左右。

| # | `kpt_names` | 语义 | `flip_idx` | 镜像伙伴 |
|---|---|---|---|---|
| 0 | `rotor_front_left` | 左前电机/旋翼中心 | 1 | 1 |
| 1 | `rotor_front_right` | 右前电机/旋翼中心 | 0 | 0 |
| 2 | `rotor_rear_left` | 左后电机/旋翼中心 | 3 | 3 |
| 3 | `rotor_rear_right` | 右后电机/旋翼中心 | 2 | 2 |
| 4 | `nose` | 机头中心点 | 4 | 自己（在镜像轴上） |
| 5 | `arm_front_left` | 左前机臂中点 | 6 | 6 |
| 6 | `arm_front_right` | 右前机臂中点 | 5 | 5 |
| 7 | `arm_rear_left` | 左后机臂中点 | 8 | 8 |
| 8 | `arm_rear_right` | 右后机臂中点 | 7 | 7 |

`kpt_shape = [9, 3]`（x, y, visibility）。

**为什么这个顺序是合理的**（论文题的线索）：论文标题里的 *centroid point-guided* 说明
有一个**质心点**在引导定位，其余点分布在机体四周（电机/机臂是最显著、最易标的外围结构）。
上表把 4 个电机放在最前（最显著），机头居中编号 4，机臂中点垫后 ——
**这是本库的设计选择，不是论文原文**。

**若你的 RflySim 机型不是四旋翼**（如六旋翼、带起落架的机型），点数与配对都要改：
本库的做法是**改 `kpt_names` 与 `flip_idx` 两行 + `kpt_shape`**，
`tests/train_pose_smoke.py --kpt-shape` 与 `tools/check_pose_dataset.py` 会跟着校验。

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

因此 `flip_idx` 是一个**置换**（通常由若干对换组成）：`flip_idx[flip_idx[i]] == i`。
写错的后果是**静默的**——增强会把关键点配到错误的关节上，训练仍能跑、损失仍会降，
只是学出一个错的解剖结构。本库因此：

* `flip_idx` 一律显式写在数据集 YAML 里（绝不依赖框架默认）；
* `tools/check_pose_dataset.py` 校验它是合法置换、且长度等于 `kpt_shape[0]`；
* `tests/test_pose.py` 断言"镜像对"与 `flip_idx` 一致。

---

## 4. σ（OKS 容忍度）怎么取

4–9 点没有 COCO 那种公认 σ（COCO 的 17 个 σ 是人体专用的）。
本库 EP7 现提供两档（见 `src/tod/loss/pose.py`，`EP7.sigma_strategy`）：

| `sigma_strategy` | 行为 | 适用 |
|---|---|---|
| `auto`（**默认**） | 形状自适应几何曲线（中点小、两端大），并**缩放到 COCO σ 的均值和 ==量级==**（`COCO_SIGMA_MEAN`，代码里现算） | 非 17 点的通用兜底，保证 OKS 项与框架/COCO 同量级、可比 |
| `balanced` | 把 σ 压平后归一到均值 1 | 只想削弱"末端点权重"、且**明确接受损失尺度改变**时（测试里有断言钉住这一点） |
| `person` / `framework` | **仅 17 点**可用（COCO 标准 σ，未缩放，与框架逐元素一致） | 人体姿态；9 点用它会退回 `auto` 并在日志里说明 |

⚠️ 对机体关键点（电机/机臂）来说，**"哪个点更难定位"与人体无关**：
四个电机通常一样好定位，机头可能因视角遮挡更难。所以**不要**沿用人体 σ 的
"末端点更宽容"假设 —— `auto` 的几何曲线只是**兜底**。

**下一步（尚未实现，登记在此以免被误以为已有）**：`joint_adaptive` —— 用
`tools/check_pose_dataset.py` 统计出的每点定位误差分布来**反推** σ。
RflySim 是仿真数据、GT 精确，它的"难度"只来自视角与遮挡，
比人体先验更贴近真实；等 RflySim 数据到位后这是一个明确的改进项。

---

## 5. 从 MAV6D 的 6D 位姿标注生成 2D 关键点（供参考）

作者数据集给的是 6D 位姿而不是 2D 关键点，官方转换脚本会在数据包里。
投影通式（与论文的 6D pose → keypoint 监督一致）：

```
p_cam = R_cam_uav @ (K_3d @ p_body) + t_cam_uav      # 机体 3D 关键点 → 相机系
u, v  = project(p_cam, camera_intrinsics)            # → 像素
vis   = 1 if inside_image and not occluded else 0     # 越界/自遮挡按 0（未标注）
```

要点：

1. **`K_3d`（9 个机体关键点的 3D 坐标）必须来自你的机型**，不能照搬论文的 MAV 尺寸；
   RflySim 里可以直接用机模配置的电机/机臂几何位置；
2. 归一化到 YOLO 格式：`u/W, v/H`；
3. **越界点写 `v=0` 而不是裁剪到边界**（写 0 = 未标注，正是框架掩码规则 `v != 0` 的语义）；
4. 自遮挡（如从正下方看四旋翼，机臂被机身挡住）建议 `v=1`（存在但不可见）而不是 0，
   这样 `kobj` 分支能学到"该点应当存在"。

RflySim 侧通常能直接导出 2D 图像点（视觉 API），若如此就跳过投影，
但**仍要保证 `v` 语义与上面一致**。
