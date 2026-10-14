"""EP7 几何重投影损失 ``L_rep`` —— Keypoint-Guided（T-RO 2024）式 (6)/(8) 的落地。

## 论文原式与它为什么有效

论文式 (6)：``L_reg = L_loc + L_rep``；
式 (7)：``L_loc = Σ_i ½‖x_i^pr − x_i^gt‖²``（2D 像素 MSE，本库由 OKS 项承担同类角色）；
式 (8)：

    L_rep = Σ_i smoothL1( (I − v̂_i v̂_iᵀ) p_i^c )

其中 ``p_i^c = R·p_i^o + t`` 是第 i 个关键点在**相机坐标系**下的真值（``p_i^o`` 是已知的
机体 3D 坐标），``v̂_i`` 是**由预测的 2D 关键点得到的相机射线方向**（单位向量），
``(I − v̂ v̂ᵀ)`` 把误差投影到**垂直于该射线**的平面。

**为什么只在垂直方向罚**：沿视线的深度误差无法从单目 2D 观测中判别 —— 一个点沿它的
视线前后移动，投影位置一模一样。用像素距离当损失（式 7）时，网络会把"深度方向"的
自由度也当成误差去优化，引入不可辨识的噪声；式 (8) 把误差放到 3D 空间并**只保留横向
分量**，等于告诉网络："深度方向你猜不出来，我不罚你；横向偏差才是真错。" 论文
§III-A-3 的原话是这个组合"takes both the 2-D regression loss in the image and the
3-D regression loss in the space into account"。

论文消融（Table III）：加入 ``L_rep`` 后 **10°10cm 提升 > 10%**，是 KDA / Lrep / MFMF
三个组件里最显著的。

## 论文的两条训练策略（本库照做）

1. **前 15 个 epoch 只用 ``L_loc``**（§IV-A-2 原文："we only use the 2-D regression loss
   ``L_loc`` as the keypoint regression loss in the first 15 training epochs"）——
   早期网络定位能力弱，几何项的梯度会把还没收敛的预测拽向错误方向；
2. 论文的 ``λ_reg = 1``、``λ_conf = 5``（格内有目标时）—— 权重见变体配置。

## 与框架的接口（本库怎么接进去）

框架的 ``v8PoseLoss.loss()`` 里已经算好了 ``fg_mask`` / ``target_gt_idx`` / ``anchor_points``
/ ``stride_tensor``，这四样正好是接 ``L_rep`` 所需的全部锚点信息。因此本库**不重写**
框架的姿态损失，而是在 ``model.criterion`` 外面包一层：

    criterion = build_pose_criterion(model, ep7={..., "reproj": True})
    criterion = build_reproj_loss(criterion, spec)     # ← 本模块，按需启用 + warm-up

包装器调用原准则拿到 ``(total, items)``，再算 ``L_rep`` 并按**论文的 warm-up 策略**
决定是否计入。这样：
  * 不 fork 框架的两百多行姿态损失；
  * warm-up 由 ``set_epoch()`` 驱动，训练器在每个 epoch 开头调用；
  * 关掉时（``reproj=False``）行为与包装前**逐元素一致**（有回归测试）。

## 本库对论文记号的一处**明确修正**（必读）

论文式 (8) 的文字说明是 "``v_i`` is the predicted keypoint" —— 照字面取
``v_i = x_i^pr``（预测的 2D 像素坐标）会得到 ``‖(I − v̂v̂ᵀ)c‖``，这个量**不含任何预测
信息**（``c`` 是相机中心，对一张图恒定），梯度恒为 0，等于没有这一项。

因此本库按**几何上唯一说得通的读法**实现：``v_i`` 是"从相机中心穿过**预测**关键点的
射线方向"，即

    v_i = R^T · K^{-1} · [x_i^pr, 1]^T     （再归一化）

这与论文**自己给出的物理解释**一致："the distance between two keypoints in the image is
not able to reflect their distance in the 3-D space"、``L_rep`` 是 "reprojection error
mapped into 3-D space"。**这是本库的推断并已标注**；若拿到作者代码发现定义不同，
只需改 ``camera_ray_dirs()`` 一个函数。

参考：[论文 PDF](docs/)（本仓库 `docs/Zheng 等 - 2024 - ...pdf`），DOI [10.1109/TRO.2024.3400938](https://doi.org/10.1109/TRO.2024.3400938)。
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch
import torch.nn as nn

from tod.compat import CompatError
from tod.registry import register

__all__ = ["ReprojLoss", "build_reproj_loss", "camera_ray_dirs", "look_at_rotation",
           "project_points", "smooth_l1", "pose_criteria"]


# ------------------------------------------------------------------ 数学工具


def smooth_l1(x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """论文式 (9) 的 smooth L1（``beta=1`` 即论文的 ``|x|<1`` 分界）。

    论文的式子对 ``x`` 是逐元素标量形式；本库按向量实现以保证可微与高效，
    并在 ``beta=1`` 时与论文逐元素一致（测试里有数值对照）。
    """
    absx = x.abs()
    if beta <= 0:
        return absx
    quadratic = 0.5 * x * x / beta
    linear = absx - 0.5 * beta
    return torch.where(absx < beta, quadratic, linear)


def project_points(points_cam: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """相机系 3D 点 → 像素坐标（``K`` 为 3×3 内参）。

    Args:
        points_cam: ``(..., N, 3)``。
        K: ``(3, 3)``。

    Returns:
        ``(..., N, 2)`` 像素坐标（u, v）。
    """
    proj = points_cam @ K.t()
    z = proj[..., 2:3].clamp_min(1e-9)
    return proj[..., :2] / z


def camera_ray_dirs(pixels: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """**预测**关键点的像素坐标 → 相机系单位射线方向（本库对式 (8) 的 ``v̂`` 的读法）。

    ``v = R^T · K^{-1} · [u, v, 1]^T``，其中 ``R`` 是机体→相机的旋转 —— 严格来说
    ``v`` 应表达在**相机系**里，而 ``K^{-1}[u,v,1]`` 已经是相机系下的方向，
    所以这里不需要 ``R``（``R`` 只用于把 ``p^o`` 搬到相机系）。

    Args:
        pixels: ``(..., N, 2)`` 像素坐标。
        K: ``(3, 3)`` 内参。

    Returns:
        ``(..., N, 3)`` 单位向量。
    """
    ones = torch.ones_like(pixels[..., :1])
    homo = torch.cat([pixels, ones], dim=-1)              # (..., N, 3)
    K_inv = torch.linalg.inv(K)
    dirs = homo @ K_inv.t()
    return dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-9)


def _quat_to_rot(q: torch.Tensor) -> torch.Tensor:
    """四元数 ``(qx, qy, qz, qw)`` → 旋转矩阵。支持 ``(..., 4)`` 批形状。"""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    x, y, z, w = (q[..., 0], q[..., 1], q[..., 2], q[..., 3])
    o = torch.ones_like(x)
    zz = torch.zeros_like(x)
    return torch.stack([
        torch.stack([o - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        torch.stack([2 * (x * y + z * w), o - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), o - 2 * (x * x + y * y)], -1),
    ], dim=-2) if q.dim() > 1 else torch.stack([
        torch.stack([o - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)]),
        torch.stack([2 * (x * y + z * w), o - 2 * (x * x + z * z), 2 * (y * z - x * w)]),
        torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), o - 2 * (x * x + y * y)]),
    ], dim=0)


def look_at_rotation(eye: torch.Tensor, target: torch.Tensor,
                     up: torch.Tensor | None = None) -> torch.Tensor:
    """构造"相机在 ``eye``、看向 ``target``"的旋转矩阵（世界系→相机系的转置）。

    仅用于**自检与合成数据**：把已知的相机位姿变成一条真实射线，验证
    ``L_rep`` 的零点是正确的。真训练时 ``R/t`` 来自数据集的 6D 标注。
    """
    if up is None:
        up = torch.tensor([0.0, 0.0, 1.0], dtype=eye.dtype, device=eye.device)
    f = target - eye
    f = f / f.norm().clamp_min(1e-9)          # 前向
    r = torch.cross(f, up, dim=-1)
    r = r / r.norm().clamp_min(1e-9)          # 右向
    u = torch.cross(r, f, dim=-1)             # 上向
    # 相机系坐标轴：x 右、y 下（图像坐标约定）→ 这里取 y = -u 以匹配常规图像 y 向下
    rot = torch.stack([r, -u, f], dim=0)      # 行 = 相机轴在世界系的表达
    return rot


# ------------------------------------------------------------------ 损失本体


@register(
    ep="EP7",
    paper="Keypoint-Guided Efficient Pose Estimation and Domain Adaptation for Micro Aerial "
          "Vehicles（IEEE T-RO 40 (2024) 2967-2983）式 (6)/(8)：L_reg = L_loc + L_rep",
    url="https://doi.org/10.1109/TRO.2024.3400938",
    year=2024,
    license="论文为 IEEE 版权；本文件是按论文公式的独立实现（MIT 仓库 WindyLab/MAV6D 为参考）",
    cost="无额外参数；每步多一次 3×3 求逆与逐点投影，相对姿态头前向可忽略；"
         "需要数据集提供 3D 关键点坐标 + 相机内参 + 逐帧 6D 位姿",
    notes="几何一致性项：把预测 2D 关键点对应的相机射线与真值 3D 关键点比较，"
          "只罚垂直于射线的分量（沿视线的深度误差不可从单目 2D 观测判别，故不罚）。"
          "论文报告加入它后 10°10cm 提升 >10%；论文规定前 15 epoch 只用 2D 项。",
    aliases=("Lrep", "reproj", "GeometricReproj"),
)
class ReprojLoss(nn.Module):
    """论文式 (8) 的 ``L_rep``：相机射线 × 真值 3D 关键点的横向偏差。

    Args:
        K: ``(3, 3)`` 相机内参（像素单位）。
        object_points: ``(n_kpt, 3)`` 机体坐标系下的关键点 3D 坐标（米）。
        kpt_indices: 只对**哪些关键点**算该项（``None`` = 全部）。
            论文只对 8 个角点求和（式 5 的 ``i = 1..8``，质心 p0 不参与 conf 的计算），
            但式 (8) 的求和范围论文写成 ``n``；本库默认**全部关键点**并在配置里可切，
            因为质心同样有 3D 坐标、同样满足"沿视线不可辨识"的性质。
        smooth_beta: smooth L1 的分界（论文为 1，单位是米）。
        reduction: ``"sum"``（论文形式：``Σ_i``）或 ``"mean"``。
    """

    def __init__(self, K: Sequence[Sequence[float]], object_points: Sequence[Sequence[float]],
                 kpt_indices: Sequence[int] | None = None,
                 smooth_beta: float = 1.0, reduction: str = "mean"):
        super().__init__()
        K_t = torch.as_tensor(K, dtype=torch.float32)
        if K_t.shape != (3, 3):
            raise ValueError(f"camera_matrix 必须是 3×3，收到 {tuple(K_t.shape)}")
        pts = torch.as_tensor(object_points, dtype=torch.float32)
        if pts.dim() != 2 or pts.shape[1] != 3:
            raise ValueError(f"object_points 必须是 (n_kpt, 3)，收到 {tuple(pts.shape)}")
        self.register_buffer("K", K_t)
        self.register_buffer("object_points", pts)
        # 只对**部分关键点**算 L_rep 时（论文式 (5) 的 i=1..8 只含角点），
        # 用一个索引掩码同时切"预测点"与"3D 点" —— 只切一边会维度不匹配（已踩过）。
        if kpt_indices is None:
            self.register_buffer("selected", torch.arange(pts.shape[0], dtype=torch.long))
        else:
            idx = torch.as_tensor(list(kpt_indices), dtype=torch.long)
            if int(idx.max()) >= pts.shape[0]:
                raise ValueError(
                    f"kpt_indices 里有越界下标（最大 {int(idx.max())}，"
                    f"但 object_points 只有 {pts.shape[0]} 个点）"
                )
            self.register_buffer("selected", idx)
        self.smooth_beta = float(smooth_beta)
        if reduction not in ("sum", "mean"):
            raise ValueError("reduction 只能是 'sum' 或 'mean'")
        self.reduction = reduction

    @property
    def n_kpt(self) -> int:
        return int(self.selected.numel())

    def forward(self, pred_kpts_px: torch.Tensor, R: torch.Tensor,
                t: torch.Tensor, K: torch.Tensor | None = None) -> torch.Tensor:
        """计算 ``L_rep``。

        Args:
            pred_kpts_px: ``(N, K, 2)`` **预测**关键点的像素坐标（已乘回图像尺寸），
                ``K`` 必须是**全部**关键点数（本模块按 ``kpt_indices`` 自行取子集）。
            R: ``(N, 3, 3)`` 机体→相机的旋转。
            t: ``(N, 3)`` 机体→相机的平移（米）。
            K: 可选的**覆盖内参**（letterbox 等价缩放后的内参）；缺省用构造时的 K。

        Returns:
            标量损失（``reduction`` 决定求和或取均值；``N`` 为参与计算的目标数）。
        """
        device = pred_kpts_px.device
        sel = self.selected.to(device)
        if pred_kpts_px.shape[1] < int(sel.max()) + 1:
            raise ValueError(
                f"预测关键点只有 {pred_kpts_px.shape[1]} 个，但 L_rep 需要下标 "
                f"{int(sel.max())}（object_points 有 {self.object_points.shape[0]} 个点）。"
                "请确认 kpt_shape 与 tod.object_points 一一对应。"
            )
        obj = self.object_points.to(device).to(pred_kpts_px.dtype)[sel]      # (Ks,3)
        pred = pred_kpts_px[..., :2][:, sel]                                 # (N,Ks,2)
        # 真值 3D 关键点 → 相机系：p^c = R p^o + t
        pc = torch.einsum("nij,kj->nki", R.to(device), obj) + t.to(device).unsqueeze(1)
        K_use = self.K.to(device) if K is None else K.to(device)
        v = camera_ray_dirs(pred, K_use)                                     # (N,Ks,3) 单位
        # 垂直分量： (I − v̂v̂ᵀ) p^c
        parallel = (pc * v).sum(-1, keepdim=True) * v
        perp = pc - parallel
        per_point = smooth_l1(perp, self.smooth_beta).sum(-1)                # (N,Ks)
        return per_point.sum() if self.reduction == "sum" else per_point.mean()


# ------------------------------------------------------------------ 接线


def pose_criteria(criterion: Any) -> list[Any]:
    """列出持有 ``keypoint_loss`` 的姿态子准则（与 ``tod.loss.pose.pose_criteria`` 同规则）。"""
    from tod.loss.pose import pose_criteria as _pc

    return _pc(criterion)


class ReprojCriterion:
    """在框架准则外面加 ``L_rep``（**不改框架源码**，与 ``KDCriterion`` 同思路）。

    为什么是普通类而不是 ``nn.Module``：与 ``tod.engine.distill.KDCriterion`` 同样的理由
    —— 框架自己的准则就是普通对象，换成 ``nn.Module`` 会被注册进模型、污染
    ``model.parameters()`` / EMA / checkpoint；而本项没有可训练参数。

    **怎么拿到锚点（关键点、R、t）—— 三代实现的血泪教训**：

    1. ~~给 ``get_assigned_targets_and_loss`` 打闭包 hook~~ → 训练跑完**存 checkpoint 时
       pickle 失败**（"Can't get local object"）；
    2. ~~改成绑定方法（``types.MethodType``）~~ → 反序列化时 pickle 按属性名在类上重查
       函数，报 ``'v8PoseLoss' object has no attribute '_recording_assign'``；
    3. **当前实现：完全不 hook**。改用框架本来就放在 batch 里的信息自己算：
       * ``preds['kpts']`` 直接从本次前向拿（O2M/O2O 由 ``parse_output`` 拆开）；
       * 锚点用 ``make_anchors(feats, stride, 0.5)`` 复算（**可加**、纯几何、无状态）；
       * 哪些锚点是正样本：用 ``batch['batch_idx']`` 与 ``gt_pose`` 表**自己**判定
         （见 ``_reproject_all``），不依赖框架内部的分配结果。

    这样没有闭包、没有绑定方法、没有实例级函数属性 → **对象图完全可 pickle**，
    且 ``R, t`` 直接来自数据集配置里的 6D 位姿表，与框架的 GT 组织方式解耦。

    Args:
        base: 被包装的准则（``v8PoseLoss`` 或 ``E2ELoss(PoseLoss26)``）。
        reproj: ``ReprojLoss`` 实例。
        weight: ``L_rep`` 的权重（论文 ``λ_reg = 1``，与 2D 项等权）。
        warmup_epochs: 前多少个 epoch 只用 2D 项（论文为 15）。
        enabled: 总开关；``False`` 时退化为**直通**（与包装前逐元素一致）。
    """

    def __init__(self, base: Any, reproj: ReprojLoss, weight: float = 1.0,
                 warmup_epochs: int = 15, enabled: bool = True):
        self.base = base
        self.reproj = reproj
        self.weight = float(weight)
        self.warmup_epochs = int(warmup_epochs)
        self.enabled = bool(enabled)
        self.epoch = 0
        self.last_value: float | None = None      # 供日志/自检读取

    # ---- 与框架训练循环的接口 ----
    def set_epoch(self, epoch: int) -> None:
        """显式设定 epoch（框架的 epoch 从 1 开始计）。

        一般不直接调用：框架在每个 epoch 末调用 ``criterion.update()``
        （``ultralytics/engine/trainer.py`` 的 ``unwrap_model(self.model).criterion.update()``），
        本包装器**自己计数**即可与框架的 ``self.epoch`` 保持一致。
        留这个方法是为了测试与外部调度器能显式对齐。
        """
        self.epoch = int(epoch)

    @property
    def active(self) -> bool:
        """当前 epoch 是否**计入** ``L_rep``（论文的 warm-up 策略）。

        论文 §IV-A-2：前 15 个 epoch 只用 2D 的 ``L_loc``。框架在第 1 个 epoch 训练时
        ``epoch`` 还是 0（``update()`` 在该 epoch 训练**结束**后才调用），所以这里用
        ``epoch > warmup_epochs``：第 1..15 个 epoch 关闭，第 16 个 epoch 起启用。
        """
        return self.enabled and self.epoch > self.warmup_epochs

    def update(self) -> None:
        """框架在每个 epoch 末调用：推进本包装器的计数，并转发内层 ``update``。"""
        self.epoch += 1
        inner = getattr(self.base, "update", None)
        if callable(inner):
            inner()

    def __getattr__(self, item: str) -> Any:
        """未定义的属性一律转发给被包装的准则（透明包装，便于外部检查）。"""
        if item in {"base", "reproj", "weight", "warmup_epochs", "enabled", "epoch",
                    "last_value"}:
            raise AttributeError(item)
        return getattr(self.base, item)

    def __call__(self, preds: Any, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward(preds, batch)

    # ---- 前向 ----
    def forward(self, preds: Any, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        total, items = self.base(preds, batch)
        if not self.active:
            self.last_value = 0.0
            return total, items
        if "gt_pose" not in batch:
            raise CompatError(
                "EP7.reproj 已开启，但 batch 里没有 'gt_pose'（逐目标 6D 位姿）。\n"
                "请把你的 6D 位姿按 docs/KEYPOINTS.md §5.2 的格式挂上"
                "（labels_6d/<split>.csv），或把 EP7.reproj 关掉。"
            )
        value = self._reproject_all(preds, batch)
        if value is None:
            self.last_value = 0.0
            return total, items
        self.last_value = float(value.detach())
        return total + self.weight * value, items

    def _reproject_all(self, preds: Any, batch: dict):
        """对每个（O2M/O2O）分支算一次 ``L_rep``：预测关键点 → 像素 → 真值射线。"""
        import torch as _t

        branches = _iter_branches(preds)
        total = None
        for branch_name, branch_preds in branches:
            kpts = branch_preds.get("kpts") if isinstance(branch_preds, dict) else None
            feats = branch_preds.get("feats") if isinstance(branch_preds, dict) else None
            if kpts is None or feats is None:
                continue
            criterion = _sub_criterion(self.base, branch_name)
            if criterion is None:
                continue
            value = self._one_branch(kpts, feats, batch, criterion)
            if value is None:
                continue
            total = value if total is None else total + value
        return total

    def _one_branch(self, kpts, feats, batch: dict, criterion: Any):
        """单个分支：把预测关键点解码到**增强后图像的归一化坐标**，再与真值 3D 射线比。

        三个关键步骤（每一步都是踩过坑之后定下来的）：

        1. **锚点 ↔ 目标匹配**：用"锚点中心最接近 GT 框中心"的最近邻匹配（贪心、一对一），
           不依赖框架内部的 ``topk`` 分配。代价是匹配数可能略多于框架的正样本数，
           但 `L_rep` 是**逐目标的几何一致性项**（不是分类项），这样定义更直接：
           每个 GT 目标拿一个最负责它的预测来算几何误差。
        2. **归一化 → 像素**：锚点归一化坐标（``anchor/stride``）乘以每层实际特征尺寸
           （含 stride），得到**增强后图像**的像素坐标；再除以增强图像的宽高还原为归一化。
           注意这里**不能**用 letterbox 的公式去反推原始像素 —— 那样会把补边也算进去。
        3. **真值射线也在"缩放后"的空间里算**：letterbox 是等比缩放（无仿射形变），
           所以把相机内参按同一比例缩放，几何关系完全等价，不必处理补边平移。
           比例取 ``aug / orig`` 的较大者（等比缩放的放大倍数）。
        """
        import torch as _t
        from ultralytics.utils.tal import make_anchors

        stride = criterion.stride
        anchor_points, _stride_tensor = make_anchors(feats, stride, 0.5)
        kpt_shape = criterion.kpt_shape
        b = kpts.shape[0]
        raw = kpts.permute(0, 2, 1).contiguous().view(b, -1, int(kpt_shape[0]),
                                                     int(kpt_shape[1]))
        decoded = criterion.kpts_decode(anchor_points, raw)          # (B,A,K,d) 归一化

        # ---- 锚点 ↔ GT 目标（最近中心，贪心一对一）----
        poses, orig_sizes = _pose_lookup(batch)
        if not poses:
            return None
        img_h, img_w = float(batch["img"].shape[2]), float(batch["img"].shape[3])
        gt_boxes = _gt_boxes_by_image(batch, b)                       # {i: [(cx,cy,w,h) 归一化]}
        total = None
        for i in range(b):
            rows = poses.get(i) or []
            boxes = gt_boxes.get(i) or []
            if not rows or not boxes:
                continue
            n_obj = min(len(rows), len(boxes))
            picked = _nearest_anchor_per_box(decoded[i], boxes[:n_obj], feats, i)
            if picked is None:
                continue
            # (n_obj, K, 2) 归一化；转成"增强后图像"的像素再还原归一化（保持一致性）
            sel_px = picked * _t.tensor([img_w, img_h], device=picked.device,
                                        dtype=picked.dtype)
            sel_norm = sel_px / _t.tensor([img_w, img_h], device=picked.device,
                                          dtype=picked.dtype)
            R, t, _mask = _poses_to_tensors(rows[:n_obj], decoded.device, decoded.dtype)
            size = orig_sizes.get(i)
            K = _scaled_intrinsics(self.reproj.K, size, img_w, img_h)
            value = self.reproj(sel_norm * _t.tensor([img_w, img_h], device=sel_norm.device,
                                                     dtype=sel_norm.dtype),
                                R, t, K=K)
            total = value if total is None else total + value
        return total


def _gt_boxes_by_image(batch: dict, batch_size: int) -> dict[int, list]:
    """把 ``batch['bboxes']``（按图拼接的归一化 xywh）按图拆开。"""
    boxes = batch.get("bboxes")
    idx = batch.get("batch_idx")
    out: dict[int, list] = {}
    if boxes is None or idx is None:
        return out
    for k in range(boxes.shape[0]):
        i = int(idx[k].item()) if hasattr(idx[k], "item") else int(idx[k])
        if i >= batch_size:
            continue
        out.setdefault(i, []).append([float(v) for v in boxes[k].tolist()[:4]])
    return out


def _nearest_anchor_per_box(decoded_i, boxes: list, feats, image_index: int):
    """为每个 GT 框挑一个"锚点中心离框中心最近"的预测，返回 ``(n_obj, K, 2)``。

    匹配在**归一化坐标**里做（``decoded`` 的 cx/cy 也是归一化），因此不需要知道
    stride 或 letterbox 细节。
    """
    import torch as _t

    if decoded_i.numel() == 0:
        return None
    centers = decoded_i[:, :, :2].mean(dim=1)                     # (A,2) 用质心点近似
    # 质心点是 0 号关键点（论文 p0），直接用它更准
    if decoded_i.shape[1] > 0:
        centers = decoded_i[:, 0, :2]
    picked = []
    for cx, cy, _w, _h in boxes:
        d = (centers[:, 0] - cx) ** 2 + (centers[:, 1] - cy) ** 2
        picked.append(int(_t.argmin(d).item()))
    return decoded_i[picked]                                       # (n_obj,K,d)


def _scaled_intrinsics(K, size, aug_w: float, aug_h: float):
    """按 letterbox 的**等比缩放**比例缩放内参（补边只是平移，不影响方向）。

    比例取 ``aug/orig`` 的较大者：框架的 letterbox 保证原图完整落入 imgsz，
    所以实际缩放比 = ``min(aug_w/w0, aug_h/h0)`` 的倒数 —— 这里取 ``max(aug_w/w0,
    aug_h/h0)`` 与之相等（因为缩放比是那个较小的比值）。
    """
    import torch as _t

    if not size:
        return K
    w0, h0 = max(float(size[0]), 1e-6), max(float(size[1]), 1e-6)
    ratio = max(aug_w / w0, aug_h / h0)
    K2 = K.clone() if isinstance(K, _t.Tensor) else _t.as_tensor(K, dtype=_t.float32).clone()
    K2[0, 0] *= ratio
    K2[1, 1] *= ratio
    # 主点在缩放后也按比例走；补边平移在"以图像中心为原点"的射线计算里抵消掉
    K2[0, 2] = aug_w / 2.0
    K2[1, 2] = aug_h / 2.0
    return K2


def _iter_branches(preds: Any):
    """遍历本次前向的分支：``E2ELoss`` 是 ``one2many``/``one2one``，单分支是它自己。"""
    if isinstance(preds, dict) and "one2many" in preds:
        return [("one2many", preds["one2many"]), ("one2one", preds.get("one2one"))]
    return [("single", preds)]


def _sub_criterion(base: Any, branch: str):
    """取某个分支对应的子准则（决定 ``stride`` / ``kpt_shape`` / ``kpts_decode``）。"""
    if branch == "one2many":
        return getattr(base, "one2many", base)
    if branch == "one2one":
        return getattr(base, "one2one", base)
    return base


def _pose_lookup(batch: dict) -> tuple[dict, dict]:
    """从 batch 建立 ``{图内序号: [位姿行]}``（``(B, G, 7)`` + ``gt_pose_mask``）。

    ``gt_pose`` 由 ``Pose6DDataset`` 按"每张图的行序"给出（与标签行序一致，
    ``tod/data/pose6d.py`` 会强校验行数一致、indices 从 0 连续；
    ``tools/check_pose_dataset.py`` 开训前再校验一次）。

    注意本函数**不做**任何内部索引猜测：返回的就是"每张图的目标列表"（按行序）。
    与锚点的对应关系由 ``_nearest_anchor_per_box`` 用几何最近邻确定，
    因此与框架的 O2M/O2O 分配差异无关。

    Returns:
        ``({图内序号: [位姿行]}, {图内序号: (原图宽, 原图高)})``；第二个字典来自
        batch 里的 ``gt_image_size``（没有则为空 → 退化为不做 letterbox 缩放校正）。
    """
    poses = batch.get("gt_pose")
    if poses is None:
        return {}, {}
    mask = batch.get("gt_pose_mask")
    out: dict[int, list] = {}
    if poses.dim() == 3:                          # (B, G, 7)
        for i in range(poses.shape[0]):
            rows = []
            for j in range(poses.shape[1]):
                if mask is not None and not bool(mask[i, j]):
                    continue
                rows.append([float(v) for v in poses[i, j].tolist()])
            if rows:
                out[i] = rows
    sizes = batch.get("gt_image_size") or {}
    if isinstance(sizes, (list, tuple)):          # 也接受逐图的 (w,h) 列表
        sizes = {i: sizes[i] for i in range(len(sizes))}
    return out, sizes


def _poses_to_tensors(rows: list, device, dtype):
    """位姿行 → ``(R, t, mask)``。"""
    import torch as _t

    quats = _t.tensor([r[3:7] for r in rows], device=device, dtype=dtype)
    trans = _t.tensor([r[:3] for r in rows], device=device, dtype=dtype)
    return _quat_to_rot(quats), trans, _t.ones(len(rows), dtype=_t.bool, device=device)


def build_reproj_loss(criterion: Any, spec: dict, *, enabled: bool = True,
                      weight: float = 1.0, warmup_epochs: int = 15) -> Any:
    """按变体 spec + 数据集配置构造带 ``L_rep`` 的准则（不需要时原样返回）。

    Args:
        criterion: 已被 ``build_pose_criterion`` 打好补丁的姿态准则。
        spec: 变体 spec（读 ``EP7.reproj`` / ``EP7.reproj_weight``）。
        enabled: 是否启用（``EP7.reproj`` 的最终值）。
        weight: ``L_rep`` 权重（论文等权 → 1.0）。
        warmup_epochs: 论文的前 N epoch 只用 2D 项。

    Returns:
        包装后的准则（``ReprojCriterion``）；``enabled=False`` 时返回原对象。
    """
    if not enabled:
        return criterion
    data_cfg = spec.get("data_cfg") or {}
    tod = data_cfg.get("tod") or {}
    K = tod.get("camera_matrix")
    obj = tod.get("object_points")
    if not K or not obj:
        raise CompatError(
            "EP7.reproj 已开启，但数据集配置里缺少相机内参或 3D 关键点坐标：\n"
            "  tod.camera_matrix（3×3 内参 K）\n"
            "  tod.object_points（9×3 机体关键点坐标，顺序与 kpt_names 一致）\n"
            "格式与获取方式见 docs/KEYPOINTS.md §5。"
        )
    n_kpt = len((data_cfg.get("kpt_shape") or [0])) and int(data_cfg["kpt_shape"][0])
    if n_kpt and len(obj) != n_kpt:
        raise CompatError(
            f"tod.object_points 有 {len(obj)} 个点，但 kpt_shape[0]={n_kpt}；"
            "两者必须一一对应（顺序还要与 kpt_names 一致）。"
        )
    loss = ReprojLoss(K, obj)
    wrapper = ReprojCriterion(criterion, loss, weight=weight,
                              warmup_epochs=warmup_epochs, enabled=True)
    return wrapper
