"""EP7 几何重投影损失 ``L_rep`` —— Keypoint-Guided（T-RO 2024）式 (6)/(8) 的落地。"""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch
import torch.nn as nn

from tod.compat import CompatError
from tod.registry import register

__all__ = ["ReprojLoss", "build_reproj_loss", "camera_ray_dirs", "look_at_rotation",
           "project_points", "smooth_l1", "pose_criteria"]


def smooth_l1(x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """论文式 (9) 的 smooth L1（``beta=1`` 即论文的 ``|x|<1`` 分界）。"""
    absx = x.abs()
    if beta <= 0:
        return absx
    quadratic = 0.5 * x * x / beta
    linear = absx - 0.5 * beta
    return torch.where(absx < beta, quadratic, linear)


def project_points(points_cam: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """相机系 3D 点 → 像素坐标（``K`` 为 3×3 内参）。"""
    proj = points_cam @ K.t()
    z = proj[..., 2:3].clamp_min(1e-9)
    return proj[..., :2] / z


def camera_ray_dirs(pixels: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """**预测**关键点的像素坐标 → 相机系单位射线方向（本库对式 (8) 的 ``v̂`` 的读法）。"""
    ones = torch.ones_like(pixels[..., :1])
    homo = torch.cat([pixels, ones], dim=-1)
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
    """构造"相机在 ``eye``、看向 ``target``"的旋转矩阵（世界系→相机系的转置）。"""
    if up is None:
        up = torch.tensor([0.0, 0.0, 1.0], dtype=eye.dtype, device=eye.device)
    f = target - eye
    f = f / f.norm().clamp_min(1e-9)
    r = torch.cross(f, up, dim=-1)
    r = r / r.norm().clamp_min(1e-9)
    u = torch.cross(r, f, dim=-1)
    rot = torch.stack([r, -u, f], dim=0)
    return rot


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
    """论文式 (8) 的 ``L_rep``：相机射线 × 真值 3D 关键点的横向偏差。"""

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
        """计算 ``L_rep``。"""
        device = pred_kpts_px.device
        sel = self.selected.to(device)
        if pred_kpts_px.shape[1] < int(sel.max()) + 1:
            raise ValueError(
                f"预测关键点只有 {pred_kpts_px.shape[1]} 个，但 L_rep 需要下标 "
                f"{int(sel.max())}（object_points 有 {self.object_points.shape[0]} 个点）。"
                "请确认 kpt_shape 与 tod.object_points 一一对应。"
            )
        obj = self.object_points.to(device).to(pred_kpts_px.dtype)[sel]
        pred = pred_kpts_px[..., :2][:, sel]
        pc = torch.einsum("nij,kj->nki", R.to(device), obj) + t.to(device).unsqueeze(1)
        K_use = self.K.to(device) if K is None else K.to(device)
        v = camera_ray_dirs(pred, K_use)
        # 只惩罚垂直于相机射线的偏差，沿射线的深度分量不计入几何损失。
        parallel = (pc * v).sum(-1, keepdim=True) * v
        perp = pc - parallel
        per_point = smooth_l1(perp, self.smooth_beta).sum(-1)
        return per_point.sum() if self.reduction == "sum" else per_point.mean()


def pose_criteria(criterion: Any) -> list[Any]:
    """列出持有 ``keypoint_loss`` 的姿态子准则（与 ``tod.loss.pose.pose_criteria`` 同规则）。"""
    from tod.loss.pose import pose_criteria as _pc

    return _pc(criterion)


class ReprojCriterion:
    """在框架准则外面加 ``L_rep``（**不改框架源码**，与 ``KDCriterion`` 同思路）。"""

    def __init__(self, base: Any, reproj: ReprojLoss, weight: float = 1.0,
                 warmup_epochs: int = 15, enabled: bool = True):
        self.base = base
        self.reproj = reproj
        self.weight = float(weight)
        self.warmup_epochs = int(warmup_epochs)
        self.enabled = bool(enabled)
        self.epoch = 0
        self.last_value: float | None = None
        self._terms = 0
        self._sum = 0.0
        self._count = 0
        self.epoch_mean: float | None = None

    def reproj_terms(self) -> int:
        """累计成功计算的几何项数量（``L_rep`` 是否真的跑过，看这个计数最客观）。"""
        return int(self._terms)

    def set_epoch(self, epoch: int) -> None:
        """显式设定 epoch（框架的 epoch 从 1 开始计）。"""
        self.epoch = int(epoch)
        self._sum = 0.0
        self._count = 0

    @property
    def active(self) -> bool:
        """当前 epoch 是否**计入** ``L_rep``（论文的 warm-up 策略）。"""
        # epoch 表示已完成的轮数；前 warmup_epochs 轮只计算二维损失。
        return self.enabled and self.epoch > self.warmup_epochs

    def update(self) -> None:
        """框架在每个 epoch 末调用：记录本 epoch 的 L_rep 均值，再推进计数。"""
        self.epoch_mean = (self._sum / self._count) if self._count else None
        self._sum = 0.0
        self._count = 0
        self.epoch += 1
        inner = getattr(self.base, "update", None)
        if callable(inner):
            inner()

    def __getattr__(self, item: str) -> Any:
        """未定义的属性一律转发给被包装的准则（透明包装，便于外部检查）。"""
        if item in {"base", "reproj", "weight", "warmup_epochs", "enabled", "epoch",
                    "last_value", "epoch_mean", "_sum", "_count", "_terms"}:
            raise AttributeError(item)
        return getattr(self.base, item)

    def __call__(self, preds: Any, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward(preds, batch)

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
        self._sum += self.last_value
        self._count += 1
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
        """单个分支：把预测关键点解码到**增强后图像的像素坐标**，再与真值 3D 射线比。"""
        import torch as _t
        from ultralytics.utils.tal import make_anchors

        stride = criterion.stride
        anchor_points, stride_tensor = make_anchors(feats, stride, 0.5)
        kpt_shape = criterion.kpt_shape
        b = kpts.shape[0]
        raw = kpts.permute(0, 2, 1).contiguous().view(b, -1, int(kpt_shape[0]),
                                                     int(kpt_shape[1]))
        decoded_grid = criterion.kpts_decode(anchor_points, raw)[..., :2]
        img_h, img_w = float(batch["img"].shape[2]), float(batch["img"].shape[3])
        scale = _t.tensor([img_w, img_h], device=decoded_grid.device,
                          dtype=decoded_grid.dtype)
        # kpts_decode 输出网格单位，必须先乘 stride 还原像素，再与相机内参配合。
        decoded_px = decoded_grid * stride_tensor.view(1, -1, 1, 1)
        decoded_norm = decoded_px / scale

        poses, orig_sizes = _pose_lookup(batch)
        if not poses:
            return None
        gt_boxes = _gt_boxes_by_image(batch, b)
        total = None
        for i in range(b):
            rows = poses.get(i) or []
            boxes = gt_boxes.get(i) or []
            if not rows or not boxes:
                continue
            n_obj = min(len(rows), len(boxes))
            idxs = _nearest_anchor_indices_per_box(decoded_norm[i, :, 0, :2],
                                                   boxes[:n_obj])
            if idxs is None:
                continue
            index = _t.tensor(idxs, device=decoded_px.device, dtype=_t.long)
            sel_px = decoded_px[i, index]
            R, t, _mask = _poses_to_tensors(rows[:n_obj], decoded_px.device,
                                            decoded_px.dtype)
            size = orig_sizes.get(i)
            K = _scaled_intrinsics(self.reproj.K, size, img_w, img_h)
            value = self.reproj(sel_px, R, t, K=K)
            if value is not None:
                self._terms += 1
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


def _nearest_anchor_indices_per_box(centers_norm, boxes: list):
    """为每个 GT 框挑一个"归一化中心最近"的锚点，返回下标列表。"""
    import torch as _t

    if centers_norm.numel() == 0:
        return None
    picked = []
    for cx, cy, _w, _h in boxes:
        d = (centers_norm[:, 0] - cx) ** 2 + (centers_norm[:, 1] - cy) ** 2
        picked.append(int(_t.argmin(d).item()))
    return picked


def _nearest_anchor_per_box(decoded_i, boxes: list, feats=None, image_index: int = 0):
    """兼容旧接口：返回每个框对应的预测点（``(n_obj,K,d)``）。"""
    idxs = _nearest_anchor_indices_per_box(decoded_i[:, 0, :2], boxes)
    if idxs is None:
        return None
    return decoded_i[idxs]


def _scaled_intrinsics(K, size, aug_w: float, aug_h: float):
    """按训练增强的**等比缩放 + 居中补边**缩放内参。"""
    import torch as _t

    if size is None:
        return K
    try:
        w0, h0 = float(size[0]), float(size[1])
    except (IndexError, TypeError, ValueError):
        return K
    if w0 <= 0.0 or h0 <= 0.0:
        return K
    w0, h0 = max(w0, 1e-6), max(h0, 1e-6)
    gain = min(aug_w / w0, aug_h / h0)
    pad_x = (aug_w - w0 * gain) / 2.0
    pad_y = (aug_h - h0 * gain) / 2.0
    K2 = K.clone() if isinstance(K, _t.Tensor) else _t.as_tensor(K, dtype=_t.float32).clone()
    K2[0, 0] *= gain
    K2[1, 1] *= gain
    K2[0, 2] = float(K[0, 2]) * gain + pad_x
    K2[1, 2] = float(K[1, 2]) * gain + pad_y
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
    """从 batch 建立 ``{图内序号: [位姿行]}``。"""
    import torch as _t

    poses = batch.get("gt_pose")
    if poses is None:
        return {}, {}
    mask = batch.get("gt_pose_mask")
    out: dict[int, list] = {}

    def _rows_of(pose_img, mask_img) -> list:
        rows = []
        n = int(pose_img.shape[0]) if hasattr(pose_img, "shape") else len(pose_img)
        for j in range(n):
            if mask_img is not None:
                keep = bool(mask_img[j]) if hasattr(mask_img, "__getitem__") else bool(mask_img)
                if not keep:
                    continue
            rows.append([float(v) for v in _t.as_tensor(pose_img[j]).reshape(-1).tolist()])
        return rows

    if isinstance(poses, (list, tuple)):
        for i, pose_img in enumerate(poses):
            mask_img = None
            if isinstance(mask, (list, tuple)) and i < len(mask):
                mask_img = mask[i]
            elif mask is not None:
                mask_img = mask
            rows = _rows_of(pose_img, mask_img)
            if rows:
                out[i] = rows
    elif hasattr(poses, "dim") and poses.dim() == 3:
        for i in range(poses.shape[0]):
            mask_img = mask[i] if mask is not None and hasattr(mask, "__getitem__") else None
            rows = _rows_of(poses[i], mask_img)
            if rows:
                out[i] = rows

    sizes = batch.get("gt_image_size") or {}
    if isinstance(sizes, (list, tuple)):
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
    """按变体 spec + 数据集配置构造带 ``L_rep`` 的准则（不需要时原样返回）。"""
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
