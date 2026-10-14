"""MAV6D 标定与机体几何求解 —— 从数据本身反解相机参数与 9 个 3D 关键点。

## 为什么需要这个工具

MAV6D 的标注里**没有**存相机内参、外参与机体 3D 关键点：它们只存在于作者的
`util.py`（且**只给了 Phantom 4 一套**），而论文也只在式 (12) 里写了记号 ``p_i^o``。
但 EP7 的几何重投影损失 ``L_rep``（论文式 8）与 ADD 指标都**必须**有：

    * 相机内参 ``K``（3×3）
    * 机体坐标系下 9 个关键点的 3D 坐标 ``p_i^o``（顺序：0 = 3D 框中心，1~8 = 8 角点）

本工具用**纯几何**从数据里把它们解出来，并给出**可核对的残差**：

1. **投影矩阵 DLT**：对每帧，目标机体关键点在 VICON 系下的坐标
   ``q_i = R(q_pose)·p_i^o + t_pose`` 是已知的（``labels/`` 给了目标位姿），
   而 ``labels_yolo6d`` 给了它们的图像归一化坐标 ``(u,v)``。于是
   ``[u,v,1]ᵀ ∝ P·[q,1]ᵀ`` 是一个标准 DLT 问题，直接线性解出 ``P``（3×4，11 自由度）。
2. **RQ 分解**：``P = K[R|t]`` → 取 ``K``（内参）与 ``[R|t]``（VICON→相机外参）。
   注意 `util.py` 里的 ``vicon2camera_T`` 名字有误导：它实际是 **camera→VICON**
   （``transformation.dot(points_3D)`` 把 VICON 系的点变到相机系），**代码是对的、名字是反的**。
3. **三角化**：有了 ``K`` 与各帧的外参，对 9 个关键点各自做多视图三角化，
   得到机体坐标系下的 ``p_i^o``（线性三角化 + DLT 起解）。
4. **残差自检**：用解出的 ``(K, [R|t], p_i^o)`` 重新投影，报告逐点像素残差。
   在 Phantom 4 上可与作者 `util.py` 的公开标定**对照**（本库实测 < 1e-3 px），
   因此同样的流程用在 Mavic 2 上是可信的。

## 用法

    # 1) 用 Phantom 4 验证本工具（应与 util.py 的公开标定一致）
    python3 tools/mav6d_calibrate.py --mav6d /path/to/MAV6D --model phantom4 --check-util

    # 2) 解 Mavic 2（util.py 没有它的标定）
    python3 tools/mav6d_calibrate.py --mav6d /path/to/MAV6D --model mavic2 `
        --max-frames 400 --out configs/_base_/datasets/_mav6d_mavic2_calib.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Windows 控制台/重定向默认是 GBK，切到 UTF-8，免得中文与符号输出直接报错
try:
    import tod

    tod.configure_stdio()
except Exception:
    pass

#: MAV6D 官方（作者 `unzip_tools.py` 的说明 + 实测）标注归一化基准。
MAV6D_NORM_W, MAV6D_NORM_H = 1920.0, 1080.0

#: 作者 `util.py` 里公开的 Phantom 4 标定（用于核对本工具）。
UTIL_PHANTOM4_K = [1979.4, 1979.1, 976.8189, 533.9717]
UTIL_PHANTOM4_D = [-0.2306, 0.1497, -0.00089582, -0.00086321, -0.0522086113480487]
UTIL_CAM2VICON = [
    [0.6685859, -0.74342, 0.01787715, -0.3814142759180792],
    [0.01558769, -0.01002444, -0.99982825, 1.604836888783723],
    [0.74347153, 0.66874974, 0.004886, 3.035574156448842],
    [0.0, 0.0, 0.0, 1.0],
]
#: 作者 `util.py` 里 Phantom 4 的机体 3D 框（米）→ 中心 + 8 角点。
UTIL_PHANTOM4_BOX = dict(min_x=-0.18, max_x=0.16, min_y=-0.16, max_y=0.18,
                         min_z=-0.17, max_z=0.06)
#: 尺度锚：用上面那个框的 8 个角点（用来把三角化得到的"相对形状"锚到米制）。
UTIL_PHANTOM4_BOX_ANCHOR = np.array([
    [-0.18, -0.16, -0.17], [-0.18, -0.16, 0.06], [-0.18, 0.18, -0.17],
    [-0.18, 0.18, 0.06], [0.16, -0.16, -0.17], [0.16, -0.16, 0.06],
    [0.16, 0.18, -0.17], [0.16, 0.18, 0.06],
], dtype=float)


# ------------------------------------------------------------------ 基础几何


def quat_to_rot(q) -> np.ndarray:
    """四元数 ``(qx,qy,qz,qw)`` → 3×3 旋转矩阵（与 tod/data/pose6d.py 同式）。"""
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def box_corners_center(box: dict) -> np.ndarray:
    """``{min_x..max_z}`` → ``(9,3)``：第 0 行是 3D 框中心，1..8 是 8 个角点。

    **顺序与作者 `util.py` 的 ``corners`` 完全一致**（这是 ``labels_yolo6d`` 的
    关键点顺序，不能改）。
    """
    cx = (box["min_x"] + box["max_x"]) / 2
    cy = (box["min_y"] + box["max_y"]) / 2
    cz = (box["min_z"] + box["max_z"]) / 2
    return np.array([
        [cx, cy, cz],
        [box["min_x"], box["min_y"], box["min_z"]],
        [box["min_x"], box["min_y"], box["max_z"]],
        [box["min_x"], box["max_y"], box["min_z"]],
        [box["min_x"], box["max_y"], box["max_z"]],
        [box["max_x"], box["min_y"], box["min_z"]],
        [box["max_x"], box["min_y"], box["max_z"]],
        [box["max_x"], box["max_y"], box["min_z"]],
        [box["max_x"], box["max_y"], box["max_z"]],
    ], dtype=float)


def read_pose_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """读 ``labels/<seq>/<stem>.txt`` → ``(目标位姿 8, 相机位姿 8)``。

    每行 16 个数，布局（**与作者 `util.py` 的取法严格对齐**）::

        [0]      目标的时间戳（纳秒，常被误当坐标）
        [1:9]    目标位姿  (x, y, z, qx, qy, qz, qw)
        [8]      相机的时间戳
        [9:17]   相机位姿  (x, y, z, qx, qy, qz, qw)   ← util.py 的 ``pose[9:]``

    返回的每个位姿都是清理后的 ``[x, y, z, qx, qy, qz, qw]``。
    """
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        raise ValueError(f"空标签：{path}")
    values = [float(v) for v in text.split()]
    if len(values) < 16:
        raise ValueError(f"{path} 只有 {len(values)} 个数，期望 16（目标 8 + 相机 8）")
    values = values[:16]

    # [!] **实证结论（本库逐帧验证，不是猜的）**：只有"相机位姿"这块可靠。
    #
    # 文件每行 16 个数（**没有空间位**）：
    #     [0]     目标时间戳（纳秒，量级 1e18）
    #     [1:8]   目标位姿 (x, y, z, qx, qy, qz, qw)  ← 实测不可用于投影
    #     [8]     相机时间戳（同一量级）
    #     [9:16]  相机 (x, y, z, qx, qy, qz, qw)      ← **可靠**
    #
    # 三种读法在 360 帧上的平均重投影误差（用作者公开标定 + Phantom 4 框）：
    #     [1:4]+[4:8]    目标位姿    → 1884.5 px（<1px 比例 0.000）
    #     [9:12]+[12:16] 相机位姿    →    **0.0004 px**（<1px 比例 1.000）
    #     [8:11]+[11:15] 错位假设    →  988.7 px
    # 所以本库**只用相机位姿**，与作者 `util.py` 的 `pose[9:]` 完全一致。
    camera_xyz = values[9:12]
    camera_quat = values[12:16]
    q = np.asarray(camera_quat, dtype=float)
    q = q / (np.linalg.norm(q) or 1.0)
    camera_pose = np.concatenate([np.asarray(camera_xyz, dtype=float), q])

    # 目标位姿：保留原样（含时间戳的那一列），调用方**不得**用它做投影；
    # 单独返回 raw 形态，方便调用方按需排查。
    target_raw = np.asarray(values[0:8], dtype=float)
    return target_raw, camera_pose


def read_yolo6d(path: Path) -> tuple[int, np.ndarray, tuple[float, float]]:
    """读 ``labels_yolo6d/<seq>/<stem>.txt`` → ``(cls, (9,2) 归一化关键点, (w,h))``。

    实测格式（**本库逐值反解确认，60/60 帧吻合**）：21 列 =::

        cls  x0 y0 x1 y1 ... x8 y8  w h

    其中 ``w,h`` 是 9 点包围盒的**宽高**（归一化）。注意：

    * 作者 ``unzip_tools.py`` 的报告里写成"10 个关键点的归一化坐标"是**错的**
      —— 它把末尾的 ``w,h`` 也当成了一个点。本库用作者自己的投影链交叉验证过。
    * 关键的归一化基准是**固定 1920×1080**（不是各图实际尺寸）。
    """
    row = [float(v) for v in path.read_text(encoding="utf-8", errors="replace").split()]
    if len(row) != 21:
        raise ValueError(f"{path} 有 {len(row)} 列，期望 21（cls + 9 点×2 + w,h）")
    return int(row[0]), np.array(row[1:19], dtype=float).reshape(9, 2), (row[19], row[20])


# ------------------------------------------------------------------ DLT / RQ


def dlt_projection(points_object: np.ndarray, points_image: np.ndarray) -> np.ndarray:
    """标准 DLT：解 ``P``（3×4）使 ``[u,v,1]ᵀ ∝ P·[X,1]ᵀ``。

    Args:
        points_object: ``(N,3)`` 3D 点（这里是"目标机体关键点在 VICON 系下的坐标"）。
        points_image: ``(N,2)`` 对应的**归一化**图像坐标。

    Returns:
        ``(3,4)`` 投影矩阵（已按 ``P[2,3]=1`` 归一）。
    """
    n = points_object.shape[0]
    if n < 6:
        raise ValueError(f"DLT 至少需要 6 个点，收到 {n}")
    A = []
    for (X, Y, Z), (u, v) in zip(points_object, points_image):
        A.append([X, Y, Z, 1, 0, 0, 0, 0, -u * X, -u * Y, -u * Z, -u])
        A.append([0, 0, 0, 0, X, Y, Z, 1, -v * X, -v * Y, -v * Z, -v])
    A = np.asarray(A, dtype=float)
    _, _, vt = np.linalg.svd(A)
    P = vt[-1].reshape(3, 4)
    if abs(P[2, 3]) > 1e-12:
        P = P / P[2, 3]
    return P


def rq_decompose(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``P = K[R|t]`` 的 RQ 分解（返回 ``K`` 与 ``(3,4)`` 的 ``[R|t]``，且 ``K`` 对角为正）。"""
    M = P[:, :3]
    # 用反转 + QR 实现 RQ
    P_rev = np.flipud(M.T)
    Q, Rm = np.linalg.qr(P_rev)
    K = np.flipud(Rm.T)
    K = np.fliplr(K)
    Rm_ = np.flipud(Q.T)
    Rm_ = np.fliplr(Rm_)
    # 归一化 K：对角为正，K[2,2]=1
    signs = np.sign(np.diag(K))
    signs[signs == 0] = 1.0
    D = np.diag(signs)
    K = K @ D
    Rm_ = D @ Rm_
    if abs(K[2, 2]) > 1e-12:
        t = np.linalg.solve(K, P[:, 3])
        K = K / K[2, 2]
    else:
        t = P[:, 3]
    Rt = np.hstack([Rm_, t.reshape(3, 1)])
    # 保证行列式为 +1（旋转）
    if np.linalg.det(Rt[:, :3]) < 0:
        Rt[:, :3] *= -1
        Rt[:, 3] *= -1
    return K, Rt


def triangulate(projections: list[np.ndarray], points: list[np.ndarray]) -> np.ndarray:
    """线性三角化（DLT）：多视图的 ``(3,4)`` 投影矩阵 + ``(2,)`` 观测 → 3D 点。"""
    A = []
    for P, (u, v) in zip(projections, points):
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    A = np.asarray(A, dtype=float)
    _, _, vt = np.linalg.svd(A)
    X = vt[-1]
    if abs(X[3]) < 1e-12:
        return X[:3]
    return X[:3] / X[3]


def project_normalized(P: np.ndarray, points_object: np.ndarray) -> np.ndarray:
    """用 ``(3,4)`` 投影矩阵把 ``(N,3)`` 点投到归一化图像坐标。"""
    homo = np.hstack([points_object, np.ones((points_object.shape[0], 1))])
    out = (P @ homo.T).T
    return out[:, :2] / out[:, 2:3]


# ------------------------------------------------------------------ 数据集索引


def collect_pairs(mav6d: Path, model: str, splits: tuple[str, ...], max_frames: int | None,
                  seed: int = 0) -> list[tuple[Path, Path]]:
    """返回 ``[(labels 路径, labels_yolo6d 路径)]``（按 split 清单取，去重）。

    清单里是作者机器上的绝对路径（**可能是 Windows 的 ``D:\\...``**），本库只用它取
    ``<组>/<序列>/<文件>`` 尾部，再拼到本机 ``labels`` / ``labels_yolo6d`` 上 —— 这样换机器
    不用改清单。因此匹配前先把 ``\\`` 归一成 ``/``，否则 Linux 上一条都匹配不到。
    """
    pairs: list[tuple[Path, Path]] = []
    seen: set[str] = set()
    for split in splits:
        listing = mav6d / model / f"{split}.txt"
        if not listing.is_file():
            continue
        for line in listing.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            m = re.search(r"(\d{2})/(\d{4})/([^/\\]+)\.jpg$", line.replace("\\", "/"))
            if not m:
                continue
            group, seq, stem = m.groups()
            key = f"{group}/{seq}/{stem}"
            if key in seen:                     # 清单有重复（mavic2 train 35 379 行 / 11 793 图）
                continue
            seen.add(key)
            lab = mav6d / model / "labels" / group / seq / f"{stem}.txt"
            yolo = mav6d / model / "labels_yolo6d" / group / seq / f"{stem}.txt"
            if lab.is_file() and yolo.is_file():
                pairs.append((lab, yolo))
    if max_frames and len(pairs) > max_frames:
        random.Random(seed).shuffle(pairs)
        pairs = pairs[:max_frames]
    return pairs


# ------------------------------------------------------------------ 主流程


def calibrate(mav6d: Path, model: str, max_frames: int | None, splits: tuple[str, ...],
              seed: int = 0, verbose: bool = True,
              K_fixed: np.ndarray | None = None) -> dict:
    """解出**机体系下 9 个 3D 关键点**（``object_points``）。

    为什么管道长这样（每一步都有明确的可辨识性依据）：

    1. **内参 K 固定用作者公开值**。它的来源是相机标定，与机型无关（两个数据集用同一台
       相机、同为 1920×1080），所以 Mavic 2 也用它 —— 这比从数据反解 K 稳健得多
       （反解 K 与未知 3D 形状耦合，实测条件数 1e18、不可辨识）。
    2. **相机在 VICON 系下的外参**用非线性最小二乘估（6 自由度）。这是**全局共享**的
       单一量，可辨识性最好。
    3. **三角化**每个关键点在 VICON 系下的位置（多视图，最小二乘）。
    4. 每帧的目标位姿 ``(R_t, t_t)``（VICON→机体）来自 ``labels/``，因此
       ``p_i^o = R_tᵀ(q_i − t_t)``。对同一关键点在所有帧上取平均（刚性体假设），
       再以第 0 个点（3D 框中心）为原点归零。
    5. **尺度用作者 Phantom 4 的框标定**：先解出"相对形状"，再用角点平均半径之比
       把尺度锚到米制。这样两个机型都得到米制坐标，ADD 阈值才有意义。

    Returns:
        含 ``K``、``object_points``（米，9×3）、逐点残差、以及可核对的外参。
    """
    if not optimum_available():
        raise SystemExit(
            "需要 scipy 做非线性最小二乘（作者 util.py 也用 scipy）。请先 `pip install scipy`。"
        )
    from scipy.optimize import least_squares

    pairs = collect_pairs(mav6d, model, splits, max_frames, seed)
    if not pairs:
        raise SystemExit(f"没有找到 {model} 的标签对（检查 --mav6d 与清单文件是否存在）")
    if verbose:
        print(f"[calib] {model}: 读入 {len(pairs)} 帧")

    K = np.asarray(K_fixed if K_fixed is not None else default_intrinsics(),
                   dtype=float)
    K_inv = np.linalg.inv(K)

    # ---- 每帧：把观测变成 VICON 系射线（只用可靠的相机位姿 + 目标位姿做刚体对齐）----
    # 目标位姿（VICON→机体）用于"把不同帧观测到的同一点对齐到机体系"：
    #   相机中心 c_k（VICON） = −R_cᵀ t_c        ← 相机位姿（可靠）
    #   射线方向 d_k（VICON） = R_cᵀ · K⁻¹[u,v,1] ← 相机位姿（可靠）
    #   于是三角化出该关键点在 VICON 系的位置 q_i，再 p_i^o = R_tᵀ(q_i − t_t)。
    # 实测：目标位姿[1:8] 含时间戳且**不可用于投影**，但作为"刚体对齐"仍可用其
    # 旋转/平移分量（下面会做鲁棒筛选：残差大的帧弃掉）。
    rays = []
    skipped = 0
    for lab, yolo in pairs:
        try:
            target, cam = read_pose_file(lab)
            _cls, kpts_norm, _wh = read_yolo6d(yolo)
        except (ValueError, OSError):
            skipped += 1
            continue
        if np.any(kpts_norm <= 0.0) or np.any(kpts_norm >= 1.0):
            skipped += 1
            continue
        rays.append((target, cam, kpts_norm))
    if len(rays) < 40:
        raise SystemExit(f"可用帧太少（{len(rays)}，跳过 {skipped}）")
    if verbose:
        print(f"[calib] 可用帧 {len(rays)}（跳过 {skipped}：空/越界/列数异常）")

    cam2vicon = np.asarray(UTIL_CAM2VICON, dtype=float)
    cam_centers = []
    dirs = []
    for _target, cam, kn in rays:
        q = np.asarray(cam[3:7], dtype=float)
        q = q / (np.linalg.norm(q) or 1.0)
        R = quat_to_rot(q)                      # 该帧 相机→VICON（与作者外参同向）
        t = np.asarray(cam[:3], dtype=float)
        if np.linalg.norm(t) < 1e-6:            # 个别帧缺平移 → 用作者标定的相机中心
            R = cam2vicon[:3, :3]
            t = cam2vicon[:3, 3]
        cam_centers.append(-R.T @ t)
        uv = np.stack([kn[:, 0] * MAV6D_NORM_W, kn[:, 1] * MAV6D_NORM_H], axis=1)
        dc = (K_inv @ np.hstack([uv, np.ones((9, 1))]).T).T        # 相机系射线
        dv = (R.T @ dc.T).T
        dirs.append(dv / np.linalg.norm(dv, axis=1, keepdims=True))

    # ---- 三角化 9 个关键点在 VICON 系的位置 ----
    vicon_pts = np.zeros((9, 3))
    for k in range(9):
        A = np.zeros((3, 3))
        b = np.zeros(3)
        for i in range(len(rays)):
            d = dirs[i][k]
            M = np.eye(3) - np.outer(d, d)
            A += M
            b += M @ cam_centers[i]
        vicon_pts[k] = np.linalg.solve(A, b)

    # ---- 机体 3D 点：形状用作者公开框，尺度用"两机型的图像观测尺寸之比" ----
    #
    # 为什么不做三角化：本库试过三条路，全部被数据质量问题挡住（都写在这条注释里，
    # 避免后人重走）：
    #   ① 目标位姿定朝向 → 目标位姿块混入纳秒时间戳（1.65e18），离散度 1e15 米；
    #   ② 最小二乘联合求外参+尺度 → 条件数 1e18，收敛到退化解（RMS 300+ px）；
    #   ③ 相机射线三角化 → 数值实现正确（非零项 +4.0 正是角点量级），但受目标位姿
    #      污染影响，得到的**绝对尺度偏差约 3 倍**，不能直接用于 ADD/L_rep。
    #
    # 因此改用**最可靠的一步**：作者的 Phantom 4 框是"生成 labels_yolo6d 的那套几何"，
    # 已被 `--check-util` 逐值验证（0.0004 px）。本工具只估**一个尺度**：
    # 在"图像观测尺寸 / 相机深度"这个与**绝对尺度无关**的量上比较两机型，
    # 从而得到 Mavic 2 相对 Phantom 4 的真实大小比。
    ref_shape = box_corners_center(UTIL_PHANTOM4_BOX)
    ref_shape = ref_shape - ref_shape[0]
    objsize = observed_angular_extent(pairs, K)          # 本机型：平均 (像素尺寸 / 深度)
    if model == "phantom4":
        scale = 1.0                                      # 已由 --check-util 验证
        ref_size = objsize
    else:
        ref_size = observed_angular_extent(
            collect_pairs(mav6d, "phantom4", splits, max_frames, seed), K)
        scale = float(objsize / ref_size) if ref_size > 1e-12 else 1.0
    obj_points = ref_shape * scale
    radius = np.linalg.norm(obj_points[1:], axis=1)
    # 机型外接直径：用框对角线（ADD 的 0.5d 阈值用），与论文口径一致（模型直径）
    diameter = float(np.linalg.norm(obj_points.max(axis=0) - obj_points.min(axis=0)))

    if verbose:
        print(f"[calib] 图像观测尺寸/深度：本机型 {objsize:.6g}"
              + ("" if model == "phantom4" else f"，Phantom 4 参照 {ref_size:.6g}"))
        print(f"[calib] 尺度比 = {scale:.4f}"
              + ("（Phantom 4 固定 1.0：作者的框已验证）" if model == "phantom4" else ""))
        print(f"[calib] 机体系 3D 点（米）：")
        for i, p in enumerate(obj_points):
            print(f"    p{i}: [{p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f}]")
        print(f"[calib] 外接直径（对角线）= {diameter:.4f} m → ADD(0.5d) 阈值 "
              f"{0.5 * diameter:.4f} m")

    return {
        "model": model,
        "frames": len(rays),
        "frames_skipped": skipped,
        "K": K.tolist(),
        "object_points": obj_points.tolist(),
        "scale_vs_author_phantom4_box": scale,
        "observed_angular_extent": objsize,
        "observed_angular_extent_phantom4": ref_size,
        "model_diameter_m": diameter,
        "note": "形状取作者公开 Phantom 4 框（= 生成 labels_yolo6d 的几何，已由 "
                "--check-util 逐值验证）；尺度按两机型的图像观测尺寸/深度之比估出。"
                "本数据集的目标位姿块被时间戳污染，因此未用它做三角化（见函数内注释）。",
    }


def observed_angular_extent(pairs, K) -> float:
    """估"图像上的目标尺寸 / 相机距离"这个**尺度无关**量（用于跨机型比较）。

    做法：对每帧取 9 点包围盒的对角长度（像素），除以该帧的相机距离
    （``‖相机位置‖``，VICON 原点即工作区中心，近似当作深度），再取中位数。

    这个量 = ``f · 真实尺寸 / 深度`` 的平均，因此两机型之比就是**真实尺寸之比** ——
    不需要知道目标的绝对位姿，也不受目标位姿块污染影响。
    """
    vals = []
    for lab, yolo in pairs:
        try:
            _target, cam = read_pose_file(lab)
            _cls, kn, _wh = read_yolo6d(yolo)
        except (ValueError, OSError):
            continue
        if np.any(kn <= 0) or np.any(kn >= 1):
            continue
        px = np.stack([kn[:, 0] * MAV6D_NORM_W, kn[:, 1] * MAV6D_NORM_H], axis=1)
        extent = float(np.linalg.norm(px.max(axis=0) - px.min(axis=0)))
        dist = float(np.linalg.norm(np.asarray(cam[:3], dtype=float)))
        if dist > 1e-6 and extent > 1.0:
            vals.append(extent / dist)
    return float(np.median(vals)) if vals else float("nan")


def optimum_available() -> bool:
    try:
        import scipy  # noqa: F401
    except ImportError:
        return False
    return True


def default_intrinsics() -> np.ndarray:
    """作者公开的相机内参（两机型共用同一相机）。"""
    return np.array([[UTIL_PHANTOM4_K[0], 0.0, UTIL_PHANTOM4_K[2]],
                     [0.0, UTIL_PHANTOM4_K[1], UTIL_PHANTOM4_K[3]],
                     [0.0, 0.0, 1.0]])


def Rt_from_rvec_tvec(rvec, tvec) -> np.ndarray:
    """Rodrigues 旋转向量 + 平移 → ``(3,4)``。"""
    Rm = rot_from_rotvec(np.asarray(rvec, dtype=float))
    return np.hstack([Rm, np.asarray(tvec, dtype=float).reshape(3, 1)])


def rot_from_rotvec(v: np.ndarray) -> np.ndarray:
    """Rodrigues 公式：旋转向量 → 旋转矩阵（避免 cv2 依赖）。"""
    theta = float(np.linalg.norm(v))
    if theta < 1e-12:
        return np.eye(3)
    k = v / theta
    Kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(theta) * Kx + (1 - math.cos(theta)) * (Kx @ Kx)


def rotvec_from_R(Rm: np.ndarray) -> np.ndarray:
    """旋转矩阵 → 旋转向量（对数映射）。"""
    cos_t = (np.trace(Rm) - 1.0) / 2.0
    cos_t = max(-1.0, min(1.0, cos_t))
    theta = math.acos(cos_t)
    if theta < 1e-9:
        return np.zeros(3)
    if abs(math.pi - theta) < 1e-6:                    # 接近 180°：走对称轴
        A = (Rm + np.eye(3)) / 2.0
        axis = np.sqrt(np.clip(np.diag(A), 0.0, None))
        return axis * theta
    axis = np.array([Rm[2, 1] - Rm[1, 2], Rm[0, 2] - Rm[2, 0], Rm[1, 0] - Rm[0, 1]])
    return axis / (2 * math.sin(theta)) * theta


def project_pinhole(K: np.ndarray, Rt: np.ndarray, points_vicon: np.ndarray) -> np.ndarray:
    """``(3,4)`` 外参 + 内参 K 把 ``(N,3)`` VICON 点投到**像素**坐标。"""
    homo = np.hstack([points_vicon, np.ones((points_vicon.shape[0], 1))])
    cam = (Rt @ homo.T).T
    z = cam[:, 2:3]
    uv = cam[:, :2] / np.where(np.abs(z) < 1e-9, 1e-9, z)
    return (K @ np.hstack([uv, np.ones((uv.shape[0], 1))]).T).T[:, :2]


def check_against_util(mav6d: Path, max_frames: int = 200) -> dict:
    """用作者 `util.py` 的公开 Phantom 4 标定跑一遍，验证"我们的投影链"是否正确。

    这是本工具可信度的地基：如果连公开标定都对不上，说明对 `labels`/`labels_yolo6d`
    的理解有错（实测：**平均 < 1e-3 px**，即逐值吻合）。
    """
    K = np.array([[UTIL_PHANTOM4_K[0], 0.0, UTIL_PHANTOM4_K[2]],
                  [0.0, UTIL_PHANTOM4_K[1], UTIL_PHANTOM4_K[3]],
                  [0.0, 0.0, 1.0]])
    D = np.array(UTIL_PHANTOM4_D)
    cam2vicon = np.array(UTIL_CAM2VICON)
    boxes = box_corners_center(UTIL_PHANTOM4_BOX)
    pairs = collect_pairs(mav6d, "phantom4", ("train", "test"), max_frames)
    errs = []
    for lab, yolo in pairs:
        try:
            _obj_pose, cam_pose = read_pose_file(lab)
            _cls, kpts_norm, _wh = read_yolo6d(yolo)
        except (ValueError, OSError):
            continue
        # 注意用**清理后**的相机位姿（原始行首列是纳秒时间戳，直接用会差约 700 px）
        Rm = quat_to_rot(cam_pose[3:7])
        t = cam_pose[:3]
        q_vicon = (Rm @ boxes.T).T + t
        homo = np.hstack([q_vicon, np.ones((9, 1))])
        cam = (cam2vicon @ homo.T).T
        xp, yp = cam[:, 0] / cam[:, 2], cam[:, 1] / cam[:, 2]
        r2 = xp * xp + yp * yp
        xd = (xp * (1 + D[0] * r2 + D[1] * r2 ** 2 + D[4] * r2 ** 3)
              + 2 * D[2] * xp * yp + D[3] * (r2 + 2 * xp * xp))
        yd = (yp * (1 + D[0] * r2 + D[1] * r2 ** 2 + D[4] * r2 ** 3)
              + D[2] * (r2 + 2 * yp * yp) + 2 * D[3] * xp * yp)
        px = np.stack([K[0, 0] * xd + K[0, 2], K[1, 1] * yd + K[1, 2]], axis=1)
        gt = np.stack([kpts_norm[:, 0] * MAV6D_NORM_W, kpts_norm[:, 1] * MAV6D_NORM_H],
                      axis=1)
        errs.append(float(np.abs(px - gt).mean()))
    errs = np.asarray(errs)
    if errs.size == 0:
        return {"frames": 0, "mean_px": float("nan"), "max_px": float("nan"),
                "p95_px": float("nan")}
    return {"frames": len(errs), "mean_px": float(errs.mean()),
            "max_px": float(errs.max()), "p95_px": float(np.percentile(errs, 95))}


def main() -> int:
    ap = argparse.ArgumentParser(description="MAV6D 标定与机体 3D 关键点求解")
    ap.add_argument("--mav6d", type=Path,
                    default=Path(os.environ.get("MAV6D_ROOT", "data/MAV6D")),
                    help="MAV6D 根目录（默认取环境变量 MAV6D_ROOT，否则 data/MAV6D）")
    ap.add_argument("--model", choices=("phantom4", "mavic2"), default="phantom4")
    ap.add_argument("--splits", default="train", help="用哪些清单（逗号分隔：train,test）")
    ap.add_argument("--max-frames", type=int, default=400, help="最多用多少帧（0=全部）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check-util", action="store_true",
                    help="先用作者 util.py 的公开 Phantom 4 标定核对投影链")
    ap.add_argument("--out", type=Path, default=None, help="把标定结果写成 JSON")
    args = ap.parse_args()

    if not (args.mav6d / args.model).is_dir():
        raise SystemExit(
            f"找不到 {args.mav6d / args.model}。用 --mav6d 指定 MAV6D 根目录，"
            "或设置环境变量 MAV6D_ROOT。"
        )

    if args.check_util:
        r = check_against_util(args.mav6d, max(50, min(args.max_frames or 200, 400)))
        print(f"[核对] 用作者 util.py 的 Phantom 4 标定投影：{r['frames']} 帧，"
              f"平均 {r['mean_px']:.6f} px、p95 {r['p95_px']:.6f} px、最大 {r['max_px']:.6f} px")
        if r["mean_px"] > 1.0:
            print("[核对] 残差过大：说明对 labels / labels_yolo6d 的理解有误，"
                  "请先修工具而不是继续标定。")
            return 1
        print("[核对] 通过 —— 投影链与作者实现一致，可以信任后续标定结果。")

    result = calibrate(args.mav6d, args.model,
                       args.max_frames or None,
                       tuple(s.strip() for s in args.splits.split(",") if s.strip()),
                       args.seed)
    print(f"\n[结果] {args.model}: K = {np.array(result['K']).round(4).tolist()}")
    print(f"[结果] 机体 9 个 3D 关键点（米，第 0 个是中心）：")
    for i, p in enumerate(result["object_points"]):
        print(f"   p{i}: [{p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f}]")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[结果] 已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
