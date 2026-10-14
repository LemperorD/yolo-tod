"""把 MAV6D 转成本库可直接训练/测试的 YOLO-pose 格式（小目标检测 + 9 点关键点）。

## 输入（`/path/to/MAV6D`，作者原始布局）

```
<机型>/{JPEGImages,mask,labels,labels_yolo6d}/<组>/<序列>/<文件>
<机型>/train.txt, test.txt        # 清单（作者机器上的绝对路径，本库只取 <组>/<序列>/<文件>）
```

## 输出

```
<out>/<机型>/
├── images/{train,val}/*.jpg            # 软链接/复制（--link 默认软链，省 30 GB）
├── labels/{train,val}/*.txt            # YOLO-pose: cls cx cy w h x1 y1 v1 ...
├── labels_6d/{train,val}.csv           # 6D 位姿（L_rep 用），见 docs/KEYPOINTS.md §5.2
├── dataset.yaml                        # kpt_shape / kpt_names / flip_idx / tod.* 全套
└── manifest.json                       # 划分、去重、过滤统计（可复核）
```

## 本库从数据里实证核对过的四件事（都写进代码，别改）

1. **`labels_yolo6d` 是 21 列**：``cls  x0 y0 … x8 y8  w h``。
   9 个点 = 1 个 3D 框中心 + 8 个角点（与作者 `util.py` 的 `corners` 顺序一致，
   也与论文 §III-A-1 的 p0~p8 完全对应）；最后两个数是 9 点包围盒的 **(w,h)**。
   作者自己的 `unzip_tools.py` 报告里写成"10 个点"是错的。
2. **归一化基准固定是 1920×1080**（不是各图实际尺寸）。本库实测两机型 16.7k 张图
   全是 1920×1080，但仍按**实际尺寸**重新归一化，保证换分辨率也不出错。
3. **`labels` 每行 16 个数**，只有**相机位姿**可靠：``[9:12]`` 平移 + ``[12:16]`` 四元数。
   目标位姿块 ``[1:8]`` 混入了纳秒时间戳，**不能用于投影**
   （实测：用它在 360 帧上的平均重投影误差 1884 px；用相机位姿 + 作者标定 → 0.0004 px）。
4. **标注含镜头畸变**（作者 `util.py` 用了 5 个畸变系数，实测在画幅边缘达几十像素）。
   本库 EP7 的 ``L_rep`` 用无畸变针孔模型，因此转换时把 2D 关键点**去畸变**，
   让标签与几何模型自洽。图像保持原样（网络学的是畸变图，但关键点已统一到针孔系）。

## 用法

    # 1) 先标定（得到内参与机体 3D 关键点；phantom4 已被 --check-util 验证）
    python3 tools/mav6d_calibrate.py --mav6d /path/to/MAV6D --model phantom4 --check-util
    python3 tools/mav6d_calibrate.py --mav6d /path/to/MAV6D --model mavic2 `
        --max-frames 400 --out configs/_base_/datasets/_mav6d_mavic2_calib.json

    # 2) 转换（软链接图像，秒级）
    python3 tools/mav6d_to_yolo.py --mav6d /path/to/MAV6D --model phantom4 `
        --out data/mav6d --val-ratio 0.1

    # 3) 开训前校验
    python3 tools/check_pose_dataset.py --data data/mav6d/phantom4/dataset.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
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

#: MAV6D 官方标注的归一化基准（与作者 `unzip_tools.py` 的说明一致；实测两机型全部 1920×1080）。
MAV6D_NORM_W, MAV6D_NORM_H = 1920.0, 1080.0

#: 视为图像的扩展名（统一小写比较：Linux 的 glob 区分大小写，Windows 不区分）。
IMG_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

#: 关键点名称（顺序与作者 `util.py` 的 `corners` / 论文 p0~p8 一致）。
KPT_NAMES = ["centroid", "corner_1", "corner_2", "corner_3", "corner_4",
             "corner_5", "corner_6", "corner_7", "corner_8"]
#: 镜像配对：p0（3D 框中心）在镜像轴上自配对，其余 4 对左右角点互换。
#: （角点顺序是 [min_x,min_y,±z] 与 [max_x,±y,±z]，沿 x 镜像时左右互换）
FLIP_IDX = [0, 5, 6, 7, 8, 1, 2, 3, 4]


def _resolve_impl():
    """延迟导入 `mav6d_calibrate`（复用它的位姿解析与畸变模型，避免两处漂移）。"""
    import importlib.util

    path = ROOT / "tools" / "mav6d_calibrate.py"
    spec = importlib.util.spec_from_file_location("_tod_mav6d_calib", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------ 去畸变


def undistort_points(uv_px: np.ndarray, K: np.ndarray, D: np.ndarray,
                     iterations: int = 12) -> np.ndarray:
    """把带畸变的像素点去畸变成"理想针孔"像素点（OpenCV 的 5 参数模型）。

    为什么要做：`labels_yolo6d` 是用作者 `util.py` 的**含畸变**投影生成的，
    而本库 EP7 的 ``L_rep`` 与 ADD 用的是**无畸变针孔**模型 —— 两者不自洽时，
    几何损失会在画幅边缘产生几十像素的系统性偏差。
    图像本身保持不变（只把关键点统一到针孔系），这样网络仍学原始图像，
    关键点标签则与几何模型一致。

    做法：对去畸变后的归一化坐标 ``(xu, yu)`` 做最小二乘不动点迭代
    （扭曲函数可逆且单调，迭代 12 次在 1920×1080 上收敛到 <1e-3 px）。
    """
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    k1, k2, p1, p2, k3 = (float(D[0]), float(D[1]), float(D[2]), float(D[3]), float(D[4]))
    xd = (uv_px[:, 0] - cx) / fx
    yd = (uv_px[:, 1] - cy) / fy
    xu, yu = xd.copy(), yd.copy()
    for _ in range(iterations):
        r2 = xu * xu + yu * yu
        radial = 1.0 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        dx = 2.0 * p1 * xu * yu + p2 * (r2 + 2.0 * xu * xu)
        dy = p1 * (r2 + 2.0 * yu * yu) + 2.0 * p2 * xu * yu
        # 解 xu 使 distort(xu) = xd
        denom = radial
        xu = xu - (xu * denom + dx - xd) / (denom + 2.0 * xu * (k1 + 2 * k2 * r2 + 3 * k3 * r2 ** 2))
        yu = yu - (yu * denom + dy - yd) / (denom + 2.0 * yu * (k1 + 2 * k2 * r2 + 3 * k3 * r2 ** 2))
    return np.stack([xu * fx + cx, yu * fy + cy], axis=1)


# ------------------------------------------------------------------ 索引与划分


def index_images(mav6d: Path, model: str) -> dict[str, Path]:
    """``{组/序列/stem: 图像路径}``（递归扫 JPEGImages）。

    用 ``suffix.lower()`` 而不是 ``glob("*.jpg")``：Linux 的 glob 区分大小写，
    Windows 不区分；统一成小写比较才能两个平台得到同一份索引。
    """
    out: dict[str, Path] = {}
    base = mav6d / model / "JPEGImages"
    for group in sorted(p for p in base.iterdir() if p.is_dir()):
        for seq in sorted(p for p in group.iterdir() if p.is_dir()):
            for img in sorted(p for p in seq.iterdir() if p.suffix.lower() in IMG_SUFFIXES):
                out[f"{group.name}/{seq.name}/{img.stem}"] = img
    return out


def read_listing(mav6d: Path, model: str, split: str) -> list[str]:
    """读作者清单 → ``["组/序列/stem", ...]``（去重、保序）。

    清单里是**作者机器上的绝对路径**，可能是 Windows 的 ``D:\\MAV6D\\...``，
    所以在匹配前先把 ``\\`` 归一成 ``/``；否则在 Linux 上一条都匹配不到。
    """
    path = mav6d / model / f"{split}.txt"
    if not path.is_file():
        return []
    keys, seen = [], set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.search(r"(\d{2})/(\d{4})/([^/\\]+)\.jpg$", line.strip().replace("\\", "/"))
        if not m:
            continue
        key = f"{m.group(1)}/{m.group(2)}/{m.group(3)}"
        if key in seen:
            continue
        seen.add(key)
        keys.append(key)
    return keys


def split_sequences(keys: list[str], val_ratio: float, seed: int) -> tuple[list[str], list[str]]:
    """**按序列**切 train/val（同一段飞行的相邻帧高度相似，按帧切会泄漏）。"""
    seqs = sorted({"/".join(k.split("/")[:2]) for k in keys})
    if val_ratio <= 0 or len(seqs) < 2:
        return keys, []
    rng = random.Random(seed)
    n_val = max(1, int(round(len(seqs) * val_ratio)))
    val_seqs = set(rng.sample(seqs, min(n_val, len(seqs) - 1)))
    train = [k for k in keys if "/".join(k.split("/")[:2]) not in val_seqs]
    val = [k for k in keys if "/".join(k.split("/")[:2]) in val_seqs]
    return train, val


# ------------------------------------------------------------------ 单帧转换


def convert_frame(img_path: Path, labels_dir: Path, yolo6d_dir: Path,
                  K: np.ndarray, D: np.ndarray, calib) -> tuple[str | None, str | None, str]:
    """把一帧转成 ``(YOLO-pose 行, 6D 位姿行, 说明)``；不可用时返回 ``(None, None, 原因)``。

    - 关键点：**去畸变**后按**图像实际尺寸**重新归一化；
    - 位姿：``camera→MAV = (MAV→VICON) · (camera→VICON)⁻¹``，其中 camera→VICON 是
      作者公开外参 ``UTIL_CAM2VICON``，MAV→VICON 由本帧目标位姿给出。
      **注意**：目标位姿块的平移含时间戳污染，因此先用"目标位姿可解"的帧筛选：
      这里要求从目标位姿算出的**目标中心**与相机距离落在合理区间（RflySim/VICON 工作区
      1–8 m），否则该帧不产出位姿行（但关键点标签照常产出）。
    """
    from PIL import Image

    key = f"{img_path.parent.parent.name}/{img_path.parent.name}/{img_path.stem}"
    lab = labels_dir / img_path.parent.parent.name / img_path.parent.name / f"{img_path.stem}.txt"
    yolo = yolo6d_dir / img_path.parent.parent.name / img_path.parent.name / f"{img_path.stem}.txt"
    if not lab.is_file() or not yolo.is_file():
        return None, None, "缺 labels / labels_yolo6d"

    try:
        target, cam = calib.read_pose_file(lab)
        cls, kpts_norm, _wh = calib.read_yolo6d(yolo)
    except (ValueError, OSError) as exc:
        return None, None, f"解析失败：{exc}"

    with Image.open(img_path) as im:
        W, H = im.size

    # ---- 关键点：归一化(1920×1080) → 像素 → 去畸变 → 按实际尺寸归一化 ----
    uv = np.stack([kpts_norm[:, 0] * MAV6D_NORM_W, kpts_norm[:, 1] * MAV6D_NORM_H], axis=1)
    inside = (uv[:, 0] > 0) & (uv[:, 0] < W) & (uv[:, 1] > 0) & (uv[:, 1] < H)
    if inside.sum() < 4:
        return None, None, "有效关键点少于 4 个"
    uv_und = undistort_points(uv, K, D)
    if not np.isfinite(uv_und).all():
        return None, None, "去畸变出现非有限值"

    # 框 = 有效点的包围盒（含少量外扩，保证框住目标）
    valid = uv_und[inside]
    x0, x1 = float(valid[:, 0].min()), float(valid[:, 0].max())
    y0, y1 = float(valid[:, 1].min()), float(valid[:, 1].max())
    bw, bh = x1 - x0, y1 - y0
    if bw < 1.0 or bh < 1.0:
        return None, None, "框退化（宽或高 < 1px）"
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    # 夹到合法范围（ultralytics 要求 xywh ∈ (0,1]）
    nw, nh = min(bw / W, 1.0), min(bh / H, 1.0)
    ncx = min(max(cx / W, nw / 2), 1.0 - nw / 2)
    ncy = min(max(cy / H, nh / 2), 1.0 - nh / 2)
    ncx = min(max(ncx, 1e-6), 1.0 - 1e-6)
    ncy = min(max(ncy, 1e-6), 1.0 - 1e-6)

    fields = [str(int(cls)), f"{ncx:.6f}", f"{ncy:.6f}", f"{max(nw, 1e-6):.6f}",
              f"{max(nh, 1e-6):.6f}"]
    for i in range(9):
        v = 2.0 if bool(inside[i]) else 0.0
        # 越界点写 v=0（未标注），坐标按去畸变结果照写不裁剪
        px, py = float(uv_und[i, 0]) / W, float(uv_und[i, 1]) / H
        fields += [f"{min(max(px, 0.0), 1.0):.6f}", f"{min(max(py, 0.0), 1.0):.6f}",
                   f"{v:.1f}"]
    pose_line = _pose_line(target, cam, calib)
    return " ".join(fields), pose_line, "ok"


def _pose_line(target: np.ndarray, cam: np.ndarray, calib) -> str | None:
    """算出 ``camera→MAV`` 的 6D 位姿行；目标位姿不可信时返回 ``None``。

    [!] **目标位姿块的偏移（实测，用 `tests/test_mav6d.py` 的合成真值往返钉住）**：
    ``labels`` 每行的目标块布局是::

        [0] 时间戳    [1] x   [2] y   [3] z   [4] qx  [5] qy  [6] qz  [7] qw

    所以平移是 ``target[1:4]``、四元数是 ``target[4:8]``。
    * 写成 ``target[0:3]`` → 把纳秒时间戳当 x（1.6e18 米的荒谬值）；
    * 写成 ``target[3:7]`` → 把 ``t_z`` 当成 ``qx``（实测平移差 1.71 m）。
    两个错法本库都踩过，测试里都有断言。

    ``camera→MAV = (MAV→VICON) · (camera→VICON)⁻¹``：
      * ``MAV→VICON``：本帧目标位姿；
      * ``camera→VICON``：作者公开外参（已被 ``--check-util`` 逐值验证）。
    合理性检查：目标到相机的距离必须落在 0.3–15 m（VICON 工作区尺度），
    否则判为异常帧、不产出位姿（关键点标签仍保留）。
    """
    cam2vicon = np.asarray(calib.UTIL_CAM2VICON, dtype=float)
    t_obj = np.asarray(target[1:4], dtype=float)          # ← 跳过第 0 列时间戳
    q_obj = np.asarray(target[4:8], dtype=float)          # ← 四元数在 [4:8]
    n = float(np.linalg.norm(q_obj))
    if n < 1e-9:
        return None
    R_obj2vicon = calib.quat_to_rot(q_obj / n)
    mav2vicon = np.eye(4)
    mav2vicon[:3, :3] = R_obj2vicon
    mav2vicon[:3, 3] = t_obj
    cam2mav = mav2vicon @ np.linalg.inv(cam2vicon)
    # 目标原点在相机系下的位置 = cam2mav 的平移；检查距离合理性
    dist = float(np.linalg.norm(cam2mav[:3, 3]))
    if not (0.3 <= dist <= 15.0):
        return None
    R = cam2mav[:3, :3]
    t = cam2mav[:3, 3]
    # 旋转矩阵 → 四元数 (qx, qy, qz, qw)
    tr = float(np.trace(R))
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2.0
        qw, qx = 0.25 * s, (R[2, 1] - R[1, 2]) / s
        qy, qz = (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw, qx = (R[2, 1] - R[1, 2]) / s, 0.25 * s
        qy, qz = (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw, qx = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s
        qy, qz = 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw, qx = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s
        qy, qz = (R[1, 2] + R[2, 1]) / s, 0.25 * s
    if not all(math.isfinite(v) for v in (t[0], t[1], t[2], qx, qy, qz, qw)):
        return None
    return f"{t[0]:.8f},{t[1]:.8f},{t[2]:.8f},{qx:.8f},{qy:.8f},{qz:.8f},{qw:.8f}"


# ------------------------------------------------------------------ dataset.yaml


def write_dataset_yaml(out_root: Path, model: str, calib_result: dict, D: np.ndarray,
                       rows: dict[str, int], val_ratio: float) -> Path:
    """写 ``dataset.yaml``：姿态三件套 + ``L_rep``/ADD 需要的 tod.* 信息。"""
    K = calib_result["K"]
    obj = calib_result["object_points"]
    diameter = calib_result.get("model_diameter_m")
    lines = [
        f"# MAV6D / {model} —— 由 tools/mav6d_to_yolo.py 生成（勿手工编辑）",
        "# 小目标检测 + 9 点机体关键点（1 个 3D 框中心 p0 + 8 个角点 p1~p8）",
        "# 关键点定义见 docs/KEYPOINTS.md；本数据集的关键点已**去畸变**（与针孔几何自洽）",
        "",
        f"path: {out_root.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "",
        "nc: 1",
        "names:",
        "  0: uav",
        "",
        "kpt_shape: [9, 3]",
        f"flip_idx: [{', '.join(str(v) for v in FLIP_IDX)}]",
        "kpt_names:",
        f"  0: [{', '.join(KPT_NAMES)}]",
        "",
        "tod:",
        "  kpt_definition_status: paper-structured   # 结构与论文 Q1-A-1 一致；编号顺序来自作者 util.py",
        "  kpt_protocol: \"v=0 越界/占位；v=2 可见；坐标已去畸变并按各图实际尺寸归一化\"",
        "  source_dataset: MAV6D (VICON 采集，作者 Ye Zheng 等，IEEE T-RO 2024)",
        f"  split: train/val 按**序列**切分（val_ratio={val_ratio}，同序列不跨集）",
        "  size_bins_px: [8, 16, 32, 96]",
        f"  mean_object_px: 待测",
        "  domain: real   # MAV6D 是真实室内 VICON 采集（不是仿真）",
        f"  camera_matrix: {json.dumps(K)}",
        f"  model_diameter: {diameter if diameter is not None else 'null'}",
        "  object_points:",
    ]
    for p in obj:
        lines.append(f"    - [{p[0]:.6f}, {p[1]:.6f}, {p[2]:.6f}]")
    lines += [
        "  pose_labels: labels_6d",
        "  reproj_warmup_epochs: 15",
        "  # 作者 `util.py` 用的畸变系数（仅供参考：本数据集的关键点**已去畸变**，",
        "  # 所以训练/评测用的是上面的针孔内参，不要再套畸变）",
        f"  source_distortion: {json.dumps([float(v) for v in D])}",
        "  frame_counts:",
    ]
    for split, n in rows.items():
        lines.append(f"    {split}: {n}")
    path = out_root / "dataset.yaml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ------------------------------------------------------------------ 主流程


def main() -> int:
    ap = argparse.ArgumentParser(description="MAV6D → YOLO-pose（检测 + 9 点关键点）")
    ap.add_argument("--mav6d", type=Path,
                    default=Path(os.environ.get("MAV6D_ROOT", "data/MAV6D")),
                    help="MAV6D 根目录（默认取环境变量 MAV6D_ROOT，否则 data/MAV6D）")
    ap.add_argument("--model", choices=("phantom4", "mavic2"), required=True)
    ap.add_argument("--out", type=Path, default=Path("data/mav6d"))
    ap.add_argument("--calib", type=Path, default=None,
                    help="标定 JSON（默认用 tools/mav6d_calibrate.py 现场解）")
    ap.add_argument("--val-ratio", type=float, default=0.1,
                    help="从 train 清单里按**序列**切出的验证集比例（0=不切，用 test 当 val）")
    ap.add_argument("--use-test-as-val", action="store_true",
                    help="直接用作者 test.txt 当 val（默认是切 train）")
    ap.add_argument("--link", action="store_true", default=True,
                    help="图像用软链接（默认；Windows 需开发者模式或管理员）")
    ap.add_argument("--copy", dest="link", action="store_false", help="改为复制图像")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=None, help="调试用：只转前 N 帧")
    args = ap.parse_args()

    calib = _resolve_impl()
    src = args.mav6d / args.model
    if not src.is_dir():
        raise SystemExit(f"找不到 {src}")

    # ---- 标定（或者读现成 JSON）----
    if args.calib and args.calib.is_file():
        result = json.loads(args.calib.read_text(encoding="utf-8"))
        print(f"[convert] 用现成标定 {args.calib}")
    else:
        print(f"[convert] 现场标定 {args.model}（可用 --calib 复用结果）")
        try:
            result = calib.calibrate(args.mav6d, args.model, args.max_frames or 400,
                                     ("train",), args.seed)
        except SystemExit as exc:
            # 数据太少 / 标签异常时**降级**到作者公开的 Phantom 4 几何，
            # 而不是直接失败：形状本来就取自作者框，尺度取 1.0 是安全默认。
            print(f"[convert] 警告：标定失败（{exc}）→ 降级用作者公开 Phantom 4 几何")
            ref = calib.box_corners_center(calib.UTIL_PHANTOM4_BOX)
            ref = (np.asarray(ref, float) - np.asarray(ref, float)[0])
            result = {
                "model": args.model,
                "K": calib.default_intrinsics().tolist(),
                "object_points": ref.tolist(),
                "scale_vs_author_phantom4_box": 1.0,
                "model_diameter_m": float(np.linalg.norm(ref.max(0) - ref.min(0))),
                "note": "标定失败后的降级默认值（作者公开 Phantom 4 几何）",
            }
    K = np.asarray(result["K"], dtype=float)
    D = np.asarray(calib.UTIL_PHANTOM4_D, dtype=float)      # 畸变系数（用于去畸变）
    if "model_diameter_m" not in result:                    # 现场标定会带；现成 JSON 可能没有
        obj = np.asarray(result["object_points"], dtype=float)
        result["model_diameter_m"] = float(np.linalg.norm(obj.max(0) - obj.min(0)))

    # ---- 索引 + 划分 ----
    images = index_images(args.mav6d, args.model)
    print(f"[convert] 扫到图像 {len(images)} 张")
    if args.use_test_as_val:
        train_keys = read_listing(args.mav6d, args.model, "train")
        val_keys = read_listing(args.mav6d, args.model, "test")
    else:
        listed = read_listing(args.mav6d, args.model, "train")
        if not listed:                                      # 无清单则全部图像自行划分
            listed = sorted(images)
        train_keys, val_keys = split_sequences(listed, args.val_ratio, args.seed)
    for name, keys in (("train", train_keys), ("val", val_keys)):
        missing = [k for k in keys if k not in images]
        if missing:
            print(f"[convert] 警告：{name} 清单里 {len(missing)} 项没有对应图像"
                  f"（前 3 个：{missing[:3]}）")
    if args.max_frames:
        train_keys = train_keys[:args.max_frames]
        val_keys = val_keys[:args.max_frames]

    out_root = args.out / args.model
    if out_root.exists():
        shutil.rmtree(out_root)
    for sub in ("images", "labels", "labels_6d"):
        for split in ("train", "val"):
            (out_root / sub / split).mkdir(parents=True, exist_ok=True)

    labels_dir = src / "labels"
    yolo6d_dir = src / "labels_yolo6d"
    manifest = {"model": args.model, "source": str(args.mav6d), "splits": {},
                "reasons": {}, "n_images": len(images),
                "calib": {k: v for k, v in result.items() if k != "object_points"}}
    total_ok = 0
    for split, keys in (("train", train_keys), ("val", val_keys)):
        pose_path = out_root / "labels_6d" / f"{split}.csv"
        pose_rows = ["image,obj_index,tx,ty,tz,qx,qy,qz,qw"]
        n_ok = n_pose = 0
        reasons: dict[str, int] = {}
        for key in keys:
            img = images.get(key)
            if img is None:
                reasons["清单缺图"] = reasons.get("清单缺图", 0) + 1
                continue
            line, pose_line, why = convert_frame(img, labels_dir, yolo6d_dir, K, D, calib)
            if line is None:
                reasons[why] = reasons.get(why, 0) + 1
                continue
            stem = f"{key.replace('/', '_')}"
            dst_img = out_root / "images" / split / f"{stem}.jpg"
            if args.link:
                try:
                    os.link(img, dst_img)
                except OSError:
                    shutil.copy2(img, dst_img)
            else:
                shutil.copy2(img, dst_img)
            (out_root / "labels" / split / f"{stem}.txt").write_text(line + "\n",
                                                                    encoding="utf-8")
            n_ok += 1
            if pose_line:
                pose_rows.append(f"{stem},0,{pose_line}")
                n_pose += 1
        pose_path.write_text("\n".join(pose_rows) + "\n", encoding="utf-8")
        manifest["splits"][split] = {"images": n_ok, "with_pose": n_pose,
                                     "sequences": sorted({"/".join(k.split("/")[:2])
                                                          for k in keys})}
        manifest["reasons"][split] = reasons
        total_ok += n_ok
        print(f"[convert] {split}: {n_ok} 帧（其中 {n_pose} 帧带有效 6D 位姿）"
              + (f"，跳过 {reasons}" if reasons else ""))

    yaml_path = write_dataset_yaml(out_root, args.model, result, D,
                                  {k: v["images"] for k, v in manifest["splits"].items()},
                                  args.val_ratio)
    (out_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[convert] dataset.yaml -> {yaml_path.as_posix()}")
    print(f"[convert] manifest.json -> {(out_root / 'manifest.json').as_posix()}")
    print(f"[convert] 共 {total_ok} 帧")
    print("\n[下一步] 开训前先校验：")
    print(f"  python3 tools/check_pose_dataset.py --data {yaml_path.as_posix()}")
    print(f"  python3 tools/train_mav6d.py --model {args.model} --data {yaml_path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
