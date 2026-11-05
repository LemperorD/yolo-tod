"""把 MAV6D 转成本库可直接训练/测试的 YOLO-pose 格式（小目标检测 + 9 点关键点）。"""

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

try:
    import tod

    tod.configure_stdio()
except Exception:
    pass

MAV6D_NORM_W, MAV6D_NORM_H = 1920.0, 1080.0

IMG_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

KPT_NAMES = ["centroid", "corner_1", "corner_2", "corner_3", "corner_4",
             "corner_5", "corner_6", "corner_7", "corner_8"]
FLIP_IDX = [0, 5, 6, 7, 8, 1, 2, 3, 4]


def _resolve_impl():
    """延迟导入 `mav6d_calibrate`（复用它的位姿解析与畸变模型，避免两处漂移）。"""
    import importlib.util

    path = ROOT / "tools" / "mav6d_calibrate.py"
    spec = importlib.util.spec_from_file_location("_tod_mav6d_calib", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def undistort_points(uv_px: np.ndarray, K: np.ndarray, D: np.ndarray,
                     iterations: int = 12) -> np.ndarray:
    """把带畸变的像素点去畸变成"理想针孔"像素点（OpenCV 的 5 参数模型）。"""
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
        denom = radial
        xu = xu - (xu * denom + dx - xd) / (denom + 2.0 * xu * (k1 + 2 * k2 * r2 + 3 * k3 * r2 ** 2))
        yu = yu - (yu * denom + dy - yd) / (denom + 2.0 * yu * (k1 + 2 * k2 * r2 + 3 * k3 * r2 ** 2))
    return np.stack([xu * fx + cx, yu * fy + cy], axis=1)


def index_images(mav6d: Path, model: str) -> dict[str, Path]:
    """``{组/序列/stem: 图像路径}``（递归扫 JPEGImages）。"""
    out: dict[str, Path] = {}
    base = mav6d / model / "JPEGImages"
    for group in sorted(p for p in base.iterdir() if p.is_dir()):
        for seq in sorted(p for p in group.iterdir() if p.is_dir()):
            for img in sorted(p for p in seq.iterdir() if p.suffix.lower() in IMG_SUFFIXES):
                out[f"{group.name}/{seq.name}/{img.stem}"] = img
    return out


def read_listing(mav6d: Path, model: str, split: str) -> list[str]:
    """读作者清单 → ``["组/序列/stem", ...]``（去重、保序）。"""
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
    # 按完整序列划分，避免相邻帧同时进入训练集和验证集。
    seqs = sorted({"/".join(k.split("/")[:2]) for k in keys})
    if val_ratio <= 0 or len(seqs) < 2:
        return keys, []
    rng = random.Random(seed)
    n_val = max(1, int(round(len(seqs) * val_ratio)))
    val_seqs = set(rng.sample(seqs, min(n_val, len(seqs) - 1)))
    train = [k for k in keys if "/".join(k.split("/")[:2]) not in val_seqs]
    val = [k for k in keys if "/".join(k.split("/")[:2]) in val_seqs]
    return train, val


def convert_frame(img_path: Path, labels_dir: Path, yolo6d_dir: Path,
                  K: np.ndarray, D: np.ndarray, calib,
                  object_points: np.ndarray | None = None,
                  max_reproj_px: float = 5.0) -> tuple[str | None, str | None, str]:
    """把一帧转成 ``(YOLO-pose 行, 6D 位姿行, 说明)``；不可用时返回 ``(None, None, 原因)``。"""
    from PIL import Image

    key = f"{img_path.parent.parent.name}/{img_path.parent.name}/{img_path.stem}"
    lab = labels_dir / img_path.parent.parent.name / img_path.parent.name / f"{img_path.stem}.txt"
    yolo = yolo6d_dir / img_path.parent.parent.name / img_path.parent.name / f"{img_path.stem}.txt"
    if not lab.is_file() or not yolo.is_file():
        return None, None, "缺 labels / labels_yolo6d"

    try:
        camera_pose, mav_pose = calib.read_pose_file(lab)
        cls, kpts_norm, _wh = calib.read_yolo6d(yolo)
    except (ValueError, OSError) as exc:
        return None, None, f"解析失败：{exc}"

    with Image.open(img_path) as im:
        W, H = im.size

    # 源坐标按固定 1920×1080 还原，输出标签再按实际图像宽高归一化。
    uv = np.stack([kpts_norm[:, 0] * MAV6D_NORM_W, kpts_norm[:, 1] * MAV6D_NORM_H], axis=1)
    inside = (uv[:, 0] > 0) & (uv[:, 0] < W) & (uv[:, 1] > 0) & (uv[:, 1] < H)
    if inside.sum() < 4:
        return None, None, "有效关键点少于 4 个"
    # 去畸变后的二维点与 L_rep 使用同一套针孔内参。
    uv_und = undistort_points(uv, K, D)
    if not np.isfinite(uv_und).all():
        return None, None, "去畸变出现非有限值"

    valid = uv_und[inside]
    x0, x1 = float(valid[:, 0].min()), float(valid[:, 0].max())
    y0, y1 = float(valid[:, 1].min()), float(valid[:, 1].max())
    bw, bh = x1 - x0, y1 - y0
    if bw < 1.0 or bh < 1.0:
        return None, None, "框退化（宽或高 < 1px）"
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    nw, nh = min(bw / W, 1.0), min(bh / H, 1.0)
    ncx = min(max(cx / W, nw / 2), 1.0 - nw / 2)
    ncy = min(max(cy / H, nh / 2), 1.0 - nh / 2)
    ncx = min(max(ncx, 1e-6), 1.0 - 1e-6)
    ncy = min(max(ncy, 1e-6), 1.0 - 1e-6)

    fields = [str(int(cls)), f"{ncx:.6f}", f"{ncy:.6f}", f"{max(nw, 1e-6):.6f}",
              f"{max(nh, 1e-6):.6f}"]
    for i in range(9):
        v = 2.0 if bool(inside[i]) else 0.0
        px, py = float(uv_und[i, 0]) / W, float(uv_und[i, 1]) / H
        fields += [f"{min(max(px, 0.0), 1.0):.6f}", f"{min(max(py, 0.0), 1.0):.6f}",
                   f"{v:.1f}"]
    pose_line = _pose_line(camera_pose, mav_pose, calib,
                           kpts_px=uv_und, K=K, valid=inside,
                           object_points=object_points,
                           max_reproj_px=max_reproj_px)
    if pose_line is None:
        return " ".join(fields), None, "6D 位姿投影不一致/不可解"
    return " ".join(fields), pose_line, "ok"


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    """旋转矩阵 → 四元数 ``(qx, qy, qz, qw)``（与 `tod.data.pose6d` 同式）。"""
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
    q = np.asarray([qx, qy, qz, qw], dtype=float)
    return q / (np.linalg.norm(q) or 1.0)


def _pose_line(camera_pose: np.ndarray, mav_pose: np.ndarray, calib,
               *, kpts_px: np.ndarray | None = None, K: np.ndarray | None = None,
               valid: np.ndarray | None = None,
               object_points: np.ndarray | None = None,
               max_reproj_px: float = 5.0) -> str | None:
    """算出 ``camera→MAV`` 的 6D 位姿行；不可信时返回 ``None``。"""
    del camera_pose
    t_mav = np.asarray(mav_pose[:3], dtype=float)
    q_mav = np.asarray(mav_pose[3:7], dtype=float)
    n = float(np.linalg.norm(q_mav))
    if n < 1e-9:
        return None
    mav2vicon = np.eye(4)
    mav2vicon[:3, :3] = calib.quat_to_rot(q_mav / n)
    mav2vicon[:3, 3] = t_mav
    cam2vicon = np.asarray(calib.UTIL_CAM2VICON, dtype=float)
    cam2mav = np.linalg.inv(cam2vicon) @ mav2vicon
    dist = float(np.linalg.norm(cam2mav[:3, 3]))
    if not (0.3 <= dist <= 15.0):
        return None
    R = cam2mav[:3, :3]
    t = cam2mav[:3, 3]

    if kpts_px is not None and K is not None and object_points is not None:
        obj = np.asarray(object_points, dtype=float)
        pc = (R @ obj.T).T + t
        if np.any(pc[:, 2] <= 1e-9):
            return None
        uv = (np.asarray(K, dtype=float)[:2, :2] @ (pc[:, :2] / pc[:, 2:3]).T).T \
            + np.asarray(K, dtype=float)[:2, 2]
        mask = np.ones(len(uv), dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
        if mask.sum() >= 4:
            err = float(np.linalg.norm(uv[mask] - np.asarray(kpts_px)[mask], axis=1).mean())
            if not math.isfinite(err) or err > float(max_reproj_px):
                return None

    q = rot_to_quat(R)
    if not all(math.isfinite(v) for v in (t[0], t[1], t[2], q[0], q[1], q[2], q[3])):
        return None
    return (f"{t[0]:.8f},{t[1]:.8f},{t[2]:.8f},"
            f"{q[0]:.8f},{q[1]:.8f},{q[2]:.8f},{q[3]:.8f}")


def write_dataset_yaml(out_root: Path, model: str, calib_result: dict, D: np.ndarray,
                       rows: dict[str, int], val_ratio: float,
                       split_note: str | None = None,
                       pose_source: str | None = None,
                       pose_strict: bool | None = None,
                       pose_reproj_px: float | None = None,
                       pose_coverage: float | None = None) -> Path:
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
        f"  split: {split_note or f'train/val 按**序列**切分（val_ratio={val_ratio}，同序列不跨集）'}",
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
    ]
    if pose_source:
        lines.append(f"  pose_source: {pose_source}")
    if pose_strict is not None:
        lines.append(f"  pose_strict: {str(bool(pose_strict)).lower()}")
    if pose_reproj_px is not None:
        lines.append(f"  pose_reproj_px: {float(pose_reproj_px):.4f}")
    if pose_coverage is not None:
        lines.append(f"  pose_coverage: {float(pose_coverage):.4f}")
    lines += [
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


def preset_calib_path(model: str) -> Path | None:
    """仓库里预置的标定 JSON（``configs/_base_/datasets/_mav6d_<机型>_calib.json``）。"""
    candidate = ROOT / "configs" / "_base_" / "datasets" / f"_mav6d_{model}_calib.json"
    return candidate if candidate.is_file() else None


def _load_calib_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _author_geometry_fallback(calib, model: str) -> dict:
    """降级到作者公开的 Phantom 4 框 + 默认内参（形状的来源，尺度取 1.0 是安全默认）。"""
    ref = calib.box_corners_center(calib.UTIL_PHANTOM4_BOX)
    ref = (np.asarray(ref, float) - np.asarray(ref, float)[0])
    return {
        "model": model,
        "K": calib.default_intrinsics().tolist(),
        "object_points": ref.tolist(),
        "scale_vs_author_phantom4_box": 1.0,
        "model_diameter_m": float(np.linalg.norm(ref.max(0) - ref.min(0))),
        "note": "标定失败后的降级默认值（作者公开 Phantom 4 几何）",
    }


def resolve_calibration(args, calib) -> tuple[dict, np.ndarray, np.ndarray]:
    """取标定结果，返回 ``(结果 dict, K, D)``（``D`` 是作者 5 参数畸变，去畸变用）。"""
    if args.calib and args.calib.is_file():
        print(f"[convert] 用现成标定 {args.calib}")
        result = _load_calib_json(args.calib)
    else:
        preset = preset_calib_path(args.model)
        src = args.mav6d / args.model
        if preset is not None:
            print(f"[convert] 用仓库预置标定 {preset}")
            result = _load_calib_json(preset)
        elif not src.is_dir():
            print("[convert] 警告：既没有 --calib / 预置 JSON，也没有原始 MAV6D "
                  "→ 降级用作者公开 Phantom 4 几何")
            result = _author_geometry_fallback(calib, args.model)
        else:
            print(f"[convert] 现场标定 {args.model}（可用 --calib 复用结果）")
            try:
                result = calib.calibrate(args.mav6d, args.model, args.max_frames or 400,
                                         ("train",), args.seed)
            except SystemExit as exc:
                print(f"[convert] 警告：标定失败（{exc}）→ 降级用作者公开 Phantom 4 几何")
                result = _author_geometry_fallback(calib, args.model)

    K = np.asarray(result["K"], dtype=float)
    D = np.asarray(calib.UTIL_PHANTOM4_D, dtype=float)
    if "model_diameter_m" not in result:
        obj = np.asarray(result["object_points"], dtype=float)
        result["model_diameter_m"] = float(np.linalg.norm(obj.max(0) - obj.min(0)))
    return result, K, D


def _count_files(directory: Path, suffixes: tuple[str, ...] | None = None) -> int:
    """数目录里的文件（递归；给 ``suffixes`` 就只数这些扩展名）。"""
    if not directory.is_dir():
        return 0
    n = 0
    for item in directory.rglob("*"):
        if not item.is_file():
            continue
        if suffixes is None or item.suffix.lower() in suffixes:
            n += 1
    return n


def _csv_data_rows(path: Path) -> int:
    """``labels_6d/<split>.csv`` 的数据行数（减表头；文件不存在为 0）。"""
    if not path.is_file():
        return 0
    with open(path, "r", encoding="utf-8") as fh:
        return max(0, sum(1 for _ in fh) - 1)


def descriptor_only(args, calib) -> int:
    """只（重新）生成 ``dataset.yaml`` + ``manifest.json``，**不动** images/labels。"""
    out_root = args.out / args.model
    if not out_root.is_dir():
        raise SystemExit(
            f"找不到 {out_root}\n"
            "--yaml-only 只在**已经转换好的**目录上补写描述文件；"
            "数据还没转换就先跑一次完整转换（去掉 --yaml-only）。"
        )
    missing = [name for name in ("images", "labels", "labels_6d")
               if not (out_root / name).is_dir()]
    if missing:
        raise SystemExit(
            f"{out_root} 缺少 {missing}：这不像一份 MAV6D 转换产物。\n"
            "要么先跑完整转换，要么确认 --out 指对了（应为 <out>/<机型> 的父目录）。"
        )

    splits = {}
    rows = {}
    for split in ("train", "val"):
        n_img = _count_files(out_root / "images" / split, IMG_SUFFIXES)
        n_lab = _count_files(out_root / "labels" / split, (".txt",))
        n_pose = _csv_data_rows(out_root / "labels_6d" / f"{split}.csv")
        if n_img == 0:
            raise SystemExit(f"{out_root}/images/{split} 里没有图像（--out 指错了？）")
        if n_lab != n_img:
            print(f"[yaml-only] 警告：{split} 图像 {n_img} 张、标签 {n_lab} 个（应该相等）")
        splits[split] = {"images": n_img, "with_pose": n_pose}
        rows[split] = n_img
        print(f"[yaml-only] {split}: {n_img} 张图 / {n_lab} 个标签 / {n_pose} 条 6D 位姿")

    result, K, D = resolve_calibration(args, calib)
    old_tod: dict = {}
    old_yaml = out_root / "dataset.yaml"
    if old_yaml.is_file():
        try:
            from tod.compat import load_yaml

            old_tod = (load_yaml(old_yaml) or {}).get("tod") or {}
        except Exception:  # noqa: BLE001
            old_tod = {}
    yaml_path = write_dataset_yaml(
        out_root, args.model, result, D, rows, args.val_ratio,
        split_note="沿用磁盘上已有的 train/val 划分（--yaml-only 未重新划分）",
        pose_source=old_tod.get("pose_source"),
        pose_strict=old_tod.get("pose_strict"),
        pose_reproj_px=old_tod.get("pose_reproj_px"),
        pose_coverage=old_tod.get("pose_coverage"),
    )
    manifest = {
        "model": args.model,
        "source": str(args.mav6d),
        "yaml_only": True,
        "splits": splits,
        "calib": {k: v for k, v in result.items() if k != "object_points"},
    }
    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    print(f"[yaml-only] dataset.yaml  -> {yaml_path.as_posix()}")
    print(f"[yaml-only] manifest.json -> {manifest_path.as_posix()}")
    print("\n[下一步] 注册表里让它对上号（或者 --data 直接给这个目录）：")
    print(f"  python3 tools/check_pose_dataset.py --data {yaml_path.as_posix()}")
    print(f"  python3 tools/train.py --list-datasets")
    return 0


def _image_size_for(out_root: Path, split: str, stem: str) -> tuple[int, int] | None:
    """找 ``images/<split>/<stem>.<ext>`` 并返回 ``(w, h)``。"""
    from PIL import Image

    for suffix in IMG_SUFFIXES:
        path = out_root / "images" / split / f"{stem}{suffix}"
        if path.is_file():
            with Image.open(path) as im:
                return int(im.size[0]), int(im.size[1])
    return None


def _pnp_from_keypoints(object_points: np.ndarray, kpts_px: np.ndarray,
                        K: np.ndarray, max_reproj_px: float):
    """单目标 PnP：返回 ``(R, t, mean_reproj_px)``；不可解/超阈值返回 ``None``。"""
    import cv2

    valid = np.isfinite(kpts_px).all(axis=1)
    if valid.sum() < 6:
        return None
    obj = np.asarray(object_points, dtype=np.float64)
    uv = np.asarray(kpts_px, dtype=np.float64)
    for flag in (cv2.SOLVEPNP_ITERATIVE, cv2.SOLVEPNP_EPNP):
        try:
            ok, rvec, tvec = cv2.solvePnP(
                obj[valid], uv[valid], np.asarray(K, dtype=np.float64), None,
                flags=flag,
            )
        except cv2.error:
            continue
        if not ok:
            continue
        R, _ = cv2.Rodrigues(rvec)
        t = np.asarray(tvec, dtype=float).reshape(3)
        pc = (R @ obj.T).T + t
        if np.any(pc[:, 2] <= 1e-9):
            continue
        proj = (np.asarray(K, dtype=float)[:2, :2] @ (pc[:, :2] / pc[:, 2:3]).T).T \
            + np.asarray(K, dtype=float)[:2, 2]
        err = float(np.linalg.norm(proj[valid] - uv[valid], axis=1).mean())
        if not np.isfinite(err) or err > float(max_reproj_px):
            continue
        if not (0.3 <= float(np.linalg.norm(t)) <= 15.0):
            continue
        return R, t, err
    return None


def _pnp_frames(label_dir: Path, max_frames: int | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
    """读 YOLO-pose 标签，返回 ``[(uv_px, visible_mask), ...]``（每帧一个目标）。"""
    frames = []
    for label_path in sorted(label_dir.glob("*.txt")):
        for line in label_path.read_text(encoding="utf-8").splitlines():
            vals = [float(v) for v in line.split()]
            if len(vals) < 5 + 3 * 9:
                continue
            kpts = np.asarray(vals[5:5 + 27], dtype=float).reshape(9, 3)
            vis = kpts[:, 2] > 0
            if vis.sum() < 6:
                continue
            frames.append((kpts[:, :2] * np.array([1920.0, 1080.0]), vis))
            break
        if max_frames and len(frames) >= max_frames:
            break
    return frames


def fit_shape_from_pnp(label_dir: Path, K: np.ndarray, obj0: np.ndarray,
                       diameter0: float, *, small: int = 40, large: int = 200,
                       seed: int = 0) -> tuple[np.ndarray, dict]:
    """用 2D 关键点 + PnP 拟合机体的**各向异性尺度**（只改形状，不改整体直径）。"""
    try:
        from scipy.optimize import minimize
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("--fit-pnp-shape 需要 scipy：pip install scipy") from exc
    import cv2

    frames = _pnp_frames(label_dir)
    if len(frames) < 20:
        raise SystemExit(f"{label_dir} 可用帧太少（{len(frames)}），无法拟合形状。")
    small_set = frames[:small]
    stride = max(1, len(frames) // large)
    large_set = frames[::stride][:large]
    if not large_set:
        large_set = frames

    obj0 = np.asarray(obj0, dtype=np.float64)

    def _normalize(shape: np.ndarray) -> np.ndarray:
        diag = float(np.linalg.norm(shape.max(axis=0) - shape.min(axis=0)))
        return shape * (diameter0 / diag) if diag > 1e-9 else shape

    def _stats(shape: np.ndarray, subset) -> tuple[float, int]:
        vals = []
        fails = 0
        for uv, vis in subset:
            ok, rvec, tvec = cv2.solvePnP(
                shape[vis], uv[vis], np.asarray(K, dtype=np.float64), None,
                flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                fails += 1
                continue
            R, _ = cv2.Rodrigues(rvec)
            pc = (R @ shape.T).T + tvec.reshape(3)
            if np.any(pc[:, 2] <= 1e-9):
                fails += 1
                continue
            proj = (np.asarray(K, dtype=float)[:2, :2]
                    @ (pc[:, :2] / pc[:, 2:3]).T).T + np.asarray(K, dtype=float)[:2, 2]
            vals.append(float(np.linalg.norm(proj[vis] - uv[vis], axis=1).mean()))
        return (float(np.mean(vals)) if vals else 1e6), fails

    def _objective(shape: np.ndarray, subset) -> float:
        mean, fails = _stats(shape, subset)
        if mean >= 1e6:
            return 1e6
        return mean + 50.0 * fails / max(len(subset), 1)

    def _shape_from_x(x: np.ndarray) -> np.ndarray:
        return _normalize(obj0 * np.exp(np.asarray(x, dtype=float)))

    res1 = minimize(lambda x: _objective(_shape_from_x(x), small_set),
                    np.zeros(3), method="Nelder-Mead",
                    options={"maxiter": 150, "xatol": 1e-3, "fatol": 1e-2})
    res2 = minimize(lambda x: _objective(_shape_from_x(x), large_set),
                    res1.x, method="Nelder-Mead",
                    options={"maxiter": 220, "xatol": 1e-3, "fatol": 1e-3})
    shape = _shape_from_x(res2.x)
    large_mean, large_fails = _stats(shape, large_set)
    info = {
        "scale_xyz": np.exp(res2.x).round(6).tolist(),
        "small_frames": len(small_set),
        "large_frames": len(large_set),
        "small_err_px": round(float(res1.fun), 4),
        "large_err_px": round(float(large_mean), 4),
        "large_failures": int(large_fails),
        "diameter_m": round(float(np.linalg.norm(shape.max(0) - shape.min(0))), 6),
    }
    return shape, info


def rebuild_pose_from_pnp(args, calib) -> int:
    """由已有的 YOLO-pose 标签反解 ``labels_6d``（原始 VICON 标签不可用时的修复路径）。"""
    out_root = args.out / args.model
    if not out_root.is_dir():
        raise SystemExit(f"找不到 {out_root}；先完整转换或确认 --out 指对了。")
    missing = [name for name in ("images", "labels", "labels_6d")
               if not (out_root / name).is_dir()]
    if missing:
        raise SystemExit(f"{out_root} 缺少 {missing}，不像转换产物。")

    result, K, D = resolve_calibration(args, calib)
    obj = np.asarray(result["object_points"], dtype=np.float64)
    if args.fit_pnp_shape:
        diameter0 = result.get("model_diameter_m")
        if not diameter0:
            diameter0 = float(np.linalg.norm(obj.max(0) - obj.min(0)))
        obj, fit_info = fit_shape_from_pnp(
            out_root / "labels" / "train", K, obj, float(diameter0),
            small=args.fit_small_frames, large=args.fit_large_frames, seed=args.seed)
        result["object_points"] = obj.tolist()
        result["model_diameter_m"] = float(np.linalg.norm(obj.max(0) - obj.min(0)))
        result["shape_fit"] = fit_info
        print(f"[pnp] 形状拟合：scale_xyz={fit_info['scale_xyz']}，"
              f"大样本残差 {fit_info['large_err_px']:.4f} px")
    rows: dict[str, int] = {}
    stats: dict[str, dict] = {}
    all_errs: list[float] = []

    for split in ("train", "val"):
        label_dir = out_root / "labels" / split
        csv_path = out_root / "labels_6d" / f"{split}.csv"
        if csv_path.is_file():
            shutil.copy2(csv_path, csv_path.with_name(csv_path.name + ".bak"))
            print(f"[pnp] 旧 CSV 已备份 -> {csv_path.name}.bak")
        out_lines = ["image,obj_index,tx,ty,tz,qx,qy,qz,qw"]
        n_frames = n_ok = n_fail = n_obj = 0
        reasons: dict[str, int] = {}

        for label_path in sorted(label_dir.glob("*.txt")):
            n_frames += 1
            stem = label_path.stem
            size = _image_size_for(out_root, split, stem)
            if size is None:
                reasons["缺图像"] = reasons.get("缺图像", 0) + 1
                n_fail += 1
                continue
            W, H = size
            frame_rows: list[list[float]] = []
            frame_ok = True
            for line in label_path.read_text(encoding="utf-8").splitlines():
                vals = [float(v) for v in line.split()]
                if len(vals) < 5 + 3 * 9:
                    frame_ok = False
                    reasons["标签列数异常"] = reasons.get("标签列数异常", 0) + 1
                    break
                kpts = np.asarray(vals[5:5 + 27], dtype=float).reshape(9, 3)
                visible = kpts[:, 2] > 0
                if visible.sum() < 6:
                    frame_ok = False
                    reasons["可见点<6"] = reasons.get("可见点<6", 0) + 1
                    break
                uv = np.stack([kpts[:, 0] * W, kpts[:, 1] * H], axis=1)
                uv[~visible] = np.nan
                solved = _pnp_from_keypoints(obj, uv, K, args.max_pose_reproj_px)
                if solved is None:
                    frame_ok = False
                    reasons["PnP 失败/超阈值"] = reasons.get("PnP 失败/超阈值", 0) + 1
                    break
                R, t, err = solved
                all_errs.append(err)
                q = rot_to_quat(R)
                frame_rows.append([*t, *q])
                n_obj += 1

            if not frame_ok:
                n_fail += 1
                continue
            for obj_index, row in enumerate(frame_rows):
                out_lines.append(
                    f"{stem},{obj_index}," + ",".join(f"{v:.8f}" for v in row))
            n_ok += 1

        csv_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
        rows[split] = _count_files(out_root / "images" / split, IMG_SUFFIXES)
        stats[split] = {"frames": n_frames, "with_pose": n_ok,
                        "skipped": n_fail, "objects": n_obj, "reasons": reasons}
        print(f"[pnp] {split}: {n_ok}/{n_frames} 帧有 6D 位姿，跳过 {n_fail}"
              + (f"（{reasons}）" if reasons else ""))

    mean_err = float(np.mean(all_errs)) if all_errs else float("nan")
    max_err = float(np.max(all_errs)) if all_errs else float("nan")
    total_frames = sum(v["frames"] for v in stats.values())
    total_pose = sum(v["with_pose"] for v in stats.values())
    coverage = (total_pose / total_frames) if total_frames else 0.0
    strict = all(v["skipped"] == 0 and v["with_pose"] == v["frames"]
                 for v in stats.values())
    if coverage < 0.5:
        print(f"[pnp] 警告：6D 位姿覆盖率只有 {coverage:.1%}；"
              "L_rep 实际几乎不生效。请用 --fit-pnp-shape 拟合该机型形状，"
              "或先用 --no-lrep 跑 2D baseline。")
    yaml_path = write_dataset_yaml(
        out_root, args.model, result, D, rows, args.val_ratio,
        split_note="沿用磁盘上已有的 train/val 划分（--pose-from-pnp 未重新划分）",
        pose_source="pnp (cv2.solvePnP from undistorted 2D labels)",
        pose_strict=strict,
        pose_reproj_px=(mean_err if math.isfinite(mean_err) else None),
        pose_coverage=coverage,
    )
    manifest_path = out_root / "manifest.json"
    manifest = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            manifest = {}
    manifest["pose_rebuild"] = {
        "source": "pnp",
        "mean_reproj_px": mean_err,
        "max_reproj_px": max_err,
        "coverage": coverage,
        "strict": strict,
        "splits": stats,
        "shape_fit": result.get("shape_fit"),
    }
    for split, info in stats.items():
        if isinstance(manifest.get("splits"), dict) and split in manifest["splits"]:
            manifest["splits"][split]["with_pose"] = info["with_pose"]
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    print(f"[pnp] 平均重投影 {mean_err:.4f} px（最大 {max_err:.4f}），"
          f"覆盖率 {coverage:.1%}")
    print(f"[pnp] dataset.yaml -> {yaml_path.as_posix()}（pose_strict={str(strict).lower()}）")
    print("\n[下一步]")
    print(f"  python3 tools/check_pose_dataset.py --data {yaml_path.as_posix()}")
    return 0


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
    ap.add_argument("--yaml-only", action="store_true",
                    help="只在已转换好的 <out>/<机型> 上补写 dataset.yaml + manifest.json"
                         "（不动 images/labels；用于描述文件丢失或数据已拷到别的盘）")
    ap.add_argument("--pose-from-pnp", action="store_true",
                    help="由已有 YOLO-pose 标签 + K + object_points 做 PnP，重建 labels_6d"
                         "（原始 VICON labels 不可用时的修复路径；旧 CSV 会备份）")
    ap.add_argument("--fit-pnp-shape", action="store_true",
                    help="PnP 前先按 2D 标签拟合机体的各向异性尺度（Mavic 2 必须；"
                         "只改形状与 model_diameter，不覆盖标定 JSON）")
    ap.add_argument("--fit-small-frames", type=int, default=40,
                    help="形状拟合第一段用的帧数（同一序列，给初值）")
    ap.add_argument("--fit-large-frames", type=int, default=200,
                    help="形状拟合第二段用的帧数（跨序列细化）")
    ap.add_argument("--max-pose-reproj-px", type=float, default=5.0,
                    help="位姿投影一致性阈值（像素）；超过则该帧不产出 6D 位姿")
    args = ap.parse_args()

    calib = _resolve_impl()
    if args.yaml_only:
        return descriptor_only(args, calib)
    if args.pose_from_pnp:
        return rebuild_pose_from_pnp(args, calib)

    src = args.mav6d / args.model
    if not src.is_dir():
        raise SystemExit(
            f"找不到 {src}\n"
            "（要转换原始 MAV6D 需要 --mav6d 指向它；"
            "如果数据已经转换好、只缺 dataset.yaml，用 --yaml-only）"
        )

    result, K, D = resolve_calibration(args, calib)

    images = index_images(args.mav6d, args.model)
    print(f"[convert] 扫到图像 {len(images)} 张")
    if args.use_test_as_val:
        train_keys = read_listing(args.mav6d, args.model, "train")
        val_keys = read_listing(args.mav6d, args.model, "test")
    else:
        listed = read_listing(args.mav6d, args.model, "train")
        if not listed:
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
            line, pose_line, why = convert_frame(
                img, labels_dir, yolo6d_dir, K, D, calib,
                object_points=np.asarray(result["object_points"], dtype=float),
                max_reproj_px=args.max_pose_reproj_px,
            )
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

    total_frames = sum(v["images"] for v in manifest["splits"].values())
    total_pose = sum(v["with_pose"] for v in manifest["splits"].values())
    coverage = (total_pose / total_frames) if total_frames else 0.0
    strict = all(v["with_pose"] == v["images"] for v in manifest["splits"].values())
    yaml_path = write_dataset_yaml(
        out_root, args.model, result, D,
        {k: v["images"] for k, v in manifest["splits"].items()},
        args.val_ratio,
        pose_source="vicon (labels[9:16])",
        pose_strict=strict,
        pose_coverage=coverage,
    )
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
