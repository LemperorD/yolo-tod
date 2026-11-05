"""生成"训练回路自检"用的极小数据集（离线、确定性、秒级）。"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

IMAGE_SIZE = 320

COCO_SKELETON = (
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12),
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
    (1, 3), (2, 4), (3, 5), (4, 6),
)
COCO_POSE_LAYOUT = (
    (0.50, 0.03),
    (0.42, 0.00), (0.58, 0.00),
    (0.33, 0.02), (0.67, 0.02),
    (0.25, 0.22), (0.75, 0.22),
    (0.12, 0.42), (0.88, 0.42),
    (0.05, 0.60), (0.95, 0.60),
    (0.36, 0.52), (0.64, 0.52),
    (0.34, 0.76), (0.66, 0.76),
    (0.32, 1.00), (0.68, 1.00),
)

UAV_BODY_POINTS = (
    (+0.000, +0.000, +0.000),
    (+0.150, +0.150, +0.040),
    (+0.150, -0.150, +0.040),
    (+0.150, +0.150, -0.040),
    (+0.150, -0.150, -0.040),
    (-0.150, +0.150, +0.040),
    (-0.150, -0.150, +0.040),
    (-0.150, +0.150, -0.040),
    (-0.150, -0.150, -0.040),
)
UAV_DIAMETER = 0.45
UAV_CAMERA = ((320.0, 0.0, 320.0), (0.0, 320.0, 240.0), (0.0, 0.0, 1.0))


def _quat_from_axis_angle(axis, angle):
    """轴角 → 四元数 ``(qx, qy, qz, qw)``（纯 Python，避免为合成数据引入 torch）。"""
    import math

    n = math.sqrt(sum(a * a for a in axis)) or 1.0
    ax = [a / n for a in axis]
    s = math.sin(angle / 2.0)
    return (ax[0] * s, ax[1] * s, ax[2] * s, math.cos(angle / 2.0))


def _rot_from_quat(q):
    """四元数 → 3×3 旋转矩阵（与 tod/data/pose6d.py 同式）。"""
    x, y, z, w = q
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def _project(points, R, t, K, width, height):
    """机体 3D 点 → 像素（论文式 (12)）→ ``(像素点列表, 可见性列表)``。"""
    out_px, out_vis = [], []
    for p in points:
        cam = [sum(R[i][j] * p[j] for j in range(3)) + t[i] for i in range(3)]
        if cam[2] <= 1e-6:
            out_px.append((0.0, 0.0))
            out_vis.append(0.0)
            continue
        u = (K[0][0] * cam[0] + K[0][2] * cam[2]) / cam[2]
        v = (K[1][1] * cam[1] + K[1][2] * cam[2]) / cam[2]
        inside = 0.0 <= u < width and 0.0 <= v < height
        out_px.append((u, v))
        out_vis.append(2.0 if inside else 0.0)
    return out_px, out_vis


def _canonical_layout(points):
    """把米制的 3D 关键点投影到一个标准视角，得到 0..1 单位方框内的 2D 布局。"""
    R = _rot_from_quat(_quat_from_axis_angle((0.4, 0.8, 0.2), 0.9))
    t = (0.0, 0.0, 2.0)
    px, _ = _project(points, R, t, UAV_CAMERA, 640, 480)
    xs = [p[0] for p in px]
    ys = [p[1] for p in px]
    w = max(max(xs) - min(xs), 1e-6)
    h = max(max(ys) - min(ys), 1e-6)
    return tuple(((x - min(xs)) / w, (y - min(ys)) / h) for x, y in px)


UAV_POSE_LAYOUT = _canonical_layout(UAV_BODY_POINTS)
UAV_SKELETON = (
    (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8),
    (1, 2), (3, 4), (5, 6), (7, 8),
    (1, 5), (2, 6), (3, 7), (4, 8),
)
UAV_MIRROR = (0, 2, 1, 4, 3, 6, 5, 8, 7)
PERSON_MIRROR = (0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15)

LAYOUTS = {
    "person": (COCO_POSE_LAYOUT, COCO_SKELETON, PERSON_MIRROR,
               ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
                "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                "left_wrist", "right_wrist", "left_hip", "right_hip",
                "left_knee", "right_knee", "left_ankle", "right_ankle"]),
    "uav": (UAV_POSE_LAYOUT, UAV_SKELETON, UAV_MIRROR,
            ["centroid", "corner_1", "corner_2", "corner_3", "corner_4",
             "corner_5", "corner_6", "corner_7", "corner_8"]),
}


def _draw_image(rng: random.Random, index: int):
    """返回 (PIL.Image, labels)，labels 为 ``[(cls, cx, cy, w, h), ...]``（归一化）。"""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter

    arr = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 40, dtype=np.uint8)
    noise = np.random.default_rng(rng.randrange(1 << 30)).integers(0, 60, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)

    labels: list[tuple[int, float, float, float, float]] = []

    def add(cls: int, size: int) -> None:
        half = size // 2
        cx = rng.randint(half + 2, IMAGE_SIZE - half - 3)
        cy = rng.randint(half + 2, IMAGE_SIZE - half - 3)
        value = 235 if cls == 0 else 190
        draw.rectangle((cx - half, cy - half, cx + half, cy + half), fill=(value, value, value))
        labels.append((cls, cx / IMAGE_SIZE, cy / IMAGE_SIZE, size / IMAGE_SIZE, size / IMAGE_SIZE))

    for _ in range(rng.randint(3, 6)):
        add(0, rng.randint(4, 16))
    for _ in range(rng.randint(1, 2)):
        add(1, rng.randint(32, 64))

    img = img.filter(ImageFilter.GaussianBlur(0.6))
    return img, labels


def _draw_pose_image(rng: random.Random, index: int, layout: str = "person",
                     n_kpt: int | None = None):
    """姿态版：画火柴人 / 无人机 → ``(PIL.Image, labels, poses)``。"""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter

    points_table, skeleton, _mirror, _names = LAYOUTS[layout]
    n_kpt = int(n_kpt or len(points_table))
    if not 1 <= n_kpt <= len(points_table):
        raise ValueError(f"layout={layout!r} 只支持 1..{len(points_table)} 个点，收到 {n_kpt}")
    points_table = points_table[:n_kpt]
    skeleton = [(a, b) for a, b in skeleton if a < n_kpt and b < n_kpt]

    arr = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 40, dtype=np.uint8)
    noise = np.random.default_rng(rng.randrange(1 << 30)).integers(0, 60, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)

    labels: list[tuple] = []
    poses: list[tuple] = []
    K = UAV_CAMERA
    S = float(IMAGE_SIZE)

    def add(cls: int, box: int) -> None:
        """在随机位置放一个 ``box`` 像素的目标。"""
        bright = 240 if cls == 0 else 200
        if layout == "uav":
            f = K[0][0]
            q = _quat_from_axis_angle(
                (rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1)),
                rng.uniform(-0.9, 0.9))
            R = _rot_from_quat(q)
            pts3d = UAV_BODY_POINTS[:n_kpt]
            rotated = [tuple(sum(R[i][j] * p[j] for j in range(3)) for i in range(3))
                       for p in pts3d]
            cx3 = tuple(sum(p[i] for p in rotated) / len(rotated) for i in range(3))
            target = max(box, 8) / 2.0
            offs = [(p[0] - cx3[0], p[1] - cx3[1], p[2] - cx3[2]) for p in rotated]
            E = [rng.uniform(-0.25, 0.25), rng.uniform(-0.25, 0.25)]
            z = max(f * max(UAV_DIAMETER / 2.0, 1e-3) / max(target, 1e-3), 0.5)
            ok = False
            for _ in range(40):
                t = (z * E[0] - cx3[0], z * E[1] - cx3[1], z - cx3[2])
                px, vis = _project(pts3d, R, t, K, S, S)
                us = [p[0] for p in px]
                vs = [p[1] for p in px]
                mu, mv = sum(us) / len(us), sum(vs) / len(vs)
                radius = max(max(abs(x - mu), abs(y - mv)) for x, y in px)
                if not (radius > 1e-9) or radius > 1e6 or z > 1e6:
                    break
                if abs(mu - S / 2) < 0.5 and abs(mv - S / 2) < 0.5 \
                        and abs(radius - target) <= max(0.02 * target, 0.3):
                    ok = True
                    break
                E[0] -= (mu - S / 2) / f
                E[1] -= (mv - S / 2) / f
                z *= radius / target
            if not ok:
                return
            px, vis = _project(pts3d, R, tuple(t), K, S, S)
            inside = [(x, y) for (x, y), v in zip(px, vis) if v > 0]
            if len(inside) != n_kpt:
                return
            xs = [p[0] for p in inside]
            ys = [p[1] for p in inside]
            x0, y0 = min(xs), min(ys)
            w, h = max(xs) - x0, max(ys) - y0
            points = px
            poses.append((t[0], t[1], t[2], *q))
            cx0, cy0 = (x0 + w / 2) / S, (y0 + h / 2) / S
            ww, hh = min(w / S, 1.0), min(h / S, 1.0)
            cx0 = min(max(cx0, ww / 2), 1.0 - ww / 2)
            cy0 = min(max(cy0, hh / 2), 1.0 - hh / 2)
            precomputed = (cls, cx0, cy0, max(ww, 1e-4), max(hh, 1e-4),
                           [(points[k][0] / S, points[k][1] / S, vis[k])
                            for k in range(n_kpt)])
        else:
            xs = [p[0] for p in points_table]
            ys = [p[1] for p in points_table]
            w = max(6, int(box * max(max(xs) - min(xs), 0.3)))
            h = max(6, int(box * max(max(ys) - min(ys), 0.3)))
            x0 = rng.randint(2, max(3, IMAGE_SIZE - w - 3))
            y0 = rng.randint(2, max(3, IMAGE_SIZE - h - 3))
            points = [(x0 + px * w, y0 + py * h) for px, py in points_table]
            precomputed = None

        width = max(1, box // 14)
        for a, b in skeleton:
            draw.line((points[a], points[b]), fill=(bright, bright, bright), width=width)
        radius = max(1, box // 10)
        for px_, py_ in points:
            draw.ellipse((px_ - radius, py_ - radius, px_ + radius, py_ + radius),
                         fill=(bright, bright, bright))
        if precomputed is not None:
            labels.append(precomputed)
        else:
            labels.append((cls, (x0 + w / 2) / S, (y0 + h / 2) / S, w / S, h / S,
                           [(px_ / S, py_ / S, 2.0) for px_, py_ in points]))

    for _ in range(rng.randint(2, 4)):
        add(0, rng.randint(12, 24))
    for _ in range(rng.randint(1, 2)):
        add(1, rng.randint(48, 112))

    img = img.filter(ImageFilter.GaussianBlur(0.4))
    return img, labels, poses


def mirror_for(layout: str, n_kpt: int) -> list[int]:
    """给出**前 n_kpt 个点**的合法镜像置换（``flip_idx``）。"""
    raw = list(LAYOUTS[layout][2])[:n_kpt]
    mirror = list(range(n_kpt))
    seen: set[int] = set()
    for i, j in enumerate(raw):
        if i in seen or j in seen or j >= n_kpt or j == i:
            continue
        if raw[j] != i:
            continue
        mirror[i], mirror[j] = j, i
        seen.add(i)
        seen.add(j)
    return mirror


def build(out: Path, n_train: int, n_val: int, seed: int = 0, task: str = "detect",
          kpt_shape: tuple[int, int] = (17, 3), layout: str = "person",
          with_6d: bool | None = None) -> Path:
    """生成合成数据集，返回 dataset.yaml。"""
    if task not in ("detect", "pose"):
        raise ValueError(f"task 只能是 detect / pose，收到 {task!r}")
    if with_6d is None:
        with_6d = (task == "pose" and layout == "uav")
    if with_6d and not (task == "pose" and layout == "uav"):
        raise ValueError(
            "with_6d=True 只在 task='pose' 且 layout='uav' 时有意义"
            "（人形布局没有 3D 几何，无法产生 6D 位姿）。"
        )
    if task == "pose":
        if layout not in LAYOUTS:
            raise ValueError(f"未知姿态布局 {layout!r}；可选 {sorted(LAYOUTS)}")
        available = len(LAYOUTS[layout][0])
        if layout == "person" and tuple(kpt_shape) != (17, 3):
            raise ValueError(
                f"合成火柴人只提供 COCO 的 17 点 3 维标注，不支持 kpt_shape={kpt_shape}。"
                "要非 17 点请用 layout='uav'（顺序见 docs/KEYPOINTS.md）。"
            )
        if not 1 <= int(kpt_shape[0]) <= available:
            raise ValueError(
                f"layout={layout!r} 最多提供 {available} 个点，收到 kpt_shape={kpt_shape}。"
            )
        if int(kpt_shape[1]) != 3:
            raise ValueError(
                f"合成姿态数据只写 3 维关键点 (x, y, visibility)，收到 kpt_shape={kpt_shape}。"
            )

    for split, count in (("train", n_train), ("val", n_val)):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        if with_6d:
            (out / "labels_6d").mkdir(parents=True, exist_ok=True)
        rng = random.Random(seed + (0 if split == "train" else 10_000))
        pose_rows = ["image,obj_index,tx,ty,tz,qx,qy,qz,qw"]
        for i in range(count):
            poses: list[tuple] = []
            if task == "pose":
                img, labels, poses = _draw_pose_image(rng, i, layout=layout,
                                                      n_kpt=int(kpt_shape[0]))
            else:
                img, labels = _draw_image(rng, i)
            stem = f"{split}_{i:04d}"
            img.save(out / "images" / split / f"{stem}.jpg", quality=92)
            lines = []
            for entry in labels:
                cls, cx, cy, w, h = entry[:5]
                text = f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                if task == "pose":
                    flat = " ".join(f"{v:.6f}" for k in entry[5] for v in k)
                    text = f"{text} {flat}"
                lines.append(text)
            (out / "labels" / split / f"{stem}.txt").write_text(
                "\n".join(lines) + "\n", encoding="utf-8")
            if with_6d:
                for j, pose in enumerate(poses):
                    pose_rows.append(f"{stem},{j}," + ",".join(f"{v:.8f}" for v in pose))
        if with_6d:
            (out / "labels_6d" / f"{split}.csv").write_text(
                "\n".join(pose_rows) + "\n", encoding="utf-8")

    header = (
        "# 由 tools/make_dummy_dataset.py 生成的合成数据集（训练回路自检用，勿用于任何结论）\n"
        f"# 任务：{task}" + (f"｜骨架布局：{layout}\n" if task == "pose" else "\n")
    )
    body = (
        f"path: {out.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
    )
    if task == "pose":
        n_kpt, n_dim = int(kpt_shape[0]), int(kpt_shape[1])
        names = list(LAYOUTS[layout][3])[:n_kpt]
        mirror = mirror_for(layout, n_kpt)
        body += (f"kpt_shape: [{n_kpt}, {n_dim}]\n"
                 f"flip_idx: [{', '.join(str(m) for m in mirror)}]\n")
        for cls in (0, 1):
            body += f"kpt_names:\n" if cls == 0 else ""
            body += f"  {cls}: [{', '.join(names)}]\n"
        if with_6d:
            pts = UAV_BODY_POINTS[:n_kpt]
            body += (
                "tod:\n"
                "  kpt_definition_status: synthetic\n"
                "  # 合成数据的相机与机体几何（与生成标签时使用的完全一致，故 L_rep 自洽）\n"
                f"  camera_matrix: {[list(r) for r in UAV_CAMERA]}\n"
                f"  model_diameter: {UAV_DIAMETER}\n"
                "  object_points:\n"
                + "".join(f"    - [{p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}]\n" for p in pts)
                + "  pose_labels: labels_6d\n"
                "  reproj_warmup_epochs: 15\n"
                "  domain: synthetic\n"
            )
    names_yaml = ("  0: small-target\n  1: large-object\n" if task == "detect" else
                  "  0: tiny-target\n  1: target\n")
    (out / "dataset.yaml").write_text(header + body + "names:\n" + names_yaml, encoding="utf-8")
    return out / "dataset.yaml"


def main() -> int:
    ap = argparse.ArgumentParser(description="生成合成数据集（自检用）：检测或姿态")
    ap.add_argument("--out", type=Path, default=Path("tests/.tmp/tiny-detect"))
    ap.add_argument("--n-train", type=int, default=12)
    ap.add_argument("--n-val", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task", choices=("detect", "pose"), default="detect",
                    help="pose 会额外画关键点并写 kpt_shape / flip_idx / kpt_names")
    ap.add_argument("--layout", choices=tuple(LAYOUTS), default="person",
                    help="姿态骨架：person（COCO 17 点）或 uav（机体，最多 9 点）")
    ap.add_argument("--kpt-shape", default=None, metavar="N,D",
                    help="关键点形状，如 9,3（默认：person=17,3 / uav=9,3）")
    args = ap.parse_args()

    kpt_shape = (17, 3) if args.layout == "person" else (9, 3)
    if args.kpt_shape:
        parts = [int(x) for x in args.kpt_shape.replace(" ", "").split(",")]
        if len(parts) != 2:
            raise SystemExit(f"--kpt-shape 需要 N,D 形式，收到 {args.kpt_shape!r}")
        kpt_shape = (parts[0], parts[1])

    path = build(args.out, args.n_train, args.n_val, args.seed, task=args.task,
                 kpt_shape=kpt_shape, layout=args.layout)
    print(f"[dummy] 任务={args.task} 布局={args.layout} kpt_shape={list(kpt_shape)} "
          f"图像 {args.n_train} 训练 / {args.n_val} 验证 → {args.out}")
    print(f"[dummy] 数据集配置 -> {path.as_posix()}")
    if args.task == "pose":
        print("[下一步] python3 tools/check_pose_dataset.py --data "
              f"{path.as_posix()}  # 校验 kpt_shape/flip_idx/标签一致性")
        print("[下一步] python3 tests/train_pose_smoke.py     # 姿态训练回路自检")
    else:
        print("[下一步] python3 tools/train.py --variant variants/SDD-YOLO26n/variant.yaml "
              f"--data {path.as_posix()} --epochs 2 --batch 2 --imgsz 320 "
              "--set workers=0 --set plots=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
