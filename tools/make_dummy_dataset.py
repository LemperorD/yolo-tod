"""生成"训练回路自检"用的极小数据集（离线、确定性、秒级）。

**为什么需要它**：VisDrone 等真实数据集体积大、还要人工准备，但"训练回路是否真的跑得通"
（Trainer 钩子、准则补丁、MuSGD + AMP、EMA、验证、checkpoint 存取）与数据集内容无关。
这里合成一批图像（噪声背景 + 若干 4–16 px 的高亮小目标 + 若干 32–64 px 的普通目标），
按 YOLO 格式写好标签与 dataset.yaml，用于：

    * `tools/train.py` 的端到端自检（几秒钟跑完 1–2 个 epoch）；
    * `tools/val.py` / `tools/ablation.py` / 尺度分层评测的冒烟数据（图里同时有小/大目标，
      正好检验 `AP_small` 与整体 AP 的差异）。

用法::

    python tools/make_dummy_dataset.py --out tests/.tmp/tiny-detect --n-train 12 --n-val 4
    python tools/train.py --variant variants/SDD-YOLO26n/variant.yaml \
        --data tests/.tmp/tiny-detect/dataset.yaml --epochs 2 --batch 2 --imgsz 320 `
        --set workers=0 --set plots=False --name sdd-tiny

**姿态模式（``--task pose``）**：额外画出**火柴人**与 17 个 COCO 关键点，
标签写成 YOLO-pose 格式 ``cls cx cy w h x1 y1 v1 …``，并在 dataset.yaml 里写
``kpt_shape: [17, 3]``。关键点数量与顺序与 COCO 一致（鼻/眼/耳/肩/肘/腕/髋/膝/踝），
因此可以直接配 ``sigma_strategy=person``；尺度覆盖 tiny（12–24 px）到日常（48–112 px），
用来验证"OKS 的分母含框面积"导致的**小目标关键点更难**这一现象。

    为什么姿态人形刻意画成**完整骨架**而不是实心块：实心块的关键点既不可辨识、
    也不符合任何真实标注分布；骨架的 17 个关节位置与像素图案强相关，
    训练 1–2 epoch 就能看到 loss 明显下降（说明回路真的在学），
    这才是有意义的"回路自检"。

类别定义（故意区分尺度，方便看分层指标）：

    ======  ==================  ====================
    id      名称                尺寸范围（像素）
    ======  ==================  ====================
    0       small-target        4 – 16（姿态模式：12 – 24）
    1       large-object        32 – 64（姿态模式：48 – 112）
    ======  ==================  ====================
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

IMAGE_SIZE = 320

#: COCO 17 点骨骼连接（人体姿态标准），姿态像素图按它连线。
COCO_SKELETON = (
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12),
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
    (1, 3), (2, 4), (3, 5), (4, 6),
)
#: COCO 17 点在 0..1 单位方框里的标准比例位置 (x, y)。
COCO_POSE_LAYOUT = (
    (0.50, 0.03),                                     # 0 鼻
    (0.42, 0.00), (0.58, 0.00),                       # 1/2 眼
    (0.33, 0.02), (0.67, 0.02),                       # 3/4 耳
    (0.25, 0.22), (0.75, 0.22),                       # 5/6 肩
    (0.12, 0.42), (0.88, 0.42),                       # 7/8 肘
    (0.05, 0.60), (0.95, 0.60),                       # 9/10 腕
    (0.36, 0.52), (0.64, 0.52),                       # 11/12 髋
    (0.34, 0.76), (0.66, 0.76),                       # 13/14 膝
    (0.32, 1.00), (0.68, 1.00),                       # 15/16 踝
)


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

    for _ in range(rng.randint(3, 6)):          # 3–6 个小目标（4–16 px）
        add(0, rng.randint(4, 16))
    for _ in range(rng.randint(1, 2)):          # 1–2 个普通目标（32–64 px）
        add(1, rng.randint(32, 64))

    img = img.filter(ImageFilter.GaussianBlur(0.6))     # 轻微模糊，避免"完美方框"过于简单
    return img, labels


def _draw_pose_image(rng: random.Random, index: int):
    """姿态版：画火柴人 → ``(PIL.Image, [(cls, cx, cy, w, h, kpts)])``。

    ``kpts`` 是归一化的 ``[(x, y, vis), ...]``（17 点，``vis=2`` 表示可见）。
    """
    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter

    arr = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 40, dtype=np.uint8)
    noise = np.random.default_rng(rng.randrange(1 << 30)).integers(0, 60, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)

    labels: list[tuple] = []

    def add(cls: int, box: int) -> None:
        """在随机位置放一个 ``box`` 像素高的人形；宽度按其 0.55 缩放（人比正方形高）。"""
        w = max(6, int(box * 0.55))
        h = box
        x0 = rng.randint(2, max(3, IMAGE_SIZE - w - 3))
        y0 = rng.randint(2, max(3, IMAGE_SIZE - h - 3))
        bright = 240 if cls == 0 else 200
        points = [(x0 + px * w, y0 + py * h) for px, py in COCO_POSE_LAYOUT]
        width = max(1, box // 14)                     # 细线，避免把 17 个点糊成一团
        for a, b in COCO_SKELETON:
            draw.line((points[a], points[b]), fill=(bright, bright, bright), width=width)
        radius = max(1, box // 10)
        for px, py in points:                         # 关节点画成小圆，给关键点可辨识的落点
            draw.ellipse((px - radius, py - radius, px + radius, py + radius),
                         fill=(bright, bright, bright))
        kpts = [((px) / IMAGE_SIZE, (py) / IMAGE_SIZE, 2.0) for px, py in points]
        labels.append((cls, (x0 + w / 2) / IMAGE_SIZE, (y0 + h / 2) / IMAGE_SIZE,
                       w / IMAGE_SIZE, h / IMAGE_SIZE, kpts))

    for _ in range(rng.randint(2, 4)):          # 2–4 个小人（12–24 px）
        add(0, rng.randint(12, 24))
    for _ in range(rng.randint(1, 2)):          # 1–2 个日常尺度（48–112 px）
        add(1, rng.randint(48, 112))

    img = img.filter(ImageFilter.GaussianBlur(0.4))
    return img, labels


def build(out: Path, n_train: int, n_val: int, seed: int = 0, task: str = "detect",
          kpt_shape: tuple[int, int] = (17, 3)) -> Path:
    """生成合成数据集，返回 dataset.yaml。

    Args:
        task: ``"detect"``（实心方块 + 框）或 ``"pose"``（火柴人 + 框 + 17 关键点）。
        kpt_shape: 姿态模式下的关键点形状（写进 dataset.yaml；必须是 ``[17, 3]``，
            因为火柴人只画了 COCO 的 17 点）。
    """
    if task not in ("detect", "pose"):
        raise ValueError(f"task 只能是 detect / pose，收到 {task!r}")
    if task == "pose" and tuple(kpt_shape) != (17, 3):
        raise ValueError(
            f"合成火柴人只提供 COCO 的 17 点 3 维标注，不支持 kpt_shape={kpt_shape}。"
        )

    for split, count in (("train", n_train), ("val", n_val)):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        rng = random.Random(seed + (0 if split == "train" else 10_000))
        for i in range(count):
            if task == "pose":
                img, labels = _draw_pose_image(rng, i)
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

    header = (
        "# 由 tools/make_dummy_dataset.py 生成的合成数据集（训练回路自检用，勿用于任何结论）\n"
        f"# 任务：{task}\n"
    )
    body = (
        f"path: {out.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
    )
    if task == "pose":
        body += (f"kpt_shape: [{int(kpt_shape[0])}, {int(kpt_shape[1])}]\n"
                 "# 关键点名称（COCO 17 点顺序；框架用它做可视化）\n"
                 "kpt_names:\n"
                 "  0: [nose, left_eye, right_eye, left_ear, right_ear, left_shoulder,\n"
                 "      right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist,\n"
                 "      left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle]\n"
                 "  1: [nose, left_eye, right_eye, left_ear, right_ear, left_shoulder,\n"
                 "      right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist,\n"
                 "      left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle]\n")
    names = ("  0: small-target\n  1: large-object\n" if task == "detect" else
             "  0: tiny-person\n  1: person\n")
    (out / "dataset.yaml").write_text(header + body + "names:\n" + names, encoding="utf-8")
    return out / "dataset.yaml"


def main() -> int:
    ap = argparse.ArgumentParser(description="生成合成数据集（自检用）：检测或姿态")
    ap.add_argument("--out", type=Path, default=Path("tests/.tmp/tiny-detect"))
    ap.add_argument("--n-train", type=int, default=12)
    ap.add_argument("--n-val", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task", choices=("detect", "pose"), default="detect",
                    help="pose 会额外画 17 个 COCO 关键点并写 kpt_shape")
    args = ap.parse_args()

    path = build(args.out, args.n_train, args.n_val, args.seed, task=args.task)
    print(f"[dummy] 任务={args.task} 图像 {args.n_train} 训练 / {args.n_val} 验证 → {args.out}")
    print(f"[dummy] 数据集配置 → {path}")
    if args.task == "pose":
        print("[下一步] python tests/train_pose_smoke.py     # 姿态训练回路自检")
    else:
        print("[下一步] python tools/train.py --variant variants/SDD-YOLO26n/variant.yaml "
              f"--data {path} --epochs 2 --batch 2 --imgsz 320 --set workers=0 --set plots=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

