"""生成"训练回路自检"用的极小检测数据集（离线、确定性、秒级）。

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

类别定义（故意区分尺度，方便看分层指标）：

    ======  ==================  ====================
    id      名称                尺寸范围（像素）
    ======  ==================  ====================
    0       small-target        4 – 16
    1       large-object        32 – 64
    ======  ==================  ====================
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

IMAGE_SIZE = 320


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


def build(out: Path, n_train: int, n_val: int, seed: int = 0) -> Path:
    for split, count in (("train", n_train), ("val", n_val)):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        rng = random.Random(seed + (0 if split == "train" else 10_000))
        for i in range(count):
            img, labels = _draw_image(rng, i)
            stem = f"{split}_{i:04d}"
            img.save(out / "images" / split / f"{stem}.jpg", quality=92)
            (out / "labels" / split / f"{stem}.txt").write_text(
                "\n".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for c, cx, cy, w, h in labels)
                + "\n",
                encoding="utf-8",
            )

    yaml_path = out / "dataset.yaml"
    yaml_path.write_text(
        "# 由 tools/make_dummy_dataset.py 生成的合成数据集（训练回路自检用，勿用于任何结论）\n"
        f"path: {out.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: small-target\n"
        "  1: large-object\n",
        encoding="utf-8",
    )
    return yaml_path


def main() -> int:
    ap = argparse.ArgumentParser(description="生成合成小目标检测数据集（自检用）")
    ap.add_argument("--out", type=Path, default=Path("tests/.tmp/tiny-detect"))
    ap.add_argument("--n-train", type=int, default=12)
    ap.add_argument("--n-val", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    path = build(args.out, args.n_train, args.n_val, args.seed)
    print(f"[dummy] 图像 {args.n_train} 训练 / {args.n_val} 验证 → {args.out}")
    print(f"[dummy] 数据集配置 → {path}")
    print("[下一步] python tools/train.py --variant variants/SDD-YOLO26n/variant.yaml "
          f"--data {path} --epochs 2 --batch 2 --imgsz 320 --set workers=0 --set plots=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
