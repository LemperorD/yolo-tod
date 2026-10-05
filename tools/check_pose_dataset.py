"""RflySim / 任意 YOLO-pose 数据集的**开训前校验**：把静默错误变成明确报错。

**为什么必须有一个独立校验器**：关键点数据的错误几乎全是**静默**的 —— 训练会照常跑、
损失会照常降、指标看起来"有个数"，但学到的是错的解剖结构。已知的静默失败类型：

    ==========================  ================================================
    错误                          后果
    ==========================  ================================================
    kpt_shape 与标签列数不一致      框架在标签缓存阶段才报错（有时被 workers 吞掉），
                                  或在多进程里变成难懂的异常
    flip_idx 缺失/写错             框架**自动关掉 fliplr/flipud**（静默），
                                  或把关键点配到错误的关节上（更糟）
    flip_idx 不是对换置换           翻转两次不回到原状，标注被系统性打乱
    v 全为 0 的点                   该点在 OKS 与 kobj 里被完全忽略（等于没标）
    v 语义用反（0=可见）            掩码整体反转，学出来的点是"不可见点"
    越界点被裁剪到边界            坐标合法但物理错误 → 关键点永远贴边
    尺度分布失衡                   没有小目标样本时，AP_small 无从谈起
    ==========================  ================================================

用法::

    python tools/check_pose_dataset.py --data configs/_base_/datasets/rflysim-pose.yaml
    python tools/check_pose_dataset.py --data ... --split val --sample 20 --json out.json

退出码：有 **错误** 时返回 1（可直接进 CI / 开训前置检查）；只有警告时返回 0。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


# ------------------------------------------------------------------ 检查工具


class Report:
    """收集错误与警告（错误 = 会让训练/评测得出错误结论的问题）。"""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.notes: list[str] = []

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def note(self, msg: str) -> None:
        self.notes.append(msg)

    @property
    def ok(self) -> bool:
        return not self.errors


# ------------------------------------------------------------------ 配置检查


def check_config(cfg: dict, report: Report) -> tuple[tuple[int, int], list[int], list[str]]:
    """校验 kpt_shape / flip_idx / kpt_names 三件套，返回 ``(kpt_shape, flip_idx, names)``。"""
    raw_shape = cfg.get("kpt_shape")
    if not raw_shape:
        report.error(
            "数据集缺少 kpt_shape。姿态数据集**必须**有它，否则框架报 "
            "'No kpt_shape in data yaml' 并中止。参考 configs/_base_/datasets/rflysim-pose.yaml。"
        )
        return (0, 0), [], []
    kpt_shape = tuple(int(x) for x in raw_shape)
    n_kpt, n_dim = kpt_shape
    if n_dim not in (2, 3):
        report.error(f"kpt_shape[1]={n_dim} 非法：只能是 2（x,y）或 3（x,y,visibility）。")
    if n_kpt <= 0:
        report.error(f"kpt_shape[0]={n_kpt} 非法：关键点数必须为正。")
    if n_dim == 2:
        report.warn(
            "kpt_shape 为 2 维（无 visibility）：关键点没有可见性标注，"
            "kobj（可见性）分支会退化成全 1 掩码，遮挡样本无法被忽略。"
        )

    # ---- flip_idx：框架的硬行为（缺失即静默关掉翻转增强）----
    flip_idx = [int(x) for x in (cfg.get("flip_idx") or [])]
    if not flip_idx:
        report.error(
            "缺少 flip_idx。ultralytics 在检测到 use_keypoints 且 flip_idx 为空时会把 "
            "fliplr/flipud **静默置 0**（只打印一条 warning），水平翻转增强直接失效。"
            "请显式写出长度 == kpt_shape[0] 的置换，例如 rflysim-pose.yaml 里的 "
            "flip_idx: [1, 0, 3, 2, 4, 6, 5, 8, 7]。"
        )
    else:
        if len(flip_idx) != n_kpt:
            report.error(
                f"flip_idx 长度 {len(flip_idx)} != kpt_shape[0]={n_kpt}；"
                "框架会直接 raise ValueError('flip_idx length must be equal to kpt_shape[0]')。"
            )
        if sorted(flip_idx) != list(range(len(flip_idx))):
            report.error(
                f"flip_idx={flip_idx} 不是 0..{len(flip_idx) - 1} 的置换（有重复或缺号）；"
                "翻转会把关键点映射到错误的关节上，而且是**静默**的。"
            )
        else:
            bad = [i for i, j in enumerate(flip_idx) if flip_idx[j] != i]
            if bad:
                report.error(
                    f"flip_idx 不是**对换**置换：点 {bad} 翻转两次回不到自身"
                    f"（flip_idx[flip_idx[i]] != i）。这会让数据增强系统性打乱标注。"
                )
            fixed = [i for i, j in enumerate(flip_idx) if i == j]
            report.note(f"flip_idx 中自配对（在镜像轴上）的点：{fixed or '无'}")

    # ---- kpt_names：按类别给；用于可视化和日志 ----
    names = cfg.get("kpt_names") or {}
    # nc 缺省时按 names 的条目数推断（很多数据集只写 names 不写 nc）
    n_cls = int(cfg.get("nc") or len(cfg.get("names") or {}) or 1)
    if not names:
        report.warn(
            "没有 kpt_names：框架会用数字编号做可视化。建议按 docs/KEYPOINTS.md 写清楚点名，"
            "否则半年后没人知道第 7 号点是哪个部位。"
        )
    else:
        for cls in range(n_cls):
            entry = names.get(cls) or names.get(str(cls))
            if entry is None:
                report.warn(f"kpt_names 缺少类别 {cls} 的点名。")
            elif len(entry) != n_kpt:
                report.error(
                    f"kpt_names[{cls}] 有 {len(entry)} 个名字，与 kpt_shape[0]={n_kpt} 不符。"
                )

    status = ((cfg.get("tod") or {}).get("kpt_definition_status") or "").strip().lower()
    if status == "inferred":
        report.warn(
            "关键点定义状态是 **inferred**（本库推断，未与论文核对）。"
            "拿到论文 Figure 11 后请校正 kpt_names/flip_idx 并改为 paper-verified —— "
            "在此之前不要对外宣称「复现了该论文的关键点定义」。"
        )
    elif not status:
        report.warn("未声明 tod.kpt_definition_status（建议标 inferred / paper-verified / custom）。")
    else:
        report.note(f"关键点定义状态：{status}")

    return kpt_shape, flip_idx, names


# ------------------------------------------------------------------ 标签检查


def iter_label_files(root: Path, split: str) -> list[Path]:
    """按 ultralytics 约定定位标签文件。

    数据集 YAML 里 ``train:``/``val:`` 写的是**图像**路径（如 ``images/train``），
    标签在同级 ``labels/train``。早先这里直接拼 ``root/split`` 会找不到任何标签、
    误报"没有数据" —— 所以按优先级依次尝试，并把 ``/images/`` 换成 ``/labels/``。
    """
    value = str(split)
    candidates: list[Path] = []
    if "images" in Path(value).parts:
        parts = list(Path(value).parts)
        parts[parts.index("images")] = "labels"
        candidates.append(root / Path(*parts))
    candidates += [root / "labels" / value, root / value]

    for target in candidates:
        if target.is_dir():
            files = sorted(target.rglob("*.txt"))
            if files:
                return files
        elif target.suffix == ".txt" and target.is_file():     # 列表文件
            lines = [ln.strip() for ln in target.read_text(encoding="utf-8").splitlines()
                     if ln.strip()]
            if lines:
                return [Path(ln) for ln in lines]
    return []


def resolve_labels_dir(cfg: dict, split: str) -> Path | None:
    """返回某个划分的标签目录（供图像/标签配对检查用）。"""
    root = Path(str(cfg.get("path") or "."))
    value = Path(str(split))
    if "images" in value.parts:
        parts = list(value.parts)
        parts[parts.index("images")] = "labels"
        return root / Path(*parts)
    candidate = root / "labels" / value
    return candidate if candidate.is_dir() else None


def check_labels(cfg: dict, kpt_shape: tuple[int, int], flip_idx: list[int],
                 split: str, sample: int, report: Report) -> dict:
    """逐行校验标签格式与数值，返回统计信息。"""
    n_kpt, n_dim = kpt_shape
    # nc 缺省时按 names 的条目数推断（很多数据集只写 names 不写 nc）
    n_cls = int(cfg.get("nc") or len(cfg.get("names") or {}) or 1)
    root = Path(str(cfg.get("path") or "."))
    files = iter_label_files(root, split)
    stats: dict = {"split": split, "files": len(files), "lines": 0, "boxes": 0,
                   "kpt_total": 0, "kpt_vis_hist": {}, "kpt_per_index_vis": {},
                   "box_sides_px": [], "out_of_range": 0, "bbox_mismatch": 0,
                   "empty_files": 0}
    if not files:
        report.error(
            f"在 {root / split} 下没找到任何标签文件（split={split}）。"
            "检查数据集配置里的 path 与 train/val 路径。"
        )
        return stats

    expected = 5 + n_kpt * n_dim
    limit = sample if sample and sample > 0 else len(files)
    for path in files[:limit]:
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if not lines:
            stats["empty_files"] += 1
            continue
        for lineno, line in enumerate(lines, 1):
            parts = line.split()
            stats["lines"] += 1
            if len(parts) != expected:
                report.error(
                    f"{path.name}:{lineno} 有 {len(parts)} 列，期望 {expected}"
                    f"（5 + {n_kpt}×{n_dim}）。要么 kpt_shape 写错，要么这一行漏了关键点。"
                )
                continue
            try:
                cls = int(float(parts[0]))
                vals = [float(x) for x in parts[1:]]
            except ValueError as exc:
                report.error(f"{path.name}:{lineno} 含非数值字段（{exc}）。")
                continue
            stats["boxes"] += 1
            if cls < 0 or cls >= n_cls:
                report.error(f"{path.name}:{lineno} 类别 {cls} 越界（nc={n_cls}）。")
            cx, cy, bw, bh = vals[:4]
            if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0):
                report.error(f"{path.name}:{lineno} 框不是合法的归一化 xywh：{vals[:4]}。")
            # 尺度统计（按框的**归一化**边长收集，最后统一汇总）
            stats["box_sides_px"].append((bw, bh))
            # 关键点
            for k in range(n_kpt):
                x, y = vals[4 + k * n_dim], vals[5 + k * n_dim]
                v = vals[4 + k * n_dim + 2] if n_dim == 3 else 1.0
                stats["kpt_total"] += 1
                stats["kpt_vis_hist"][str(v)] = stats["kpt_vis_hist"].get(str(v), 0) + 1
                per = stats["kpt_per_index_vis"].setdefault(k, {"n": 0, "vis": 0})
                per["n"] += 1
                per["vis"] += 1 if v > 0 else 0
                if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                    stats["out_of_range"] += 1
                    if v != 0:
                        report.warn(
                            f"{path.name}:{lineno} 第 {k} 号关键点坐标越界 "
                            f"({x:.3f},{y:.3f}) 但 v={v}≠0；越界点应写 v=0（未标注），"
                            "不要裁剪到边界（裁剪会得到「永远贴边」的假样本）。"
                        )
                # 点是否落在框内（Looser 判定：允许一定外扩，因为电机点常在框边缘）
                if v > 0:
                    x1, y1 = cx - bw / 2, cy - bh / 2
                    x2, y2 = cx + bw / 2, cy + bh / 2
                    margin = 0.15 * max(bw, bh)
                    if not (x1 - margin <= x <= x2 + margin and y1 - margin <= y <= y2 + margin):
                        stats["bbox_mismatch"] += 1

    if stats["bbox_mismatch"]:
        report.warn(
            f"有 {stats['bbox_mismatch']} 个可见关键点明显落在框外（超过框尺寸 15% 的容差）。"
            "要么框标小了，要么点标错了 —— 值得抽查可视化。"
        )
    if stats["empty_files"]:
        report.warn(f"{stats['empty_files']} 个标签文件是空的（该图无目标；空图过多会让训练偏负样本）。")

    # 每点的可见率：全 0 的点等于没标
    dead = [k for k, s in stats["kpt_per_index_vis"].items() if s["n"] and s["vis"] == 0]
    if dead:
        report.error(
            f"第 {dead} 号关键点在**全部**样本里 v=0（未标注）：它们在 OKS 与 kobj 损失里"
            "被完全忽略，等于这些点白标了。检查标注流程或 v 的语义是否用反。"
        )
    if n_dim == 3:
        vis_vals = set(stats["kpt_vis_hist"])
        unexpected = {v for v in vis_vals if v not in {"0.0", "1.0", "2.0"}}
        if unexpected:
            report.warn(
                f"visibility 出现了非 0/1/2 的取值 {sorted(unexpected)}。"
                "框架只判断 `v != 0`，所以 0.5 之类会被当成「可见」，容易埋雷。"
            )
    return stats


def summarize_scale(stats: dict, cfg: dict, report: Report) -> None:
    """按归一化框边长估计像素尺度分布（需要图像尺寸；这里给出相对分布）。"""
    sides = stats.get("box_sides_px") or []
    if not sides:
        return
    longest = sorted(max(w, h) for w, h in sides)
    n = len(longest)
    stats["side_rel_p50"] = longest[n // 2]
    stats["side_rel_p90"] = longest[int(n * 0.9)]
    stats["side_rel_min"] = longest[0]
    stats["side_rel_max"] = longest[-1]
    report.note(
        f"{stats['split']} 框长边（归一化）：min={longest[0]:.4f} / p50={longest[n // 2]:.4f} / "
        f"p90={longest[int(n * 0.9)]:.4f} / max={longest[-1]:.4f}"
        "（换算成像素请乘以输入分辨率；小目标变体要确认有足够多的小框）"
    )


def check_image_pairing(cfg: dict, split: str, report: Report) -> None:
    """标签与图像的配对是否完整（漏标/多余标签是常见的数据集事故）。"""
    root = Path(str(cfg.get("path") or "."))
    value = Path(str(split))
    images = (root / value) if (root / value).is_dir() else (root / "images" / value)
    labels = resolve_labels_dir(cfg, split)
    if not images.is_dir() or labels is None or not labels.is_dir():
        report.note(f"跳过图像/标签配对检查（images={images}，labels={labels}）。")
        return
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    stems = {p.stem for p in images.iterdir() if p.suffix.lower() in suffixes}
    lbl = {p.stem for p in labels.glob("*.txt")}
    missing = sorted(stems - lbl)
    extra = sorted(lbl - stems)
    if missing:
        report.warn(f"{len(missing)} 张图没有对应标签（前 5 个：{missing[:5]}）。"
                    "无目标图是允许的（空标签），但若本该有目标就是漏标。")
    if extra:
        report.error(f"{len(extra)} 个标签没有对应图像（前 5 个：{extra[:5]}）——"
                     "训练时会报 corrupt 或被静默跳过。")
    if not missing and not extra:
        report.note(f"图像/标签配对完整（{len(stems)} 张）。")


# ------------------------------------------------------------------ 入口


def main() -> int:
    ap = argparse.ArgumentParser(description="YOLO-pose 数据集开训前校验")
    ap.add_argument("--data", type=Path, required=True, help="数据集 YAML")
    ap.add_argument("--split", default="train", help="检查哪个划分（默认 train）")
    ap.add_argument("--sample", type=int, default=0, help="只检查前 N 个标签文件（0=全部）")
    ap.add_argument("--json", type=Path, default=None, help="把统计写成 JSON")
    args = ap.parse_args()

    from tod.compat import load_yaml

    if not args.data.is_file():
        raise SystemExit(f"数据集配置不存在：{args.data}")
    cfg = load_yaml(args.data)
    report = Report()

    kpt_shape, flip_idx, _names = check_config(cfg, report)
    stats: dict = {}
    if kpt_shape[0] > 0:
        stats = check_labels(cfg, kpt_shape, flip_idx, args.split, args.sample, report)
        summarize_scale(stats, cfg, report)
        check_image_pairing(cfg, args.split, report)

    print(f"\n[check] {args.data}  (split={args.split})")
    print(f"        kpt_shape={list(kpt_shape)}  flip_idx={flip_idx}  nc={cfg.get('nc')}")
    if stats:
        per = stats.get("kpt_per_index_vis") or {}
        vis_rate = "  ".join(
            f"{k}:{(s['vis'] / s['n'] * 100 if s['n'] else 0):.0f}%" for k, s in sorted(per.items())
        )
        print(f"        目标数={stats['boxes']}  关键点数={stats['kpt_total']}  "
              f"文件={stats['files']}")
        print(f"        逐点可见率：{vis_rate}")
    for line in report.notes:
        print(f"  note  {line}")
    for line in report.warnings:
        print(f"  warn  {line}")
    for line in report.errors:
        print(f"  ERROR {line}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(
            {"data": str(args.data), "kpt_shape": list(kpt_shape), "flip_idx": flip_idx,
             "stats": {k: v for k, v in stats.items() if k != "kpt_per_index_vis"},
             "kpt_per_index_vis": {str(k): v for k, v in (stats.get("kpt_per_index_vis") or {}).items()},
             "errors": report.errors, "warnings": report.warnings, "notes": report.notes},
            ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
        print(f"[check] 统计已写入 {args.json}")

    verdict = "通过" if report.ok else "**有错误**（先修数据再开训）"
    print(f"\n[check] 结论：{verdict}；{len(report.warnings)} 条警告、{len(report.errors)} 条错误\n")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
