"""尺度分层评测（PLAN §7.2）：固定同时报告整体 AP 与 `AP_small` / `AP_tiny`。

**为什么不能只看 AP50 / 整体 mAP**（PLAN §1.1）：小目标变体的收益几乎全部体现在
`AP_small` 上；只报整体 mAP 会让"小目标涨点、大目标掉点"和"什么都没发生"看起来一样。
本模块按**目标边长（sqrt(area)，px）**分层，口径来自数据集配置里的
``tod.size_bins_px``（默认 ``[8, 16, 32, 96]``，与 configs/_base_/datasets/*.yaml 一致）：

    ============  ==========================  =========================
    层            COCO 口径                   本库口径（边长）
    ============  ==========================  =========================
    tiny          —                           边长 < 16px
    small         area < 32²                   边长 < 32px
    medium        32² ≤ area < 96²             32 ≤ 边长 < 96
    large         area ≥ 96²                   边长 ≥ 96
    ============  ==========================  =========================

实现要点（与 COCO 一致，便于横向对比）：
    * AP 用 101 点插值；`AP50:95` 对 IoU 0.50:0.05:0.95 求平均；
    * 每层只按**目标边长**过滤 GT；预测框**不**按尺寸过滤，但命中"本层之外 GT"的预测按
      COCO 的 **ignore 语义**忽略（既不算 TP 也不算 FP）—— 否则一个"大小目标都会检"的
      检测器会因为大目标预测在小目标层里变成 FP，被莫名压低 AP_small；
    * **有 GT 但一个预测都没有的类别计 0 分**（不能悄悄跳过，否则漏检整类反而"好看"）；
    * 每个类别先算 AP 再对类别取平均（得到该层的 mAP），因此类内分层、类间可比；
    * 低置信阈值（默认 0.001）+ 每图最多 300 框，保证召回不被后处理截断。

本模块**不依赖** ultralytics 的 validator 内部结构：预测可以来自 YOLO 模型，也可以来自
任何 ``(image_path) -> [(cls, conf, xyxy), ...]`` 的可调用对象（测试里用它注入"完美预测"）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

#: 图像后缀（与 ultralytics 一致）
IMG_SUFFIXES = (".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp")
#: 默认边长分层（px）；数据集里可用 tod.size_bins_px 覆盖
DEFAULT_BINS = (8, 16, 32, 96)
IOU_THRESHOLDS = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))


@dataclass(frozen=True)
class Bin:
    """一个尺度层：``lo <= 边长 < hi``（hi 为 None 表示无上界）。"""

    name: str
    lo: float
    hi: float | None

    def contains(self, side: float) -> bool:
        return side >= self.lo and (self.hi is None or side < self.hi)


def bins_from_edges(edges: Sequence[float] = DEFAULT_BINS) -> list[Bin]:
    """把 ``[8,16,32,96]`` 变成 5 个层（含 <8 与 ≥96 两端）。"""
    edges = sorted(float(e) for e in edges)
    out = [Bin(f"lt{edges[0]:g}", 0.0, edges[0])]
    for lo, hi in zip(edges, edges[1:]):
        out.append(Bin(f"{lo:g}-{hi:g}", lo, hi))
    out.append(Bin(f"ge{edges[-1]:g}", edges[-1], None))
    return out


def dataset_bins(data_cfg: dict) -> list[Bin]:
    """从数据集配置里读 ``tod.size_bins_px``（没有就用默认）。"""
    edges = ((data_cfg.get("tod") or {}).get("size_bins_px")) or DEFAULT_BINS
    return bins_from_edges(edges)


# ------------------------------------------------------------------ 数据读取


def resolve_split(data_cfg: dict, split: str, yaml_path: Path) -> tuple[Path, list[Path]]:
    """解析数据集 YAML 里的 split 路径，返回 ``(数据根目录, 图像列表)``。

    支持两种写法：目录（递归找图像）与 ``.txt`` 列表文件（每行一个路径）。
    """
    root = Path(str(data_cfg.get("path") or yaml_path.parent))
    if not root.is_absolute():
        root = (yaml_path.parent / root).resolve()
    value = data_cfg.get(split)
    if value is None:
        raise KeyError(f"数据集配置里没有 {split} 划分（现有键：{sorted(data_cfg)}）")
    target = Path(str(value))
    target = target if target.is_absolute() else root / target
    if target.suffix == ".txt" and target.is_file():      # 列表文件形式
        images = [Path(line.strip()) for line in target.read_text(encoding="utf-8").splitlines()
                  if line.strip()]
        return root, images
    images = sorted(p for p in target.rglob("*") if p.suffix.lower() in IMG_SUFFIXES)
    return root, images


def label_path_for(image: Path) -> Path:
    """图像路径 → 标签路径（ultralytics 约定：``/images/`` 换成 ``/labels/``）。"""
    parts = list(image.parts)
    if "images" in parts:
        parts[parts.index("images")] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image.parent.parent / "labels" / image.parent.name / f"{image.stem}.txt"


def load_labels(image: Path, img_w: int, img_h: int) -> tuple[list[float], list[int]]:
    """读一张图的 YOLO 标签 → ``(boxes_xyxy, classes)``（像素坐标）。"""
    path = label_path_for(image)
    boxes: list[float] = []
    classes: list[int] = []
    if not path.is_file():
        return boxes, classes
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        cx, cy, w, h = (float(x) for x in parts[1:5])
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h
        boxes.extend((x1, y1, x2, y2))
        classes.append(cls)
    return boxes, classes


# ------------------------------------------------------------------ 匹配 / AP


def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    """两个 xyxy 框的 IoU。"""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def average_precision(tp: list[bool], fp: list[bool], n_gt: int) -> float:
    """COCO 式 101 点插值 AP（输入需按置信度**降序**）。"""
    if n_gt == 0:
        return float("nan")
    tp_cum = fp_cum = 0
    recalls: list[float] = []
    precisions: list[float] = []
    for t, f in zip(tp, fp):
        tp_cum += int(t)
        fp_cum += int(f)
        recalls.append(tp_cum / n_gt)
        precisions.append(tp_cum / max(tp_cum + fp_cum, 1e-9))
    # 精度包络：从右往左取累计最大
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    ap = 0.0
    for k in range(101):
        r = k / 100
        p = 0.0
        for rec, prec in zip(recalls, precisions):
            if rec >= r:
                p = prec
                break
        ap += p / 101
    return ap


def match_image(preds: list[tuple[int, float, Sequence[float]]],
                gts: list[tuple[int, Sequence[float]]],
                iou_thr: float) -> list[tuple[int, float, int]]:
    """按类别做贪心匹配（单图）。

    Args:
        preds: ``[(cls, conf, xyxy), ...]``（任意顺序）。
        gts: ``[(cls, xyxy), ...]``（本层保留的 GT）。
        iou_thr: IoU 阈值。

    Returns:
        ``[(cls, conf, matched_gt_index), ...]``，按置信度降序；``matched_gt_index`` 为
        ``-1`` 表示未匹配（FP）。
    """
    order = sorted(range(len(preds)), key=lambda i: -preds[i][1])
    used: set[int] = set()
    out: list[tuple[int, float, int]] = []
    for i in order:
        pcls, conf, pbox = preds[i]
        best, best_iou = -1, iou_thr
        for j, (gcls, gbox) in enumerate(gts):
            if j in used or gcls != pcls:
                continue
            value = iou_xyxy(pbox, gbox)
            if value >= best_iou:
                best, best_iou = j, value
        if best >= 0:
            used.add(best)
        out.append((pcls, conf, best))
    return out


# ------------------------------------------------------------------ 主流程


def _default_predictor(weights: str | Path, imgsz: int, conf: float, iou: float,
                       device: Any, max_det: int, batch: int):
    """用 ultralytics YOLO 构造预测器：``(image_path) -> [(cls, conf, xyxy), ...]``。"""
    from tod.compat import ensure_runtime_env

    ensure_runtime_env()
    from ultralytics import YOLO

    model = YOLO(str(weights))

    def predict(image: Path):
        result = model.predict(str(image), imgsz=imgsz, conf=conf, iou=iou, device=device,
                               max_det=max_det, verbose=False)[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().tolist()
        cls = boxes.cls.cpu().tolist()
        scores = boxes.conf.cpu().tolist()
        return [(int(c), float(s), b) for c, s, b in zip(cls, scores, xyxy)]

    return predict, model


def evaluate(
    data_yaml: str | Path,
    *,
    weights: str | Path | None = None,
    predictor: Callable[[Path], list[tuple[int, float, Sequence[float]]]] | None = None,
    split: str = "val",
    imgsz: int = 640,
    conf: float = 0.001,
    iou: float = 0.7,
    device: Any = None,
    max_det: int = 300,
    max_images: int | None = None,
    verbose: bool = True,
) -> dict:
    """在 ``data_yaml`` 的 ``split`` 上做尺度分层评测。

    Args:
        weights: YOLO 权重路径（与 ``predictor`` 二选一）。
        predictor: 自定义预测器（测试注入用），签名 ``(image_path) -> [(cls, conf, xyxy)]``。
        max_images: 只评测前 N 张（自检/冒烟用）。

    Returns:
        ``{bins: {层名: {ap50, ap50_95, n_gt, n_pred, n_classes}}, overall: {...}, meta: {...}}``
    """
    from PIL import Image

    from tod.compat import load_yaml

    data_yaml = Path(data_yaml)
    if not data_yaml.is_file():
        raise FileNotFoundError(f"数据集配置不存在：{data_yaml}")
    data_cfg = load_yaml(data_yaml)
    bins = dataset_bins(data_cfg)
    root, images = resolve_split(data_cfg, split, data_yaml)
    if max_images:
        images = images[:max_images]
    if not images:
        raise FileNotFoundError(f"{data_yaml} 的 {split} 划分里没找到图像（root={root}）")

    if predictor is None:
        if weights is None:
            raise ValueError("需要 weights 或 predictor 之一")
        predictor, _ = _default_predictor(weights, imgsz, conf, iou, device, max_det, 1)

    # 逐图收集：预测与 GT（GT 记住边长，供分层过滤）
    collected: list[dict[str, Any]] = []
    for image in images:
        with Image.open(image) as im:
            w, h = im.size
        gbox, gcls = load_labels(image, w, h)
        gts = []
        for cls, k in zip(gcls, range(len(gcls))):
            box = gbox[4 * k:4 * k + 4]
            side = math.sqrt(max(0.0, (box[2] - box[0]) * (box[3] - box[1])))
            gts.append({"cls": cls, "box": box, "side": side})
        collected.append({"image": image, "gts": gts, "preds": list(predictor(image))})

    result: dict[str, Any] = {"bins": {}, "overall": None, "meta": {
        "data": str(data_yaml), "split": split, "images": len(collected),
        "imgsz": imgsz, "conf": conf, "iou": iou, "size_bins_px": [b.lo for b in bins[1:]],
    }}

    for bin_spec in [Bin("all", 0.0, None), *bins]:
        n_gt_per_class: dict[int, int] = {}
        #: (cls, iou_thr) → [(是否 TP, 置信度), ...]
        records: dict[tuple[int, float], list[tuple[bool, float]]] = {}
        n_pred = 0
        for item in collected:
            gts = item["gts"]
            in_bin = {i for i, g in enumerate(gts) if bin_spec.contains(g["side"])}
            for i in in_bin:
                cls = gts[i]["cls"]
                n_gt_per_class[cls] = n_gt_per_class.get(cls, 0) + 1
            pairs = [(g["cls"], g["box"]) for g in gts]
            for thr in IOU_THRESHOLDS:
                for cls, conf_v, gt_index in match_image(item["preds"], pairs, thr):
                    if thr == IOU_THRESHOLDS[0]:
                        n_pred += 1
                    if gt_index != -1 and gt_index not in in_bin:
                        # 命中**本层之外**的 GT → 忽略该预测（COCO 的 area-range ignore 语义：
                        # 既不算 TP 也不算 FP，否则"检测器同时会检大目标"会莫名拉低 AP_small）
                        continue
                    records.setdefault((cls, thr), []).append((gt_index != -1, conf_v))

        # 逐类别逐阈值算 AP；**有 GT 但一个预测都没有的类别必须计 0 分**（不能悄悄跳过）
        ap_by_class: dict[int, list[float]] = {}
        for cls, n_gt in sorted(n_gt_per_class.items()):
            if n_gt == 0:
                continue
            per_thr = []
            for thr in IOU_THRESHOLDS:
                recs = sorted(records.get((cls, thr), []), key=lambda r: -r[1])
                tp = [r[0] for r in recs]
                fp = [not r[0] for r in recs]
                per_thr.append(average_precision(tp, fp, n_gt))
            ap_by_class[cls] = per_thr

        ap50 = [vals[0] for vals in ap_by_class.values()]
        ap = [sum(vals) / len(vals) for vals in ap_by_class.values()]
        entry = {
            "n_gt": sum(n_gt_per_class.values()),
            "n_pred": n_pred,
            "n_classes": len(ap_by_class),
            "ap50": _nanmean(ap50),
            "ap50_95": _nanmean(ap),
        }
        result["bins"][bin_spec.name] = entry
        if bin_spec.name == "all":
            result["overall"] = entry

    if verbose:
        print(format_table(result))
    return result


def _nanmean(values: Iterable[float]) -> float:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def format_table(result: dict) -> str:
    """把评测结果渲染成终端表格（整体在前，各层按尺度从小到大）。"""
    order = ["all", *[b for b in result["bins"] if b != "all"]]
    lines = ["", f"{'层(边长px)':<12}{'n_gt':>7}{'n_pred':>8}{'AP50':>9}{'AP50:95':>10}"]
    lines.append("-" * 46)
    for name in order:
        entry = result["bins"][name]
        label = "整体" if name == "all" else name
        lines.append(f"{label:<12}{entry['n_gt']:>7}{entry['n_pred']:>8}"
                     f"{_fmt(entry['ap50']):>9}{_fmt(entry['ap50_95']):>10}")
    smallest = [b for b in order[1:] if b not in ("ge96",)][:2]
    lines += [
        "",
        f"注：n_pred 是该层**计入评测**的预测框数（每层都会看到全部预测框；命中本层之外 GT 的"
        f"预测按 COCO ignore 语义忽略）；n_gt 是本层的 GT 数。",
        f"提示：小目标变体必须同时看「整体」与最小两层（{' / '.join(smallest)}）——"
        "只看整体会把『小目标涨点、大目标掉点』看成什么都没发生（PLAN §7.2）。",
    ]
    return "\n".join(lines)


def _fmt(value: float) -> str:
    return "nan" if value is None or math.isnan(value) else f"{value:.4f}"


__all__ = ["Bin", "average_precision", "bins_from_edges", "dataset_bins", "evaluate",
           "format_table", "iou_xyxy", "label_path_for", "load_labels", "match_image",
           "resolve_split"]
