"""姿态关键点评测（EP8 口径）：OKS-AP 50:95 + 按**目标尺度**与**关键点尺度**分层。

**为什么必须单独一个评测模块**（PLAN §7.2 的同一条理由，在关键点上更严重）：
    目标检测里"小目标退化"表现为 AP_small 掉点；关键点检测里还有第二层掩盖：
    **OKS 的分母含框面积**，同一个 3 px 的定位误差，在大目标上 OKS≈0.95、
    在 6 px 小人上 OKS≈0.2。于是"整体 OKS-AP 涨了"完全可能对应"小目标关键点全废"。
    只报一个总体 OKS-AP，和只看整体 mAP 一样，会把退化看成没发生。

本模块固定同时给出四组数字（全部 101 点插值 AP，口径与 ``tod.eval.scales`` 一致）：

    ==========================  ==========================================================
    指标                         含义
    ==========================  ==========================================================
    OKS-AP / OKS-AP50            COCO sigma 口径（与框架 validator / COCO 可横向对比）
    OKS-AP@自定义 sigma           本库 EP7 训练时用的 sigma 口径（应与训练目标一致）
    按**目标边长**分层             <8 / 8-16 / 16-32 / 32-96 / ≥96 px 的目标各自的 OKS-AP
    按**关键点尺度**分层           逐点统计：每个点落在哪个尺度桶里，各自的 <1e-5 命中率
    ==========================  ==========================================================

**匹配规则（刻意与 COCO 不同，必须知道）**：COCO keypoints 评测把 OKS 同时当作匹配
准则；本模块沿用 ``tod.eval.scales`` 的**按框 IoU 匹配**（阈值可调，默认 0.5），
再对匹配上的预测/GT 对计算 OKS。理由有两条：
  1. 只有一个匹配口径时，框的 AP 与关键点的 AP 可以直接对比（差多少是"关键点带来的额外
     损失"）；COCO 的 OKS 匹配会把"框对但点偏"的预测直接判为 FP，与框指标不可比；
  2. 小目标上 OKS 匹配阈值本身极难标定（见上），用它当匹配准则会让 AP 对 σ 的选择
     高度敏感 —— 那正是我们要**测量**的量，不该混进匹配里。

因此本模块报出的 OKS-AP **不等于** COCO keypoints AP，变体卡片里必须写清口径。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Sequence

from tod.eval.scales import (
    Bin,
    IOU_THRESHOLDS,
    average_precision,
    dataset_bins,
    label_path_for,
    resolve_split,
)
from tod.loss.pose import COCO_SIGMA_17, sigma_tensor

#: 单点定位误差低于该像素数视为命中（关键点尺度的诊断指标）。
KPX_HIT_PX = 1.0


# ------------------------------------------------------------------ 标签 / 预测


def load_pose_labels(image: Path, img_w: int, img_h: int,
                     kpt_shape: Sequence[int] = (17, 3)
                     ) -> tuple[list[float], list[int], list[list[list[float]]]]:
    """读一张 YOLO-pose 标签 → ``(boxes_xyxy, classes, keypoints)`。

    Args:
        image: 图像路径（标签路径按 ultralytics 约定由 ``/images/`` → ``/labels/`` 推出）。
        img_w, img_h: 图像尺寸（关键点坐标是归一化的，需要还原到像素）。
        kpt_shape: ``(n_kpt, n_dim)``；``n_dim=3`` 时第三维是可见性。

    Returns:
        ``(boxes, classes, kpts)``：``boxes`` 是扁平像素 xyxy，``classes`` 与之等长，
        ``kpts[i]`` 是第 i 个目标的 ``[[x, y, vis], ...]``（像素坐标）。
        第三维缺失（``n_dim=2``）时 ``vis`` 记为 1。
    """
    n_kpt, n_dim = int(kpt_shape[0]), int(kpt_shape[1])
    path = label_path_for(image)
    boxes: list[float] = []
    classes: list[int] = []
    kpts: list[list[list[float]]] = []
    if not path.is_file():
        return boxes, classes, kpts
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        cx, cy, w, h = (float(x) for x in parts[1:5])
        boxes.extend(((cx - w / 2) * img_w, (cy - h / 2) * img_h,
                      (cx + w / 2) * img_w, (cy + h / 2) * img_h))
        classes.append(cls)
        values = [float(x) for x in parts[5:]]
        row: list[list[float]] = []
        for k in range(n_kpt):
            chunk = values[k * n_dim:(k + 1) * n_dim]
            if len(chunk) < 2:
                row.append([0.0, 0.0, 0.0])
                continue
            x = chunk[0] * img_w
            y = chunk[1] * img_h
            if n_dim >= 3 and len(chunk) >= 3:
                # vis 保持原始值（0=未标注，1/2=可见），与训练侧掩码规则一致
                row.append([x, y, float(chunk[2])])
            else:
                row.append([x, y, 1.0])
        kpts.append(row)
    return boxes, classes, kpts


def oks(pred_kpts: Sequence[Sequence[float]], gt_kpts: Sequence[Sequence[float]],
        box_xyxy: Sequence[float], sigmas: Sequence[float]) -> float:
    """Object Keypoint Similarity（COCO 口径，逐点可见性掩码）。

        OKS = Σ_k exp(−d_k² / (2·s²·σ_k²)) · δ(v_k>0) / Σ_k δ(v_k>0)

    其中 ``s²`` 用 GT **框面积**（px²，COCO 为保持稳定取 max(area, 1)），
    ``d_k`` 是预测点与 GT 点的欧氏距离（px）。返回 ``nan`` 表示该目标没有任何
    有效关键点（COCO 里这种目标对 AP 无贡献，本模块按 nan 剔除）。

    本函数与 ``tod.loss.pose.TinyPoseLoss`` 的损失式**同源**：损失是
    ``1 − exp(−e)``（归一化坐标 + 平方距离），本函数是指标形式（像素坐标 + 面积）。
    一一对应关系见 variants/visdrone-yolo26n-pose-p2-p16/paper-notes.md。
    """
    area = max(1.0, (box_xyxy[2] - box_xyxy[0]) * (box_xyxy[3] - box_xyxy[1]))
    total = 0.0
    n_valid = 0
    for k, point in enumerate(pred_kpts):
        if k >= len(gt_kpts):
            break
        px, py = float(point[0]), float(point[1])
        gx, gy = gt_kpts[k][0], gt_kpts[k][1]
        vis = gt_kpts[k][2] if len(gt_kpts[k]) > 2 else 1.0
        if vis <= 0:                       # 未标注的点不参与（与训练侧掩码一致）
            continue
        sigma = float(sigmas[k]) if k < len(sigmas) else float(sigmas[-1])
        d2 = (px - gx) ** 2 + (py - gy) ** 2
        total += math.exp(-d2 / (2.0 * area * sigma ** 2 + 1e-9))
        n_valid += 1
    return total / n_valid if n_valid else float("nan")


def keypoint_predictor(model: Any, imgsz: int, conf: float, iou: float,
                       device: Any = None, max_det: int = 300):
    """把 ultralytics 姿态模型包成 ``(image_path) -> [(cls, conf, xyxy, kpts)]``。

    ``kpts`` 是**像素坐标**的 ``[[x, y], ...]``（可视化与 OKS 都用它）。
    """
    def predict(image: Path):
        result = model.predict(str(image), imgsz=imgsz, conf=conf, iou=iou,
                               device=device, max_det=max_det, verbose=False)[0]
        return extract_pose_preds(result)

    return predict


def match_pose_image(preds: list[tuple], gts: list[dict], iou_thr: float
                     ) -> list[tuple[int, float, int, int]]:
    """姿态用的按类贪心匹配（单图）—— 与 ``scales.match_image`` 同一套规则。

    **为什么不复用 ``scales.match_image``**：它只吃 ``(cls, conf, xyxy)`` 三元组，
    而姿态预测是四元组（多一个关键点），解包会直接
    ``ValueError: too many values to unpack``。更重要的是：本函数**直接返回预测的下标**，
    调用方不必再靠"类别 + 置信度"去反查预测（置信度可能重复，反查是脆的）。

    Args:
        preds: ``[(cls, conf, xyxy, kpts), ...]``。
        gts: ``[{"cls":…, "box":…, "side":…, "kpts":…}, ...]``。
        iou_thr: 匹配用的框 IoU 阈值（**不是** OKS 阈值，见模块 docstring）。

    Returns:
        ``[(cls, conf, gt_index, pred_index), ...]``（按置信度降序）；
        ``gt_index == -1`` 表示未匹配（FP）。
    """
    from tod.eval.scales import iou_xyxy

    order = sorted(range(len(preds)), key=lambda i: -preds[i][1])
    used: set[int] = set()
    out: list[tuple[int, float, int, int]] = []
    for i in order:
        pcls, conf, pbox = preds[i][0], preds[i][1], preds[i][2]
        best, best_iou = -1, iou_thr
        for j, gt in enumerate(gts):
            if j in used or gt["cls"] != pcls:
                continue
            value = iou_xyxy(pbox, gt["box"])
            if value >= best_iou:
                best, best_iou = j, value
        if best >= 0:
            used.add(best)
        out.append((pcls, conf, best, i))
    return out


def extract_pose_preds(result: Any) -> list[tuple[int, float, list[float], list[list[float]]]]:
    """从 ultralytics 的 ``Results`` 取出 ``[(cls, conf, xyxy, kpts_px)]``。

    关键点是 Results 里的公共契约（``result.keypoints.xy``），比读原始输出张量的
    形状更稳（Pose / Pose26 / NMS-free 三种后处理路径的张量布局并不相同）。
    """
    boxes = getattr(result, "boxes", None)
    kpts = getattr(result, "keypoints", None)
    if boxes is None or len(boxes) == 0 or kpts is None:
        return []
    xyxy = boxes.xyxy.cpu().tolist()
    cls = boxes.cls.cpu().tolist()
    scores = boxes.conf.cpu().tolist()
    points = kpts.xy.cpu().tolist()          # (n, K, 2) 已是原图像素坐标
    return [(int(c), float(s), [float(v) for v in b], [[float(x), float(y)] for x, y in k])
            for c, s, b, k in zip(cls, scores, xyxy, points)]


def default_pose_predictor(weights: str | Path, imgsz: int, conf: float, iou: float,
                           device: Any, max_det: int):
    """用 ultralytics 构造姿态预测器（评测工具的默认路径）。"""
    from tod.compat import ensure_runtime_env

    ensure_runtime_env()
    from ultralytics import YOLO

    model = YOLO(str(weights), task="pose")
    return keypoint_predictor(model, imgsz, conf, iou, device, max_det), model


# ------------------------------------------------------------------ 主流程


def evaluate_pose(
    data_yaml: str | Path,
    *,
    weights: str | Path | None = None,
    predictor: Callable[[Path], list[tuple]] | None = None,
    split: str = "val",
    imgsz: int = 640,
    conf: float = 0.001,
    iou: float = 0.7,
    device: Any = None,
    max_det: int = 300,
    match_iou: float = 0.5,
    sigma_strategy: str = "auto",
    max_images: int | None = None,
    verbose: bool = True,
) -> dict:
    """在 ``data_yaml`` 的 ``split`` 上做姿态分层评测。

    Args:
        match_iou: **匹配**用的框 IoU 阈值（与 OKS 无关，见模块 docstring 的匹配规则）。
        sigma_strategy: 第二套 sigma 口径（``person``/``auto``/``balanced``），
            一般填与 EP7 训练时一致的策略，用来衡量"训练目标与评测目标是否对齐"。
        max_images: 只评测前 N 张（自检/冒烟用）。

    Returns:
        ``{bins, overall, bins_custom, overall_custom, kpt_scale, meta}``，
        每项含 ``oks_ap`` / ``oks_ap50`` / ``oks50``（OKS≥0.5 的命中率）/ ``n_gt`` / ``n_pred``。
    """
    from PIL import Image

    from tod.compat import load_yaml

    data_yaml = Path(data_yaml)
    if not data_yaml.is_file():
        raise FileNotFoundError(f"数据集配置不存在：{data_yaml}")
    data_cfg = load_yaml(data_yaml)
    kpt_shape = tuple(data_cfg.get("kpt_shape") or (17, 3))
    bins = dataset_bins(data_cfg)
    root, images = resolve_split(data_cfg, split, data_yaml)
    if max_images:
        images = images[:max_images]
    if not images:
        raise FileNotFoundError(f"{data_yaml} 的 {split} 划分里没找到图像（root={root}）")

    if predictor is None:
        if weights is None:
            raise ValueError("需要 weights 或 predictor 之一")
        predictor, _ = default_pose_predictor(weights, imgsz, conf, iou, device, max_det)

    n_kpt = int(kpt_shape[0])
    if n_kpt == len(COCO_SIGMA_17):
        sigmas = list(COCO_SIGMA_17)
        custom, custom_note = sigma_tensor(kpt_shape, strategy=sigma_strategy)
        custom = [float(v) for v in custom.tolist()]
    else:
        custom, custom_note = sigma_tensor(kpt_shape, strategy=sigma_strategy)
        custom = [float(v) for v in custom.tolist()]
        sigmas = list(custom)                 # 非 17 点没有 COCO 标准值，两套口径合一
        custom_note += "（非 17 点：COCO 口径不可用，两套指标相同）"

    collected: list[dict[str, Any]] = []
    for image in images:
        with Image.open(image) as im:
            w, h = im.size
        gbox, gcls, gkpt = load_pose_labels(image, w, h, kpt_shape)
        gts = []
        for i, cls in enumerate(gcls):
            box = gbox[4 * i:4 * i + 4]
            side = math.sqrt(max(0.0, (box[2] - box[0]) * (box[3] - box[1])))
            gts.append({"cls": cls, "box": box, "side": side, "kpts": gkpt[i]})
        collected.append({"image": image, "gts": gts, "preds": list(predictor(image))})

    # ---- 主循环：整体 + 各尺度层，两套 sigma 口径 ----
    bins_std: dict[str, dict[str, float]] = {}
    bins_custom: dict[str, dict[str, float]] = {}
    same_sigma = list(custom) == list(sigmas)
    for bin_spec in [Bin("all", 0.0, None), *bins]:
        bins_std[bin_spec.name] = _accumulate(collected, bin_spec, sigmas, match_iou)
        bins_custom[bin_spec.name] = (bins_std[bin_spec.name] if same_sigma
                                      else _accumulate(collected, bin_spec, custom, match_iou))

    # ---- 关键点尺度诊断（只统计"匹配上的目标对"的逐点误差）----
    kpt_scale: dict[str, dict[str, float]] = {
        b.name: {"n": 0.0, "hit": 0.0, "err_px_sum": 0.0, "oks_sum": 0.0, "oks_n": 0.0}
        for b in bins
    }
    for item in collected:
        gts = item["gts"]
        for _cls, _conf, gt_index, pred_index in match_pose_image(item["preds"], gts, match_iou):
            if gt_index < 0:
                continue
            gt = gts[gt_index]
            pred = item["preds"][pred_index]
            if len(pred) < 4 or not pred[3]:
                continue
            bucket = next((b.name for b in bins if b.contains(gt["side"])), bins[-1].name)
            stat = kpt_scale[bucket]
            for k in range(min(len(gt["kpts"]), len(pred[3]))):
                gx, gy, vis = gt["kpts"][k]
                if vis <= 0:
                    continue
                px, py = pred[3][k][0], pred[3][k][1]
                err = math.hypot(px - gx, py - gy)
                stat["n"] += 1
                stat["err_px_sum"] += err
                stat["hit"] += 1.0 if err <= KPX_HIT_PX else 0.0
                stat["oks_sum"] += oks(pred[3], gt["kpts"], gt["box"], sigmas)
                stat["oks_n"] += 1

    out = {
        "bins": bins_std,
        "bins_custom": bins_custom,
        "overall": bins_std["all"],
        "overall_custom": bins_custom["all"],
        "kpt_scale": {k: _finalize_kpt(v) for k, v in kpt_scale.items()},
        "meta": {
            "data": str(data_yaml), "split": split, "images": len(collected),
            "imgsz": imgsz, "conf": conf, "iou": iou, "match_iou": match_iou,
            "kpt_shape": list(kpt_shape), "sigma_strategy": sigma_strategy,
            "sigma_note": custom_note, "kpt_hit_px": KPX_HIT_PX,
            "size_bins_px": [b.lo for b in bins[1:]],
            "note": "OKS-AP 用框 IoU 匹配（非 COCO 的 OKS 匹配），与 COCO keypoints AP 不可直接比较",
        },
    }
    if verbose:
        print(format_pose_table(out))
    return out


def _accumulate(collected: list[dict], bin_spec: Bin, sigmas: Sequence[float],
                match_iou: float) -> dict[str, float]:
    """在一个尺度层内累计 OKS-AP（按类别算 AP 再取平均，与 scales.py 同规则）。"""
    n_gt_per_class: dict[int, int] = {}
    #: cls → [(oks_value, conf), ...]（按 conf 降序后，以 oks>=thr 判 TP）
    records: dict[int, list[tuple[float, float]]] = {}
    n_pred = 0

    for item in collected:
        gts = item["gts"]
        in_bin = {i for i, g in enumerate(gts) if bin_spec.contains(g["side"])}
        for i in in_bin:
            cls = gts[i]["cls"]
            n_gt_per_class[cls] = n_gt_per_class.get(cls, 0) + 1
        for cls, conf_v, gt_index, pred_index in match_pose_image(item["preds"], gts, match_iou):
            n_pred += 1
            if gt_index != -1 and gt_index not in in_bin:
                continue                     # 命中本层之外的 GT → ignore（COCO 语义）
            if gt_index < 0:
                records.setdefault(cls, []).append((0.0, conf_v))
                continue
            pred = item["preds"][pred_index]
            if len(pred) < 4 or not pred[3]:
                records.setdefault(cls, []).append((0.0, conf_v))
                continue
            value = oks(pred[3], gts[gt_index]["kpts"], gts[gt_index]["box"], sigmas)
            records.setdefault(cls, []).append((0.0 if math.isnan(value) else value, conf_v))

    oks_ap: list[float] = []
    oks_ap50: list[float] = []
    oks50_pairs: list[tuple[bool, float]] = []
    for cls, n_gt in sorted(n_gt_per_class.items()):
        if n_gt == 0:
            continue
        recs = sorted(records.get(cls, []), key=lambda r: -r[1])
        per_thr = []
        for thr in IOU_THRESHOLDS:           # 复用同一组阈值：0.50, 0.55, …, 0.95
            tp = [r[0] >= thr for r in recs]
            fp = [not t for t in tp]
            per_thr.append(average_precision(tp, fp, n_gt))
        oks_ap.append(sum(per_thr) / len(per_thr))
        oks_ap50.append(per_thr[0])
        oks50_pairs += [(r[0] >= 0.5, r[1]) for r in recs]

    return {
        "n_gt": float(sum(n_gt_per_class.values())),
        "n_pred": float(n_pred),
        "n_classes": float(len(oks_ap)),
        "oks_ap50": _nanmean(oks_ap50),
        "oks_ap": _nanmean(oks_ap),
        "oks50_rate": (sum(1 for t, _ in oks50_pairs if t) / len(oks50_pairs)
                       if oks50_pairs else float("nan")),
    }


def _finalize_kpt(stat: dict[str, float]) -> dict[str, float]:
    n = stat["n"]
    return {
        "n_kpt": n,
        "hit_rate@1px": stat["hit"] / n if n else float("nan"),
        "mean_err_px": stat["err_px_sum"] / n if n else float("nan"),
        "mean_oks": stat["oks_sum"] / stat["oks_n"] if stat["oks_n"] else float("nan"),
    }


def _nanmean(values: Sequence[float]) -> float:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def format_pose_table(result: dict) -> str:
    """渲染终端表格：整体 + 各尺度层的 OKS-AP（含自定义 sigma 口径）。"""
    order = ["all", *[b for b in result["bins"] if b != "all"]]
    lines = ["", f"{'层(边长px)':<12}{'n_gt':>7}{'n_pred':>8}{'OKS-AP':>10}{'OKS-AP50':>10}"
                  f"{'OKS≥.5':>10}{'自定义σ':>10}"]
    lines.append("-" * 68)
    for name in order:
        entry = result["bins"][name]
        custom = result["bins_custom"][name]
        label = "整体" if name == "all" else name
        lines.append(f"{label:<12}{int(entry['n_gt']):>7}{int(entry['n_pred']):>8}"
                     f"{_fmt(entry['oks_ap']):>10}{_fmt(entry['oks_ap50']):>10}"
                     f"{_fmt(entry['oks50_rate']):>10}{_fmt(custom['oks_ap']):>10}")
    lines += ["", "关键点尺度诊断（只统计匹配上的目标对，逐点）：",
              f"{'层(边长px)':<12}{'n_kpt':>8}{'命中≤1px':>11}{'平均误差px':>12}{'平均OKS':>10}"]
    for name, stat in result["kpt_scale"].items():
        lines.append(f"{name:<12}{int(stat['n_kpt']):>8}{_fmt(stat['hit_rate@1px']):>11}"
                     f"{_fmt(stat['mean_err_px']):>12}{_fmt(stat['mean_oks']):>10}")
    lines += [
        "",
        f"口径：匹配用框 IoU（阈值 {result['meta']['match_iou']}），OKS 用 COCO sigma；"
        f"「自定义σ」列是 sigma_strategy={result['meta']['sigma_strategy']} 的同一套预测"
        f"（{result['meta']['sigma_note']}）。",
        "注意：本表 OKS-AP **不等于** COCO keypoints AP（COCO 用 OKS 当匹配准则），"
        "两者不可直接比较；同一张表内部的对比才是有效的。",
        "提示：小目标关键点必须看 lt8 / 8-16 两层与「关键点尺度诊断」——"
        "整体 OKS-AP 涨而小目标层掉，是 OKS 分母面积小导致的典型假象。",
    ]
    return "\n".join(lines)


def _fmt(value: float) -> str:
    return "nan" if value is None or math.isnan(value) else f"{value:.4f}"


def evaluate_pose_ap(*args: Any, **kwargs: Any) -> dict:
    """``evaluate_pose`` 的稳定别名（工具脚本与测试都用这个入口名）。"""
    return evaluate_pose(*args, **kwargs)


__all__ = ["KPX_HIT_PX", "default_pose_predictor", "evaluate_pose", "evaluate_pose_ap",
           "extract_pose_preds", "format_pose_table", "keypoint_predictor",
           "load_pose_labels", "match_pose_image", "oks"]
