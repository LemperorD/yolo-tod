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

    # ---- 论文的 ADD / ADD(0.5d) 所需：6D 位姿 + 相机内参 + 机体 3D 点（有则算）----
    tod = data_cfg.get("tod") or {}
    K = tod.get("camera_matrix")
    obj_pts = tod.get("object_points")
    diameter = tod.get("model_diameter")
    pose_table: dict[str, list[list[float]]] = {}
    if K and obj_pts and tod.get("pose_labels"):
        from tod.data.pose6d import Pose6DError, load_pose_csv

        pose_csv = Path(str(data_cfg.get("path") or ".")) / str(tod["pose_labels"]) / f"{split}.csv"
        if pose_csv.is_file():
            try:
                pose_table = load_pose_csv(pose_csv)
            except Pose6DError as exc:
                print(f"[pose-eval] 警告：6D 位姿不可用，跳过 ADD（{exc}）")
        else:
            print(f"[pose-eval] 提示：{pose_csv} 不存在，跳过 ADD 指标。")

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
        collected.append({"image": image, "gts": gts, "preds": list(predictor(image)),
                          "poses": pose_table.get(image.stem, [])})

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
    # ---- 论文的 ADD / ADD(0.5d)（有 6D 位姿时才算）----
    add_stats: dict[str, Any] = {"available": False}
    if pose_table and K and obj_pts:
        add_stats = _add_stats(collected, K, obj_pts, diameter)
        out["add"] = add_stats
    if verbose:
        print(format_pose_table(out))
    return out


def _add_stats(collected: list[dict], K, obj_pts, diameter) -> dict[str, Any]:
    """逐图算论文式 (13) 的 ADD（用真值深度，见 ``add_metric`` 的口径说明）。"""
    values: list[float] = []
    per_bin: dict[str, list[float]] = {}
    n_skip = 0
    bins = dataset_bins({"tod": {}})                     # 默认分层（与其它表一致）
    for item in collected:
        poses = item.get("poses") or []
        gts = item["gts"]
        for cls, _conf, gt_index, pred_index in match_pose_image(item["preds"], gts, 0.5):
            if gt_index < 0:
                continue
            if gt_index >= len(poses):
                n_skip += 1
                continue
            pred = item["preds"][pred_index]
            if len(pred) < 4 or not pred[3]:
                continue
            depths = gt_depths(poses[gt_index], obj_pts, K)
            value = add_metric(pred[3], gts[gt_index]["kpts"], depths, K)
            if math.isnan(value):
                continue
            values.append(value)
            side = gts[gt_index]["side"]
            for b in bins:
                if b.contains(side):
                    per_bin.setdefault(b.name, []).append(value)
                    break
    mean_add = sum(values) / len(values) if values else float("nan")
    threshold = (ADD_THRESHOLD_RATIO * float(diameter)) if diameter else float("nan")
    ratio_correct = (sum(1 for v in values if v < threshold) / len(values)
                     if values and diameter else float("nan"))
    return {
        "available": bool(values),
        "n": len(values),
        "n_skipped_no_pose": n_skip,
        "add_m": mean_add,
        "add_median_m": sorted(values)[len(values) // 2] if values else float("nan"),
        "threshold_m": threshold,
        "add_0.5d_ratio": ratio_correct,
        "by_bin": {k: sum(v) / len(v) for k, v in per_bin.items()},
        "note": "本库的 ADD 用真值深度（未做 PnP），是「关键点层面」的 3D 误差，"
                "与论文含 PnP 的 ADD 不可直接比较",
    }


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
    add = result.get("add")
    if add is not None:
        if add.get("available"):
            lines += [
                "",
                "论文指标（Keypoint-Guided, T-RO 2024 §IV-A-3）：",
                f"  ADD = {_fmt(add['add_m'])} m（中位数 {_fmt(add['add_median_m'])} m，"
                f"{add['n']} 个匹配目标）",
                f"  ADD(0.5d) 命中率 = {_fmt(add['add_0.5d_ratio'])}"
                f"（阈值 {_fmt(add['threshold_m'])} m = 0.5 × 机型直径）",
                f"  {add['note']}",
            ]
            if add.get("by_bin"):
                per = "  ".join(f"{k}={v:.4f}" for k, v in sorted(add["by_bin"].items()))
                lines.append(f"  按尺度层：{per}")
        else:
            skipped = add.get("n_skipped_no_pose", 0)
            lines += [
                "",
                "论文指标 ADD：本次**没有可用的匹配对**"
                f"（6D 位姿缺失跳过 {skipped} 个）。"
                "常见原因：模型还没检出任何目标（早期 epoch 正常），或 conf 阈值过高。",
            ]
    return "\n".join(lines)


def _fmt(value: float) -> str:
    return "nan" if value is None or math.isnan(value) else f"{value:.4f}"


# ------------------------------------------------------------------ 论文指标（ADD）

#: 论文（Keypoint-Guided, T-RO 2024 §IV-A-3）的 ADD 判据：ADD < 50% × 机型直径。
ADD_THRESHOLD_RATIO = 0.5


def add_metric(pred_kpts_px: Sequence[Sequence[float]],
               gt_kpts_px: Sequence[Sequence[float]],
               depth_m: float,
               K: Sequence[Sequence[float]]) -> float:
    """按论文式 (13) 的口径算 **ADD**（米）：真值 3D 关键点与预测 3D 关键点的平均距离。

    论文的 ADD 需要先把预测的 2D 关键点做 PnP 估出位姿 ``R̃, t̃``，再算
    ``mean_p ‖(R p + t) − (R̃ p + t̃)‖``。**本库不做 PnP**，改用等价而更直接的做法：

    每个关键点在相机系下的位置是 ``p_c = p_o + d·u``（``u`` 是相机射线方向，
    ``d`` 是该点的深度）。因此同一个关键点的"真值 vs 预测"之差**与深度无关**：

        ‖p_c^gt − p_c^pred‖ = d · ‖u^gt − u^pred‖

    于是 ``ADD = mean_i[ d_i · ‖u_i^gt − u_i^pred‖ ]``。
    其中 ``d_i`` 用**真值深度**（由 6D 位姿与机体 3D 点算出）—— 这也是它相对纯像素
    误差的价值所在：**同样的像素误差，远处的目标 ADD 更大**，正好符合"深度越大、
    同一像素对应的物理偏差越大"的直觉。

    ⚠️ **口径差异（引用前必读）**：论文的 ADD 含"PnP 估计位姿"这一步，本函数没有。
    所以本库的 ADD 是"关键点层面的 3D 误差"，**不是**位姿层面的 ADD，两者不可直接比较。
    变体卡片与论文数字对比时必须写清这一点。

    Args:
        pred_kpts_px: ``(n_kpt, 2)`` 预测关键点像素坐标。
        gt_kpts_px: ``(n_kpt, 2)`` 真值关键点像素坐标。
        depth_m: 各关键点的真值深度（米）；标量或长度 ``n_kpt`` 的序列。
        K: 3×3 相机内参。

    Returns:
        ADD（米）；无有效点时返回 ``nan``。
    """
    fx, fy = float(K[0][0]), float(K[1][1])
    cx, cy = float(K[0][2]), float(K[1][2])
    if not fx or not fy:
        return float("nan")
    depths = ([float(depth_m)] * len(pred_kpts_px) if isinstance(depth_m, (int, float))
              else [float(d) for d in depth_m])
    total, n = 0.0, 0
    for i, (pp, gp) in enumerate(zip(pred_kpts_px, gt_kpts_px)):
        xu, yu = (pp[0] - cx) / fx, (pp[1] - cy) / fy
        xv, yv = (gp[0] - cx) / fx, (gp[1] - cy) / fy
        d = depths[i] if i < len(depths) else depths[-1]
        total += d * math.sqrt((xu - xv) ** 2 + (yu - yv) ** 2)
        n += 1
    return total / n if n else float("nan")


def gt_depths(pose_row: Sequence[float], object_points: Sequence[Sequence[float]],
              K: Sequence[Sequence[float]]) -> list[float]:
    """由一帧 6D 位姿 ``(tx,ty,tz,qx,qy,qz,qw)`` 与机体 3D 点算出各关键点的**相机系深度**。

    这是 ``add_metric`` 需要的 ``d_i``（也是论文式 (8) 里 ``p_i^c`` 的 z 分量）。
    """
    if len(pose_row) < 7:
        raise ValueError(f"位姿需要 7 个数 (t + 四元数)，收到 {len(pose_row)}")
    R = _quat_to_rot(list(pose_row[3:7]))
    t = list(pose_row[:3])
    out = []
    for p in object_points:
        cam = [sum(R[i][j] * p[j] for j in range(3)) + t[i] for i in range(3)]
        out.append(float(cam[2]))
    return out


def _quat_to_rot(q: Sequence[float]) -> list[list[float]]:
    """四元数 ``(qx,qy,qz,qw)`` → 3×3 旋转（与 tod/data/pose6d.py 同式）。"""
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def evaluate_pose_ap(*args: Any, **kwargs: Any) -> dict:
    """``evaluate_pose`` 的稳定别名（工具脚本与测试都用这个入口名）。"""
    return evaluate_pose(*args, **kwargs)


# ------------------------------------------------------ 论文的位姿层面指标（PnP）

#: 10°10cm 判据（论文式 14/15）：旋转误差 < 10° 且平移误差 < 10 cm。
ROT_THRESHOLD_DEG = 10.0
TRANS_THRESHOLD_M = 0.10


def _numpy():
    """延迟导入 numpy。

    为什么不在模块顶部导入：本模块的 OKS/ADD/10°10cm 核心计算是纯 Python，
    只有 PnP 需要 numpy+cv2。让"没装 numpy 也能评测 OKS"成为可能
    （`tests/test_pose.py` 里的纯数值断言就靠这个）。
    """
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PnP 需要 numpy：pip install numpy") from exc
    return np


def rotation_error_deg(R_pred: Sequence[Sequence[float]],
                       R_gt: Sequence[Sequence[float]]) -> float:
    """论文式 (14)：``arccos((trace(R̃·Rᵀ) − 1)/2)``（返回**度**）。

    数值保护：``trace`` 会因浮点误差略微越界，直接 ``acos`` 会得到 nan。
    """
    a = [[float(v) for v in row] for row in R_pred]
    b = [[float(v) for v in row] for row in R_gt]
    tr = 0.0
    for i in range(3):
        for k in range(3):
            tr += a[i][k] * b[i][k]          # trace(A·Bᵀ)
    cos_t = max(-1.0, min(1.0, (tr - 1.0) / 2.0))
    return math.degrees(math.acos(cos_t))


def translation_error_m(t_pred: Sequence[float], t_gt: Sequence[float]) -> float:
    """论文式 (15)：``‖t − t̃‖₂``（米）。"""
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(t_pred, t_gt)))


def pose_to_rt(pose_row: Sequence[float]) -> tuple[list[list[float]], list[float]]:
    """``(tx,ty,tz,qx,qy,qz,qw)`` → ``(R, t)``；``t`` 单位与输入一致。"""
    if len(pose_row) < 7:
        raise ValueError(f"位姿需要 7 个数，收到 {len(pose_row)}")
    return _quat_to_rot(list(pose_row[3:7])), [float(v) for v in pose_row[:3]]


def add_pose_metric(R_pred, t_pred, object_points: Sequence[Sequence[float]],
                    R_gt, t_gt) -> float:
    """论文式 (13) 的**原始形式**：``mean_p ‖(R̃p + t̃) − (Rp + t)‖₂``（米）。

    与 ``add_metric``（关键点层面、用真值深度）不同，这里用的是 PnP 估出来的位姿，
    **与论文完全同口径**，可直接比较。
    """
    total, n = 0.0, 0
    for p in object_points:
        a = [sum(R_pred[i][j] * p[j] for j in range(3)) + t_pred[i] for i in range(3)]
        b = [sum(R_gt[i][j] * p[j] for j in range(3)) + t_gt[i] for i in range(3)]
        total += math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))
        n += 1
    return total / n if n else float("nan")


def solve_pnp(kpts_px: Sequence[Sequence[float]],
              object_points: Sequence[Sequence[float]],
              K: Sequence[Sequence[float]]):
    """``cv2.solvePnP`` → ``(R, t, ok)``。

    为什么把 ``ok`` 单独返回：PnP 会失败（点太少 / 共面退化 / 不收敛）。
    失败帧**不能**算成"误差很大"，否则会把 ADD/10°10cm 拉飞；本库的做法是单独计数
    （``n_pnp_failed``），与论文口径的差异在报告里写明。
    """
    try:
        import cv2
    except ImportError:
        return None, None, False
    np = _numpy()
    pts2d = np.asarray(kpts_px, dtype=np.float64).reshape(-1, 1, 2)
    pts3d = np.asarray(object_points, dtype=np.float64).reshape(-1, 1, 3)
    K_np = np.asarray(K, dtype=np.float64)
    try:
        ok, rvec, tvec = cv2.solvePnP(pts3d, pts2d, K_np, None,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
    except Exception:  # noqa: BLE001 - PnP 失败不应中断评测
        return None, None, False
    if not ok:
        return None, None, False
    Rm, _ = cv2.Rodrigues(rvec)
    return Rm.tolist(), [float(v) for v in np.asarray(tvec).reshape(-1)], True


def evaluate_pnp_metrics(collected: list[dict], K, object_points, diameter) -> dict:
    """对"匹配上的目标对"做 PnP，算论文式的 ADD / ADD(0.5d) / 10°10cm。"""
    adds: list[float] = []
    rots: list[float] = []
    transs: list[float] = []
    adds_hit: list[float] = []
    ten_ten: list[float] = []
    n_failed = 0
    for item in collected:
        poses = item.get("poses") or []
        gts = item["gts"]
        for _cls, _conf, gt_index, pred_index in match_pose_image(item["preds"], gts, 0.5):
            if gt_index < 0 or gt_index >= len(poses):
                continue
            pred = item["preds"][pred_index]
            if len(pred) < 4 or not pred[3]:
                continue
            R_pred, t_pred, ok = solve_pnp(pred[3], object_points, K)
            if not ok:
                n_failed += 1
                continue
            R_gt, t_gt = pose_to_rt(poses[gt_index])
            a = add_pose_metric(R_pred, t_pred, object_points, R_gt, t_gt)
            r = rotation_error_deg(R_pred, R_gt)
            tr = translation_error_m(t_pred, t_gt)
            adds.append(a)
            rots.append(r)
            transs.append(tr)
            if diameter:
                adds_hit.append(1.0 if a < 0.5 * float(diameter) else 0.0)
            ten_ten.append(1.0 if (r < ROT_THRESHOLD_DEG
                                   and tr < TRANS_THRESHOLD_M) else 0.0)

    def _mean(v: Sequence[float]) -> float:
        return sum(v) / len(v) if v else float("nan")

    return {
        "available": bool(adds),
        "n": len(adds),
        "n_pnp_failed": n_failed,
        "add_m": _mean(adds),
        "add_median_m": sorted(adds)[len(adds) // 2] if adds else float("nan"),
        "add_0.5d_ratio": _mean(adds_hit),
        "rot_err_deg_mean": _mean(rots),
        "rot_err_deg_median": sorted(rots)[len(rots) // 2] if rots else float("nan"),
        "trans_err_m_mean": _mean(transs),
        "trans_err_m_median": sorted(transs)[len(transs) // 2] if transs else float("nan"),
        "10deg10cm_ratio": _mean(ten_ten),
        "threshold_note": (f"判据：ADD < {0.5 * float(diameter):.3f} m（=0.5d）、"
                           f"旋转 < {ROT_THRESHOLD_DEG:.0f}°、平移 < "
                           f"{TRANS_THRESHOLD_M * 100:.0f} cm")
        if diameter else "缺 tod.model_diameter，ADD(0.5d) 不可用",
        "note": "与论文式 (13)/(14)/(15) **同口径**（用 PnP 估出的位姿）。",
    }


__all__ = ["ADD_THRESHOLD_RATIO", "KPX_HIT_PX", "ROT_THRESHOLD_DEG",
           "TRANS_THRESHOLD_M", "add_metric", "add_pose_metric",
           "default_pose_predictor", "evaluate_pnp_metrics", "evaluate_pose",
           "evaluate_pose_ap", "extract_pose_preds", "format_pose_table",
           "gt_depths", "keypoint_predictor", "load_pose_labels",
           "match_pose_image", "oks", "pose_to_rt", "rotation_error_deg",
           "solve_pnp", "translation_error_m"]
