"""MAV6D 测试入口：小目标检测 + 9 点关键点的完整评测（含论文全部指标）。

## 报什么（一个脚本给全）

| 指标 | 口径 | 与论文关系 |
|---|---|---|
| **AP / AP50**（框） | 按 GT 框边长分层（<8 / 8-16 / 16-32 / 32-96 / ≥96 px） | 论文的 AP（框由关键点外接得到） |
| **OKS-AP 50:95** | 本库口径（框 IoU 匹配）+ 按目标尺度分层 + 逐关键点诊断 | 论文没有这项（本库新增，对尺度更鲁棒） |
| **ADD / ADD(0.5d)** | ① **PnP 位姿层面**（论文式 13，同口径）② 关键点层面（用真值深度，补充诊断） | ✅ 与论文可比 |
| **10°10cm** | 论文式 (14)/(15) + PnP | ✅ 与论文可比 |
| 关键点像素误差 | 平均误差 / ≤1px 命中率，按尺度分层 | 论文没有（诊断用） |

## 用法

    # 单机型评测（自动找 results/mav6d-<机型>/weights/best.pt）
    python tools/test_mav6d.py --model phantom4 --data data/mav6d/phantom4/dataset.yaml

    # 指定权重 / 划分 / 输出 JSON
    python tools/test_mav6d.py --model phantom4 --weights results/xxx/weights/best.pt `
        --split val --json results/mav6d-phantom4/test.json

    # 两个机型一起报（论文 Table I/IV 就是分机型报的）
    python tools/test_mav6d.py --model both --data-root data/mav6d

    # 只看前 200 张（冒烟）
    python tools/test_mav6d.py --model phantom4 --max-images 200

## 重要口径说明（写进 JSON 的 meta，引用数字前必读）

* `ADD` 与 `10°10cm` 用 **PnP** 估位姿，与论文**同口径**；PnP 失败的帧**单独计数**、
  不计入误差（否则会把平均值拉飞）。论文未说明失败帧如何处理，这是本库的明确选择。
* `OKS-AP` 用**框 IoU 匹配**（不是 COCO 的 OKS 匹配），**不等于** COCO keypoints AP。
* 关键点标签在转换时**已去畸变**（见 `tools/mav6d_to_yolo.py`），因此相机内参用
  针孔模型，不要再套作者 `util.py` 的畸变系数。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MODELS = ("phantom4", "mavic2")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="MAV6D 测试（小目标检测 + 9 点关键点；含论文 ADD/ADD(0.5d)/10°10cm）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--model", choices=(*MODELS, "both"), default="phantom4")
    ap.add_argument("--weights", type=Path, default=None, help="权重 .pt（默认自动找）")
    ap.add_argument("--data", type=Path, default=None, help="数据集 YAML")
    ap.add_argument("--data-root", type=Path, default=ROOT / "data" / "mav6d",
                    help="--model both 时用它的 <机型>/dataset.yaml")
    ap.add_argument("--results-root", type=Path, default=ROOT / "results")
    ap.add_argument("--split", default="val")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--device", default=None)
    ap.add_argument("--conf", type=float, default=0.001, help="低阈值，避免截断召回")
    ap.add_argument("--iou", type=float, default=0.7, help="NMS IoU")
    ap.add_argument("--match-iou", type=float, default=0.5, help="与 GT 匹配的框 IoU")
    ap.add_argument("--sigma-strategy", default="uniform",
                    choices=("uniform", "auto", "person", "framework", "balanced"))
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--all-splits", action="store_true",
                    help="val 与 test 都测（test 用作者原始划分时需要单独转换）")
    return ap


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def find_weights(model: str, results_root: Path, explicit: Path | None) -> Path:
    """找权重：显式给的优先，否则在 ``results/**`` 里找该机型的最新 best.pt。"""
    if explicit:
        if not explicit.is_file():
            raise SystemExit(f"权重不存在：{explicit}")
        return explicit
    candidates = sorted(results_root.rglob("best.pt"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    hit = [p for p in candidates if model in str(p)]
    if not hit:
        raise SystemExit(
            f"没找到 {model} 的权重（在 {results_root} 下）。\n"
            f"请先训练：python tools/train_mav6d.py --model {model} "
            "--data data/mav6d/%s/dataset.yaml" % model
        )
    return hit[0]


def evaluate_one(model: str, args) -> dict:
    """对一个机型完整评测，返回可写 JSON 的 dict。"""
    from tod.compat import load_yaml
    from tod.data.pose6d import load_pose_csv
    from tod.eval.pose import (evaluate_pnp_metrics, evaluate_pose,
                               format_pose_table, load_pose_labels,
                               match_pose_image)

    data = args.data or (args.data_root / model / "dataset.yaml")
    if not data.is_file():
        raise SystemExit(
            f"数据集不存在：{data}\n"
            f"请先转换：python tools/mav6d_to_yolo.py --mav6d D:/MAV6D --model {model} "
            "--out data/mav6d"
        )
    weights = find_weights(model, args.results_root, args.weights)
    cfg = load_yaml(data)
    tod = cfg.get("tod") or {}
    K = tod.get("camera_matrix")
    obj_pts = tod.get("object_points")
    diameter = tod.get("model_diameter")

    print(f"\n{'=' * 78}\n[{model}] weights={weights}\n[{model}] data={data} "
          f"split={args.split} imgsz={args.imgsz}\n{'=' * 78}")

    started = time.time()
    result = evaluate_pose(
        data, weights=weights, split=args.split, imgsz=args.imgsz,
        conf=args.conf, iou=args.iou, device=args.device,
        match_iou=args.match_iou, sigma_strategy=args.sigma_strategy,
        max_images=args.max_images, verbose=True,
    )
    # ---- 框的尺度分层 AP（与 OKS 用同一套尺度层，便于横向对比）----
    from tod.eval.scales import evaluate as evaluate_boxes, format_table

    box_result = evaluate_boxes(
        data, weights=weights, split=args.split, imgsz=args.imgsz,
        conf=args.conf, iou=args.iou, device=args.device,
        max_images=args.max_images, verbose=False,
    )
    print("\n框 AP（尺度分层；与上面 OKS 表用同一套层，可直接横向对比）")
    print(format_table(box_result))
    elapsed = time.time() - started

    # ---- 论文的位姿层面指标（PnP）：需要重新收集一遍匹配对 ----
    pnp = None
    if K and obj_pts:
        collected = _collect_for_pnp(data, weights, args, cfg)
        pnp = evaluate_pnp_metrics(collected, K, obj_pts, diameter)
        _print_pnp(pnp)

    payload = {
        "model": model,
        "weights": str(weights),
        "weights_mtime": weights.stat().st_mtime,
        "data": str(data),
        "split": args.split,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "match_iou": args.match_iou,
        "sigma_strategy": args.sigma_strategy,
        "git_commit": _git_commit(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_s": round(elapsed, 1),
        "oks": {"overall": result["overall"], "bins": result["bins"],
                "kpt_scale": result["kpt_scale"], "meta": result["meta"]},
        "box_ap": {"overall": box_result["overall"], "bins": box_result["bins"],
                   "meta": box_result["meta"]},
        "add_keypoint_level": result.get("add"),
        "pnp": pnp,
        "dataset_tod": {k: v for k, v in tod.items() if k != "object_points"},
    }
    return payload


def _collect_for_pnp(data: Path, weights: Path, args, cfg: dict) -> list[dict]:
    """重新跑一遍推理收集"匹配对"，供 PnP 指标使用（与 evaluate_pose 的匹配规则一致）。"""
    from PIL import Image

    from tod.eval.pose import (default_pose_predictor, load_pose_labels,
                               match_pose_image)
    from tod.eval.scales import resolve_split

    predictor, _model = default_pose_predictor(weights, args.imgsz, args.conf,
                                               args.iou, args.device, 300)
    _, images = resolve_split(cfg, args.split, data)
    if args.max_images:
        images = images[:args.max_images]
    kpt_shape = tuple(cfg.get("kpt_shape") or (9, 3))
    tod = cfg.get("tod") or {}
    pose_table: dict = {}
    if tod.get("pose_labels"):
        from tod.data.pose6d import load_pose_csv

        csv_path = Path(str(cfg.get("path") or ".")) / str(tod["pose_labels"]) / \
            f"{args.split}.csv"
        if csv_path.is_file():
            pose_table = load_pose_csv(csv_path)

    out = []
    for image in images:
        with Image.open(image) as im:
            w, h = im.size
        gbox, gcls, gkpt = load_pose_labels(image, w, h, kpt_shape)
        gts = []
        for i, cls in enumerate(gcls):
            box = gbox[4 * i:4 * i + 4]
            side = ((max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])) ** 0.5)
            gts.append({"cls": cls, "box": box, "side": side, "kpts": gkpt[i]})
        out.append({"image": image, "gts": gts, "preds": list(predictor(image)),
                    "poses": pose_table.get(image.stem, [])})
    return out


def _print_pnp(pnp: dict) -> None:
    if not pnp.get("available"):
        print(f"\n论文位姿指标：不可用（匹配对 {pnp.get('n', 0)}，"
              f"PnP 失败 {pnp.get('n_pnp_failed', 0)}）")
        return
    print("\n" + "-" * 78)
    print("论文指标（PnP 位姿层面，与论文式 13/14/15 同口径）")
    print("-" * 78)
    print(f"  匹配目标数 {pnp['n']}（PnP 失败 {pnp['n_pnp_failed']}）")
    print(f"  ADD          = {pnp['add_m']:.4f} m（中位 {pnp['add_median_m']:.4f} m）")
    print(f"  ADD(0.5d)    = {pnp['add_0.5d_ratio']:.4f}")
    print(f"  旋转误差     = {pnp['rot_err_deg_mean']:.3f}°（中位 "
          f"{pnp['rot_err_deg_median']:.3f}°）")
    print(f"  平移误差     = {pnp['trans_err_m_mean']:.4f} m（中位 "
          f"{pnp['trans_err_m_median']:.4f} m）")
    print(f"  10°10cm      = {pnp['10deg10cm_ratio']:.4f}")
    print(f"  {pnp['threshold_note']}")


def main() -> int:
    args = build_parser().parse_args()
    models = list(MODELS) if args.model == "both" else [args.model]
    if args.model == "both" and args.weights:
        raise SystemExit("--model both 与 --weights 不能同时用（每机型各需一个权重）")

    reports = {}
    failures = []
    for model in models:
        try:
            reports[model] = evaluate_one(model, args)
        except SystemExit as exc:
            failures.append(f"[{model}] {exc}")
        except Exception as exc:  # noqa: BLE001 - 一个机型失败不应吞掉另一个
            import traceback
            failures.append(f"[{model}] {type(exc).__name__}: {exc}\n"
                            f"{traceback.format_exc()}")

    # ---- 汇总表 ----
    if reports:
        print("\n" + "=" * 78)
        print("汇总（分机型；论文 Table I/IV 也是分机型报的）")
        print("=" * 78)
        head = (f"{'机型':<10}{'OKS-AP':>9}{'OKS-AP50':>10}{'AP50:95(框)':>12}"
                f"{'ADD(m)':>9}{'ADD(.5d)':>10}{'10°10cm':>9}")
        print(head)
        print("-" * len(head))
        for model, rep in reports.items():
            o = rep["oks"]["overall"]
            pnp = rep.get("pnp") or {}
            box = (rep.get("box_ap") or {}).get("overall") or {}
            print(f"{model:<10}{o['oks_ap']:>9.4f}{o['oks_ap50']:>10.4f}"
                  f"{box.get('ap50_95', float('nan')):>12.4f}"
                  f"{pnp.get('add_m', float('nan')):>9.4f}"
                  f"{pnp.get('add_0.5d_ratio', float('nan')):>10.4f}"
                  f"{pnp.get('10deg10cm_ratio', float('nan')):>9.4f}")
        print("\n注意：OKS-AP 用框 IoU 匹配，**不等于** COCO keypoints AP；")
        print("      ADD/10°10cm 用 PnP，与论文同口径。关键点标签已去畸变。")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(reports, ensure_ascii=False, indent=2,
                                        allow_nan=True), encoding="utf-8")
        print(f"\n[json] 已写入 {args.json}")

    for f in failures:
        print("\n" + f)
    return 1 if failures else 0


if __name__ == "__main__":
    from tod.compat import CompatError as _CompatError

    try:
        raise SystemExit(main())
    except _CompatError as exc:
        raise SystemExit(f"[环境错误] {exc}") from exc
