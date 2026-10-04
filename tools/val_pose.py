"""姿态评测入口：OKS-AP（整体 / 按目标尺度分层 / 按关键点尺度诊断）。

**为什么不用 `yolo pose val`**：框架的 ``PoseValidator`` 只给一个总体
``metrics.pose``（OKS-AP 50:95），拿不到"小目标层的关键点怎么样"。
而本库的主战场是航拍小目标 —— OKS 的分母含框面积，小目标关键点的退化会被
整体数字完全掩盖（详见 ``src/tod/eval/pose.py`` 的模块说明）。

本入口固定同时输出三组东西：

    1. 整体与**按目标边长分层**的 OKS-AP / OKS-AP50 / OKS≥0.5 命中率；
    2. 同一套预测在**自定义 sigma** 口径下的 OKS-AP（衡量"训练目标与评测目标是否对齐"）；
    3. **逐关键点尺度诊断**：每个尺度层的关键点平均误差（px）、≤1px 命中率、平均 OKS。

用法::

    # 真实权重评测（把 results.json 的 metrics 直接贴进变体卡片）
    python tools/val_pose.py \
        --variant variants/visdrone-yolo26n-pose-p2-p16/variant.yaml \
        --weights results/visdrone-yolo26n-pose-p2-p16/weights/best.pt \
        --data configs/_base_/datasets/visdrone2019-pose.yaml \
        --imgsz 1024 --sigma-strategy person \
        --json variants/visdrone-yolo26n-pose-p2-p16/results.json

    # 合成数据冒烟（几秒钟）
    python tools/val_pose.py --weights runs/x/weights/best.pt \
        --data tests/.tmp/tiny-pose/dataset.yaml --max-images 4 --device cpu

results.json 里带 git commit / 权重 mtime（PLAN §2.7）。

⚠️ 口径提醒：本入口的 OKS-AP 用**框 IoU 匹配**（默认 0.5），不是 COCO 的 OKS 匹配，
因此**不等于** COCO keypoints AP，两者不可直接比较。
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


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:  # noqa: BLE001 - 复现信息缺失不应中断评测
        return "unknown"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="姿态 OKS-AP 评测（含尺度分层）")
    ap.add_argument("--variant", type=Path, default=None, help="变体 spec（用于记录 meta）")
    ap.add_argument("--weights", type=Path, required=True, help="姿态模型权重 .pt")
    ap.add_argument("--data", type=Path, required=True, help="数据集 YAML（必须含 kpt_shape）")
    ap.add_argument("--split", default="val", help="val / test / train")
    ap.add_argument("--imgsz", type=int, default=None, help="默认取变体配置里的 imgsz")
    ap.add_argument("--conf", type=float, default=0.001, help="评测用低阈值，避免截断召回")
    ap.add_argument("--iou", type=float, default=0.7, help="NMS IoU")
    ap.add_argument("--match-iou", type=float, default=0.5,
                    help="预测与 GT 的**匹配**框 IoU 阈值（与 OKS 无关，见模块说明）")
    ap.add_argument("--sigma-strategy", default="auto",
                    choices=("person", "framework", "auto", "balanced"),
                    help="第二套 sigma 口径；一般填与 EP7 训练时一致的策略")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--max-images", type=int, default=None, help="只评测前 N 张（冒烟用）")
    ap.add_argument("--json", type=Path, default=None, help="把结果写成 JSON")
    return ap


def main() -> int:
    args = build_parser().parse_args()

    from tod.compat import load_yaml
    from tod.eval.pose import evaluate_pose

    if not args.weights.is_file():
        raise SystemExit(f"权重不存在：{args.weights}")

    spec = load_yaml(args.variant) if args.variant and args.variant.is_file() else {}
    imgsz = args.imgsz or int((spec.get("data") or {}).get("imgsz", 640))

    data_cfg = load_yaml(args.data)
    if not data_cfg.get("kpt_shape"):
        raise SystemExit(
            f"{args.data} 缺少 kpt_shape（姿态数据集必需）。"
            "参考 configs/_base_/datasets/visdrone2019-pose.yaml。"
        )

    print(f"[pose-eval] weights={args.weights} | data={args.data} | split={args.split} "
          f"| imgsz={imgsz} | conf={args.conf} | match_iou={args.match_iou} "
          f"| sigma={args.sigma_strategy}")
    started = time.time()
    result = evaluate_pose(
        args.data, weights=args.weights, split=args.split, imgsz=imgsz,
        conf=args.conf, iou=args.iou, device=args.device, max_det=args.max_det,
        match_iou=args.match_iou, sigma_strategy=args.sigma_strategy,
        max_images=args.max_images,
    )
    print(f"[pose-eval] 用时 {time.time() - started:.1f}s，"
          f"评测 {result['meta']['images']} 张图")

    payload = {
        "variant": spec.get("id") or (args.variant.stem if args.variant else None),
        "base": spec.get("base"),
        "task": spec.get("task", "pose"),
        "eps": spec.get("eps"),
        "weights": str(args.weights),
        "weights_mtime": args.weights.stat().st_mtime,
        "dataset": str(args.data),
        "split": args.split,
        "imgsz": imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "match_iou": args.match_iou,
        "sigma_strategy": args.sigma_strategy,
        "git_commit": _git_commit(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "overall": result["overall"],
        "overall_custom_sigma": result["overall_custom"],
        "metrics": result["bins"],
        "metrics_custom_sigma": result["bins_custom"],
        "keypoint_scale": result["kpt_scale"],
        "meta": result["meta"],
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                       allow_nan=True), encoding="utf-8")
        print(f"[pose-eval] 结果已写入 {args.json}")
    return 0


if __name__ == "__main__":
    from tod.compat import CompatError as _CompatError

    try:
        raise SystemExit(main())
    except _CompatError as exc:
        raise SystemExit(f"[环境错误] {exc}") from exc
