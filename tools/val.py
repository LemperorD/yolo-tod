"""评测入口：在指定数据集划分上做**尺度分层**评测，输出整体 AP 与 AP_small/AP_tiny。

为什么不用 `yolo val`：ultralytics 的 validator 不给按目标面积分层的指标，
而小目标变体的收益几乎全在 `AP_small` 上（PLAN §7.2）。本入口固定同时报告：

    * 整体 AP50 / AP50:95
    * 按**目标边长**分层（默认 <8 / 8–16 / 16–32 / 32–96 / ≥96 px，
      可用数据集配置里的 ``tod.size_bins_px`` 覆盖）

数据集位置与训练脚本同源：``--data`` 可以写**注册表里的名字**（``configs/datasets.yaml``）、
数据集目录或数据集 YAML —— 换盘只改注册表，不改命令行。

用法::

    # 单变体评测（并把结果写成 machine-readable 的 results.json）
    python3 tools/val.py --variant variants/SDD-YOLO26n/variant.yaml \
        --weights results/SDD-YOLO26n/weights/best.pt \
        --data visdrone2019-det \
        --imgsz 1024 --split val --json variants/SDD-YOLO26n/results.json

    # 临时换盘：命令行覆盖数据根目录
    python3 tools/val.py --weights results/x/weights/best.pt \
        --data visdrone2019-det --data-root /mnt/hdd/datasets

    # 只跑前 20 张做冒烟（合成数据集自检用）
    python3 tools/val.py --weights runs/x/weights/best.pt \
        --data tests/.tmp/tiny-detect/dataset.yaml --max-images 20 --device cpu

results.json 里带 git commit / 权重 mtime 等复现信息（PLAN §2.7 要求记录 commit hash），
也记录了数据集注册表解析出来的**绝对**根目录（换盘后仍能追溯这次评测读的是哪块盘）。
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

# Windows 控制台/重定向默认是 GBK，切到 UTF-8，免得中文与符号输出直接报错
try:
    import tod

    tod.configure_stdio()
except Exception:
    pass


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:  # noqa: BLE001 - 复现信息缺失不应中断评测
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description="尺度分层评测（AP_small / AP_tiny）")
    ap.add_argument("--variant", type=Path, default=None, help="变体 spec（仅用于记录 meta）")
    ap.add_argument("--weights", type=Path, required=True, help="模型权重 .pt")
    ap.add_argument("--data", required=True, metavar="名字|目录|YAML",
                    help="数据集：注册表里的名字 / 数据集根目录 / 数据集 YAML")
    ap.add_argument("--data-root", default=None, metavar="DIR",
                    help="覆盖注册表里的数据根目录（也可用环境变量 TOD_DATA_ROOT）")
    ap.add_argument("--datasets", type=Path, default=None, metavar="YAML",
                    help="数据集注册表 YAML（默认 configs/datasets.yaml）")
    ap.add_argument("--split", default="val", help="val / test / train")
    ap.add_argument("--imgsz", type=int, default=None, help="默认取变体配置里的 imgsz")
    ap.add_argument("--conf", type=float, default=0.001, help="评测用低阈值，避免截断召回")
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--max-images", type=int, default=None, help="只评测前 N 张（冒烟用）")
    ap.add_argument("--json", type=Path, default=None, help="把结果写成 JSON")
    args = ap.parse_args()

    from tod.compat import load_yaml
    from tod.eval.scales import evaluate
    from tod.paths import DatasetPathError, resolve

    if not args.weights.is_file():
        raise SystemExit(f"权重不存在：{args.weights}")

    try:
        paths = resolve(args.data, registry_path_hint=args.datasets,
                        data_root=args.data_root, splits=(args.split,),
                        require_train=False)
    except DatasetPathError as exc:
        raise SystemExit(f"[错误] {exc}") from exc
    if paths is None:
        raise SystemExit("--data 不能为空")
    for line in paths.report():
        print(line)

    spec = load_yaml(args.variant) if args.variant and args.variant.is_file() else {}
    imgsz = args.imgsz or int((spec.get("data") or {}).get("imgsz", 640))

    print(f"[eval] weights={args.weights} | data={paths.root} | split={args.split} "
          f"| imgsz={imgsz} | conf={args.conf}")
    started = time.time()
    result = evaluate(
        paths.yaml, weights=args.weights, split=args.split, imgsz=imgsz,
        conf=args.conf, iou=args.iou, device=args.device,
        max_det=args.max_det, max_images=args.max_images,
    )
    print(f"[eval] 用时 {time.time() - started:.1f}s，评测 {result['meta']['images']} 张图")

    payload = {
        "variant": spec.get("id") or (args.variant.stem if args.variant else None),
        "base": spec.get("base"),
        "eps": spec.get("eps"),
        "weights": str(args.weights),
        "weights_mtime": args.weights.stat().st_mtime,
        "dataset": str(args.data),
        "dataset_name": paths.name,
        "dataset_root": str(paths.root),
        "dataset_yaml": str(paths.yaml),
        "split": args.split,
        "imgsz": imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "git_commit": _git_commit(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "metrics": result["bins"],
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[eval] 结果已写入 {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
