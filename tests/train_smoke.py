"""训练回路自检（opt-in）：合成数据集 → 真训练 1–2 epoch → 载权重推理 → 查证据。"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import tod

    tod.configure_stdio()
except Exception:
    pass

PASSED: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    if not cond:
        raise AssertionError(f"[FAIL] {label} {detail}")
    PASSED.append(label)


def _load_tool(name: str):
    """从 tools/ 目录加载脚本（tools 不是包）。"""
    path = ROOT / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_tod_tool_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    ap = argparse.ArgumentParser(description="训练回路自检（合成数据）")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--device", default="0")
    ap.add_argument("--variant", type=Path, default=ROOT / "variants" / "SDD-YOLO26n" / "variant.yaml")
    ap.add_argument("--work", type=Path, default=ROOT / "tests" / ".tmp" / "train-smoke")
    ap.add_argument("--keep", action="store_true", help="保留实验输出目录")
    ap.add_argument("--allow-threaded-cache", action="store_true",
                    help="保留框架默认的并行标签缓存（受限沙箱下会 WinError 5）")
    args = ap.parse_args()

    for pkg in ("torch", "ultralytics", "PIL"):
        if importlib.util.find_spec(pkg) is None:
            print(f"[skip] 未安装 {pkg}")
            return 0

    import tod
    from tod import compat, runtime
    from tod.compat import load_yaml
    from tod.compose import Variant
    from tod.engine.trainer import TODDetectionTrainer

    tod.bootstrap()

    if not args.allow_threaded_cache and compat.allow_threadless_label_cache():
        print("[env ] 标签缓存已切换为顺序模式（沙箱兼容；--allow-threaded-cache 可关闭）")

    data_yaml = _load_tool("make_dummy_dataset").build(args.work / "data", 12, 4, seed=0)
    check("合成数据集配置已生成", data_yaml.is_file())

    spec = load_yaml(args.variant)
    variant = Variant.from_spec(spec)
    model_yaml = args.work / "model.yaml"
    variant.model_yaml(model_yaml, write=True)
    runtime.set_active(spec)

    from ultralytics import YOLO

    project = args.work / "runs"
    model = YOLO(str(model_yaml))
    model.train(
        trainer=TODDetectionTrainer,
        data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, workers=0, plots=False, val=True, exist_ok=True,
        project=str(project), name=variant.name, verbose=False,
        **{k: v for k, v in (spec.get("train") or {}).items()
           if k in {"dfl", "optimizer", "lr0", "lrf", "momentum", "weight_decay",
                    "warmup_epochs", "warmup_momentum", "close_mosaic", "mosaic", "mixup",
                    "multi_scale", "amp", "seed", "deterministic"}},
    )

    run_dir = project / variant.name
    best, last = run_dir / "weights" / "best.pt", run_dir / "weights" / "last.pt"
    check("训练产出 best.pt", best.is_file())
    check("训练产出 last.pt", last.is_file())

    with open(run_dir / "results.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    check("results.csv 有训练记录", len(rows) >= args.epochs, f"实际 {len(rows)} 行")
    dfl_col = next((c for c in rows[0] if "dfl_loss" in c), None)
    check("results.csv 含 dfl_loss 列（框架仍会记录它）", dfl_col is not None,
          f"列名：{list(rows[0])}")
    dfl_values = [float(r[dfl_col]) for r in rows]
    check("dfl_loss 恒为 0（论文 §4.3 的 dfl=0.0 / 无 DFL 生效）",
          all(abs(v) < 1e-9 for v in dfl_values), f"实际 {dfl_values}")

    reloaded = YOLO(str(best))
    check("best.pt 可被重新载入（含动态分配的 STAL 类）", reloaded.model is not None)
    images = sorted((args.work / "data" / "images" / "val").glob("*.jpg"))
    results = reloaded.predict(str(images[0]), imgsz=args.imgsz, device=args.device, verbose=False)
    check("载入后能推理并返回结果", len(results) == 1 and hasattr(results[0], "boxes"))
    print(f"       -> 推理结果：{len(results[0].boxes)} 个框，"
          f"图像 {Path(images[0]).name}")
    print(f"       -> 实验目录：{run_dir}")

    if not args.keep:
        shutil.rmtree(args.work, ignore_errors=True)
        print("       -> 已清理临时实验目录（--keep 可保留）")

    print(f"\n训练回路自检通过：{len(PASSED)} 项\n")
    for label in PASSED:
        print(f"  ok  {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
