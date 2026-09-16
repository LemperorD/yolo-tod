"""消融流水线：一个变体的 leave-one-out / 逐项开关批量跑「训练 + 尺度分层评测」。

**为什么必须自动化**（PLAN §2.3、§4.2）：变体的价值不在"涨了几个点"，而在
"这个点是谁贡献的、代价多少"。手工改配置跑消融一定会让配置与代码漂移；
本工具直接从配方（``recipe.py``）派生消融配置，并调用与
``tools/train.py`` / ``tools/val.py`` **完全相同**的路径，保证可比。

用法::

    # 1) 只看消融网格（不训练，秒级；用于评审/确认配置正确）
    python tools/ablation.py --recipe variants/SDD-YOLO26n/recipe.py --plan

    # 2) 真跑：默认 baseline + 逐项剔除（leave-one-out）
    python tools/ablation.py --recipe variants/SDD-YOLO26n/recipe.py \
        --data configs/_base_/datasets/visdrone2019-det.yaml \
        --epochs 100 --imgsz 1024 --batch 4 --device 0

    # 3) 自检（合成数据、1 epoch、CPU 也能跑通整条流水线）
    python tools/ablation.py --recipe variants/SDD-YOLO26n/recipe.py \
        --data tests/.tmp/tiny-detect/dataset.yaml --epochs 1 --imgsz 320 --batch 2 \
        --device cpu --set workers=0 --max-images 8

产物（全部在 ``--out`` 下，默认 results/ablations/<变体名>/）：

    specs/<配置名>.yaml    派生的变体 spec（可追溯：由哪个 recipe + 去掉什么得到）
    runs/<配置名>/         训练输出（权重、results.csv、args.yaml）
    summary.csv            每次配置的 整体/各尺度层 AP50、AP50:95、参数量、耗时
    summary.md             人读的汇总表（可直接贴进变体卡片）

消融项怎么指定：
    * ``--drop`` 给出"要剔掉的项"，每项单独生成一个配置（leave-one-out）；
    * ``--keep-only`` 反向：只保留列出的项，其余全部剔掉；
    * 不指定时 **自动** 扫描配方 spec 里所有可剥离的键（EP 覆盖值 + model 段开关）。
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

#: 这些键是"结构性开关"，参与自动消融扫描（其余如 nc/imgsz 属于固定设定）
ABLATABLE_MODEL_KEYS = ("add_p2", "p2_pre", "p2_fuse_block")
#: 这些键只是"从框架继承"的标记位或超参，其**值**不作为消融目标
NEVER_DROP = ("prefer_native", "prog_loss", "kd_levels")

#: 这些取值是"模式开关"而不是组件（剔掉它们没有意义）
NEVER_TARGETS = {"native", "input", "framework", "auto", "none", "None", ""}

#: 组件的"关掉"动作：**不能一律用 without** —— 由框架继承来的能力（STAL / MuSGD）
#: 删掉键只会退回框架默认（8.4 的默认 TAL 本身就带小目标先验），
#: 必须显式切到经典替代物，否则消融会得出"该组件没用"的假结论。
ABLATION_ACTIONS: dict[str, "callable"] = {
    "DualAttention": lambda v: v.without("DualAttention"),
    "STAL": lambda v: v.assigner("TAL", small_target_aware=False),
    "wiou": lambda v: v.without("wiou"),
    "MuSGD": lambda v: v.strategy(optimizer="SGD"),
    "add_p2": lambda v: v.without("add_p2"),
    "p2_fuse_block": lambda v: v.without("C3"),
    "FeatureAlignKD": lambda v: v.strategy(distill=None),
}


def load_recipe(path: Path):
    """加载配方并返回 ``tod.compose.Variant``（同时触发模块注册，卡片/指标才有来源）。"""
    import tod

    tod.import_libraries()
    spec = importlib.util.spec_from_file_location(f"_tod_ablation_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    variant = getattr(module, "variant", None)
    if variant is None:
        raise SystemExit(f"{path} 必须暴露一个名为 variant 的 Variant 对象")
    return variant


def discover_ablations(variant) -> list[str]:
    """自动扫描：EP 覆盖里的值 + model 段的开关，作为可剥离项。"""
    items: list[str] = []
    spec = variant.spec()
    for ep, cfg in (spec.get("eps") or {}).items():
        for key, value in cfg.items():
            if key in NEVER_DROP:
                continue
            if isinstance(value, str) and value not in NEVER_TARGETS:
                items.append(value)
            elif key in ABLATABLE_MODEL_KEYS:
                items.append(key)
    for key in ABLATABLE_MODEL_KEYS:
        if spec.get("model", {}).get(key):
            items.append(key)
    # 去重并保序
    seen: set[str] = set()
    return [x for x in items if not (x in seen or seen.add(x))]


def diff_spec(base: dict, other: dict) -> list[str]:
    """列出 ``other`` 相对 ``base`` 的配置差异（人读；用于 --plan 与日志）。"""
    out: list[str] = []

    def walk(prefix: str, a, b) -> None:
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b)):
                walk(f"{prefix}.{key}" if prefix else key, a.get(key, "<缺>"), b.get(key, "<缺>"))
            return
        if a != b:
            out.append(f"{prefix}: {a!r} → {b!r}")

    walk("", base, other)
    return out


#: 这些字段的差异不代表"配置变了"（只是名字/来源清单）
IGNORABLE_DIFF_KEYS = {"id", "modules", "notes", "tags"}


def is_equivalent(diff: list[str]) -> bool:
    """差异只落在名字/来源清单上 ⇒ 该消融与 baseline 等价（例如剔掉的键本就是默认值）。"""
    return all(d.split(":", 1)[0] in IGNORABLE_DIFF_KEYS for d in diff)


def make_variants(variant, drop: list[str], keep_only: list[str], auto: bool):
    """生成 ``[(配置名, Variant), ...]``：baseline 在前。"""
    from tod.compose import Variant

    def clone() -> "Variant":
        return Variant.from_spec(variant.spec())

    def switch_off(v: "Variant", item: str) -> None:
        action = ABLATION_ACTIONS.get(item)
        if action is None:
            v.without(item)
        else:
            action(v)

    out = [("baseline", clone())]
    if keep_only:
        base = clone()
        for item in discover_ablations(variant):
            if item not in keep_only:
                switch_off(base, item)
        base.name = f"only-{'+'.join(keep_only)}"
        return out + [(base.name, base)]

    targets = drop or (discover_ablations(variant) if auto else [])
    for item in targets:
        v = clone()
        switch_off(v, item)
        v.name = f"without-{item}"
        out.append((v.name, v))
    return out


# --------------------------------------------------------------------- 执行


def train_one(variant, data: Path, out_dir: Path, args) -> Path:
    """跑一个配置的训练，返回 best.pt 路径。"""
    import tod
    from tod import runtime

    tod.bootstrap()
    from ultralytics import YOLO

    from tod.engine.trainer import TODDetectionTrainer

    spec = variant.spec()
    run_dir = out_dir / "runs" / variant.name
    model_yaml = out_dir / "specs" / f"{variant.name}.yaml"
    variant.model_yaml(out_dir / "specs" / f"{variant.name}.model.yaml", write=True)
    variant.dump(model_yaml)
    runtime.set_active(spec)

    train_args = dict(spec.get("train") or {})
    train_args.update({
        "data": str(data), "epochs": args.epochs, "imgsz": args.imgsz, "batch": args.batch,
        "device": args.device, "workers": args.workers, "plots": False, "val": True,
        "exist_ok": True, "project": str(out_dir / "runs"), "name": variant.name,
        "verbose": False,
    })
    for item in args.set:
        key, _, value = item.partition("=")
        train_args[key] = _coerce(value)
    model = YOLO(str(out_dir / "specs" / f"{variant.name}.model.yaml"))
    model.train(trainer=TODDetectionTrainer, **train_args)
    return run_dir / "weights" / "best.pt"


def eval_one(variant, weights: Path, data: Path, args) -> dict:
    """对训练结果做尺度分层评测，返回 ``{层名: 指标}``（含 "all"）。"""
    from tod.eval.scales import evaluate

    result = evaluate(data, weights=weights, split=args.split, imgsz=args.imgsz,
                      conf=args.conf, device=args.device, max_images=args.max_images,
                      verbose=False)
    return result["bins"]          # 只取分层指标；"all" 也在里面，便于统一列表头


def _coerce(text: str):
    import ast

    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def write_summary(rows: list[dict], out_dir: Path) -> None:
    """把结果写成 summary.csv 与人读的 summary.md。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    bins = sorted({b for row in rows for b in row["metrics"]})
    csv_path = out_dir / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["config", "params", "minutes"] + [f"{b}_AP50" for b in bins]
                        + [f"{b}_AP50_95" for b in bins])
        for row in rows:
            writer.writerow([row["config"], row["params"], f"{row['minutes']:.2f}"]
                            + [f"{row['metrics'].get(b, {}).get('ap50', float('nan')):.4f}"
                               for b in bins]
                            + [f"{row['metrics'].get(b, {}).get('ap50_95', float('nan')):.4f}"
                               for b in bins])

    md = ["# 消融汇总", "", f"- 数据集：`{rows[0]['data']}`",
          f"- imgsz：{rows[0]['imgsz']}｜epochs：{rows[0]['epochs']}｜git：`{rows[0]['git']}`", "",
          "| 配置 | 参数量 | 分钟 | " + " | ".join(f"{b} AP50" for b in bins) + " |",
          "|---|---|---|" + "---|" * len(bins)]
    for row in rows:
        cells = " | ".join(f"{row['metrics'].get(b, {}).get('ap50', float('nan')):.4f}" for b in bins)
        md.append(f"| `{row['config']}` | {row['params']:,} | {row['minutes']:.2f} | {cells} |")
    md += ["", "> 判读：与前一行 `baseline` 的差才是该组件的贡献；"
           "小目标变体必须看最小两层，别只看某一列的绝对值。", ""]
    (out_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[ablation] 汇总 → {csv_path} / {out_dir / 'summary.md'}")


def _git_commit() -> str:
    import subprocess

    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description="消融流水线（leave-one-out + 尺度分层评测）")
    ap.add_argument("--recipe", type=Path, required=True, help="配方文件（暴露 variant）")
    ap.add_argument("--data", type=Path, default=None, help="数据集 YAML（--plan 可省）")
    ap.add_argument("--drop", nargs="*", default=None, help="要剔除的项（可多个）")
    ap.add_argument("--keep-only", nargs="*", default=None, help="只保留这些项，其余剔除")
    ap.add_argument("--no-auto", action="store_true", help="不自动扫描，仅用 --drop 指定")
    ap.add_argument("--plan", action="store_true", help="只打印消融网格与配置差异，不训练")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "ablations")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--device", default=None)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--split", default="val")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--set", action="append", default=[], metavar="K=V")
    args = ap.parse_args()

    variant = load_recipe(args.recipe)
    variants = make_variants(variant, args.drop or [], args.keep_only or [],
                             auto=not args.no_auto)
    out_root = args.out / variant.name
    args.imgsz = args.imgsz or int((variant.spec().get("data") or {}).get("imgsz", 640))

    print(f"[ablation] 变体={variant.name}｜配置数={len(variants)}｜输出={out_root}")
    baseline_spec = variants[0][1].spec()
    runnable: list[tuple[str, object]] = [variants[0]]
    for name, v in variants[1:]:
        diff = diff_spec(baseline_spec, v.spec())
        if is_equivalent(diff):
            print(f"  - {name:<28} ⚠ 与 baseline 等价（剔掉的键本来就是默认值），已跳过")
            continue
        print(f"  - {name:<28} " + "；".join(diff))
        runnable.append((name, v))
    variants = runnable

    (out_root / "specs").mkdir(parents=True, exist_ok=True)
    for name, v in variants:
        v.dump(out_root / "specs" / f"{name}.yaml")     # 与变体 spec 同格式，可追溯
    if args.plan:
        print(f"[ablation] --plan：已写出 {len(variants)} 份配置到 {out_root / 'specs'}")
        return 0

    if args.data is None:
        raise SystemExit("真跑必须给 --data（或用 --plan 只看网格）")
    data = args.data if args.data.is_absolute() else (ROOT / args.data).resolve()
    if not data.is_file():
        raise SystemExit(f"数据集配置不存在：{data}")

    rows: list[dict] = []
    for name, v in variants:
        started = time.time()
        weights = train_one(v, data, out_root, args)
        metrics = eval_one(v, weights, data, args)
        params = 0
        try:
            import torch

            params = sum(p.numel() for p in torch.load(weights, map_location="cpu",
                                                       weights_only=False)["model"].parameters())
        except Exception:  # noqa: BLE001 - 参数量只是附注信息
            pass
        rows.append({"config": name, "metrics": metrics, "params": params,
                     "minutes": (time.time() - started) / 60, "data": str(data),
                     "imgsz": args.imgsz, "epochs": args.epochs, "git": _git_commit()})
        overall = metrics.get("all", {})
        print(f"[ablation] {name}: 整体 AP50={overall.get('ap50', float('nan')):.4f} "
              f"AP50:95={overall.get('ap50_95', float('nan')):.4f}（{rows[-1]['minutes']:.2f} min）")
    write_summary(rows, out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
