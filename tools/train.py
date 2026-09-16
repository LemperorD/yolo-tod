"""训练入口：读变体 spec → 注入魔改模块 → 生成模型图 → 交给 ultralytics 训练。

用法::

    # 0) 先物化变体（生成 variant.yaml / model.yaml / card.md）
    python tools/make_variant.py variants/SPAE-YOLOv8n/recipe.py

    # 1) 只做结构自检：建模型 + 一次前向，不训练（不需要数据集）
    python tools/train.py --variant variants/SPAE-YOLOv8n/variant.yaml --dry-run

    # 2) 真训练
    python tools/train.py --variant variants/SPAE-YOLOv8n/variant.yaml \
        --data configs/_base_/datasets/visdrone2019-det.yaml --epochs 150 --batch 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="TOD 训练入口")
    ap.add_argument("--variant", type=Path, required=True, help="变体 spec YAML")
    ap.add_argument("--data", type=Path, default=None, help="数据集 YAML（覆盖变体设置）")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--device", default=None, help="如 0 或 cpu")
    ap.add_argument("--name", default=None, help="实验名，默认用变体 id")
    ap.add_argument("--set", action="append", default=[], metavar="K=V",
                    help="覆盖任意训练参数（可重复），如 --set workers=0 --set plots=False")
    ap.add_argument("--dry-run", action="store_true", help="只建模型并前向一次，不训练")
    return ap


def _coerce(text: str):
    """把 ``--set`` 的字符串值转成 int/float/bool/None，失败则原样保留字符串。"""
    import ast

    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def main() -> int:
    args = build_parser().parse_args()

    import tod
    from tod import runtime
    from tod.compat import CompatError, load_yaml
    from tod.compose import Variant

    if not args.variant.is_file():
        raise SystemExit(f"变体文件不存在：{args.variant}")
    spec = load_yaml(args.variant)
    variant = Variant.from_spec(spec)

    # 现场重新生成模型图：保证 model.yaml 与变体配置永不脱节
    out_dir = args.variant.parent
    model_path = out_dir / "model.yaml"
    cfg = variant.model_yaml(model_path, write=True)
    print(f"[模型] {model_path}")
    print(f"       P2 分支：{cfg.get('_p2_status', '未启用')}；"
          f"下采样替换：{cfg.get('_downsample_replaced') or '无'}；"
          f"节点类型替换：{cfg.get('_type_map_applied') or '无'}")

    try:
        tod.bootstrap()          # 导入模块库并把魔改注入框架命名空间
    except ImportError as exc:
        raise SystemExit(
            f"载入魔改模块失败：{exc}\n"
            "请确认已完成环境安装（ultralytics + torch）。"
        ) from exc

    runtime.set_active(spec)

    from ultralytics import YOLO

    model = YOLO(str(model_path))

    if args.dry_run:
        import torch
        from ultralytics.cfg import get_cfg

        from tod.engine.surgery import apply_spec
        from tod.loss.criterion import build_detection_loss

        imgsz = args.imgsz or int(spec.get("data", {}).get("imgsz", 640))
        n_before = sum(p.numel() for p in model.model.parameters())
        applied = apply_spec(model.model, spec)
        if applied:
            print("[EP4/EP5] 建模后手术：")
            for line in applied:
                print(f"       {line}")
        n_after = sum(p.numel() for p in model.model.parameters())

        # ---- EP6/EP7/EP9：准则与优化器接线自检（同样不需要数据集）----
        eps = spec.get("eps") or {}
        ep6, ep7, ep9 = (eps.get("EP6") or {}), (eps.get("EP7") or {}), (eps.get("EP9") or {})
        model.model.args = _dry_run_args(spec, get_cfg)      # 准则会读 model.args 当 hyp

        from tod.engine.trainer import stal_flag

        dfl_gain = float(getattr(model.model.args, "dfl", 1.5) or 0.0)
        use_dfl = False if (dfl_gain == 0.0 and ep7.get("box")) else None
        criterion = build_detection_loss(
            model.model, kind=ep7.get("box"),
            theta=float(ep7.get("theta", 4.0) or 4.0), use_dfl=use_dfl,
            stal=stal_flag(ep6), **(ep7.get("kind_kwargs") or {}),
        )
        print("[EP6/EP7] 训练准则：" + ("；".join(criterion._tod_patched) or "框架默认（未做替换）"))
        prog = callable(getattr(criterion, "update", None)) or callable(
            getattr(getattr(criterion, "base", None), "update", None))
        print(f"[EP9] 优化器={ep9.get('optimizer', 'auto')}"
              f" | dfl 增益={dfl_gain}"
              f" | 蒸馏={'开启' if ep9.get('distill') and ep9.get('teacher') else '关闭'}"
              f" | ProgLoss={'框架原生 E2ELoss.update' if prog else '不适用'}")

        model.model.eval()
        head = model.model.model[-1]
        with torch.no_grad():
            out = model.model(torch.zeros(1, 3, imgsz, imgsz))
        print(f"[自检] imgsz={imgsz}  参数量={n_after:,} ({n_after / 1e6:.2f} M)"
              + (f"  手术影响 {n_after - n_before:+,} 参数" if applied else ""))
        print(f"[自检] 检测层数 nl={getattr(head, 'nl', '?')}  stride={list(getattr(head, 'stride', []))}")
        print(f"[自检] 头输入特征图索引：{list(getattr(head, 'f', []))}")
        print(f"[自检] 前向输出层数：{len(out) if isinstance(out, (list, tuple)) else 1}")
        return 0

    train_args = dict(spec.get("train") or {})
    for item in args.set:                      # --set k=v 覆盖（消融/自检常用）
        if "=" not in item:
            raise SystemExit(f"--set 需要 K=V 形式，收到 {item!r}")
        key, value = item.split("=", 1)
        train_args[key.strip()] = _coerce(value.strip())

    data = args.data or _default_data(variant.dataset)
    if data is not None:
        train_args["data"] = str(data)
    elif "data" not in train_args:
        raise SystemExit(
            "未指定数据集：请用 --data 传入数据集 YAML，"
            "或在变体配置里写 data.dataset。"
        )

    # 实验输出目录：变体里写的可能是相对路径（如 "results"），必须锚到仓库根，
    # 否则 ultralytics 会把它解析成 runs/detect/results/...（散落在仓库里）
    project = train_args.get("project")
    if not project or not Path(str(project)).is_absolute():
        train_args["project"] = str(ROOT / (str(project) if project else "results"))
    train_args["name"] = args.name or variant.name
    for key, value in (("epochs", args.epochs), ("imgsz", args.imgsz),
                       ("batch", args.batch), ("device", args.device)):
        if value is not None:
            train_args[key] = value
    train_args.setdefault("imgsz", spec.get("data", {}).get("imgsz"))
    train_args = {k: v for k, v in train_args.items() if v is not None}

    from tod.engine.trainer import TODDetectionTrainer

    print(f"[训练] {variant.name} | data={train_args.get('data')} | "
          f"imgsz={train_args.get('imgsz')} | epochs={train_args.get('epochs')} | "
          f"box_loss={runtime.get('eps.EP7.box', '框架默认')}")
    model.train(trainer=TODDetectionTrainer, **train_args)
    return 0


def _dry_run_args(spec: dict, get_cfg):
    """给 ``--dry-run`` 造一份训练超参命名空间。

    准则（``v8DetectionLoss``/``E2ELoss``）会把 ``model.args`` 当超参读，
    直接建图得到的 ``model.args`` 是 dict，会报 ``'dict' object has no attribute 'box'``。
    """
    args = get_cfg()
    for key, value in (spec.get("train") or {}).items():
        if hasattr(args, key):
            setattr(args, key, value)
    return args


def _default_data(dataset: str) -> Path | None:
    """按数据集名推断本库的 base 配置路径。"""
    if not dataset:
        return None
    candidate = ROOT / "configs" / "_base_" / "datasets" / f"{dataset}.yaml"
    return candidate if candidate.is_file() else None


if __name__ == "__main__":
    # CompatError 在 main() 内部才被导入（那时 sys.path 才加好），
    # 所以这里也要延迟导入：早年写成模块级 except 会抛 NameError 掩盖真实退出码。
    from tod.compat import CompatError as _CompatError

    try:
        raise SystemExit(main())
    except _CompatError as exc:  # 环境/版本问题给出可读提示
        raise SystemExit(f"[环境错误] {exc}") from exc
