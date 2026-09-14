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
    ap.add_argument("--dry-run", action="store_true", help="只建模型并前向一次，不训练")
    return ap


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

        from tod.engine.surgery import apply_spec

        imgsz = args.imgsz or int(spec.get("data", {}).get("imgsz", 640))
        applied = apply_spec(model.model, spec)
        if applied:
            print("[EP5] 检测头手术：")
            for line in applied:
                print(f"       {line}")

        model.model.eval()
        head = model.model.model[-1]
        n_before = sum(p.numel() for p in model.model.parameters())
        with torch.no_grad():
            out = model.model(torch.zeros(1, 3, imgsz, imgsz))
        n_after = sum(p.numel() for p in model.model.parameters())
        print(f"[自检] imgsz={imgsz}  参数量={n_after:,} ({n_after / 1e6:.2f} M)"
              + (f"  头部手术影响 {n_before - n_after:+,} 参数" if applied else ""))
        print(f"[自检] 检测层数 nl={getattr(head, 'nl', '?')}  stride={list(getattr(head, 'stride', []))}")
        print(f"[自检] 头输入特征图尺寸：{getattr(head, 'f', None)}")
        print(f"[自检] 前向输出层数：{len(out) if isinstance(out, (list, tuple)) else 1}")
        return 0

    train_args = dict(spec.get("train") or {})
    data = args.data or _default_data(variant.dataset)
    if data is not None:
        train_args["data"] = str(data)
    elif "data" not in train_args:
        raise SystemExit(
            "未指定数据集：请用 --data 传入数据集 YAML，"
            "或在变体配置里写 data.dataset。"
        )

    train_args.setdefault("project", str(ROOT / "results"))
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


def _default_data(dataset: str) -> Path | None:
    """按数据集名推断本库的 base 配置路径。"""
    if not dataset:
        return None
    candidate = ROOT / "configs" / "_base_" / "datasets" / f"{dataset}.yaml"
    return candidate if candidate.is_file() else None


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CompatError as exc:  # 环境/版本问题给出可读提示
        raise SystemExit(f"[环境错误] {exc}") from exc
