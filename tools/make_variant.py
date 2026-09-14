"""把配方（Python 代码）物化成变体产物：variant.yaml + model.yaml + card.md。

架构约定（PLAN.md §4.2）：变体由**代码**定义，不由人手写 YAML 维护。
配方文件里只需暴露一个名为 ``variant`` 的 ``tod.compose.Variant`` 对象::

    # variants/SPAE-YOLOv8n/recipe.py
    from tod.compose import Variant

    variant = Variant("SPAE-YOLOv8n", base="yolov8n").data(...)...

用法::

    python tools/make_variant.py variants/SPAE-YOLOv8n/recipe.py
    python tools/make_variant.py variants/SPAE-YOLOv8n/recipe.py --no-model
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tod.compat import CompatError  # noqa: E402
from tod.compose import Variant  # noqa: E402


def load_recipe(path: Path) -> Variant:
    spec = importlib.util.spec_from_file_location(f"_tod_recipe_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"无法加载配方：{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    variant = getattr(module, "variant", None)
    if not isinstance(variant, Variant):
        raise SystemExit(f"{path} 必须暴露一个名为 variant 的 Variant 对象。")
    return variant


def main() -> int:
    ap = argparse.ArgumentParser(description="物化变体：variant.yaml / model.yaml / card.md")
    ap.add_argument("recipe", type=Path, help="配方文件（.py）")
    ap.add_argument("--out", type=Path, default=None, help="输出目录，默认 recipe 所在目录")
    ap.add_argument("--no-model", action="store_true", help="跳过模型 YAML 生成")
    ap.add_argument("--no-card", action="store_true", help="跳过卡片生成")
    args = ap.parse_args()

    out = args.out or args.recipe.parent
    out.mkdir(parents=True, exist_ok=True)

    # 先导入模块库/损失库，让卡片能自动带出论文来源与许可证。
    # 这两步依赖 torch，环境未就绪时降级为警告，不阻塞 spec 生成。
    try:
        import tod.modules  # noqa: F401
        import tod.loss  # noqa: F401
    except ImportError as exc:
        print(f"[warn ] 未能导入模块库（{exc}）：卡片里的来源表可能为空。")

    variant = load_recipe(args.recipe)

    spec_path = variant.dump(out / "variant.yaml")
    print(f"[spec ] {spec_path}")

    if not args.no_model:
        try:
            cfg = variant.model_yaml(write=False)
            from tod.compat import dump_yaml

            model_path = dump_yaml(cfg, out / "model.yaml")
            replaced = cfg.get("_downsample_replaced") or {}
            print(f"[model] {model_path}  "
                  f"(P2: {cfg.get('_p2_status', '未启用')}, 下采样替换 {len(replaced)} 处, "
                  f"类型替换 {cfg.get('_type_map_applied') or '无'})")
        except CompatError as exc:
            print(f"[model] 跳过：{exc}")

    if not args.no_card:
        card_path = variant.card(out / "card.md")
        print(f"[card ] {card_path}")

    print("\n下一步：python tools/train.py --variant", spec_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
