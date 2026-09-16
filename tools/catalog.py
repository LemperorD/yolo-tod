"""生成 docs/VARIANTS.md（模块总表）。

用法::

    python tools/catalog.py            # 打印到标准输出
    python tools/catalog.py --write    # 写入 docs/VARIANTS.md

为什么不用 ``python -m tod.registry``：包 __init__ 已导入该子模块，
runpy 会抛 RuntimeWarning，输出不干净。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tod.registry import catalog  # noqa: E402


def import_modules() -> None:
    """导入模块库以触发注册。

    模块文件依赖 torch；环境未就绪时总表会缺条目（这是环境问题，不是 bug）。
    """
    try:
        import tod.assigner  # noqa: F401
        import tod.loss  # noqa: F401
        import tod.modules  # noqa: F401
        import tod.optim  # noqa: F401
    except ImportError as exc:
        print(f"[warn] 未能导入模块库（{exc}）：总表不完整。\n", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description="生成模块总表")
    ap.add_argument("--write", action="store_true", help="写入 docs/VARIANTS.md")
    ap.add_argument("--no-modules", action="store_true", help="跳过模块导入（仅看注册表本身）")
    ap.add_argument("--out", default=str(ROOT / "docs" / "VARIANTS.md"))
    args = ap.parse_args()

    if not args.no_modules:
        import_modules()

    text = catalog()
    if args.write:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"已写入 {out}")
    else:
        sys.stdout.reconfigure(encoding="utf-8")
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
