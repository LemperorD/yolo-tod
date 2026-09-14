"""模块级测试：形状 / 数值 / 端到端建图。

需要 torch + ultralytics（跑 ``python tests/smoke.py`` 的那套无依赖检查在另一个文件）。

运行::

    python tests/test_modules.py            # 全部
    python tests/test_modules.py --fast     # 跳过端到端建图
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PASSED: list[str] = []
SKIPPED: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    if not cond:
        raise AssertionError(f"[FAIL] {label} {detail}")
    PASSED.append(label)


def skip(label: str, why: str) -> None:
    SKIPPED.append(f"{label} —— {why}")


def _need_torch():
    import importlib.util

    if importlib.util.find_spec("torch") is None:
        return None
    import torch

    return torch


# --------------------------------------------------------------------- ADown


def test_adown() -> None:
    torch = _need_torch()
    if torch is None:
        return skip("ADown", "未安装 torch")

    import tod.modules  # noqa: F401  触发注册
    from tod.modules.conv.adown import ADown

    m = ADown(64, 128).eval()
    with torch.no_grad():
        y = m(torch.randn(2, 64, 40, 40))
    check("ADown 输出形状", tuple(y.shape) == (2, 128, 20, 20), f"实际 {tuple(y.shape)}")

    # 非方形输入
    with torch.no_grad():
        y = m(torch.randn(1, 64, 33, 41))
    check("ADown 非方形输入", tuple(y.shape) == (1, 128, 17, 21), f"实际 {tuple(y.shape)}")

    # 奇数输入通道必须报错（要沿通道一分为二）
    try:
        ADown(63, 128)
        raise AssertionError("[FAIL] 奇数输入通道未报错")
    except ValueError:
        PASSED.append("ADown 拒绝奇数输入通道")

    # 两个分支各占一半通道
    check("ADown 分支通道", ADown(64, 128).c == 64)


# ---------------------------------------------------------------------- SIoU


def test_siou() -> None:
    torch = _need_torch()
    if torch is None:
        return skip("SIoU", "未安装 torch")

    from tod.loss.box import siou

    # 1) 完全重合 → ≈ 1
    box = torch.tensor([[10.0, 10.0, 20.0, 20.0]])          # xywh
    v = siou(box, box.clone(), xywh=True)
    check("SIoU 完全重合≈1", bool((v - 1).abs().max() < 1e-4), f"实际 {v.item():.6f}")

    # 2) SIoU ≤ IoU（距离/形状代价非负）
    a = torch.tensor([[10.0, 10.0, 20.0, 20.0]])
    b = torch.tensor([[12.0, 11.0, 18.0, 22.0]])
    s = siou(a, b, xywh=True)
    iou = _plain_iou(a, b)
    check("SIoU ≤ IoU", bool((s <= iou + 1e-6).all()), f"siou={s.item():.4f} iou={iou.item():.4f}")

    # 3) 对称性
    s_ab = siou(a, b, xywh=True)
    s_ba = siou(b, a, xywh=True)
    check("SIoU 对称", bool((s_ab - s_ba).abs().max() < 1e-5),
          f"{s_ab.item():.6f} vs {s_ba.item():.6f}")

    # 4) xywh 与 xyxy 两种输入应给出相同结果
    a_xyxy = _xywh2xyxy(a)
    b_xyxy = _xywh2xyxy(b)
    s_xyxy = siou(a_xyxy, b_xyxy, xywh=False)
    check("SIoU xywh/xyxy 一致", bool((s_ab - s_xyxy).abs().max() < 1e-5),
          f"{s_ab.item():.6f} vs {s_xyxy.item():.6f}")

    # 5) 小目标敏感度：1 像素偏移对 SIoU 的伤害应小于对 IoU 的伤害
    tiny = torch.tensor([[50.0, 50.0, 8.0, 8.0]])
    tiny_shift = torch.tensor([[51.0, 50.0, 8.0, 8.0]])
    s_tiny = siou(tiny, tiny_shift, xywh=True)
    iou_tiny = _plain_iou(tiny, tiny_shift)
    check("小目标上 SIoU 比 IoU 平滑", bool(s_tiny > iou_tiny - 0.5),
          f"siou={s_tiny.item():.4f} iou={iou_tiny.item():.4f}")


def _xywh2xyxy(b):
    cx, cy, w, h = b.chunk(4, -1)
    return __import__("torch").cat([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)


def _plain_iou(a, b):
    ax1, ay1, ax2, ay2 = _xywh2xyxy(a).chunk(4, -1)
    bx1, by1, bx2, by2 = _xywh2xyxy(b).chunk(4, -1)
    inter = (ax2.min(bx2) - ax1.max(bx1)).clamp(0) * (ay2.min(by2) - ay1.max(by1)).clamp(0)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter + 1e-7
    return inter / union


# ----------------------------------------------------------------- 命名空间注入


def test_namespace_injection() -> None:
    try:
        import tod
        from tod import compat
    except ImportError as exc:
        return skip("命名空间注入", f"导入失败：{exc}")

    if not compat.installed():
        return skip("命名空间注入", "未安装 ultralytics")
    try:
        tod.bootstrap()
    except ImportError as exc:
        return skip("命名空间注入", f"模块库导入失败：{exc}")

    ns = compat.model_globals()
    for name in ("ADown", "Efficient_UAVDet", "siou"):
        check(f"命名空间含 {name}", name in ns)


# --------------------------------------------------- Efficient_UAVDet 换头手术


def test_efficient_uavdet_surgery() -> None:
    torch = _need_torch()
    if torch is None:
        return skip("Efficient_UAVDet", "未安装 torch")
    import importlib.util

    if importlib.util.find_spec("ultralytics") is None:
        return skip("Efficient_UAVDet", "未安装 ultralytics")

    from ultralytics.nn.modules import Detect

    from tod.modules.head.efficient_uavdet import describe, swap_detect_head

    ch = (32, 64, 128, 256)          # 论文 Table 3 的 x
    head = Detect(nc=10, ch=ch)
    n_before = sum(p.numel() for p in head.parameters())
    out_ch_before = head.cv2[0][-1].bias.numel()

    swap_detect_head(head, per_group=16, channels="native")
    n_after = sum(p.numel() for p in head.parameters())

    check("换头后参数量下降", n_after < n_before, f"{n_before:,} -> {n_after:,}")
    check("输出层通道保持 4*reg_max",
          head.cv2[0][-1].bias.numel() == out_ch_before,
          f"{out_ch_before} -> {head.cv2[0][-1].bias.numel()}")

    groups = [head.cv2[i][0].groups for i in range(4)]
    check("分组数 g = x/16 = [2,4,8,16]", groups == [2, 4, 8, 16], f"实际 {groups}")
    check("每组通道数恒为 16",
          all(c // g == 16 for c, g in zip(ch, groups)),
          f"{[c // g for c, g in zip(ch, groups)]}")

    head.train()
    x = [torch.randn(1, c, s, s) for c, s in zip(ch, (160, 80, 40, 20))]
    y = head(x)
    check("换头后前向可跑通", isinstance(y, (list, tuple)) and len(y) == 4)
    check("前向输出通道 = nc + 4*reg_max", y[0].shape[1] == 10 + 64, f"实际 {y[0].shape[1]}")
    print("       ↳ " + describe(head).replace("\n", "\n         "))

    # 另一种通道策略（in=out=x）也必须可用
    head2 = Detect(nc=10, ch=ch)
    swap_detect_head(head2, per_group=16, channels="input")
    check("input 策略：stem 内 in=out=x",
          head2.cv2[1][0].cv1.conv.in_channels == head2.cv2[1][0].cv2.conv.out_channels == 64)


# ---------------------------------------------------------------------- 端到端


def test_end_to_end_build(fast: bool = False) -> None:
    if fast:
        return skip("端到端建图", "--fast")

    import importlib.util

    for pkg in ("torch", "ultralytics"):
        if importlib.util.find_spec(pkg) is None:
            return skip("端到端建图", f"未安装 {pkg}")

    recipe = ROOT / "variants" / "SPAE-YOLOv8n" / "recipe.py"
    if not recipe.is_file():
        return skip("端到端建图", "缺少 SPAE 配方")

    import tod
    from tod.compose import Variant

    spec = _load_recipe_variant(recipe)
    tod.bootstrap()
    spec_dict = spec.spec()

    cfg = spec.model_yaml(write=False)
    check("SPAE：P2 已注入", cfg.get("_p2_status") == "injected")
    check("SPAE：下采样替换 4 处", len(cfg.get("_downsample_replaced") or {}) == 4)
    check("SPAE：检测头记录为建模后手术",
          cfg.get("_head_surgery") == "Efficient_UAVDet", f"实际 {cfg.get('_head_surgery')}")
    check("SPAE：YAML 里检测头仍是原生 Detect",
          str((cfg.get("head") or cfg.get("model"))[-1][2]) == "Detect")

    from ultralytics import YOLO

    model_path = recipe.parent / "model.yaml"
    from tod.compat import dump_yaml

    dump_yaml(cfg, model_path)
    yolo = YOLO(str(model_path))
    head = yolo.model.model[-1]
    n_yaml = sum(p.numel() for p in yolo.model.parameters())
    check("SPAE：检测层数 = 4", getattr(head, "nl", 0) == 4, f"实际 {getattr(head, 'nl', '?')}")

    from tod.engine.surgery import apply_spec

    applied = apply_spec(yolo.model, spec_dict)
    check("SPAE：EP5 手术已应用", bool(applied), f"applied={applied}")

    import torch

    yolo.model.eval()
    with torch.no_grad():
        out = yolo.model(torch.zeros(1, 3, 640, 640))
    n_final = sum(p.numel() for p in yolo.model.parameters())
    check("SPAE：stride 含 4（P2 层生效）", 4 in list(getattr(head, "stride", [])),
          f"实际 {list(getattr(head, 'stride', []))}")
    check("SPAE：前向可跑通", out is not None)
    print(f"       ↳ 参数量 {n_final:,} ({n_final / 1e6:.2f} M)"
          f"，换头减少 {n_yaml - n_final:,}，头输入尺寸 {getattr(head, 'f', None)}")


def _load_recipe_variant(path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("_tod_recipe_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.variant


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="跳过端到端建图")
    args = ap.parse_args()

    tests = [test_adown, test_siou, test_namespace_injection,
             test_efficient_uavdet_surgery]
    failures: list[str] = []
    for fn in tests:
        try:
            fn()
        except Exception:  # noqa: BLE001
            failures.append(f"{fn.__name__}:\n{traceback.format_exc()}")
    try:
        test_end_to_end_build(args.fast)
    except Exception:  # noqa: BLE001
        failures.append(f"test_end_to_end_build:\n{traceback.format_exc()}")

    print(f"\n通过 {len(PASSED)} 项，跳过 {len(SKIPPED)} 项，失败 {len(failures)} 项\n")
    for label in PASSED:
        print(f"  ok    {label}")
    for label in SKIPPED:
        print(f"  skip  {label}")
    for fail in failures:
        print(f"\n{fail}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
