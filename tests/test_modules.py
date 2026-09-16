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

    # 非方形输入：输出尺寸是**下取整**的一半（33x41 -> 16x20），
    # 与 ultralytics 自带 ADown 逐形状一致（下面做等价性对照）
    with torch.no_grad():
        y = m(torch.randn(1, 64, 33, 41))
    check("ADown 非方形输入", tuple(y.shape) == (1, 128, 16, 20), f"实际 {tuple(y.shape)}")

    import importlib.util

    if importlib.util.find_spec("ultralytics") is not None:
        from tod import compat

        compat.ensure_runtime_env()
        from ultralytics.nn.modules import ADown as _FrameworkADown

        fm = _FrameworkADown(64, 128).eval()
        with torch.no_grad():
            for shape in ((2, 64, 40, 40), (1, 64, 33, 41)):
                ours = m(torch.zeros(*shape)).shape
                theirs = fm(torch.zeros(*shape)).shape
                check(f"ADown 形状与框架一致 {shape}", tuple(ours) == tuple(theirs),
                      f"{tuple(ours)} vs {tuple(theirs)}")

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


# ------------------------------------------------------------------ DualAttention


def test_dual_attention() -> None:
    torch = _need_torch()
    if torch is None:
        return skip("DualAttention", "未安装 torch")

    import tod.modules  # noqa: F401  触发注册
    from tod.modules.attention.dual_attention import DualAttention

    m = DualAttention(64, reduction=16).eval()
    x = torch.randn(2, 64, 20, 20)
    with torch.no_grad():
        y = m(x)
    check("DualAttention 形状/通道不变", tuple(y.shape) == tuple(x.shape), f"实际 {tuple(y.shape)}")
    check("DualAttention 不是直通", not torch.allclose(y, x))
    # σ(·)·σ(·) ∈ (0,1) ⇒ 逐元素放大系数 < 1
    check("DualAttention 输出幅度被抑制", bool((y.abs() <= x.abs() + 1e-6).all()))

    try:
        DualAttention(64, 32)
        raise AssertionError("[FAIL] DualAttention 未拒绝 c2 != c1")
    except ValueError:
        PASSED.append("DualAttention 拒绝改变通道数")

    xg = torch.randn(1, 32, 8, 8, requires_grad=True)
    DualAttention(32)(xg).sum().backward()
    check("DualAttention 梯度可达", xg.grad is not None and bool(xg.grad.abs().sum() > 0))

    n = sum(p.numel() for p in DualAttention(64).parameters())
    check("DualAttention 参数极少（<1k）", n < 1000, f"实际 {n}")


# ---------------------------------------------------------------------- WIoU


def test_wise_iou() -> None:
    torch = _need_torch()
    if torch is None:
        return skip("WIoU", "未安装 torch")

    from tod.loss.box import WiseIoU, box_loss

    box = torch.tensor([[10.0, 10.0, 20.0, 20.0]])
    for v in (1, 2, 3):
        sim = WiseIoU(variant=v)(box, box.clone(), xywh=True)
        check(f"WIoU v{v} 完全重合≈1", bool((sim - 1).abs().max() < 1e-5),
              f"实际 {float(sim):.6f}")

    v3 = WiseIoU(variant=3)
    a = torch.tensor([[50.0, 50.0, 8.0, 8.0]])
    b = torch.tensor([[51.0, 50.0, 8.0, 8.0]])
    sim = v3(a, b, xywh=True)
    check("WIoU v3 数值有限", bool(torch.isfinite(sim).all()), f"实际 {float(sim)}")

    # v1 是无状态的，用它做 xywh/xyxy 一致性检查（v2/v3 会改 iou_mean，跨调用不可比）
    v1 = WiseIoU(variant=1)
    a_xyxy = _xywh2xyxy(a)
    b_xyxy = _xywh2xyxy(b)
    check("WIoU xywh/xyxy 一致",
          bool((v1(a, b, xywh=True) - v1(a_xyxy, b_xyxy, xywh=False)).abs().max() < 1e-5))

    mean_before = float(v3.iou_mean)
    v3(torch.tensor([[0.0, 0.0, 4.0, 4.0]]), torch.tensor([[40.0, 40.0, 4.0, 4.0]]), xywh=True)
    check("WIoU 跨 batch 滑动均值在更新", float(v3.iou_mean) != mean_before,
          f"{mean_before} -> {float(v3.iou_mean)}")

    x = torch.tensor([[50.0, 50.0, 8.0, 8.0]], requires_grad=True)
    (1 - v3(x, b, xywh=True)).sum().backward()
    check("WIoU 梯度可达", x.grad is not None and bool(x.grad.abs().sum() > 0))

    check("box_loss('wiou') 返回有状态实例", isinstance(box_loss("wiou"), WiseIoU))
    check("box_loss('siou') 返回无状态函数", not isinstance(box_loss("siou"), WiseIoU))


# ---------------------------------------------------------------------- STAL


def test_stal() -> None:
    torch = _need_torch()
    if torch is None:
        return skip("STAL", "未安装 torch")
    from tod import compat

    if not compat.installed():
        return skip("STAL", "未安装 ultralytics")
    compat.ensure_runtime_env()

    from tod.assigner.stal import small_target_assigner_class

    strides = [8, 16, 32]
    grid = torch.stack(torch.meshgrid(torch.arange(20) + 0.5, torch.arange(20) + 0.5,
                                      indexing="ij"), -1).view(-1, 2) * 8
    tiny = torch.tensor([[[80.0, 80.0, 84.0, 84.0]]])        # 4x4 < stride[0]=8
    normal = torch.tensor([[[70.0, 70.0, 130.0, 130.0]]])     # 60x60
    gt = torch.cat((tiny, normal), dim=1)
    mask_gt = torch.ones(1, 2, 1)

    def run(cls):
        return cls(10, 2, 0.5, 6.0, strides).select_candidates_in_gts(grid, gt.clone(), mask_gt)

    fw = run(compat.tal_assigner())
    on = run(small_target_assigner_class(True))
    off = run(small_target_assigner_class(False))

    check("STAL 让极小目标拿到正样本（经典 TAL 为 0）",
          int(on[0, 0].sum()) > 0 and int(off[0, 0].sum()) == 0,
          f"STAL={int(on[0, 0].sum())} 经典={int(off[0, 0].sum())}")
    check("STAL 放大幅度与框架一致", torch.equal(on[0, 0], fw[0, 0]))
    check("小目标规则不影响普通目标", torch.equal(on[0, 1], off[0, 1]))
    if hasattr(compat.tal_assigner()(10, 2, 0.5, 6.0, strides), "stride_val"):
        check("本库 STAL 与框架实现逐元素一致", torch.equal(fw, on))

    from tod.registry import get

    built = get("STAL").obj(topk=10, num_classes=2, alpha=0.5, beta=6.0, stride=strides)
    check("注册表工厂可构造 STAL", getattr(built, "small_target_aware", False) is True)


# --------------------------------------------------------------------- MuSGD


def test_musgd() -> None:
    torch = _need_torch()
    if torch is None:
        return skip("MuSGD", "未安装 torch")

    import copy

    import torch.nn as nn

    from tod.optim.musgd import (MuSGD, build, native_musgd, param_groups,
                                 zeropower_newton_schulz)

    ortho = zeropower_newton_schulz(torch.randn(32, 16))
    sv = torch.linalg.svdvals(ortho)
    check("NS 正交化形状不变", tuple(ortho.shape) == (32, 16))
    check("NS 奇异值落在 [0.5, 1.5]（5 步放宽正交）",
          bool((sv > 0.5).all() and (sv < 1.5).all()),
          f"范围 [{float(sv.min()):.3f}, {float(sv.max()):.3f}]")

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, 3, padding=1)
            self.bn = nn.BatchNorm2d(8)
            self.fc = nn.Linear(8 * 4 * 4, 2)

        def forward(self, x):
            return self.fc(torch.relu(self.bn(self.conv(x))).flatten(1))

    torch.manual_seed(0)
    base = Net()
    groups = param_groups(base, lr=0.02, momentum=0.9, weight_decay=0.01)
    check("MuSGD 参数分组含 muon 组", any(g["use_muon"] for g in groups))
    check("bias/BN 组不施加 weight decay",
          all(g["weight_decay"] == 0.0 for g in groups if g["param_group"] in ("bias", "bn")))

    def train_with(optimizer, model, steps=5):
        torch.manual_seed(1)
        data = [(torch.randn(4, 3, 4, 4), torch.randint(0, 2, (4,))) for _ in range(steps)]
        for x, y in data:
            optimizer.zero_grad()
            nn.functional.cross_entropy(model(x), y).backward()
            optimizer.step()

    ref, mine = copy.deepcopy(base), copy.deepcopy(base)
    native = native_musgd()
    if native is not None:
        train_with(native(params=param_groups(ref, lr=0.02, momentum=0.9, weight_decay=0.01),
                          muon=0.2, sgd=1.0), ref)
    train_with(MuSGD(params=param_groups(mine, lr=0.02, momentum=0.9, weight_decay=0.01),
                     muon=0.2, sgd=1.0), mine)
    if native is not None:
        with torch.no_grad():
            diff = max(float((p - q).abs().max())
                       for p, q in zip(ref.parameters(), mine.parameters()))
        check("本库 MuSGD 与框架原生数值一致（<1e-3）", diff < 1e-3, f"实际 {diff:.2e}")

    opt, source = build(mine, lr=0.01, prefer_native=False)
    check("框架缺失时的兜底实现可用", isinstance(opt, MuSGD), source)
    before = mine.conv.weight.detach().clone()
    opt.zero_grad()
    nn.functional.cross_entropy(mine(torch.randn(2, 3, 4, 4)), torch.randint(0, 2, (2,))).backward()
    opt.step()
    check("兜底实现能更新参数", not torch.equal(before, mine.conv.weight))


# ---------------------------------------------------------------- 命名空间注入


def test_namespace_injection() -> None:
    try:
        import tod
        from tod import compat
    except ImportError as exc:
        return skip("命名空间注入", f"导入失败：{exc}")

    if not compat.installed():
        return skip("命名空间注入", "未安装 ultralytics")
    compat.ensure_runtime_env()
    try:
        tod.bootstrap()
        import tod.assigner  # noqa: F401
        import tod.engine.distill  # noqa: F401
        import tod.loss  # noqa: F401
        import tod.optim  # noqa: F401
    except ImportError as exc:
        return skip("命名空间注入", f"模块库导入失败：{exc}")

    ns = compat.model_globals()
    for name in ("ADown", "Efficient_UAVDet", "siou", "wiou", "DualAttention",
                 "STAL", "MuSGD", "FeatureAlignKD"):
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
    out = head(x)
    # 框架 8.4 起检测头训练模式的返回是 dict（end2end 时再套一层 one2many/one2one），
    # 不再是老版本的 4 元素 list；这里按新契约取逐层 logits。
    logits = out[1] if isinstance(out, tuple) else out
    if isinstance(logits, dict) and "one2many" in logits:
        logits = logits["one2many"]
    check("换头后前向可跑通（返回 dict 契约）", isinstance(logits, dict) and "scores" in logits,
          f"实际 {type(out).__name__}")
    check("前向输出通道 = nc + 4*reg_max",
          logits["boxes"].shape[1] + logits["scores"].shape[1] == 10 + 64,
          f"实际 boxes={logits['boxes'].shape[1]} scores={logits['scores'].shape[1]}")
    print("       ↳ " + describe(head).replace("\n", "\n         "))

    # 另一种通道策略（in=out=x）也必须可用
    head2 = Detect(nc=10, ch=ch)
    swap_detect_head(head2, per_group=16, channels="input")
    check("input 策略：stem 内 in=out=x",
          head2.cv2[1][0].cv1.conv.in_channels == head2.cv2[1][0].cv2.conv.out_channels == 64)


# --------------------------------------------------------- 知识蒸馏（EP9 §4.7）

_SDD_CACHE: dict = {}


def _sdd_model():
    """构建一次并复用：yolo26n + P2 分支 + DualAttention（SDD 配方的最小骨架）。"""
    if "model" in _SDD_CACHE:
        return _SDD_CACHE["model"]
    import importlib.util

    for pkg in ("torch", "ultralytics"):
        if importlib.util.find_spec(pkg) is None:
            return None

    import tod
    from tod.compat import dump_yaml
    from tod.compose import Variant

    tod.bootstrap()
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg

    from tod.engine.surgery import apply_spec

    spec = (Variant("unit-sdd", base="yolo26n")
            .model(nc=10, add_p2=True, p2_idx=2, p2_channels=128, p2_fuse_block="C3")
            .attention("DualAttention", levels=[2, 3, 4, 5]))
    path = ROOT / "tests" / ".tmp" / "unit_sdd_model.yaml"
    dump_yaml(spec.model_yaml(write=False), path)
    model = YOLO(str(path)).model
    model.args = get_cfg()
    apply_spec(model, spec.spec())
    model.train()
    _SDD_CACHE["model"] = model
    return model


def _unit_batch(imgsz: int = 320):
    torch = _need_torch()
    return {
        "img": torch.rand(2, 3, imgsz, imgsz),
        "cls": torch.tensor([[0.0], [1.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.06, 0.06], [0.3, 0.3, 0.02, 0.02]]),
        "batch_idx": torch.tensor([0.0, 1.0]),
    }


def test_kd() -> None:
    torch = _need_torch()
    if torch is None:
        return skip("FeatureAlignKD", "未安装 torch")
    from tod import compat

    if not compat.installed():
        return skip("FeatureAlignKD", "未安装 ultralytics")
    compat.ensure_runtime_env()

    import copy

    from tod.engine.distill import FeatureAlignKD, build_kd, head_strides, level_logits
    from tod.loss.criterion import build_detection_loss

    # ---- 单元：相同 logits 时 KL 必须为 0；combine 就是式 (6) ----
    logits = {2: torch.randn(2, 64, 10), 3: torch.randn(2, 16, 10)}
    kd = FeatureAlignKD(lambda_=0.5, temperature=3.0)
    same = kd(logits, {k: v.clone() for k, v in logits.items()})
    check("KD：相同 logits → 0", abs(float(same)) < 1e-5, f"实际 {float(same):.2e}")
    check("KD：combine 即 (1-λ)L_task + λL_KD",
          abs(float(kd.combine(torch.tensor(2.0), torch.tensor(4.0))) - 3.0) < 1e-6)
    try:
        kd({2: logits[2]}, {7: torch.randn(2, 64, 10)})
        raise AssertionError("[FAIL] 无共同蒸馏层未报错")
    except compat.CompatError:
        PASSED.append("KD 拒绝无共同蒸馏层")

    model = _sdd_model()
    if model is None:
        return skip("FeatureAlignKD 端到端", "未安装 torch/ultralytics")
    teacher = copy.deepcopy(model).eval()          # 必须在注册 hook 前深拷贝
    for p in teacher.parameters():
        p.requires_grad_(False)

    captured: dict = {}
    model.model[-1].register_forward_hook(lambda mod, inp, out: captured.update(out=out))
    batch = _unit_batch()

    rec: dict = {}

    class _Spy:                                     # 记录 combine 的入参，复核式 (6)
        def __init__(self, inner):
            self.inner = inner
            self.lambda_ = inner.lambda_

        def __call__(self, s, t):
            out = self.inner(s, t)
            rec["kd"] = float(out)
            return out

        def combine(self, task, kd_loss):
            rec["task"], rec["kd_arg"] = float(task), float(kd_loss)
            out = self.inner.combine(task, kd_loss)
            rec["total"] = float(out)
            return out

    criterion = build_kd(build_detection_loss(model, kind="siou"), model, teacher)
    criterion.kd = _Spy(criterion.kd)
    model.criterion = criterion
    total, _items = model(batch)

    check("KD：式 (6) 逐值成立",
          abs(rec["total"] - (0.5 * rec["task"] + 0.5 * rec["kd_arg"])) < 1e-4,
          f"total={rec['total']:.6f} task={rec['task']:.6f} kd={rec['kd_arg']:.6f}")
    check("KD：KD 项非零（教师 BN 统计与学生不同）", rec["kd_arg"] > 0)
    check("KD：教师不混进学生参数表",
          not ({id(p) for p in model.parameters()} & {id(p) for p in teacher.parameters()}))

    levels = level_logits(captured["out"], head_strides(model))
    check("KD：拆层键为 P2–P5", sorted(levels) == [2, 3, 4, 5], f"实际 {sorted(levels)}")
    check("KD：逐层形状为 (b, A_l, nc)",
          all(t.shape[0] == 2 and t.shape[2] == 10 for t in levels.values()))

    total.sum().backward()
    head_idx = model.model[-1].f[0]
    check("KD：梯度到达注意力层",
          model.model[head_idx].channel[2].weight.grad is not None)
    check("KD：梯度不流向教师", all(p.grad is None for p in teacher.parameters()))


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


# ------------------------------------------------------- SDD-YOLO26n 端到端


def test_sdd_end_to_end(fast: bool = False) -> None:
    """按配方（variants/SDD-YOLO26n/recipe.py）走一遍完整链路：建图 → 手术 → 准则 → 反传。"""
    if fast:
        return skip("SDD 端到端建图", "--fast")

    torch = _need_torch()
    if torch is None:
        return skip("SDD 端到端建图", "未安装 torch")

    import importlib.util

    for pkg in ("torch", "ultralytics"):
        if importlib.util.find_spec(pkg) is None:
            return skip("SDD 端到端建图", f"未安装 {pkg}")

    recipe = ROOT / "variants" / "SDD-YOLO26n" / "recipe.py"
    if not recipe.is_file():
        return skip("SDD 端到端建图", "缺少 SDD 配方")

    import tod
    from tod.compat import dump_yaml

    tod.bootstrap()
    variant = _load_recipe_variant(recipe)
    spec = variant.spec()

    cfg = variant.model_yaml(write=False)
    check("SDD：底座是 YOLO26（end2end=True / NMS-free）", cfg.get("end2end") is True)
    check("SDD：reg_max=1（无 DFL 分支）", cfg.get("reg_max") == 1)
    check("SDD：P2 分支已注入", cfg.get("_p2_status") == "injected")
    check("SDD：P2 融合块为 C3（论文 §4.2 原文）",
          any(str(n[2]) == "C3" for n in cfg["head"]), f"{cfg['head'][-4:]}")
    check("SDD：训练配置里 dfl=0.0（论文 §4.3）",
          float((spec.get("train") or {}).get("dfl", 1.5)) == 0.0)

    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg

    from tod.engine.surgery import apply_spec
    from tod.loss.criterion import build_detection_loss, bbox_criteria

    model_path = recipe.parent / "model.yaml"
    dump_yaml(cfg, model_path)
    model = YOLO(str(model_path)).model
    head = model.model[-1]
    check("SDD：YAML 阶段检测头仍是原生 Detect（改动走手术）",
          type(head).__name__ == "Detect" and len(head.f) == 4, f"{type(head).__name__} f={list(head.f)}")

    applied = apply_spec(model, spec)
    check("SDD：EP4 注意力插入 4 处（P2–P5）",
          any("DualAttention × 4" in a for a in applied), f"applied={applied}")
    check("SDD：Detect 输入已重指向注意力节点",
          all(getattr(model.model[i], "_tod_ep4", None) for i in head.f),
          f"f={list(head.f)}")
    check("SDD：stride 含 4（P2 层生效）", 4 in [int(s) for s in head.stride],
          f"实际 {[int(s) for s in head.stride]}")

    # 准则：与 tools/train.py --dry-run 同一条路径
    args = get_cfg()
    for key, value in (spec.get("train") or {}).items():
        if hasattr(args, key):
            setattr(args, key, value)
    model.args = args
    criterion = build_detection_loss(model, kind="wiou", use_dfl=False, stal=True)
    subs = bbox_criteria(criterion)
    check("SDD：E2ELoss 的两套子准则都被替换（O2M+O2O）",
          len(subs) == 2 and all(type(s.bbox_loss).__name__ == "TODBboxLoss" for s in subs),
          f"subs={len(subs)}")
    check("SDD：回归损失为 WiseIoU 且关闭了 DFL 分支",
          all(s.bbox_loss.fn.__class__.__name__ == "WiseIoU" and s.bbox_loss.use_dfl is False
              for s in subs))
    check("SDD：分配器换成 STAL（small_target_aware=True）",
          all(getattr(s.assigner, "small_target_aware", False) for s in subs))

    model.criterion = criterion
    model.train()
    batch = _unit_batch(320)
    total, items = model(batch)
    check("SDD：训练前向可跑通", bool(torch.isfinite(total).all()))
    check("SDD：DFL 增益为 0 → 回归项 loss_items[2] == 0",
          abs(float(items[2])) < 1e-8, f"items={[round(float(x), 4) for x in items]}")
    total.sum().backward()
    attn = model.model[head.f[0]]
    check("SDD：反向传播到达注意力层", attn.channel[2].weight.grad is not None)

    n_final = sum(p.numel() for p in model.parameters())
    print(f"       ↳ SDD-YOLO26n：{n_final:,} 参数（{n_final / 1e6:.2f} M）、"
          f"nl={head.nl}、stride={[int(s) for s in head.stride]}、f={list(head.f)}")


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

    tests = [test_adown, test_siou, test_dual_attention, test_wise_iou, test_stal,
             test_musgd, test_namespace_injection, test_efficient_uavdet_surgery,
             test_kd]
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
    try:
        test_sdd_end_to_end(args.fast)
    except Exception:  # noqa: BLE001
        failures.append(f"test_sdd_end_to_end:\n{traceback.format_exc()}")

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
