"""姿态关键点模块测试：损失数值 / sigma 策略 / OKS 口径 / 建图 / 换头手术。

需要 torch + ultralytics（不需要数据集与训练；真训练自检在 ``tests/train_pose_smoke.py``）。

运行::

    python tests/test_pose.py            # 全部
    python tests/test_pose.py --fast     # 跳过端到端建图（只跑数值）
"""

from __future__ import annotations

import argparse
import math
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


def _need(pkg: str = "torch"):
    import importlib.util

    if importlib.util.find_spec(pkg) is None:
        return None
    return __import__(pkg)


def _need_framework() -> bool:
    from tod import compat

    if not compat.installed():
        return False
    compat.ensure_runtime_env()
    return True


# ---------------------------------------------------------------- EP7 关键点损失


def test_tiny_pose_loss_vs_framework() -> None:
    """本库 TinyPoseLoss 在 person/auto 策略下必须与框架逐元素一致。"""
    torch = _need()
    if torch is None:
        return skip("TinyPoseLoss", "未安装 torch")
    if not _need_framework():
        return skip("TinyPoseLoss 对照", "未安装 ultralytics")

    from ultralytics.utils.loss import OKS_SIGMA, KeypointLoss

    from tod.loss.pose import COCO_SIGMA_17, TinyPoseLoss, sigma_tensor

    # 容差 1e-7：框架的 OKS_SIGMA 是 numpy float32，字面量 0.026 存成 0.025999999
    check("本库 COCO sigma 与框架 OKS_SIGMA 一致（float32 精度内）",
          all(abs(float(a) - float(b)) < 1e-7 for a, b in zip(COCO_SIGMA_17, OKS_SIGMA)),
          f"本库 {COCO_SIGMA_17[:4]} vs 框架 {list(OKS_SIGMA)[:4]}")

    torch.manual_seed(0)
    pred = torch.rand(8, 17, 3)
    gt = pred + torch.randn(8, 17, 3) * 0.05
    mask = torch.ones(8, 17, dtype=torch.bool)
    area = torch.rand(8, 1) + 0.01

    reference = KeypointLoss(sigmas=torch.tensor(OKS_SIGMA))
    expected = float(reference(pred, gt, mask, area))

    for strategy in ("person", "framework", "auto"):
        sigma, _note = sigma_tensor((17, 3), strategy)
        value = float(TinyPoseLoss(sigma, strategy=strategy)(pred, gt, mask, area))
        check(f"TinyPoseLoss[{strategy}] 与框架数值一致",
              abs(value - expected) < 1e-9, f"{value:.8f} vs {expected:.8f}")

    # 位置误差为 0 时损失为 0；误差越大损失越大且趋于 1（OKS 的有界性）
    zero = float(TinyPoseLoss(torch.tensor(COCO_SIGMA_17))(gt, gt, mask, area))
    check("关键点完全重合 → 损失 0", abs(zero) < 1e-12, f"实际 {zero}")
    sigma = torch.tensor(COCO_SIGMA_17)
    far = float(TinyPoseLoss(sigma)(gt + 1e3, gt, mask, area))
    check("关键点极远 → 损失趋于 1（有界）", 0.99 < far <= 1.0, f"实际 {far}")

    # 可见性掩码：被掩掉的点必须**完全不贡献**损失。
    # 注意两个反直觉之处（都是框架原式的行为，本库照抄）：
    #   * 掩掉点后因子 N/Σvis 变大（可见点越少、单点权重越大），所以"掩掉误差为 0 的点"
    #     反而会让损失**上升**；
    #   * 极小框 + 大偏差时 1-exp(-e) 饱和到 1，此时怎么掩都是 1。
    # 因此这里用"只在被掩点上报错"来测：若掩码失效，损失会立刻变正。
    mask_part = mask.clone()
    mask_part[:, 9:] = False
    area_mid = torch.full((8, 1), 0.01)
    pred_hidden = gt.clone()
    pred_hidden[:, 9:, :2] += 0.3                     # 只在 9..16 号点上制造大误差
    hidden_loss = float(TinyPoseLoss(sigma)(pred_hidden, gt, mask_part, area_mid))
    check("被掩掉的点不贡献损失（只在被掩点报错 → 损失为 0）",
          hidden_loss < 1e-9, f"实际 {hidden_loss:.8f}")
    visible_loss = float(TinyPoseLoss(sigma)(pred_hidden, gt, mask, area_mid))
    check("取消掩码后同一误差立刻产生损失", visible_loss > 0.1,
          f"实际 {visible_loss:.6f}")

    # 极小框 + 大偏差 → 损失饱和到 1：这就是"框架 OKS 在 tiny 目标上近似全或无"的证据
    saturated = float(TinyPoseLoss(sigma)(gt + 0.2, gt, mask, torch.full((8, 1), 0.001)))
    check("极小框上 OKS 项饱和到 ≈1（本库要暴露的 tiny 现象）",
          saturated > 0.999, f"实际 {saturated:.6f}")

    # balanced 策略：改变绝对尺度但不得产生 nan / inf，且 σ 全为正
    bal, note = sigma_tensor((17, 3), "balanced")
    check("balanced σ 全为正", bool((bal > 0).all()))
    check("balanced 说明里标注了尺度改变", "尺度" in note, note)
    value = float(TinyPoseLoss(bal)(pred, gt, mask, area))
    check("balanced 损失有限", math.isfinite(value), f"实际 {value}")


def test_sigma_strategies() -> None:
    torch = _need()
    if torch is None:
        return skip("sigma 策略", "未安装 torch")

    from tod.loss.pose import COCO_SIGMA_MEAN, sigma_tensor

    # 非 17 点：没有公认标准值 → 用自研几何 sigma，但量级必须与 COCO 对齐
    sigma, note = sigma_tensor((6, 2), "auto")
    check("非 17 点 auto 给出 6 个 sigma", sigma.numel() == 6)
    check("非 17 点 auto 的均值对齐 COCO sigma 量级",
          abs(float(sigma.mean()) - COCO_SIGMA_MEAN) < 1e-6,
          f"{float(sigma.mean()):.6f} vs {COCO_SIGMA_MEAN:.6f}")
    check("非 17 点 auto 说明里标注为推断", "非论文参数" in note, note)
    check("两端 σ 大于中间（形状自适应）",
          float(sigma[0]) > float(sigma[2]) and float(sigma[-1]) > float(sigma[2]))

    # person 策略在非 17 点上不应假装有 COCO 值
    sigma2, note2 = sigma_tensor((4, 3), "person")
    check("非 17 点的 person 回退为 auto", "auto" in note2, note2)

    # min_sigma 下限：极小 sigma 被夹住（防止 (2σ)²·area 下溢导致饱和）
    from tod.loss.pose import TinyPoseLoss

    tiny = TinyPoseLoss(torch.tensor([0.0, 1e-9, 0.5]), min_sigma=1e-3)
    check("σ 被夹到 min_sigma 以上", float(tiny.sigmas.min()) >= 1e-3,
          f"实际 {tiny.sigmas.tolist()}")
    check("describe 可读", "TinyPoseLoss(strategy=" in tiny.describe())


def test_pose_criteria_detection() -> None:
    """单分支 / E2ELoss 双分支都必须能被识别（否则只替换一半会静默漏改）。"""
    torch = _need()
    if torch is None:
        return skip("pose_criteria", "未安装 torch")

    from tod.loss.pose import pose_criteria

    class _Branch:
        def __init__(self):
            self.keypoint_loss = "kpt"

    class _Single:
        keypoint_loss = "kpt"

    class _E2E:
        def __init__(self):
            self.one2many = _Branch()
            self.one2one = _Branch()

    check("单分支返回 1 个", len(pose_criteria(_Single())) == 1)
    check("E2ELoss 双分支返回 2 个", len(pose_criteria(_E2E())) == 2)

    from tod.compat import CompatError

    try:
        pose_criteria(object())
        raise AssertionError("[FAIL] 非姿态准则未报错")
    except CompatError as exc:
        check("非姿态准则报可操作的错", "keypoint_loss" in str(exc), str(exc)[:100])


# ---------------------------------------------------------------- EP8 OKS 指标


def test_oks_metric() -> None:
    torch = _need()
    if torch is None and _need("math") is None:  # math 恒在；这里只是保持结构一致
        return skip("OKS", "未安装 torch")

    from tod.loss.pose import COCO_SIGMA_17
    from tod.eval.pose import oks

    box = (0.0, 0.0, 20.0, 20.0)                      # 20×20 px 的小目标
    kpts = [[5.0 + i, 5.0 + i, 2.0] for i in range(17)]

    check("完全命中 → OKS = 1", abs(oks(kpts, kpts, box, COCO_SIGMA_17) - 1.0) < 1e-9)

    shifted = [[x + 1.0, y, v] for x, y, v in kpts]
    value = oks(shifted, kpts, box, COCO_SIGMA_17)
    check("1px 偏移 → OKS 下降", 0.0 < value < 1.0, f"实际 {value:.4f}")

    # 尺度敏感性：同样的像素误差，小目标上的 OKS 必须低于大目标（口径的核心）
    big_box = (0.0, 0.0, 200.0, 200.0)
    small_oks = oks(shifted, kpts, box, COCO_SIGMA_17)
    big_oks = oks(shifted, kpts, big_box, COCO_SIGMA_17)
    check("同一像素误差：小目标的 OKS 低于大目标",
          small_oks < big_oks, f"小 {small_oks:.4f} vs 大 {big_oks:.4f}")

    # vis=0 的点被忽略：**GT** 标为 vis=0 的点即使预测离得极远也不影响 OKS
    # （掩码读的是 GT 的可见性，不是预测的 —— 这是 COCO 的口径）
    gt_with_hole = [row[:] for row in kpts]
    gt_with_hole[3] = [kpts[3][0], kpts[3][1], 0.0]
    far_pred = [row[:] for row in kpts]
    far_pred[3] = [999.0, 999.0, 2.0]
    check("GT 里 vis=0 的点不参与 OKS",
          abs(oks(far_pred, gt_with_hole, box, COCO_SIGMA_17) - 1.0) < 1e-9,
          f"实际 {oks(far_pred, gt_with_hole, box, COCO_SIGMA_17):.6f}")
    # 反例：同一个点在 GT 里 vis=2 时，跑偏就必须扣分
    check("GT 里 vis=2 的点跑偏会扣分",
          oks(far_pred, kpts, box, COCO_SIGMA_17) < 1.0 - 1e-6,
          f"实际 {oks(far_pred, kpts, box, COCO_SIGMA_17):.6f}")

    nan = oks(kpts, [[0.0, 0.0, 0.0]] * 17, box, COCO_SIGMA_17)
    check("全部 vis=0 → nan（该目标对 AP 无贡献）", math.isnan(nan))


def test_pose_label_io() -> None:
    from tod.eval.pose import load_pose_labels

    tmp = ROOT / "tests" / ".tmp" / "pose-label-io"
    (tmp / "images").mkdir(parents=True, exist_ok=True)
    (tmp / "labels").mkdir(parents=True, exist_ok=True)
    image = tmp / "images" / "a.jpg"
    image.write_bytes(b"")                            # 只用于推标签路径，不需要真图
    kpts = " ".join(f"{0.5 + 0.01 * i:.6f} {0.4 + 0.01 * i:.6f} 2.000000" for i in range(17))
    (tmp / "labels" / "a.txt").write_text(
        f"0 0.5 0.5 0.2 0.4 {kpts}\n", encoding="utf-8")

    boxes, classes, keypoints = load_pose_labels(image, 100, 200, (17, 3))
    check("读到 1 个目标", len(classes) == 1 and len(keypoints) == 1)
    check("框还原到像素", boxes[:4] == [40.0, 60.0, 60.0, 140.0], f"实际 {boxes[:4]}")
    check("关键点数量正确", len(keypoints[0]) == 17)
    check("关键点 x 还原到像素", abs(keypoints[0][0][0] - 50.0) < 1e-6,
          f"实际 {keypoints[0][0][0]}")
    check("关键点 y 还原到像素", abs(keypoints[0][0][1] - 80.0) < 1e-6,
          f"实际 {keypoints[0][0][1]}")
    check("可见性被保留", keypoints[0][0][2] == 2.0)


def test_pose_match_image() -> None:
    """姿态匹配器必须接受**四元组**预测（含关键点）—— 曾经复用三元组匹配器直接崩。"""
    from tod.eval.pose import match_pose_image

    gts = [{"cls": 0, "box": (0, 0, 10, 10), "side": 10.0, "kpts": [[1, 1, 2]] * 17},
           {"cls": 0, "box": (20, 20, 30, 30), "side": 10.0, "kpts": [[21, 21, 2]] * 17}]
    preds = [(0, 0.9, [0, 0, 10, 10], [[1, 1]] * 17),
             (0, 0.8, [21, 21, 31, 31], [[22, 22]] * 17),
             (1, 0.7, [0, 0, 10, 10], [[1, 1]] * 17)]

    matched = match_pose_image(preds, gts, 0.5)
    check("姿态匹配按置信度降序", [m[1] for m in matched] == [0.9, 0.8, 0.7])
    check("姿态匹配返回预测下标", [m[3] for m in matched] == [0, 1, 2],
          f"实际 {[m[3] for m in matched]}")
    check("姿态匹配命中对的 GT", matched[0][2] == 0 and matched[1][2] == 1)
    check("类别不同算 FP", matched[2][2] == -1)

    dup = match_pose_image([(0, 0.9, [0, 0, 10, 10], [[1, 1]] * 17),
                            (0, 0.8, [0, 0, 10, 10], [[1, 1]] * 17)], gts[:1], 0.5)
    check("同一 GT 不会被匹配两次", dup[1][2] == -1)


# ---------------------------------------------------------------- EP0 合成数据


def test_pose_dummy_dataset() -> None:
    import importlib.util

    if importlib.util.find_spec("PIL") is None or importlib.util.find_spec("numpy") is None:
        return skip("合成姿态数据集", "未安装 pillow/numpy")

    path = ROOT / "tools" / "make_dummy_dataset.py"
    spec = importlib.util.spec_from_file_location("_tod_tool_dummy_pose", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    out = ROOT / "tests" / ".tmp" / "pose-dummy"
    yaml_path = module.build(out, 2, 1, seed=1, task="pose")
    text = yaml_path.read_text(encoding="utf-8")
    check("dataset.yaml 写入了 kpt_shape", "kpt_shape: [17, 3]" in text, text[:200])
    check("dataset.yaml 写入了 kpt_names", "kpt_names" in text)

    label = (out / "labels" / "train" / "train_0000.txt").read_text(encoding="utf-8")
    first = label.splitlines()[0].split()
    check("标签行 = 5 + 17*3 个数", len(first) == 5 + 51, f"实际 {len(first)}")
    check("类别列为整数", first[0] in ("0", "1"), first[0])
    check("可见性列全为 2", all(abs(float(first[5 + i * 3 + 2]) - 2.0) < 1e-6
                              for i in range(17)))

    # 拒绝不支持的 kpt_shape（火柴人只有 COCO 17 点）
    try:
        module.build(out, 1, 1, task="pose", kpt_shape=(5, 2))
        raise AssertionError("[FAIL] 非 17 点的合成姿态未报错")
    except ValueError:
        PASSED.append("合成姿态拒绝非 17 点 kpt_shape")


# ------------------------------------------------------------- DSL / 建图 / 手术


def test_pose_variant_dsl() -> None:
    from tod.compose import Variant

    v = Variant("unit-pose", base="yolo26n").pose(kpt_shape=(17, 3)).model(nc=10)
    check("pose() 设置任务", v.task == "pose")
    check("pose() 写入 kpt_shape", list(v.model_cfg["kpt_shape"]) == [17, 3])

    data = v.spec()
    check("spec 记录 task", data["task"] == "pose")
    check("spec_version 已升版（新增 task 字段）", data["spec_version"] == 2)
    check("from_spec 往返保留 task", Variant.from_spec(data).task == "pose")

    try:
        Variant("bad", task="keypoint")
        raise AssertionError("[FAIL] 非法任务未报错")
    except ValueError:
        PASSED.append("非法 task 被拒绝")

    card = v.card(ROOT / "tests" / ".tmp" / "pose-card.md").read_text(encoding="utf-8")
    check("卡片含 task 行", "- **task**: `pose`" in card, card[:300])
    check("姿态卡片含 OKS 指标行", "OKS-AP" in card)


def test_pose_model_yaml(fast: bool = False) -> None:
    if fast:
        return skip("姿态建图", "--fast")

    from tod import compat

    if not compat.installed():
        return skip("姿态建图", "未安装 ultralytics")

    import torch
    from ultralytics import YOLO

    from tod.compose import Variant

    v = (Variant("unit-pose-build", base="yolo26n", task="pose")
         .pose(kpt_shape=(17, 3))
         .model(nc=10, add_p2=True, p2_idx=2, p2_channels=128, p2_fuse_block="C3"))
    cfg = v.model_yaml(write=False)

    check("姿态图保留 end2end=True（NMS-free 底座）", cfg.get("end2end") is True)
    check("姿态图保留 reg_max=1（DFL-free 底座）", cfg.get("reg_max") == 1)
    check("kpt_shape 写进模型图", list(cfg.get("kpt_shape")) == [17, 3])
    check("P2 已注入", cfg.get("_p2_status") == "injected")

    head_node = cfg["head"][-1]
    check("头类型是 Pose26", str(head_node[2]) == "Pose26", f"实际 {head_node[2]}")
    check("Pose26 的 args 未被 P2 注入破坏（仍是 [nc, kpt_shape]）",
          list(head_node[3]) == ["nc", "kpt_shape"], f"实际 {head_node[3]}")
    check("头输入为 4 路（P2-P5）", len(head_node[0]) == 4, f"实际 {head_node[0]}")

    from tod.compat import dump_yaml

    path = ROOT / "tests" / ".tmp" / "unit_pose_model.yaml"
    dump_yaml(cfg, path)
    model = YOLO(str(path), task="pose").model
    head = model.model[-1]
    check("真实建图为 Pose26", type(head).__name__ == "Pose26", type(head).__name__)
    check("nl = 4（P2-P5）", head.nl == 4, f"实际 {head.nl}")
    check("stride 含 4", [int(s) for s in head.stride] == [4, 8, 16, 32],
          f"实际 {[int(s) for s in head.stride]}")
    check("kpt_shape 生效", list(head.kpt_shape) == [17, 3], f"实际 {head.kpt_shape}")
    check("nk = 51", head.nk == 51, f"实际 {head.nk}")
    check("Pose26 有归一化流与 sigma 分支",
          hasattr(head, "flow_model") and hasattr(head, "cv4_sigma"))

    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 320, 320))
    check("姿态模型前向可跑通", out is not None)


def test_keypoint_head_surgery() -> None:
    """EP5 换头必须支持姿态头，且关键点分支可单独不压（消融开关）。"""
    torch = _need()
    if torch is None:
        return skip("关键点头手术", "未安装 torch")
    if not _need_framework():
        return skip("关键点头手术", "未安装 ultralytics")

    from ultralytics.nn.modules import Pose, Pose26

    from tod.modules.head.efficient_uavdet import swap_detect_head

    ch = (32, 64, 128, 256)

    # --- Pose（yolov8 系关键点分支是单条 cv4）---
    head = Pose(nc=10, kpt_shape=(17, 3), ch=ch)
    before = sum(p.numel() for p in head.parameters())
    swap_detect_head(head, per_group=16, channels="native")
    after = sum(p.numel() for p in head.parameters())
    check("Pose 换头后参数量下降", after < before, f"{before:,} -> {after:,}")
    check("Pose 的关键点分支也被替换（cv4[0][0] 是分组卷积）",
          head.cv4[0][0].__class__.__name__ == "GroupedStem", head.cv4[0][0].__class__.__name__)
    check("关键点输出通道保持 nk=51", head.cv4[0][-1].bias.numel() == 51,
          f"实际 {head.cv4[0][-1].bias.numel()}")

    # --- Pose26（关键点被拆成 cv4_kpts / cv4_sigma，单层 1x1 无 stem）---
    head26 = Pose26(nc=10, kpt_shape=(17, 3), ch=ch)
    n_before26 = sum(p.numel() for p in head26.parameters())
    swap_detect_head(head26, per_group=16, channels="native")
    n_after26 = sum(p.numel() for p in head26.parameters())
    check("Pose26 换头后参数量下降", n_after26 < n_before26,
          f"{n_before26:,} -> {n_after26:,}")
    check("Pose26 的关键点特征块 cv4 被替换",
          head26.cv4[0][0].__class__.__name__ == "GroupedStem",
          head26.cv4[0][0].__class__.__name__)
    check("Pose26 的单层 1x1 关键点/sigma 分支原样保留（没有 stem 可换）",
          head26.cv4_kpts[0].__class__.__name__ == "Conv2d"
          and head26.cv4_sigma[0].__class__.__name__ == "Conv2d")
    check("Pose26 关键点输出通道仍为 nk=51", head26.cv4_kpts[0].out_channels == 51,
          f"实际 {head26.cv4_kpts[0].out_channels}")
    check("Pose26 sigma 输出通道仍为 34", head26.cv4_sigma[0].out_channels == 34,
          f"实际 {head26.cv4_sigma[0].out_channels}")

    # --- 消融开关：keypoint_branches=False 时关键点特征块保持原生 ---
    head26b = Pose26(nc=10, kpt_shape=(17, 3), ch=ch)
    native_cv4 = head26b.cv4[0][0].__class__.__name__
    swap_detect_head(head26b, per_group=16, channels="native", keypoint_branches=False)
    check("keypoint_branches=False：关键点分支不被替换",
          head26b.cv4[0][0].__class__.__name__ == native_cv4,
          f"{native_cv4} -> {head26b.cv4[0][0].__class__.__name__}")
    check("keypoint_branches=False：框/分类分支仍被替换",
          head26b.cv2[0][0].__class__.__name__ == "GroupedStem")

    # Pose（yolov8 系）上同一个开关也必须真的有区别（cv4 是带 stem 的三层结构）
    head_pose_off = Pose(nc=10, kpt_shape=(17, 3), ch=ch)
    native_pose_cv4 = head_pose_off.cv4[0][0].__class__.__name__
    swap_detect_head(head_pose_off, keypoint_branches=False)
    check("Pose：keypoint_branches=False 时 cv4 stem 保持原生",
          head_pose_off.cv4[0][0].__class__.__name__ == native_pose_cv4,
          f"{native_pose_cv4} -> {head_pose_off.cv4[0][0].__class__.__name__}")

    # 前向：换完头仍能跑通并输出正确维度
    head26c = Pose26(nc=10, kpt_shape=(17, 3), ch=ch)
    swap_detect_head(head26c, per_group=16)
    head26c.train()
    feats = [torch.randn(1, c, s, s) for c, s in zip(ch, (40, 20, 10, 5))]
    out = head26c(feats)
    preds = out[1] if isinstance(out, tuple) else out
    if isinstance(preds, dict) and "one2many" in preds:
        preds = preds["one2many"]
    check("换头后前向可跑通且带 kpts", isinstance(preds, dict) and "kpts" in preds,
          f"键：{list(preds) if isinstance(preds, dict) else type(preds)}")
    if isinstance(preds, dict) and "kpts" in preds:
        check("前向 kpts 通道 = 51", preds["kpts"].shape[1] == 51,
              f"实际 {tuple(preds['kpts'].shape)}")


def test_pose_criterion_patching(fast: bool = False) -> None:
    """姿态准则：O2M/O2O 两套的关键点项与框损失项都必须被替换。"""
    if fast:
        return skip("姿态准则接线", "--fast")

    torch = _need()
    if torch is None:
        return skip("姿态准则接线", "未安装 torch")
    if not _need_framework():
        return skip("姿态准则接线", "未安装 ultralytics")

    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg

    from tod import runtime
    from tod.compat import dump_yaml
    from tod.compose import Variant
    from tod.loss.criterion import build_pose_criterion

    v = (Variant("unit-pose-crit", base="yolo26n", task="pose")
         .pose(kpt_shape=(17, 3))
         .model(nc=4, add_p2=True, p2_idx=2, p2_channels=128, p2_fuse_block="C3"))
    path = ROOT / "tests" / ".tmp" / "unit_pose_crit.yaml"
    dump_yaml(v.model_yaml(write=False), path)
    model = YOLO(str(path), task="pose").model
    args = get_cfg()
    args.pose, args.kobj, args.dfl = 12.0, 1.0, 0.0
    model.args = args

    ep7 = {"box": "wiou", "pose": "oks", "sigma_strategy": "person",
           "kind_kwargs": {"variant": 3}}
    criterion, report = build_pose_criterion(model, ep7=ep7, stal=None, use_dfl=False)
    check("准则类型为 E2ELoss（YOLO26 NMS-free 双分支）",
          type(criterion).__name__ == "E2ELoss", type(criterion).__name__)
    check("报告含框损失替换", any("bbox_loss -> wiou" in r for r in report), str(report))
    check("报告含关键点损失替换", any("TinyPoseLoss" in r for r in report), str(report))

    from tod.loss.pose import TinyPoseLoss

    for branch in ("one2many", "one2one"):
        sub = getattr(criterion, branch)
        check(f"{branch} 的关键点损失已替换为 TinyPoseLoss",
              isinstance(sub.keypoint_loss, TinyPoseLoss),
              type(sub.keypoint_loss).__name__)
        check(f"{branch} 的框损失已替换为 TODBboxLoss",
              type(sub.bbox_loss).__name__ == "TODBboxLoss", type(sub.bbox_loss).__name__)
    check("两套子准则共用同一个 TinyPoseLoss 实例",
          criterion.one2many.keypoint_loss is criterion.one2one.keypoint_loss)

    # pose=False → 保留框架原生 KeypointLoss（消融列）
    criterion2, report2 = build_pose_criterion(model, ep7={"pose": False}, stal=None)
    from ultralytics.utils.loss import KeypointLoss

    check("pose=False 时保留框架 KeypointLoss",
          isinstance(criterion2.one2many.keypoint_loss, KeypointLoss)
          and not isinstance(criterion2.one2many.keypoint_loss, TinyPoseLoss))
    check("pose=False 的报告写明未替换",
          any("未替换" in r or "保留" in r for r in report2), str(report2))
    check("pose=False 时框损失不被误替换（EP7 里没有 box 键）",
          not any("bbox_loss" in r for r in report2), str(report2))

    # 前向 + 反传：损失有限、梯度能到关键点分支
    runtime.set_active(v.spec())
    model.criterion = criterion
    model.train()
    batch = {
        "img": torch.rand(2, 3, 320, 320),
        "cls": torch.tensor([[0.0], [1.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.15, 0.2], [0.3, 0.3, 0.06, 0.09]]),
        "keypoints": torch.rand(2, 17, 3) * 0.4 + 0.3,
        "batch_idx": torch.tensor([0.0, 1.0]),
    }
    total, items = model(batch)
    loss_scalar = float(total.sum() if total.dim() else total)
    check("姿态训练前向损失有限", bool(torch.isfinite(total).all()),
          f"loss={loss_scalar:.4f}")
    check("姿态损失项含 pose/kobj 两项（非零）",
          float(items[1]) > 0 and float(items[2]) > 0,
          f"items={[round(float(x), 4) for x in items]}")
    check("dfl 增益为 0 → 该项为 0", abs(float(items[4])) < 1e-8,
          f"items={[round(float(x), 4) for x in items]}")
    total.sum().backward()
    head = model.model[-1]
    check("梯度到达关键点分支（cv4_kpts）",
          any(p.grad is not None for p in head.cv4_kpts.parameters()))
    check("梯度到达归一化流（flow_model）",
          any(p.grad is not None for p in head.flow_model.parameters()))


def test_pose_variant_recipe() -> None:
    """仓库里的姿态变体配方必须可物化、任务与关键点形状自洽。"""
    import importlib.util

    recipe = ROOT / "variants" / "visdrone-yolo26n-pose-p2-p16" / "recipe.py"
    if not recipe.is_file():
        return skip("姿态变体配方", "缺少 recipe.py")

    spec = importlib.util.spec_from_file_location("_tod_pose_recipe", recipe)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    variant = module.variant

    check("变体任务是 pose", variant.task == "pose", variant.task)
    check("变体声明 kpt_shape=(17,3)", tuple(variant.model_cfg.get("kpt_shape")) == (17, 3),
          str(variant.model_cfg.get("kpt_shape")))
    check("变体启用了 P2 头", bool(variant.model_cfg.get("add_p2")))
    check("EP7 显式启用 OKS 损失", variant.eps["EP7"].get("pose") == "oks",
          str(variant.eps["EP7"]))
    check("EP7 sigma 策略为 person（COCO 17 点标准）",
          variant.eps["EP7"].get("sigma_strategy") == "person")
    check("数据集为 visdrone2019-pose", variant.dataset == "visdrone2019-pose",
          variant.dataset)
    check("数据集占位配置存在",
          (ROOT / "configs" / "_base_" / "datasets" / "visdrone2019-pose.yaml").is_file())
    check("卡片里标注了数据集待准备", "待准备" in variant.notes or "占位" in variant.notes)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="跳过端到端建图与准则接线")
    args = ap.parse_args()

    tests = [test_tiny_pose_loss_vs_framework, test_sigma_strategies,
             test_pose_criteria_detection, test_oks_metric, test_pose_label_io,
             test_pose_match_image, test_pose_dummy_dataset, test_pose_variant_dsl,
             test_keypoint_head_surgery, test_pose_variant_recipe]
    failures: list[str] = []
    for fn in tests:
        try:
            fn()
        except Exception:  # noqa: BLE001
            failures.append(f"{fn.__name__}:\n{traceback.format_exc()}")
    for fn in (test_pose_model_yaml, test_pose_criterion_patching):
        try:
            fn(args.fast)
        except Exception:  # noqa: BLE001
            failures.append(f"{fn.__name__}:\n{traceback.format_exc()}")

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
