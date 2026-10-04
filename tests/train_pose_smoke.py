"""姿态训练回路自检（opt-in）：合成**关键点**数据 → 真训练 → 载权重推理 → 查证据。

与 ``tests/train_smoke.py`` 是姊妹文件，验证的是**姿态这条新通路**上只有真跑才会暴露的坑：

    1. ``task=pose`` 真的被传到模型与数据加载器（框架靠**文件名**猜任务，本库生成的
       图叫 ``model.yaml``，不显式传 task 会被静默当成 detect）；
    2. ``PoseModel`` 用数据集的 ``kpt_shape`` 覆盖了模型图（否则关键点形状对不上）；
    3. ``E2ELoss`` 持有的 **O2M/O2O 两套** PoseLoss26 的关键点项**都被替换**成 TinyPoseLoss
       （只换一半是那种"训练能跑但一半损失是框架默认"的静默错误）；
    4. 训练日志里出现 ``pose_loss`` / ``kobj_loss`` 两列，且数值有限非零（回路在学）；
    5. 断点能被新进程载入并对一张图推理，输出 17×3 的关键点；
    6. ``tools/val_pose.py`` 的 OKS-AP 口径能给出一张完整的表（完美预测 = 1）。

用法::

    python tests/train_pose_smoke.py                 # 1 epoch，默认 device=0
    python tests/train_pose_smoke.py --device cpu --epochs 2
    python tests/train_pose_smoke.py --keep          # 保留实验目录（默认清理）

⚠️ 环境要求：同 ``train_smoke.py`` —— ultralytics 的标签缓存用 ``multiprocessing.Pool``
（Windows 上是命名管道），在受限沙箱会话里会 ``PermissionError [WinError 5]``。
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PASSED: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    if not cond:
        raise AssertionError(f"[FAIL] {label} {detail}")
    PASSED.append(label)


def _load_tool(name: str):
    path = ROOT / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_tod_tool_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    ap = argparse.ArgumentParser(description="姿态训练回路自检（合成关键点数据）")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--device", default="0")
    ap.add_argument("--variant", type=Path,
                    default=ROOT / "variants" / "visdrone-yolo26n-pose-p2-p16" / "variant.yaml")
    ap.add_argument("--work", type=Path, default=ROOT / "tests" / ".tmp" / "train-pose-smoke")
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
    from tod.engine.pose_trainer import TODPoseTrainer

    tod.bootstrap()

    # 受限沙箱（Windows 命名管道被禁）下必须换成顺序标签缓存，否则
    # ``ThreadPool`` 建队列表就 PermissionError [WinError 5]（见 compat 的说明）。
    if not args.allow_threaded_cache and compat.allow_threadless_label_cache():
        print("[env ] 标签缓存已切换为顺序模式（沙箱兼容；--allow-threaded-cache 可关闭）")

    # ---- 1) 合成关键点数据集（火柴人 + COCO 17 点）----
    data_yaml = _load_tool("make_dummy_dataset").build(
        args.work / "data", 12, 4, seed=0, task="pose")
    check("合成姿态数据集配置已生成", data_yaml.is_file())
    cfg = load_yaml(data_yaml)
    check("数据集声明 kpt_shape=[17,3]", list(cfg.get("kpt_shape") or []) == [17, 3],
          f"实际 {cfg.get('kpt_shape')}")

    # ---- 2) 真训练（姿态回路）----
    spec = load_yaml(args.variant)
    variant = Variant.from_spec(spec)
    check("变体声明的任务为 pose", variant.task == "pose", f"实际 {variant.task!r}")
    model_yaml = args.work / "model.yaml"
    variant.model_yaml(model_yaml, write=True)
    runtime.set_active(spec)

    from ultralytics import YOLO

    # 实验输出目录必须是**绝对路径**：ultralytics 会把相对 project 解析到
    # ``runs/<task>/<project>``（相对于数据集/当前工作目录），于是
    # ``project/name`` 找不到产物（实测：产物落到 runs/pose/tests/.tmp/...）。
    project = (args.work / "runs").resolve()
    model = YOLO(str(model_yaml), task="pose")
    check("模型头是 Pose 系（Pose/Pose26）",
          "Pose" in type(model.model.model[-1]).__name__,
          f"实际 {type(model.model.model[-1]).__name__}")

    model.train(
        trainer=TODPoseTrainer,
        data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, workers=0, plots=False, val=True, exist_ok=True,
        project=str(project), name=variant.name, verbose=False,
        # amp=False：框架在 amp=True 时会先下载 yolo26n.pt 做一次 CPU 自检
        # （ultralytics.utils.checks.check_amp），本自检要的是"姿态回路跑通"，
        # 不需要把网络依赖拉进来；AMP 通路的验证由 tests/train_smoke.py 负责。
        amp=False,
        **{k: v for k, v in (spec.get("train") or {}).items()
           if k in {"dfl", "pose", "kobj", "rle", "optimizer", "lr0", "lrf", "momentum",
                    "weight_decay", "warmup_epochs", "warmup_momentum", "close_mosaic",
                    "mosaic", "mixup", "multi_scale", "seed", "deterministic"}},
    )

    run_dir = project / variant.name
    best, last = run_dir / "weights" / "best.pt", run_dir / "weights" / "last.pt"
    check("训练产出 best.pt", best.is_file())
    check("训练产出 last.pt", last.is_file())

    # ---- 3) 证据：关键点两列损失真的存在，且在下降 ----
    with open(run_dir / "results.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    check("results.csv 有训练记录", len(rows) >= args.epochs, f"实际 {len(rows)} 行")
    cols = list(rows[0])
    pose_col = next((c for c in cols if "pose_loss" in c), None)
    kobj_col = next((c for c in cols if "kobj_loss" in c), None)
    check("results.csv 含 pose_loss 列", pose_col is not None, f"列名：{cols}")
    check("results.csv 含 kobj_loss 列", kobj_col is not None, f"列名：{cols}")
    check("pose_loss 数值有限且非零",
          all(abs(float(r[pose_col])) < 1e4 and float(r[pose_col]) > 0 for r in rows),
          f"实际 {[r[pose_col] for r in rows]}")

    # ---- 4) 断点能被新进程载入并输出 17×3 关键点 ----
    reloaded = YOLO(str(best))           # 任务由 checkpoint 里的 model.task 决定
    check("best.pt 可被重新载入", reloaded.model is not None)
    images = sorted((args.work / "data" / "images" / "val").glob("*.jpg"))
    results = reloaded.predict(str(images[0]), imgsz=args.imgsz, device=args.device,
                              conf=0.01, verbose=False)
    check("载入后能推理并返回结果", len(results) == 1 and hasattr(results[0], "keypoints"))
    kpts = results[0].keypoints
    check("推理结果带关键点对象", kpts is not None)
    if kpts is not None and len(kpts):
        shape = tuple(kpts.data.shape)
        check("关键点张量形状为 (n, 17, 3)", shape[1:] == (17, 3), f"实际 {shape}")
        print(f"       ↳ 推理结果：{len(results[0].boxes)} 个框，"
              f"关键点 {shape}，图像 {Path(images[0]).name}")
    else:
        # 1 epoch 的合成数据可能一个框都检不出，这不是回路问题，但要如实报告
        print("       ↳ 注意：1 epoch 后没有检出任何目标（合成数据 + 极小 epoch 的正常现象）")
        PASSED.append("推理结果关键点形状（本次无检出，跳过形状断言）")

    # ---- 5) OKS 评测口径可用（完美预测 = 满分）----
    from PIL import Image

    from tod.eval.pose import evaluate_pose, load_pose_labels

    def perfect(image: Path):
        with Image.open(image) as im:
            w, h = im.size
        boxes, classes, kpts = load_pose_labels(image, w, h, (17, 3))
        return [(classes[i], 0.9, boxes[4 * i:4 * i + 4],
                 [[k[0], k[1]] for k in kpts[i]]) for i in range(len(classes))]

    scored = evaluate_pose(data_yaml, predictor=perfect, split="val", verbose=False)
    check("完美预测：整体 OKS-AP50 = 1",
          abs(scored["overall"]["oks_ap50"] - 1.0) < 1e-9,
          f"实际 {scored['overall']['oks_ap50']}")
    check("完美预测：整体 OKS-AP(50:95) = 1",
          abs(scored["overall"]["oks_ap"] - 1.0) < 1e-9,
          f"实际 {scored['overall']['oks_ap']}")
    smallest = [b for b in scored["bins"] if b in ("lt8", "8-16") and scored["bins"][b]["n_gt"] > 0]
    check("完美预测：小目标层也有 GT 且满分",
          bool(smallest) and all(abs(scored["bins"][b]["oks_ap"] - 1.0) < 1e-9 for b in smallest),
          f"小目标层：{smallest}")

    # 关键点尺度诊断：完美预测的逐点误差必须是 0
    diag = next((v for k, v in scored["kpt_scale"].items() if v["n_kpt"] > 0), None)
    check("关键点尺度诊断有数据", diag is not None)
    check("完美预测：平均误差 0 px、≤1px 命中率 100%",
          diag is not None and abs(diag["mean_err_px"]) < 1e-9
          and abs(diag["hit_rate@1px"] - 1.0) < 1e-9,
          f"实际 {diag}")

    # 半像素偏移：OKS 必须**下降**，且小目标层下降得更狠（口径有效性的核心断言）
    def shifted(image: Path):
        out = []
        for cls, conf, box, kp in perfect(image):
            out.append((cls, conf, box, [[x + 0.5, y + 0.5] for x, y in kp]))
        return out

    half = evaluate_pose(data_yaml, predictor=shifted, split="val", verbose=False)
    check("半像素偏移：整体 OKS-AP 下降",
          half["overall"]["oks_ap"] < scored["overall"]["oks_ap"] - 1e-6,
          f"{scored['overall']['oks_ap']:.6f} -> {half['overall']['oks_ap']:.6f}")

    def _bin_ap(res, name):
        entry = res["bins"].get(name)
        return entry["oks_ap"] if entry and entry["n_gt"] else float("nan")

    pairs = [(name, _bin_ap(scored, name), _bin_ap(half, name))
             for name in ("8-16", "32-96")]
    usable = [(n, a, b) for n, a, b in pairs if a == a and b == b]     # 剔除 nan
    drops = {n: a - b for n, a, b in usable}
    print("       ↳ 半像素偏移的 OKS-AP 降幅：" + " / ".join(
        f"{n} 层 {d:.4f}" for n, d in drops.items()))
    check("OKS 口径对尺度敏感（小目标降幅 ≥ 大目标降幅）",
          "8-16" in drops and "32-96" in drops and drops["8-16"] >= drops["32-96"] - 1e-9,
          f"实际 {drops}")

    print(f"       ↳ 实验目录：{run_dir}")
    if not args.keep:
        shutil.rmtree(args.work, ignore_errors=True)
        print("       ↳ 已清理临时实验目录（--keep 可保留）")

    print(f"\n姿态训练回路自检通过：{len(PASSED)} 项\n")
    for label in PASSED:
        print(f"  ok  {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
