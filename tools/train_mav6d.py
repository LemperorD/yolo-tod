"""MAV6D 训练入口：SPAE-YOLOv8n + 9 点机体关键点（小目标检测 + 关键点检测）。

## 这是什么

一个把「MAV6D 数据集」直接喂给「SPAE + 关键点」变体的训练脚本，覆盖：

    * **小目标检测**：MAV6D 的目标只有画面长边的 10–17%（实测框长边归一化
      0.10–0.17 → 1920 宽下约 190–320 px），因此按 PLAN §1.1 的口径属于
      "小目标偏中等"；脚本默认按**小目标**配置高分辨率 + P2 头 + 关闭会伤小目标的增强。
    * **关键点检测**：9 点（1 个 3D 框中心 + 8 个角点），EP7 的 OKS 损失 +
      论文式 (8) 的 **L_rep 几何重投影损失**（前 15 epoch 只用 2D 项，见 paper-notes §3）。

## 用法

    # 0) 先转换数据（会现场标定；phantom4 的标定已被 --check-util 验证）
    python3 tools/mav6d_to_yolo.py --mav6d /path/to/MAV6D --model phantom4 --out data/mav6d

    # 1) 结构自检（不需要数据，几秒钟）
    python3 tools/train_mav6d.py --model phantom4 --dry-run

    # 2) 只跑 2 个 epoch 做冒烟（确认数据/损失/显存都没问题）
    python3 tools/train_mav6d.py --model phantom4 --smoke

    # 3) 正式训练
    python3 tools/train_mav6d.py --model phantom4 --epochs 100 --batch 8 --imgsz 1280

    # 4) 两个机型分别训练（论文 Table IV 显示跨机型零样本很差，建议分别训）
    python3 tools/train_mav6d.py --model mavic2 --epochs 100 --batch 8 --imgsz 1280

    # 5) 断点续训
    python3 tools/train_mav6d.py --model phantom4 --resume results/mav6d-phantom4/weights/last.pt

## 为什么默认这些超参（都可覆盖）

| 参数 | 默认 | 理由 |
|---|---|---|
| `imgsz` | 1280 | MAV6D 原图 1920×1080；目标只占 10–17% 长边。640 会让 190 px 的目标缩到 63 px，直接损失关键点定位精度。1280 在 8 GB 卡上配 batch 4–8 可跑。 |
| `batch` | 4 | 1280 + P2 头的显存约束（PLAN §11.1）；`--batch 8` 视显存可上。 |
| `mosaic` | 0.0 | **关键点任务不建议 mosaic**：拼接会改变目标尺度分布，而关键点回归依赖尺度一致性；且 MAV6D 单目标、无遮挡，mosaic 的收益很低。 |
| `close_mosaic` | 0 | 与上面配套（mosaic 本来就关了）。 |
| `degrees` | 10.0 | 论文 KDA 用了 ±30°；本库保守取 10°，并**必须**配好 `flip_idx`（已写在 dataset.yaml）。 |
| `translate` | 0.1 | 论文 KDA 用 ±10 px（≈0.005 归一化）；这里给 0.1 与框架默认一致。 |
| `fliplr` | 0.5 | dataset.yaml 里已显式给了 `flip_idx`，框架不会静默关掉它（见 docs/KEYPOINTS.md §3）。 |
| `scale` | 0.4 | 让网络见到稍大/稍小的目标；再大就会破坏关键点的像素尺度一致性。 |
| `hsv_*` | 框架默认 | 论文 KDA 的亮度/对比度/色调/饱和度抖动范围已在注释里给出，可对齐。 |
| `optimizer` | auto | 框架会在 SGD/AdamW 间自动选（本库实测 1280 下会选 AdamW）。 |
| `workers` | 4 | Windows 下建议 ≤8；沙箱环境请用 `--set workers=0`。 |

## 与论文的差异（必读）

* 论文用 **CPKL-Net**（1×1 卷积 + EfficientRep + Rep-PAN + MFMF + 网格三分支），
  本脚本用 **YOLOv8 Pose 头**承载关键点 —— **结构不等价，不能宣称复现其精度**。
* 论文 2D 损失是像素 MSE，本库用 OKS；论文的 6D 位姿/10°10cm 指标需要 PnP，本脚本不产出。
* MAV6D 是**真实室内 VICON 采集**（不是仿真），域差异与 RflySim 不同。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
# Windows 控制台/重定向默认是 GBK，切到 UTF-8，免得中文与符号输出直接报错
try:
    import tod

    tod.configure_stdio()
except Exception:
    pass

#: 机型 → 变体 spec 与**注册表里的数据集名**（都可用命令行覆盖）。
#: 数据集位置由 configs/datasets.yaml 决定（换盘只改注册表，不改这里）：
#:   mav6d-phantom4 / mav6d-mavic2 → <data_root>/mav6d/<机型>
MODEL_DEFAULTS = {
    "phantom4": {
        "variant": ROOT / "variants" / "spae-yolov8n-pose" / "variant.yaml",
        "data": "mav6d-phantom4",
        "name": "mav6d-phantom4",
    },
    "mavic2": {
        "variant": ROOT / "variants" / "spae-yolov8n-pose" / "variant.yaml",
        "data": "mav6d-mavic2",
        "name": "mav6d-mavic2",
    },
}

#: 默认训练超参（见模块 docstring 的理由表）。
DEFAULT_TRAIN = dict(
    imgsz=1280,
    batch=4,
    epochs=100,
    # 关键点任务：关 mosaic（改动尺度分布），保留几何/颜色抖动
    mosaic=0.0,
    close_mosaic=0,
    copy_paste=0.0,
    mixup=0.0,
    degrees=10.0,
    translate=0.1,
    scale=0.4,
    fliplr=0.5,
    flipud=0.0,
    # 关键点损失增益（与变体里一致，显式写出便于调）
    pose=12.0,
    kobj=1.0,
    amp=True,
    workers=4,
    seed=0,
    deterministic=True,
    project=str(ROOT / "results"),
    plots=True,
    val=True,
    save_period=-1,
)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="MAV6D 训练（SPAE-YOLOv8n + 9 点关键点；检测 + 关键点）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--model", choices=tuple(MODEL_DEFAULTS), required=True,
                    help="MAV6D 机型（两机型要**分别**训练，见论文 Table IV）")
    ap.add_argument("--data", default=None, metavar="名字|目录|YAML",
                    help="数据集：默认取注册表里的 mav6d-<机型>，"
                         "也可写数据集目录或 dataset.yaml 路径")
    ap.add_argument("--data-root", default=None, metavar="DIR",
                    help="覆盖注册表里的数据根目录（换盘/临时切换用；"
                         "也可用环境变量 TOD_DATA_ROOT）")
    ap.add_argument("--datasets", type=Path, default=None, metavar="YAML",
                    help="数据集注册表 YAML（默认 configs/datasets.yaml）")
    ap.add_argument("--variant", type=Path, default=None, help="变体 spec（默认 SPAE-pose）")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--device", default=None, help="如 0 / cpu")
    ap.add_argument("--name", default=None, help="实验名（默认 mav6d-<机型>）")
    ap.add_argument("--resume", type=Path, default=None, help="从 checkpoint 续训")
    ap.add_argument("--smoke", action="store_true",
                    help="冒烟：2 epoch + 小分辨率，只为确认回路能跑")
    ap.add_argument("--dry-run", action="store_true", help="只建模型/准则自检，不训练")
    ap.add_argument("--no-pretrained", action="store_true",
                    help="不用 COCO/ImageNet 预训练（默认会用 yolov8n-pose.pt）")
    ap.add_argument("--set", action="append", default=[], metavar="K=V",
                    help="覆盖任意训练参数（可重复），如 --set workers=0")
    ap.add_argument("--allow-threaded-cache", action="store_true",
                    help="保留框架默认并行标签缓存（沙箱下会 WinError 5）")
    return ap


def _coerce(text: str):
    import ast

    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def _reproj_enabled(spec: dict) -> bool:
    """变体是否启用了 L_rep（EP7.reproj）。"""
    flag = ((spec.get("eps") or {}).get("EP7") or {}).get("reproj", False)
    return not (flag is False or flag is None
                or str(flag).lower() in {"", "off", "none", "false"})


def explicit_set_keys(items: list[str]) -> set[str]:
    """``--set K=V`` 里显式给出的键（这些优先于脚本施加的安全默认）。"""
    return {item.split("=", 1)[0].strip() for item in items if "=" in item}


def main() -> int:
    args = build_parser().parse_args()

    import tod
    from tod import compat, runtime
    from tod.compat import CompatError, load_yaml
    from tod.compose import Variant
    from tod.data.pose6d import pose_safe_augment_overrides
    from tod.paths import DatasetPathError, resolve

    variant_path = args.variant or MODEL_DEFAULTS[args.model]["variant"]
    if not variant_path.is_file():
        raise SystemExit(f"变体 spec 不存在：{variant_path}")

    # ---- 训练集位置：注册表（configs/datasets.yaml）说了算 ----
    try:
        data_paths = resolve(args.data or MODEL_DEFAULTS[args.model]["data"],
                             registry_path_hint=args.datasets, data_root=args.data_root,
                             require_train=not args.dry_run)
    except DatasetPathError as exc:
        if not args.dry_run:
            raise SystemExit(
                f"[数据] {exc}\n"
                "换盘后请把 configs/datasets.yaml 的 data_root 指向新硬盘，或先重新转换数据：\n"
                f"  python3 tools/mav6d_to_yolo.py --mav6d /path/to/MAV6D "
                f"--model {args.model} --out <新硬盘>/mav6d"
            ) from exc
        print(f"[warn ] 数据集未就绪（--dry-run 继续）：{exc}")
        data_paths = None
    if data_paths is not None:
        for line in data_paths.report():
            print(line)
    data_path = data_paths.yaml if data_paths is not None else (
        ROOT / "data" / "mav6d" / args.model / "dataset.yaml")
    if not args.dry_run and not data_path.is_file():
        raise SystemExit(
            f"数据集不存在：{data_path}\n"
            "请先转换数据：\n"
            f"  python3 tools/mav6d_to_yolo.py --mav6d /path/to/MAV6D --model {args.model} "
            "--out <新硬盘>/mav6d"
        )

    if not args.allow_threaded_cache and compat.allow_threadless_label_cache():
        print("[env ] 标签缓存已切换为顺序模式（沙箱兼容；--allow-threaded-cache 可关闭）")

    spec = load_yaml(variant_path)
    variant = Variant.from_spec(spec)
    if data_path.is_file():
        # L_rep 需要数据集侧的相机内参与机体 3D 关键点
        spec["dataset_cfg"] = load_yaml(data_path)
    runtime.set_active(spec)

    tod.bootstrap()
    from ultralytics import YOLO

    trainer_cls = _task_trainer(variant.task)
    model_yaml = (variant_path.parent / "model.yaml")
    variant.model_yaml(model_yaml, write=True)

    # 预训练权重：MAV6D 是真实小样本域，从姿态预训练起步比从零快很多。
    weights = None
    if not args.no_pretrained and not args.dry_run:
        weights = _pretrained_for(variant.base)

    print(f"[mav6d] 机型={args.model} | 任务={variant.task} | 变体={variant.name}")
    print(f"[mav6d] 数据={data_paths.root if data_paths else data_path.parent}")
    print(f"[mav6d] 预训练={weights or '不用（从零训）'}")

    if args.dry_run:
        return _dry_run(model_yaml, spec, variant, data_path, args)

    # ---- 组装训练参数：变体给的 + MAV6D 默认的 + 命令行覆盖 ----
    train_args = dict(spec.get("train") or {})
    # 变体里的 imgsz/batch 是按别的数据集定的（640/4），MAV6D 用自己的默认
    for key, value in DEFAULT_TRAIN.items():
        train_args.setdefault(key, value)
    if args.smoke:
        train_args.update(epochs=2, imgsz=640, batch=2, workers=0, plots=False)
    for key, value in (("epochs", args.epochs), ("batch", args.batch),
                       ("imgsz", args.imgsz), ("device", args.device)):
        if value is not None:
            train_args[key] = value
    for item in args.set:
        if "=" not in item:
            raise SystemExit(f"--set 需要 K=V 形式，收到 {item!r}")
        key, value = item.split("=", 1)
        train_args[key.strip()] = _coerce(value.strip())

    train_args["data"] = str(data_path)
    train_args["name"] = args.name or MODEL_DEFAULTS[args.model]["name"]
    project = train_args.get("project")
    if not project or not Path(str(project)).is_absolute():
        train_args["project"] = str(ROOT / (str(project) if project else "results"))

    # ---- L_rep 与"会丢目标的几何增强"互斥（见 pose_safe_augment_overrides）----
    # 启用 6D 位姿监督时，框与位姿必须保持一一对应；而 RandomPerspective 会把
    # 变换后面积占比 <10% 的框连同类丢掉（实测 MAV6D 有 0.68% 的图被丢成 0 框），
    # 那之后就无法认定"CSV 的哪几行还对应得上"。命令行显式 --set 的优先级更高。
    if _reproj_enabled(spec):
        safe = pose_safe_augment_overrides()
        forced = {k: v for k, v in safe.items() if k not in explicit_set_keys(args.set)}
        train_args.update(forced)
        print(f"[mav6d] L_rep 启用 -> 关闭会丢目标的几何增强："
              f"{', '.join(f'{k}={v}' for k, v in forced.items())}")
        print("         （要保留某项就显式 --set 覆盖它，自担位姿错配风险）")

    train_args = {k: v for k, v in train_args.items() if v is not None}

    print(f"[mav6d] imgsz={train_args.get('imgsz')} batch={train_args.get('batch')} "
          f"epochs={train_args.get('epochs')} | mosaic={train_args.get('mosaic')} "
          f"degrees={train_args.get('degrees')} scale={train_args.get('scale')}")
    print(f"[mav6d] 关键点：EP7.pose={runtime.get('eps.EP7.pose')}、"
          f"L_rep={runtime.get('eps.EP7.reproj')}、"
          f"warm-up={runtime.get('dataset_cfg.tod.reproj_warmup_epochs')} epoch")

    model = YOLO(str(model_yaml), task=variant.task)
    if args.resume:
        if not args.resume.is_file():
            raise SystemExit(f"checkpoint 不存在：{args.resume}")
        model = YOLO(str(args.resume), task=variant.task)
        train_args["resume"] = True
    model.train(trainer=trainer_cls, **train_args)
    return 0


def _task_trainer(task: str):
    from tod.engine.pose_trainer import TODPoseTrainer
    from tod.engine.trainer import TODDetectionTrainer

    return {"pose": TODPoseTrainer, "detect": TODDetectionTrainer}.get(
        task, TODDetectionTrainer)


def _pretrained_for(base: str) -> str | None:
    """按底座名找一个可用的预训练权重（找不到就返回 None，让框架自己决定）。"""
    candidate = f"{base}-pose.pt" if "yolo" in base else None
    if candidate is None:
        return None
    root = ROOT / candidate
    if root.is_file():
        return candidate            # 框架会从当前目录/缓存加载
    return candidate                # 交给框架尝试下载


def _dry_run(model_yaml: Path, spec: dict, variant, data_path: Path,
             args) -> int:
    """结构自检：建图 + 手术 + 准则 + 一次前向，不需要数据集内容。"""
    import torch
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg

    from tod.compat import dump_yaml
    from tod.engine.surgery import apply_spec
    from tod.loss.criterion import build_pose_criterion, build_reproj_criterion

    cfg = variant.model_yaml(write=False)
    dump_yaml(cfg, model_yaml)
    model = YOLO(str(model_yaml), task=variant.task).model
    n0 = sum(p.numel() for p in model.parameters())
    applied = apply_spec(model, spec)
    n1 = sum(p.numel() for p in model.parameters())
    args_ns = get_cfg()
    for key, value in (spec.get("train") or {}).items():
        if hasattr(args_ns, key) and not isinstance(value, dict):
            setattr(args_ns, key, value)
    model.args = args_ns
    criterion, report = build_pose_criterion(model, ep7=(spec.get("eps") or {}).get("EP7"))
    criterion = build_reproj_criterion(criterion, spec)
    report = list(getattr(criterion, "_tod_patched", report))
    model.eval()
    imgsz = args.imgsz or 1280
    with torch.no_grad():
        model(torch.zeros(1, 3, imgsz, imgsz))
    head = model.model[-1]
    print("[自检] task=%s imgsz=%d 参数量=%s%s"
          % (variant.task, imgsz, f"{n1:,}",
             f"（手术 {n1 - n0:+,}）" if applied else ""))
    print(f"[自检] nl={getattr(head, 'nl', '?')} stride={[int(s) for s in head.stride]}")
    print(f"[自检] kpt_shape={getattr(head, 'kpt_shape', '?')} nk={getattr(head, 'nk', '?')}")
    print("[自检] 准则：" + ("；".join(report) or "框架默认"))
    if hasattr(criterion, "active"):
        print(f"[自检] L_rep：权重={criterion.weight} warm-up={criterion.warmup_epochs} epoch")
    print(f"[自检] 数据配置：{data_path}（存在={data_path.is_file()}）")
    return 0


if __name__ == "__main__":
    from tod.compat import CompatError as _CompatError

    try:
        raise SystemExit(main())
    except _CompatError as exc:
        raise SystemExit(f"[环境错误] {exc}") from exc
