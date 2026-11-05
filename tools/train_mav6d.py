"""MAV6D 训练入口：SPAE-YOLOv8n + 9 点机体关键点（小目标检测 + 关键点检测）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    import tod

    tod.configure_stdio()
except Exception:
    pass

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

DEFAULT_TRAIN = dict(
    imgsz=1280,
    batch=4,
    epochs=100,
    mosaic=0.0,
    close_mosaic=0,
    copy_paste=0.0,
    mixup=0.0,
    degrees=10.0,
    translate=0.1,
    scale=0.4,
    fliplr=0.5,
    flipud=0.0,
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

MAV6D_OVERRIDES = (
    "imgsz", "batch", "epochs", "mosaic", "close_mosaic", "copy_paste", "mixup",
    "degrees", "translate", "scale", "fliplr", "flipud", "pose", "kobj",
    "amp", "workers", "seed", "deterministic", "project", "plots", "val",
    "save_period",
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
    ap.add_argument("--no-lrep", action="store_true",
                    help="关闭 EP7.reproj（2D baseline；几何增强保持开启）")
    ap.add_argument("--no-pretrained", action="store_true",
                    help="兼容保留；当前自定义结构始终从零训练（不自动迁移预训练）")
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
    if args.no_lrep:
        ep7 = spec.setdefault("eps", {}).setdefault("EP7", {})
        ep7["reproj"] = False
        print("[mav6d] --no-lrep：EP7.reproj=False（2D baseline，几何增强保持开启）")
    variant = Variant.from_spec(spec)
    if data_path.is_file():
        spec["dataset_cfg"] = load_yaml(data_path)
        if args.smoke:
            spec["dataset_cfg"].setdefault("tod", {})["reproj_warmup_epochs"] = 0
            print("[mav6d] --smoke：L_rep warm-up 强制为 0（2 个 epoch 内会真正算几何项）")
        coverage = (spec["dataset_cfg"].get("tod") or {}).get("pose_coverage")
        if _reproj_enabled(spec) and coverage is not None and float(coverage) < 0.5:
            print(f"[mav6d] 警告：6D 位姿覆盖率只有 {float(coverage):.1%}，"
                  "L_rep 实际几乎不生效；建议先跑 --no-lrep，"
                  "或用 --pose-from-pnp --fit-pnp-shape 重拟合形状。")
    runtime.set_active(spec)

    tod.bootstrap()
    from ultralytics import YOLO

    trainer_cls = _task_trainer(variant.task)
    model_yaml = (variant_path.parent / "model.yaml")
    variant.model_yaml(model_yaml, write=True)

    print(f"[mav6d] 机型={args.model} | 任务={variant.task} | 变体={variant.name}")
    print(f"[mav6d] 数据={data_paths.root if data_paths else data_path.parent}")
    print("[mav6d] 预训练=不自动加载（自定义结构；从零训练，见 docs/TRAINING_REVIEW.md P1-1）")

    if args.dry_run:
        return _dry_run(model_yaml, spec, variant, data_path, args)

    train_args = dict(spec.get("train") or {})
    train_args.update({k: DEFAULT_TRAIN[k] for k in MAV6D_OVERRIDES
                       if k in DEFAULT_TRAIN})
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
    train_args["pretrained"] = False
    project = train_args.get("project")
    if not project or not Path(str(project)).is_absolute():
        train_args["project"] = str(ROOT / (str(project) if project else "results"))

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
