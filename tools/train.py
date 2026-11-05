"""训练入口：读变体 spec → 注入魔改模块 → 生成模型图 → 交给 ultralytics 训练。"""

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


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="TOD 训练入口")
    ap.add_argument("--variant", type=Path, default=None, help="变体 spec YAML")
    ap.add_argument("--data", default=None, metavar="名字|目录|YAML",
                    help="训练集：注册表里的名字（如 visdrone2019-det）、数据集根目录，"
                         "或数据集 YAML 路径（覆盖变体设置）")
    ap.add_argument("--data-root", default=None, metavar="DIR",
                    help="覆盖注册表里的数据根目录（换盘/临时切换用；"
                         "也可用环境变量 TOD_DATA_ROOT）")
    ap.add_argument("--datasets", type=Path, default=None, metavar="YAML",
                    help="数据集注册表 YAML（默认 configs/datasets.yaml）")
    ap.add_argument("--list-datasets", action="store_true",
                    help="打印注册表里的数据集与解析结果后退出（不训练）")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--device", default=None, help="如 0 或 cpu")
    ap.add_argument("--name", default=None, help="实验名，默认用变体 id")
    ap.add_argument("--set", action="append", default=[], metavar="K=V",
                    help="覆盖任意训练参数（可重复），如 --set workers=0 --set plots=False")
    ap.add_argument("--dry-run", action="store_true", help="只建模型并前向一次，不训练")
    ap.add_argument("--no-lrep", action="store_true",
                    help="关闭 EP7.reproj（2D baseline；几何增强保持开启）")
    ap.add_argument("--allow-threaded-cache", action="store_true",
                    help="保留框架默认的并行标签缓存（受限沙箱下会 WinError 5）")
    return ap


def _coerce(text: str):
    """把 ``--set`` 的字符串值转成 int/float/bool/None，失败则原样保留字符串。"""
    import ast

    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def _reproj_enabled(spec: dict) -> bool:
    """变体是否启用了 L_rep（``EP7.reproj``）。"""
    flag = ((spec.get("eps") or {}).get("EP7") or {}).get("reproj", False)
    return not (flag is False or flag is None
                or str(flag).lower() in {"", "off", "none", "false"})


def main() -> int:
    args = build_parser().parse_args()

    import tod
    from tod import runtime
    from tod.compat import CompatError, load_yaml
    from tod.compose import Variant
    from tod.paths import describe

    if args.list_datasets:
        for line in describe(registry_path_hint=args.datasets):
            print(line)
        print("\n提示：改 configs/datasets.yaml 的 data_root（或 configs/datasets.local.yaml）"
              "即可换盘；临时换盘用 --data-root 或环境变量 TOD_DATA_ROOT。")
        return 0

    if args.variant is None:
        raise SystemExit("缺少 --variant（或改用 --list-datasets 只看数据集注册表）")
    if not args.variant.is_file():
        raise SystemExit(f"变体文件不存在：{args.variant}")
    spec = load_yaml(args.variant)
    if args.no_lrep:
        spec.setdefault("eps", {}).setdefault("EP7", {})["reproj"] = False
        print("[train] --no-lrep：EP7.reproj=False（2D baseline）")
    variant = Variant.from_spec(spec)

    out_dir = args.variant.parent
    model_path = out_dir / "model.yaml"
    cfg = variant.model_yaml(model_path, write=True)
    print(f"[模型] {model_path}")
    print(f"       P2 分支：{cfg.get('_p2_status', '未启用')}；"
          f"下采样替换：{cfg.get('_downsample_replaced') or '无'}；"
          f"节点类型替换：{cfg.get('_type_map_applied') or '无'}")

    resolved = _resolve_data(args.data or variant.dataset, args,
                             task=str(spec.get("task") or "detect"),
                             require=not args.dry_run)

    try:
        tod.bootstrap()
    except ImportError as exc:
        raise SystemExit(
            f"载入魔改模块失败：{exc}\n"
            "请确认已完成环境安装（ultralytics + torch）。"
        ) from exc

    runtime.set_active(spec)

    def _attach_dataset_cfg(data_path) -> None:
        if not data_path:
            return
        try:
            spec["dataset_cfg"] = load_yaml(data_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn ] 未能读取数据集配置（L_rep 将不可用）：{exc}")

    _attach_dataset_cfg(resolved.yaml if resolved is not None else None)

    if not args.allow_threaded_cache:
        from tod.compat import allow_threadless_label_cache

        if allow_threadless_label_cache():
            print("[env ] 标签缓存已切换为顺序模式（沙箱兼容；--allow-threaded-cache 可关闭）")

    task = str(spec.get("task") or "detect")
    if task not in _task_trainers():
        raise SystemExit(
            f"变体 {variant.name} 的 task={task!r} 还没有训练回路（可选 {sorted(TASK_TRAINERS)}）。"
            "本库目前接了 detect 与 pose 两条；segment/obb 需要各自的 Validator 与准则接线。"
        )

    from ultralytics import YOLO

    model = YOLO(str(model_path), task=task)

    if args.dry_run:
        import torch
        from ultralytics.cfg import get_cfg

        from tod.engine.surgery import apply_spec
        from tod.loss.criterion import (build_detection_loss, build_pose_criterion,
                                       build_reproj_criterion)

        imgsz = args.imgsz or int(spec.get("data", {}).get("imgsz", 640))
        n_before = sum(p.numel() for p in model.model.parameters())
        applied = apply_spec(model.model, spec)
        if applied:
            print("[EP4/EP5] 建模后手术：")
            for line in applied:
                print(f"       {line}")
        n_after = sum(p.numel() for p in model.model.parameters())

        eps = spec.get("eps") or {}
        ep6, ep7, ep9 = (eps.get("EP6") or {}), (eps.get("EP7") or {}), (eps.get("EP9") or {})
        model.model.args = _dry_run_args(spec, get_cfg)

        from tod.engine.trainer import stal_flag

        dfl_gain = float(getattr(model.model.args, "dfl", 1.5) or 0.0)
        box_kind = ep7.get("box")
        use_dfl = False if (dfl_gain == 0.0 and box_kind) else None
        if task == "pose":
            criterion, report = build_pose_criterion(
                model.model, ep7=ep7, stal=stal_flag(ep6), use_dfl=use_dfl,
            )
            criterion = build_reproj_criterion(criterion, spec)
            report = list(getattr(criterion, "_tod_patched", report))
        else:
            criterion = build_detection_loss(
                model.model, kind=box_kind,
                theta=float(ep7.get("theta", 4.0) or 4.0), use_dfl=use_dfl,
                stal=stal_flag(ep6), **(ep7.get("kind_kwargs") or {}),
            )
            report = list(getattr(criterion, "_tod_patched", []))
        print("[EP6/EP7] 训练准则：" + ("；".join(report) or "框架默认（未做替换）"))
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
        print(f"[自检] task={task}  imgsz={imgsz}  参数量={n_after:,} ({n_after / 1e6:.2f} M)"
              + (f"  手术影响 {n_after - n_before:+,} 参数" if applied else ""))
        print(f"[自检] 检测层数 nl={getattr(head, 'nl', '?')}  stride={list(getattr(head, 'stride', []))}")
        print(f"[自检] 头输入特征图索引：{list(getattr(head, 'f', []))}")
        if task == "pose":
            kpt_branch = ("cv4_kpts + cv4_sigma（Pose26，带归一化流）"
                          if hasattr(head, "cv4_kpts") else "cv4（Pose）")
            print(f"[自检] 关键点：kpt_shape={getattr(head, 'kpt_shape', '?')} "
                  f"nk={getattr(head, 'nk', '?')}  分支={kpt_branch}")
        print(f"[自检] 前向输出层数：{len(out) if isinstance(out, (list, tuple)) else 1}")
        return 0

    train_args = dict(spec.get("train") or {})
    for item in args.set:
        if "=" not in item:
            raise SystemExit(f"--set 需要 K=V 形式，收到 {item!r}")
        key, value = item.split("=", 1)
        train_args[key.strip()] = _coerce(value.strip())

    if resolved is not None:
        train_args["data"] = str(resolved.yaml)
    elif "data" not in train_args:
        raise SystemExit(
            "未指定数据集：请用 --data 传入注册表名（如 visdrone2019-det）、数据集目录或"
            "数据集 YAML，或在变体配置里写 data.dataset。"
        )

    if task == "pose" and _reproj_enabled(spec):
        from tod.data.pose6d import pose_safe_augment_overrides

        explicit = {item.split("=", 1)[0].strip() for item in args.set if "=" in item}
        forced = {k: v for k, v in pose_safe_augment_overrides().items()
                  if k not in explicit}
        train_args.update(forced)
        print("[增强] L_rep 启用 -> 关闭会丢目标的几何增强："
              + ", ".join(f"{k}={v}" for k, v in forced.items())
              + "（要保留某项就显式 --set 覆盖它）")

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

    trainer = _task_trainers()[task]
    print(f"[训练] {variant.name} | task={task} | data={resolved.root if resolved else train_args.get('data')} | "
          f"imgsz={train_args.get('imgsz')} | epochs={train_args.get('epochs')} | "
          f"box_loss={runtime.get('eps.EP7.box', '框架默认')}"
          + (f" | pose_loss={runtime.get('eps.EP7.pose', '框架默认')}" if task == "pose" else ""))
    model.train(trainer=trainer, **train_args)
    return 0


def _task_trainers() -> dict:
    """task → 训练器类（延迟导入：未装 ultralytics 时 tools/train.py 仍可 --help）。"""

    from tod.engine.pose_trainer import TODPoseTrainer
    from tod.engine.trainer import TODDetectionTrainer

    return {"detect": TODDetectionTrainer, "pose": TODPoseTrainer}


def _dry_run_args(spec: dict, get_cfg):
    """给 ``--dry-run`` 造一份训练超参命名空间。"""
    args = get_cfg()
    for key, value in (spec.get("train") or {}).items():
        if hasattr(args, key):
            setattr(args, key, value)
    return args


def _resolve_data(spec, args, *, task: str = "detect", require: bool):
    """按**注册表**把 ``--data`` / 变体里的 ``dataset`` 解析成数据集路径。"""
    from tod.paths import DatasetPathError, resolve

    if not spec:
        if require:
            raise SystemExit(
                "未指定数据集：用 --data 传注册表里的名字（如 visdrone2019-det）、"
                "数据集目录或 YAML 路径，或在变体配置里写 data.dataset。\n"
                "想看注册表里有哪些数据集：--list-datasets"
            )
        print("[warn ] 未指定数据集（--dry-run 继续：结构自检不需要数据）")
        return None

    candidates = [spec]
    if task == "pose" and isinstance(spec, str) and "2019-det" in spec:
        candidates.append(spec.replace("2019-det", "2019-pose"))

    last: Exception | None = None
    for candidate in candidates:
        try:
            paths = resolve(candidate, registry_path_hint=args.datasets,
                            data_root=args.data_root, require_train=require)
        except DatasetPathError as exc:
            last = exc
            continue
        if paths is not None:
            for line in paths.report():
                print(line)
        return paths

    if require:
        raise SystemExit(f"[错误] {last}")
    fallback = _template_only(spec)
    if fallback is not None:
        print(f"[数据] 数据集={fallback.name} | 根目录未就位（--dry-run 模式，不读图片）")
        print(f"[数据] 只读字段的模板：{fallback.yaml}")
        for warning in fallback.warnings:
            print(f"[warn ] {warning}")
        return fallback
    print(f"[warn ] 数据集未就绪（--dry-run 继续，不需要数据）：{last}")
    return None


def _template_only(spec):
    """按数据集名找 ``configs/_base_/datasets/<名>.yaml``（只读字段，不碰图片）。"""
    from tod.paths import TEMPLATE_DIR_REL, DatasetPaths

    name = Path(str(spec)).stem
    candidate = ROOT / TEMPLATE_DIR_REL / f"{name}.yaml"
    if not candidate.is_file():
        return None
    return DatasetPaths(
        name=name, root=candidate.parent, yaml=candidate,
        source=f"模板 {candidate.relative_to(ROOT).as_posix()}（数据尚未就位，--dry-run 只读字段）",
        warnings=(f"数据集根目录未就位（{spec}）：本次只用模板字段做结构自检",),
    )


if __name__ == "__main__":
    from tod.compat import CompatError as _CompatError

    try:
        raise SystemExit(main())
    except _CompatError as exc:
        raise SystemExit(f"[环境错误] {exc}") from exc
