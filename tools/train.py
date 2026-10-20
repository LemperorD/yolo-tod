"""训练入口：读变体 spec → 注入魔改模块 → 生成模型图 → 交给 ultralytics 训练。

训练集位置由**注册表 YAML**（默认 ``configs/datasets.yaml``）决定，脚本本身不含任何
绝对路径 —— 服务器换盘 / 换机器只改注册表，不改代码：

用法::

    # 0) 先物化变体（生成 variant.yaml / model.yaml / card.md）
    python3 tools/make_variant.py variants/SPAE-YOLOv8n/recipe.py

    # 1) 看注册表里有哪些数据集、根目录指向哪里、各分划有多少张图
    python3 tools/train.py --list-datasets

    # 2) 只做结构自检：建模型 + 一次前向，不训练（不需要数据集）
    python3 tools/train.py --variant variants/SPAE-YOLOv8n/variant.yaml --dry-run

    # 3) 真训练：--data 可以直接写注册表里的名字
    python3 tools/train.py --variant variants/SPAE-YOLOv8n/variant.yaml \
        --data visdrone2019-det --epochs 150 --batch 4

    # 4) 临时换盘/换布局：命令行覆盖根目录（优先级最高）
    python3 tools/train.py --variant variants/SPAE-YOLOv8n/variant.yaml \
        --data visdrone2019-det --data-root /mnt/hdd/datasets
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

    # ---- 0) 只看注册表：换盘之后先跑这一条，确认根目录指向新硬盘、分划里有图 ----
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
    variant = Variant.from_spec(spec)

    # 现场重新生成模型图：保证 model.yaml 与变体配置永不脱节
    out_dir = args.variant.parent
    model_path = out_dir / "model.yaml"
    cfg = variant.model_yaml(model_path, write=True)
    print(f"[模型] {model_path}")
    print(f"       P2 分支：{cfg.get('_p2_status', '未启用')}；"
          f"下采样替换：{cfg.get('_downsample_replaced') or '无'}；"
          f"节点类型替换：{cfg.get('_type_map_applied') or '无'}")

    # ---- 训练集位置：由注册表 YAML 决定，脚本里没有任何绝对路径 ----
    # --data（名字/目录/YAML）> 变体里的 data.dataset > 注册表 data_root/<名>
    resolved = _resolve_data(args.data or variant.dataset, args,
                             task=str(spec.get("task") or "detect"),
                             require=not args.dry_run)

    try:
        tod.bootstrap()          # 导入模块库并把魔改注入框架命名空间
    except ImportError as exc:
        raise SystemExit(
            f"载入魔改模块失败：{exc}\n"
            "请确认已完成环境安装（ultralytics + torch）。"
        ) from exc

    runtime.set_active(spec)

    # L_rep（论文式 6/8）需要数据集侧的相机内参 / 机体 3D 关键点。这里把数据集 YAML
    # 的内容快照塞进 spec，训练器与 --dry-run 都能从 spec 读到，不必各自再解析路径。
    def _attach_dataset_cfg(data_path) -> None:
        if not data_path:
            return
        try:
            spec["dataset_cfg"] = load_yaml(data_path)
        except Exception as exc:  # noqa: BLE001 - 不影响其它 EP
            print(f"[warn ] 未能读取数据集配置（L_rep 将不可用）：{exc}")

    _attach_dataset_cfg(resolved.yaml if resolved is not None else None)

    # 受限沙箱（Windows 命名管道被禁）下必须把标签缓存换成顺序实现，
    # 否则 dataloader 一建就 PermissionError [WinError 5]（见 tod/compat.py 的说明）。
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

    # task 必须显式传给 YOLO：本库生成的模型图叫 model.yaml，而框架的
    # ``guess_model_task`` **靠文件名**判任务（只认 *-pose 之类），不看网络结构。
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

        # ---- EP6/EP7/EP9：准则与优化器接线自检（同样不需要数据集）----
        eps = spec.get("eps") or {}
        ep6, ep7, ep9 = (eps.get("EP6") or {}), (eps.get("EP7") or {}), (eps.get("EP9") or {})
        model.model.args = _dry_run_args(spec, get_cfg)      # 准则会读 model.args 当 hyp

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
    for item in args.set:                      # --set k=v 覆盖（消融/自检常用）
        if "=" not in item:
            raise SystemExit(f"--set 需要 K=V 形式，收到 {item!r}")
        key, value = item.split("=", 1)
        train_args[key.strip()] = _coerce(value.strip())

    # 训练集 YAML 已在 main 开头由注册表解析好（绝对路径副本），这里只做交接。
    if resolved is not None:
        train_args["data"] = str(resolved.yaml)
    elif "data" not in train_args:
        raise SystemExit(
            "未指定数据集：请用 --data 传入注册表名（如 visdrone2019-det）、数据集目录或"
            "数据集 YAML，或在变体配置里写 data.dataset。"
        )

    # ---- L_rep 与"会丢目标的几何增强"互斥（与 tools/train_mav6d.py 同一口径）----
    # 启用 6D 位姿监督时，框与 labels_6d 的位姿必须**一一对应**；而框架的
    # RandomPerspective 会把变换后面积占比 <10% 的框连同类一起丢掉（mosaic 更会
    # 把多张图拼在一起），之后就无法认定"CSV 的哪几行还对应得上"，训练会直接报
    # "labels_6d 里有 N 个目标的位姿，但标签文件里有 M 个 GT 框"。命令行 --set 显式
    # 给出的键优先级更高（自担位姿错配风险）。
    if task == "pose" and _reproj_enabled(spec):
        from tod.data.pose6d import pose_safe_augment_overrides

        explicit = {item.split("=", 1)[0].strip() for item in args.set if "=" in item}
        forced = {k: v for k, v in pose_safe_augment_overrides().items()
                  if k not in explicit}
        train_args.update(forced)
        print("[增强] L_rep 启用 -> 关闭会丢目标的几何增强："
              + ", ".join(f"{k}={v}" for k, v in forced.items())
              + "（要保留某项就显式 --set 覆盖它）")

    # 实验输出目录：变体里写的可能是相对路径（如 "results"），必须锚到仓库根，
    # 否则 ultralytics 会把它解析成 runs/detect/results/...（散落在仓库里）
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
    """给 ``--dry-run`` 造一份训练超参命名空间。

    准则（``v8DetectionLoss``/``E2ELoss``）会把 ``model.args`` 当超参读，
    直接建图得到的 ``model.args`` 是 dict，会报 ``'dict' object has no attribute 'box'``。
    """
    args = get_cfg()
    for key, value in (spec.get("train") or {}).items():
        if hasattr(args, key):
            setattr(args, key, value)
    return args


def _resolve_data(spec, args, *, task: str = "detect", require: bool):
    """按**注册表**把 ``--data`` / 变体里的 ``dataset`` 解析成数据集路径。

    这是"训练集在哪"的唯一决定点：解析出来的是**绝对**根目录 + 一份 ``path`` 已被
    改写成绝对路径的数据集 YAML（``tod.paths`` 生成的副本），所以框架不会再拿它自己的
    ``datasets_dir`` 去猜目录 —— 训练集在新挂的机械硬盘上时尤其重要。

    Args:
        spec: ``--data``（名字 / 目录 / YAML）或变体里写的 ``data.dataset``。
        args: 命令行命名空间（提供 ``--datasets`` / ``--data-root``）。
        task: 任务类型；姿态任务的 ``*-det`` 名字会顺带试一次 ``*-pose``。
        require: ``True`` = 训练集不可用就是致命错误；``False`` = ``--dry-run``
            用，缺数据集只警告（结构自检本来就不需要数据）。

    Returns:
        :class:`tod.paths.DatasetPaths`，或 ``None``（``--dry-run`` 且数据集缺失）。
    """
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
        # 检测数据集与姿态数据集不是同一份配置（后者多 kpt_shape），沿用老口径
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
    # --dry-run 只读数据集 YAML 里的**字段**（kpt_shape / 内参 / 3D 点），不需要图片。
    # 数据还没拷到新盘上时退回到模板 YAML，保证结构自检不会被"数据不在场"卡住
    # （老版本就是这么做的；这里只是把它显式化，并说明读的是哪份模板）。
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
    # CompatError 在 main() 内部才被导入（那时 sys.path 才加好），
    # 所以这里也要延迟导入：早年写成模块级 except 会抛 NameError 掩盖真实退出码。
    from tod.compat import CompatError as _CompatError

    try:
        raise SystemExit(main())
    except _CompatError as exc:  # 环境/版本问题给出可读提示
        raise SystemExit(f"[环境错误] {exc}") from exc
