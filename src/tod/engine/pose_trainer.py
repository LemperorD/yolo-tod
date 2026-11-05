"""姿态训练器：把检测侧的 EP 接线**全部继承**过来，只换"准则怎么建"这一步。"""

from __future__ import annotations

from pathlib import Path

from tod import runtime
from tod.engine.distill import build_kd
from tod.engine.trainer import TODDetectionTrainer, _has_prog_loss, stal_flag
from tod.loss.criterion import build_pose_criterion, build_reproj_criterion

try:  # pragma: no cover
    from ultralytics.models.yolo.pose import PoseTrainer as _PoseTrainer
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "tod.engine.pose_trainer 需要 ultralytics。请先完成环境安装：\n"
        "  pip install -e .[dev]"
    ) from exc


# MRO 复用检测侧扩展与框架 PoseModel；保留框架构造器的默认 cfg。
class TODPoseTrainer(TODDetectionTrainer, _PoseTrainer):
    """姿态训练器 = 框架 ``PoseTrainer`` + 本库的 EP6/EP7/EP9 接线。"""

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0,
                       mode: str = "train"):
        """训练/验证 dataloader：启用 ``L_rep`` 时把数据集换成带 **6D 位姿**的版本。"""
        if not self._needs_pose6d():
            return super().get_dataloader(dataset_path, batch_size, rank, mode)

        from tod import compat
        from tod.data.pose6d import pose6d_dataset_context, verify_pose_in_loader

        data = self.data or {}
        tod_cfg = data.get("tod") or {}
        pose_dir = tod_cfg.get("pose_labels")
        if not pose_dir:
            raise compat.CompatError(
                "EP7.reproj 已开启，但数据集配置里没有 tod.pose_labels（labels_6d 目录名）。\n"
                "请按 docs/KEYPOINTS.md §5.2 提供逐目标 6D 位姿，或把 EP7.reproj 关掉。"
            )
        # PnP 不可解的帧可通过 pose_strict=false 保留二维监督，由 mask 跳过几何项。
        strict = bool(tod_cfg.get("pose_strict", True))
        if not strict:
            print("[pose6d] pose_strict=false：缺 6D 位姿的帧会被 L_rep 跳过")
        root = Path(str(data.get("path") or "."))
        if not root.is_absolute():
            root = (Path(self.args.data).parent / root).resolve()

        csv_name = f"{mode}.csv"
        with pose6d_dataset_context(root / str(pose_dir), mode, strict=strict) as dataset_cls:
            loader = super().get_dataloader(dataset_path, batch_size, rank, mode)
            verify_pose_in_loader(loader, source=f"{mode}：{root / str(pose_dir) / csv_name}")
            return loader

    def _needs_pose6d(self) -> bool:
        """当前变体是否真的需要 6D 位姿（即 ``L_rep`` 已启用）。"""
        ep7 = ((runtime.active().get("eps") or {}).get("EP7") or {})
        flag = ep7.get("reproj", False)
        return not (flag is False or flag is None
                    or str(flag).lower() in {"", "off", "none", "false"})

    def _install_criterion(self):
        """按变体 spec 重建姿态准则（EP6 分配 + EP7 框/关键点损失 + EP9 蒸馏）。"""
        spec = runtime.active()
        model = self.model
        eps = spec.get("eps") or {}
        ep6, ep7, ep9 = (eps.get("EP6") or {}), (eps.get("EP7") or {}), (eps.get("EP9") or {})

        kind = ep7.get("box")
        dfl_gain = float(getattr(self.args, "dfl", 1.5) or 0.0)
        # 只有替换框损失时才按零增益跳过 DFL，保留原生准则的归一化行为。
        use_dfl = False if (dfl_gain == 0.0 and kind is not None) else None

        criterion, report = build_pose_criterion(
            model, ep7=ep7, stal=stal_flag(ep6), use_dfl=use_dfl,
        )
        criterion = build_reproj_criterion(criterion, spec)
        report = list(getattr(criterion, "_tod_patched", report))

        distill = ep9.get("distill")
        teacher = ep9.get("teacher")
        if isinstance(distill, str) and distill and teacher:
            criterion = build_kd(
                criterion, model, teacher,
                lambda_=float(ep9.get("kd_lambda", ep9.get("lambda", 0.5)) or 0.5),
                temperature=float(ep9.get("temperature", 3.0) or 3.0),
                levels=tuple(ep9.get("kd_levels", (2, 3, 4, 5))),
            )
            print(f"[EP9] 蒸馏：{distill}(λ={criterion.kd.lambda_}, T={criterion.kd.temperature}, "
                  f"levels={list(criterion.kd.levels)})  教师={teacher}")
        elif isinstance(distill, str) and distill:
            print(f"[EP9] 蒸馏已配置（{distill}）但缺少 teacher 权重路径，本次不启用。")

        model.criterion = criterion
        head = model.model[-1] if hasattr(model, "model") else None
        print("[EP7] 姿态准则：" + ("；".join(report) if report else "框架默认（未做替换）"))
        print(f"[EP7] 关键点分支：kpt_shape={getattr(head, 'kpt_shape', '?')}、"
              f"增益 pose={getattr(self.args, 'pose', '?')} / kobj={getattr(self.args, 'kobj', '?')}"
              f" | dfl 增益={dfl_gain}")
        if hasattr(criterion, "active"):
            if criterion.active:
                state = "本期生效"
            elif not criterion.enabled:
                state = "本期未生效（L_rep 已关闭）"
            else:
                remaining = max(0, int(criterion.warmup_epochs) - int(criterion.epoch))
                state = (f"本期未生效：warm-up 还剩 {remaining} 个 epoch"
                         f"（已训完 {criterion.epoch} / {criterion.warmup_epochs}）")
            print(f"[EP7] 几何重投影 L_rep：权重={criterion.weight}、"
                  f"warm-up={criterion.warmup_epochs} epoch、当前 epoch={criterion.epoch}、"
                  f"{state}")
        print(f"[EP4/EP5] 已生效 EP：{sorted(k for k, v in eps.items() if v)}"
              f" | ProgLoss={'框架原生 E2ELoss.update' if _has_prog_loss(criterion) else '不适用'}")
        if not getattr(self, "_lrep_cb_added", False):
            self.add_callback("on_train_epoch_end", self._log_lrep)
            self._lrep_cb_added = True

    def _log_lrep(self, trainer) -> None:
        """epoch 末打印 ``L_rep`` 的均值/累计项数，避免几何项在日志里“隐身”。"""
        criterion = getattr(trainer.model, "criterion", None)
        if criterion is None or not hasattr(criterion, "reproj_terms"):
            return
        mean = getattr(criterion, "epoch_mean", None)
        if mean is None:
            return
        print(f"[L_rep] epoch={getattr(criterion, 'epoch', '?')} "
              f"mean={mean:.6f} terms={criterion.reproj_terms()} "
              f"active={getattr(criterion, 'active', '?')}")


__all__ = ["TODPoseTrainer"]
