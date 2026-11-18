"""姿态训练器：把检测侧的 EP 接线**全部继承**过来，只换"准则怎么建"这一步。"""

from __future__ import annotations

import csv
from copy import copy
from pathlib import Path

from tod import runtime
from tod.engine.distill import build_kd
from tod.engine.trainer import TODDetectionTrainer, _has_prog_loss, stal_flag
from tod.loss.criterion import build_pose_criterion, build_reproj_criterion

try:  # pragma: no cover
    from ultralytics.models.yolo.pose import PoseTrainer as _PoseTrainer
    from ultralytics.utils import LOGGER, RANK
    from ultralytics.utils.torch_utils import unwrap_model
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "tod.engine.pose_trainer 需要 ultralytics。请先完成环境安装：\n"
        "  pip install -e .[dev]"
    ) from exc


# MRO 复用检测侧扩展与框架 PoseModel；保留框架构造器的默认 cfg。
class TODPoseTrainer(TODDetectionTrainer, _PoseTrainer):
    """姿态训练器 = 框架 ``PoseTrainer`` + 本库的 EP6/EP7/EP9 接线。"""

    def preprocess_batch(self, batch):
        mask = batch.get("gt_pose_mask")
        if RANK in {-1, 0} and mask is not None:
            masks = mask if isinstance(mask, (list, tuple)) else (mask,)
            for image_mask in masks:
                self._pose_targets += image_mask.numel()
                self._pose_valid += int(image_mask.sum())
        return super().preprocess_batch(batch)

    def get_validator(self):
        interval = runtime.active().get("pose6d_log_interval", 0)
        if not interval or RANK != -1:
            return super().get_validator()

        from tod.data.pose6d import Pose6DError, load_pose_csv
        from tod.engine.pose_validator import TODPoseValidator

        geometry = self.data.get("tod") or {}
        pose_dir = geometry.get("pose_labels")
        csv_path = Path(self.data["path"]) / pose_dir / "val.csv" if pose_dir else None
        if (not csv_path or not csv_path.is_file() or not geometry.get("camera_matrix")
                or not geometry.get("object_points")):
            LOGGER.warning("[PnP] 训练期 6D 验证需要 val.csv、相机内参和 3D 关键点")
            return super().get_validator()

        try:
            self._pnp_rows = load_pose_csv(csv_path)
        except (OSError, Pose6DError) as exc:
            LOGGER.warning(f"[PnP] 无法读取验证集位姿：{exc}")
            return super().get_validator()
        self._pnp_interval = interval
        self.loss_names = "box_loss", "pose_loss", "kobj_loss", "cls_loss", "dfl_loss"
        if getattr(unwrap_model(self.model).model[-1], "flow_model", None) is not None:
            self.loss_names += ("rle_loss",)
        return TODPoseValidator(self.test_loader, save_dir=self.save_dir,
                                args=copy(self.args), _callbacks=self.callbacks)

    def validate(self):
        interval = getattr(self, "_pnp_interval", 0)
        epoch = self.epoch + 1
        if not interval or (epoch != 1 and epoch % interval and epoch != self.epochs):
            return super().validate()

        self.validator.pose_rows = self._pnp_rows
        self.validator.pose_samples = samples = []
        try:
            result = super().validate()
        finally:
            self.validator.pose_rows = self.validator.pose_samples = None
        if result[0] is not None:
            self._log_pnp(epoch, samples)
        return result

    def _log_pnp(self, epoch: int, samples: list) -> None:
        from tod.eval.pose import evaluate_pnp_metrics

        geometry = self.data["tod"]
        stats = evaluate_pnp_metrics(samples, geometry["camera_matrix"],
                                     geometry["object_points"], geometry.get("model_diameter"))
        targets = sum(min(len(sample["gts"]), len(sample["poses"])) for sample in samples)
        metrics = {
            "epoch": epoch,
            "targets_with_pose": targets,
            "pnp_success": stats["n"],
            "pnp_failed": stats["n_pnp_failed"],
            "pnp_success_over_targets": stats["n"] / targets if targets else None,
            "add_m": stats["add_m"],
            "add_0.5d_ratio": stats["add_0.5d_ratio"],
            "rotation_error_deg": stats["rot_err_deg_mean"],
            "translation_error_m": stats["trans_err_m_mean"],
            "10deg10cm_ratio": stats["10deg10cm_ratio"],
        }
        errors = "ADD / 10°10cm 暂无有效结果"
        if stats["available"]:
            errors = (f"ADD={metrics['add_m']:.3f}m ADD(0.5d)={metrics['add_0.5d_ratio']:.1%}"
                      f" | 旋转误差={metrics['rotation_error_deg']:.2f}°"
                      f" 平移误差={metrics['translation_error_m']:.3f}m"
                      f" | 10°10cm={metrics['10deg10cm_ratio']:.1%}")
        LOGGER.info(f"[PnP val] epoch={epoch} | 解算成功={stats['n']}/{targets}"
                    f" | 匹配后解算失败={stats['n_pnp_failed']} | {errors}")
        self._write_pose_metrics("pose_val_metrics.csv", metrics)

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
            self._pose_targets = self._pose_valid = 0
            self.add_callback("on_train_epoch_end", self._log_lrep)
            self._lrep_cb_added = True

    def _log_lrep(self, trainer) -> None:
        """记录刚结束一轮的几何监督状态和实际位姿覆盖。"""
        if RANK not in {-1, 0}:
            return
        criterion = getattr(unwrap_model(trainer.model), "criterion", None)
        if criterion is None or not hasattr(criterion, "reproj_terms"):
            return
        mean = criterion.epoch_mean
        state = "warm-up"
        if criterion.epoch_active:
            state = "active" if criterion.epoch_batches else "no-valid-terms"
        coverage = self._pose_valid / self._pose_targets if self._pose_targets else None
        metrics = {
            "epoch": trainer.epoch + 1,
            "scope": "rank0" if RANK == 0 else "all",
            "lrep_state": state,
            "lrep_mean_per_batch": mean,
            "lrep_batches": criterion.epoch_batches,
            "lrep_image_branch_terms": criterion.epoch_terms,
            "pose_valid_targets": self._pose_valid,
            "pose_total_targets": self._pose_targets,
            "pose_coverage": coverage,
        }
        mean_text = "-" if mean is None else f"{mean:.6g}"
        coverage_text = "-" if coverage is None else f"{coverage:.1%}"
        LOGGER.info(f"[L_rep] epoch={metrics['epoch']} scope={metrics['scope']} {state}"
                    f" | mean={mean_text} batches={criterion.epoch_batches}"
                    f" terms={criterion.epoch_terms}"
                    f" | 6D targets={self._pose_valid}/{self._pose_targets} ({coverage_text})")
        self._write_pose_metrics("pose_train_metrics.csv", metrics)
        self._pose_targets = self._pose_valid = 0

    def _write_pose_metrics(self, filename: str, metrics: dict) -> None:
        path = self.save_dir / filename
        written = getattr(self, "_pose_metric_files", set())
        append = path.exists() and (self.resume or filename in written)
        with path.open("a" if append else "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(metrics))
            if stream.tell() == 0:
                writer.writeheader()
            writer.writerow(metrics)
        written.add(filename)
        self._pose_metric_files = written


__all__ = ["TODPoseTrainer"]
