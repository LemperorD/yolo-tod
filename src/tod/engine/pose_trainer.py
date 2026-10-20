"""姿态训练器：把检测侧的 EP 接线**全部继承**过来，只换"准则怎么建"这一步。

**为什么姿态要单独一个 Trainer**：ultralytics 的 ``PoseTrainer`` 负责三件检测侧
不存在的事 —— 用例程 ``PoseModel`` 建图（用数据集的 ``kpt_shape`` 覆盖模型图）、
把 ``self.loss_names`` 换成 ``box/pose/kobj/cls/dfl``、以及用 ``PoseValidator`` 验证。
本库不改这些，**只接管训练准则**（EP6/EP7/EP9），其余钩子（EP4/EP5 建模后手术、
MuSGD、蒸馏、resume 兜底）与 ``TODDetectionTrainer`` 完全一致 —— 所以这里靠继承
而不是复制：任何在检测训练器上修好的 bug 会自动作用到姿态。

姿态准则与检测准则的关系（决定了接线的写法）：
    ``v8PoseLoss`` / ``PoseLoss26`` **继承自** ``v8DetectionLoss``，所以
      * 换框回归损失（EP7 ``box``）走的是同一套 ``bbox_loss`` 替换，**完全复用**；
      * 关键点项是**额外**的一层（``keypoint_loss`` + ``pose``/``kobj`` 两个增益），
        由 ``tod.loss.pose.TinyPoseLoss`` 替换，且必须同时命中 O2M/O2O **两套**子准则
        （YOLO26 ``end2end`` 下 ``E2ELoss`` 持有两份 ``PoseLoss26``）。

用法（``tools/train.py`` 会按变体的 ``task`` 自动选）::

    from tod.engine.trainer import TODPoseTrainer
    model.train(trainer=TODPoseTrainer, **train_args)
"""

from __future__ import annotations

from pathlib import Path

from tod import runtime
from tod.engine.distill import build_kd
from tod.engine.trainer import TODDetectionTrainer, _has_prog_loss, stal_flag
from tod.loss.criterion import build_pose_criterion, build_reproj_criterion

try:  # pragma: no cover - 环境相关
    from ultralytics.models.yolo.pose import PoseTrainer as _PoseTrainer
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "tod.engine.pose_trainer 需要 ultralytics。请先完成环境安装：\n"
        "  pip install -e .[dev]"
    ) from exc


class TODPoseTrainer(TODDetectionTrainer, _PoseTrainer):
    """姿态训练器 = 框架 ``PoseTrainer`` + 本库的 EP6/EP7/EP9 接线。

    MRO 说明：``TODDetectionTrainer`` 在前，因此
      * ``get_model`` / ``set_model_attributes`` / ``build_optimizer`` / ``resume_training``
        这些"手术 + 优化器 + resume 兜底"用本库版本；
      * ``get_model`` 里 ``super().get_model()`` 沿 MRO 落到 ``PoseTrainer.get_model``，
        于是建的仍是 **PoseModel**（关键点形状由数据集给出）。

    **刻意不写 ``__init__``**（这是一个踩过的坑）：MRO 的第一个 ``__init__`` 是
    ``DetectionTrainer`` 的，它的签名是 ``(cfg, overrides, _callbacks)``，而框架实例化
    trainer 时是 ``Trainer(overrides=args)`` —— ``cfg`` 取**形参默认值**。
    如果本类自己写一个 ``cfg=None`` 的 ``__init__`` 再 ``super().__init__(cfg, ...)``，
    就会把 ``None`` 显式传给 ``BaseTrainer``，触发
    ``AttributeError: 'NoneType' object has no attribute 'keys'``（框架的
    ``get_cfg(cfg, overrides)`` 会去读 ``cfg.keys()``）。
    不写 ``__init__`` → 直接继承 ``PoseTrainer.__init__``，由它设置
    ``overrides["task"] = "pose"`` 并正确传递 ``DEFAULT_CFG``。
    """

    # ---------------------------------------------------------------- 数据
    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0,
                       mode: str = "train"):
        """训练/验证 dataloader：启用 ``L_rep`` 时把数据集换成带 **6D 位姿**的版本。

        为什么必须在这里替换：``L_rep``（论文式 8）需要逐目标的 ``R, t``，而框架标准的
        ``YOLODataset`` 只产出框 + 关键点。本库不改框架，而是在 ``build_dataset``
        外面把框架的 ``YOLODataset`` 临时换成子类（子类在 ``__getitem__`` 里附加
        ``gt_pose`` / ``gt_pose_mask``，由框架的 ``collate_fn`` 自动收集进 batch）。这样：
          * 不改框架源码，也不复制它的 dataset 实现；
          * 只有真的启用 ``L_rep`` 时才生效（关掉时行为与纯姿态训练**完全一致**）；
          * 校验失败会**直接报错**，不会静默退化成"没有几何监督"。

        **注意替换的是两个命名空间**（``ultralytics.data.dataset`` 与
        ``ultralytics.data.build``）：``build_yolo_dataset()`` 用的是后者自己的全局名，
        只换前者等于没换 —— 详见 ``tod.data.pose6d.pose6d_dataset_context``。
        建完 loader 立刻用 ``verify_pose_in_loader`` 校验，避免"替换失效"拖到
        warm-up 结束后的第一个 batch 才炸。
        """
        if not self._needs_pose6d():
            return super().get_dataloader(dataset_path, batch_size, rank, mode)

        from tod import compat
        from tod.data.pose6d import pose6d_dataset_context, verify_pose_in_loader

        data = self.data or {}
        pose_dir = (data.get("tod") or {}).get("pose_labels")
        if not pose_dir:
            raise compat.CompatError(
                "EP7.reproj 已开启，但数据集配置里没有 tod.pose_labels（labels_6d 目录名）。\n"
                "请按 docs/KEYPOINTS.md §5.2 提供逐目标 6D 位姿，或把 EP7.reproj 关掉。"
            )
        root = Path(str(data.get("path") or "."))
        if not root.is_absolute():
            root = (Path(self.args.data).parent / root).resolve()

        csv_name = f"{mode}.csv"
        with pose6d_dataset_context(root / str(pose_dir), mode) as dataset_cls:
            loader = super().get_dataloader(dataset_path, batch_size, rank, mode)
            verify_pose_in_loader(loader, source=f"{mode}：{root / str(pose_dir) / csv_name}")
            return loader

    def _needs_pose6d(self) -> bool:
        """当前变体是否真的需要 6D 位姿（即 ``L_rep`` 已启用）。"""
        ep7 = ((runtime.active().get("eps") or {}).get("EP7") or {})
        flag = ep7.get("reproj", False)
        return not (flag is False or flag is None
                    or str(flag).lower() in {"", "off", "none", "false"})

    # ---------------------------------------------------------------- 准则
    def _install_criterion(self):
        """按变体 spec 重建姿态准则（EP6 分配 + EP7 框/关键点损失 + EP9 蒸馏）。"""
        spec = runtime.active()
        model = self.model
        eps = spec.get("eps") or {}
        ep6, ep7, ep9 = (eps.get("EP6") or {}), (eps.get("EP7") or {}), (eps.get("EP9") or {})

        kind = ep7.get("box")
        # "无 DFL"只在真的换了框损失时才省计算（否则连框架的 reg_max=1 归一化项一起丢掉，
        # 那属于改变了框架默认行为，不该由 dfl 增益单独决定）
        dfl_gain = float(getattr(self.args, "dfl", 1.5) or 0.0)
        use_dfl = False if (dfl_gain == 0.0 and kind is not None) else None

        criterion, report = build_pose_criterion(
            model, ep7=ep7, stal=stal_flag(ep6), use_dfl=use_dfl,
        )
        # EP7 的几何重投影项 L_rep（论文式 6/8）：需要数据集侧的相机内参与机体 3D 关键点
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
        if hasattr(criterion, "active"):        # ReprojCriterion：报告 L_rep 的 warm-up 状态
            # 文案要说准：warmup=0 时"warm-up 中"是错的说法（前 0 个 epoch 只用 2D 项
            # 意味着从第 1 个 epoch 起就该生效，只是框架此刻还没训完第 1 个 epoch）。
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


__all__ = ["TODPoseTrainer"]
