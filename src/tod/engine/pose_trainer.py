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

from tod import runtime
from tod.engine.distill import build_kd
from tod.engine.trainer import TODDetectionTrainer, _has_prog_loss, stal_flag
from tod.loss.criterion import build_pose_criterion

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
        print(f"[EP4/EP5] 已生效 EP：{sorted(k for k, v in eps.items() if v)}"
              f" | ProgLoss={'框架原生 E2ELoss.update' if _has_prog_loss(criterion) else '不适用'}")


__all__ = ["TODPoseTrainer"]
