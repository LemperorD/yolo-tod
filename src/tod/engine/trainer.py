"""自定义 Trainer：把变体 spec 里的 EP 改动接进框架训练循环。

**为什么必须有这个文件（也是一个已修的坑）**：ultralytics 在
``DetectionModel.__init__`` 里就调用 ``self.init_criterion()`` 建准则，
训练器**不会**调用 ``Trainer.init_criterion``。因此"在 Trainer 上覆盖
``init_criterion``"是**死代码**（本库 v0.1 就是这么写的，EP7 的损失替换从未生效）。
正确做法是两个钩子：

    1. ``get_model()`` —— 模型构建完成后做 EP4/EP5 的**建模后手术**
       （模型对象级改动，不 fork 框架）；
    2. ``set_model_attributes()`` —— 此时 ``model.args`` 才被赋成训练参数，
       准则必须在这之后重建（``v8DetectionLoss`` 会读 ``model.args`` 当超参，
       早于此读到的是 dict，直接报 ``'dict' object has no attribute 'box'``）；
       resume 路径（框架会 ``criterion = model.init_criterion()``）用
       ``resume_training()`` 兜一次。

接线内容：
    EP7  回归损失（``wiou``/``siou``…）+ 无 DFL（``dfl=0.0`` 时连计算一起省掉）
    EP6  标签分配（STAL 开关；框架 8.4 原生已含小目标先验，仅在此做显式化/消融）
    EP4  注意力注入（建模后手术）
    EP5  检测头手术
    EP9  MuSGD 优化器（优先框架原生，缺失时用本库实现）+ 特征对齐蒸馏（可选）

用法见 ``tools/train.py``::

    model.train(trainer=TODDetectionTrainer, **train_args)
"""

from __future__ import annotations

from tod import runtime
from tod.engine.surgery import apply_spec
from tod.loss.criterion import build_detection_loss

try:  # pragma: no cover - 环境相关
    from ultralytics.models.yolo.detect import DetectionTrainer as _DetectionTrainer
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "tod.engine.trainer 需要 ultralytics。请先完成环境安装：\n"
        "  conda create -n tod python=3.12 -y && conda activate tod\n"
        "  pip install -e .[dev]"
    ) from exc


class TODDetectionTrainer(_DetectionTrainer):
    """框架 Trainer + 本库的 EP 接线（损失 / 分配 / 手术 / 优化器 / 蒸馏）。"""

    # ---------------------------------------------------------------- 模型
    def get_model(self, cfg=None, weights=None, verbose=True):
        """构建模型后立刻应用 EP4/EP5 的建模后手术（见 tod.engine.surgery）。

        手术放在 ``super().get_model()`` 之后：此时预训练权重已加载完毕，
        分支末端 1×1 输出卷积会被原地复用，权重不受影响；
        被替换的两层 stem 卷积形状不同，只能重新初始化。
        """
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        applied = apply_spec(model, runtime.active())
        if applied:
            print("[EP4/EP5] 建模后手术已应用：")
            for line in applied:
                print(f"      {line}")
            if weights:
                print("      注意：EP5 的分支 stem 预训练权重因形状改变已被丢弃（重新初始化）。")
        return model

    # ------------------------------------------------------------ 属性/准则
    def set_model_attributes(self):
        """框架在这里把 ``self.args`` 写进模型；准则必须紧随其后重建。"""
        super().set_model_attributes()
        self._install_criterion()

    def resume_training(self, ckpt):
        """resume 时框架会用 ``model.init_criterion()`` 覆盖准则，这里补回我们的版本。"""
        super().resume_training(ckpt)
        self._install_criterion()

    # ---------------------------------------------------------------- 优化器
    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9,
                        decay=1e-5, iterations=1e5):
        """MuSGD 优先走框架原生实现；框架 < 8.4 时退回本库实现。"""
        from tod.optim import musgd

        requested = runtime.get("eps.EP9.optimizer")
        if isinstance(requested, str) and requested.lower() in {"musgd", "musgd_fallback"}:
            prefer_native = bool(runtime.get("eps.EP9.prefer_native", True))
            if prefer_native and musgd.native_musgd() is not None:
                print("[EP9] 优化器：ultralytics.optim.MuSGD（框架原生，"
                      "含 YOLO26 辅助头 3x lr 等细节）")
                return super().build_optimizer(
                    model, name="MuSGD", lr=lr, momentum=momentum, decay=decay,
                    iterations=iterations,
                )
            optimizer, source = musgd.build(
                model, lr=lr, momentum=momentum, weight_decay=decay,
                iterations=iterations, prefer_native=False,
            )
            print(f"[EP9] 优化器：{source}")
            return optimizer
        return super().build_optimizer(
            model, name=name, lr=lr, momentum=momentum, decay=decay, iterations=iterations,
        )

    # ---------------------------------------------------------------- 内部
    def _install_criterion(self):
        """按变体 spec 重建训练准则，并打印实际生效的改动。"""
        spec = runtime.active()
        model = self.model
        eps = spec.get("eps") or {}
        ep6, ep7, ep9 = (eps.get("EP6") or {}), (eps.get("EP7") or {}), (eps.get("EP9") or {})

        kind = ep7.get("box")
        theta = float(ep7.get("theta", 4.0) or 4.0)
        # 论文 §4.3 的 "setting dfl=0.0"：增益为 0 时连计算一起省掉
        dfl_gain = float(getattr(self.args, "dfl", 1.5) or 0.0)
        use_dfl = False if (dfl_gain == 0.0 and kind is not None) else None
        stal = stal_flag(ep6)

        criterion = build_detection_loss(
            model,
            kind=kind,
            theta=theta,
            use_dfl=use_dfl,
            stal=stal,
            **(ep7.get("kind_kwargs") or {}),
        )

        distill = ep9.get("distill")
        teacher = ep9.get("teacher")
        if isinstance(distill, str) and distill and teacher:
            from tod.engine.distill import build_kd

            criterion = build_kd(
                criterion, model, teacher,
                lambda_=float(ep9.get("kd_lambda", ep9.get("lambda", 0.5)) or 0.5),
                temperature=float(ep9.get("temperature", 3.0) or 3.0),
                levels=tuple(ep9.get("kd_levels", (2, 3, 4, 5))),
            )
            print(f"[EP9] 蒸馏：{distill}(λ={criterion.kd.lambda_}, T={criterion.kd.temperature}, "
                  f"levels={list(criterion.kd.levels)}, anchor_reduction="
                  f"{criterion.kd.anchor_reduction})  教师={teacher}")
        elif isinstance(distill, str) and distill:
            print(f"[EP9] 蒸馏已配置（{distill}）但缺少 teacher 权重路径，本次不启用。")

        model.criterion = criterion
        patched = getattr(criterion, "_tod_patched", None)
        if patched is None:
            patched = getattr(getattr(criterion, "base", None), "_tod_patched", []) or []
        print("[EP7/EP6] 训练准则：" + ("；".join(patched) if patched else "框架默认（未做替换）"))
        print(f"[EP4/EP5] 已生效 EP：{sorted(k for k, v in eps.items() if v)}"
              f" | dfl 增益={dfl_gain}"
              f" | ProgLoss={'框架原生 E2ELoss.update' if _has_prog_loss(criterion) else '不适用'}")


def stal_flag(ep6: dict) -> bool | None:
    """把 EP6 配置翻译成 ``build_detection_loss(stal=...)`` 的三态参数。

    供训练器与 ``tools/train.py --dry-run`` 共用，避免两处判断不一致。
    """
    value = ep6.get("assigner", ep6.get("small_target_aware"))
    if value is None:
        return None
    if isinstance(value, str):
        low = value.lower()
        if low in {"stal", "smalltargetassigner", "small_target_assigner", "true"}:
            return True
        if low in {"tal", "classic", "classictalassigner", "false"}:
            return False
        raise ValueError(f"无法识别的 EP6.assigner={value!r}（可选 STAL / TAL）。")
    return bool(value)


def _has_prog_loss(criterion) -> bool:
    """准则（或包装后的基准准则）是否带 ``update()``（框架的 ProgLoss 调度）。"""
    for candidate in (criterion, getattr(criterion, "base", None)):
        if callable(getattr(candidate, "update", None)):
            return True
    return False
