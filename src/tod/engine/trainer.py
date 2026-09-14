"""自定义 Trainer：让框架用我们注册的回归损失/训练策略。

导入本模块需要已安装 ultralytics（否则直接报错，而不是悄悄退化）。
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
    """在框架 Trainer 之上，只做两件事：替换损失准则、应用检测头手术。"""

    def init_criterion(self):
        kind = runtime.get("eps.EP7.box", "siou")
        theta = float(runtime.get("eps.EP7.theta", 4.0) or 4.0)
        return build_detection_loss(self.model, kind=kind, theta=theta)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """构建模型后立刻应用 EP5 检测头手术（见 tod.engine.surgery 的说明）。

        手术放在 ``super().get_model()`` 之后：此时预训练权重已加载完毕，
        分支末端 1×1 输出卷积会被原地复用，权重不受影响；
        被替换的两层 stem 卷积形状不同，只能重新初始化。
        """
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        applied = apply_spec(model, runtime.active())
        if applied:
            print("[EP5] 检测头手术已应用：")
            for line in applied:
                print(f"      {line}")
            if weights:
                print("      注意：分支 stem 的预训练权重因形状改变已被丢弃（重新初始化）。")
        return model
