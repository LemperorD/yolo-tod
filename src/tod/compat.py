"""框架兼容层 —— 本库唯一允许触碰 ultralytics 内部结构的文件。

原因（PLAN.md §4.7）：ultralytics 内部 API（``parse_model``、``DetectionModel``、
``v8DetectionLoss`` 等）在次版本间时有变动。把全部适配集中在这里，
升级框架时只需要改这一个文件 + 重跑 ``tests/smoke.py``。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

FRAMEWORK = "ultralytics"
#: 本库开发/验证所基于的框架版本区间。
MIN_VERSION = "8.2.0"
MAX_TESTED_VERSION = "8.3.999"


class CompatError(RuntimeError):
    """框架缺失或版本不受支持。"""


def installed() -> bool:
    """是否已安装 ultralytics。"""
    import importlib.util

    return importlib.util.find_spec(FRAMEWORK) is not None


def version() -> str:
    """返回 ultralytics 版本号；未安装时抛 CompatError。"""
    if not installed():
        raise CompatError(
            "未检测到 ultralytics。请先建独立虚拟环境并安装依赖：\n"
            "  py -3.12 -m venv .venv && .venv\\Scripts\\activate\n"
            "  pip install -e .[dev]\n"
            "注意：本机 Python 3.14 下 torch 轮子可能不可用，建议用 3.11/3.12。"
        )
    import ultralytics

    return getattr(ultralytics, "__version__", "unknown")


def _parse(v: str) -> tuple[int, ...]:
    parts: list[int] = []
    for chunk in v.split("."):
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def assert_supported() -> str:
    """校验框架版本落在受支持区间内，返回版本号。"""
    v = version()
    if v == "unknown":
        return v
    if _parse(v) < _parse(MIN_VERSION):
        raise CompatError(f"ultralytics {v} 过旧，本库要求 >= {MIN_VERSION}。")
    if _parse(v) > _parse(MAX_TESTED_VERSION):
        raise CompatError(
            f"ultralytics {v} 超出本库验证区间（<= {MAX_TESTED_VERSION}）。"
            "请先跑 tests/smoke.py 与一次小规模训练确认无误，再放宽 MAX_TESTED_VERSION。"
        )
    return v


@lru_cache(maxsize=1)
def _tasks():
    """缓存 ``ultralytics.nn.tasks`` 模块对象。"""
    import ultralytics.nn.tasks as tasks

    return tasks


def model_globals() -> dict:
    """返回 ``parse_model`` 解析 YAML 时查名字所用的命名空间。

    ultralytics 通过 ``eval(m)`` 在该模块的 globals 中查找模块类，
    因此把注册的类塞进这个 ``__dict__`` 即可让 YAML 里直接写我们的模块名。
    """
    if not installed():
        raise CompatError("未安装 ultralytics，无法注入模块命名空间。")
    return _tasks().__dict__


@lru_cache(maxsize=1)
def cfg_root() -> Path:
    """ultralytics 内置配置目录（内含 models/ 与 datasets/）。"""
    if not installed():
        raise CompatError("未安装 ultralytics。")
    import ultralytics

    return Path(ultralytics.__file__).parent / "cfg"


def base_model_yaml(key: str = "yolov8") -> Path:
    """定位内置模型 YAML，例如 ``base_model_yaml("yolov8")`` / ``"yolov8-p2"``。

    不把官方 yaml 复制进本仓库：避免与上游版本漂移。
    """
    root = cfg_root() / "models"
    if not root.is_dir():
        raise CompatError(f"未找到 {root}，ultralytics 目录结构可能已变化。")
    matches = sorted(root.rglob(f"{key}.yaml"))
    if not matches:
        raise CompatError(
            f"内置模型 {key!r} 不存在。可选项示例："
            f"{sorted(p.stem for p in root.rglob('*.yaml'))[:20]} ..."
        )
    return matches[0]


def load_yaml(path: str | Path) -> dict:
    """读 YAML（依赖 PyYAML，缺失时给出明确提示）。"""
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise CompatError("需要 PyYAML：pip install pyyaml") from exc
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def dump_yaml(data: dict, path: str | Path) -> Path:
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise CompatError("需要 PyYAML：pip install pyyaml") from exc
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        # 保留 list-of-list 的紧凑可读形式（ultralytics yaml 风格）
        yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True, default_flow_style=None)
    return path
