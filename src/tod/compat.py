"""框架兼容层 —— 本库唯一允许触碰 ultralytics 内部结构的文件。

原因（PLAN.md §4.7）：ultralytics 内部 API（``parse_model``、``DetectionModel``、
``v8DetectionLoss`` 等）在次版本间时有变动。把全部适配集中在这里，
升级框架时只需要改这一个文件 + 重跑 ``tests/smoke.py``。
"""

from __future__ import annotations

import importlib
import importlib.util
import os
from functools import lru_cache
from pathlib import Path

FRAMEWORK = "ultralytics"
#: 本库开发/验证所基于的框架版本区间。
MIN_VERSION = "8.2.0"
#: 8.4 起：``BboxLoss.forward`` 追加 imgsz/stride、检测头支持 ``end2end``（NMS-free）、
#: 训练器自带 ``MuSGD``、TAL 自带小目标先验（STAL）。SDD-YOLO 依赖这些能力，
#: 故本地实测区间上探到 8.4（见 variants/SDD-YOLO26n/paper-notes.md §框架依赖）。
MAX_TESTED_VERSION = "8.4.999"

#: 仓库根目录（``src/tod/compat.py`` → 上溯三级）。
REPO_ROOT = Path(__file__).resolve().parents[2]


class CompatError(RuntimeError):
    """框架缺失或版本不受支持。"""


def installed() -> bool:
    """是否已安装 ultralytics。"""
    import importlib.util

    return importlib.util.find_spec(FRAMEWORK) is not None


def _default_user_config_dir() -> Path:
    """复刻 ultralytics 的默认用户配置目录，且**不 import 框架**。

    框架在 ``ultralytics.utils.get_user_config_dir()`` 里用同样规则取路径，
    并在 import 阶段就 ``mkdir`` 它——所以不能靠 ``import`` 来探路。
    """
    if os.name == "nt":
        base = os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming")
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config")
    return Path(base) / "Ultralytics"


def _writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".tod_write_probe"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def ensure_runtime_env() -> str | None:
    """保证 ultralytics 能拿到一个可写的用户配置目录。

    为什么必须放在 ``import ultralytics`` 之前：框架 import 期就会
    ``mkdir`` 用户配置目录，目录不可写时抛的是 ``PermissionError``（不是
    ``ImportError``），调用方无法用常规的 try/except 兜住，整个进程直接死掉。
    在受限环境（只读 HOME / 文件沙箱）里，这里把 ``YOLO_CONFIG_DIR`` 改指到
    仓库内 ``.cache/ultralytics``（已 gitignore），让框架正常工作。

    Returns:
        回退生效时返回新的配置目录，否则返回 None。
    """
    current = os.environ.get("YOLO_CONFIG_DIR")
    if current and _writable(Path(current)):
        return None
    target = _default_user_config_dir()
    if _writable(target):
        return None
    fallback = REPO_ROOT / ".cache" / "ultralytics"
    if not _writable(fallback):
        raise CompatError(
            f"ultralytics 配置目录不可写：默认 {target}，回退 {fallback} 也不可写。"
            "请设置环境变量 YOLO_CONFIG_DIR 指向一个可写目录。"
        )
    os.environ["YOLO_CONFIG_DIR"] = str(fallback)
    return str(fallback)


def version() -> str:
    """返回 ultralytics 版本号；未安装时抛 CompatError。"""
    if not installed():
        raise CompatError(
            "未检测到 ultralytics。请先建独立虚拟环境并安装依赖：\n"
            "  py -3.12 -m venv .venv && .venv\\Scripts\\activate\n"
            "  pip install -e .[dev]\n"
            "注意：本机 Python 3.14 下 torch 轮子可能不可用，建议用 3.11/3.12。"
        )
    ensure_runtime_env()
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
    ensure_runtime_env()
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
    ensure_runtime_env()
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


@lru_cache(maxsize=1)
def base_conv():
    """框架的基础卷积类（含 BN + 激活 + autopad）。

    优先使用框架实现，保证魔改模块与主干在初始化/激活配置上完全一致。
    ultralytics 的导入路径在历史上变动过，这里按可靠性依次尝试。
    """
    if not installed():
        raise CompatError("未安装 ultralytics，无法获取基础 Conv 类。")
    ensure_runtime_env()
    for mod_path in ("ultralytics.nn.modules.conv",
                     "ultralytics.nn.modules",
                     "ultralytics.nn.modules.block"):
        try:
            mod = importlib.import_module(mod_path)
        except ImportError:  # pragma: no cover
            continue
        conv = getattr(mod, "Conv", None)
        if conv is not None:
            return conv
    raise CompatError("未能在 ultralytics 中找到 Conv 类，请检查框架版本。")


def framework_has(name: str) -> bool:
    """框架自身的模型解析命名空间里是否已有同名模块。

    用于检测我们的注册名是否会遮蔽框架自带实现
    （例如 ultralytics 可能自带 ADown，名字不冲突才不会出意外）。
    """
    try:
        return name in model_globals()
    except CompatError:
        return False


def allow_threadless_label_cache(enable: bool = True) -> bool:
    """让框架的**标签缓存**在受限沙箱（Windows 命名管道被禁）里也能跑。

    背景（这是环境限制，不是本库的 bug）：
        ultralytics 的 ``YOLODataset.cache_labels`` 无条件用
        ``with ThreadPool(min(8, os.cpu_count())) as pool:`` 并行校验图片与标签。
        Windows 上 ``multiprocessing.pool.ThreadPool`` 的实现会创建
        ``SimpleQueue`` → ``multiprocessing.connection.Pipe`` →
        ``_winapi.CreateFile`` 打开一个**命名管道**。在文件/进程受限的沙箱会话里
        这一步会直接 ``PermissionError: [WinError 5] 拒绝访问``，训练与验证都进不去。

    本函数做的事（**不改框架源码**）：
        把 ``ultralytics.data.dataset.ThreadPool`` 换成一个**顺序执行的假池**，
        接口只用到 ``__enter__/__exit__/imap/imap_unordered``，语义与单线程串行完全一致
        —— 结果正确性不受影响，只是放弃并行加速（合成数据的标签缓存本来就是毫秒级）。

    为什么要放在 ``compat.py``：这正是"唯一允许触碰框架内部结构"的文件的职责；
    其它模块只调用 ``allow_threadless_label_cache()``，不直接 import 框架内部。

    Returns:
        ``True`` 表示补丁已生效（或已生效过），``False`` 表示未能打补丁（框架结构变化）。
    """
    if not installed():
        return False
    if not enable:
        return False
    ensure_runtime_env()
    import importlib

    try:
        module = importlib.import_module("ultralytics.data.dataset")
    except ImportError:  # pragma: no cover
        return False
    if getattr(module, "_tod_threadless_cache", False):
        return True

    class _SequentialPool:
        """``ThreadPool`` 的最小顺序替身（只实现框架实际用到的那几个方法）。"""

        def __init__(self, *args, **kwargs):
            self._closed = False

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self._closed = True
            return False

        def imap(self, func, iterable):
            return (func(item) for item in iterable)

        def imap_unordered(self, func, iterable):
            return (func(item) for item in iterable)

        def map(self, func, iterable):
            return [func(item) for item in iterable]

        def close(self):
            self._closed = True

        def join(self):
            return None

        def terminate(self):
            self._closed = True

    module.ThreadPool = _SequentialPool
    module._tod_threadless_cache = True
    return True


@lru_cache(maxsize=1)
def pose_model_class():
    """框架的姿态模型类 ``PoseModel``（``DetectionModel`` 的姿态特化）。

    为什么本库需要它：模型图是 ``*-pose.yaml`` 时，用 ``YOLO(yaml, task="pose")``
    也能跑，但**任务判定依赖文件名**（``guess_model_task`` 先看路径后看结构，
    而变体生成的图叫 ``model.yaml``，判不出 pose）。直接用 ``PoseModel`` 建图
    可以完全绕开文件名启发式，也让 ``--dry-run`` 不依赖 ``YOLO`` 包装器。

    导入路径在历史版本里变动过，按可靠性依次尝试。
    """
    if not installed():
        raise CompatError("未安装 ultralytics，无法获取 PoseModel。")
    ensure_runtime_env()
    for mod_path in ("ultralytics.nn.tasks",):
        try:
            mod = importlib.import_module(mod_path)
        except ImportError:  # pragma: no cover
            continue
        cls = getattr(mod, "PoseModel", None)
        if cls is not None:
            return cls
    raise CompatError(
        "未能找到 ultralytics.nn.tasks.PoseModel，框架结构可能已变更，"
        "请检查 src/tod/compat.py。"
    )


@lru_cache(maxsize=1)
def tal_assigner():
    """框架的 ``TaskAlignedAssigner``（EP6 的基类）。

    导入路径在历史版本里变动过（``ultralytics.utils.tal`` → 有时被再导出到
    ``ultralytics.utils``），按可靠性依次尝试，失败时给出可操作的报错。
    """
    if not installed():
        raise CompatError("未安装 ultralytics，无法获取 TaskAlignedAssigner。")
    ensure_runtime_env()
    for mod_path in ("ultralytics.utils.tal", "ultralytics.utils"):
        try:
            mod = importlib.import_module(mod_path)
        except ImportError:  # pragma: no cover
            continue
        cls = getattr(mod, "TaskAlignedAssigner", None)
        if cls is not None:
            return cls
    raise CompatError(
        "未能在 ultralytics 中找到 TaskAlignedAssigner，"
        "框架结构可能已变更，请检查 src/tod/compat.py。"
    )


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
