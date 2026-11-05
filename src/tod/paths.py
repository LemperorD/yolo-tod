"""数据集路径解析：训练/评测脚本「训练集在哪」由一份注册表 YAML 决定。"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from tod.compat import REPO_ROOT, dump_yaml, load_yaml

REGISTRY_REL = Path("configs") / "datasets.yaml"
LOCAL_REL = Path("configs") / "datasets.local.yaml"
ENV_DATA_ROOT = "TOD_DATA_ROOT"
# 交给框架的是绝对路径 YAML 副本，避免其按全局 datasets_dir 重新定位。
CACHE_REL = Path(".cache") / "tod" / "datasets"
TEMPLATE_DIR_REL = Path("configs") / "_base_" / "datasets"

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff",
                            ".dng", ".mpo", ".cr2", ".gif"})
SCAN_LIMIT = 100_000


class DatasetPathError(RuntimeError):
    """数据集定位失败。消息里必须给出**可操作**的下一步（改哪个文件/加哪个参数）。"""


def _merge(dst: dict, src: Mapping[str, Any]) -> dict:
    """就地深合并 ``src`` 到 ``dst``（后写覆盖先写，dict 递归合并）。"""
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _merge(dst[key], value)
        else:
            dst[key] = value
    return dst


def _as_path(value: Any, base: Path) -> Path:
    """把注册表里的路径写法转成 Path：支持 ``~``、``$VAR`` / ``%VAR%``、相对 ``base``。"""
    text = os.path.expandvars(os.path.expanduser(str(value).strip().strip("'\"")))
    path = Path(text)
    return path if path.is_absolute() else (base / path)


def _norm(path: Path) -> Path:
    """规范化（不要求存在）：去掉 ``..``/``.``，统一成绝对路径。"""
    try:
        return path.resolve()
    except OSError:
        return Path(os.path.normpath(str(path)))


def _inside_repo(path: Path, repo_root: Path) -> str:
    """仓库内的路径显示成相对形式（日志更短，也不会泄漏用户名）。"""
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return str(path)


def registry_path(path: str | Path | None = None, *, repo_root: Path | None = None) -> Path:
    """注册表文件位置：显式传入 > 环境变量 ``TOD_DATASETS`` > ``configs/datasets.yaml``。"""
    if path:
        return _as_path(path, Path(repo_root or REPO_ROOT))
    env = os.environ.get("TOD_DATASETS")
    if env:
        return _as_path(env, Path(repo_root or REPO_ROOT))
    return Path(repo_root or REPO_ROOT) / REGISTRY_REL


def load_registry(path: str | Path | None = None, *, repo_root: Path | None = None,
                  required: bool = False) -> dict:
    """读注册表，并用 ``configs/datasets.local.yaml``（若存在）覆盖。"""
    root = Path(repo_root or REPO_ROOT)
    file = registry_path(path, repo_root=root)
    registry: dict = {}
    if file.is_file():
        registry = dict(load_yaml(file) or {})
    elif required:
        raise DatasetPathError(
            f"数据集注册表不存在：{file}\n"
            "本库用一份 YAML 决定训练集位置（换盘/换机器只改它）。"
            "请确认仓库完整，或用 --datasets 指定注册表路径。"
        )

    local = root / LOCAL_REL
    if local.is_file() and local != file:
        _merge(registry, dict(load_yaml(local) or {}))
        registry["_local_file"] = str(local)

    registry.setdefault("datasets", {})
    if not isinstance(registry["datasets"], dict):
        raise DatasetPathError(f"注册表 {file} 的 datasets 段必须是映射（名 → 配置）。")
    return registry


def known_names(registry: Mapping[str, Any]) -> list[str]:
    """注册表里所有可用名字（含 ``aliases``），用于报错时给出候选。"""
    names: list[str] = []
    for key, entry in (registry.get("datasets") or {}).items():
        names.append(str(key))
        if isinstance(entry, Mapping):
            names.extend(str(a) for a in (entry.get("aliases") or []))
    return names


def _entry(registry: Mapping[str, Any], name: str) -> tuple[str, dict] | None:
    """按名字或别名查注册表项，返回 ``(规范名, 展开后的配置)``。"""
    datasets = registry.get("datasets") or {}
    for key, value in datasets.items():
        entry = _normalize_entry(value)
        if str(key) == name or name in [str(a) for a in entry.get("aliases") or []]:
            return str(key), entry
    return None


def _normalize_entry(value: Any) -> dict:
    """``数据集名: 路径`` 的简写与完整写法统一成 dict。"""
    if isinstance(value, Mapping):
        return dict(value)
    if value is None:
        return {}
    return {"root": str(value)}


def _template_for(name: str, entry: Mapping[str, Any], repo_root: Path) -> Path | None:
    """该数据集的类别/字段模板 YAML（存在才算数）。"""
    if entry.get("yaml"):
        return _as_path(entry["yaml"], repo_root)
    candidate = repo_root / TEMPLATE_DIR_REL / f"{name}.yaml"
    return candidate if candidate.is_file() else None


def _count_images(directory: Path, limit: int = SCAN_LIMIT) -> int:
    """数目录下的图片（递归，到上限就停）。目录不存在返回 0。"""
    if not directory.is_dir():
        return 0
    n = 0
    stack = [directory]
    while stack and n < limit:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                for item in it:
                    try:
                        if item.is_dir():
                            stack.append(Path(item.path))
                        elif Path(item.name).suffix.lower() in IMAGE_SUFFIXES:
                            n += 1
                            if n >= limit:
                                break
                    except OSError:
                        continue
        except OSError:
            continue
    return n


def _labels_dir(images: Path) -> Path | None:
    """按框架的约定从图片目录推出标签目录（``.../images/train`` → ``.../labels/train``）。"""
    parts = list(images.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            return Path(*parts[:i], "labels", *parts[i + 1:])
    return None


def _split_dirs(data: Mapping[str, Any], root: Path,
                splits: Iterable[str]) -> dict[str, Path]:
    """数据集 YAML 里各分划的**绝对**目录（``train``/``val``/``test`` ...）。"""
    out: dict[str, Path] = {}
    for split in splits:
        value = data.get(split)
        if value is None and split == "val":
            value = data.get("validation")
        if not value:
            continue
        first = value[0] if isinstance(value, (list, tuple)) else value
        out[split] = _as_path(first, root)
    return out


@dataclass(frozen=True)
class DatasetPaths:
    """一次解析的结果：数据集名、绝对根目录、交给框架的 YAML、各分划统计。"""

    name: str | None
    root: Path
    yaml: Path
    source: str
    splits: Mapping[str, Path] = field(default_factory=dict)
    counts: Mapping[str, int] = field(default_factory=dict)
    label_counts: Mapping[str, int] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def report(self, *, compact: bool = False) -> list[str]:
        """人话版说明（训练脚本开训前打印，出问题时第一眼就看这里）。"""
        title = self.name or self.root.name
        lines = [f"[数据] 数据集={title} | 根目录={self.root}（来源：{self.source}）"]
        if not compact:
            parts = []
            for split in self.splits:
                n_img = self.counts.get(split, 0)
                n_lab = self.label_counts.get(split, 0)
                more = "≥" if n_img >= SCAN_LIMIT else ""
                parts.append(f"{split}={more}{n_img:,} 张（标签 {n_lab:,}）")
            if parts:
                lines.append("[数据] 分划：" + " | ".join(parts))
            lines.append(f"[数据] 交给框架的 YAML：{self.yaml}")
        for warning in self.warnings:
            lines.append(f"[warn ] {warning}")
        return lines


def resolve(spec: str | Path | None, *, registry: dict | None = None,
            registry_path_hint: str | Path | None = None,
            data_root: str | Path | None = None,
            repo_root: str | Path | None = None,
            cache_dir: str | Path | None = None,
            splits: Iterable[str] = ("train", "val"),
            require_train: bool = False) -> DatasetPaths | None:
    """把「注册表名 / 目录 / 数据集 YAML」解析成 :class:`DatasetPaths`。"""
    if spec is None or (isinstance(spec, str) and not spec.strip()):
        return None

    root_repo = Path(repo_root or REPO_ROOT)
    registry = dict(registry) if registry is not None else load_registry(
        registry_path_hint, repo_root=root_repo)
    cache = _as_path(cache_dir, root_repo) if cache_dir else root_repo / CACHE_REL

    text = str(spec).strip()
    given = _as_path(text, root_repo)
    name: str | None = None
    entry: dict = {}
    yaml_path: Path | None = None
    root_hint: Path | None = None
    source = ""

    looks_like_path = (
        isinstance(spec, Path)
        or given.exists()
        or "/" in text or "\\" in text
        or text.lower().endswith((".yaml", ".yml"))
    )

    if looks_like_path and given.exists():
        if given.is_dir():
            root_hint, source = _norm(given), f"命令行/配置给的目录 {text}"
            yaml_path = root_hint / "dataset.yaml"
            if not yaml_path.is_file():
                raise DatasetPathError(
                    f"{root_hint} 里没有 dataset.yaml。\n"
                    "目录方式要求数据集自身带一份描述文件（MAV6D 转换产物就是这样）；"
                    "只有 images/labels 的话，请在 configs/datasets.yaml 里为它登记 yaml 模板。"
                )
            name = root_hint.name
        else:
            yaml_path = _norm(given)
            data_peek = load_yaml(yaml_path) or {}
            name = str(data_peek.get("name") or yaml_path.stem)
            hit = _entry_by_yaml(registry, yaml_path, root_repo)
            if hit is not None:
                name, entry = hit
                source = f"注册表 datasets.{name}（由模板 YAML {_inside_repo(yaml_path, root_repo)} 反查）"
            else:
                source = f"数据集 YAML {_inside_repo(yaml_path, root_repo)} 自带的 path"
    elif looks_like_path:
        raise DatasetPathError(
            f"数据集路径不存在：{given}\n"
            "（--data 也可以直接写注册表里的名字，如 --data visdrone2019-det，"
            "训练脚本会按 configs/datasets.yaml 去找）"
        )
    else:
        hit = _entry(registry, text)
        if hit is None:
            available = known_names(registry)
            reg = registry_path(registry_path_hint, repo_root=root_repo)
            raise DatasetPathError(
                f"注册表 {reg} 里没有数据集 {text!r}。\n"
                f"可用的名字：{available or '（注册表是空的）'}\n"
                "请在 configs/datasets.yaml 的 datasets: 下新增一项，"
                "或用 --data 直接给数据集目录 / YAML 路径。"
            )
        name, entry = hit
        root_hint = None
        source = f"注册表 {REGISTRY_REL.as_posix()} 的 datasets.{name}"
        yaml_path = _template_for(name, entry, root_repo)

    override = data_root if data_root is not None else os.environ.get(ENV_DATA_ROOT)
    explicit_root = root_hint is not None
    if root_hint is None:
        if override:
            base = _as_path(override, root_repo)
            root_hint = _norm(base / str(entry.get("root") or name))
            source = ("--data-root" if data_root is not None else f"${ENV_DATA_ROOT}")
            if entry.get("root"):
                source += f" + 注册表 datasets.{name}.root"
            explicit_root = True
        elif entry.get("root"):
            base = _as_path(registry.get("data_root") or ".", root_repo)
            root_hint = _norm(_as_path(entry["root"], base))
            source = f"注册表 {REGISTRY_REL.as_posix()} 的 datasets.{name}.root"
            explicit_root = True
        else:
            base = _as_path(registry.get("data_root") or ".", root_repo)
            root_hint = _norm(base / str(name))
            source = f"注册表 data_root + 数据集名 {name!r}"
    elif override:
        base = _as_path(override, root_repo)
        root_hint = _norm(base / root_hint.name)
        explicit_root = True

    data = load_yaml(yaml_path) if yaml_path is not None and yaml_path.is_file() else None

    if not explicit_root and (root_hint is None or not root_hint.is_dir()):
        own = (data or {}).get("path")
        if own:
            candidate = _as_path(own, root_repo)
            if not candidate.is_dir() and yaml_path is not None:
                candidate = _as_path(own, yaml_path.parent)
            root_hint = _norm(candidate)
            if candidate.is_dir() and yaml_path is not None:
                source = f"数据集 YAML {_inside_repo(yaml_path, root_repo)} 自带的 path"
    root = _norm(root_hint) if root_hint is not None else None
    if root is None:
        raise DatasetPathError(f"数据集 {name!r} 的根目录无法确定，请检查注册表里的 root。")

    if not root.is_dir():
        raise DatasetPathError(
            f"数据集根目录不存在：{root}\n"
            f"  来历：{source}\n"
            "服务器新挂了盘的话，改 configs/datasets.yaml 里的 data_root（或在"
            " configs/datasets.local.yaml 里覆盖它），也可以用 --data-root / "
            f"环境变量 {ENV_DATA_ROOT} 临时指定。"
        )

    if data is None:
        candidate = root / "dataset.yaml"
        if candidate.is_file():
            yaml_path = candidate
            data = load_yaml(candidate) or {}
            source = (f"{source} + {_inside_repo(candidate, root_repo)}" if source
                      else f"{_inside_repo(candidate, root_repo)}（数据集自带的描述文件）")
        else:
            raise DatasetPathError(
                f"数据集 {name!r} 找不到描述文件（dataset.yaml）。\n"
                f"  根目录：{root}\n"
                "请在 configs/datasets.yaml 里给它补一个 yaml: 模板（类别/关键点口径），"
                "或确认该目录里有 dataset.yaml。\n"
                "若是 MAV6D 的转换产物（它里面的内参/3D 关键点是标定产物，不能手写）：\n"
                "  # 图/标签已在盘上、只缺 dataset.yaml（不删除任何数据）\n"
                f"  python3 tools/mav6d_to_yolo.py --model {root.name} --out {root.parent} --yaml-only\n"
                "  # 还没转换过（需要原始 MAV6D）\n"
                f"  python3 tools/mav6d_to_yolo.py --mav6d <原始数据> --model {root.name} "
                f"--out {root.parent}"
            )

    resolved = dict(data)
    resolved["path"] = str(root)
    resolved.setdefault("train", "images/train")
    resolved.setdefault("val", "images/val")
    cache.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(str(root).encode("utf-8")).hexdigest()[:8]
    key = _safe_name(name or root.name)
    out_yaml = cache / f"{key}-{digest}.yaml"
    dump_yaml(resolved, out_yaml)

    split_dirs = _split_dirs(resolved, root, splits)
    counts, label_counts, problems, warnings = {}, {}, [], []
    for split, directory in split_dirs.items():
        if not directory.exists():
            problems.append(f"{split} 分划目录不存在：{directory}")
            continue
        n_img = _count_images(directory) if directory.is_dir() else _count_lines(directory)
        counts[split] = n_img
        labels = _labels_dir(directory)
        if labels is not None:
            label_counts[split] = (_count_txt(labels) if labels.is_dir() else 0)
            if n_img and label_counts[split] == 0:
                warnings.append(f"{split} 有 {n_img} 张图但标签目录为空/不存在：{labels}")
        if n_img == 0:
            problems.append(f"{split} 分划里没有图片：{directory}")

    if "val" not in split_dirs:
        warnings.append("数据集没有 val 分划（框架训练时会报错，评测也无从下手）")
    if require_train:
        train_problems = [p for p in problems if p.startswith("train")]
        if not split_dirs.get("train"):
            train_problems.append("数据集 YAML 里没有 train 键")
        if train_problems:
            raise DatasetPathError(
                "训练集不可用：\n  - " + "\n  - ".join(train_problems)
                + f"\n  根目录：{root}（来历：{source}）\n"
                "最常见原因：新硬盘挂载点变了 / 数据还没拷过去。改 configs/datasets.yaml "
                "的 data_root 即可。"
            )
        warnings.extend(p for p in problems if not p.startswith("train"))

    return DatasetPaths(name=name, root=root, yaml=out_yaml, source=source,
                        splits=split_dirs, counts=counts,
                        label_counts=label_counts, warnings=tuple(warnings))


def _entry_by_yaml(registry: Mapping[str, Any], yaml_path: Path,
                   repo_root: Path) -> tuple[str, dict] | None:
    """反查：这个数据集 YAML 是否是注册表里某项的模板。"""
    target = _norm(yaml_path)
    for key, value in (registry.get("datasets") or {}).items():
        entry = _normalize_entry(value)
        template = _template_for(str(key), entry, repo_root)
        if template is not None and _norm(template) == target:
            return str(key), entry
    return _entry(registry, yaml_path.stem)


def _safe_name(name: str) -> str:
    """把数据集名转成安全的文件名（保留中文/字母/数字，其它换成下划线）。"""
    keep = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(name))
    return keep.strip("._") or "dataset"


def _count_lines(path: Path, limit: int = SCAN_LIMIT) -> int:
    """``train: list.txt`` 这种清单文件的行数。"""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return sum(1 for _ in fh) if path.stat().st_size < 8 << 20 else limit
    except OSError:
        return 0


def _count_txt(directory: Path, limit: int = SCAN_LIMIT) -> int:
    """标签目录里的 ``*.txt`` 数量（到上限就停）。"""
    n = 0
    stack = [directory]
    while stack and n < limit:
        try:
            with os.scandir(stack.pop()) as it:
                for item in it:
                    try:
                        if item.is_dir():
                            stack.append(Path(item.path))
                        elif item.name.endswith(".txt"):
                            n += 1
                            if n >= limit:
                                break
                    except OSError:
                        continue
        except OSError:
            continue
    return n


def describe(registry: dict | None = None, *, registry_path_hint: str | Path | None = None,
             repo_root: str | Path | None = None) -> list[str]:
    """把注册表列成一张表（``--list-datasets`` 用）：名字、根目录、是否存在、分划规模。"""
    root_repo = Path(repo_root or REPO_ROOT)
    registry = registry if registry is not None else load_registry(
        registry_path_hint, repo_root=root_repo)
    reg_file = registry_path(registry_path_hint, repo_root=root_repo)
    lines = [f"[注册表] {reg_file}"
             + (f"（+ 覆盖 {registry.get('_local_file')}）" if registry.get("_local_file") else "")]
    data_root = _as_path(registry.get("data_root") or ".", root_repo)
    lines.append(f"[注册表] data_root = {data_root}"
                 + ("" if data_root.is_dir() else "   ← 目录不存在，改这一行指向新硬盘"))
    for name, value in sorted((registry.get("datasets") or {}).items()):
        entry = _normalize_entry(value)
        try:
            paths = resolve(str(name), registry=registry, repo_root=root_repo)
        except DatasetPathError as exc:
            lines.append(f"  [ MISS ] {name}: {str(exc).splitlines()[0]}")
            continue
        assert paths is not None
        train = paths.counts.get("train", 0)
        val = paths.counts.get("val", 0)
        alias = f"（别名 {', '.join(str(a) for a in entry.get('aliases') or [])}）" \
            if entry.get("aliases") else ""
        lines.append(f"  [ ok  ] {name:<18} root={paths.root}  train={train:,}  val={val:,}{alias}")
    return lines
