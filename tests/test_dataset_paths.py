"""数据集注册表（``configs/datasets.yaml``）与路径解析的自检：不需要 torch / 框架。

守的是「训练集在哪」这条链路上最容易悄悄错的地方：

1. **相对 path 的锚点**。ultralytics 把数据集 YAML 里的相对 ``path`` 解析成
   ``<datasets_dir>/...``（不是 YAML 所在目录），所以必须由本库先把根目录算成**绝对路径**，
   再写一份副本交给框架 —— 测试直接断言副本里的 ``path`` 是绝对的、且等于解析出的根目录。
2. **换盘**。只改注册表的 ``data_root`` / 机器覆盖文件，解析结果就必须跟着变；
   数据不在盘上时必须**报错**而不是回落到仓库里那份旧的同名数据（静默训练错数据比崩更糟）。
3. **优先级**。``--data-root`` > ``$TOD_DATA_ROOT`` > 注册表 ``root`` > 数据集 YAML 自带的 path。
4. **三种 ``--data`` 写法**（注册表名 / 目录 / YAML）都要落到同一个根目录 ——
   否则同一份数据用不同写法会训练到不同地方。

用法::

    python3 tests/test_dataset_paths.py
    python3 tests/test_dataset_paths.py --keep      # 保留临时目录
"""

from __future__ import annotations

import os
import shutil
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

from tod.compat import dump_yaml, load_yaml          # noqa: E402
from tod.paths import (  # noqa: E402
    DatasetPathError,
    describe,
    load_registry,
    resolve,
)

PASSED: list[str] = []
FAILED: list[str] = []

WORK = ROOT / "tests" / ".tmp" / "test-dataset-paths"


def check(label: str, cond: bool, detail: str = "") -> None:
    if cond:
        PASSED.append(label)
        print(f"  ok    {label}")
    else:
        FAILED.append(f"{label} {detail}".strip())
        print(f"  FAIL  {label} {detail}")


def _write(path: Path, text: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _fake_dataset(root: Path, *, n_train: int = 3, n_val: int = 1,
                  with_labels: bool = True) -> Path:
    """造一个最小的 YOLO 目录（图/标签都是空文件，只验数量与路径，不验内容）。"""
    for split, n in (("train", n_train), ("val", n_val)):
        for i in range(n):
            _write(root / "images" / split / f"{i}.jpg")
            if with_labels:
                _write(root / "labels" / split / f"{i}.txt", "0 0.5 0.5 0.2 0.2\n")
    # 预处理产物那种"数据集自带 dataset.yaml"的形态（path 是转换时的绝对路径）
    dump_yaml({"path": str(root.resolve()), "train": "images/train", "val": "images/val",
               "nc": 1, "names": {0: "obj"}}, root / "dataset.yaml")
    return root


def _make_world(base: Path) -> tuple[Path, Path]:
    """造一套「仓库 + 新硬盘」的假世界，返回 (仓库根, 注册表路径)。"""
    repo = base / "repo"
    (repo / "configs").mkdir(parents=True, exist_ok=True)
    # 仓库里那份"旧的同名数据"：如果解析错，就会悄悄跑在这上面
    _fake_dataset(repo / "data" / "visdrone2019-det", n_train=99, n_val=99)
    # 新硬盘上的真数据
    _fake_dataset(base / "hdd" / "visdrone2019-det")
    _fake_dataset(base / "hdd" / "rflysim-pose", n_train=2, n_val=1)
    _fake_dataset(base / "hdd" / "mav6d" / "phantom4", n_train=4, n_val=2)
    # MAV6D 转换产物：dataset.yaml 里带 L_rep 需要的 kpt_shape
    dump_yaml({"path": str((base / "hdd" / "mav6d" / "phantom4").resolve()),
               "train": "images/train", "val": "images/val",
               "nc": 1, "names": {0: "uav"}, "kpt_shape": [9, 3]},
              base / "hdd" / "mav6d" / "phantom4" / "dataset.yaml")

    # 数据集模板（相对 path：正是会导致框架解析到 datasets_dir 的写法）
    _write(repo / "configs" / "_base_" / "datasets" / "visdrone2019-det.yaml",
           "path: ./data/visdrone2019-det\ntrain: images/train\nval: images/val\n"
           "nc: 1\nnames:\n  0: car\n")
    _write(repo / "configs" / "_base_" / "datasets" / "rflysim-pose.yaml",
           "path: ./data/rflysim-pose\ntrain: images/train\nval: images/val\n"
           "nc: 1\nnames:\n  0: uav\nkpt_shape: [9, 3]\nflip_idx: [0, 8, 7, 6, 5, 4, 3, 2, 1]\n")

    registry = _write(repo / "configs" / "datasets.yaml", "")
    dump_yaml({
        "data_root": str(base / "hdd"),
        "datasets": {
            "visdrone2019-det": {"root": "visdrone2019-det",
                                 "yaml": "configs/_base_/datasets/visdrone2019-det.yaml",
                                 "aliases": ["visdrone"]},
            "rflysim-pose": {"root": "rflysim-pose",
                             "yaml": "configs/_base_/datasets/rflysim-pose.yaml"},
            "mav6d-phantom4": {"root": "mav6d/phantom4", "aliases": ["phantom4"]},
            "not-copied-yet": {"root": "visdrone2019-obb"},
        },
    }, registry)
    return repo, registry


def test_registry_loading(repo: Path, registry: Path) -> None:
    print("\n[1] 注册表加载与机器覆盖")
    loaded = load_registry(registry, repo_root=repo)
    check("读到 data_root", str(loaded["data_root"]).replace("\\", "/").endswith("hdd"),
          str(loaded.get("data_root")))
    check("读到 4 个数据集", len(loaded["datasets"]) == 4, str(list(loaded["datasets"])))

    # 机器相关覆盖（configs/datasets.local.yaml）只改 data_root，其它字段保留
    _write(repo / "configs" / "datasets.local.yaml",
           f"data_root: {(WORK / 'hdd').as_posix()}\n"
           "datasets:\n  visdrone2019-det:\n    root: hdd-visdrone\n")
    merged = load_registry(registry, repo_root=repo)
    check("local 覆盖生效（data_root 换成 hdd）",
          str(merged["data_root"]).replace("\\", "/").endswith("hdd"), str(merged["data_root"]))
    check("local 覆盖是深合并（没被整段替换）",
          set(merged["datasets"]) == set(loaded["datasets"]), str(list(merged["datasets"])))
    check("local 记录的来源在注册表里", merged.get("_local_file"), str(merged.get("_local_file")))
    (repo / "configs" / "datasets.local.yaml").unlink()

    try:
        load_registry(repo / "configs" / "nope.yaml", repo_root=repo, required=True)
        check("注册表缺失时 required=True 报错", False)
    except DatasetPathError:
        check("注册表缺失时 required=True 报错", True)


def test_resolve_by_name(repo: Path, registry: Path) -> None:
    print("\n[2] 按注册表名字解析（换盘的核心路径）")
    paths = resolve("visdrone2019-det", registry_path_hint=registry,
                    repo_root=repo, require_train=True)
    assert paths is not None
    hdd = (WORK / "hdd" / "visdrone2019-det").resolve()
    check("根目录=新硬盘上的目录", paths.root == hdd, f"{paths.root} != {hdd}")
    check("没有回落到仓库里的旧数据", "repo" not in str(paths.root), str(paths.root))
    check("train 张数=3、val 张数=1",
          paths.counts.get("train") == 3 and paths.counts.get("val") == 1, str(paths.counts))
    check("标签张数与图一致", paths.label_counts.get("train") == 3, str(paths.label_counts))

    cfg = load_yaml(paths.yaml)
    check("交给框架的 YAML 里 path 是**绝对**路径", Path(str(cfg["path"])).is_absolute(),
          str(cfg.get("path")))
    check("path 等于解析出的根目录", Path(str(cfg["path"])).resolve() == hdd, str(cfg.get("path")))
    check("副本仍带类别口径（nc/names 没丢）",
          cfg.get("nc") == 1 and list((cfg.get("names") or {}).values()) == ["car"], str(cfg))
    check("副本落在缓存目录里（不污染仓库配置目录）",
          paths.yaml.is_file() and ".cache" in str(paths.yaml), str(paths.yaml))

    alias = resolve("visdrone", registry_path_hint=registry, repo_root=repo)
    assert alias is not None
    check("别名解析到同一根目录", alias.root == hdd, str(alias.root))

    pose = resolve("rflysim-pose", registry_path_hint=registry, repo_root=repo)
    assert pose is not None
    pose_cfg = load_yaml(pose.yaml)
    check("姿态数据集的 kpt_shape 跟着模板走", pose_cfg.get("kpt_shape") == [9, 3],
          str(pose_cfg.get("kpt_shape")))


def test_three_ways_agree(repo: Path, registry: Path) -> None:
    print("\n[3] 三种 --data 写法必须落到同一个根目录")
    hdd = (WORK / "hdd" / "visdrone2019-det").resolve()
    by_name = resolve("visdrone2019-det", registry_path_hint=registry, repo_root=repo)
    by_yaml = resolve(repo / "configs" / "_base_" / "datasets" / "visdrone2019-det.yaml",
                      registry_path_hint=registry, repo_root=repo)
    by_dir = resolve(WORK / "hdd" / "visdrone2019-det", registry_path_hint=registry,
                     repo_root=repo)
    for label, got in (("名字", by_name), ("模板 YAML（反查注册表）", by_yaml),
                       ("数据集目录", by_dir)):
        assert got is not None
        check(f"{label} → 新硬盘根目录", got.root == hdd, f"{got.root} != {hdd}")

    # 目录方式要求目录里自带 dataset.yaml（MAV6D 转换产物就是这样）
    mav = resolve("mav6d-phantom4", registry_path_hint=registry, repo_root=repo,
                  require_train=True)
    assert mav is not None
    check("没有模板的数据集用 <root>/dataset.yaml",
          mav.yaml.is_file() and load_yaml(mav.yaml).get("kpt_shape") == [9, 3],
          str(mav.yaml))


def test_priority(repo: Path, registry: Path) -> None:
    print("\n[4] 优先级：--data-root > $TOD_DATA_ROOT > 注册表")
    alt = WORK / "hdd2"
    _fake_dataset(alt / "visdrone2019-det", n_train=7, n_val=2)

    cli = resolve("visdrone2019-det", registry_path_hint=registry, repo_root=repo,
                  data_root=alt)
    assert cli is not None
    check("--data-root 覆盖注册表", cli.root == (alt / "visdrone2019-det").resolve(),
          str(cli.root))

    os.environ["TOD_DATA_ROOT"] = str(alt)
    try:
        env = resolve("visdrone2019-det", registry_path_hint=registry, repo_root=repo)
        both = resolve("visdrone2019-det", registry_path_hint=registry, repo_root=repo,
                       data_root=WORK / "hdd")
    finally:
        os.environ.pop("TOD_DATA_ROOT", None)
    assert env is not None and both is not None
    check("$TOD_DATA_ROOT 覆盖注册表", env.root == (alt / "visdrone2019-det").resolve(),
          str(env.root))
    check("--data-root 又盖过 $TOD_DATA_ROOT",
          both.root == (WORK / "hdd" / "visdrone2019-det").resolve(), str(both.root))

    # 数据集 YAML 里写的相对 path 必须按**仓库根**解析（不是框架的 datasets_dir）
    outside = _write(WORK / "loose.yaml",
                     "path: data/visdrone2019-det\ntrain: images/train\nval: images/val\n"
                     "nc: 1\nnames:\n  0: car\n")
    loose = resolve(outside, registry_path_hint=registry, repo_root=repo)
    assert loose is not None
    check("YAML 里的相对 path 锚在仓库根",
          loose.root == (repo / "data" / "visdrone2019-det").resolve(), str(loose.root))


def test_errors(repo: Path, registry: Path) -> None:
    print("\n[5] 出错时必须报错（而不是静默用错的目录）")

    try:
        resolve("不存在的数据集", registry_path_hint=registry, repo_root=repo)
        check("未知数据集名 → 报错", False)
    except DatasetPathError as exc:
        msg = str(exc)
        check("未知数据集名 → 报错", "visdrone2019-det" in msg, msg.splitlines()[0])
        check("报错里列出可用名字", "可用的名字" in msg, msg.splitlines()[-1])

    try:
        resolve("not-copied-yet", registry_path_hint=registry, repo_root=repo,
                require_train=True)
        check("根目录不存在 → 报错", False)
    except DatasetPathError as exc:
        msg = str(exc)
        check("根目录不存在 → 报错", "不存在" in msg, msg.splitlines()[0])
        check("报错里指出怎么修（改 data_root / --data-root / 环境变量）",
              "data_root" in msg and "--data-root" in msg, msg.splitlines()[-1])

    # 图在、标签空：只警告；图 0 张：require_train 必报错
    empty = _fake_dataset(WORK / "hdd" / "empty-set", n_train=0, n_val=0)
    _write(empty / "images" / "train" / ".keep")
    try:
        resolve(empty, require_train=True)
        check("训练集 0 张图 → require_train 报错", False)
    except DatasetPathError as exc:
        check("训练集 0 张图 → require_train 报错", "没有图片" in str(exc),
              str(exc).splitlines()[1])

    nolab = _fake_dataset(WORK / "hdd" / "nolab", n_train=2, n_val=1,
                          with_labels=False)
    warned = resolve(nolab, require_train=True)
    assert warned is not None
    check("有图无标签 → 警告（不中断）",
          any("标签" in w for w in warned.warnings), str(warned.warnings))


def test_describe(repo: Path) -> None:
    print("\n[6] --list-datasets 的表格")
    lines = describe(registry_path_hint=repo / "configs" / "datasets.yaml", repo_root=repo)
    text = "\n".join(lines)
    check("表里列出已就绪的数据集（ok）", "[ ok  ] visdrone2019-det" in text, text)
    check("表里把没拷过去的数据集标成 MISS", "[ MISS ] not-copied-yet" in text, text)
    check("表里显示 data_root", "data_root" in text, text.splitlines()[1])


def main() -> int:
    if WORK.exists():
        shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    repo, registry = _make_world(WORK)

    print(f"[paths] 假仓库：{repo}")
    print(f"[paths] 假新硬盘：{WORK / 'hdd'}")
    test_registry_loading(repo, registry)
    test_resolve_by_name(repo, registry)
    test_three_ways_agree(repo, registry)
    test_priority(repo, registry)
    test_errors(repo, registry)
    test_describe(repo)

    if "--keep" not in sys.argv:
        shutil.rmtree(WORK, ignore_errors=True)

    print(f"\n通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    for item in FAILED:
        print(f"  [FAIL] {item}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
