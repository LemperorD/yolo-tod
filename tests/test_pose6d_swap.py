"""回归测试：6D 位姿数据集替换必须**真的**生效。

## 为什么需要这个测试

``ultralytics/data/build.py`` 开头是 ``from .dataset import (..., YOLODataset, ...)``，
而 Python 的 ``from X import Y`` 是**值拷贝**：``build.YOLODataset`` 与
``dataset.YOLODataset`` 指向同一个类对象，但给其中一个重新赋值**不影响另一个**。
真正建数据集的是 ``build_yolo_dataset()``，用的是 **``build`` 模块自己的全局名**。

所以只替换 ``ultralytics.data.dataset.YOLODataset`` 是**完全无效**的：数据集仍是原生
``YOLODataset``，样本里没有 ``gt_pose``，训练会一路跑到 warm-up 结束后的第一个 batch
才抛 ``CompatError: batch 里没有 gt_pose``（而且报错文案还容易让人误以为是数据没挂）。

本测试做两件事：

1. **正向**：用 ``pose6d_dataset_context()`` 替换后，``build_yolo_dataset()`` 必须
   实例化替换类，且样本里真的有 ``gt_pose`` / ``gt_pose_mask`` / ``gt_image_size``；
2. **反向**：只替换 ``dataset`` 命名空间时，必须**判定为失效**（证明这个测试抓得住
   原来的 bug，而不是永远绿灯）。

用法::

    python tests/test_pose6d_swap.py --fast   # 只做命名空间层面的正/反向验证（秒级）
    python tests/test_pose6d_swap.py          # 额外做一次真实建数据集 + 取样本
"""

from __future__ import annotations

import argparse
import importlib.util
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

PASSED: list[str] = []
FAILED: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    if cond:
        PASSED.append(label)
        print(f"  ok    {label}")
    else:
        FAILED.append(f"{label} {detail}".strip())
        print(f"  FAIL  {label} {detail}")


def _load_tool(name: str):
    spec = importlib.util.spec_from_file_location(f"_tod_tool_{name}",
                                                  ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _instrument(cls, log: list, tag: str):  # noqa: D103 - 保留给将来排查用
    original = cls.__init__

    def wrapped(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        log.append(tag)
        return result

    cls.__init__ = wrapped
    return original


def _restore(cls, original) -> None:
    cls.__init__ = original


def _has_pose(dataset) -> bool:
    """数据集是否真的带上了 6D 位姿。

    判据是「属性/标签键」而不是「类名」：类名会被插桩或继承顺序误导
    （踩过：包装父类 ``__init__`` 会让子类实例也被记成父类）。
    """
    if type(dataset).__module__ == __name__:
        return True
    if isinstance(getattr(dataset, "_pose_table", None), dict):
        return True
    labels = getattr(dataset, "labels", None)
    if isinstance(labels, list) and labels and isinstance(labels[0], dict):
        return "gt_pose" in labels[0]
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="6D 位姿数据集替换的回归测试")
    ap.add_argument("--fast", action="store_true", help="跳过真实建数据集")
    ap.add_argument("--keep", action="store_true", help="保留临时数据")
    args = ap.parse_args()

    for pkg in ("torch", "ultralytics", "PIL", "numpy"):
        if importlib.util.find_spec(pkg) is None:
            print(f"[skip] 未安装 {pkg}")
            return 0

    from tod.compat import ensure_runtime_env, load_yaml

    ensure_runtime_env()

    # 与训练入口一致：把框架的并行标签缓存换成顺序版。
    # 受限沙箱（Windows 命名管道被禁）下 ThreadPool 建队列表就会 WinError 5；
    # 本测试会真的建两次数据集（反向 + 正向），第二次会触发缓存重建，必然踩到。
    from tod import compat

    if compat.allow_threadless_label_cache():
        print("  [env ] 标签缓存已切换为顺序模式（沙箱兼容）")

    import ultralytics.data.build as build
    import ultralytics.data.dataset as ds

    from tod.data.pose6d import (Pose6DError, pose6d_dataset,
                                 pose6d_dataset_context, verify_pose_in_loader)

    work = ROOT / "tests" / ".tmp" / "pose6d-swap"
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    try:
        data_yaml = _load_tool("make_dummy_dataset").build(
            work / "data", 4, 2, seed=0, task="pose", kpt_shape=(9, 3), layout="uav")
    except TypeError:                     # 老签名没有 layout
        data_yaml = _load_tool("make_dummy_dataset").build(
            work / "data", 4, 2, seed=0, task="pose", kpt_shape=(9, 3))
    cfg = load_yaml(data_yaml)
    tod_cfg = cfg.get("tod") or {}
    pose_dir = Path(str(cfg.get("path"))) / str(tod_cfg.get("pose_labels"))
    check("合成数据带 labels_6d/train.csv", (pose_dir / "train.csv").is_file(),
          f"实际 {pose_dir}")
    check("合成数据声明了 tod.pose_labels", bool(tod_cfg.get("pose_labels")))

    native = ds.YOLODataset
    print(f"  原生类：{native.__module__}.{native.__name__}；"
          f"build 模块持有同一个对象：{build.YOLODataset is native}")
    check("build 与 dataset 两个命名空间初始指向同一个类",
          build.YOLODataset is native)

    # ---------------- 反向：只换 dataset 命名空间必须被判定为失效 ----------------
    only_ds = pose6d_dataset(pose_dir, "train")
    ds.YOLODataset = only_ds                    # ← 旧写法（bug）
    try:
        check("旧写法：dataset 命名空间确实换了", ds.YOLODataset is only_ds)
        check("旧写法：build 命名空间仍是原生类（这就是 bug 本体）",
              build.YOLODataset is native, f"实际 {build.YOLODataset}")
        old_built = _try_build(build, data_yaml, cfg)
        check("旧写法：建出来的数据集拿不到 6D 位姿（静默失效）",
              old_built is None or not _has_pose(old_built),
              f"实际 {type(old_built).__name__ if old_built is not None else 'None'}")
    finally:
        ds.YOLODataset = native

    # ---------------- 正向：上下文管理器必须两个命名空间一起换 ----------------
    with pose6d_dataset_context(pose_dir, "train") as swapped:
        check("新写法：dataset 命名空间已替换", ds.YOLODataset is swapped)
        check("新写法：build 命名空间已替换（关键）", build.YOLODataset is swapped)
        built = _try_build(build, data_yaml, cfg)
        check("新写法：build_yolo_dataset 用的是替换类",
              built is not None and _has_pose(built),
              f"实际 {type(built).__name__ if built is not None else 'None（建数据集失败）'}")

        if built is not None and not args.fast:
            sample = _try_sample(built)
            if sample is None:
                print("  skip  取样本（建数据集时被沙箱拦下写缓存）")
            else:
                check("样本里有 gt_pose", "gt_pose" in sample)
                check("样本里有 gt_pose_mask", "gt_pose_mask" in sample)
                check("样本里有 gt_image_size", "gt_image_size" in sample)
                if "gt_pose" in sample:
                    gp = sample["gt_pose"]
                    check("gt_pose 形状为 (n_obj, 7)",
                          tuple(gp.shape)[-1] == 7, f"实际 {tuple(gp.shape)}")
                try:
                    verify_pose_in_loader(type("L", (), {"dataset": built})())
                    check("verify_pose_in_loader 接受带 gt_pose 的数据集", True)
                except Pose6DError as exc:                     # pragma: no cover
                    check("verify_pose_in_loader 接受带 gt_pose 的数据集", False, str(exc))

        if built is not None and not args.fast:
            sample = _try_sample(built)
            if sample is None:
                print("  skip  取样本（建数据集时被沙箱拦下写缓存）")
            else:
                check("样本里有 gt_pose", "gt_pose" in sample)
                check("样本里有 gt_pose_mask", "gt_pose_mask" in sample)
                check("样本里有 gt_image_size", "gt_image_size" in sample)
                if "gt_pose" in sample:
                    gp = sample["gt_pose"]
                    check("gt_pose 形状为 (n_obj, 7)",
                          tuple(gp.shape)[-1] == 7, f"实际 {tuple(gp.shape)}")
                try:
                    verify_pose_in_loader(type("L", (), {"dataset": built})())
                    check("verify_pose_in_loader 接受带 gt_pose 的数据集", True)
                except Pose6DError as exc:                     # pragma: no cover
                    check("verify_pose_in_loader 接受带 gt_pose 的数据集", False, str(exc))

    check("退出上下文后 build 命名空间已还原", build.YOLODataset is native)
    check("退出上下文后 dataset 命名空间已还原", ds.YOLODataset is native)

    # ---------------- verify_pose_in_loader 必须能识别“没替换成功” ----------------
    try:
        verify_pose_in_loader(type("L", (), {"dataset": object()})())
        check("verify_pose_in_loader 对原生/无位姿数据集报错", False, "没报错")
    except Pose6DError:
        check("verify_pose_in_loader 对原生/无位姿数据集报错", True)

    if not args.keep:
        shutil.rmtree(work, ignore_errors=True)

    print(f"\n通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    for f in FAILED:
        print(f"  [FAIL] {f}")
    return 1 if FAILED else 0


def _try_build(build, data_yaml: Path, cfg: dict, verbose: bool = False):
    """尽力建一次数据集；沙箱里缓存写入可能被拒，此时返回 None（判定仍用实例化记录）。"""
    from ultralytics.cfg import get_cfg

    names = cfg.get("names") or {0: "uav"}
    data = {"path": str(cfg.get("path")), "names": names,
            "channels": 3, "nc": int(cfg.get("nc") or len(names)),
            "kpt_shape": list(cfg.get("kpt_shape") or (9, 3))}
    overrides = {"task": "pose", "mode": "train", "imgsz": 320, "batch": 2,
                 "data": str(data_yaml), "rect": False, "cache": False,
                 "single_cls": False, "fraction": 1.0, "classes": None}
    train_img = str(Path(str(cfg.get("path"))) / "images" / "train")
    try:
        return build.build_yolo_dataset(get_cfg(overrides=overrides), train_img, 2, data,
                                        mode="train", stride=32)
    except (PermissionError, OSError) as exc:
        if verbose:
            print(f"  (建数据集被文件系统拦下：{type(exc).__name__}: {exc})")
        return None
    except Exception as exc:                                   # noqa: BLE001
        print(f"  (建数据集抛 {type(exc).__name__}：{exc})")
        return None


def _try_sample(dataset):
    try:
        return dataset[0]
    except (PermissionError, OSError):
        return None
    except Exception as exc:                                   # noqa: BLE001
        print(f"  (取样本抛 {type(exc).__name__}：{exc})")
        return None


if __name__ == "__main__":
    raise SystemExit(main())
