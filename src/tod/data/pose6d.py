"""6D 位姿旁挂数据（EP0）—— 为论文式 (8) 的 ``L_rep`` 提供逐目标位姿。

## 数据格式（与 ``docs/KEYPOINTS.md`` §5.2 一致）

``<path>/labels_6d/<split>.csv``，带表头::

    image,obj_index,tx,ty,tz,qx,qy,qz,qw
    train_0000,0,0.12,-0.03,1.85,0.02,-0.01,0.71,0.70
    train_0000,1,0.20, 0.01,2.10,0.00, 0.00,0.02,1.00

* ``image``：图像文件名（**不含扩展名**）；
* ``obj_index``：该图内目标序号（0 起），**必须与标签文件里的行序一致**；
* ``t``：机体→相机的平移（米）；``q``：旋转四元数 ``(qx, qy, qz, qw)``，机体→相机。

## 为什么行序必须对齐（以及本库怎么防它）

``ultralytics`` 的 ``YOLODataset`` 把一张图的所有目标**按标签行序**打包成一个
``Instances`` 数组；``L_rep`` 又是通过框架分配器给出的 ``target_gt_idx``
（指向"该图第 i 个 GT"）来取位姿的。所以 CSV 的 ``obj_index`` 只要错位一格，
几何监督就会配到别的目标上 —— 而**训练仍然能跑**（损失照样下降），
属于最难发现的静默错误。因此本库：

* 读取时校验 ``obj_index`` 从 0 连续、无重复；
* ``Pose6DDataset`` 在每张图上校验"CSV 行数 == 标签行数"，不一致**直接报错**；
* ``tools/check_pose_dataset.py`` 会做同样的检查（开训前就能拦住）。
"""

from __future__ import annotations

import contextlib
import csv
from pathlib import Path
from typing import Any, Iterable

import torch

#: CSV 必需列（顺序不限，按表头定位）。
REQUIRED_COLUMNS = ("image", "obj_index", "tx", "ty", "tz", "qx", "qy", "qz", "qw")
#: 每目标位姿向量的长度：3 平移 + 4 四元数。
POSE_DIM = 7


class Pose6DError(RuntimeError):
    """6D 位姿旁挂数据缺失或与标签不一致（**必须报错，不能静默跳过**）。"""


def quat_to_rot(q: torch.Tensor) -> torch.Tensor:
    """四元数 ``(..., 4)`` ``(qx,qy,qz,qw)`` → 旋转矩阵 ``(..., 3, 3)``。

    与 ``tod.loss.reproj`` 里的实现同式（那边为了不引入 data 依赖单独写了一份；
    两处都有测试逐元素对照，改动时请一起改）。
    """
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    two = 2.0
    row0 = torch.stack([1 - two * (y * y + z * z), two * (x * y - z * w),
                        two * (x * z + y * w)], dim=-1)
    row1 = torch.stack([two * (x * y + z * w), 1 - two * (x * x + z * z),
                        two * (y * z - x * w)], dim=-1)
    row2 = torch.stack([two * (x * z - y * w), two * (y * z + x * w),
                        1 - two * (x * x + y * y)], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def parse_pose_rows(rows: Iterable[dict[str, str]], source: str = "") -> dict[str, list[list[float]]]:
    """把 CSV 行解析成 ``{图像名: [[7 个数], ...]}``（按 ``obj_index`` 排序并校验连续性）。

    Raises:
        Pose6DError: 缺列、缺图像键、``obj_index`` 不从 0 连续或重复、数值非法。
    """
    table: dict[str, dict[int, list[float]]] = {}
    for lineno, row in enumerate(rows, start=2):          # 2 = 表头占第 1 行
        missing = [c for c in REQUIRED_COLUMNS if c not in row]
        if missing:
            raise Pose6DError(
                f"{source}:{lineno} 缺少列 {missing}；表头必须是 {list(REQUIRED_COLUMNS)}。"
            )
        stem = str(row["image"]).strip()
        if not stem:
            raise Pose6DError(f"{source}:{lineno} 的 image 为空。")
        try:
            idx = int(float(row["obj_index"]))
            values = [float(row[k]) for k in
                      ("tx", "ty", "tz", "qx", "qy", "qz", "qw")]
        except (TypeError, ValueError) as exc:
            raise Pose6DError(f"{source}:{lineno} 数值非法：{exc}") from exc
        norm = sum(v * v for v in values[3:]) ** 0.5
        if norm < 1e-6:
            raise Pose6DError(
                f"{source}:{lineno} 四元数 (qx,qy,qz,qw) 的模≈0，不是合法旋转。"
            )
        bucket = table.setdefault(stem, {})
        if idx in bucket:
            raise Pose6DError(f"{source}:{lineno} 图像 {stem!r} 的 obj_index={idx} 重复。")
        bucket[idx] = values

    out: dict[str, list[list[float]]] = {}
    for stem, bucket in table.items():
        expect = list(range(len(bucket)))
        if sorted(bucket) != expect:
            raise Pose6DError(
                f"{source} 图像 {stem!r} 的 obj_index 必须是 0..{len(bucket) - 1} 连续，"
                f"实际 {sorted(bucket)}；行序错位会让几何监督配到别的目标上（静默错误）。"
            )
        out[stem] = [bucket[i] for i in expect]
    return out


def load_pose_csv(path: str | Path) -> dict[str, list[list[float]]]:
    """读取 6D 位姿 CSV；文件不存在时抛 ``Pose6DError``（不静默返回空）。"""
    path = Path(path)
    if not path.is_file():
        raise Pose6DError(
            f"找不到 6D 位姿文件 {path}。EP7 的 L_rep 需要它（格式见 docs/KEYPOINTS.md §5.2），"
            "或者把变体里的 EP7.reproj 关掉。"
        )
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        return parse_pose_rows(csv.DictReader(fh), source=str(path))


# ------------------------------------------------------------------ 数据集


def _dataset_class():
    """拿到框架的 ``YOLODataset``（延迟导入，未装 ultralytics 时本模块仍可读 CSV）。"""
    from tod.compat import ensure_runtime_env

    ensure_runtime_env()
    from ultralytics.data.dataset import YOLODataset

    return YOLODataset


def pose6d_dataset(pose_dir: str | Path, split: str, strict: bool = True):
    """返回一个"带 6D 位姿"的 ``YOLODataset`` 子类（按 split 绑定 CSV）。

    Args:
        pose_dir: ``labels_6d`` 目录（内含 ``<split>.csv``）。
        split: ``train`` / ``val`` / ...
        strict: ``True``（默认）时"CSV 行数与标签行数不一致"直接报错；
            ``False`` 时只把缺失的目标标记为无效（``gt_pose_mask=False``）并继续，
            用于数据还在补齐的阶段（此时日志里会有警告）。

    Returns:
        一个类，可直接传给 ``build_yolo_dataset`` 或作为 ``dataset`` 覆写用。
    """
    base = _dataset_class()
    table = load_pose_csv(Path(pose_dir) / f"{split}.csv")

    class _Pose6DDataset(base):                          # type: ignore[misc, valid-type]
        """带 6D 位姿的 YOLO 姿态数据集。"""

        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(*args, **kwargs)
            self._pose_table = table
            self._pose_missing_images: list[str] = []
            # 原图尺寸：``L_rep`` 需要它把"原始相机内参"换算到增强后的尺度
            # （letterbox 是等比缩放，把 K 按同一比例缩放即可，补边不影响射线方向）。
            self._image_sizes: dict[str, tuple[float, float]] = {}
            try:
                from PIL import Image

                for path in self.im_files:
                    with Image.open(path) as im:
                        self._image_sizes[Path(path).stem] = (float(im.size[0]),
                                                              float(im.size[1]))
            except Exception:  # noqa: BLE001 - 取不到就退化为不做缩放校正
                self._image_sizes = {}

        def __getitem__(self, index: int) -> dict:
            labels = super().__getitem__(index)
            stem = Path(self.im_files[index]).stem
            n_obj = len(labels["cls"])
            poses = self._pose_table.get(stem)
            if poses is None:
                self._pose_missing_images.append(stem)
                poses = []
            if len(poses) != n_obj:
                msg = (
                    f"图像 {stem!r}：labels_6d 里有 {len(poses)} 个目标的位姿，"
                    f"但标签文件里有 {n_obj} 个 GT 框（第 {index} 张）。"
                    "两者必须一一对应（行序 = 标签行序），否则几何监督会配错目标。"
                )
                if strict:
                    raise Pose6DError(msg)
                self._pose_missing_images.append(stem)
            # 逐目标位姿与有效性掩码：labels 字典里的键会被 collate_fn 收集，
            # 框架在 build_targets 里用 target_gt_idx 索引到"该图第 i 个 GT"。
            gt_pose = torch.zeros(n_obj, POSE_DIM, dtype=torch.float32)
            mask = torch.zeros(n_obj, dtype=torch.bool)
            for i in range(min(n_obj, len(poses))):
                gt_pose[i] = torch.tensor(poses[i], dtype=torch.float32)
                mask[i] = True
            labels["gt_pose"] = gt_pose
            labels["gt_pose_mask"] = mask
            size = self._image_sizes.get(stem)
            labels["gt_image_size"] = torch.tensor(size or (0.0, 0.0), dtype=torch.float32)
            return labels

    _Pose6DDataset.__name__ = f"Pose6DDataset[{split}]"
    _Pose6DDataset.__qualname__ = _Pose6DDataset.__name__
    return _Pose6DDataset


# ------------------------------------------------------------------ 替换生效
# 下面的辅助必须放在 pose6d_dataset 之后（它们要用到它）。

__all__ = ["Pose6DError", "POSE_DIM", "REQUIRED_COLUMNS", "load_pose_csv",
           "parse_pose_rows", "pose6d_dataset", "pose6d_dataset_context", "quat_to_rot"]


def pose_safe_augment_overrides() -> dict:
    """启用 6D 位姿监督（``L_rep``）时必须关掉的几何增强。

    ## 为什么必须关

    ``labels_6d`` 里的位姿是**按原图逐目标**给的，数据集侧强校验
    "CSV 行数 == 标签行数"。但框架的 ``RandomPerspective`` 在增强后会把
    "变换后面积占比 < 10%" 的框**连同类一起丢掉**（``apply_instances`` 里的
    ``box_candidates`` + ``cls[i]``）。实测 MAV6D（``scale=0.4`` / ``degrees=10`` /
    ``translate=0.1``）：4000 张里有 **27 张（0.68%）** 被丢成 0 个框 ——
    于是"1 行位姿 vs 0 个框"，严格模式下必然报错。

    这不是数据错误，是增强的合法行为；而一旦丢了框，就无法可靠认定
    "CSV 的哪几行还对应得上"（框架不暴露保留索引）。所以本库的口径是：
    **要几何监督，就关掉会丢目标的几何增强**。

    保留的增强：``fliplr``（配好 ``flip_idx`` 时安全，位姿本身与翻转无关）、
    颜色抖动（``hsv_*``）、``erasing`` 等不影响框数的项。

    Returns:
        可直接展开进 ``model.train(**overrides)`` 的字典。
    """
    return {
        "degrees": 0.0,        # 旋转会把目标转出画面
        "scale": 0.0,          # 缩放会改变目标面积占比 -> 触发 box_candidates 丢弃
        "translate": 0.0,      # 平移同理
        "shear": 0.0,
        "perspective": 0.0,
        "mosaic": 0.0,         # 拼图：框与关键点变 4 倍，位姿无法跟随
        "close_mosaic": 0,
        "copy_paste": 0.0,
        "mixup": 0.0,
    }


@contextlib.contextmanager
def pose6d_dataset_context(pose_dir: str | Path, split: str, strict: bool = True):
    """上下文管理器：把框架的 ``YOLODataset`` 暂时换成带 6D 位姿的子类。

    **必须同时替换两个命名空间** —— 这是踩过的坑：

    ``ultralytics/data/build.py`` 开头是 ``from .dataset import (..., YOLODataset, ...)``，
    而 Python 的 ``from X import Y`` 是**值拷贝**：``build.YOLODataset`` 与
    ``dataset.YOLODataset`` 初始指向同一个类对象，但给其中一个重新赋值**不会影响另一个**。
    真正建数据集的是 ``build_yolo_dataset()``，它用的是 **``build`` 模块自己的全局名**。
    只换 ``dataset`` 那个名字的话替换**完全无效**：数据集仍是原生 ``YOLODataset``，
    样本里没有 ``gt_pose``，于是训练一路跑到 ``ReprojCriterion`` 才炸
    （"batch 里没有 gt_pose"），而且是在 warm-up 结束后才炸，极难定位。

    用法::

        with pose6d_dataset_context(pose_dir, "train") as cls:
            loader = build(...)          # 期间生效

    Args:
        pose_dir: ``labels_6d`` 目录（内含 ``<split>.csv``）。
        split: ``train`` / ``val`` / ...
        strict: 见 :func:`pose6d_dataset`。

    Yields:
        替换用的数据集类（可直接断言 ``build_module.YOLODataset is cls``）。
    """
    import ultralytics.data.build as build_module
    import ultralytics.data.dataset as ds_module

    dataset_cls = pose6d_dataset(pose_dir, split, strict=strict)
    original = (ds_module.YOLODataset, build_module.YOLODataset)
    ds_module.YOLODataset = dataset_cls
    build_module.YOLODataset = dataset_cls
    try:
        yield dataset_cls
    finally:
        ds_module.YOLODataset, build_module.YOLODataset = original


def verify_pose_in_loader(loader: Any, source: str = "") -> None:
    """校验刚建好的 DataLoader 真的带上了 6D 位姿；没带上就直接报错。

    "替换静默失效"是本模块最危险的一类故障：训练能正常开跑，直到 warm-up 结束后
    第一个 batch 才抛 ``CompatError``。这里在**建完 loader 的当下**就检查，
    把故障点前移到一眼能看懂的地方。
    """
    dataset = getattr(loader, "dataset", None)
    if dataset is None:
        return                                    # 拿不到（如某些自定义 loader）就不拦
    pose_cls = getattr(dataset, "__class__", None)
    if getattr(pose_cls, "__module__", "") == __name__:
        return                                    # 已是本模块的子类，必然带 gt_pose
    probe = getattr(dataset, "_pose_table", None)
    if isinstance(probe, dict):
        return
    sample = None
    for attr in ("labels", "samples", "im_files"):
        seq = getattr(dataset, attr, None)
        if isinstance(seq, list) and seq and isinstance(seq[0], dict):
            sample = seq[0]
            break
    if sample is not None and "gt_pose" in sample:
        return
    raise Pose6DError(
        f"数据集替换没有生效{source and f'（{source}）'}："
        f"建出来的 dataset 是 {type(dataset).__module__}.{type(dataset).__name__}，"
        "样本里没有 'gt_pose'。\n"
        "常见原因：只替换了 ultralytics.data.dataset.YOLODataset，而 build_yolo_dataset() "
        "用的是 ultralytics.data.build 自己的全局名 —— 请改用 pose6d_dataset_context()。"
    )

