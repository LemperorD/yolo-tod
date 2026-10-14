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
