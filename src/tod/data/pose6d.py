"""6D 位姿旁挂数据（EP0）—— 为论文式 (8) 的 ``L_rep`` 提供逐目标位姿。"""

from __future__ import annotations

import contextlib
import csv
from pathlib import Path
from typing import Any, Iterable

import torch

REQUIRED_COLUMNS = ("image", "obj_index", "tx", "ty", "tz", "qx", "qy", "qz", "qw")
POSE_DIM = 7


class Pose6DError(RuntimeError):
    """6D 位姿旁挂数据缺失或与标签不一致（**必须报错，不能静默跳过**）。"""


def quat_to_rot(q: torch.Tensor) -> torch.Tensor:
    """四元数 ``(..., 4)`` ``(qx,qy,qz,qw)`` → 旋转矩阵 ``(..., 3, 3)``。"""
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
    """把 CSV 行解析成 ``{图像名: [[7 个数], ...]}``（按 ``obj_index`` 排序并校验连续性）。"""
    table: dict[str, dict[int, list[float]]] = {}
    for lineno, row in enumerate(rows, start=2):
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


def _dataset_class():
    """拿到框架的 ``YOLODataset``（延迟导入，未装 ultralytics 时本模块仍可读 CSV）。"""
    from tod.compat import ensure_runtime_env

    ensure_runtime_env()
    from ultralytics.data.dataset import YOLODataset

    return YOLODataset


def pose6d_dataset(pose_dir: str | Path, split: str, strict: bool = True):
    """返回一个"带 6D 位姿"的 ``YOLODataset`` 子类（按 split 绑定 CSV）。"""
    base = _dataset_class()
    table = load_pose_csv(Path(pose_dir) / f"{split}.csv")

    class _Pose6DDataset(base):                          # type: ignore[misc, valid-type]
        """带 6D 位姿的 YOLO 姿态数据集。"""

        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(*args, **kwargs)
            self._pose_table = table
            self._pose_missing_images: list[str] = []
            self._image_sizes: dict[str, tuple[float, float]] = {}
            try:
                from PIL import Image

                for path in self.im_files:
                    with Image.open(path) as im:
                        self._image_sizes[Path(path).stem] = (float(im.size[0]),
                                                              float(im.size[1]))
            except Exception:  # noqa: BLE001
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
            # CSV 的 obj_index 必须与二维标签行序一致；缺失位姿由 mask 排除。
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


__all__ = ["Pose6DError", "POSE_DIM", "REQUIRED_COLUMNS", "load_pose_csv",
           "parse_pose_rows", "pose6d_dataset", "pose6d_dataset_context", "quat_to_rot"]


def pose_safe_augment_overrides() -> dict:
    """启用 6D 位姿监督（``L_rep``）时必须关掉的几何增强。"""
    # 这些几何增强不会同步更新 R、t 和内参，启用 L_rep 时必须关闭。
    return {
        "degrees": 0.0,
        "scale": 0.0,
        "translate": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "mosaic": 0.0,
        "close_mosaic": 0,
        "copy_paste": 0.0,
        "mixup": 0.0,
        "fliplr": 0.0,
        "flipud": 0.0,
    }


@contextlib.contextmanager
def pose6d_dataset_context(pose_dir: str | Path, split: str, strict: bool = True):
    """上下文管理器：把框架的 ``YOLODataset`` 暂时换成带 6D 位姿的子类。"""
    import ultralytics.data.build as build_module
    import ultralytics.data.dataset as ds_module

    dataset_cls = pose6d_dataset(pose_dir, split, strict=strict)
    # build 模块持有独立的全局引用，两个命名空间必须一起替换并在 finally 恢复。
    original = (ds_module.YOLODataset, build_module.YOLODataset)
    ds_module.YOLODataset = dataset_cls
    build_module.YOLODataset = dataset_cls
    try:
        yield dataset_cls
    finally:
        ds_module.YOLODataset, build_module.YOLODataset = original


def verify_pose_in_loader(loader: Any, source: str = "") -> None:
    """校验刚建好的 DataLoader 真的带上了 6D 位姿；没带上就直接报错。"""
    dataset = getattr(loader, "dataset", None)
    if dataset is None:
        return
    pose_cls = getattr(dataset, "__class__", None)
    if getattr(pose_cls, "__module__", "") == __name__:
        return
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
