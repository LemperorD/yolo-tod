"""Collect 6D evaluation inputs during the regular pose validation pass."""

from pathlib import Path

from ultralytics.models.yolo.pose import PoseValidator
from ultralytics.utils import ops


class TODPoseValidator(PoseValidator):
    pose_rows = None
    pose_samples = None

    def _process_batch(self, preds, batch):
        if self.pose_rows is not None:
            self._collect_pose_sample(preds, batch)
        return super()._process_batch(preds, batch)

    def _collect_pose_sample(self, preds, batch):
        rows = self.pose_rows.get(Path(batch["im_file"]).stem, [])
        if not rows or not len(batch["cls"]):
            return
        gts = [
            {"cls": int(cls), "box": box}
            for cls, box in zip(batch["cls"].tolist(), batch["bboxes"].tolist())
        ]
        # Match in validation input coordinates. PnP uses the original image intrinsics.
        keep = preds["conf"] >= 0.25
        points = ops.scale_coords(
            batch["imgsz"], preds["keypoints"][keep, :, :2],
            batch["ori_shape"], ratio_pad=batch["ratio_pad"],
        )
        predictions = [
            (int(cls), conf, box, kpts)
            for cls, conf, box, kpts in zip(
                preds["cls"][keep].tolist(), preds["conf"][keep].tolist(),
                preds["bboxes"][keep].tolist(), points.tolist(),
            )
        ]
        self.pose_samples.append({"gts": gts, "preds": predictions, "poses": rows})
