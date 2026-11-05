"""SPAE-YOLOv8n 变体配方（论文架构的严格复现）。"""

from __future__ import annotations

from tod.compose import Variant

P2_PRE = ["Conv", [128, 1, 1]]

variant = (
    Variant(
        "SPAE-YOLOv8n",
        base="yolov8n",
        tags=["paper-repro", "small-object", "lightweight", "uav"],
        notes=(
            "论文原设定：Det-Fly 数据集、batch=32、SGD、200 epochs；"
            "本变体迁移到 VisDrone2019-DET 且 batch 降到 8（8GB 显存），"
            "属于域迁移 + 超参改动，论文 mAP 数字不可直接引用。\n\n"
            "论文消融（对 baseline 0.850）：P2 +7.5pp（主贡献）、ADown +0.4pp、"
            "SIoU +0.1pp、Efficient_UAVDet −0.4pp（论文自认是压缩/加速手段）。\n\n"
            "引用论文数字时注意其表间冲突：Table 6 baseline 记 0.850，"
            "而 Table 5/7 记 0.922；同一配置 FPS 在 Table 6 为 203.0、Table 5 为 161.5。"
        ),
    )
    .backbone(downsample="ADown", downsample_indices=[1, 3, 5, 7])
    .head("Efficient_UAVDet", per_group=16, channels="native", levels=[2, 3, 4, 5])
    .loss(box="siou", theta=4.0)
    .data("visdrone2019-det", imgsz=640)
    .model(
        nc=10,
        add_p2=True,
        p2_idx=2,
        p2_channels=128,
        p2_pre=P2_PRE,
        p2_fuse_block="C2f",
    )
    .train(
        optimizer="SGD",
        epochs=200,
        batch=8,
        imgsz=640,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        close_mosaic=10,
        workers=4,
        seed=0,
        deterministic=True,
        amp=True,
        project="results",
    )
)
