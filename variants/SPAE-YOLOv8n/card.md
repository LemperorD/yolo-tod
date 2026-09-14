# SPAE-YOLOv8n

- **id**: `SPAE-YOLOv8n`
- **base**: `yolov8n`
- **dataset**: `visdrone2019-det`
- **status**: `planned`
- **tags**: paper-repro, small-object, lightweight, uav

## EP 改动

| EP | 名称 | 覆盖 |
|---|---|---|
| EP1 | Backbone | `downsample=ADown`; `downsample_indices=[1, 3, 5, 7]` |
| EP5 | Head | `head=Efficient_UAVDet`; `per_group=16`; `channels=native`; `levels=[2, 3, 4, 5]` |
| EP7 | 损失 | `box=siou`; `theta=4.0` |

## 引用模块来源

_（无，或引用的模块尚未注册）_

## 相对基线

| 指标 | 基线 | 本变体 | Δ |
|---|---|---|---|
| AP50:95 | | | |
| **AP_small** | | | |
| AP_tiny (<16px) | | | |
| Params / FLOPs | | | |
| 延迟 (ms, batch=1) | | | |
| 峰值显存 (GB) | | | |

## 已知坑 / 冲突 / 结论

论文原设定：Det-Fly 数据集、batch=32、SGD、200 epochs；本变体迁移到 VisDrone2019-DET 且 batch 降到 8（8GB 显存），属于域迁移 + 超参改动，论文 mAP 数字不可直接引用。

论文消融（对 baseline 0.850）：P2 +7.5pp（主贡献）、ADown +0.4pp、SIoU +0.1pp、Efficient_UAVDet −0.4pp（论文自认是压缩/加速手段）。

引用论文数字时注意其表间冲突：Table 6 baseline 记 0.850，而 Table 5/7 记 0.922；同一配置 FPS 在 Table 6 为 203.0、Table 5 为 161.5。
