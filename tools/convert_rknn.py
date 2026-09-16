"""RKNN 转换：ONNX → .rknn（Rockchip RK3588 NPU）。

**必须在 x86 主机上运行**：rknn-toolkit2 只提供 x86_64 的轮子，板子上不做转换
（板子只跑 librknnrt.so 做推理）。转换产物 .rknn 可以拷到任意同型号设备。

为什么归一化要在这一步"烧进模型"（而不是留到 C++ 侧做）：
    RKNN 的量化流程需要看到完整的数据流才能把 scale 校准准。若把 (x - mean) / std
    留在推理端做，量化模型看到的是"未归一化的整数"，校准出来的 scale 会完全错位。
    所以这里用 std=[255,255,255] 等参数把归一化表达进模型：
      * 转换期：toolkit 把输入 scale 设为 1/std，等价于"模型内部先 /255"；
      * 推理期：C++ 侧给 **uint8 原始像素**（todrt 的 preprocess.output = "uint8"）。

这两个设置必须成对出现 —— 只改一边会得到"不报错但框全错"的结果，所以本脚本会把
结果写回部署配置（--deploy-config），让 C++ 侧不可能配错。

用法::

    # 1) 只转模型（推荐先跑，看量化后精度掉多少）
    python tools/convert_rknn.py \\
        --onnx exports/SPAE-YOLOv8n.onnx \\
        --target rk3588 --out exports/spae_yolov8n.rk3588.rknn \\
        --dataset exports/calib.txt

    # 2) 顺带把 C++ 部署配置改成 RKNN 路径（前处理自动切成 uint8 NHWC）
    python tools/convert_rknn.py --onnx ... --out ... \\
        --deploy-config configs/deploy/spae-yolov8n.json

    # 3) 若已有 VisDrone 的量化数据集描述（rknn-toolkit2 的 dataset.txt 格式），
    #    直接用它做混合量化，精度更稳：
    python tools/convert_rknn.py --onnx ... --out ... --quantized-dataset dataset.txt

校准集（--dataset）是每行一个图片路径的文本文件，建议 100–300 张，覆盖真实场景
（白天/夜间、不同高度、有无小目标）。小目标对量化误差非常敏感，**务必复测 AP_small**。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# 支持的 SoC：决定 NPU 的算子支持与量化策略
TARGETS = ("rk3588", "rk3588s", "rk3568", "rk3566", "rk3562", "rk3576")

DEPLOY_SCHEMA = "todrt.deploy/v1"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="ONNX → RKNN 转换（x86 主机上运行）")
    ap.add_argument("--onnx", type=Path, required=True, help="输入 ONNX（由 tools/export_onnx.py 产出）")
    ap.add_argument("--out", type=Path, required=True, help="输出 .rknn 路径")
    ap.add_argument("--target", default="rk3588", choices=TARGETS, help="目标 SoC")
    ap.add_argument("--dataset", type=Path, default=None,
                    help="量化校准图片列表（每行一个路径）")
    ap.add_argument("--quantized-dataset", type=Path, default=None,
                    help="rknn-toolkit2 的 dataset.txt（含 quant_img_RGB2/mean/std 等配置）")
    ap.add_argument("--quantized-algorithm", default="normal",
                    choices=("normal", "mmse", "kl_divergence"), help="量化算法")
    ap.add_argument("--no-quantize", action="store_true",
                    help="只做 fp16 转换（不量化）；精度最高、速度最慢，用于定位"
                         "“精度掉得多不多是量化造成的”")
    ap.add_argument("--mean", default="0 0 0", help='归一化 mean，空格分隔（默认 "0 0 0"）')
    ap.add_argument("--std", default="255 255 255",
                    help='归一化 std，空格分隔（默认 "255 255 255" = 把 /255 烧进模型）')
    ap.add_argument("--optimization-level", type=int, default=3, choices=(0, 1, 2, 3))
    ap.add_argument("--core-num", type=int, default=3,
                    help="NPU core 数（RK3588=3）；写进部署配置供 C++ 侧使用")
    ap.add_argument("--single-core", action="store_true",
                    help="转换时约束为单 core（调试用；正常应留 3 核由 C++ 侧分配）")
    ap.add_argument("--deploy-config", type=Path, default=None,
                    help="同时把 C++ 部署配置改成 RKNN 路径（前处理切 uint8 NHWC）")
    ap.add_argument("--model-name", default=None, help="覆盖部署配置里的 model 名")
    return ap


def load_rknn():
    try:
        from rknn.api import RKNN  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "未安装 rknn-toolkit2。请在 **x86 主机**上安装：\n"
            "  pip install rknn-toolkit2\n"
            "注意：rknn-toolkit2 不提供 aarch64 轮子，板子上装不了（板子只要 librknnrt.so）。\n"
            "若只想知道“这套代码在 RK3588 上怎么接线”，可以不转换，直接用\n"
            "  todrt_cli dryrun <cfg.json> --preset=rk3588\n"
            "验证装配链路（不需要模型文件）。"
        ) from exc
    return RKNN


def parse_triplet(s: str) -> list[float]:
    parts = [p for p in s.replace(",", " ").split() if p]
    if len(parts) != 3:
        raise SystemExit(f"需要 3 个数值（空格分隔），实际收到：{s!r}")
    return [float(p) for p in parts]


def convert(args) -> Path:
    RKNN = load_rknn()
    try:
        from rknn.api import RKNN  # noqa: F401
    except ImportError:  # pragma: no cover
        pass

    if not args.onnx.is_file():
        raise SystemExit(f"ONNX 不存在：{args.onnx}")

    rknn = RKNN(verbose=False)

    mean = parse_triplet(args.mean)
    std = parse_triplet(args.std)
    print(f"[RKNN] 目标={args.target}  mean={mean} std={std}")
    if std != [1.0, 1.0, 1.0] and std != [0.0, 0.0, 0.0]:
        print("[RKNN] 归一化将烧进模型 → C++ 侧前处理必须给 uint8 原始像素"
              "（部署配置 preprocess.output=\"uint8\", layout=\"nhwc\"）")

    # config：mean_values/std_values 会被写进模型输入端，推理时 toolkit 期望你给
    # "原始像素"（0–255 的 uint8）。这与 todrt 的 preprocess.output=uint8 是一对。
    rknn.config(
        mean_values=[mean],
        std_values=[std],
        target_platform=args.target,
        quantized_algorithm=args.quantized_algorithm,
        optimization_level=args.optimization_level,
    )

    print(f"[RKNN] 加载 ONNX：{args.onnx}")
    if rknn.load_onnx(model=str(args.onnx)) != 0:
        raise SystemExit("加载 ONNX 失败（看上面的 RKNN 日志；常见原因：opset 过高、"
                         "含 RKNN 不支持的算子）")

    do_quant = not args.no_quantize
    dataset = None
    if args.quantized_dataset:
        dataset = str(args.quantized_dataset)
    elif args.dataset:
        dataset = str(args.dataset)
    if do_quant and dataset is None:
        print("[RKNN] ⚠️ 没有给 --dataset：toolkit 会用随机数据做量化校准，"
              "精度通常明显更差。强烈建议给 100–300 张真实场景图片。")

    print(f"[RKNN] 构建（do_quantization={do_quant}）...")
    if rknn.build(do_quantization=do_quant, dataset=dataset) != 0:
        raise SystemExit("构建 RKNN 失败（看上面的日志；常见原因：算子不支持、"
                         "量化数据集路径不对）")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if rknn.export_rknn(str(args.out)) != 0:
        raise SystemExit("导出 .rknn 失败")
    rknn.release()
    size_mb = args.out.stat().st_size / 1e6
    print(f"[RKNN] 已生成：{args.out}（{size_mb:.1f} MB）")
    return args.out


def patch_deploy_config(args, rknn_path: Path) -> None:
    """把部署配置改成 RKNN 路径。

    关键点：engine.path 指向 .rknn，preprocess 切成 uint8 + nhwc。
    这两处必须一起改 —— 只改一边就是"双重归一化"或"dtype 不匹配"。
    """
    cfg_path = args.deploy_config
    if not cfg_path.is_file():
        raise SystemExit(f"部署配置不存在：{cfg_path}（先用 tools/export_onnx.py --deploy-config 生成）")

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["schema"] = cfg.get("schema", DEPLOY_SCHEMA)

    # 后端选择：builder 决定走哪个运行时
    cfg["builder"] = "rknn"
    if args.model_name:
        cfg["model"] = args.model_name

    engine = cfg.setdefault("engine", {})
    engine["path"] = str(rknn_path)          # .rknn 作为"引擎产物"
    engine["serialize_out"] = ""             # .rknn 不需要再序列化

    hardware = cfg.setdefault("hardware", {})
    hardware["device"] = "npu"
    hardware["precision"] = "fp16" if args.no_quantize else "int8"
    hardware["accelerator_core"] = 0
    hardware["dynamic_shape"] = False
    hardware["dynamic_batch"] = False

    cfg.setdefault("rknn", {})
    cfg["rknn"]["model"] = str(rknn_path)
    cfg["rknn"]["core_num"] = 1 if args.single_core else max(1, args.core_num)
    cfg["rknn"]["dequantize_output"] = True

    # ★ 与转换参数成对的前处理设置
    pre = cfg.setdefault("preprocess", {})
    if args.no_quantize:
        pre["output"] = "float32"
        pre["layout"] = "nchw"
    else:
        pre["output"] = "uint8"
        pre["layout"] = "nhwc"

    cfg["notes"] = (cfg.get("notes", "") +
                    f"\n[RKNN] target={args.target} quantized={not args.no_quantize} "
                    f"core_num={cfg['rknn']['core_num']}（由 tools/convert_rknn.py 写入）").strip()

    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[配置] 已切到 RKNN 路径：{cfg_path}")
    print(f"        builder=rknn  device=npu  engine.path={rknn_path}")
    print(f"        preprocess={pre['output']}/{pre['layout']}  rknn.core_num={cfg['rknn']['core_num']}")


def main() -> int:
    args = build_parser().parse_args()
    rknn_path = convert(args)
    if args.deploy_config:
        patch_deploy_config(args, rknn_path)

    print("\n下一步（RK3588 板上）：")
    print("  # 把 librknnrt.so 与 .rknn 拷到板上，然后：")
    print(f"  ./todrt_cli probe                        # 看 /dev/rknpu* 与设备权限")
    if args.deploy_config:
        print(f"  ./todrt_cli dryrun {args.deploy_config}   # 装配自检（不需要真跑）")
        print(f"  ./todrt_cli bench  {args.deploy_config} sample.ppm 200")
    print("\n提醒：量化后必须复测 AP_small（小目标对量化误差很敏感），"
          "不能只看 FPS。")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
