"""导出入口：变体/权重 → ONNX → （可选）TensorRT engine + C++ 部署配置。

为什么必须有这个脚本（而不是直接 `yolo export`）：
  1. **检测头被"建模后手术"改过**（`tod.engine.surgery`：Efficient_UAVDet 的分组卷积
     stem）。`yolo export` 拿到的是未经手术的模型，导出的 ONNX 与训练权重不一致。
  2. 手术会改变分支输出，**检测头的 `stride` 缓存必须重算**，否则 ultralytics 会按
     旧的 stride 生成 anchor，导出的 detect 输出与解码器对不上（框会全乱）。
  3. C++ 端解码需要知道 `nc / reg_max / strides / layout`，这些信息只能由这里
     **一次性导出成部署配置 JSON**（src/todrt 的唯一契约），避免在两处手抄。

用法::

    # 1) 检查导出链路（不需要 GPU 也能跑到 ONNX 这一步）
    python tools/export_onnx.py --variant variants/SPAE-YOLOv8n/variant.yaml \\
        --weights results/SPAE-YOLOv8n/weights/best.pt \\
        --imgsz 640 --out exports/ --deploy-config configs/deploy/spae-yolov8n.json

    # 2) 顺带构建 TensorRT engine（需要目标机上的 tensorrt Python 包）
    python tools/export_onnx.py --variant ... --weights ... --build-engine \\
        --engine-preset orin --save-engine exports/spae_orin_fp16.engine

    # 3) INT8（需要校准数据列表）
    python tools/export_onnx.py ... --build-engine --engine-preset orin \\
        --int8 --calib-data exports/calib.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

#: C++ 端（src/todrt）识别这个 schema；改动字段请同时改 src/todrt/src/config_io.cpp
DEPLOY_SCHEMA = "todrt.deploy/v1"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="TOD 导出入口（ONNX / TensorRT / 部署配置）")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--variant", type=Path, help="变体 spec YAML（推荐：会应用头部手术）")
    src.add_argument("--weights", type=Path, help="已训练权重 .pt（需配合 --variant 才能手术）")
    ap.add_argument("--imgsz", type=int, default=None, help="输入尺寸，默认取变体 data.imgsz")
    ap.add_argument("--batch", type=int, default=1, help="导出 batch，默认 1")
    ap.add_argument("--opset", type=int, default=12)
    ap.add_argument("--out", type=Path, default=ROOT / "exports", help="输出目录")
    ap.add_argument("--half", action="store_true", help="导出 FP16 ONNX（部分后端才需要）")
    ap.add_argument("--dynamic", action="store_true", help="动态 batch/尺寸（DLA 上不建议）")
    ap.add_argument("--nms", action="store_true", help="把 NMS 加进 ONNX（输出变 plugin-nms）")
    ap.add_argument("--simplify", action="store_true", default=True)
    ap.add_argument("--no-simplify", dest="simplify", action="store_false")
    ap.add_argument("--devices", default=None, help="导出用的设备，如 0 或 cpu")

    ap.add_argument("--deploy-config", type=Path, default=None,
                    help="同时写出 C++ 部署配置 JSON")
    ap.add_argument("--class-names", default=None,
                    help="类别名，逗号分隔（仅写入部署配置，便于日志可读）")

    g = ap.add_argument_group("TensorRT engine（可选）")
    g.add_argument("--build-engine", action="store_true", help="用 tensorrt Python 包构建 engine")
    g.add_argument("--engine-preset", default="orin",
                   choices=["orin", "dgp", "x86", "fp32", "int8"],
                   help="硬件预设：orin=DLA+FP16，dgp/x86=FP16，fp32=调试，int8=INT8")
    g.add_argument("--save-engine", type=Path, default=None, help="engine 落盘路径")
    g.add_argument("--dla-core", type=int, default=0)
    g.add_argument("--no-gpu-fallback", action="store_true", help="DLA standalone（要求全部层可上 DLA）")
    g.add_argument("--int8", action="store_true", help="INT8（需 TensorRT 的校准流程）")
    g.add_argument("--calib-data", type=Path, default=None, help="INT8 校准图片列表（每行一个路径）")
    g.add_argument("--workspace-mb", type=int, default=1024)
    return ap


# --------------------------------------------------------------------- 模型准备


def load_model(args):
    """返回 (YOLO 模型, spec dict)。权重与变体同时给出时，先建变体图再压权重。"""
    import tod
    from tod import runtime
    from tod.compat import load_yaml
    from tod.compose import Variant
    from tod.engine.surgery import apply_spec

    tod.bootstrap()

    from ultralytics import YOLO

    spec: dict = {}
    variant = None
    if args.variant:
        if not args.variant.is_file():
            raise SystemExit(f"变体文件不存在：{args.variant}")
        spec = load_yaml(args.variant)
        variant = Variant.from_spec(spec)
        model_yaml = args.variant.parent / "model.yaml"
        variant.model_yaml(model_yaml, write=True)
        print(f"[模型] 由变体生成结构：{model_yaml}")
        runtime.set_active(spec)
        model = YOLO(str(model_yaml))
    else:
        model = YOLO(str(args.weights))
        print(f"[模型] 直接加载权重：{args.weights}")
        print("       ⚠️ 未提供 --variant：无法应用 EP5 头部手术，"
              "导出的图只适合没有做检测头改造的变体。")

    # 权重覆盖：变体图 + 训练好的权重
    if args.weights and variant is not None:
        import torch

        ckpt = torch.load(str(args.weights), map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt)
        state = getattr(state, "state_dict", state)
        missing, unexpected = model.model.load_state_dict(state, strict=False)
        print(f"[权重] 载入 {args.weights}：缺失 {len(missing)} 项，多余 {len(unexpected)} 项")
        if missing:
            print(f"       缺失示例：{missing[:5]}")
        if unexpected:
            print(f"       多余示例：{unexpected[:5]}")

    # ---- EP5 头部手术（必须与训练时一致，否则权重与图不匹配）----
    if spec:
        applied = apply_spec(model.model, spec)
        if applied:
            print("[EP5] 检测头手术已应用：")
            for line in applied:
                print(f"       {line}")
            recompute_stride(model.model)
        else:
            print("[EP5] 变体未要求头部手术。")
    return model, spec


def recompute_stride(net) -> None:
    """重算检测头的 stride 缓存。

    为什么必须做：ultralytics 的 `Detect.__init__` 会按 stride=[8,16,32] 造 anchor 网格
    （`_make_anchors`），并在 `bias_init` 里给出先验。头部手术改变了分支通道，但不会
    自动更新 stride；一旦 stride 与真实下采样不一致，导出的输出与解码器就对不上。
    """
    import torch

    head = net.model[-1]
    if not hasattr(head, "stride"):
        print("[stride] 检测头没有 stride 属性，跳过重算。")
        return
    try:
        s = 256
        net.eval()
        with torch.no_grad():
            _ = net(torch.zeros(1, 3, s, s))
        print(f"[stride] 重算完成：{list(head.stride)}")
    except Exception as exc:  # pragma: no cover - 依赖具体框架版本
        print(f"[stride] 重算失败（{exc}），沿用框架缓存值：{list(head.stride)}")


def infer_layout_and_strides(model, spec: dict, imgsz: int) -> tuple[str, list[int], int]:
    """推断解码布局与 strides，并返回 (layout, strides, reg_max)。"""
    head = model.model.model[-1]
    strides = [int(s) for s in getattr(head, "stride", [8, 16, 32])]
    reg_max = int(getattr(head, "reg_max", 16))
    # ultralytics 的 Detect.forward（推理态）是 anchor-major：cat([cv2, cv3]) 后按 anchor 展开
    layout = "anchor-major-dfl"
    return layout, strides, reg_max


# --------------------------------------------------------------------- 部署配置


def build_deploy_config(model, spec: dict, args, onnx_path: Path, layout: str,
                        strides: list[int], reg_max: int, imgsz: int,
                        anchors: int | None) -> dict:
    nc = int(getattr(model.model.model[-1], "nc", 80))
    engine = {"onnx": str(onnx_path)}
    if args.save_engine:
        engine["path"] = str(args.save_engine)
    engine["serialize_out"] = str(Path(args.out) / f"{onnx_path.stem}.engine")

    hardware = {"device": "gpu", "precision": "fp16"}
    if args.engine_preset == "orin":
        hardware.update({"device": "dla", "dla_core": args.dla_core,
                         "precision": "int8" if args.int8 else "fp16",
                         "allow_gpu_fallback": not args.no_gpu_fallback})
    elif args.engine_preset == "int8":
        hardware.update({"device": "gpu", "precision": "int8"})
    elif args.engine_preset == "fp32":
        hardware.update({"device": "gpu", "precision": "fp32"})

    cfg: dict = {
        "schema": DEPLOY_SCHEMA,
        "model": (spec.get("id") if spec else onnx_path.stem),
        "model_id": (spec.get("id") if spec else onnx_path.stem),
        "dataset": (spec.get("dataset") if spec else ""),
        "nc": nc,
        "reg_max": reg_max,
        "layout": layout,
        "strides": strides,
        "input": [imgsz, imgsz],
        "pad_multiple": 32,
        "engine": engine,
        "hardware": hardware,
        "runtime": {"cuda_graphs": args.engine_preset in ("orin", "dgp", "x86"),
                    "workspace_mb": args.workspace_mb},
        "postprocess": {"conf": 0.25, "iou": 0.45, "max_det": 300, "nms": "hard"},
        "preprocess": {"mode": "letterbox"},
        "source": " ".join([Path(sys.argv[0]).name] + sys.argv[1:]),
        "notes": "由 tools/export_onnx.py 生成；结构信息是 C++ 端解码的唯一依据，请勿手工改动。",
    }
    if anchors:
        cfg["anchors"] = anchors
    if args.class_names:
        cfg["class_names"] = [s.strip() for s in args.class_names.split(",") if s.strip()]
    elif spec:
        data_cfg = spec.get("data") or {}
        if isinstance(data_cfg, dict) and data_cfg.get("names"):
            cfg["class_names"] = list(data_cfg["names"])
    return cfg


def expected_anchors(imgsz: int, strides: list[int]) -> int:
    return sum((imgsz // s) ** 2 for s in strides)


# --------------------------------------------------------------------- TensorRT


def build_engine(onnx_path: Path, cfg: dict, args) -> Path:
    """用 tensorrt Python 包构建 engine（仅在实机上可用；开发机跳过即可）。"""
    try:
        import tensorrt as trt
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "未安装 tensorrt Python 包，无法构建 engine。\n"
            "  实机（JetPack）上通常已随系统提供：python3 -c 'import tensorrt'\n"
            "  也可以只用 --deploy-config 导出配置，到实机上用 todrt_cli 构建：\n"
            "    todrt_cli dryrun cfg.json --save-engine=xxx.engine"
        ) from exc

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        for i in range(parser.num_errors):
            print(f"[TRT] {parser.get_error(i)}")
        raise SystemExit("ONNX 解析失败")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_mb << 20)

    hw = cfg["hardware"]
    try:
        config.set_flag(trt.BuilderFlag.FP16)
    except Exception:  # TensorRT 11 强类型后该 flag 可能已移除
        pass
    if hw.get("precision") == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        if args.calib_data:
            print("[TRT] INT8：请使用 TensorRT 的 calibrator（本脚本只传 flag，"
                  "校准数据由 tensorrt Python 的 IInt8EntropyCalibrator2 子类消费）")
    if hw.get("device") == "dla":
        print("[TRT] ⚠️ DLA 配置需要在 C++ 端（todrt_cli）或自行调用 set_device_type 完成；"
              "Python 端这里只做精度与 workspace 设置。")

    imgsz = cfg["input"][0]
    if args.dynamic:
        profile = builder.create_optimization_profile()
        name = network.get_input(0).name
        profile.set_shape(name, (1, 3, imgsz, imgsz), (1, 3, imgsz, imgsz),
                          (args.batch, 3, imgsz, imgsz))
        config.add_optimization_profile(profile)

    print("[TRT] 正在构建 engine（Orin 上可能需要几分钟）...")
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise SystemExit("engine 构建失败（见上方 TensorRT 日志）")
    out_path = args.save_engine or (Path(args.out) / f"{onnx_path.stem}.engine")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(bytes(blob))
    print(f"[TRT] engine 已保存：{out_path}（{out_path.stat().st_size / 1e6:.1f} MB）")
    return out_path


# --------------------------------------------------------------------- 主流程


def main() -> int:
    args = build_parser().parse_args()
    args.out = Path(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    model, spec = load_model(args)
    imgsz = args.imgsz or int(((spec.get("data") or {}).get("imgsz")) or 640)
    model_name = (spec.get("id") if spec else args.weights.stem)

    layout, strides, reg_max = infer_layout_and_strides(model, spec, imgsz)
    anchors = expected_anchors(imgsz, strides)
    print(f"[导出] {model_name}: imgsz={imgsz} strides={strides} reg_max={reg_max} "
          f"layout={layout} 预期 anchor={anchors}")

    # ---- ONNX ----
    export_kwargs = dict(
        format="onnx",
        imgsz=imgsz,
        opset=args.opset,
        simplify=args.simplify,
        dynamic=args.dynamic,
        half=args.half,
        nms=args.nms,
        batch=args.batch,
    )
    if args.devices:
        export_kwargs["device"] = args.devices
    try:
        onnx_path = Path(model.export(**export_kwargs))
    except TypeError:
        # 不同 ultralytics 版本对 batch/nms 参数的支持不一致，降级重试
        for drop in ("batch", "nms", "simplify"):
            export_kwargs.pop(drop, None)
        onnx_path = Path(model.export(**export_kwargs))
    print(f"[ONNX] {onnx_path}")

    if args.nms:
        layout = "plugin-nms"
        print("[ONNX] 已把 NMS 导出进图：解码布局切换为 plugin-nms "
              "（阈值改动需重建 engine，注意记录）")

    # ---- 校验输出形状（这一步能在导出阶段抓住绝大多数配置错误）----
    verify_onnx(onnx_path, anchors, layout)

    # ---- 部署配置（C++ 端的唯一契约）----
    cfg = build_deploy_config(model, spec, args, onnx_path, layout, strides, reg_max, imgsz,
                              None if args.nms else anchors)
    deploy_path = args.deploy_config or (args.out / f"{model_name}.deploy.json")
    deploy_path.parent.mkdir(parents=True, exist_ok=True)
    deploy_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[配置] 部署配置已写出：{deploy_path}")

    if args.build_engine:
        build_engine(onnx_path, cfg, args)

    print("\n下一步（实机）：")
    print(f"  src/todrt/build/todrt_cli dryrun {deploy_path}")
    print(f"  src/todrt/build/todrt_cli bench  {deploy_path} sample.ppm 200")
    return 0


def verify_onnx(onnx_path: Path, anchors: int, layout: str) -> None:
    """用 onnx 包（若安装）核对输出形状；没装就只打印提示。"""
    try:
        import onnx
    except ImportError:
        print("[校验] 未安装 onnx 包，跳过输出形状校验"
              "（建议 pip install onnx 以便在导出阶段就发现形状错误）")
        return
    try:
        m = onnx.load(str(onnx_path))
        shapes = []
        for out in m.graph.output:
            dims = [d.dim_value if d.dim_value else (d.dim_param or "dyn")
                    for d in out.type.tensor_type.shape.dim]
            shapes.append((out.name, dims))
        for name, dims in shapes:
            print(f"[校验] 输出 {name}: {dims}")
        if layout == "plugin-nms":
            print("[校验] plugin-nms 布局：期望 [N,6] 或 [N,7]")
            return
        found = False
        for name, dims in shapes:
            if len(dims) == 3 and dims[2] == anchors and dims[1] == 4 * 16 + 10:
                found = True
            if len(dims) == 3 and dims[2] == anchors:
                found = True
        if not found:
            print(f"[校验] ⚠️ 没有找到 anchor={anchors} 的输出；"
                  "请确认 strides 与导出图一致，否则 C++ 解码会直接报错。")
        else:
            print(f"[校验] ✅ 输出 anchor 数与预期一致（{anchors}）")
    except Exception as exc:
        print(f"[校验] onnx 校验失败：{exc}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
