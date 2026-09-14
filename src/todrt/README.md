# todrt —— yolo-tod 的 C++ 推理运行时

> **一句话**：把 `tod/` 训练出来的魔改变体，用**一个名字 + 一份 JSON** 部署到实机
> （TensorRT / Jetson Orin DLA / dGPU），并且让"新增一个变体"的代价保持在**一个文件 + 两行宏**。

`src/tod` 是训练侧（Python，ultralytics 插件化魔改库）；
`src/todrt` 是部署侧（C++17，无第三方依赖，TensorRT 可选）。
两者**唯一**的耦合是一份部署配置 JSON，由 `tools/export_onnx.py` 生成。

```
Python 训练端                         C++ 部署端
──────────────────                   ──────────────────────────────────
variants/SPAE-YOLOv8n/recipe.py  ─┐
  ↓ 训练 + EP5 头部手术            │  tools/export_onnx.py
best.pt → ONNX ──────────────────┼─→ configs/deploy/spae-yolov8n.json
                                  │        │
                                  │        ▼
                                  │   todrt_cli / libtodrt
                                  │   Detector::CreateFromFile(json)
                                  └─  nc / reg_max / strides / layout
                                      （结构信息只写一次，C++ 不手抄）
```

---

## 1. 为什么是「工厂 + 注册表」

部署代码最常见的腐化方式不是写错算法，而是**接线散落各处**：

```cpp
// ✗ 反面教材：每换一个变体/精度/设备，这些 if 就多一层
if (model == "spae") { build_spae(); } else if (model == "v8") { ... }
if (use_dla) { ... } else if (use_fp16) { ... }
int nc = 10;  // 从训练脚本里抄来的，抄错了也没人知道
```

本项目把推理端拆成 **4 座自注册工厂 + 1 个变体配方**，调用方只看名字：

| 工厂 | 回答什么问题 | 注册宏 | 现有实现 |
|---|---|---|---|
| `model`（变体配方） | 这个变体用哪条链路、什么解码参数 | `TOD_RT_DEFINE_RECIPE` | `SPAE_YOLOv8n`、`YOLOv8n_baseline` |
| `builder` | 引擎从哪来（ONNX 构建 / engine 反序列化） | `TOD_RT_REGISTER_BUILDER` | `yolov8-trt`、`trt` |
| `preproc` | 图 → 张量（letterbox / stretch / integer-scale） | `TOD_RT_REGISTER_PREPROC` | `detect_letterbox` |
| `postproc` | 张量 → 检测框（NMS / soft-NMS / 插件输出） | `TOD_RT_REGISTER_POSTPROC` | `detect_nms` |

于是**调用方只需要两行**：

```cpp
auto det = todrt::Detector::CreateFromFile("configs/deploy/spae-yolov8n-orin.json");
auto results = det->Run(bgr_image);   // vector<Detection>，坐标已是原图像素
```

异步（多路视频、ROS 回调里不阻塞）：

```cpp
det->Submit(frame);                   // 内部拷贝像素，可立刻复用缓冲
todrt::InferenceResult r;
while (det->TryGet(&r)) { render(r.detections); }
```

实测延迟（**必须来自目标设备**）：

```cpp
auto b = det->Bench(frame, /*warmup=*/10, /*iters=*/200);
std::printf("%.2f ms  %.1f FPS\n", b.end_to_end_ms, b.fps);
```

### 新增一个可部署变体 = 一个文件

```cpp
// src/todrt/src/models/my_variant.cpp
#include "todrt/factory.hpp"

TOD_RT_DEFINE_RECIPE(My_Variant, MyVariantRecipe) {
  todrt::ModelRecipe r;
  r.builder  = "yolov8-trt";
  r.layout   = todrt::OutputLayout::kAnchorMajorDfl;
  r.decode.strides = {4, 8, 16, 32};
  r.decode.num_classes = 10;
  r.hardware_preset = "orin";
  r.tune = [](todrt::DetectorOptions& o) {
    // 变体特有的校验/补全；配置写错在这里就报出来，而不是等到框全乱
  };
  return r;
}

TOD_RT_META(My_Variant, {"my-variant"}, "论文标题 / 仓库", "许可证",
            "硬件与精度要求", "成本提示")
```

CMake 里加一行（或直接用 glob），**不需要改任何调用方代码**。
`REGISTER_*` 走的是静态初始化，注册表在 `main()` 之前就已填满。

---

## 2. 目录

```
src/todrt/
├─ CMakeLists.txt
├─ cmake/aarch64-linux-gnu.cmake     交叉编译到 Jetson
├─ include/todrt/
│  ├─ core.hpp                       错误 / 日志 / 自注册表（无任何依赖）
│  ├─ json.hpp                       极简 JSON（容忍注释与尾随逗号）
│  ├─ modules.hpp                    前处理 / 解码 / 后处理接口 + 配置结构
│  ├─ detector_options.hpp           「一次部署」的完整声明
│  ├─ factory.hpp                    ★ 四座工厂 + Detector（调用方唯一要认识的类型）
│  ├─ backend/factories.hpp          内部构造助手（按配置造模块）
│  ├─ backend/engine_trt.hpp         TensorRT 引擎封装（PIMPL，头里无 TRT 类型）
│  └─ backend/trt_factory.hpp        TensorRT 后端入口
├─ src/
│  ├─ core.cpp                       注册表 / 日志 / 枚举解析
│  ├─ json.cpp                       JSON 解析与序列化
│  ├─ config_io.cpp                  部署配置 <-> 结构体
│  ├─ factory.cpp                    ★ 工厂装配、异步骨架、PlanAssembly 自检
│  ├─ preprocess.cpp                 letterbox / stretch / integer-scale（自实现，无 OpenCV）
│  ├─ decode.cpp                     DFL + 四种输出布局解码（CPU 唯一真相）
│  ├─ nms.cpp                        硬 NMS / soft-NMS / 后处理器
│  ├─ models/*.cpp                   ★ 变体配方（一个变体一个文件）
│  └─ backend/                       TensorRT 引擎构建与执行；无 TRT 时走 stub
├─ apps/todrt_cli.cpp                命令行入口
└─ tests/cpp_smoke.cpp               架构自检（**不需要 GPU / TensorRT**）
```

---

## 3. 构建

### 3.1 开发机（无 GPU）：只验证工厂 / 配置 / 解码

```bash
cmake -S src/todrt -B build/todrt -DTODRT_WITH_TENSORRT=OFF
cmake --build build/todrt -j
./build/todrt/todrt_smoke          # 或 ctest --test-dir build/todrt
```

这一步覆盖：工厂自注册与别名、配置解析与校验、letterbox 几何、DFL 解码、
四种输出布局对拍、NMS、坐标反变换、以及**错误配置必须报错**的路径。
它不覆盖：引擎构建、DLA 归属、真实延迟 —— 那些只能在目标机上验证。

### 3.2 实机（Jetson Orin / dGPU，**Linux**）

```bash
# JetPack 6.x 上 TensorRT/CUDA 已随系统安装在 /usr
cmake -S src/todrt -B build/todrt -DTODRT_WITH_TENSORRT=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build/todrt -j$(nproc)
sudo cmake --install build/todrt --prefix /usr/local     # 可选
```

TensorRT 不在默认路径时：

```bash
cmake -S src/todrt -B build/todrt -DTODRT_WITH_TENSORRT=ON \
      -DTensorRT_ROOT=/path/to/TensorRT -DCUDAToolkit_ROOT=/usr/local/cuda
```

### 3.3 x86 构建机交叉编译到 Jetson

```bash
cmake -S src/todrt -B build/aarch64 \
      -DCMAKE_TOOLCHAIN_FILE=src/todrt/cmake/aarch64-linux-gnu.cmake \
      -DTODRT_WITH_TENSORRT=ON -DTensorRT_ROOT=/opt/trt-aarch64
```

推荐直接用 NVIDIA 官方 `<tag>-cross-aarch64` 容器（头/库/工具链都齐）。
注意：**engine 不能跨设备拷贝**（与 TensorRT 版本 + GPU 架构强绑定），
交叉编译只解决"编出可执行文件"，engine 必须在目标机上构建或用本工具构建。

---

## 4. 使用

```bash
# ① 不需要 GPU：看看工厂里有什么
./build/todrt/todrt_cli list
./build/todrt/todrt_cli info SPAE_YOLOv8n

# ② 不需要 GPU：配置装配自检（配置写错在这里就暴露）
./build/todrt/todrt_cli dryrun configs/deploy/spae-yolov8n-orin.json

# ③ 实机：先看硬件能力（TensorRT 版本 / DLA core 数 / 是否还有 DLA）
./build/todrt/todrt_cli probe

# ④ 实机：构建 engine 并落盘
./build/todrt/todrt_cli dryrun cfg.json --preset=orin --save-engine=spae_orin.engine

# ⑤ 实机：实测延迟（目标设备上的数字才算数）
./build/todrt/todrt_cli bench cfg.json sample.ppm 200
```

---

## 5. 硬件加速：能开什么、怎么开、会踩什么坑

配置项都在部署 JSON 的 `hardware` / `runtime` 段，或用 CLI 的 `--preset` 覆盖。

| 手段 | 配置 | 适用 | 注意事项 |
|---|---|---|---|
| **DLA**（固定功能加速器） | `device: "dla"`, `dla_core: 0` | Jetson Orin/Xavier | 只支持 FP16/INT8；不支持 FP32。**必须**配 `allow_gpu_fallback: true`，否则不支持的层直接构建失败 |
| **FP16** | `precision: "fp16"` | 几乎所有 NVIDIA GPU | dGPU 上基本是免费的加速；Orin 上小模型可能变成带宽瓶颈 |
| **INT8** | `precision: "int8"` + 校准 | Orin（尤其多路视频） | TensorRT 10+ 是强类型，需要 ONNX 带 Q/DQ 或显式量化；校准集要覆盖真实场景（小目标尤其敏感） |
| **CUDA Graph** | `runtime.cuda_graphs: true` | 固定 shape + 小 batch | Orin 的 CPU 弱，launch 开销占比高，收益明显。动态 shape 下自动关闭 |
| **固定 shape** | `dynamic_shape: false` | DLA 场景 | DLA 上动态 shape 支持有限；要换分辨率就重编 engine |
| **多分辨率** | `dynamic_shape: true` + `shape.min/opt/max` | dGPU | DLA + 动态 shape 组合容易不生效 |
| **多 DLA core 并行** | 每个核心一个 engine + 一个 Detector | Orin NX/AGX（2 个 DLA） | 两路视频各占一个 core，比单核轮转更稳 |

**判断 DLA 到底吃到了多少**：`TrtEngine::DlaLayerCount()/GpuLayerCount()` 会统计每层
能否上 DLA，构建日志里也会打印。一次 DLA↔GPU 切换的代价往往比那层本身的计算还大，
所以"DLA 层数很多但 FPS 没涨"通常意味着中间有几个被迫回退的层把流水线切碎了。

**已知与本项目相关的坑**（值得单独记一笔）：

1. **TensorRT 11 起上游移除了 DLA**（官方文档明确说明）。Orin 上请用 JetPack 6.x
   自带的 TensorRT 10.x；本库在 TRT ≥ 11 上会给出显式报错，而不是静默回落到 GPU。
2. **Efficient_UAVDet 用的是分组卷积**（`g = x/16`）。分组卷积在 DLA 上的支持随
   版本与分组数变化，务必用 `allow_gpu_fallback: true` 先跑通，再看层分布决定是否值得调。
3. **P2 头把 anchor 从 8400 抬到 34000**（640×640）。解码与 NMS 的候选量约 4 倍，
   如果后处理成为瓶颈，可以：降低 `conf` 之前的候选量（`max_det` 不影响解码成本）、
   改用插件版 NMS（`--nms` 导出，代价是阈值改了要重建 engine）、或把后处理搬到 GPU。
4. **letterbox 的 `pad_multiple` 必须与导出时一致**（Ultralytics 默认 32）。
   不一致会带来几个像素的系统性偏移 —— 对小目标就是致命的。
5. **`strides` 写错不会报错，只会给出全错的框**。所以解码器会在启动时把
   anchor 数与引擎输出对拍，不一致直接抛异常（`decode.cpp: check_anchors`）。

---

## 6. 与训练侧的对应关系

| 训练侧（`src/tod`） | 部署侧（`src/todrt`） |
|---|---|
| `registry.py`（`@register` + 强制元数据） | `core.hpp`（`REGISTER_*` + `TOD_RT_META`） |
| `compose.py`（变体 DSL） | `factory.hpp` 的 `ModelRecipe` + `DetectorOptions` |
| `compat.py`（唯一触碰框架内部） | `backend/engine_trt.cpp`（唯一触碰 TensorRT/CUDA） |
| `modules/head/efficient_uavdet.py` | 只影响图结构 → 体现在 ONNX 与部署配置的 `strides/layout` |
| `engine/surgery.py`（EP5 建模后手术） | `tools/export_onnx.py` 必须先做同样的手术再导出 |
| `variants/<name>/variant.yaml` | `configs/deploy/<name>.json`（`todrt.deploy/v1`） |

**唯一契约**：`configs/deploy/*.json`。字段定义见 `src/todrt/src/config_io.cpp`，
由 `tools/export_onnx.py --deploy-config` 生成。schema 版本不匹配会被拒绝启动。
