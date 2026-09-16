# todrt —— yolo-tod 的 C++ 推理运行时

> **一句话**：把 `tod/` 训练出来的魔改变体，用**一个名字 + 一份 JSON** 部署到实机
> （TensorRT / RKNN / ONNX Runtime / OpenVINO），并且让"新增一个变体"的代价保持在
> **一个文件 + 两行宏**、"新增一个后端"的代价保持在**一个 .cpp + 一个分派分支**。

`src/tod` 是训练侧（Python，ultralytics 插件化魔改库）；
`src/todrt` 是部署侧（C++17，无第三方依赖，推理后端全部可选）。
两者**唯一**的耦合是一份部署配置 JSON，由 `tools/export_onnx.py` 生成。

## 四个后端

同一份部署配置换个 `builder` 就能落到不同硬件上：

| 后端 | 目标硬件 | 模型产物 | 输入契约 | 构建开关 |
|---|---|---|---|---|
| **TensorRT** | NVIDIA dGPU / Jetson DLA | `.engine`（板上构建，不可移植） | `float32 + nchw` | `-DTODRT_WITH_TENSORRT=ON` |
| **RKNN** | Rockchip RK3588 NPU（3 核） | `.rknn`（x86 离线转换，可移植） | **`uint8 + nhwc`** | `-DTODRT_WITH_RKNN=ON` |
| **ONNX Runtime** | 任意 CPU（AMD x86 主力） | 无（直接读 `.onnx`） | `float32 + nchw` | `-DTODRT_WITH_ORT=ON` |
| **OpenVINO** | x86 CPU / Intel iGPU-NPU | 无（可缓存编译产物） | `float32 + nchw` | `-DTODRT_WITH_OPENVINO=ON` |

> **输入契约是硬约束**：RKNN 的量化模型把归一化烧进了模型，宿主机必须给原始 uint8。
> 配错不会报错，只会让所有框都错。所以 `resolve_preprocess_options()` 以**引擎声明**
> 为准，并在冲突时打 warn；`preprocess.cpp` 用同一个几何核心同时支持两条输出路径。

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
| `builder` | 引擎从哪来（TRT 构建 / RKNN 加载 / ORT 会话 / OV 编译） | `TOD_RT_REGISTER_BUILDER` | `yolov8-trt`、`rknn`、`ort`、`openvino` |
| `preproc` | 图 → 张量（letterbox / stretch / integer-scale；float 或 uint8） | `TOD_RT_REGISTER_PREPROC` | `detect_letterbox`、`detect_letterbox_uint8` |
| `postproc` | 张量 → 检测框（NMS / soft-NMS / 插件输出） | `TOD_RT_REGISTER_POSTPROC` | `detect_nms` |

后端之间**共享**的装配逻辑只有一份（`backend_support.cpp` + `simple_detector.cpp`）：
定契约 → 建引擎 → 配前处理 → 配后处理 → 解码 → 坐标反变换。
每个后端只需实现 `IEngineRunner`（"跑一次"），于是不会出现
"TRT 用 conf 0.25、RKNN 用 0.2"这类不可追溯的错配。

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

### 3.2 实机（**Linux**）：按目标硬件选后端

```bash
# Jetson Orin / Xavier / dGPU（TensorRT 随 JetPack 装在 /usr）
cmake -S src/todrt -B build/orin -DTODRT_WITH_TENSORRT=ON && cmake --build build/orin -j$(nproc)

# RK3588（librknnrt.so + rknn_api.h 已在 /usr/{lib,include}）
cmake -S src/todrt -B build/rk3588 -DTODRT_WITH_RKNN=ON && cmake --build build/rk3588 -j$(nproc)

# AMD/Intel x86（CPU 后路；两个都开，实测取快的）
cmake -S src/todrt -B build/x86 -DTODRT_WITH_ORT=ON -DTODRT_WITH_OPENVINO=ON \
      -DTODRT_ORT_ROOT=/opt/onnxruntime-linux-x64-1.18.0
cmake --build build/x86 -j$(nproc)

sudo cmake --install build/orin --prefix /usr/local     # 可选
```

同一个构建里可以同时开多个后端（`probe` 会逐个报可用性）。SDK 不在默认路径时用
`-DTensorRT_ROOT=` / `-DTODRT_RKNN_ROOT=` / `-DTODRT_ORT_ROOT=` / `-DTODRT_OPENVINO_ROOT=`。

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

---

## 7. 版本敏感点（首次上实机时优先看这里）

三个非 TensorRT 后端的代码都按各自 SDK 的公开 API 写，并用编译期开关隔离
（`TODRT_HAVE_RKNN` / `TODRT_HAVE_ORT` / `TODRT_HAVE_OPENVINO`），
但 SDK 小版本之间确实有漂移。首次接入时若编译不过，按此表定位：

| 后端 | 敏感 API | 说明 |
|---|---|---|
| RKNN | `rknn_init(ctx, data, size, flag, rknn_init_extend*)` | 用 5 参数带 extend 的形式（rknpu2 现行版本）。更老的 SDK 只有 4 参数 —— 删掉最后一个实参即可，`core_mask` 也就无法指定 |
| RKNN | `rknn_dup_context` | 多核必需；若该符号不存在说明 SDK 过旧，先退回单核（`rknn_core_num: 1`） |
| RKNN | 头文件位置 | `rknn_api.h` 与 `librknnrt.so` 都在 rknpu2 的 `runtime/Linux/librknn_api/aarch64/` |
| ONNX Runtime | `OrtCUDAProviderOptions` | 新版倾向 `AppendExecutionProvider_CUDA_V2`；只影响 CUDA EP，CPU EP 各版本都稳 |
| ONNX Runtime | `GetInputNameAllocated` | ≥ 1.13 才有；更老的用 `GetInputName(i, alloc)`（代码里已按 `ORT_API_VERSION` 分支） |
| ONNX Runtime | `AppendExecutionProvider("XNNPACK"/"ACL")` | 字符串版 EP 接口，各构建带哪些 EP 取决于发行包 |
| OpenVINO | `Model::get_shape()` | 2024.x 起为 `get_input_shape()`/`get_output_shape()` —— 代码里用 `TODRT_OV_AT_LEAST_2024` 收敛在这个文件内 |
| OpenVINO | `ov::hint::performance_mode` | 2.0 起稳定；更老的 Inference Engine API 不在支持范围 |

> 这三份实现里**没有**任何"只有真机能编"的语法技巧：它们的 CMake 探测一旦成功
> 就会参与编译，编译错误会立刻暴露版本不匹配，而不是留到运行时。

### 怎么在没有对应硬件时先验证接线

```bash
# 装配链路（工厂 + 配置 + 前处理契约 + 解码器配置）完全不依赖 SDK：
todrt_cli dryrun cfg.json --preset=rk3588 --builder=rknn
#   → 会报告：前处理自动切成 uint8/NHWC、预期 anchor 数、构建器是否可用
#   → 只差"把模型喂给 NPU"这一步，配置错误这时就能发现
```

这一点是本库把"装配"与"执行"分开的原因：配置错误（strides/nc/前处理类型）
在开发机上就能全部抓出来，实机上只剩硬件相关问题。
