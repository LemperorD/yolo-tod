# 部署：从训练变体到实机（TensorRT / RKNN / ONNX Runtime / OpenVINO）

> 面向 `src/tod`（训练）→ `src/todrt`（推理）的落地流程。
> 架构与 API 见 [`src/todrt/README.md`](../src/todrt/README.md)；这里只讲**怎么做**。

## 后端矩阵：先选路，再动手

同一份部署配置（`todrt.deploy/v1`）可以喂给四个后端，**换后端只改一个字段**
（`builder`）加对应的设备/精度设置：

| 目标设备 | builder | device | 典型配置 | 说明 |
|---|---|---|---|---|
| Jetson Orin / Xavier | `yolov8-trt` | `dla` | `--preset=orin` | DLA + FP16；NVIDIA dGPU 用 `gpu` |
| **RK3588 / RK3588S** | `rknn` | `npu` | `--preset=rk3588` | NPU 3 核 + INT8；**uint8 NHWC 输入** |
| **AMD x86（无独显）** | `openvino` 或 `ort` | `cpu` | `--preset=amd` | 两条 CPU 后路，实测取快的那个 |
| Intel x86（有 iGPU） | `openvino` | `gpu` | `--preset=amd` + `ov_device=GPU` | 同一份代码切设备 |
| 任意机器（对照基线） | `ort` | `cpu` | `--preset=amd --builder=ort` | 回答"加速器快多少、精度掉多少" |

> 为什么 CPU 上留**两个**后端：ONNX Runtime 与 OpenVINO 的 kernel 覆盖与线程策略不同，
> 同一台 AMD 机器上二者互有胜负（实测差异常在 20% 量级）。留着两条路，换 `builder`
> 就能对比，比事后重写一遍省事。

---

## 0. 全流程一览

```
① 训练（本机 GPU）                       ② 导出（本机或构建机）
   variants/SPAE-YOLOv8n/variant.yaml       tools/export_onnx.py
   tools/train.py                           ├─ 重建变体图 + EP5 头部手术
        │                                   ├─ 重算检测头 stride
        ▼                                   ├─ 导出 ONNX（含 anchor 数校验）
   results/.../weights/best.pt  ──────────▶ └─ 写部署配置 JSON
                                                 │
③ 建产物（**看后端**）                            │  configs/deploy/spae-yolov8n.json
   TensorRT : 目标机上 todrt_cli --save-engine   │
   RKNN     : x86 主机上 tools/convert_rknn.py   │
   ORT/OV   : 不需要（直接读 .onnx）              ▼
                                          ④ 集成（目标机）
                                              libtodrt + Detector::CreateFromFile()
```

**四条铁律**（踩过就知道为什么）：

1. **ONNX 必须由本库的导出脚本产出**。`tod.engine.surgery` 会在建模后替换检测头分支
   卷积，`yolo export` 拿到的是没做过手术的图 —— 权重与图不匹配。
2. **TensorRT engine 不能跨设备拷贝**（与 TRT 版本 + GPU 架构 + DLA 配置强绑定）。
   **.rknn 可以跨同型号设备拷贝**（离线转换产物，与板上驱动无关）。
   ONNX Runtime / OpenVINO 直接读 ONNX，没有这份产物。
3. **结构信息只写一次**。`nc / reg_max / strides / layout` 全部由 Python 写进部署配置，
   C++ 侧读取。解码器会在启动时把 anchor 数与引擎输出对拍，不一致直接抛异常
   （见 `decode.cpp: check_anchors`）—— 因为「框全乱但程序不报错」是最贵的 bug。
4. **前处理的输出类型/布局必须与后端成对**：TensorRT/ORT/OpenVINO 是 `float32 + nchw`，
   RKNN 量化模型是 `uint8 + nhwc`。配错不会报错，只会给出全错的框 ——
   `resolve_preprocess_options()` 会在冲突时打 warn，并以**引擎声明**为准。

---

## 1. 导出 ONNX 与部署配置

```bash
# 变体 + 权重 → ONNX + 部署配置
python tools/export_onnx.py \
    --variant variants/SPAE-YOLOv8n/variant.yaml \
    --weights results/SPAE-YOLOv8n/weights/best.pt \
    --imgsz 640 \
    --out exports/ \
    --deploy-config configs/deploy/spae-yolov8n.json
```

导出阶段会打印并校验：

```
[导出] SPAE-YOLOv8n: imgsz=640 strides=[4, 8, 16, 32] reg_max=16
       layout=anchor-major-dfl 预期 anchor=34000
[ONNX] exports/SPAE-YOLOv8n.onnx
[校验] 输出 output0: [1, 84, 34000]
[校验] ✅ 输出 anchor 数与预期一致（34000）
[配置] 部署配置已写出：configs/deploy/spae-yolov8n.json
```

> `anchor=34000` 是 P2–P5 四层头的结果（160²+80²+40²+20²）。
> 官方 YOLOv8n 只有 P3–P5（`anchor=8400`）。**这里差 4 倍**，写在部署配置里，
> C++ 端只信这个数字。

### 关于 NMS 放哪

| 方案 | 命令 | 优点 | 代价 |
|---|---|---|---|
| **CPU 解码 + NMS**（默认） | 不带 `--nms` | 阈值可随时改；支持 soft-NMS / 按类别抑制；布局清晰 | 34000 候选的 CPU 解码在 Orin 上约 5–10 ms |
| **插件 NMS 进图** | `--nms` | 后处理几乎免费 | 改阈值要重建 engine；`layout` 变成 `plugin-nms` |

小目标场景的候选量本来就大（P2 层 25600 个 anchor），所以两条路都值得实测一遍。

---

## 2. 在目标机上构建 engine

### 2.1 先看硬件能力

```bash
cd src/todrt && cmake -B build -DTODRT_WITH_TENSORRT=ON && cmake --build build -j$(nproc)
./build/todrt_cli probe
```

输出会明确告诉你：TensorRT 版本、GPU 名称与 compute capability、**DLA core 数**，
以及当前 TensorRT 是否还支持 DLA（**11.0 起上游移除了 DLA**，Orin 请用 JetPack 6.x 的 10.x）。

### 2.2 配置装配自检（不需要 GPU，几毫秒）

```bash
./build/todrt_cli dryrun configs/deploy/spae-yolov8n.json
```

```
装配自检（未加载引擎）
  模型      : SPAE_YOLOv8n
  构建器    : yolov8-trt  [可用: TensorRT 10.7 / Orin / DLA core=2]
  前处理    : detect_letterbox
  后处理    : detect_nms
  设备/精度 : dla / fp16
  输出布局  : anchor-major-dfl  strides=4,8,16,32
输入 640×640 strides={4,8,16,32} → anchor=160×160 + 80×80 + 40×40 + 20×20 = 34000
```

配置写错（strides 少一层、nc 不对、DLA+FP32、feature-major 缺 `level_channels`）
在这一步就会报出来，**不用等 5 分钟的 engine 构建**。

### 2.3 构建并落盘

```bash
# Orin：DLA + FP16 + 固定 shape（最稳）
./build/todrt_cli dryrun configs/deploy/spae-yolov8n.json \
    --preset=orin --save-engine=engines/spae_orin_fp16.engine

# dGPU：FP16 + CUDA Graph
./build/todrt_cli dryrun cfg.json --preset=dgp --save-engine=engines/spae_dgp.engine

# 想要 INT8（Orin 多路视频时值得）：见 §5
```

---

## 3. 集成到业务代码

```cpp
#include "todrt/factory.hpp"

// 一行装配：模型名 + 部署配置（结构信息全在配置里）
auto det = todrt::Detector::CreateFromFile("configs/deploy/spae-yolov8n-orin.json");
std::cout << det->Describe();          // 打印实际生效的链路，便于日志留证

// 同步（最简单）
todrt::ImageView frame{bgr.data, w, h, 3, (size_t)stride, /*bgr=*/true};
std::vector<todrt::Detection> dets = det->Run(frame);
for (const auto& d : dets) {
  // d.box 已经是**原图像素坐标**（已做 letterbox 反变换并按原图裁剪）
  draw(d.box, d.class_id, d.score);
}
```

异步（多路视频 / ROS 回调里不阻塞）：

```cpp
det->Submit(frame);                    // 内部拷贝像素，帧缓冲可立刻复用
todrt::InferenceResult r;
while (det->TryGet(&r)) { publish(r.detections); }
```

CMake 集成：

```cmake
add_subdirectory(src/todrt)
target_link_libraries(your_app PRIVATE todrt)
# 注意：src/models/*.cpp 必须编进你的可执行目标（自注册，见 src/todrt/README §1）
target_sources(your_app PRIVATE ${TODRT_REGISTRY_SOURCES})
```

---

## 4. 硬件加速：怎么选、怎么验

### 4.1 选择顺序（Orin 上）

1. **先 FP16 + GPU 跑通**，记录延迟与 mAP（这是基准）；
2. 打开 **CUDA Graph**（`runtime.cuda_graphs: true`）：Orin 的 CPU 弱，launch 开销占比高；
3. 再试 **DLA**（`hardware: {device: "dla", allow_gpu_fallback: true}`）：
   看构建日志里的 `DLA 层分布：DLA=x GPU=y`；
4. 最后才考虑 **INT8**（需要校准集，小目标对量化误差敏感）。

### 4.2 为什么 `allow_gpu_fallback` 默认是 true

DLA 只支持有限的算子集合。SPAE-YOLOv8n 里有两处需要留意：

- **ADown**（主干下采样）：含 avg_pool / max_pool / 分块拼接，属于逐元素+池化范畴，
  通常可上 DLA；
- **Efficient_UAVDet 的分组卷积**（`g = x/16`）：分组卷积在 DLA 上的支持随版本与
  分组数变化，**必须实测**。

任何一层落回 GPU，都会引入一次 DLA↔GPU 切换 + 额外的中间张量搬运，
代价常常超过那层本身的计算。所以「DLA 层数很多但 FPS 没涨」通常意味着
中间有几个被迫回退的层把流水线切碎了 —— 这也是为什么先跑 `allow_gpu_fallback: true`
拿到层分布，再决定要不要调结构。

### 4.3 多 DLA core（Orin NX/AGX 有 2 个）

```cpp
todrt::DetectorOptions a = todrt::DetectorOptions::FromFile(cfg);
todrt::DetectorOptions b = a;
a.dla_core = 0; b.dla_core = 1;
auto detA = todrt::Detector::Create("SPAE-YOLOv8n", a);  // 两路视频各占一个 core
auto detB = todrt::Detector::Create("SPAE-YOLOv8n", b);
```

注意每个 core 各自持有一份 engine 与缓冲，内存要算够。

### 4.4 实测延迟

```bash
./build/todrt_cli bench cfg.json sample.ppm 200
```

```
== 实测（200 次，预热 10 次）==
  推理   :    4.312 ms
  后处理 :    7.845 ms
  端到端 :   12.401 ms  →  80.6 FPS
```

**必须看数字再下结论**。经验上小目标变体的瓶颈常常在后处理（P2 层的 25600 个
anchor 带来的解码量），而不是卷积 —— 这种情况换成插件版 NMS 立刻见效。

---

## 5. RK3588（RKNN NPU）

RK3588 有 **3 个 NPU core**（每个约 2 TOPS INT8，合计 6 TOPS），走 `librknnrt.so`。
与 NVIDIA 路线最大的不同：**模型是离线转换产物**，板上不编译。

### 5.1 构建（板上）

```bash
# 从 rknpu2 取 runtime：Linux/librknn_api/aarch64/librknnrt.so 与 rknn_api.h
sudo cp librknnrt.so /usr/lib/ && sudo cp rknn_api.h /usr/include/
sudo ldconfig

cmake -S src/todrt -B build/rk3588 -DTODRT_WITH_RKNN=ON && cmake --build build/rk3588 -j
./build/rk3588/todrt_cli probe     # 看 librknnrt 版本、/dev/rknpu* 是否存在
```

权限是第一个坑：`/dev/rknpu*` 通常需要 `video` 或 `render` 组：

```bash
sudo usermod -aG video,render $USER   # 重新登录生效
ls -l /dev/rknpu*                     # crw-rw---- root video
```

### 5.2 转换（**x86 主机**上做，不在板上）

```bash
pip install rknn-toolkit2

python tools/convert_rknn.py \
    --onnx exports/SPAE-YOLOv8n.onnx \
    --target rk3588 \
    --out exports/spae_yolov8n.rk3588.rknn \
    --dataset exports/calib.txt \
    --deploy-config configs/deploy/spae-yolov8n.json
```

转换脚本做三件事，其中第三件是**这个后端最容易出错的地方**：

1. 把 ONNX 转成 `.rknn`（含量化）；
2. 校准用真实场景图片（`--dataset`，建议 100–300 张）。不给的话 toolkit 用随机数据
   校准，精度通常明显更差；
3. **把归一化写进模型**（`--std 255 255 255`），因此 C++ 侧前处理必须给 **uint8 原始
   像素**。`convert_rknn.py --deploy-config` 会自动把部署配置改成
   `preprocess: {output: "uint8", layout: "nhwc"}` + `builder: "rknn"`。

   为什么必须成对：归一化在宿主机与模型里各做一次 = 双重归一化。它**不会报错**，
   只会让所有框都错 —— 这正是本库把前处理输出类型做成显式契约的原因
   （见 `preprocess.cpp` 顶部注释）。

### 5.3 运行

```bash
./build/rk3588/todrt_cli dryrun configs/deploy/spae-yolov8n.json   # 装配自检，秒级
./build/rk3588/todrt_cli bench  configs/deploy/spae-yolov8n.json sample.ppm 200
```

多路视频用满 3 个 core：

```json
{ "rknn": { "core_num": 3, "model": "exports/spae_yolov8n.rk3588.rknn" } }
```

`core_num=3` 时引擎内部 `rknn_dup_context` 出 3 个独立上下文，推理请求按 core 轮转，
每路视频天然落到不同 core。**单 context 只会用一个 core** —— "买了 6 TOPS 只跑出
2 TOPS" 基本都是这个原因（`engine_info()` 会提示）。

### 5.4 RKNN 上的已知限制

| 限制 | 说明 / 处理 |
|---|---|
| batch 固定为 1 | `.rknn` 的 batch 在转换期定死；多 batch 要重新转换 |
| 输入尺寸固定 | 同上；要换分辨率就重新转换（DLA 也是这个性质） |
| 前处理必须是 uint8 NHWC | 归一化已烧进模型；配错不报错但全错 |
| 算子支持有限 | ADown 的 pool/chunk、Efficient_UAVDet 的分组卷积都要实测；转换失败时先看 toolkit 报的算子 |
| 量化精度 | 小目标对量化极敏感 → **必须复测 `AP_small`**；必要时用 `--quantized-algorithm mmse` 或 `--no-quantize` 做对照 |

---

## 6. ONNX Runtime / OpenVINO（CPU 后路，含 AMD x86）

目标：**没有独显/NPU 的机器也能跑这一套**，同时给加速器提供对照基线。

### 6.1 构建

```bash
# ONNX Runtime：下载官方预编译包（含头文件与 .so）
#   https://github.com/microsoft/onnxruntime/releases
cmake -S src/todrt -B build/x86 \
      -DTODRT_WITH_ORT=ON -DTODRT_ORT_ROOT=/opt/onnxruntime-linux-x64-1.18.0 \
      -DTODRT_WITH_OPENVINO=ON
cmake --build build/x86 -j
```

> AMD 机器上装 `onnxruntime`（CPU 包）即可，不需要 ROCm。
> 装 `onnxruntime-gpu` 且机器有 NVIDIA GPU 时，`ort_provider=cuda` 会自动用上。

### 6.2 运行

```bash
./build/x86/todrt_cli dryrun cfg.json --preset=amd --builder=openvino
./build/x86/todrt_cli bench  cfg.json sample.ppm 200

# 两条 CPU 后路对比（同一份配置，只换 builder）
./build/x86/todrt_cli bench cfg.json sample.ppm 200 --builder=ort
./build/x86/todrt_cli bench cfg.json sample.ppm 200 --builder=openvino
```

可调项：

```json
{
  "builder": "openvino",
  "preset": "amd",
  "openvino": {
    "device": "CPU",
    "cache_dir": "ov_cache",
    "num_streams": 4,
    "num_threads": 4,
    "performance_hint": "LATENCY"
  },
  "onnxruntime": { "provider": "cpu", "intra_threads": 4, "optimization": "all" }
}
```

### 6.3 两点实践经验

1. **OpenVINO 首次编译慢（几十秒）**：`cache_dir` 开起来后第二次启动接近瞬时。
   `--preset=amd` 默认就设了缓存目录。
2. **EP 不可用必须能看见**：ORT 侧默认"EP 不可用则退回 CPU"，但每次都打 warn；
   配 `flags: ["ort_disable_cpu_fallback"]` 可以让它直接失败。
   "以为在用 GPU，其实在跑 CPU" 是最常见的部署错觉。

### 6.4 什么时候选哪个

| 场景 | 建议 |
|---|---|
| AMD x86，纯 CPU | 两个都测；OpenVINO 通常在 Intel 上更强，AMD 上 ORT 也常胜 —— 用实测决定 |
| Intel x86 有 iGPU | OpenVINO + `ov_device: "GPU"` |
| 需要和 GPU/NPU 结果对拍 | `--builder=ort`（CPU EP 的数值最"标准"，便于定位是模型问题还是后端问题） |
| 只想快速验证配置链路 | 任意后端 `dryrun` 即可（不需要模型文件也能跑装配自检） |

---

## 7. INT8（TensorRT 路线，可选）

TensorRT 10+ 是**强类型**：精度由 ONNX 里的张量类型与 Q/DQ 节点决定，
不再靠 `BuilderFlag::kINT8`。两条路：

1. **PTQ（推荐先试）**：在导出阶段做量化感知的 ONNX（`int8=True`），C++ 侧
   `precision: "int8"`。需要校准集覆盖真实场景（小目标、低对比度、夜航都要有）。
2. **TensorRT 自己的校准**（TensorRT ≤ 10.6 的 `IInt8EntropyCalibrator2`）：
   `quantization: {calib_cache, calib_data}`。

⚠️ 小目标对量化误差非常敏感：**任何 INT8（TRT 或 RKNN）之后都必须复测 `AP_small`**。

---

## 8. 常见故障与定位

| 现象 | 后端 | 原因 | 处理 |
|---|---|---|---|
| `anchor 数与 strides 推算不一致` | 全部 | 配置的 strides/input 与导出图不一致 | 用 `export_onnx.py` 重新生成配置，别手改 |
| `输出通道数 x 与 4*reg_max+nc 不符` | 全部 | nc/reg_max 写错 | 同上；这是**保护**，不是 bug |
| 框整体偏移几个像素 | 全部 | `pad_multiple` 与导出时不一致 | 统一为 32（Ultralytics auto=True 默认） |
| 框全乱 | 全部 | ONNX 没做 EP5 手术 / 权重不匹配 | 必须用 `tools/export_onnx.py` 导出 |
| 框全乱，且刚切到 RKNN | RKNN | **双重归一化**（前处理没切成 uint8） | 用 `convert_rknn.py --deploy-config` 重写配置 |
| `RKNN 错误码 -1 / -3` | RKNN | `.rknn` 与 runtime 版本不匹配，或模型无效 | 用匹配版本的 rknn-toolkit2 重新转换 |
| `NPU 设备不可用` | RKNN | `/dev/rknpu*` 权限或驱动缺失 | `ls -l /dev/rknpu*`；把用户加进 `video/render` 组 |
| FPS 远低于预期 | RKNN | 只用了一个 core | `rknn.core_num: 3` |
| `engine 反序列化失败` | TRT | engine 换了机器 / TRT 版本 | 重新构建；engine 不可移植（.rknn 可移植） |
| `DLA core=0` 但设备是 Orin | TRT | TensorRT ≥ 11（DLA 已移除） | 用 JetPack 6.x 自带的 TRT 10.x |
| `buildSerializedNetwork 失败` | TRT | DLA standalone 有层不支持 / workspace 不足 | 开 `allow_gpu_fallback`；调 `workspace_mb` |
| 延迟明显偏高，`probe` 说在用 CPU | ORT | 请求了 cuda EP 但装的是 CPU 版 onnxruntime | 装 `onnxruntime-gpu`，或接受 CPU 路径 |
| OpenVINO 启动很慢 | OpenVINO | 首次 `compile_model` | 配 `ov_cache_dir`（`--preset=amd` 默认开） |
| 想验证配置但手头没模型 | 全部 | — | `todrt_cli dryrun` 不需要模型文件 |

---

## 9. 与训练侧的口径对齐（别忘了）

部署不是终点。按 `PLAN.md §7.2`，**每种后端/精度都要给出精度代价**：

- 建议基线矩阵：`ONNX(CPU)` → `FP16` → `INT8`/`RKNN` → `DLA`，每档都要有
  `AP50:95` 与 **`AP_small`**；
- 端到端延迟（含预处理与后处理）与峰值内存；
- 切片推理（SAHI）与非切片两套结果。

只报 FPS 不报 `AP_small`，会掩盖"加速把小目标加速没了"这种情况 ——
这在 RKNN 的 INT8 量化上尤其容易发生（P2 层的小目标激活值动态范围很小）。
