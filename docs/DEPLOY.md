# 部署：从训练变体到实机（TensorRT / Jetson Orin）

> 面向 `src/tod`（训练）→ `src/todrt`（推理）的落地流程。
> 架构与 API 见 [`src/todrt/README.md`](../src/todrt/README.md)；这里只讲**怎么做**。

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
③ 构建 engine（目标机，必做）                     │  configs/deploy/spae-yolov8n-orin.json
   todrt_cli dryrun cfg.json --save-engine       │
        │                                        ▼
        ▼                              ④ 集成（目标机）
   spae_orin_fp16.engine                    libtodrt + Detector::CreateFromFile()
```

**三条铁律**（踩过就知道为什么）：

1. **ONNX 必须由本库的导出脚本产出**。`tod.engine.surgery` 会在建模后替换检测头分支
   卷积，`yolo export` 拿到的是没做过手术的图 —— 权重与图不匹配。
2. **engine 不能跨设备拷贝**。它与「TensorRT 版本 + GPU 架构 + DLA 配置」强绑定；
   换机器、换 JetPack 版本都必须重新构建（`--onnx` 重新编）。
3. **结构信息只写一次**。`nc / reg_max / strides / layout` 全部由 Python 写进部署配置，
   C++ 侧读取。解码器会在启动时把 anchor 数与引擎输出对拍，不一致直接抛异常
   （见 `decode.cpp: check_anchors`）—— 因为「框全乱但程序不报错」是最贵的 bug。

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

## 5. INT8（可选，收益与风险并存）

TensorRT 10+ 是**强类型**：精度由 ONNX 里的张量类型与 Q/DQ 节点决定，
不再靠 `BuilderFlag::kINT8`。两条路：

1. **PTQ（推荐先试）**：在导出阶段做量化感知的 ONNX（`int8=True`），C++ 侧
   `precision: "int8"`。需要校准集覆盖真实场景（小目标、低对比度、夜航都要有）。
2. **TensorRT 自己的校准**（TensorRT ≤ 10.6 的 `IInt8EntropyCalibrator2`）：
   `quantization: {calib_cache, calib_data}`。

⚠️ 小目标对量化误差非常敏感：**INT8 之后必须复测 `AP_small`，不能只看 FPS**。

---

## 6. 常见故障与定位

| 现象 | 原因 | 处理 |
|---|---|---|
| `anchor 数与 strides 推算不一致` | 部署配置的 strides/input 与导出图不一致 | 用 `export_onnx.py` 重新生成配置，别手改 |
| `输出通道数 x 与 4*reg_max+nc 不符` | nc/reg_max 写错 | 同上；这是**保护**，不是 bug |
| 框整体偏移几个像素 | `pad_multiple` 与导出时不一致 | 统一为 32（Ultralytics auto=True 默认） |
| 框全乱 | ONNX 没做 EP5 手术 / 权重不匹配 | 必须用 `tools/export_onnx.py` 导出 |
| `engine 反序列化失败` | engine 换了机器 / TensorRT 版本 | 重新构建；engine 不可移植 |
| `DLA core=0` 但设备是 Orin | TensorRT ≥ 11（DLA 已移除） | 用 JetPack 6.x 自带的 TRT 10.x |
| `buildSerializedNetwork 失败` | DLA standalone 有层不支持 / workspace 不足 | 开 `allow_gpu_fallback`；调 `workspace_mb` |
| 未找到 TensorRT 但仍要自检 | 想跑工厂/配置/解码检查 | `-DTODRT_WITH_TENSORRT=OFF` 就够 |

---

## 7. 与训练侧的口径对齐（别忘了）

部署不是终点。按 `PLAN.md §7.2`，**部署产物也要给出精度代价**：

- ONNX / FP16 / INT8 / DLA 四档各自的 `AP50:95`、**`AP_small`**；
- 端到端延迟（含预处理与后处理）与峰值内存；
- 切片推理（SAHI）与非切片两套结果。

只报 FPS 不报 `AP_small`，会掩盖"加速把小目标加速没了"这种情况。
