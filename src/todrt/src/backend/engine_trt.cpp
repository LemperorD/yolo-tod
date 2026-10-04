// engine_trt.cpp —— TensorRT 后端实现（仅在有 TensorRT 的开发/构建环境编译）
//
// 目标平台：**Linux（JetPack 6.x / Orin、x86_64 + dGPU）**。本文件不使用任何
// Windows 专有 API；交叉编译只需把 TRT/CUDA 的头与库指向 aarch64 版本。
//
// 版本策略：
//   * TensorRT 10.x   —— 主路径：强类型张量 + enqueueV3 + setTensorAddress
//   * TensorRT 8.5/8.6—— 兼容分支：binding 索引 API + enqueueV2 + kFP16/kINT8 flag
//   * TensorRT >= 11  —— DLA 已被上游移除：显式告警并回落 GPU，而不是静默忽略
//
// 性能要点（都能在下面找到对应代码）：
//   1. 输出缓冲按引擎声明的 dtype 分配（FP16 引擎输出就是 half）；
//   2. 计时用 CUDA event（wall clock 会把 memcpy 等待算进去，虚高）；
//   3. CUDA Graph 在固定 shape 下捕获一次、复用多次（Orin 上 CPU 弱，收益明显）；
//   4. 全程只用一条 non-blocking stream，方便与相机/解码流水线并行。
#include "todrt/backend/engine_trt.hpp"

#if TODRT_HAVE_TENSORRT

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "todrt/backend/factories.hpp"
#include "todrt/backend/simple_detector.hpp"
#include "todrt/backend/trt_factory.hpp"
#include "todrt/factory.hpp"

// 版本比较统一成 (major*1000 + minor)，避免到处写多层 #if
#define TRT_VER (NV_TENSORRT_MAJOR * 1000 + NV_TENSORRT_MINOR)
#define TRT_AT_LEAST(maj, min) (TRT_VER >= ((maj) * 1000 + (min)))

namespace todrt {
namespace {

std::string dims_to_string(const nvinfer1::Dims& d) {
  std::ostringstream oss;
  oss << "[";
  for (int i = 0; i < d.nbDims; ++i) {
    if (i) oss << ",";
    if (d.d[i] < 0) {
      oss << "dyn";
    } else {
      oss << d.d[i];
    }
  }
  oss << "]";
  return oss.str();
}

std::vector<int64_t> dims_to_vec(const nvinfer1::Dims& d) {
  std::vector<int64_t> v;
  v.reserve(static_cast<size_t>(d.nbDims));
  for (int i = 0; i < d.nbDims; ++i) v.push_back(d.d[i]);
  return v;
}

/// 用初值列表构造 nvinfer1::Dims。
///
/// 为什么不直接用 `Dims4{...}`：**TensorRT 10 起 Dims4/Dims3/... 已不存在**
/// （Dims.d 也从 int32_t 改成了 int64_t）。用这个助手可以同时兼容两代 API，
/// 而且把"维度数量"这一处易错点收敛在一个函数里。
nvinfer1::Dims dims_of(std::initializer_list<int64_t> dims) {
  nvinfer1::Dims d;
  int32_t n = 0;
  for (int64_t v : dims) {
    if (n >= nvinfer1::Dims::MAX_DIMS) break;
    d.d[n++] = v;
  }
  d.nbDims = n;
  return d;
}

size_t volume(const std::vector<int64_t>& shape) {
  size_t n = 1;
  for (int64_t d : shape) {
    if (d <= 0) return 0;
    n *= static_cast<size_t>(d);
  }
  return n;
}

/// 半精度 → 单精度。FP16 引擎的输出是 half，而工具链的解码器只吃 float32。
inline float half_to_float(uint16_t h) {
  const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1Fu;
  uint32_t man = h & 0x3FFu;
  uint32_t bits;
  if (exp == 0) {
    if (man == 0) {
      bits = sign;
    } else {  // 次正规数
      exp = 127 - 15 + 1;
      while ((man & 0x400u) == 0) {
        man <<= 1;
        --exp;
      }
      man &= 0x3FFu;
      bits = sign | (exp << 23) | (man << 13);
    }
  } else if (exp == 0x1Fu) {
    bits = sign | 0x7F800000u | (man << 13);
  } else {
    bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
  }
  float out;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

DataType from_trt(nvinfer1::DataType t) {
  switch (t) {
    case nvinfer1::DataType::kFLOAT: return DataType::kF32;
    case nvinfer1::DataType::kHALF: return DataType::kF16;
    case nvinfer1::DataType::kINT8: return DataType::kI8;
    case nvinfer1::DataType::kINT32: return DataType::kI32;
    case nvinfer1::DataType::kUINT8: return DataType::kU8;
    default: return DataType::kF32;
  }
}

/// TensorRT 日志 → todrt 日志，保证全部日志只有一个出口。
class TrtLogger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    if (!msg) return;
    switch (severity) {
      case Severity::kINTERNAL_ERROR:
      case Severity::kERROR: log_error(std::string("[TRT] ") + msg); break;
      case Severity::kWARNING: log_warn(std::string("[TRT] ") + msg); break;
      case Severity::kINFO: log_info(std::string("[TRT] ") + msg); break;
      case Severity::kVERBOSE:
        if (verbose_) log_debug(std::string("[TRT] ") + msg);
        break;
    }
  }
  void set_verbose(bool v) { verbose_ = v; }

 private:
  bool verbose_ = false;
};

#if !TRT_AT_LEAST(10, 0)
/// INT8 熵校准（TRT 10 起改为显式量化，不再需要这个接口）。
class EntropyCalibrator : public nvinfer1::IInt8EntropyCalibrator2 {
 public:
  EntropyCalibrator(int batch, int channels, int height, int width, const std::string& cache)
      : batch_(batch), cache_(cache), ch_(channels), h_(height), w_(width) {
    host_.assign(static_cast<size_t>(batch) * channels * height * width * sizeof(float), 0.f);
    if (cudaMalloc(&device_, host_.size()) != cudaSuccess) {
      throw TritError("INT8 校准：cudaMalloc 失败");
    }
  }
  ~EntropyCalibrator() override {
    if (device_) cudaFree(device_);
  }

  int getBatchSize() const noexcept override { return batch_; }

  bool getBatch(void* bindings[], const char* names[], int nb_bindings) noexcept override {
    (void)names;
    (void)nb_bindings;
    if (bindings == nullptr || bindings[0] == nullptr) return false;
    // 由调用方（Detector 层）在需要时填充 host_；这里只做一次 H2D 拷贝。
    if (cudaMemcpyAsync(device_, host_.data(), host_.size(), cudaMemcpyHostToDevice, stream_) !=
        cudaSuccess) {
      return false;
    }
    cudaStreamSynchronize(stream_);
    bindings[0] = device_;
    return true;
  }

  const void* readCalibrationCache(size_t& length) noexcept override {
    blob_.clear();
    if (cache_.empty()) return nullptr;
    std::ifstream in(cache_, std::ios::binary);
    if (!in) return nullptr;
    in >> std::noskipws;
    blob_.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    length = blob_.size();
    return blob_.empty() ? nullptr : blob_.data();
  }

  void writeCalibrationCache(const void* ptr, size_t length) noexcept override {
    if (cache_.empty()) return;
    std::ofstream out(cache_, std::ios::binary);
    out.write(static_cast<const char*>(ptr), static_cast<std::streamsize>(length));
  }

 private:
  int batch_;
  std::string cache_;
  int ch_, h_, w_;
  std::vector<float> host_;
  std::vector<char> blob_;
  void* device_ = nullptr;
  cudaStream_t stream_ = nullptr;
};
#endif

}  // namespace

// ------------------------------------------------------------------ Impl

struct TrtEngine::Impl {
  TrtLogger logger;
  std::unique_ptr<nvinfer1::IRuntime> runtime;
  std::unique_ptr<nvinfer1::ICudaEngine> engine;
  std::unique_ptr<nvinfer1::IExecutionContext> context;
  std::unique_ptr<nvinfer1::IBuilder> builder;
  std::unique_ptr<nvinfer1::INetworkDefinition> network;
  std::unique_ptr<nvonnxparser::IParser> parser;
  std::vector<std::unique_ptr<nvinfer1::IOptimizationProfile>> profiles;
#if !TRT_AT_LEAST(10, 0)
  std::unique_ptr<EntropyCalibrator> calibrator;
#endif

  cudaStream_t stream = nullptr;
  cudaEvent_t ev_start = nullptr;
  cudaEvent_t ev_stop = nullptr;
  double last_infer_ms = 0.0;

  std::string input_name;
  std::vector<std::string> output_names;
  std::vector<DataType> output_types;
  std::vector<std::vector<int64_t>> output_shapes;
  std::vector<int64_t> input_shape;

  void* dev_input = nullptr;
  size_t dev_input_bytes = 0;
  std::vector<void*> dev_outputs;
  std::vector<size_t> dev_output_bytes;

  cudaGraphExec_t graph_exec = nullptr;
  bool graph_ready = false;
  bool cuda_graphs = false;
  bool dynamic = false;

  // DLA 层归属统计（ClassifyLayers 填充，DlaLayerCount/GpuLayerCount/LayerInfo 读取）。
  // 这是排查"为什么没上 DLA / FPS 没起来"的第一手信息，必须保留。
  int dla_layers = 0;
  int gpu_layers = 0;
  std::vector<std::string> layer_info;

  BuildConfig cfg;

  ~Impl() { Release(); }

  void Release() {
    if (graph_exec) {
      cudaGraphExecDestroy(graph_exec);
      graph_exec = nullptr;
    }
    graph_ready = false;
    if (dev_input) {
      cudaFree(dev_input);
      dev_input = nullptr;
    }
    for (void* p : dev_outputs) {
      if (p) cudaFree(p);
    }
    dev_outputs.clear();
    dev_output_bytes.clear();
    if (ev_start) {
      cudaEventDestroy(ev_start);
      ev_start = nullptr;
    }
    if (ev_stop) {
      cudaEventDestroy(ev_stop);
      ev_stop = nullptr;
    }
    if (stream) {
      cudaStreamDestroy(stream);
      stream = nullptr;
    }
    context.reset();
    engine.reset();
    parser.reset();
    network.reset();
    builder.reset();
    runtime.reset();
  }
};

TrtEngine::TrtEngine() : impl_(new Impl()) {}
TrtEngine::~TrtEngine() = default;
bool TrtEngine::valid() const { return impl_ && impl_->engine && impl_->context; }

// ------------------------------------------------------------------ Probe

TrtEngine::PlanInfo TrtEngine::Probe(const BuildConfig& cfg) {
  PlanInfo info;
  info.trt_version = std::to_string(NV_TENSORRT_MAJOR) + "." + std::to_string(NV_TENSORRT_MINOR);
  info.max_batch = cfg.max_batch;
#if TRT_AT_LEAST(11, 0)
  info.dla_supported_build = false;
#else
  info.dla_supported_build = true;
#endif
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) {
    log_warn("Probe: 无法访问 CUDA 设备（无 GPU 的构建机上属正常现象）");
    return info;
  }
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, device) == cudaSuccess) {
    info.device_name = prop.name;
    info.compute_capability = std::to_string(prop.major) + "." + std::to_string(prop.minor);
  }
#if !TRT_AT_LEAST(11, 0)
  if (info.dla_supported_build) {
    static TrtLogger probe_logger;
    std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(probe_logger));
    if (builder) {
      info.nb_dla_cores = builder->getNbDLACores();
      if (info.nb_dla_cores > 0) info.max_batch = builder->getMaxDLABatchSize();
    }
  }
#endif
  return info;
}

// ------------------------------------------------------------------ Build

bool TrtEngine::Build(const BuildConfig& cfg, std::string* err) {
  auto fail = [&](const std::string& m) {
    if (err) *err = m;
    log_error(m);
    return false;
  };

  if (cfg.onnx_path.empty()) return fail("BuildConfig.onnx_path 为空");
  impl_->cfg = cfg;
  impl_->logger.set_verbose(cfg.verbose);
  impl_->cuda_graphs = cfg.has(BuilderFlag::kCudaGraphs);
  impl_->dynamic = cfg.dynamic_shape || cfg.dynamic_batch;

  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) {
    return fail("无法访问 CUDA 设备：请确认在目标机上运行，且驱动/nvidia-smi 正常。");
  }

#if TRT_AT_LEAST(11, 0)
  if (cfg.device == Device::kDla) {
    return fail("当前 TensorRT " + std::to_string(NV_TENSORRT_MAJOR) + "." +
                std::to_string(NV_TENSORRT_MINOR) +
                " 已移除 DLA 支持（上游自 11.0 起不再支持）。\n"
                "  可选：① 使用 TensorRT 10.x（JetPack 6.x 默认）；② device 设为 gpu。");
  }
#endif

  impl_->builder.reset(nvinfer1::createInferBuilder(impl_->logger));
  if (!impl_->builder) return fail("createInferBuilder 失败（TensorRT 运行库是否可用？）");

  uint32_t net_flags = 0;
#if !TRT_AT_LEAST(10, 0)
  net_flags |= 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
#endif
  impl_->network.reset(impl_->builder->createNetworkV2(net_flags));
  if (!impl_->network) return fail("createNetworkV2 失败");

  impl_->parser.reset(nvonnxparser::createParser(*impl_->network, impl_->logger));
  if (!impl_->parser) return fail("createParser 失败（onnx parser 是否随 TensorRT 一起安装？）");
  if (!impl_->parser->parseFromFile(cfg.onnx_path.c_str(),
                                    static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
    std::ostringstream oss;
    oss << "解析 ONNX 失败：" << cfg.onnx_path << "\n";
    for (int i = 0; i < impl_->parser->getNbErrors(); ++i) {
      const auto* e = impl_->parser->getError(i);
      if (e) oss << "  - " << e->desc() << "\n";
    }
    return fail(oss.str());
  }

  // 输入名与 ONNX 对齐检查（写错名字是 profile 设置失败最常见的原因）
  if (impl_->network->getNbInputs() < 1) return fail("ONNX 里没有输入张量");
  const char* onnx_input = impl_->network->getInput(0)->getName();
  if (!cfg.input_name.empty() && cfg.input_name != onnx_input) {
    log_warn("配置里 input_name=" + cfg.input_name + "，但 ONNX 输入名是 " + onnx_input +
             "；以 ONNX 为准。");
  }
  const std::string input_name = onnx_input;
  log_info(std::string("ONNX 输入：") + input_name + " " +
           dims_to_string(impl_->network->getInput(0)->getDimensions()));

  // 先把"哪些层能上 DLA"统计出来（在构建前问 builder 最准）
  ClassifyLayers();

  std::unique_ptr<nvinfer1::IBuilderConfig> config(impl_->builder->createBuilderConfig());
  if (!config) return fail("createBuilderConfig 失败");
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                             cfg.workspace_mb * 1024ULL * 1024ULL);
#if TRT_AT_LEAST(8, 5)
  if (cfg.timing_cache) {
    config->setProfilingVerbosity(nvinfer1::ProfilingVerbosity::kLAYER_NAMES_ONLY);
  }
#endif
#if TRT_AT_LEAST(8, 6)
  if (cfg.num_build_threads > 0) impl_->builder->setMaxThreads(cfg.num_build_threads);
#endif
#if TRT_AT_LEAST(8, 6)
  if (cfg.hardware_compatibility) {
    config->setHardwareCompatibilityLevel(nvinfer1::HardwareCompatibilityLevel::kAMPERE_PLUS);
  }
#endif

  // ---- 精度 ----
#if !TRT_AT_LEAST(10, 0)
  if (cfg.precision == Precision::kFP16) {
    if (!impl_->builder->platformHasFastFp16()) log_warn("设备不支持快速 FP16（性能可能下降）。");
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
  }
  if (cfg.precision == Precision::kINT8) {
    if (!impl_->builder->platformHasFastInt8()) log_warn("设备不支持快速 INT8（性能可能下降）。");
    config->setFlag(nvinfer1::BuilderFlag::kINT8);
    if (!cfg.int8_calib_cache.empty()) {
      impl_->calibrator.reset(new EntropyCalibrator(1, 3, cfg.shape_opt[1], cfg.shape_opt[0],
                                                    cfg.int8_calib_cache));
      config->setInt8Calibrator(impl_->calibrator.get());
    } else {
      log_warn("未提供 int8_calib_cache：将只使用 ONNX 中已有的量化信息。");
    }
  }
  if (cfg.has(BuilderFlag::kSparsity)) {
    config->setFlag(nvinfer1::BuilderFlag::kSPARSE_WEIGHTS);
  }
#else
  if (cfg.precision == Precision::kINT8) {
    log_info(
        "TensorRT 10+ 为强类型：INT8 需要在 ONNX 中携带 Q/DQ 量化信息"
        "（Python 侧导出时用 int8=True 或 PTQ 流程），否则精度不会真正变为 INT8。");
  }
#endif

  // ---- DLA ----
  const bool want_dla = (cfg.device == Device::kDla);
#if !TRT_AT_LEAST(11, 0)
  if (want_dla) {
    const int cores = impl_->builder->getNbDLACores();
    if (cores <= 0) {
      if (cfg.has(BuilderFlag::kDlaStandalone)) {
        return fail(
            "请求 dla_standalone 但当前设备没有 DLA core。"
            "x86 dGPU 没有 DLA；请改用 Jetson/Orin，或把 device 设为 gpu。");
      }
      log_warn("当前设备没有可用 DLA core → 自动回落到 GPU 执行。");
    } else {
      if (cfg.dla_core < 0 || cfg.dla_core >= cores) {
        return fail("dla_core=" + std::to_string(cfg.dla_core) + " 超出可用范围 [0," +
                    std::to_string(cores - 1) + "]（本机 DLA core 数=" + std::to_string(cores) +
                    "）");
      }
      config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
      config->setDLACore(cfg.dla_core);
      if (cfg.allow_gpu_fallback) {
        config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
      } else {
        log_warn(
            "allow_gpu_fallback=false：要求全部层都跑在 DLA 上，任何不支持的层都会让构建失败。"
            "（分组卷积/softmax 等算子在 DLA 上的支持随版本变化，建议先用 true 跑通。）");
      }
      if (cfg.dla_memory_limit_mb > 0) {
        config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kDLA_MANAGED_SRAM,
                                   static_cast<size_t>(cfg.dla_memory_limit_mb) * 1024ULL *
                                       1024ULL);
      }
      log_info("DLA 启用：core=" + std::to_string(cfg.dla_core) + "/" + std::to_string(cores) +
               " gpu_fallback=" + (cfg.allow_gpu_fallback ? "true" : "false") +
               " precision=" + to_string(cfg.precision));
    }
  }
#endif

  // ---- 动态 shape profile ----
  if (impl_->dynamic) {
    const int64_t bmin = cfg.dynamic_batch ? 1 : cfg.max_batch;
    const int64_t bopt = cfg.dynamic_batch ? std::max<int64_t>(1, cfg.max_batch / 2) : cfg.max_batch;
    const int64_t bmax = cfg.max_batch;
    const int64_t wmin = cfg.shape_min[0], hmin = cfg.shape_min[1];
    const int64_t wopt = cfg.shape_opt[0], hopt = cfg.shape_opt[1];
    const int64_t wmax = cfg.shape_max[0], hmax = cfg.shape_max[1];
    if (wmin > wmax || hmin > hmax) {
      return fail("shape profile 非法：min 大于 max（检查配置里的 shape.min / shape.max）");
    }
#if TRT_AT_LEAST(8, 5)
    auto* profile = impl_->builder->createOptimizationProfile();
    if (!profile) return fail("createOptimizationProfile 失败");
    const bool ok =
        profile->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kMIN,
                               dims_of({bmin, 3, hmin, wmin})) &&
        profile->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kOPT,
                               dims_of({bopt, 3, hopt, wopt})) &&
        profile->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kMAX,
                               dims_of({bmax, 3, hmax, wmax}));
    if (!ok) {
      return fail("设置优化 profile 失败：请确认 ONNX 输入名（" + input_name +
                  "）是 4 维 NCHW，且 min<=opt<=max。");
    }
    if (config->addOptimizationProfile(profile) < 0) return fail("addOptimizationProfile 失败");
    log_info("动态 shape：min=" + std::to_string(wmin) + "×" + std::to_string(hmin) + " opt=" +
             std::to_string(wopt) + "×" + std::to_string(hopt) + " max=" +
             std::to_string(wmax) + "×" + std::to_string(hmax) +
             (cfg.dynamic_batch ? "（batch 动态）" : ""));
#else
    return fail("动态 shape 需要 TensorRT >= 8.5");
#endif
  }

  // ---- 构建 ----
  const auto t0 = std::chrono::steady_clock::now();
  std::unique_ptr<nvinfer1::IHostMemory> serialized(
      impl_->builder->buildSerializedNetwork(*impl_->network, *config));
  const auto t1 = std::chrono::steady_clock::now();
  if (!serialized) {
    return fail(
        "buildSerializedNetwork 失败。常见原因：\n"
        "  1) DLA standalone 下有层不支持 → 去掉 dla_standalone，或保持 allow_gpu_fallback=true；\n"
        "  2) 动态 shape 的 min>max 或 opt 不在区间内；\n"
        "  3) 工作区/显存不足 → 调小 workspace_mb、max_batch 或输入分辨率；\n"
        "  4) ORT/ONNX 算子版本过高 → 降低 ONNX opset 重新导出。");
  }
  log_info("引擎构建完成：用时 " +
           std::to_string(std::chrono::duration<double>(t1 - t0).count()) + " s，engine " +
           std::to_string(serialized->size() / (1024.0 * 1024.0)) + " MB");

  // ---- 反序列化 ----
  impl_->runtime.reset(nvinfer1::createInferRuntime(impl_->logger));
  if (!impl_->runtime) return fail("createInferRuntime 失败");
#if !TRT_AT_LEAST(11, 0)
  if (want_dla && impl_->builder->getNbDLACores() > 0) impl_->runtime->setDLACore(cfg.dla_core);
#endif
  impl_->engine.reset(
      impl_->runtime->deserializeCudaEngine(serialized->data(), serialized->size()));
  if (!impl_->engine) return fail("deserializeCudaEngine 失败");

  const bool ok = Finalize();
  if (!ok) return fail("引擎初始化失败（见上面的日志）");

  if (!cfg.serialize_out.empty()) {
    std::string serr;
    if (!Serialize(cfg.serialize_out, &serr)) log_warn(serr);
  }
  return true;
}

bool TrtEngine::Load(const std::string& engine_path, std::string* err) {
  auto fail = [&](const std::string& m) {
    if (err) *err = m;
    log_error(m);
    return false;
  };
  std::ifstream in(engine_path, std::ios::binary);
  if (!in) return fail("无法打开 engine 文件：" + engine_path);
  in.seekg(0, std::ios::end);
  const std::streamsize size = in.tellg();
  in.seekg(0, std::ios::beg);
  if (size <= 0) return fail("engine 文件为空：" + engine_path);
  std::vector<char> blob(static_cast<size_t>(size));
  if (!in.read(blob.data(), size)) return fail("读取 engine 文件失败：" + engine_path);

  impl_->runtime.reset(nvinfer1::createInferRuntime(impl_->logger));
  if (!impl_->runtime) return fail("createInferRuntime 失败");
  impl_->engine.reset(impl_->runtime->deserializeCudaEngine(blob.data(), blob.size()));
  if (!impl_->engine) {
    return fail("engine 反序列化失败：" + engine_path +
                "\n  engine 与「TensorRT 版本 + GPU 架构 + DLA 配置」强绑定，不能跨设备拷贝。"
                "\n  换机器/换 JetPack 版本请重新构建（本工具 --onnx，或 trtexec）。");
  }
  const bool ok = Finalize();
  if (!ok) return fail("engine 初始化失败（见上面的日志）");
  return true;
}

bool TrtEngine::Serialize(const std::string& path, std::string* err) const {
  if (!impl_->engine) {
    if (err) *err = "Serialize: 引擎为空";
    return false;
  }
  std::unique_ptr<nvinfer1::IHostMemory> blob(impl_->engine->serialize());
  if (!blob) {
    if (err) *err = "engine->serialize() 失败";
    return false;
  }
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    if (err) *err = "无法写入 engine 文件：" + path;
    return false;
  }
  out.write(static_cast<const char*>(blob->data()), static_cast<std::streamsize>(blob->size()));
  if (!out) {
    if (err) *err = "写 engine 文件失败：" + path;
    return false;
  }
  log_info("engine 已落盘：" + path + "（" +
           std::to_string(blob->size() / (1024.0 * 1024.0)) + " MB）");
  return true;
}

// ------------------------------------------------------------------ Finalize

bool TrtEngine::Finalize() {
  Impl& im = *impl_;
  im.context.reset(im.engine->createExecutionContext());
  if (!im.context) {
    log_error("createExecutionContext 失败");
    return false;
  }

  const int nb = im.engine->getNbIOTensors();
  bool got_input = false;
  for (int i = 0; i < nb; ++i) {
    const char* name = im.engine->getIOTensorName(i);
    if (!name) continue;
    const nvinfer1::TensorIOMode mode = im.engine->getTensorIOMode(name);
    const nvinfer1::Dims dims = im.engine->getTensorShape(name);
    const DataType dt = from_trt(im.engine->getTensorDataType(name));
    if (mode == nvinfer1::TensorIOMode::kINPUT) {
      im.input_name = name;
      im.input_shape = dims_to_vec(dims);
      got_input = true;
      log_info(std::string("输入张量  ") + name + " " + dims_to_string(dims) +
               " dtype=" + to_string(dt));
    } else {
      im.output_names.push_back(name);
      im.output_types.push_back(dt);
      im.output_shapes.push_back(dims_to_vec(dims));
      log_info(std::string("输出张量  ") + name + " " + dims_to_string(dims) +
               " dtype=" + to_string(dt));
    }
  }
  if (!got_input) {
    log_error("引擎里没有输入张量");
    return false;
  }
  if (im.output_names.empty()) {
    log_error("引擎里没有输出张量");
    return false;
  }

  if (cudaStreamCreateWithFlags(&im.stream, cudaStreamNonBlocking) != cudaSuccess) {
    log_error("cudaStreamCreateWithFlags 失败");
    return false;
  }
  if (cudaEventCreate(&im.ev_start) != cudaSuccess || cudaEventCreate(&im.ev_stop) != cudaSuccess) {
    log_error("cudaEventCreate 失败");
    return false;
  }

  // 输入设备缓冲
  const size_t in_elems = volume(im.input_shape);
  const DataType in_dt = from_trt(im.engine->getTensorDataType(im.input_name.c_str()));
  im.dev_input_bytes = in_elems * dtype_size(in_dt);
  if (im.dev_input_bytes == 0) {
    im.dev_input_bytes = static_cast<size_t>(im.cfg.max_batch) * 3 *
                         static_cast<size_t>(im.cfg.shape_opt[0]) *
                         static_cast<size_t>(im.cfg.shape_opt[1]) * sizeof(float);
    log_warn("输入为动态 shape：按 opt profile 预分配 " +
             std::to_string(im.dev_input_bytes / (1024 * 1024)) + " MB 设备缓冲");
  }
  if (cudaMalloc(&im.dev_input, im.dev_input_bytes) != cudaSuccess) {
    log_error("cudaMalloc 输入缓冲失败（" + std::to_string(im.dev_input_bytes / (1024 * 1024)) +
              " MB）：请检查显存余量，或减小 max_batch / 输入分辨率。");
    return false;
  }
  im.dev_outputs.assign(im.output_names.size(), nullptr);
  im.dev_output_bytes.assign(im.output_names.size(), 0);
  for (size_t k = 0; k < im.output_names.size(); ++k) {
    size_t bytes = volume(im.output_shapes[k]) * dtype_size(im.output_types[k]);
    if (bytes == 0) {
      // 动态 shape 时按 max profile 估一个上界
      const size_t anchors = static_cast<size_t>(im.cfg.shape_max[0] / 4) *
                                 static_cast<size_t>(im.cfg.shape_max[1] / 4) +
                             static_cast<size_t>(im.cfg.shape_max[0] / 8) *
                                 static_cast<size_t>(im.cfg.shape_max[1] / 8) +
                             static_cast<size_t>(im.cfg.shape_max[0] / 16) *
                                 static_cast<size_t>(im.cfg.shape_max[1] / 16) +
                             static_cast<size_t>(im.cfg.shape_max[0] / 32) *
                                 static_cast<size_t>(im.cfg.shape_max[1] / 32);
      bytes = static_cast<size_t>(im.cfg.max_batch) * 256 * anchors * sizeof(float);
    }
    im.dev_output_bytes[k] = bytes;
    if (cudaMalloc(&im.dev_outputs[k], bytes) != cudaSuccess) {
      log_error("cudaMalloc 输出缓冲失败（" + im.output_names[k] + "，" +
                std::to_string(bytes) + " B）");
      return false;
    }
  }

  if (im.dynamic) {
    log_info("动态 shape 模式：每次 Infer 会按输入尺寸重设 profile（必要时自动切换）");
  }
  if (im.cuda_graphs && !im.dynamic) {
    log_info("CUDA Graph 已启用（首次 Infer 捕获后复用）");
  } else if (im.cuda_graphs) {
    log_warn("CUDA Graph 与动态 shape 不兼容，已自动关闭。");
    im.cuda_graphs = false;
  }
  return true;
}

// ------------------------------------------------------------------ 查询

std::vector<int64_t> TrtEngine::InputShape() const { return impl_->input_shape; }
std::vector<std::vector<int64_t>> TrtEngine::OutputShapes() const {
  // 动态 shape 下重新查询一次，保证与当前 profile 一致
  if (impl_->dynamic && impl_->context) {
    std::vector<std::vector<int64_t>> shapes;
    shapes.reserve(impl_->output_names.size());
    for (const auto& n : impl_->output_names) {
      shapes.push_back(dims_to_vec(impl_->context->getTensorShape(n.c_str())));
    }
    return shapes;
  }
  return impl_->output_shapes;
}
std::vector<DataType> TrtEngine::OutputTypes() const { return impl_->output_types; }
std::string TrtEngine::InputName() const { return impl_->input_name; }
std::vector<std::string> TrtEngine::OutputNames() const { return impl_->output_names; }
double TrtEngine::LastInferMs() const { return impl_->last_infer_ms; }

int TrtEngine::DlaLayerCount() const { return impl_->dla_layers; }
int TrtEngine::GpuLayerCount() const { return impl_->gpu_layers; }

/// 逐层统计"这层能不能上 DLA"。排查"为什么 FPS 没起来"时这是第一手信息：
/// DLA 只支持卷积/池化/逐元素/拼接等有限算子，任何不支持的层都会退回 GPU，
/// 而一次 DLA↔GPU 切换的代价往往比这层本身的计算还大。
void TrtEngine::ClassifyLayers() {
#if !TRT_AT_LEAST(11, 0)
  Impl& im = *impl_;
  if (!im.builder || !im.network) return;
  const int n = im.network->getNbLayers();
  for (int i = 0; i < n; ++i) {
    nvinfer1::ILayer* layer = im.network->getLayer(i);
    if (!layer) continue;
    const char* lname = layer->getName();
    const bool ok = im.builder->canRunOnDLA(layer);
    if (ok) {
      ++im.dla_layers;
    } else {
      ++im.gpu_layers;
      im.layer_info.push_back(std::string(lname ? lname : "(unnamed)") +
                              "  type=" + std::to_string(static_cast<int>(layer->getType())) +
                              " → 只能跑 GPU");
    }
  }
  log_info("DLA 层分布：DLA=" + std::to_string(im.dla_layers) + " GPU=" +
           std::to_string(im.gpu_layers));
  if (im.gpu_layers > 0) {
    for (const auto& l : im.layer_info) log_debug("  " + l);
  }
#endif
}

std::vector<std::string> TrtEngine::LayerInfo() {
  std::vector<std::string> out;
#if TRT_AT_LEAST(8, 5)
  if (!impl_->engine) return out;
  // 10.x：用 IEngineInspector 输出逐层信息（含执行精度与设备）
  std::unique_ptr<nvinfer1::IEngineInspector> inspector(impl_->engine->createEngineInspector());
  if (!inspector) return out;
  inspector->setExecutionContext(impl_->context.get());
  const int32_t n = impl_->engine->getNbLayers();
  for (int32_t i = 0; i < n; ++i) {
    const char* line = inspector->getLayerInformation(
        i, nvinfer1::LayerInformationFormat::kJSON);
    if (line) out.push_back(line);
  }
#endif
  return out;
}

// ------------------------------------------------------------------ 推理

bool TrtEngine::Infer(const void* input, size_t input_bytes, int batch,
                      const std::vector<void*>& outputs, const std::vector<size_t>& output_bytes,
                      std::string* err) {
  Impl& im = *impl_;
  if (!valid()) {
    if (err) *err = "Infer: 引擎未就绪";
    return false;
  }
  if (!input || input_bytes > im.dev_input_bytes) {
    if (err) {
      *err = "Infer: 输入缓冲不合法（bytes=" + std::to_string(input_bytes) + "，设备缓冲=" +
             std::to_string(im.dev_input_bytes) + "）";
    }
    return false;
  }
  if (outputs.size() != im.dev_outputs.size()) {
    if (err) *err = "Infer: 输出缓冲数量与引擎不符";
    return false;
  }

  // 动态 shape：按本次输入尺寸设定
  if (im.dynamic) {
    // 实际尺寸由调用方通过 input_bytes 体现（w*h*3*4*batch）；这里从字节数反推。
    // 注意：这里假设输入是方图（检测模型绝大多数如此）；非方图请在
    // DetectorOptions.shape_* 里把 profile 设成对应的 w/h。
    const size_t per_sample = input_bytes / static_cast<size_t>(std::max(1, batch));
    const size_t hw = per_sample / (3 * sizeof(float));
    const int64_t side = static_cast<int64_t>(std::lround(std::sqrt(static_cast<double>(hw))));
    const nvinfer1::Dims d = dims_of({batch, 3, side, side});
    if (!im.context->setInputShape(im.input_name.c_str(), d)) {
      if (err) *err = "Infer: setInputShape 失败（尺寸是否在 profile 范围内？）";
      return false;
    }
    im.dev_input_bytes = input_bytes;
  }

  if (cudaMemcpyAsync(im.dev_input, input, input_bytes, cudaMemcpyHostToDevice, im.stream) !=
      cudaSuccess) {
    if (err) *err = "Infer: H2D 拷贝失败";
    return false;
  }

#if TRT_AT_LEAST(10, 0)
  if (!im.context->setTensorAddress(im.input_name.c_str(), im.dev_input)) {
    if (err) *err = "Infer: setTensorAddress(input) 失败";
    return false;
  }
  for (size_t k = 0; k < im.output_names.size(); ++k) {
    if (!im.context->setTensorAddress(im.output_names[k].c_str(), im.dev_outputs[k])) {
      if (err) *err = "Infer: setTensorAddress(" + im.output_names[k] + ") 失败";
      return false;
    }
  }
#else
  // 8.x：binding 索引 API（输入/输出索引需要按引擎名称表查一次）
  int in_idx = -1;
  std::vector<int> out_idx(im.output_names.size(), -1);
  for (int i = 0; i < im.engine->getNbBindings(); ++i) {
    const char* n = im.engine->getBindingName(i);
    if (!n) continue;
    if (im.engine->bindingIsInput(i)) {
      in_idx = i;
    } else {
      for (size_t k = 0; k < im.output_names.size(); ++k) {
        if (im.output_names[k] == n) out_idx[k] = i;
      }
    }
  }
  if (in_idx < 0) {
    if (err) *err = "Infer: 找不到输入 binding";
    return false;
  }
  im.context->setBindingAddress(in_idx, im.dev_input);
  for (size_t k = 0; k < out_idx.size(); ++k) {
    if (out_idx[k] >= 0) im.context->setBindingAddress(out_idx[k], im.dev_outputs[k]);
  }
#endif

  cudaEventRecord(im.ev_start, im.stream);

  bool launched = false;
  if (im.cuda_graphs && im.graph_ready && im.graph_exec) {
    launched = (cudaGraphLaunch(im.graph_exec, im.stream) == cudaSuccess);
    if (!launched) log_warn("CUDA Graph launch 失败，回退到普通 enqueue");
  }
  if (!launched) {
#if TRT_AT_LEAST(10, 0)
    launched = im.context->enqueueV3(im.stream);
#else
    launched = im.context->enqueueV2(nullptr, im.stream, nullptr);
#endif
    if (!launched) {
      if (err) *err = "Infer: enqueue 失败（DLA/GPU 执行出错，检查日志）";
      return false;
    }
    // 固定 shape 下捕获一次 graph，后续复用
    if (im.cuda_graphs && !im.dynamic && !im.graph_ready) {
      cudaStreamSynchronize(im.stream);
      cudaGraph_t graph = nullptr;
      if (cudaStreamBeginCapture(im.stream, cudaStreamCaptureModeThreadLocal) == cudaSuccess) {
#if TRT_AT_LEAST(10, 0)
        const bool ok = im.context->enqueueV3(im.stream);
#else
        const bool ok = im.context->enqueueV2(nullptr, im.stream, nullptr);
#endif
        if (ok && cudaStreamEndCapture(im.stream, &graph) == cudaSuccess && graph) {
          // CUDA 12 起 cudaGraphInstantiate 只有 3 个参数（旧的 5 参数版本已移除）
          if (cudaGraphInstantiate(&im.graph_exec, graph, 0) == cudaSuccess) {
            im.graph_ready = true;
            log_info("CUDA Graph 捕获成功，后续推理复用。");
          } else {
            log_warn("CUDA Graph instantiate 失败，继续使用普通 enqueue。");
          }
          cudaGraphDestroy(graph);
        } else {
          cudaStreamEndCapture(im.stream, &graph);
          log_warn("CUDA Graph 捕获失败，继续使用普通 enqueue。");
        }
      }
      cudaEventRecord(im.ev_start, im.stream);
    }
  }

  cudaEventRecord(im.ev_stop, im.stream);

  for (size_t k = 0; k < outputs.size(); ++k) {
    const size_t bytes = std::min(output_bytes[k], im.dev_output_bytes[k]);
    if (cudaMemcpyAsync(outputs[k], im.dev_outputs[k], bytes, cudaMemcpyDeviceToHost, im.stream) !=
        cudaSuccess) {
      if (err) *err = "Infer: D2H 拷贝失败";
      return false;
    }
  }

  if (cudaStreamSynchronize(im.stream) != cudaSuccess) {
    if (err) *err = "Infer: stream 同步失败";
    return false;
  }
  float ms = 0.f;
  if (cudaEventElapsedTime(&ms, im.ev_start, im.ev_stop) == cudaSuccess) {
    im.last_infer_ms = static_cast<double>(ms);
  }
  return true;
}

void TrtEngine::Release() {
  if (impl_) impl_->Release();
}

namespace {

// ------------------------------------------------------------------ Runner
//
// 与另外三个后端同构：本文件只实现"跑一次"（IEngineRunner），
// 流水线（前处理 → 引擎 → 解码 → NMS → 坐标反变换）由共用的 RunnerDetector 负责。
//
// 早先这里是一个手写的 `TrtDetector : Detector`，结果它没能跟上 Detector 接口的演进
// （Detector 改成 protected RunBatch() + 共用异步骨架后，那份实现就编不过了）。
// 收到共用外壳上以后，这类接口漂移由 simple_detector.cpp 一处承担。
class TrtRunner : public IEngineRunner {
 public:
  TrtRunner(std::shared_ptr<TrtEngine> engine, EngineInputSpec spec)
      : engine_(std::move(engine)), spec_(std::move(spec)) {}

  const std::string& name() const override { return name_; }
  EngineInputSpec input_spec() const override { return spec_; }

  std::vector<std::vector<int64_t>> output_shapes() const override {
    return engine_ ? engine_->OutputShapes() : std::vector<std::vector<int64_t>>{};
  }
  std::vector<DataType> output_types() const override {
    // 通用外壳只接受 float32 输出。TRT 在 FP16 引擎下会给出 half，
    // 这里如实声明，外壳会明确报错而不是把 half 当 float 解出乱框。
    return engine_ ? engine_->OutputTypes() : std::vector<DataType>{};
  }

  void Run(const void* input, size_t input_bytes, int batch, const std::vector<float*>& outputs,
           const std::vector<size_t>& output_elems) override {
    if (!engine_) throw TritError("TensorRT 引擎未初始化");
    const std::vector<std::vector<int64_t>> shapes = engine_->OutputShapes();
    const std::vector<DataType> types = engine_->OutputTypes();
    if (shapes.size() != outputs.size()) {
      throw TritError("引擎输出数量与调用方缓冲不匹配（内部错误）");
    }

    // 外层给的是 float* 缓冲；TRT 需要的是"按引擎声明 dtype 大小"的缓冲区。
    // float32 输出可直接复用调用方缓冲；非 float32（FP16 引擎）先用内部缓冲再接。
    scratch_.resize(shapes.size());
    std::vector<void*> ptrs(shapes.size(), nullptr);
    std::vector<size_t> bytes(shapes.size(), 0);
    for (size_t k = 0; k < shapes.size(); ++k) {
      size_t elems = 1;
      for (int64_t d : shapes[k]) elems *= static_cast<size_t>(d > 0 ? d : 1);
      const size_t need = elems * dtype_size(types[k]);
      bytes[k] = need;
      if (types[k] == DataType::kF32 && output_elems[k] * sizeof(float) >= need) {
        ptrs[k] = outputs[k];  // 零拷贝：直接写进调用方缓冲
        scratch_[k].clear();
      } else {
        if (scratch_[k].size() < need) scratch_[k].resize(need);
        ptrs[k] = scratch_[k].data();
      }
    }

    std::string err;
    if (!engine_->Infer(input, input_bytes, batch, ptrs, bytes, &err)) {
      throw TritError("TensorRT 推理失败：" + err);
    }

    // FP16 引擎输出转成 float32（通用外壳的契约）
    for (size_t k = 0; k < shapes.size(); ++k) {
      if (ptrs[k] == outputs[k]) continue;
      const size_t n = std::min(output_elems[k], bytes[k] / dtype_size(types[k]));
      if (types[k] == DataType::kF16) {
        const uint16_t* src = reinterpret_cast<const uint16_t*>(scratch_[k].data());
        for (size_t i = 0; i < n; ++i) outputs[k][i] = half_to_float(src[i]);
      } else if (types[k] == DataType::kF32) {
        std::memcpy(outputs[k], scratch_[k].data(), n * sizeof(float));
      } else {
        throw TritError("TensorRT 输出 " + std::to_string(k) + " 的类型 " + to_string(types[k]) +
                        " 无法转成 float32（该引擎的输出类型不受支持）");
      }
    }
  }

  double last_infer_ms() const override { return engine_ ? engine_->LastInferMs() : 0.0; }

  std::string engine_info() const override {
    std::ostringstream oss;
    oss << "TensorRT    : DLA 层=" << (engine_ ? engine_->DlaLayerCount() : 0)
        << " GPU 层=" << (engine_ ? engine_->GpuLayerCount() : 0) << "\n";
    if (engine_ && engine_->GpuLayerCount() > 0) {
      oss << "提示        : 有层退回 GPU 执行（DLA↔GPU 切换的代价常超过该层本身的计算），"
             "用 --verbose 看逐层归属。\n";
    }
    return oss.str();
  }

 private:
  std::string name_ = "yolov8-trt";
  std::shared_ptr<TrtEngine> engine_;
  EngineInputSpec spec_;
  std::vector<std::vector<char>> scratch_;  // 仅非 float32 输出时使用
};

// ------------------------------------------------------------------ 工厂
//
// 与另外三个后端（RKNN / ORT / OpenVINO）同构：builder 是**本文件的私有实现**，
// 对外只暴露 CreateTrtDetector()。engine / runner / builder 的职责划分见文件头注释。
class TrtModelBuilder : public IModelBuilder {
 public:
  const std::string& name() const override { return name_; }

  void Build(const BuildConfig& cfg) override {
    engine_ = std::make_shared<TrtEngine>();
    std::string err;
    if (!cfg.engine_path.empty()) {
      if (!engine_->Load(cfg.engine_path, &err)) throw TritError(err);
    } else {
      if (!engine_->Build(cfg, &err)) throw TritError(err);
    }
  }

  /// `--dry-run` / `PlanAssembly` 要能在**没有引擎**的情况下回答"这台机器能不能用 TRT"，
  /// 所以这里只探 CUDA 设备 + TensorRT 版本 + DLA core 数，不建引擎。
  bool Available(std::string* reason) const override {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
      if (reason) *reason = "无法访问 CUDA 设备（无 GPU 或驱动未加载）";
      return false;
    }
    const TrtEngine::PlanInfo info = TrtEngine::Probe(BuildConfig{});
    if (reason) {
      std::ostringstream oss;
      oss << "TensorRT " << info.trt_version;
      if (!info.device_name.empty()) oss << " / " << info.device_name;
      oss << " / DLA core=" << info.nb_dla_cores;
      if (!info.dla_supported_build) oss << "（该版本已移除 DLA）";
      *reason = oss.str();
    }
    return true;
  }

  bool Run(const RunIO& io, std::string* err) override {
    if (!engine_) {
      if (err) *err = "引擎未构建";
      return false;
    }
    return engine_->Infer(io.input, io.input_bytes, io.batch, io.outputs, io.output_bytes, err);
  }

  std::vector<int64_t> input_shape() const override {
    return engine_ ? engine_->InputShape() : std::vector<int64_t>{};
  }
  std::vector<std::vector<int64_t>> output_shapes() const override {
    return engine_ ? engine_->OutputShapes() : std::vector<std::vector<int64_t>>{};
  }
  std::vector<DataType> output_types() const override {
    return engine_ ? engine_->OutputTypes() : std::vector<DataType>{};
  }
  std::string input_name() const override { return engine_ ? engine_->InputName() : std::string(); }
  std::vector<std::string> output_names() const override {
    return engine_ ? engine_->OutputNames() : std::vector<std::string>{};
  }
  bool serialize(const std::string& path, std::string* err) override {
    if (!engine_) {
      if (err) *err = "引擎未构建，无法序列化";
      return false;
    }
    return engine_->Serialize(path, err);
  }
  double last_infer_ms() const override { return engine_ ? engine_->LastInferMs() : 0.0; }
  void Release() override {
    if (engine_) engine_->Release();
  }

  /// Detector 需要复用**同一个** TrtEngine 实例（构建一次，别构建两次）。
  const std::shared_ptr<TrtEngine>& engine() const { return engine_; }

 private:
  std::string name_ = "yolov8-trt";
  std::shared_ptr<TrtEngine> engine_;
};

std::unique_ptr<IModelBuilder> MakeTrtBuilder() {
  return std::unique_ptr<IModelBuilder>(new TrtModelBuilder());
}

}  // namespace

TOD_RT_REGISTER_BUILDER(yolov8_trt, MakeTrtBuilder)
TOD_RT_REGISTER_BUILDER(trt, MakeTrtBuilder)

std::unique_ptr<Detector> CreateTrtDetector(const std::string& model_name,
                                            const DetectorOptions& opts, const ModelRecipe& recipe,
                                            const std::string& builder_name,
                                            const std::string& preproc_name,
                                            const std::string& postproc_name) {
  (void)recipe;
  (void)preproc_name;
  (void)postproc_name;

  DetectorOptions o = opts;
  if (o.device == Device::kAuto) o.device = Device::kGpu;

  if (builder_name.find("trt") == std::string::npos &&
      builder_name.find("yolov8") == std::string::npos) {
    log_warn("builder=" + builder_name + " 不是 TensorRT 实现，已改用 TensorRT 后端。");
  }

  BuildConfig bc = o.ToBuildConfig();
  bc.verbose = o.verbose;
  if (bc.engine_path.empty() && bc.onnx_path.empty()) {
    throw TritError(
        "既没有 engine_path 也没有 onnx_path。请在部署配置里给出 engine.path 或 engine.onnx。");
  }
  if (!bc.engine_path.empty()) {
    log_info("加载已序列化 engine：" + bc.engine_path);
  } else {
    log_info("从 ONNX 构建 engine：" + bc.onnx_path + "（device=" + to_string(bc.device) +
             " precision=" + to_string(bc.precision) + "）");
  }

  TrtModelBuilder builder;
  builder.Build(bc);
  std::shared_ptr<TrtEngine> engine = builder.engine();

  // 输入契约：TensorRT 的输入是 NCHW float32（本工具链固定这样导出）
  EngineInputSpec spec;
  const std::vector<int64_t> in_shape = engine->InputShape();
  if (in_shape.size() == 4 && in_shape[2] > 0 && in_shape[3] > 0) {
    spec.height = static_cast<int>(in_shape[2]);
    spec.width = static_cast<int>(in_shape[3]);
  } else {
    spec.height = o.preproc_opts.input_height;
    spec.width = o.preproc_opts.input_width;
    spec.dynamic_shape = true;  // 引擎输入含动态维
    spec.dynamic_min = o.shape_min;
    spec.dynamic_max = o.shape_max;
  }
  spec.output = PreprocOutput::kFloat32;
  spec.layout = TensorLayout::kNchw;

  PreprocessOptions po = resolve_preprocess_options(o, spec);
  PostprocessOptions pp = resolve_postprocess_options(o);
  // 解码用的输入尺寸来自引擎（不是配置里的猜测）；动态 shape 时用本次前处理的尺寸
  pp.decode.input_width = po.input_width;
  pp.decode.input_height = po.input_height;

  auto runner = std::unique_ptr<IEngineRunner>(new TrtRunner(engine, spec));
  auto pre = make_detect_preprocessor(po);
  auto post = make_nms_postprocessor(pp);

  log_info("TensorRT 装配：preproc=" + preproc_name + " postproc=" + postproc_name + " 输入=" +
           std::to_string(spec.width) + "×" + std::to_string(spec.height) + " " +
           to_string(po.output_dtype()) + " " + to_string(po.layout));

  return std::unique_ptr<Detector>(
      new RunnerDetector(model_name, o, std::move(runner), std::move(pre), std::move(post)));
}

// backend_available 统一在 backend_dispatch.cpp 实现（它知道所有后端的编译状态）

}  // namespace todrt

#endif  // TODRT_HAVE_TENSORRT
