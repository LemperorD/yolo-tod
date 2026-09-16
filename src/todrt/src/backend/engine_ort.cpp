// engine_ort.cpp —— ONNX Runtime 后端
//
// 定位：**通用 CPU 后路**。典型场景是 AMD x86 主机（无 NVIDIA GPU、无 NPU），
// 以及作为 NPU/GPU 的对照基线 —— 同一份部署配置换个 builder（`"builder": "ort"`）
// 就能跑，便于回答"加速器到底快多少、精度掉多少"。
//
// 设计要点：
//   1. **EP 只是配置**：cpu / cuda / tensorrt / openvino / xnnpack / acl 都走同一份
//      代码，靠 SessionOptionsAppendExecutionProvider_* 切换。新增 EP 不需要动架构。
//   2. **EP 不可用要能看见**：默认按"可用则用、不可用则退回 CPU"处理，但每次都打
//      warn（"你以为在用 GPU，其实在跑 CPU" 是最常见的部署错觉）。配
//      `ort_disable_cpu_fallback` 可以在 EP 不可用时直接失败。
//   3. **动态 shape 按输入尺寸重建输入张量**：ORT 的输入形状可以带 -1，Run 时按实际
//      H/W 构造 Ort::Value；输出形状每次 Run 后重新查询（RunnerDetector 会据此分配）。
#include "todrt/backend/ort_factory.hpp"

#if TODRT_HAVE_ORT

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "todrt/backend/factories.hpp"
#include "todrt/backend/simple_detector.hpp"
#include "todrt/factory.hpp"

namespace todrt {
namespace {

/// 环境是进程级的：Ort::Env 允许每进程一个，重复创建会报警。
Ort::Env& OrtEnv() {
  static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "todrt");
  return env;
}

std::string lower_compact(const std::string& s) {
  std::string k;
  k.reserve(s.size());
  for (char c : s) {
    if (c >= 'A' && c <= 'Z') {
      k.push_back(static_cast<char>(c - 'A' + 'a'));
    } else if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')) {
      k.push_back(c);
    }
  }
  return k;
}

GraphOptimizationLevel parse_opt_level(const std::string& s) {
  const std::string k = lower_compact(s);
  if (k == "disabled" || k == "none" || k == "off") return GraphOptimizationLevel::ORT_DISABLE_ALL;
  if (k == "basic") return GraphOptimizationLevel::ORT_ENABLE_BASIC;
  if (k == "extended") return GraphOptimizationLevel::ORT_ENABLE_EXTENDED;
  return GraphOptimizationLevel::ORT_ENABLE_ALL;
}

/// 把 C++ API 的例外转成我们的错误（不然只有一句 "Exception occurred"）。
std::string ort_error(const Ort::Exception& e) {
  std::ostringstream oss;
  oss << "ONNX Runtime 错误：" << e.what() << "（code=" << e.GetOrtErrorCode() << "）";
  return oss.str();
}

/// ONNX Runtime 引擎。
class OrtRunner : public IEngineRunner {
 public:
  ~OrtRunner() override = default;

  const std::string& name() const override { return name_; }

  void Init(const BuildConfig& cfg, std::string* err);
  void Release() override { session_.reset(); }

  EngineInputSpec input_spec() const override { return spec_; }
  std::vector<std::vector<int64_t>> output_shapes() const override;
  std::vector<DataType> output_types() const override { return out_types_extra_; }

  void Run(const void* input, size_t input_bytes, int batch, const std::vector<float*>& outputs,
           const std::vector<size_t>& output_elems) override;

  double last_infer_ms() const override { return last_ms_; }
  std::string engine_info() const override;

 private:
  std::string name_ = "ort";
  std::unique_ptr<Ort::Session> session_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::vector<std::vector<int64_t>> out_shapes_;
  std::vector<DataType> out_types_extra_;
  EngineInputSpec spec_;
  std::string provider_used_ = "CPU";
  bool input_is_uint8_ = false;
  double last_ms_ = 0.0;
};

void OrtRunner::Init(const BuildConfig& cfg, std::string* err) {
  auto fail = [&](const std::string& m) {
    if (err) *err = m;
    throw TritError(m);
  };
  if (cfg.onnx_path.empty()) {
    fail(
        "ONNX Runtime 需要 .onnx 模型：请在部署配置里给出 engine.onnx。\n"
        "  （ONNX Runtime 不做 .engine 序列化，直接读 ONNX；首次运行会做图优化与"
        "kernel 选择，可用 ov/trt 那种缓存机制另行处理。）");
  }

  Ort::SessionOptions so;
  try {
    so.SetGraphOptimizationLevel(parse_opt_level(cfg.ort_optimization));
    so.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    if (cfg.ort_intra_threads > 0) so.SetIntraOpNumThreads(cfg.ort_intra_threads);
    if (cfg.ort_inter_threads > 0) so.SetInterOpNumThreads(cfg.ort_inter_threads);
  } catch (const Ort::Exception& e) {
    fail(ort_error(e));
  }

  // ---- 执行提供者 ----
  // 顺序：用户指定 > auto（cuda → tensorrt → openvino → cpu）
  const std::string want = lower_compact(cfg.ort_provider);
  std::vector<std::string> order;
  if (want.empty() || want == "auto") {
    order = {"cuda", "tensorrt", "openvino", "cpu"};
  } else {
    order = {want};
  }

  bool provider_ok = false;
  for (const std::string& p : order) {
    try {
      if (p == "cuda") {
        OrtCUDAProviderOptions cuda{};
        cuda.device_id = 0;
        so.AppendExecutionProvider_CUDA(cuda);
        provider_used_ = "CUDA";
        provider_ok = true;
      } else if (p == "tensorrt") {
        // TRT EP 的参数结构各版本差异较大，用默认值 + 显式 cache 选项
        OrtTensorRTProviderOptions trt{};
        trt.device_id = 0;
        so.AppendExecutionProvider_TensorRT(trt);
        provider_used_ = "TensorRT";
        provider_ok = true;
      } else if (p == "openvino") {
        so.AppendExecutionProvider_OpenVINO("");
        provider_used_ = "OpenVINO";
        provider_ok = true;
      } else if (p == "xnnpack") {
        // XNNPACK 在 ARM/x86 上都有优化；ORT 1.12+ 提供
        so.AppendExecutionProvider("XNNPACK");
        provider_used_ = "XNNPACK";
        provider_ok = true;
      } else if (p == "acl") {
        // Arm Compute Library EP（仅 aarch64 构建的 ORT 带）
        so.AppendExecutionProvider("ACL");
        provider_used_ = "ACL";
        provider_ok = true;
      } else if (p == "cpu") {
        provider_used_ = "CPU";
        provider_ok = true;  // CPU 永远可用
      } else {
        fail("未知的 ONNX Runtime 执行提供者：" + cfg.ort_provider +
             "（可选 auto/cpu/cuda/tensorrt/openvino/xnnpack/acl）");
      }
      if (provider_ok) break;
    } catch (const Ort::Exception& e) {
      const std::string msg = ort_error(e);
      if (want != "auto" && cfg.has(BuilderFlag::kOrtDisableCpuFallback)) {
        fail("执行提供者 " + p + " 不可用，且已禁用 CPU 回退：\n  " + msg);
      }
      log_warn("执行提供者 " + p + " 不可用，尝试下一个：" + msg);
    } catch (const std::exception& e) {
      log_warn(std::string("执行提供者 ") + p + " 初始化失败：" + e.what());
    }
  }
  if (!provider_ok) {
    if (cfg.has(BuilderFlag::kOrtDisableCpuFallback)) {
      fail("没有可用的执行提供者，且已禁用 CPU 回退（ort_disable_cpu_fallback）。");
    }
    log_warn("所有指定 EP 都不可用，回退到 CPU。");
    provider_used_ = "CPU";
  }
  if (provider_used_ == "CPU" && want != "cpu" && want != "auto") {
    log_warn("请求的 EP 是 " + cfg.ort_provider + "，实际在用 CPU —— 延迟数字会明显偏高。");
  }

  // ---- 建会话 ----
  try {
#if defined(_WIN32)
    std::wstring wpath(cfg.onnx_path.begin(), cfg.onnx_path.end());
    session_.reset(new Ort::Session(OrtEnv(), wpath.c_str(), so));
#else
    session_.reset(new Ort::Session(OrtEnv(), cfg.onnx_path.c_str(), so));
#endif
  } catch (const Ort::Exception& e) {
    fail(std::string("创建 ONNX Runtime 会话失败：") + ort_error(e) +
         "\n  排查：① 模型路径是否正确；② opset 是否被当前 ORT 版本支持；"
         "\n        ③ 若要 CUDA EP，是否装了 onnxruntime-gpu 且驱动正常。");
  }

  Ort::AllocatorWithDefaultOptions alloc;

  // ---- 输入 ----
  const size_t n_in = session_->GetInputCount();
  if (n_in < 1) fail("ONNX 模型没有输入");
  if (n_in > 1) {
    log_warn("模型有 " + std::to_string(n_in) + " 个输入，只使用第 0 个（检测模型通常只有 1 个）");
  }
  for (size_t i = 0; i < n_in; ++i) {
#if ORT_API_VERSION >= 13
    input_names_.push_back(session_->GetInputNameAllocated(i, alloc).get());
#else
    input_names_.push_back(session_->GetInputName(i, alloc));
#endif
  }
  {
    const Ort::TypeInfo ti = session_->GetInputTypeInfo(0);
    const auto info = ti.GetTensorTypeAndShapeInfo();
    input_is_uint8_ = (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);
    const std::vector<int64_t> shape = info.GetShape();
    // 典型 NCHW: [N, 3, H, W]
    if (shape.size() == 4) {
      spec_.height = shape[2] > 0 ? static_cast<int>(shape[2]) : 640;
      spec_.width = shape[3] > 0 ? static_cast<int>(shape[3]) : 640;
      spec_.dynamic_shape = (shape[0] < 0 || shape[2] < 0 || shape[3] < 0);
    } else if (shape.size() == 3) {
      log_warn("输入是 3 维（可能是 [N,H,W] 灰度或已展平），按 640×640 处理，请核对。");
      spec_.height = shape[1] > 0 ? static_cast<int>(shape[1]) : 640;
      spec_.width = shape[2] > 0 ? static_cast<int>(shape[2]) : 640;
    }
    log_info("ORT 输入: " + input_names_[0] + " shape=[");
    for (size_t i = 0; i < shape.size(); ++i) {
      log_info(std::string(i ? "," : "") + std::to_string(shape[i]));
    }
    if (input_is_uint8_) {
      spec_.output = PreprocOutput::kUint8Raw;
      log_info("ORT 输入元素类型是 uint8：按量化模型处理，前处理产出原始像素。");
    } else {
      spec_.output = PreprocOutput::kFloat32;
    }
    spec_.layout = TensorLayout::kNchw;  // ONNX 检测模型默认 NCHW
  }

  // ---- 输出 ----
  const size_t n_out = session_->GetOutputCount();
  if (n_out < 1) fail("ONNX 模型没有输出");
  for (size_t i = 0; i < n_out; ++i) {
#if ORT_API_VERSION >= 13
    output_names_.push_back(session_->GetOutputNameAllocated(i, alloc).get());
#else
    output_names_.push_back(session_->GetOutputName(i, alloc));
#endif
    const Ort::TypeInfo ti = session_->GetOutputTypeInfo(i);
    const auto info = ti.GetTensorTypeAndShapeInfo();
    const ONNXTensorElementDataType t = info.GetElementType();
    // 通用外壳只接受 float32；非 float 输出在此声明出来，Run 时会做转换
    out_types_extra_.push_back(t == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ? DataType::kF32
                                                                        : DataType::kF32);
    const std::vector<int64_t> shape = info.GetShape();
    log_info("ORT 输出 " + std::to_string(i) + ": " + output_names_[i] + " 元素类型=" +
             std::to_string(static_cast<int>(t)));
    out_shapes_.push_back(shape);
  }
}

std::vector<std::vector<int64_t>> OrtRunner::output_shapes() const {
  if (!session_) return {};
  // 动态 shape：每次重新查询（Run 之后形状才确定）
  std::vector<std::vector<int64_t>> shapes;
  shapes.reserve(output_names_.size());
  for (size_t i = 0; i < output_names_.size(); ++i) {
    const Ort::TypeInfo ti = session_->GetOutputTypeInfo(i);
    const auto info = ti.GetTensorTypeAndShapeInfo();
    shapes.push_back(info.GetShape());
  }
  return shapes;
}

void OrtRunner::Run(const void* input, size_t input_bytes, int batch,
                    const std::vector<float*>& outputs, const std::vector<size_t>& output_elems) {
  if (!session_) throw TritError("ORT 会话未初始化");

  const auto t0 = std::chrono::steady_clock::now();

  // 输入张量：按实际尺寸构造（动态 shape 时形状来自本次前处理）
  const size_t elem = input_is_uint8_ ? 1u : sizeof(float);
  const size_t per_sample = input_bytes / static_cast<size_t>(std::max(1, batch));
  const size_t hw = per_sample / (3u * elem);
  const int64_t side = static_cast<int64_t>(std::lround(std::sqrt(static_cast<double>(hw))));
  const std::array<int64_t, 4> in_shape{batch, 3, side, side};

  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value in_tensor = input_is_uint8_
                             ? Ort::Value::CreateTensor<uint8_t>(
                                   mem, const_cast<uint8_t*>(static_cast<const uint8_t*>(input)),
                                   input_bytes, in_shape.data(), in_shape.size())
                             : Ort::Value::CreateTensor<float>(
                                   mem, const_cast<float*>(static_cast<const float*>(input)),
                                   input_bytes / sizeof(float), in_shape.data(), in_shape.size());

  std::vector<const char*> in_names;
  for (const auto& n : input_names_) in_names.push_back(n.c_str());
  std::vector<const char*> out_names;
  for (const auto& n : output_names_) out_names.push_back(n.c_str());

  std::vector<Ort::Value> out_tensors;
  try {
    out_tensors = session_->Run(Ort::RunOptions{nullptr}, in_names.data(), &in_tensor, 1,
                                out_names.data(), out_names.size());
  } catch (const Ort::Exception& e) {
    throw TritError(std::string("ORT 推理失败：") + ort_error(e) +
                    "\n  常见原因：输入尺寸不在模型允许范围、输入 dtype 与模型不符"
                    "（uint8 vs float）。");
  }

  for (size_t i = 0; i < out_tensors.size() && i < outputs.size(); ++i) {
    const auto info = out_tensors[i].GetTensorTypeAndShapeInfo();
    const size_t n = info.GetElementCount();
    const size_t want = std::min(output_elems[i], n);
    const ONNXTensorElementDataType t = info.GetElementType();
    if (t == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      std::memcpy(outputs[i], out_tensors[i].GetTensorData<float>(), want * sizeof(float));
    } else if (t == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
      const int64_t* src = out_tensors[i].GetTensorData<int64_t>();
      for (size_t k = 0; k < want; ++k) outputs[i][k] = static_cast<float>(src[k]);
    } else if (t == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
      const int32_t* src = out_tensors[i].GetTensorData<int32_t>();
      for (size_t k = 0; k < want; ++k) outputs[i][k] = static_cast<float>(src[k]);
    } else {
      throw TritError("ORT 输出 " + std::to_string(i) + " 的元素类型 " +
                      std::to_string(static_cast<int>(t)) +
                      " 暂不支持（通用外壳只接受 float32/int32/int64）。"
                      "纯量化输出请改用 RKNN 后端，或让模型输出 float。");
    }
  }

  const auto t1 = std::chrono::steady_clock::now();
  last_ms_ = std::chrono::duration<double, std::milli>(t1 - t0).count();
}

std::string OrtRunner::engine_info() const {
  std::ostringstream oss;
  oss << "ORT         : provider=" << provider_used_ << " 输入 dtype="
      << (input_is_uint8_ ? "uint8" : "float32") << "\n";
  if (provider_used_ == "CPU") {
    oss << "提示        : 当前是 CPU 执行。AMD x86 上这是预期路径；"
           "若机器有 NVIDIA GPU，可装 onnxruntime-gpu 并把 ort_provider 设为 cuda/auto。\n";
  }
  return oss.str();
}

// ------------------------------------------------------------------ 装配

std::unique_ptr<Detector> BuildOrtDetector(const std::string& model_name, DetectorOptions o,
                                           const std::string& preproc_name,
                                           const std::string& postproc_name) {
  BuildConfig bc = o.ToBuildConfig();
  bc.verbose = o.verbose;

  auto runner = std::unique_ptr<OrtRunner>(new OrtRunner());
  runner->Init(bc, nullptr);

  const EngineInputSpec spec = runner->input_spec();
  PreprocessOptions po = resolve_preprocess_options(o, spec);
  PostprocessOptions pp = resolve_postprocess_options(o);
  pp.decode.input_width = spec.width;
  pp.decode.input_height = spec.height;

  auto pre = make_detect_preprocessor(po);
  auto post = make_nms_postprocessor(pp);
  log_info("ORT 装配：builder=ort preproc=" + preproc_name + " postproc=" + postproc_name +
           " 输入=" + std::to_string(spec.width) + "×" + std::to_string(spec.height) + " " +
           to_string(po.output_dtype()));

  return std::unique_ptr<Detector>(
      new RunnerDetector(model_name, o, std::move(runner), std::move(pre), std::move(post)));
}

/// IModelBuilder 适配（供工厂清单 / dry-run 报告可用性）。
class OrtModelBuilder : public IModelBuilder {
 public:
  const std::string& name() const override { return name_; }
  void Build(const BuildConfig& cfg) override {
    runner_ = std::make_shared<OrtRunner>();
    runner_->Init(cfg, nullptr);
  }
  bool Available(std::string* reason) const override {
    if (reason) {
      *reason = std::string("onnxruntime ") + OrtGetApiBase()->GetVersionString() +
                "（EP 支持取决于所用构建：CPU 一定可用）";
    }
    return true;
  }
  bool Run(const RunIO& io, std::string* err) override {
    if (!runner_) {
      if (err) *err = "ORT 会话未构建";
      return false;
    }
    try {
      std::vector<float*> ptrs(io.outputs.size(), nullptr);
      for (size_t i = 0; i < io.outputs.size(); ++i) ptrs[i] = static_cast<float*>(io.outputs[i]);
      runner_->Run(io.input, io.input_bytes, io.batch, ptrs, io.output_bytes);
      return true;
    } catch (const std::exception& e) {
      if (err) *err = e.what();
      return false;
    }
  }
  std::vector<int64_t> input_shape() const override {
    if (!runner_) return {};
    const EngineInputSpec s = runner_->input_spec();
    return {1, 3, s.height, s.width};
  }
  std::vector<std::vector<int64_t>> output_shapes() const override {
    return runner_ ? runner_->output_shapes() : std::vector<std::vector<int64_t>>{};
  }
  std::vector<DataType> output_types() const override {
    return runner_ ? runner_->output_types() : std::vector<DataType>{};
  }
  std::string input_name() const override { return "images"; }
  std::vector<std::string> output_names() const override { return {"output0"}; }
  bool serialize(const std::string&, std::string* err) override {
    if (err) {
      *err = "ONNX Runtime 不需要序列化：直接读 .onnx（图优化在会话创建时完成）。";
    }
    return false;
  }
  double last_infer_ms() const override { return runner_ ? runner_->last_infer_ms() : 0.0; }
  void Release() override { runner_.reset(); }

 private:
  std::string name_ = "ort";
  std::shared_ptr<OrtRunner> runner_;
};

std::unique_ptr<IModelBuilder> MakeOrtBuilder() {
  return std::unique_ptr<IModelBuilder>(new OrtModelBuilder());
}

}  // namespace

TOD_RT_REGISTER_BUILDER(ort, MakeOrtBuilder)
TOD_RT_REGISTER_BUILDER(onnxruntime, MakeOrtBuilder)

std::unique_ptr<Detector> CreateOrtDetector(const std::string& model_name,
                                            const DetectorOptions& opts, const ModelRecipe& recipe,
                                            const std::string& builder_name,
                                            const std::string& preproc_name,
                                            const std::string& postproc_name) {
  (void)recipe;
  (void)builder_name;
  DetectorOptions o = opts;
  if (o.device == Device::kCpu || o.device == Device::kAuto) {
    o.device = Device::kCpu;
  } else {
    // 允许 gpu（CUDA EP）等：真正决定路径的是 ort_provider
    log_info(std::string("ORT builder + device=") + to_string(o.device) +
             "：实际执行路径由 ort_provider 决定。");
  }
  return BuildOrtDetector(model_name, o, preproc_name, postproc_name);
}

}  // namespace todrt

#endif  // TODRT_HAVE_ORT
