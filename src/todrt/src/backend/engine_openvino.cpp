// engine_openvino.cpp —— Intel OpenVINO 后端
//
// 定位：与 ONNX Runtime 互补的通用 CPU 后路。两点现实价值：
//   1. **AMD x86 上直接用 CPU 设备**（OpenVINO 的 CPU 插件对 x86 有 AVX2/AVX512 深度优化，
//      且不需要任何 Intel 硬件）；
//   2. **Intel 机器上可切 GPU/NPU** —— 同一份配置改 `ov_device` 即可，不用改代码。
//
// 设计要点：
//   1. **编译产物有缓存**：OpenVINO 首次 compile_model 要几十秒（图优化+kernel 选择），
//      开 `ov_cache_dir` 后第二次启动接近瞬时。这也顺带解决了"换机器要重编"的问题。
//   2. **动态 shape 走 reshape**：输入带 -1 时按本次 H/W reshape 再推理。
//   3. **API 跨版本**：2.0 的 get_input_shape/get_shape 在 2024.x 改为
//      get_input_shape/get_output_shape，这里用 TODRT_OV_AT_LEAST 收敛。
#include "todrt/backend/openvino_factory.hpp"

#if TODRT_HAVE_OPENVINO

#include <openvino/openvino.hpp>

#include <algorithm>
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

// OpenVINO 2024.0 起 get_shape() → get_output_shape()/get_input_shape()
#if defined(OPENVINO_VERSION_MAJOR) && (OPENVINO_VERSION_MAJOR >= 2024)
#define TODRT_OV_AT_LEAST_2024 1
#else
#define TODRT_OV_AT_LEAST_2024 0
#endif

namespace todrt {
namespace {

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

/// OpenVINO 的设备名要用它自己的写法（CPU / GPU / NPU / AUTO），
/// 这里做一次宽松映射，让配置能写 "cpu"/"amd"/"x86" 这类词。
std::string parse_device(const std::string& s) {
  const std::string k = lower_compact(s);
  if (k.empty() || k == "auto") return "AUTO";
  if (k == "cpu" || k == "x86" || k == "amd" || k == "host") return "CPU";
  if (k == "gpu" || k == "igpu" || k == "intelgpu") return "GPU";
  if (k == "npu" || k == "intelnpu" || k == "vpu") return "NPU";
  if (k == "multidevice" || k == "multi") return "MULTI";
  // 允许直接透传 OpenVINO 的复合写法，如 "AUTO:GPU,CPU"
  return s;
}

std::string parse_hint(const std::string& s) {
  const std::string k = lower_compact(s);
  if (k == "throughput") return "THROUGHPUT";
  if (k == "cumulativethroughput") return "CUMULATIVE_THROUGHPUT";
  if (k == "none") return "NO_HINT";
  return "LATENCY";
}

/// OpenVINO 引擎。
class OpenVinoRunner : public IEngineRunner {
 public:
  const std::string& name() const override { return name_; }

  void Init(const BuildConfig& cfg, std::string* err);
  void Release() override {
    compiled_.reset();
    model_.reset();
  }

  EngineInputSpec input_spec() const override { return spec_; }
  std::vector<std::vector<int64_t>> output_shapes() const override { return out_shapes_; }
  std::vector<DataType> output_types() const override { return out_types_; }

  void Run(const void* input, size_t input_bytes, int batch, const std::vector<float*>& outputs,
           const std::vector<size_t>& output_elems) override;

  double last_infer_ms() const override { return last_ms_; }
  std::string engine_info() const override;

 private:
  /// 按实际输入尺寸 reshape（动态 shape 时才需要）。
  void MaybeReshape(int batch, int h, int w);

  std::string name_ = "openvino";
  ov::Core core_;
  std::shared_ptr<ov::Model> model_;
  ov::CompiledModel compiled_;
  ov::InferRequest request_;

  EngineInputSpec spec_;
  std::vector<std::vector<int64_t>> out_shapes_;
  std::vector<DataType> out_types_;
  std::string device_used_ = "CPU";
  bool dynamic_ = false;
  int64_t cur_h_ = 0;
  int64_t cur_w_ = 0;
  int64_t cur_b_ = 0;
  double last_ms_ = 0.0;
};

void OpenVinoRunner::Init(const BuildConfig& cfg, std::string* err) {
  auto fail = [&](const std::string& m) {
    if (err) *err = m;
    throw TritError(m);
  };
  if (cfg.onnx_path.empty()) {
    fail(
        "OpenVINO 需要 .onnx 模型（或 IR .xml）：请在部署配置里给出 engine.onnx。\n"
        "  OpenVINO 直接读 ONNX 并在首次 compile_model 时做图优化；开启 ov_cache_dir "
        "可把编译产物缓存下来，避免每次启动重编。");
  }

  const std::string dev = parse_device(cfg.ov_device);
  device_used_ = dev;

  // 可用设备列表：这是排查"GPU/NPU 插件没装"的第一手信息
  try {
    const std::vector<std::string> devices = core_.get_available_devices();
    std::ostringstream oss;
    for (size_t i = 0; i < devices.size(); ++i) oss << (i ? ", " : "") << devices[i];
    log_info("OpenVINO 可用设备：" + oss.str());
    if (!devices.empty() &&
        std::find(devices.begin(), devices.end(), dev) == devices.end() &&
        dev.find(',') == std::string::npos && dev.find(':') == std::string::npos) {
      log_warn("配置的 ov_device=" + dev + " 不在可用设备列表里，OpenVINO 可能回退到 CPU。");
    }
  } catch (const std::exception& e) {
    log_warn(std::string("查询 OpenVINO 设备失败：") + e.what());
  }

  // ---- 读模型 ----
  try {
    model_ = core_.read_model(cfg.onnx_path);
  } catch (const std::exception& e) {
    fail(std::string("OpenVINO 读取模型失败：") + e.what() + "\n  模型：" + cfg.onnx_path);
  }

  // ---- 输入形状 → 契约 ----
  const std::vector<ov::Output<ov::Node>> inputs = model_->inputs();
  if (inputs.empty()) fail("模型没有输入");
  const ov::Shape in_shape = inputs[0].get_shape();
  if (in_shape.size() == 4) {
    spec_.width = in_shape[3] > 0 ? static_cast<int>(in_shape[3]) : 640;
    spec_.height = in_shape[2] > 0 ? static_cast<int>(in_shape[2]) : 640;
  } else if (in_shape.size() == 3) {
    spec_.width = in_shape[2] > 0 ? static_cast<int>(in_shape[2]) : 640;
    spec_.height = in_shape[1] > 0 ? static_cast<int>(in_shape[1]) : 640;
  }
  const ov::element::Type et = inputs[0].get_element_type();
  const bool is_u8 = (et == ov::element::u8);
  spec_.output = is_u8 ? PreprocOutput::kUint8Raw : PreprocOutput::kFloat32;
  spec_.layout = TensorLayout::kNchw;
  spec_.dynamic_shape = model_->is_dynamic();
  dynamic_ = spec_.dynamic_shape;

  std::ostringstream is;
  for (size_t i = 0; i < in_shape.size(); ++i) is << (i ? "," : "") << in_shape[i];
  log_info("OpenVINO 输入：" + inputs[0].get_any_name() + " [" + is.str() + "] dtype=" +
           et.get_type_name() + (dynamic_ ? "（动态 shape）" : ""));
  if (is_u8) log_info("输入是 u8：按量化模型处理，前处理产出原始像素（不做归一化）。");

  // ---- 输出 ----
  const std::vector<ov::Output<ov::Node>> outputs = model_->outputs();
  if (outputs.empty()) fail("模型没有输出");
  for (size_t i = 0; i < outputs.size(); ++i) {
    const ov::Shape s = outputs[i].get_shape();
    std::vector<int64_t> shape;
    for (size_t d = 0; d < s.size(); ++d) {
      shape.push_back(s[d] > 0 ? static_cast<int64_t>(s[d]) : -1);
    }
    out_shapes_.push_back(shape);
    out_types_.push_back(DataType::kF32);
    std::ostringstream os;
    for (size_t d = 0; d < s.size(); ++d) os << (d ? "," : "") << s[d];
    log_info("OpenVINO 输出 " + std::to_string(i) + ": " + outputs[i].get_any_name() + " [" +
             os.str() + "]");
  }

  // ---- 编译（可缓存）----
  ov::AnyMap props;
  props.emplace(ov::hint::performance_mode.name(), parse_hint(cfg.ov_performance_hint));
  if (cfg.ov_num_streams > 0) props.emplace(ov::num_streams.name(), cfg.ov_num_streams);
  if (cfg.ov_num_threads > 0) {
    props.emplace(ov::inference_num_threads.name(), cfg.ov_num_threads);
  }
  if (!cfg.ov_cache_dir.empty() && cfg.has(BuilderFlag::kOvCacheCompiled)) {
    try {
      core_.set_property(ov::cache_dir.name(), cfg.ov_cache_dir);
      log_info("OpenVINO 编译缓存目录：" + cfg.ov_cache_dir);
    } catch (const std::exception& e) {
      log_warn(std::string("设置 OpenVINO 缓存目录失败（忽略）：") + e.what());
    }
  }

  const auto t0 = std::chrono::steady_clock::now();
  try {
    compiled_ = core_.compile_model(model_, dev, props);
  } catch (const std::exception& e) {
    fail(std::string("OpenVINO 编译模型失败（device=") + dev + "）：" + e.what() +
         "\n  排查：① 该设备插件是否随 OpenVINO 安装（GPU/NPU 是独立插件）；"
         "\n        ② ov_device 写法（CPU / GPU / NPU / AUTO）；"
         "\n        ③ 首次编译较慢，可开 ov_cache_dir 复用。");
  }
  const auto t1 = std::chrono::steady_clock::now();
  log_info("OpenVINO 编译完成：" + std::to_string(std::chrono::duration<double>(t1 - t0).count()) +
           " s（device=" + dev + " hint=" + parse_hint(cfg.ov_performance_hint) + "）");

  request_ = compiled_.create_infer_request();
}

void OpenVinoRunner::MaybeReshape(int batch, int h, int w) {
  if (!dynamic_) return;
  if (cur_b_ == batch && cur_h_ == h && cur_w_ == w) return;
  // NCHW；非 4 维输入不 reshape（交给用户保证）
  try {
    ov::Shape target;
    const ov::Shape in = model_->input().get_shape();
    if (in.size() == 4) {
      target = ov::Shape{static_cast<size_t>(batch), in[1], static_cast<size_t>(h),
                         static_cast<size_t>(w)};
    } else {
      return;
    }
    model_->reshape(ov::PartialShape(target));
    compiled_ = core_.compile_model(model_, device_used_);
    request_ = compiled_.create_infer_request();
    cur_b_ = batch;
    cur_h_ = h;
    cur_w_ = w;
    log_info("OpenVINO reshape → " + std::to_string(batch) + "×3×" + std::to_string(h) + "×" +
             std::to_string(w));
  } catch (const std::exception& e) {
    log_warn(std::string("OpenVINO reshape 失败（沿用原 shape）：") + e.what());
  }
}

void OpenVinoRunner::Run(const void* input, size_t input_bytes, int batch,
                         const std::vector<float*>& outputs,
                         const std::vector<size_t>& output_elems) {
  if (!request_) throw TritError("OpenVINO 推理请求未初始化");

  const bool u8 = (spec_.output == PreprocOutput::kUint8Raw);
  const size_t per_sample = input_bytes / static_cast<size_t>(std::max(1, batch));
  const size_t hw = per_sample / (3u * (u8 ? 1u : sizeof(float)));
  const int side = static_cast<int>(std::lround(std::sqrt(static_cast<double>(hw))));
  MaybeReshape(batch, side, side);

  // ---- 输入张量：直接用前处理缓冲（零拷贝）----
  const ov::Shape in_shape = request_.get_input_shape(0);
  ov::Tensor in_tensor;
  if (u8) {
    in_tensor = ov::Tensor(ov::element::u8, in_shape,
                           const_cast<uint8_t*>(static_cast<const uint8_t*>(input)));
  } else {
    in_tensor = ov::Tensor(ov::element::f32, in_shape,
                           const_cast<float*>(static_cast<const float*>(input)));
  }
  request_.set_input_tensor(0, in_tensor);

  const auto t0 = std::chrono::steady_clock::now();
  try {
    request_.infer();
  } catch (const std::exception& e) {
    throw TritError(std::string("OpenVINO 推理失败：") + e.what() +
                    "\n  常见原因：输入尺寸与模型不符、dtype 与模型不符（u8 vs f32）。");
  }
  const auto t1 = std::chrono::steady_clock::now();
  last_ms_ = std::chrono::duration<double, std::milli>(t1 - t0).count();

  // ---- 输出：拷进调用方给的 float 缓冲 ----
  for (size_t i = 0; i < outputs.size(); ++i) {
    if (!outputs[i]) continue;
    const ov::Tensor out = request_.get_output_tensor(i);
    const size_t n = out.get_size();
    const size_t want = std::min(output_elems[i], n);
    const ov::element::Type t = out.get_element_type();
    if (t == ov::element::f32) {
      std::memcpy(outputs[i], out.data<float>(), want * sizeof(float));
    } else if (t == ov::element::f16) {
      const ov::float16* src = out.data<ov::float16>();
      for (size_t k = 0; k < want; ++k) outputs[i][k] = static_cast<float>(src[k]);
    } else if (t == ov::element::i64) {
      const int64_t* src = out.data<int64_t>();
      for (size_t k = 0; k < want; ++k) outputs[i][k] = static_cast<float>(src[k]);
    } else if (t == ov::element::i32) {
      const int32_t* src = out.data<int32_t>();
      for (size_t k = 0; k < want; ++k) outputs[i][k] = static_cast<float>(src[k]);
    } else {
      throw TritError("OpenVINO 输出 " + std::to_string(i) + " 的类型 " + t.get_type_name() +
                      " 暂不支持（通用外壳只接受 f32/f16/i32/i64）。"
                      "若模型输出是量化整型，请在转换时让输出为 float。");
    }
  }
}

std::string OpenVinoRunner::engine_info() const {
  std::ostringstream oss;
  oss << "OpenVINO    : device=" << device_used_ << " 输入 dtype="
      << (spec_.output == PreprocOutput::kUint8Raw ? "u8" : "f32") << "\n";
#if TODRT_OV_AT_LEAST_2024
  oss << "版本        : " << OPENVINO_VERSION_MAJOR << "." << OPENVINO_VERSION_MINOR << "\n";
#endif
  if (device_used_ == "CPU") {
    oss << "提示        : 走 CPU 插件（x86 上已针对 AVX2/AVX512 优化；AMD 同样适用）。"
           "Intel 机器可设 ov_device=GPU 对比。\n";
  }
  return oss.str();
}

// ------------------------------------------------------------------ 装配

std::unique_ptr<Detector> BuildOvDetector(const std::string& model_name, DetectorOptions o,
                                          const std::string& preproc_name,
                                          const std::string& postproc_name) {
  BuildConfig bc = o.ToBuildConfig();
  bc.verbose = o.verbose;

  auto runner = std::unique_ptr<OpenVinoRunner>(new OpenVinoRunner());
  runner->Init(bc, nullptr);

  const EngineInputSpec spec = runner->input_spec();
  PreprocessOptions po = resolve_preprocess_options(o, spec);
  PostprocessOptions pp = resolve_postprocess_options(o);
  pp.decode.input_width = spec.width;
  pp.decode.input_height = spec.height;

  auto pre = make_detect_preprocessor(po);
  auto post = make_nms_postprocessor(pp);
  log_info("OpenVINO 装配：builder=openvino preproc=" + preproc_name + " postproc=" +
           postproc_name + " 输入=" + std::to_string(spec.width) + "×" +
           std::to_string(spec.height) + " " + to_string(po.output_dtype()));

  return std::unique_ptr<Detector>(
      new RunnerDetector(model_name, o, std::move(runner), std::move(pre), std::move(post)));
}

class OvModelBuilder : public IModelBuilder {
 public:
  const std::string& name() const override { return name_; }
  void Build(const BuildConfig& cfg) override {
    runner_ = std::make_shared<OpenVinoRunner>();
    runner_->Init(cfg, nullptr);
  }
  bool Available(std::string* reason) const override {
    try {
      ov::Core core;
      const std::vector<std::string> devices = core.get_available_devices();
      std::ostringstream oss;
      oss << "OpenVINO 已编译；可用设备：";
      for (size_t i = 0; i < devices.size(); ++i) oss << (i ? ", " : "") << devices[i];
      if (reason) *reason = oss.str();
    } catch (const std::exception& e) {
      if (reason) *reason = std::string("OpenVINO 初始化失败：") + e.what();
      return false;
    }
    return true;
  }
  bool Run(const RunIO& io, std::string* err) override {
    if (!runner_) {
      if (err) *err = "OpenVINO 未构建";
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
      *err =
          "OpenVINO 不需要序列化：直接读 .onnx，编译产物可用 ov_cache_dir 缓存"
          "（比自定义序列化格式更省心）。";
    }
    return false;
  }
  double last_infer_ms() const override { return runner_ ? runner_->last_infer_ms() : 0.0; }
  void Release() override { runner_.reset(); }

 private:
  std::string name_ = "openvino";
  std::shared_ptr<OpenVinoRunner> runner_;
};

std::unique_ptr<IModelBuilder> MakeOvBuilder() {
  return std::unique_ptr<IModelBuilder>(new OvModelBuilder());
}

}  // namespace

TOD_RT_REGISTER_BUILDER(openvino, MakeOvBuilder)
TOD_RT_REGISTER_BUILDER(ov, MakeOvBuilder)

std::unique_ptr<Detector> CreateOpenVinoDetector(const std::string& model_name,
                                                 const DetectorOptions& opts,
                                                 const ModelRecipe& recipe,
                                                 const std::string& builder_name,
                                                 const std::string& preproc_name,
                                                 const std::string& postproc_name) {
  (void)recipe;
  (void)builder_name;
  DetectorOptions o = opts;
  // 真正决定执行设备的是 ov_device；这里只保证 device 语义不误导日志
  if (o.device == Device::kAuto) o.device = Device::kCpu;
  return BuildOvDetector(model_name, o, preproc_name, postproc_name);
}

}  // namespace todrt

#endif  // TODRT_HAVE_OPENVINO
