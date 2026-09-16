// engine_rknn.cpp —— Rockchip RKNN 后端（RK3588 / RK3588S 的 NPU）
//
// 目标平台：**Linux + librknnrt.so**（rknn_api.h 由 rknpu2 提供）。
// RK3588 有 3 个 NPU core，每个约 2 TOPS（INT8），合计 6 TOPS。
//
// 与 TensorRT 的关键差异（决定了本文件的结构）：
//   1. **模型格式是 .rknn**，由 rknn-toolkit2 在 x86 主机上离线转换（板上不做转换）；
//      C API 没有"从 ONNX 构建引擎"的等价物，所以 Init() 只做加载 + 校验。
//   2. **量化模型的输入是 uint8（通常 NHWC）**，归一化已烧进模型。前处理必须走
//      uint8 路径（preprocess.cpp 的 kUint8Raw），否则双重归一化 → 框全错。
//   3. **输出走 want_float=1 让 SDK 反量化**（内部按 scale/zp 还原），对外统一 float32 ——
//      与 RunnerDetector 的契约一致。
//   4. **多核要显式 dup context**：rknn_dup_context 复制出独立上下文，各占一个 core。
//      单 context 只会用一个 core —— "买了 6 TOPS 只跑出 2 TOPS" 的常见原因。
//   5. **NPU 有原生计时**：rknn_query(RKNN_QUERY_PERF_RUN) 给的是 NPU 侧耗时，
//      比 wall-clock 干净。
//
// rknn_api.h 在各版本间有漂移（尤其 init 的 extend 结构与 mem API），
// 所有版本相关调用都收敛在本文件的 rknn 命名空间内。
#include "todrt/backend/rknn_factory.hpp"

#if TODRT_HAVE_RKNN

#include <rknn_api.h>

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
#include "todrt/factory.hpp"

namespace todrt {
namespace {

/// 把 RKNN 的错误码转成可读信息。
std::string rknn_err(int ret) {
  std::ostringstream oss;
  oss << "RKNN 错误码 " << ret;
  switch (ret) {
    case RKNN_ERR_FAIL: oss << "（执行失败）"; break;
    case RKNN_ERR_TIMEOUT: oss << "（超时）"; break;
    case RKNN_ERR_DEVICE_UNAVAILABLE:
      oss << "（NPU 设备不可用：检查 /dev/rknpu* 与权限，以及 librknnrt 与内核驱动是否匹配）";
      break;
    case RKNN_ERR_MALLOC_FAIL: oss << "（内存分配失败）"; break;
    case RKNN_ERR_PARAM_INVALID: oss << "（参数无效）"; break;
    case RKNN_ERR_MODEL_INVALID:
      oss << "（模型无效：.rknn 是否由匹配版本的 rknn-toolkit2 转换？）";
      break;
    case RKNN_ERR_CTX_INVALID: oss << "（上下文无效）"; break;
    case RKNN_ERR_INPUT_INVALID: oss << "（输入无效：尺寸/格式是否与模型一致？）"; break;
    case RKNN_ERR_OUTPUT_INVALID: oss << "（输出无效）"; break;
    default: break;
  }
  return oss.str();
}

void check_rknn(int ret, const std::string& what) {
  if (ret != RKNN_SUCC) throw TritError(what + " 失败：" + rknn_err(ret));
}

std::string rknn_shape_to_string(const rknn_tensor_attr& a) {
  std::ostringstream oss;
  oss << "[";
  for (uint32_t i = 0; i < a.n_dims; ++i) {
    if (i) oss << ",";
    // RKNN 的 dims 是反序存放的（idx 0 = 最后一维），按 n_dims 反着打印更直观
    oss << a.dims[a.n_dims - 1 - i];
  }
  oss << "]";
  return oss.str();
}

const char* rknn_type_name(rknn_tensor_type t) {
  switch (t) {
    case RKNN_TENSOR_FLOAT32: return "float32";
    case RKNN_TENSOR_FLOAT16: return "float16";
    case RKNN_TENSOR_INT8: return "int8";
    case RKNN_TENSOR_UINT8: return "uint8";
    case RKNN_TENSOR_INT16: return "int16";
    default: return "other";
  }
}

const char* rknn_format_name(rknn_tensor_format f) {
  switch (f) {
    case RKNN_TENSOR_NCHW: return "nchw";
    case RKNN_TENSOR_NHWC: return "nhwc";
    case RKNN_TENSOR_NC1HWC2: return "nc1hwc2";
    default: return "other";
  }
}

/// 一个 RKNN 上下文 = 一个 NPU core 上的一份模型实例。
struct RknnContext {
  rknn_context ctx = 0;
  int core = 0;
  std::vector<rknn_tensor_attr> in_attrs;
  std::vector<rknn_tensor_attr> out_attrs;
  /// 主机侧输入缓冲（uint8 是常态；float 模型也走这里）
  std::vector<uint8_t> host_input;
  /// 主机侧输出缓冲（SDK want_float=1 后的 float32 结果）
  std::vector<std::vector<float>> host_outputs;

  ~RknnContext() { release(); }
  void release() {
    if (ctx) {
      rknn_destroy(ctx);
      ctx = 0;
    }
  }
  RknnContext(const RknnContext&) = delete;
  RknnContext& operator=(const RknnContext&) = delete;
  RknnContext() = default;
};

/// RKNN 引擎：管理 1..N 个 context（多核轮转），对外统一 float32 输出。
class RknnRunner : public IEngineRunner {
 public:
  ~RknnRunner() override { Release(); }

  const std::string& name() const override { return name_; }

  void Init(const BuildConfig& cfg, std::string* err);
  void Release() override;

  EngineInputSpec input_spec() const override { return spec_; }
  std::vector<std::vector<int64_t>> output_shapes() const override { return out_shapes_; }
  std::vector<DataType> output_types() const override { return out_types_; }

  void Run(const void* input, size_t input_bytes, int batch, const std::vector<float*>& outputs,
           const std::vector<size_t>& output_elems) override;

  double last_infer_ms() const override { return last_ms_; }
  std::string engine_info() const override;

 private:
  std::string name_ = "rknn";
  EngineInputSpec spec_;
  std::vector<std::vector<int64_t>> out_shapes_;
  std::vector<DataType> out_types_;
  std::vector<std::unique_ptr<RknnContext>> contexts_;
  int start_core_ = 0;
  double last_ms_ = 0.0;
  size_t input_elems_ = 0;
  DataType input_dtype_ = DataType::kU8;
  size_t round_robin_ = 0;
};

void RknnRunner::Init(const BuildConfig& cfg, std::string* err) {
  auto fail = [&](const std::string& m) {
    if (err) *err = m;
    throw TritError(m);
  };

  start_core_ = std::max(0, cfg.accelerator_core);

  // ---- 找模型文件：.rknn 优先 ----
  std::string model_path = cfg.rknn_model_path;
  if (model_path.empty()) model_path = cfg.engine_path;
  if (model_path.empty()) {
    fail(
        "RKNN 需要 .rknn 模型文件：请在部署配置里给出 engine.path（.rknn）或 rknn_model。\n"
        "  .rknn 由 rknn-toolkit2 在 **x86 主机上离线转换**得到（板上不做转换）：\n"
        "    pip install rknn-toolkit2\n"
        "    python tools/convert_rknn.py --onnx exports/SPAE-YOLOv8n.onnx \\\n"
        "        --target rk3588 --std 255 255 255 --out exports/spae_yolov8n.rk3588.rknn\n"
        "  --std 255 表示\"归一化烧进模型\"，因此宿主机前处理必须给 uint8 原始像素（默认如此）。");
  }
  std::ifstream in(model_path, std::ios::binary);
  if (!in) fail("无法打开 .rknn 模型：" + model_path);
  in.seekg(0, std::ios::end);
  const std::streamsize size = in.tellg();
  in.seekg(0, std::ios::beg);
  if (size <= 0) fail(".rknn 文件为空：" + model_path);
  std::vector<char> blob(static_cast<size_t>(size));
  if (!in.read(blob.data(), size)) fail("读取 .rknn 模型失败：" + model_path);
  log_info("加载 RKNN 模型：" + model_path + "（" +
           std::to_string(static_cast<double>(size) / (1024.0 * 1024.0)) + " MB）");

  // ---- SDK 版本：排查"toolkit 与 runtime 版本不匹配"的第一手信息 ----
  {
    rknn_sdk_version sdk{};
    if (rknn_query(0, RKNN_QUERY_SDK_VERSION, &sdk) == RKNN_SUCC) {
      log_info(std::string("RKNN SDK: api=") + sdk.api_version + " driver=" + sdk.drv_version);
    }
  }

  // ---- 建第一个 context ----
  auto make_ctx = [&](int core_index) -> std::unique_ptr<RknnContext> {
    rknn_init_extend ext{};
    rknn_context c = 0;
    // core_mask：0 = 让 SDK 自行分配；否则用 (1 << core) 指定（RKNN 的约定）
    ext.core_mask = static_cast<int32_t>(1U << core_index);
    int ret = rknn_init(&c, blob.data(), static_cast<uint32_t>(blob.size()), 0, &ext);
    if (ret != RKNN_SUCC) {
      std::ostringstream oss;
      oss << "rknn_init(core=" << core_index << ") 失败：" << rknn_err(ret);
      if (ret == RKNN_ERR_DEVICE_UNAVAILABLE) {
        oss << "\n  排查：① ls -l /dev/rknpu*；② 用户是否在 video/render 组；"
               "\n        ③ librknnrt.so 与内核驱动版本是否匹配（dmesg | grep -i rknpu）。";
      }
      throw TritError(oss.str());
    }
    auto p = std::unique_ptr<RknnContext>(new RknnContext());
    p->ctx = c;
    p->core = core_index;
    return p;
  };

  auto first = make_ctx(start_core_);
  rknn_context first_handle = first->ctx;
  contexts_.push_back(std::move(first));

  // ---- 多核：dup 出独立 context ----
  const int want_cores = std::max(1, cfg.rknn_core_num);
  for (int i = 1; i < want_cores; ++i) {
    rknn_context dup = 0;
    const int ret = rknn_dup_context(&first_handle, &dup);
    if (ret != RKNN_SUCC) {
      log_warn("rknn_dup_context 失败（第 " + std::to_string(i + 1) + " 个 core）：" +
               rknn_err(ret) + " —— 退化为 " + std::to_string(contexts_.size()) +
               " 个 core（RK3588 有 3 个，请确认 rknn_core_num 与设备匹配）。");
      break;
    }
    auto p = std::unique_ptr<RknnContext>(new RknnContext());
    p->ctx = dup;
    p->core = start_core_ + i;
    contexts_.push_back(std::move(p));
  }

  // ---- 查询输入属性 → 确定前处理契约 ----
  RknnContext& c0 = *contexts_.front();
  rknn_input_output_num io{};
  check_rknn(rknn_query(c0.ctx, RKNN_QUERY_IN_OUT_NUM, &io), "查询输入输出数量");
  if (io.n_input < 1) fail("RKNN 模型没有输入张量");
  if (io.n_output < 1) fail("RKNN 模型没有输出张量");

  c0.in_attrs.resize(io.n_input);
  c0.in_attrs[0].index = 0;
  check_rknn(rknn_query(c0.ctx, RKNN_QUERY_INPUT_ATTR, &c0.in_attrs[0]), "查询输入属性");
  const rknn_tensor_attr& ia = c0.in_attrs[0];

  log_info(std::string("RKNN 输入: ") + rknn_shape_to_string(ia) + " " +
           rknn_type_name(ia.type) + " " + rknn_format_name(ia.fmt) +
           " scale=" + std::to_string(ia.scale) + " zp=" + std::to_string(ia.zp) +
           " n_elems=" + std::to_string(ia.n_elems));

  input_elems_ = ia.n_elems;
  switch (ia.type) {
    case RKNN_TENSOR_UINT8: input_dtype_ = DataType::kU8; break;
    case RKNN_TENSOR_INT8: input_dtype_ = DataType::kI8; break;
    case RKNN_TENSOR_FLOAT16: input_dtype_ = DataType::kF16; break;
    default: input_dtype_ = DataType::kF32; break;
  }

  // 契约：把 RKNN 的要求翻译成"前处理该怎么产出数据"
  spec_.output =
      (input_dtype_ == DataType::kU8) ? PreprocOutput::kUint8Raw : PreprocOutput::kFloat32;
  spec_.layout = (ia.fmt == RKNN_TENSOR_NHWC) ? TensorLayout::kNhwc : TensorLayout::kNchw;
  // RKNN 的 dims 反序：NHWC 时 dims[2]=W、dims[1]=H；NCHW 时 dims[1]=H、dims[0]=W
  if (ia.n_dims >= 3) {
    spec_.width = static_cast<int>(ia.dims[2]);
    spec_.height = static_cast<int>(ia.dims[1]);
  }
  spec_.dynamic_shape = false;  // .rknn 的输入尺寸编译期固定
  log_info("RKNN 输入契约：dtype=" + to_string(spec_.output_dtype()) +
           " layout=" + to_string(spec_.layout) + " " + std::to_string(spec_.width) + "×" +
           std::to_string(spec_.height));

  if (input_dtype_ != DataType::kU8) {
    log_warn(std::string("RKNN 输入不是 uint8 而是 ") + rknn_type_name(ia.type) +
             "。若模型是「归一化烧进模型」的量化模型，这里应该是 uint8 —— "
             "请确认 rknn-toolkit2 转换时的 mean/std 设置。");
  }

  // ---- 查询输出属性 ----
  c0.out_attrs.resize(io.n_output);
  out_shapes_.clear();
  out_types_.clear();
  for (uint32_t i = 0; i < io.n_output; ++i) {
    c0.out_attrs[i].index = i;
    check_rknn(rknn_query(c0.ctx, RKNN_QUERY_OUTPUT_ATTR, &c0.out_attrs[i]),
               "查询输出属性 " + std::to_string(i));
    const rknn_tensor_attr& oa = c0.out_attrs[i];
    std::vector<int64_t> shape;
    for (uint32_t d = 0; d < oa.n_dims; ++d) {
      shape.push_back(static_cast<int64_t>(oa.dims[oa.n_dims - 1 - d]));
    }
    out_shapes_.push_back(shape);
    out_types_.push_back(DataType::kF32);  // Run 内按 want_float=1 取 float32
    log_info("RKNN 输出 " + std::to_string(i) + ": " + rknn_shape_to_string(oa) + " " +
             rknn_type_name(oa.type) + " n_elems=" + std::to_string(oa.n_elems));
  }

  // ---- 每个 context 各自准备缓冲区（互不共享，多线程安全）----
  for (auto& cp : contexts_) {
    RknnContext& cc = *cp;
    cc.in_attrs = c0.in_attrs;
    cc.out_attrs = c0.out_attrs;
    cc.host_input.assign(static_cast<size_t>(input_elems_) * dtype_size(input_dtype_), 0);
    cc.host_outputs.resize(cc.out_attrs.size());
    for (size_t i = 0; i < cc.out_attrs.size(); ++i) {
      cc.host_outputs[i].assign(cc.out_attrs[i].n_elems, 0.f);
    }
  }

  log_info("RKNN 就绪：" + std::to_string(contexts_.size()) + " 个 NPU core（起始 core=" +
           std::to_string(start_core_) + "），输入 " + std::to_string(input_elems_) + " 元素");
}

void RknnRunner::Run(const void* input, size_t input_bytes, int batch,
                     const std::vector<float*>& outputs, const std::vector<size_t>& output_elems) {
  if (contexts_.empty()) throw TritError("RKNN 引擎未初始化");
  if (batch != 1) {
    throw TritError(
        "RKNN 引擎的 batch 在转换时固定为 1（当前请求 batch=" + std::to_string(batch) +
        "）。多 batch 请在 rknn-toolkit2 转换时指定 batch_size，或逐帧调用。");
  }
  RknnContext& c = *contexts_[round_robin_ % contexts_.size()];
  round_robin_ = (round_robin_ + 1) % contexts_.size();

  const size_t need = static_cast<size_t>(input_elems_) * dtype_size(input_dtype_);
  if (input_bytes > need) {
    throw TritError("输入字节数 " + std::to_string(input_bytes) + " 超过引擎输入尺寸 " +
                    std::to_string(need) +
                    "（检查前处理尺寸/类型是否与 .rknn 一致：RKNN 输入是编译期固定的）");
  }
  if (input_bytes > c.host_input.size()) {
    throw TritError("输入缓冲不足（内部错误）");
  }
  std::memcpy(c.host_input.data(), input, input_bytes);

  rknn_input inputs[1];
  std::memset(inputs, 0, sizeof(inputs));
  inputs[0].index = 0;
  inputs[0].type = c.in_attrs[0].type;
  inputs[0].fmt = c.in_attrs[0].fmt;
  inputs[0].size = static_cast<uint32_t>(input_bytes);
  inputs[0].buf = c.host_input.data();
  inputs[0].pass_through = 0;  // 让 SDK 处理类型/布局转换，最稳
  check_rknn(rknn_inputs_set(c.ctx, 1, inputs), "设置输入");

  const auto t0 = std::chrono::steady_clock::now();
  check_rknn(rknn_run(c.ctx, nullptr), "NPU 推理");
  const auto t1 = std::chrono::steady_clock::now();

  rknn_perf_run perf{};
  if (rknn_query(c.ctx, RKNN_QUERY_PERF_RUN, &perf) == RKNN_SUCC) {
    last_ms_ = static_cast<double>(perf.run_duration) / 1000.0;  // 微秒 → 毫秒
  } else {
    last_ms_ = std::chrono::duration<double, std::milli>(t1 - t0).count();
  }

  // want_float=1：让 SDK 按 scale/zp 反量化为 float32（省掉手写反量化，也少一类错）
  std::vector<rknn_output> outs(outputs.size());
  std::memset(outs.data(), 0, sizeof(rknn_output) * outs.size());
  for (size_t i = 0; i < outs.size(); ++i) {
    outs[i].index = static_cast<uint32_t>(i);
    outs[i].want_float = 1;
    outs[i].is_prealloc = 0;
  }
  check_rknn(rknn_outputs_get(c.ctx, static_cast<uint32_t>(outs.size()), outs.data(), nullptr),
             "取输出");

  std::string err_msg;
  for (size_t i = 0; i < outs.size(); ++i) {
    if (!outputs[i]) continue;
    if (!outs[i].buf) {
      err_msg = "RKNN 输出 " + std::to_string(i) + " 为空";
      break;
    }
    const size_t available = static_cast<size_t>(c.out_attrs[i].n_elems);
    const size_t want = std::min(output_elems[i], available);
    std::memcpy(outputs[i], outs[i].buf, want * sizeof(float));
  }
  rknn_outputs_release(c.ctx, static_cast<uint32_t>(outs.size()), outs.data());
  if (!err_msg.empty()) throw TritError(err_msg);
}

std::string RknnRunner::engine_info() const {
  std::ostringstream oss;
  oss << "RKNN        : core 数=" << contexts_.size() << " 起始 core=" << start_core_
      << " 输入 dtype=" << to_string(input_dtype_) << "\n";
  if (contexts_.size() > 1) {
    oss << "多核        : 推理请求按 core 轮转分配（每路视频天然落到不同 core）\n";
  } else {
    oss << "提示        : 只用了 1 个 NPU core。RK3588 有 3 个，多路视频可设 "
           "rknn_core_num=3 提升吞吐。\n";
  }
  return oss.str();
}

void RknnRunner::Release() {
  for (auto& c : contexts_) {
    if (c) c->release();
  }
  contexts_.clear();
}

// ------------------------------------------------------------------ 装配

std::unique_ptr<Detector> BuildRknnDetector(const std::string& model_name, DetectorOptions o,
                                            const std::string& preproc_name,
                                            const std::string& postproc_name) {
  BuildConfig bc = o.ToBuildConfig();
  bc.verbose = o.verbose;

  auto runner = std::unique_ptr<RknnRunner>(new RknnRunner());
  runner->Init(bc, nullptr);

  // 前处理必须按**引擎声明**的契约来（RKNN 上就是 uint8 NHWC）
  const EngineInputSpec spec = runner->input_spec();
  PreprocessOptions po = resolve_preprocess_options(o, spec);
  PostprocessOptions pp = resolve_postprocess_options(o);
  // 解码用的输入尺寸来自引擎，而不是配置里的猜测
  pp.decode.input_width = spec.width;
  pp.decode.input_height = spec.height;

  auto pre = make_detect_preprocessor(po);
  auto post = make_nms_postprocessor(pp);
  log_info("RKNN 装配：preproc=" + preproc_name + " postproc=" + postproc_name + " 输入=" +
           std::to_string(spec.width) + "×" + std::to_string(spec.height) + " " +
           to_string(po.output_dtype()) + " " + to_string(po.layout));

  return std::unique_ptr<Detector>(
      new RunnerDetector(model_name, o, std::move(runner), std::move(pre), std::move(post)));
}

/// IModelBuilder 适配：让工厂清单 / dry-run 也能报告 RKNN 的可用性。
class RknnModelBuilder : public IModelBuilder {
 public:
  const std::string& name() const override { return name_; }

  void Build(const BuildConfig& cfg) override {
    runner_ = std::make_shared<RknnRunner>();
    runner_->Init(cfg, nullptr);
  }
  bool Available(std::string* reason) const override {
    std::ifstream dev0("/dev/rknpu0");
    std::ifstream dev("/dev/rknpu");
    const bool has_dev = dev0.good() || dev.good();
    if (reason) {
      *reason = std::string("librknnrt 已编译；/dev/rknpu* ") +
                (has_dev ? "存在" : "未发现（非 Rockchip 板子，或当前用户无权限）");
    }
    return true;
  }
  bool Run(const RunIO& io, std::string* err) override {
    if (!runner_) {
      if (err) *err = "RKNN 引擎未构建";
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
    if (s.layout == TensorLayout::kNhwc) {
      return {1, s.height, s.width, 3};
    }
    return {1, 3, s.height, s.width};
  }
  std::vector<std::vector<int64_t>> output_shapes() const override {
    return runner_ ? runner_->output_shapes() : std::vector<std::vector<int64_t>>{};
  }
  std::vector<DataType> output_types() const override {
    return runner_ ? runner_->output_types() : std::vector<DataType>{};
  }
  std::string input_name() const override { return "input"; }
  std::vector<std::string> output_names() const override {
    std::vector<std::string> names;
    if (runner_) {
      for (size_t i = 0; i < runner_->output_shapes().size(); ++i) {
        names.push_back("output" + std::to_string(i));
      }
    }
    return names;
  }
  bool serialize(const std::string&, std::string* err) override {
    if (err) {
      *err =
          "RKNN 不需要序列化：.rknn 本身就是最终产物，由 rknn-toolkit2 在 x86 上转换得到，"
          "可跨同型号（同目标平台参数）设备复制。";
    }
    return false;
  }
  double last_infer_ms() const override { return runner_ ? runner_->last_infer_ms() : 0.0; }
  void Release() override { runner_.reset(); }

 private:
  std::string name_ = "rknn";
  std::shared_ptr<RknnRunner> runner_;
};

std::unique_ptr<IModelBuilder> MakeRknnBuilder() {
  return std::unique_ptr<IModelBuilder>(new RknnModelBuilder());
}

}  // namespace

TOD_RT_REGISTER_BUILDER(rknn, MakeRknnBuilder)
TOD_RT_REGISTER_BUILDER(rknn_npu, MakeRknnBuilder)

std::unique_ptr<Detector> CreateRknnDetector(const std::string& model_name,
                                             const DetectorOptions& opts, const ModelRecipe& recipe,
                                             const std::string& builder_name,
                                             const std::string& preproc_name,
                                             const std::string& postproc_name) {
  (void)recipe;
  (void)builder_name;
  DetectorOptions o = opts;
  if (o.device != Device::kNpu) {
    if (o.device != Device::kAuto) {
      log_warn(std::string("所选 builder 是 RKNN，但 device=") + to_string(o.device) +
               "，已按 NPU 执行。");
    }
    o.device = Device::kNpu;
  }
  o.precision = Precision::kINT8;  // RKNN 量化模型即 INT8
  return BuildRknnDetector(model_name, o, preproc_name, postproc_name);
}

}  // namespace todrt

#endif  // TODRT_HAVE_RKNN
