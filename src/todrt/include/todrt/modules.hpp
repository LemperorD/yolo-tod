#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "todrt/core.hpp"

namespace todrt {

namespace json {
class Value;
}

// 框坐标使用当前图像的像素 xyxy；对外结果已还原到源图。
struct BBox {
  float x1 = 0.f;
  float y1 = 0.f;
  float x2 = 0.f;
  float y2 = 0.f;

  float w() const { return x2 - x1; }
  float h() const { return y2 - y1; }
  float area() const { return w() * h(); }
  float cx() const { return 0.5f * (x1 + x2); }
  float cy() const { return 0.5f * (y1 + y2); }
};

struct Detection {
  BBox box;
  float score = 0.f;
  int class_id = -1;
  float box_score = 0.f;
};

// 非拥有的图像视图；step 为每行字节数，0 表示连续存储。
struct ImageView {
  const uint8_t* data = nullptr;
  int width = 0;
  int height = 0;
  int channels = 3;
  size_t step = 0;
  bool bgr = true;

  size_t row_bytes() const {
    return step ? step : static_cast<size_t>(width) * static_cast<size_t>(channels);
  }
  bool valid() const {
    return data != nullptr && width > 0 && height > 0 &&
           (channels == 1 || channels == 3 || channels == 4);
  }
};

enum class DataType { kF32 = 0, kF16 = 1, kI8 = 2, kU8 = 3, kI32 = 4 };

const char* to_string(DataType t);
size_t dtype_size(DataType t);

// 非拥有的张量视图，解释 data 前必须核对 dtype 与 shape。
struct TensorView {
  const void* data = nullptr;
  DataType dtype = DataType::kF32;
  std::vector<int64_t> shape;

  bool valid() const { return data != nullptr && !shape.empty(); }
  int64_t rank() const { return static_cast<int64_t>(shape.size()); }
  int64_t dim(int i) const {
    const int64_t r = rank();
    return shape[static_cast<size_t>(i < 0 ? r + i : i)];
  }
  int64_t numel() const {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
  }
};

enum class ResizeMode {
  kLetterbox = 0,
  kStretch = 1,
  kIntegerScale = 2
};

enum class PadValue { kGray114 = 0, kZero = 1, kEdge = 2 };

enum class TensorLayout { kNchw = 0, kNhwc = 1 };

const char* to_string(TensorLayout l);

struct PreprocessResult {
  std::vector<uint8_t> bytes;
  DataType dtype = DataType::kF32;
  TensorLayout layout = TensorLayout::kNchw;
  int width = 0;
  int height = 0;
  int batch = 0;
  int channels = 3;

  std::vector<float> scale;
  std::vector<float> pad_x;
  std::vector<float> pad_y;
  std::vector<int> src_w;
  std::vector<int> src_h;

  size_t sample_bytes() const {
    return static_cast<size_t>(channels) * static_cast<size_t>(height) *
           static_cast<size_t>(width) * dtype_size(dtype);
  }
  size_t sample_elems() const {
    return static_cast<size_t>(channels) * static_cast<size_t>(height) *
           static_cast<size_t>(width);
  }
  bool empty() const { return bytes.empty(); }

  const float* float_ptr() const {
    if (dtype != DataType::kF32) {
      throw TritError("前处理输出是 " + std::string(to_string(dtype)) +
                      "，不能按 float 读取（RKNN 量化输入是 uint8）");
    }
    return reinterpret_cast<const float*>(bytes.data());
  }
  size_t elems() const { return bytes.size() / dtype_size(dtype); }
};

enum class PreprocOutput {
  kFloat32 = 0,
  // 量化输入保留 uint8 原值，由模型处理归一化。
  kUint8Raw = 1
};

struct PreprocessOptions {
  int input_width = 640;
  int input_height = 640;
  int pad_multiple = 32;
  ResizeMode mode = ResizeMode::kLetterbox;
  PadValue pad_value = PadValue::kGray114;
  float norm_scale = 1.f / 255.f;
  float norm_bias = 0.f;
  bool to_rgb = true;
  PreprocOutput output = PreprocOutput::kFloat32;
  TensorLayout layout = TensorLayout::kNchw;

  DataType output_dtype() const {
    return output == PreprocOutput::kUint8Raw ? DataType::kU8 : DataType::kF32;
  }
  bool is_quantized_input() const { return output == PreprocOutput::kUint8Raw; }

  static PreprocessOptions ForInput(int w, int h) {
    PreprocessOptions o;
    o.input_width = w;
    o.input_height = h;
    return o;
  }
  static PreprocessOptions ForRknnQuantized(int w, int h) {
    PreprocessOptions o;
    o.input_width = w;
    o.input_height = h;
    o.output = PreprocOutput::kUint8Raw;
    o.layout = TensorLayout::kNhwc;
    return o;
  }
};

BBox inv_transform(const BBox& net_box, float scale, float pad_x, float pad_y);

class IPreprocessor {
 public:
  virtual ~IPreprocessor() = default;
  virtual const std::string& name() const = 0;
  virtual const PreprocessOptions& options() const = 0;
  virtual PreprocessResult Run(const std::vector<ImageView>& images) = 0;
  virtual int out_width() const = 0;
  virtual int out_height() const = 0;

  std::vector<std::vector<Detection>> ToSourceCoords(std::vector<std::vector<Detection>> dets,
                                                     const PreprocessResult& pre) const;
};

// 输出布局必须与 Python 导出的部署配置一致，不能根据模型名称猜测。
enum class OutputLayout {
  kAnchorMajorDfl = 0,
  kAnchorMajorDflTransposed = 1,
  kFeatureMajorDfl = 2,
  kPluginNms = 3
};

const char* to_string(OutputLayout l);

struct DecodeOptions {
  int num_classes = 10;
  int reg_max = 16;
  int max_det = 300;
  float conf_threshold = 0.25f;
  std::vector<int> strides{4, 8, 16, 32};
  OutputLayout layout = OutputLayout::kAnchorMajorDfl;
  std::vector<int> level_channels;
  int input_width = 640;
  int input_height = 640;
};

struct Grid {
  int stride = 0;
  int w = 0;
  int h = 0;
  int64_t anchors() const { return static_cast<int64_t>(w) * h; }
};

std::vector<Grid> make_grids(const std::vector<int>& strides, int input_w, int input_h);

struct DflResult {
  float l = 0.f;
  float t = 0.f;
  float r = 0.f;
  float b = 0.f;
};

DflResult dfl_decode(const float* dist, int reg_max);
float dfl_expectation(const float* dist, int reg_max);
float sigmoid(float x);

std::vector<Detection> decode_predictions(const TensorView& output, const DecodeOptions& opt);

enum class NmsKind { kHard = 0, kSoft = 1 };

struct PostprocessOptions {
  DecodeOptions decode;
  NmsKind nms = NmsKind::kHard;
  float iou_threshold = 0.45f;
  float sigma = 0.5f;
  float soft_score_threshold = 0.001f;
  bool class_agnostic = false;
  int max_det = 300;
};

float iou(const BBox& a, const BBox& b);

void nms_hard(const std::vector<Detection>& dets, float iou_threshold, bool class_agnostic,
              std::vector<int>& keep);

void nms_soft(std::vector<Detection>& dets, float iou_threshold, float sigma, float score_floor,
              bool class_agnostic, std::vector<int>& keep);

class IPostprocessor {
 public:
  virtual ~IPostprocessor() = default;
  virtual const std::string& name() const = 0;
  virtual const PostprocessOptions& options() const = 0;
  virtual std::vector<std::vector<Detection>> Run(const std::vector<TensorView>& outputs,
                                                 const PreprocessResult& pre) = 0;
};

enum class BuilderFlag : uint32_t {
  kNone = 0,
  kFp16Fallback = 1u << 0,
  kInt8Fallback = 1u << 1,
  kDlaGpuFallback = 1u << 2,
  kDlaStandalone = 1u << 3,
  kCudaGraphs = 1u << 4,
  kSparsity = 1u << 5,
  kProfilingVerbosity = 1u << 6,
  kStronglyTyped = 1u << 7,
  kRknnMultiCore = 1u << 8,
  kRknnPacked = 1u << 9,
  kOrtGraphOptimize = 1u << 10,
  kOrtDisableCpuFallback = 1u << 11,
  kOvCacheCompiled = 1u << 12
};

struct BuildConfig {
  std::string onnx_path;
  std::string engine_path;
  std::string serialize_out;
  std::string input_name;
  std::string output_name;
  std::string fallback_input_name = "images";

  Precision precision = Precision::kFP16;
  Device device = Device::kAuto;
  int accelerator_core = 0;
  int dla_core = 0;
  int dla_memory_limit_mb = 512;
  bool allow_gpu_fallback = true;

  std::string rknn_model_path;
  int rknn_core_num = 1;
  bool rknn_dequantize_output = true;

  std::string ort_provider = "auto";
  int ort_intra_threads = 0;
  int ort_inter_threads = 0;
  std::string ort_optimization = "all";

  std::string ov_device = "CPU";
  std::string ov_cache_dir;
  int ov_num_streams = 0;
  int ov_num_threads = 0;
  std::string ov_performance_hint = "LATENCY";

  bool dynamic_batch = false;
  int max_batch = 8;
  bool dynamic_shape = false;
  std::array<int, 2> shape_min{640, 640};
  std::array<int, 2> shape_opt{640, 640};
  std::array<int, 2> shape_max{640, 640};

  size_t workspace_mb = 1024;
  size_t dla_workspace_mb = 0;
  int num_build_threads = 0;
  bool timing_cache = true;
  bool tactic_sources = true;
  bool hardware_compatibility = false;

  std::string int8_calib_cache;
  std::string int8_calib_data;
  std::string int8_calib_alg = "entropy";
  int int8_calib_batches = 10;

  bool verbose = false;
  bool dump_tactics = false;
  uint32_t flags = 0;

  bool has(BuilderFlag f) const { return (flags & static_cast<uint32_t>(f)) != 0; }
  void set(BuilderFlag f, bool on = true) {
    if (on) {
      flags |= static_cast<uint32_t>(f);
    } else {
      flags &= ~static_cast<uint32_t>(f);
    }
  }
  bool set_flag(const std::string& name, bool on = true);
};

struct DeployConfig {
  std::string schema = "todrt.deploy/v1";
  std::string model;
  std::string model_id;

  BuildConfig build;

  int num_classes = 10;
  int reg_max = 16;
  OutputLayout layout = OutputLayout::kAnchorMajorDfl;
  std::vector<int> strides{4, 8, 16, 32};
  std::vector<int> level_channels;
  std::array<int, 2> input{640, 640};
  int pad_multiple = 32;

  std::string source;
  std::string notes;
  std::string dataset;
  std::vector<std::string> class_names;

  static DeployConfig FromJson(const json::Value& v);
  static DeployConfig FromFile(const std::string& path);

  void Validate() const;
};

}
