// modules.hpp —— 可插拔模块的接口集（前处理 / 解码 / 后处理）
//
// 一个「模块」= 纯接口 + Options + 一个 create 函数。工厂只认接口与名字，因此：
//   * 换 letterbox 策略不影响解码与 NMS；
//   * 换解码器不影响引擎与流水线；
//   * 新增变体不需要改任何调用方代码。
//
// 本文件不依赖 TensorRT/CUDA，任何 C++17 工具链都能编译（便于在 x86 上做逻辑自检、
// 在 Linux 构建机上交叉编译到 Jetson）。
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
}  // namespace json

// ------------------------------------------------------------------ 基础几何/图像

/// 图像外框（左上+右下，像素坐标）。
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

/// 一条检测结果（坐标为**原图**像素坐标）。
struct Detection {
  BBox box;
  float score = 0.f;
  int class_id = -1;
  /// 可选：框的置信度（IoU-aware 头才有；YOLOv8 默认 = score）。
  float box_score = 0.f;
};

/// 一张图的像素视图（交错 HWC）。只借用、不拥有内存：
/// 调用方需保证在 Run()/Submit() 返回前缓冲有效（Submit 内部会拷贝）。
struct ImageView {
  const uint8_t* data = nullptr;
  int width = 0;
  int height = 0;
  int channels = 3;  ///< 仅支持 1/3/4
  size_t step = 0;   ///< 一行字节数；0 = width*channels（无 padding）
  bool bgr = true;   ///< 默认 OpenCV 式 BGR（V4L2/GStreamer/相机多为 BGR）

  size_t row_bytes() const {
    return step ? step : static_cast<size_t>(width) * static_cast<size_t>(channels);
  }
  bool valid() const {
    return data != nullptr && width > 0 && height > 0 &&
           (channels == 1 || channels == 3 || channels == 4);
  }
};

// ------------------------------------------------------------------ 张量视图

enum class DataType { kF32 = 0, kF16 = 1, kI8 = 2, kU8 = 3, kI32 = 4 };

const char* to_string(DataType t);
size_t dtype_size(DataType t);

/// 对某个 blob 的只读视图：仅描述布局，不持有内存。
/// data 为设备内存时，仅在拷贝回主机后才有效（Detector::last_outputs 保证）。
struct TensorView {
  const void* data = nullptr;
  DataType dtype = DataType::kF32;
  std::vector<int64_t> shape;  ///< 例如 {1, 84, 8400}

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

// ------------------------------------------------------------------ 前处理

enum class ResizeMode {
  kLetterbox = 0,    ///< 等比缩放 + 居中填充（保持长宽比，默认）
  kStretch = 1,      ///< 直接拉伸到目标尺寸（省时间，但比例变化大时伤小目标）
  kIntegerScale = 2  ///< 只允许整数倍缩放，再填到目标尺寸（航拍小目标常用）
};

enum class PadValue { kGray114 = 0, kZero = 1, kEdge = 2 };

/// 前处理结果：一批图 → NCHW RGB float 缓冲 + 可逆变换记录。
struct PreprocessResult {
  /// NCHW float：索引 = ((n*3 + c)*H + y)*W + x
  std::vector<float> tensor;
  int width = 0;
  int height = 0;
  int batch = 0;
  int channels = 3;

  /// 坐标反变换参数（每张图一份）
  std::vector<float> scale;
  std::vector<float> pad_x;
  std::vector<float> pad_y;
  std::vector<int> src_w;
  std::vector<int> src_h;

  size_t sample_stride() const {
    return static_cast<size_t>(channels) * static_cast<size_t>(height) *
           static_cast<size_t>(width);
  }
};

struct PreprocessOptions {
  int input_width = 640;
  int input_height = 640;
  /// letterbox 对齐倍数；YOLOv8 导出一般取 32（Ultralytics 的 auto=32）。
  int pad_multiple = 32;
  ResizeMode mode = ResizeMode::kLetterbox;
  PadValue pad_value = PadValue::kGray114;
  /// 归一化：value = pixel * norm_scale + norm_bias（默认 1/255，0）
  float norm_scale = 1.f / 255.f;
  float norm_bias = 0.f;
  bool to_rgb = true;

  static PreprocessOptions ForInput(int w, int h) {
    PreprocessOptions o;
    o.input_width = w;
    o.input_height = h;
    return o;
  }
};

/// 坐标反变换：网络坐标 → 原图坐标。
BBox inv_transform(const BBox& net_box, float scale, float pad_x, float pad_y);

/// 前处理策略接口。**实现必须可被多线程并发调用**（Run 内不得改自身状态）。
class IPreprocessor {
 public:
  virtual ~IPreprocessor() = default;
  virtual const std::string& name() const = 0;
  virtual const PreprocessOptions& options() const = 0;
  /// 一批图 → 一个 NCHW 张量。batch 内共用一组 letterbox 参数
  /// （取 batch 内最保守的缩放比），保证能拼成单一张量。
  virtual PreprocessResult Run(const std::vector<ImageView>& images) = 0;
  virtual int out_width() const = 0;
  virtual int out_height() const = 0;

  /// 默认实现：坐标反变换 + 按原图裁剪 + 按分数排序（供后处理复用）。
  std::vector<std::vector<Detection>> ToSourceCoords(std::vector<std::vector<Detection>> dets,
                                                     const PreprocessResult& pre) const;
};

// ------------------------------------------------------------------ 解码

/// 网络输出的排布方式。**选错会导致框全乱但程序不报错，务必与导出方式对齐。**
enum class OutputLayout {
  /// [1, 4*reg_max+nc, A]：ultralytics 官方 YOLOv8 Detect 导出。A = 各尺度 anchor 之和。
  kAnchorMajorDfl = 0,
  /// [1, A, 4*reg_max+nc]：转置版（部分自定义导出工具链偏好）。
  kAnchorMajorDflTransposed = 1,
  /// [1, sum(4*reg_max+nc), A]：每层的 DFL 值全在前、类别分数在后，层间拼接。
  /// SPAE-YOLOv8 的 Efficient_UAVDet 独立实现即此排布。
  kFeatureMajorDfl = 2,
  /// [N, 6] 或 [N, 7]：插件（EfficientNMS_TRT 等）已完成解码，
  /// 列为 [x1,y1,x2,y2,score,class] 或 [batch,x1,y1,x2,y2,score,class]。
  kPluginNms = 3
};

const char* to_string(OutputLayout l);

struct DecodeOptions {
  int num_classes = 10;
  int reg_max = 16;  ///< DFL 分箱数；YOLOv8 = 16
  int max_det = 300;
  float conf_threshold = 0.25f;
  /// 特征层步长（顺序与 head 输出顺序一致）。P2–P5 → {4, 8, 16, 32}
  std::vector<int> strides{4, 8, 16, 32};
  OutputLayout layout = OutputLayout::kAnchorMajorDfl;
  /// kFeatureMajorDfl 时每层的通道数；空 = 每层都是 4*reg_max+nc
  std::vector<int> level_channels;
  /// 网络输入尺寸：解码必须知道它才能生成 anchor 网格（与 strides 一起决定 anchor 数）。
  /// 正常由部署配置注入，不需要手工填写。
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

/// DFL 分布 → 到框四边的距离（单位 = 网络输入像素）。
struct DflResult {
  float l = 0.f;
  float t = 0.f;
  float r = 0.f;
  float b = 0.f;
};

/// softmax 后求期望（等价 ultralytics 的 DFL 解码）。
DflResult dfl_decode(const float* dist, int reg_max);
/// 只求期望（输入已是概率分布时用）。
float dfl_expectation(const float* dist, int reg_max);
/// 数值稳定的 sigmoid。
float sigmoid(float x);

/// CPU 参考解码器：网络原始输出 → **网络坐标系**下的检测框（未做坐标反变换）。
/// 这是"唯一真相"：无 GPU 时的兜底路径，也是与 Python 实现对拍的基准。
std::vector<Detection> decode_predictions(const TensorView& output, const DecodeOptions& opt);

// ------------------------------------------------------------------ 后处理

enum class NmsKind { kHard = 0, kSoft = 1 };

struct PostprocessOptions {
  DecodeOptions decode;
  NmsKind nms = NmsKind::kHard;
  float iou_threshold = 0.45f;
  float sigma = 0.5f;  ///< soft-NMS 高斯方差
  float soft_score_threshold = 0.001f;
  bool class_agnostic = false;
  int max_det = 300;  ///< 每图保留上限
};

/// 两个框的 IoU（同坐标系即可）。
float iou(const BBox& a, const BBox& b);

/// 硬 NMS：dets 需按分数降序；返回保留索引（复用调用方缓冲）。
void nms_hard(const std::vector<Detection>& dets, float iou_threshold, bool class_agnostic,
              std::vector<int>& keep);

/// soft-NMS（高斯）：就地衰减 dets 的分数并返回保留索引。
void nms_soft(std::vector<Detection>& dets, float iou_threshold, float sigma, float score_floor,
              bool class_agnostic, std::vector<int>& keep);

/// 后处理策略接口：网络输出 → 每图检测结果（**网络坐标系**；坐标反变换
/// 由 IPreprocessor::ToSourceCoords 统一负责，避免两处各写一遍）。
class IPostprocessor {
 public:
  virtual ~IPostprocessor() = default;
  virtual const std::string& name() const = 0;
  virtual const PostprocessOptions& options() const = 0;
  /// @param outputs 引擎输出张量（按 binding 顺序）
  /// @param pre     本次前处理记录（batch 大小由它决定；batch=0 视为 1）
  virtual std::vector<std::vector<Detection>> Run(const std::vector<TensorView>& outputs,
                                                 const PreprocessResult& pre) = 0;
};

// ------------------------------------------------------------------ 构建选项

enum class BuilderFlag : uint32_t {
  kNone = 0,
  kFp16Fallback = 1u << 0,    ///< 允许 FP16 层（TensorRT < 11 用；10.x 起为强类型）
  kInt8Fallback = 1u << 1,
  kDlaGpuFallback = 1u << 2,  ///< DLA 不支持的层退回 GPU（Orin 上基本必须）
  kDlaStandalone = 1u << 3,   ///< 只保留 DLA 支持的层（要求全部算子可上 DLA）
  kCudaGraphs = 1u << 4,      ///< 运行期 CUDA Graph（小 batch 降 CPU 开销）
  kSparsity = 1u << 5,
  kProfilingVerbosity = 1u << 6,
  kStronglyTyped = 1u << 7
};

struct BuildConfig {
  // --- 来源 ---
  std::string onnx_path;
  std::string engine_path;    ///< 非空 = 直接反序列化（engine 与硬件/TRT 版本绑定）
  std::string serialize_out;  ///< 构建后落盘的 engine 路径
  std::string input_name;     ///< 空 = 自动探测
  std::string output_name;
  std::string fallback_input_name = "images";

  // --- 精度与硬件 ---
  Precision precision = Precision::kFP16;
  Device device = Device::kAuto;
  int dla_core = 0;
  int dla_memory_limit_mb = 512;
  bool allow_gpu_fallback = true;

  // --- shape ---
  bool dynamic_batch = false;
  int max_batch = 8;
  bool dynamic_shape = false;
  std::array<int, 2> shape_min{640, 640};
  std::array<int, 2> shape_opt{640, 640};
  std::array<int, 2> shape_max{640, 640};

  // --- 资源 ---
  size_t workspace_mb = 1024;
  size_t dla_workspace_mb = 0;  ///< 0 = 用 TensorRT 默认 DLA 工作区
  int num_build_threads = 0;
  bool timing_cache = true;
  bool tactic_sources = true;
  bool hardware_compatibility = false;

  // --- 量化 ---
  std::string int8_calib_cache;
  std::string int8_calib_data;
  std::string int8_calib_alg = "entropy";
  int int8_calib_batches = 10;

  // --- 调试 ---
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
  /// 从文本解析 flag（"fp16" / "dla_standalone" …）；返回是否识别。
  bool set_flag(const std::string& name, bool on = true);
};

/// 部署配置（JSON，由 Python 侧 `tools/export_onnx.py --deploy-config` 生成）。
///
/// 这是 **Python 训练端与 C++ 部署端之间唯一的契约**：网络结构信息（nc / reg_max /
/// strides / layout）只写一次，C++ 侧不再手抄——抄错一个数字就是"框全乱但程序不报错"
/// 这种最难查的 bug。
struct DeployConfig {
  std::string schema = "todrt.deploy/v1";
  /// 注册名（工厂查找键），通常与 Python variant.yaml 的 id 一致。
  /// 落盘时**必须**写进去：否则 C++ 端读回来时不知道该用哪个变体配方。
  std::string model;
  std::string model_id;  ///< 训练侧变体 id（可与 model 不同：id 里常有连字符）

  BuildConfig build;

  int num_classes = 10;
  int reg_max = 16;
  OutputLayout layout = OutputLayout::kAnchorMajorDfl;
  std::vector<int> strides{4, 8, 16, 32};
  std::vector<int> level_channels;
  std::array<int, 2> input{640, 640};  ///< (w, h)
  int pad_multiple = 32;

  // ---- 仅作记录，不参与推理 ----
  std::string source;  ///< 生成该配置的命令行
  std::string notes;
  std::string dataset;
  std::vector<std::string> class_names;

  static DeployConfig FromJson(const json::Value& v);
  static DeployConfig FromFile(const std::string& path);

  /// schema 大版本必须匹配，且关键字段必须自洽（否则抛 TritError）。
  void Validate() const;
};

}  // namespace todrt
