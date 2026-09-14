// factory.hpp —— 工厂层（本文件的全部内容就是「怎么方便地调用」的答案）
//
// 分层：
//
//     DeployConfig (Python 导出：网络结构是谁)
//            +
//     DetectorOptions (部署侧：跑在哪块硬件、什么精度、什么阈值)
//            │
//            ▼
//   ┌────────────────────┐   名字 → 实现，全部来自自注册表
//   │  DetectorFactory   │   model × builder × preproc × postproc
//   └────────────────────┘
//            │
//            ▼
//     Detector (统一句柄：Run / Submit / TryGet / Bench / Describe)
//
// 三个设计要点：
//   1. **调用方不 new 任何具体类**，只给名字；新增变体 = 新增一个 .cpp + 一行宏。
//   2. **变体配方**（ModelRecipe）是"一键式"入口：一个变体把「用哪个 builder、
//      什么 layout、哪些 strides、默认后处理」全部封装好，调用方只写一行
//      `Detector::Create("SPAE-YOLOv8n", opts)`。
//   3. 子工厂（preproc / postproc / builder）保持可覆盖：做消融或换加速方案时
//      不必改任何变体代码，也不必重新编译（只要实现已注册）。
#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "todrt/core.hpp"
#include "todrt/detector_options.hpp"
#include "todrt/modules.hpp"

namespace todrt {

// ------------------------------------------------------------------ 构建器

/// 引擎来源接口。TensorRT 实现见 backend/engine_trt.hpp；
/// 单元测试可用假实现（tests/cpp_smoke.cpp 里的 MockEngineBuilder）。
class IModelBuilder {
 public:
  virtual ~IModelBuilder() = default;
  virtual const std::string& name() const = 0;
  /// 构建或加载引擎；失败抛 TritError。
  virtual void Build(const BuildConfig& cfg) = 0;
  /// 该实现当前是否可用（未编译 TensorRT / 无 GPU 时返回 false + 原因）。
  virtual bool Available(std::string* reason) const = 0;
  /// 推理入口：把输入张量与输出张量指针交给后端执行一次。
  struct RunIO {
    const void* input = nullptr;   ///< 主机侧 NCHW float
    size_t input_bytes = 0;
    std::vector<void*> outputs;    ///< 主机侧输出缓冲（按输出序号）
    std::vector<size_t> output_bytes;
    int batch = 1;
  };
  /// 执行一次推理；成功返回 true。device-side 拷贝由实现负责。
  virtual bool Run(const RunIO& io, std::string* err) = 0;
  /// 引擎输入形状（NCHW）；不可用时返回空。
  virtual std::vector<int64_t> input_shape() const = 0;
  /// 引擎每个输出的形状（**当前实际**，动态 shape 下随输入变化）。
  virtual std::vector<std::vector<int64_t>> output_shapes() const = 0;
  /// 引擎每个输出的数据类型。
  virtual std::vector<DataType> output_types() const = 0;
  /// 输入张量名（空 = 默认名）。
  virtual std::string input_name() const = 0;
  /// 输出张量名列表。
  virtual std::vector<std::string> output_names() const = 0;
  /// 把引擎序列化到文件。
  virtual bool serialize(const std::string& path, std::string* err) = 0;
  /// 一次推理的实测耗时（ms，CUDA event 计时）；无 GPU 时返回 0。
  virtual double last_infer_ms() const = 0;
  /// 释放所有设备资源（析构前显式调用，避免 CUDA 上下文销毁顺序问题）。
  virtual void Release() = 0;
};

// ------------------------------------------------------------------ 变体配方

/// 一个变体构建时用的"配方"：把「工厂链路 + 默认解码参数 + 硬件预设」固化下来。
struct ModelRecipe {
  std::string builder;   ///< 空 = "auto"
  std::string preproc;   ///< 空 = "auto"
  std::string postproc;  ///< 空 = "auto"

  DecodeOptions decode;  ///< 默认解码参数（被 DeployConfig / DetectorOptions 覆盖）
  PreprocessOptions preproc_opts;
  PostprocessOptions postproc_opts;

  /// 期望的输出布局提示：PostprocessOptions / DecodeOptions 未指定时的兜底。
  OutputLayout layout = OutputLayout::kAnchorMajorDfl;

  std::string hardware_preset;  ///< "orin" / "dgp" / "x86" / 空
  std::string source;           ///< 论文/仓库
  std::string license;
  std::string notes;

  /// 可选钩子：创建 Detector 前对 options 做变体特有的补全/校验。
  std::function<void(DetectorOptions&)> tune;
};

using ModelRecipeFn = ModelRecipe (*)();
using BuilderFn = std::unique_ptr<IModelBuilder> (*)();
using PreprocFn = std::unique_ptr<IPreprocessor> (*)();
using PostprocFn = std::unique_ptr<IPostprocessor> (*)();

/// 自注册钩子（由宏展开调用，实现见 src/factory.cpp）。
void register_model_recipe(const std::string& name, ModelRecipeFn fn);
void register_builder_factory(const std::string& name, BuilderFn fn);
void register_preproc_factory(const std::string& name, PreprocFn fn);
void register_postproc_factory(const std::string& name, PostprocFn fn);

/// ★ 新增一个可部署变体只需要这一段（放在 src/models/<name>.cpp 里）::
///
///     TOD_RT_DEFINE_RECIPE(SPAE_YOLOv8n, SpaeYolov8nRecipe) {
///       ModelRecipe r;
///       r.builder = "yolov8-trt";
///       r.decode.strides = {4, 8, 16, 32};
///       r.notes = "SPAE-YOLOv8n：ADown + P2 + Efficient_UAVDet + SIoU 训练";
///       return r;
///     }
///     // 强制登记来源与许可证（用函数而非宏：别名列表里的逗号不会引发预处理问题）
///     namespace {
///     const bool g_meta = []() {
///       register_model_metadata("SPAE_YOLOv8n", {"spae-yolov8", "spae"},
///                               "SPAE-YOLOv8 (Sensors 2026)", "论文 CC BY 4.0",
///                               "Jetson Orin / dGPU；FP16 或 INT8", "anchor 34000");
///       return true;
///     }();
///     }
#define TOD_RT_DEFINE_RECIPE(NAME, FN)                                          \
  static ::todrt::ModelRecipe FN();                                             \
  namespace {                                                                   \
  const bool tod_rt_recipe_##NAME = []() {                                      \
    ::todrt::register_model_recipe(#NAME, &FN);                                 \
    return true;                                                                \
  }();                                                                          \
  }                                                                             \
  static ::todrt::ModelRecipe FN()

/// 给已有配方追加别名（不登记元数据）。
///
/// 首选做法是直接在 `register_model_metadata()` 的别名列表里写 —— 元数据与别名
/// 一处维护。这个宏只用于"某些别名需要在别的 TU 里补登记"的场景。
/// ALIAS 必须是**合法 C++ 标识符**（用于生成变量名）；连字符写法请走元数据列表。
#define TOD_RT_ALIAS_RECIPE(ALIAS, FN)                                          \
  namespace {                                                                   \
  const bool tod_rt_alias_##ALIAS = []() {                                      \
    ::todrt::register_model_recipe(#ALIAS, &FN);                                \
    return true;                                                                \
  }();                                                                          \
  }

/// 登记一个变体的元数据（来源 / 许可证 / 硬件要求 / 成本）。
///
/// **强制**：部署产物必须能追溯到论文与许可证（与 Python 侧 registry 同一原则）。
/// 用普通函数而不是宏，是因为别名列表里的逗号/花括号会让宏在部分编译器上
/// 展开出错——注册元数据这种"必须写对"的东西不值得冒这个险。
void register_model_metadata(const std::string& name, const std::vector<std::string>& aliases,
                             const std::string& source, const std::string& license,
                             const std::string& hardware, const std::string& cost,
                             const char* file = nullptr, int line = 0);

/// 子工厂自注册（preproc / postproc / builder 实现文件里用）。
#define TOD_RT_REGISTER_BUILDER(NAME, FN)                                       \
  namespace {                                                                   \
  const bool tod_rt_builder_##NAME = []() {                                     \
    ::todrt::register_builder_factory(#NAME, &FN);                              \
    return true;                                                                \
  }();                                                                          \
  }
#define TOD_RT_REGISTER_PREPROC(NAME, FN)                                       \
  namespace {                                                                   \
  const bool tod_rt_preproc_##NAME = []() {                                     \
    ::todrt::register_preproc_factory(#NAME, &FN);                              \
    return true;                                                                \
  }();                                                                          \
  }
#define TOD_RT_REGISTER_POSTPROC(NAME, FN)                                      \
  namespace {                                                                   \
  const bool tod_rt_postproc_##NAME = []() {                                    \
    ::todrt::register_postproc_factory(#NAME, &FN);                             \
    return true;                                                                \
  }();                                                                          \
  }

// ------------------------------------------------------------------ Detector

/// 一次异步请求的结果。
struct InferenceResult {
  uint64_t request_id = 0;
  std::vector<Detection> detections;
  double preprocess_ms = 0.0;
  double infer_ms = 0.0;
  double postprocess_ms = 0.0;
};

/// 统一推理句柄。**这是调用方唯一需要认识的类型。**
///
/// 同步（最简单的用法）::
///
///     auto det = todrt::Detector::CreateFromFile(cfg);   // 一行
///     auto results = det->Run(bgr_image);                // 一行
///
/// 异步（多路视频 / ROS 回调里不阻塞）::
///
///     det->Submit(bgr_image);
///     InferenceResult r;
///     while (det->TryGet(&r)) { render(r.detections); }
///
/// 性能实测（**目标设备上的数字才算数**）::
///
///     auto b = det->Bench(bgr_image, 10, 200);
///     std::printf("%.2f ms  %.1f FPS\n", b.end_to_end_ms, b.fps);
class Detector {
 public:
  virtual ~Detector() = default;

  /// ★ 工厂入口：按名字装配完整流水线。
  /// @param model_name 注册的变体名（空 = 用 opts.model）
  static std::unique_ptr<Detector> Create(const std::string& model_name,
                                          const DetectorOptions& opts);
  /// ★ 便捷入口：直接读部署配置 JSON（实机上的推荐用法）。
  static std::unique_ptr<Detector> CreateFromFile(const std::string& config_path);

  // ---- 元信息 ----
  virtual const std::string& model_name() const = 0;
  virtual const DetectorOptions& options() const = 0;
  virtual void input_shape(int* w, int* h) const = 0;
  virtual std::vector<std::vector<int64_t>> output_shapes() const = 0;
  /// 人类可读摘要（模型/精度/设备/工厂链/形状），可安全打印到日志。
  virtual std::string Describe() const = 0;

  // ---- 推理 ----
  /// 同步：一批图 → 每图检测结果（**原图**坐标）。
  virtual std::vector<std::vector<Detection>> Run(const std::vector<ImageView>& images) = 0;
  std::vector<Detection> Run(const ImageView& image) {
    return Run(std::vector<ImageView>{image}).at(0);
  }

  /// 异步提交单图（内部拷贝像素，调用方可立刻复用缓冲）。
  virtual uint64_t Submit(const ImageView& image) = 0;
  /// 非阻塞取结果；无已完成请求返回 false。
  virtual bool TryGet(InferenceResult* out) = 0;
  /// 阻塞等待最早完成的请求（timeout_ms < 0 = 无限等待）。
  virtual bool Wait(InferenceResult* out, int timeout_ms = -1) = 0;
  /// 已提交但未取回的请求数。
  virtual size_t pending() const = 0;

  // ---- 性能 ----
  struct BenchResult {
    double preprocess_ms = 0.0;
    double infer_ms = 0.0;
    double postprocess_ms = 0.0;
    double end_to_end_ms = 0.0;
    double fps = 0.0;
    int iters = 0;
    std::string device;
    std::string precision;
  };
  virtual BenchResult Bench(const ImageView& image, int warmup, int iters) = 0;

  /// 直接访问最近一次推理的引擎输出（调试/与 Python 对拍用）。
  virtual std::vector<TensorView> last_outputs() const = 0;

 protected:
  /// **子类唯一需要实现的方法**：一次前处理结果 → 引擎推理 + 解码 + 坐标反变换。
  ///
  /// 为什么返回 `vector<Detection>` 而不是 `vector<vector<Detection>>`：
  /// 调用方（Run / Submit）永远只处理"这一次进来的图"，批量化属于引擎内部细节
  /// （TensorRT 的 batch 维由引擎自己管）。这样契约最简单，也不会出现
  /// "外层是 batch 还是 image" 的歧义——小目标检测里这种歧义会导致框全错。
  ///
  /// @param pp       前处理结果（含 scale/pad，用于坐标反变换）
  /// @param timings  [0]=前处理(ms) [1]=推理(ms) [2]=后处理(ms)；可传 nullptr
  /// @return 该图的检测结果（**原图像素坐标**，已裁剪、已按分数降序）
  virtual std::vector<Detection> RunBatch(const PreprocessResult& pp, double* preprocess_ms,
                                          double* infer_ms, double* postprocess_ms) = 0;

  /// 供后端在初始化阶段写入实际生效的配置（精度回落、动态 shape 等）。
  DetectorOptions& mutable_options() { return opts_; }

  /// 只读的原型配置（构造时传入的那份，尚未被后端补全）；用于诊断输出。
  const DetectorOptions& prototype_options() const { return opts_; }

 private:
  DetectorOptions opts_;
};

// ------------------------------------------------------------------ 工厂清单

bool has_model(const std::string& name);
std::vector<std::string> model_names();
ModelRecipe model_recipe(const std::string& name);

std::unique_ptr<IPreprocessor> make_preprocessor(const std::string& name);
std::unique_ptr<IPostprocessor> make_postprocessor(const std::string& name);
std::unique_ptr<IModelBuilder> make_builder(const std::string& name);

/// 一次装配的链路描述（用于日志与产物存档）。
struct AssemblyReport {
  std::string model;
  std::string builder;
  std::string preproc;
  std::string postproc;
  std::string device;
  std::string precision;
  std::string layout;
  std::string strides;
  bool builder_available = false;
  std::string builder_unavailable_reason;
  std::vector<std::string> warnings;
  std::vector<std::string> registry_missing;  ///< check_registry() 缺失的依赖

  std::string ToText() const;
  std::string ToJson() const;
};

/// 解析「名字」链路并做一致性校验，但不真正建引擎（干跑自检）。
/// CLI `--dry-run` 与单元测试都走这里：**没有 GPU 的机器上也能验证工厂装配**。
AssemblyReport PlanAssembly(const std::string& model_name, const DetectorOptions& opts);

// ------------------------------------------------------------------ 后端入口
//
// 由后端实现（backend/engine_trt.cpp 或 backend/engine_stub.cpp）。
// 未编译 TensorRT 时，stub 版本会给出带修复建议的明确错误，而不是链接失败。
std::unique_ptr<Detector> CreateDeviceDetector(const std::string& model_name,
                                               const DetectorOptions& opts,
                                               const ModelRecipe& recipe,
                                               const std::string& builder_name,
                                               const std::string& preproc_name,
                                               const std::string& postproc_name);

/// 后端是否已编译进来（TensorRT）。
bool backend_available(std::string* reason);

}  // namespace todrt
