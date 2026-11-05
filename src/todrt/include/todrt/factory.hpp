#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "todrt/core.hpp"
#include "todrt/detector_options.hpp"
#include "todrt/modules.hpp"

namespace todrt {

class IModelBuilder {
 public:
  virtual ~IModelBuilder() = default;
  virtual const std::string& name() const = 0;
  virtual void Build(const BuildConfig& cfg) = 0;
  virtual bool Available(std::string* reason) const = 0;
  struct RunIO {
    const void* input = nullptr;
    size_t input_bytes = 0;
    std::vector<void*> outputs;
    std::vector<size_t> output_bytes;
    int batch = 1;
  };
  virtual bool Run(const RunIO& io, std::string* err) = 0;
  virtual std::vector<int64_t> input_shape() const = 0;
  virtual std::vector<std::vector<int64_t>> output_shapes() const = 0;
  virtual std::vector<DataType> output_types() const = 0;
  virtual std::string input_name() const = 0;
  virtual std::vector<std::string> output_names() const = 0;
  virtual bool serialize(const std::string& path, std::string* err) = 0;
  virtual double last_infer_ms() const = 0;
  virtual void Release() = 0;
};

// 配方只声明组件与默认配置；SDK 资源由具体 builder/runner 持有。
struct ModelRecipe {
  std::string builder;
  std::string preproc;
  std::string postproc;

  DecodeOptions decode;
  PreprocessOptions preproc_opts;
  PostprocessOptions postproc_opts;

  OutputLayout layout = OutputLayout::kAnchorMajorDfl;

  std::string hardware_preset;
  std::string source;
  std::string license;
  std::string notes;

  std::function<void(DetectorOptions&)> tune;
};

using ModelRecipeFn = ModelRecipe (*)();
using BuilderFn = std::unique_ptr<IModelBuilder> (*)();
using PreprocFn = std::unique_ptr<IPreprocessor> (*)();
using PostprocFn = std::unique_ptr<IPostprocessor> (*)();

void register_model_recipe(const std::string& name, ModelRecipeFn fn);
void register_builder_factory(const std::string& name, BuilderFn fn);
void register_preproc_factory(const std::string& name, PreprocFn fn);
void register_postproc_factory(const std::string& name, PostprocFn fn);

#define TOD_RT_DEFINE_RECIPE(NAME, FN)                                          \
  static ::todrt::ModelRecipe FN();                                             \
  namespace {                                                                   \
  const bool tod_rt_recipe_##NAME = []() {                                      \
    ::todrt::register_model_recipe(#NAME, &FN);                                 \
    return true;                                                                \
  }();                                                                          \
  }                                                                             \
  static ::todrt::ModelRecipe FN()

#define TOD_RT_ALIAS_RECIPE(ALIAS, FN)                                          \
  namespace {                                                                   \
  const bool tod_rt_alias_##ALIAS = []() {                                      \
    ::todrt::register_model_recipe(#ALIAS, &FN);                                \
    return true;                                                                \
  }();                                                                          \
  }

void register_model_metadata(const std::string& name, const std::vector<std::string>& aliases,
                             const std::string& source, const std::string& license,
                             const std::string& hardware, const std::string& cost,
                             const char* file = nullptr, int line = 0);

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

struct InferenceResult {
  uint64_t request_id = 0;
  std::vector<Detection> detections;
  double preprocess_ms = 0.0;
  double infer_ms = 0.0;
  double postprocess_ms = 0.0;
};

class Detector {
 public:
  virtual ~Detector() = default;

  static std::unique_ptr<Detector> Create(const std::string& model_name,
                                          const DetectorOptions& opts);
  static std::unique_ptr<Detector> CreateFromFile(const std::string& config_path);

  virtual const std::string& model_name() const = 0;
  virtual const DetectorOptions& options() const = 0;
  virtual void input_shape(int* w, int* h) const = 0;
  virtual std::vector<std::vector<int64_t>> output_shapes() const = 0;
  virtual std::string Describe() const = 0;

  virtual std::vector<std::vector<Detection>> Run(const std::vector<ImageView>& images) = 0;
  std::vector<Detection> Run(const ImageView& image) {
    return Run(std::vector<ImageView>{image}).at(0);
  }

  virtual uint64_t Submit(const ImageView& image) = 0;
  virtual bool TryGet(InferenceResult* out) = 0;
  virtual bool Wait(InferenceResult* out, int timeout_ms = -1) = 0;
  virtual size_t pending() const = 0;

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

  virtual std::vector<TensorView> last_outputs() const = 0;

 protected:
  void set_model_name(std::string n) { model_name_ = std::move(n); }
  const std::string& model_name_impl() const { return model_name_; }

  virtual std::vector<Detection> RunBatch(const PreprocessResult& pp, double* preprocess_ms,
                                          double* infer_ms, double* postprocess_ms) = 0;

  DetectorOptions& mutable_options() { return opts_; }

  const DetectorOptions& prototype_options() const { return opts_; }

  std::string model_name_;

 private:
  DetectorOptions opts_;
};

bool has_model(const std::string& name);
std::vector<std::string> model_names();
ModelRecipe model_recipe(const std::string& name);

std::unique_ptr<IPreprocessor> make_preprocessor(const std::string& name);
std::unique_ptr<IPostprocessor> make_postprocessor(const std::string& name);
std::unique_ptr<IModelBuilder> make_builder(const std::string& name);

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
  std::vector<std::string> registry_missing;

  std::string ToText() const;
  std::string ToJson() const;
};

AssemblyReport PlanAssembly(const std::string& model_name, const DetectorOptions& opts);

std::unique_ptr<Detector> CreateDeviceDetector(const std::string& model_name,
                                               const DetectorOptions& opts,
                                               const ModelRecipe& recipe,
                                               const std::string& builder_name,
                                               const std::string& preproc_name,
                                               const std::string& postproc_name);

bool backend_available(std::string* reason);

}
