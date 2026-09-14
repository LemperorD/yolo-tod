// trt_factory.hpp —— TensorRT 后端入口（仅在编译了 TensorRT 时使用）
//
// 该头文件**不包含任何 TensorRT 头**，因此可以在无 TensorRT 的机器上被引用做
// 装配自检；真正的实现在 src/backend/*.cpp 里（受 TODRT_HAVE_TENSORRT 保护）。
#pragma once

#include <memory>
#include <string>

#include "todrt/factory.hpp"

namespace todrt {

class TrtEngine;

/// TensorRT 版引擎构建器。除了实现 IModelBuilder，还把内部 TrtEngine 暴露出来，
/// 以便 Detector 复用同一个引擎实例（**构建一次，别构建两次**）。
class TrtModelBuilder final : public IModelBuilder {
 public:
  const std::string& name() const override;
  void Build(const BuildConfig& cfg) override;
  bool Available(std::string* reason) const override;
  bool Run(const RunIO& io, std::string* err) override;
  std::vector<int64_t> input_shape() const override;
  std::vector<std::vector<int64_t>> output_shapes() const override;
  std::vector<DataType> output_types() const override;
  std::string input_name() const override;
  std::vector<std::string> output_names() const override;
  bool serialize(const std::string& path, std::string* err) override;
  double last_infer_ms() const override;
  void Release() override;

  const std::shared_ptr<TrtEngine>& engine() const { return engine_; }

 private:
  std::shared_ptr<TrtEngine> engine_;
};

/// 创建 TensorRT 构建器（供工厂与 CreateTrtDetector 共用）。
std::unique_ptr<IModelBuilder> make_trt_builder();

/// 构造走 TensorRT 的 Detector。仅当 TODRT_HAVE_TENSORRT=1 时存在实现。
std::unique_ptr<Detector> CreateTrtDetector(const std::string& model_name,
                                            const DetectorOptions& opts,
                                            const ModelRecipe& recipe,
                                            const std::string& builder_name,
                                            const std::string& preproc_name,
                                            const std::string& postproc_name);

}  // namespace todrt
