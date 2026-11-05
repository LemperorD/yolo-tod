#include <memory>
#include <string>

#include "todrt/factory.hpp"

#if !TODRT_HAVE_ANY_BACKEND

namespace todrt {
namespace {

class UnavailableBuilder : public IModelBuilder {
 public:
  const std::string& name() const override { return name_; }
  void Build(const BuildConfig&) override {
    throw TritError(
        "本次构建没有编译任何推理后端。\n"
        "  TensorRT  : cmake -DTODRT_WITH_TENSORRT=ON ...\n"
        "  RKNN      : cmake -DTODRT_WITH_RKNN=ON ...（RK3588 / librknnrt.so）\n"
        "  ONNX RT   : cmake -DTODRT_WITH_ORT=ON ...（通用 CPU 后路）\n"
        "  OpenVINO  : cmake -DTODRT_WITH_OPENVINO=ON ...（x86 CPU / Intel iGPU）\n"
        "  只想验证工厂装配与配置：todrt_cli dryrun <cfg.json>（不需要后端）");
  }
  bool Available(std::string* reason) const override {
    if (reason) {
      *reason =
          "未编译任何后端。装配自检 / 工厂清单 / 前处理 / 解码 / NMS 都可用，"
          "但无法建引擎。";
    }
    return false;
  }
  std::vector<int64_t> input_shape() const override { return {}; }
  std::vector<std::vector<int64_t>> output_shapes() const override { return {}; }
  std::vector<DataType> output_types() const override { return {}; }
  std::string input_name() const override { return {}; }
  std::vector<std::string> output_names() const override { return {}; }
  bool serialize(const std::string&, std::string* err) override {
    if (err) *err = "未编译任何后端";
    return false;
  }
  bool Run(const RunIO&, std::string* err) override {
    if (err) *err = "未编译任何后端";
    return false;
  }
  double last_infer_ms() const override { return 0.0; }
  void Release() override {}

 private:
  std::string name_ = "unavailable";
};

std::unique_ptr<IModelBuilder> MakeUnavailableBuilder() {
  return std::unique_ptr<IModelBuilder>(new UnavailableBuilder());
}

}

TOD_RT_REGISTER_BUILDER(unavailable, MakeUnavailableBuilder)

}

#endif
