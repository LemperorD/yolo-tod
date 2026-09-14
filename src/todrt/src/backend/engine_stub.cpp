// engine_stub.cpp —— 未编译 TensorRT 时的后端占位
//
// 存在的意义：**没有 GPU / 没有 TensorRT 的机器上，工厂装配自检依然可用**。
//   * `Detector::Create()` 会给出带修复建议的明确错误，而不是链接失败或段错误；
//   * `PlanAssembly()` / `--dry-run` / `--list-modules` 完全不依赖这个文件；
//   * 于是 CI / 开发机可以只跑 CPU 部分（见 tests/cpp_smoke.cpp）。
#include "todrt/factory.hpp"

#if !TODRT_HAVE_TENSORRT

namespace todrt {
namespace {

class UnavailableBuilder : public IModelBuilder {
 public:
  const std::string& name() const override { return name_; }
  void Build(const BuildConfig&) override {
    throw TritError(
        "本次构建没有启用 TensorRT（TODRT_HAVE_TENSORRT=0）。\n"
        "  重新配置：cmake -DTODRT_WITH_TENSORRT=ON -DTensorRT_ROOT=<TRT 安装路径> ..\n"
        "  Jetson/Orin 上 TensorRT 通常已随 JetPack 提供（/usr/include/aarch64-linux-gnu）。");
  }
  bool Available(std::string* reason) const override {
    if (reason) {
      *reason =
          "未启用 TensorRT（TODRT_HAVE_TENSORRT=0）。装配自检/工厂清单可用，"
          "但无法构建或加载 engine。";
    }
    return false;
  }
  std::vector<int64_t> input_shape() const override { return {}; }
  std::vector<std::vector<int64_t>> output_shapes() const override { return {}; }
  std::vector<DataType> output_types() const override { return {}; }
  std::string input_name() const override { return {}; }
  std::vector<std::string> output_names() const override { return {}; }
  bool serialize(const std::string&, std::string* err) override {
    if (err) *err = "未启用 TensorRT";
    return false;
  }
  bool Run(const RunIO&, std::string* err) override {
    if (err) *err = "未启用 TensorRT";
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

}  // namespace

TOD_RT_REGISTER_BUILDER(yolov8_trt, MakeUnavailableBuilder)
TOD_RT_REGISTER_BUILDER(trt, MakeUnavailableBuilder)
TOD_RT_REGISTER_BUILDER(auto, MakeUnavailableBuilder)

std::unique_ptr<Detector> CreateDeviceDetector(const std::string&, const DetectorOptions&,
                                               const ModelRecipe&, const std::string&,
                                               const std::string&, const std::string&) {
  throw TritError(
      "本构建中没有 TensorRT 后端，无法创建 Detector。\n"
      "  实机部署（Linux/Jetson）请用 -DTODRT_WITH_TENSORRT=ON 重新构建；\n"
      "  只想验证工厂装配与配置，请用 apps/todrt_cli --dry-run。");
}

bool backend_available(std::string* reason) {
  if (reason) *reason = "TODRT_HAVE_TENSORRT=0（本次构建未启用 TensorRT）";
  return false;
}

}  // namespace todrt

#endif  // !TODRT_HAVE_TENSORRT
