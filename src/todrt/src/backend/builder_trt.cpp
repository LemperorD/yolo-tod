// builder_trt.cpp —— TensorRT 引擎构建器（IModelBuilder 的 TensorRT 实现）
//
// 这一层刻意做得很薄：把 BuildConfig 转成 TrtEngine 的构建/加载调用，并把
// 「引擎的 IO 形状」报给上层。构建细节都在 engine_trt.cpp。
//
// 它同时也是 `--dry-run` / PlanAssembly 能给出「[可用] TensorRT x.y / DLA core=2」
// 这类信息的来源，因此要在**没有引擎**的情况下也能回答 Available()。
#include <cuda_runtime_api.h>

#include <memory>
#include <sstream>
#include <string>

#include "todrt/backend/engine_trt.hpp"
#include "todrt/backend/trt_factory.hpp"

namespace todrt {

const std::string& TrtModelBuilder::name() const {
  static const std::string kName = "yolov8-trt";
  return kName;
}

void TrtModelBuilder::Build(const BuildConfig& cfg) {
  Release();
  engine_ = std::make_shared<TrtEngine>();
  std::string err;
  if (!cfg.engine_path.empty()) {
    if (!engine_->Load(cfg.engine_path, &err)) throw TritError(err);
  } else {
    if (!engine_->Build(cfg, &err)) throw TritError(err);
  }
}

bool TrtModelBuilder::Available(std::string* reason) const {
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) {
    if (reason) *reason = "无法访问 CUDA 设备（无 GPU 或驱动未加载）";
    return false;
  }
  const TrtEngine::PlanInfo info = TrtEngine::Probe(BuildConfig{});
  if (reason) {
    std::ostringstream oss;
    oss << "TensorRT " << info.trt_version;
    if (!info.device_name.empty()) oss << " / " << info.device_name;
    oss << " / DLA core=" << info.nb_dla_cores;
    if (!info.dla_supported_build) oss << "（该版本已移除 DLA）";
    *reason = oss.str();
  }
  return true;
}

bool TrtModelBuilder::Run(const RunIO& io, std::string* err) {
  if (!engine_) {
    if (err) *err = "引擎未构建";
    return false;
  }
  return engine_->Infer(io.input, io.input_bytes, io.batch, io.outputs, io.output_bytes, err);
}

std::vector<int64_t> TrtModelBuilder::input_shape() const {
  return engine_ ? engine_->InputShape() : std::vector<int64_t>{};
}

std::vector<std::vector<int64_t>> TrtModelBuilder::output_shapes() const {
  return engine_ ? engine_->OutputShapes() : std::vector<std::vector<int64_t>>{};
}

std::vector<DataType> TrtModelBuilder::output_types() const {
  return engine_ ? engine_->OutputTypes() : std::vector<DataType>{};
}

std::string TrtModelBuilder::input_name() const {
  return engine_ ? engine_->InputName() : std::string();
}

std::vector<std::string> TrtModelBuilder::output_names() const {
  return engine_ ? engine_->OutputNames() : std::vector<std::string>{};
}

bool TrtModelBuilder::serialize(const std::string& path, std::string* err) {
  if (!engine_) {
    if (err) *err = "引擎未构建，无法序列化";
    return false;
  }
  return engine_->Serialize(path, err);
}

double TrtModelBuilder::last_infer_ms() const {
  return engine_ ? engine_->LastInferMs() : 0.0;
}

void TrtModelBuilder::Release() { engine_.reset(); }

namespace {
std::unique_ptr<IModelBuilder> MakeTrtBuilder() {
  return std::unique_ptr<IModelBuilder>(new TrtModelBuilder());
}
}  // namespace

TOD_RT_REGISTER_BUILDER(yolov8_trt, MakeTrtBuilder)
TOD_RT_REGISTER_BUILDER(trt, MakeTrtBuilder)

std::unique_ptr<IModelBuilder> make_trt_builder() { return MakeTrtBuilder(); }

}  // namespace todrt
