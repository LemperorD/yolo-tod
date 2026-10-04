// simple_detector.cpp —— RunnerDetector 的实现（后端无关）
//
// 这里只有一件事：把 IEngineRunner 的输出整流成工具链解码器能吃的形状，
// 然后交给后处理器与坐标反变换。异步/计时/生命周期都在 AsyncDetectorBase 里。
#include "todrt/backend/simple_detector.hpp"

#include <algorithm>
#include <sstream>

namespace todrt {

std::string RunnerDetector::Describe() const {
  const DetectorOptions& o = options();
  std::ostringstream oss;
  int w = 0, h = 0;
  input_shape(&w, &h);
  oss << "模型        : " << model_name() << "\n";
  oss << "引擎        : " << (runner_ ? runner_->name() : "-") << "\n";
  oss << "工厂链      : preproc=" << (preprocessor() ? preprocessor()->name() : "-")
      << " postproc=" << (post_ ? post_->name() : "-") << "\n";
  oss << "硬件        : device=" << to_string(o.device) << " precision=" << to_string(o.precision)
      << " core=" << o.deploy.build.accelerator_core << "\n";
  if (preprocessor()) {
    const PreprocessOptions& po = preprocessor()->options();
    oss << "输入        : " << w << "×" << h << "  " << to_string(po.output_dtype()) << " "
        << to_string(po.layout) << "  pad_multiple=" << po.pad_multiple << "\n";
  }
  if (post_) {
    const PostprocessOptions& pp = post_->options();
    oss << "解码        : layout=" << to_string(pp.decode.layout)
        << " nc=" << pp.decode.num_classes << " reg_max=" << pp.decode.reg_max << " strides=";
    for (size_t i = 0; i < pp.decode.strides.size(); ++i) {
      oss << (i ? "," : "") << pp.decode.strides[i];
    }
    oss << "\n";
    oss << "阈值        : conf=" << pp.decode.conf_threshold << " iou=" << pp.iou_threshold
        << " max_det=" << pp.max_det << " nms="
        << (pp.nms == NmsKind::kSoft ? "soft" : "hard") << "\n";
  }
  if (runner_) {
    const std::string info = runner_->engine_info();
    if (!info.empty()) oss << info;
    const std::vector<std::vector<int64_t>> shapes = runner_->output_shapes();
    const std::vector<DataType> types = runner_->output_types();
    for (size_t k = 0; k < shapes.size(); ++k) {
      oss << "引擎输出    : [";
      for (size_t i = 0; i < shapes[k].size(); ++i) oss << (i ? "," : "") << shapes[k][i];
      oss << "] " << to_string(k < types.size() ? types[k] : DataType::kF32) << "\n";
    }
  }
  return oss.str();
}

std::vector<Detection> RunnerDetector::RunBatch(const PreprocessResult& pp, double* preprocess_ms,
                                                double* infer_ms, double* postprocess_ms) {
  if (!runner_) throw TritError("Detector 未配置引擎");
  if (!preprocessor() || !post_) throw TritError("Detector 未配置前/后处理器");
  if (preprocess_ms) *preprocess_ms = 0.0;  // 前处理在上层完成

  // 1) 准备主机侧输出缓冲（后端负责把结果写到这里，统一 float32）
  const std::vector<std::vector<int64_t>> shapes = runner_->output_shapes();
  const std::vector<DataType> types = runner_->output_types();
  if (shapes.empty()) throw TritError("引擎没有输出张量");

  host_outputs_.resize(shapes.size());
  std::vector<float*> ptrs(shapes.size(), nullptr);
  std::vector<size_t> elems(shapes.size(), 0);
  for (size_t k = 0; k < shapes.size(); ++k) {
    size_t n = 1;
    for (int64_t d : shapes[k]) n *= static_cast<size_t>(d > 0 ? d : 0);
    if (n == 0) {
      // 动态 shape 且尚未确定：后端应在 Run 里重新查询；这里给不出尺寸就报错，
      // 而不是静默跳过（静默跳过会表现为"一个目标都检测不到"，更难查）。
      throw TritError("输出 " + std::to_string(k) +
                      " 的形状含动态维且未确定，无法分配缓冲（动态 shape 请把 profile 设全）");
    }
    elems[k] = n;
    if (host_outputs_[k].size() < n) host_outputs_[k].resize(n);
    ptrs[k] = host_outputs_[k].data();
  }

  // 2) 跑引擎
  runner_->Run(pp.bytes.data(), pp.bytes.size(), pp.batch, ptrs, elems);
  if (infer_ms) *infer_ms = runner_->last_infer_ms();

  // 3) 解码 + NMS（后处理器只吃 float32；后端必须在 Run 内完成类型转换）
  last_.clear();
  last_.reserve(shapes.size());
  for (size_t k = 0; k < shapes.size(); ++k) {
    TensorView v;
    v.data = host_outputs_[k].data();
    v.dtype = DataType::kF32;
    if (types.size() > k && types[k] != DataType::kF32) {
      throw TritError("输出 " + std::to_string(k) + " 声明为 " + to_string(types[k]) +
                      "，但通用外壳只接受 float32（后端应在 Run 内完成反量化/类型转换）");
    }
    v.shape = shapes[k];
    last_.push_back(v);
  }
  std::vector<std::vector<Detection>> per_image = post_->Run(last_, pp);
  per_image = preprocessor()->ToSourceCoords(std::move(per_image), pp);
  if (postprocess_ms) *postprocess_ms = 0.0;  // 由 Bench 的 wall-clock 覆盖
  if (per_image.empty()) return std::vector<Detection>();
  return std::move(per_image.front());
}

}  // namespace todrt
