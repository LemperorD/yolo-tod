#include "todrt/factory.hpp"

TOD_RT_DEFINE_RECIPE(YOLOv8n_baseline, Yolov8nBaselineRecipe) {
  ::todrt::ModelRecipe r;
  r.builder = "yolov8-trt";
  r.preproc = "detect_letterbox";
  r.postproc = "detect_nms";

  r.layout = ::todrt::OutputLayout::kAnchorMajorDfl;
  r.decode.strides = {8, 16, 32};
  r.decode.reg_max = 16;
  r.decode.num_classes = 80;
  r.decode.max_det = 300;
  r.postproc_opts.decode = r.decode;
  r.preproc_opts = ::todrt::PreprocessOptions::ForInput(640, 640);
  r.hardware_preset = "";

  r.tune = [](::todrt::DetectorOptions& o) {
    const int w = o.preproc_opts.input_width;
    int64_t total = 0;
    for (int s : o.deploy.strides) total += static_cast<int64_t>(w / s) * (w / s);
    ::todrt::log_info("YOLOv8 基线: 输入 " + std::to_string(w) + " → 预期 anchor=" +
                      std::to_string(total) + "（官方 640 配置为 8400）");
  };
  return r;
}

namespace {
const bool g_yolov8n_meta = []() {
  ::todrt::register_model_metadata(
      "YOLOv8n_baseline", {"yolov8n", "yolov8n_baseline", "YOLOv8n"},
      "Ultralytics YOLOv8（AGPL-3.0）官方 P3-P5 检测头，作为无 P2 的干净基线",
      "AGPL-3.0（ultralytics）",
      "任意支持 TensorRT 10 的 GPU / Jetson Orin；FP16 或 INT8",
      "640x640 P3-P5；anchor=8400", __FILE__, __LINE__);
  return true;
}();
}
