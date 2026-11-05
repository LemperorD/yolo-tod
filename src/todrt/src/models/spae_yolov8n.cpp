#include "todrt/factory.hpp"

namespace {

int64_t AnchorCountFor(int size, const std::vector<int>& strides) {
  int64_t total = 0;
  for (int s : strides) total += static_cast<int64_t>(size / s) * (size / s);
  return total;
}

}

TOD_RT_DEFINE_RECIPE(SPAE_YOLOv8n, SpaeYolov8nRecipe) {
  ::todrt::ModelRecipe r;

  r.builder = "yolov8-trt";
  r.preproc = "detect_letterbox";
  r.postproc = "detect_nms";

  r.layout = ::todrt::OutputLayout::kAnchorMajorDfl;
  r.decode.strides = {4, 8, 16, 32};
  r.decode.reg_max = 16;
  r.decode.num_classes = 10;
  r.decode.max_det = 300;
  r.decode.conf_threshold = 0.25f;
  r.postproc_opts.decode = r.decode;
  r.postproc_opts.iou_threshold = 0.45f;
  r.postproc_opts.max_det = 300;

  r.preproc_opts.input_width = 640;
  r.preproc_opts.input_height = 640;
  r.preproc_opts.pad_multiple = 32;
  r.preproc_opts.mode = ::todrt::ResizeMode::kLetterbox;

  r.hardware_preset = "orin";

  r.tune = [](::todrt::DetectorOptions& o) {
    const ::todrt::DeployConfig& d = o.deploy;
    if (d.strides.size() != 4) {
      ::todrt::log_warn(
          "SPAE-YOLOv8n 期望 4 个检测层（P2–P5，strides=4,8,16,32），当前配置为 " +
          std::to_string(d.strides.size()) +
          " 层。若确实只导出 P3–P5，请确认 strides 与导出图一致。");
    }
    const int64_t expect = AnchorCountFor(o.preproc_opts.input_width, d.strides);
    ::todrt::log_info("SPAE-YOLOv8n: 输入 " + std::to_string(o.preproc_opts.input_width) + "×" +
                      std::to_string(o.preproc_opts.input_height) + " → 预期 anchor=" +
                      std::to_string(expect));
  };

  return r;
}

namespace {
const bool g_spae_meta = []() {
  ::todrt::register_model_metadata(
      "SPAE_YOLOv8n",
      {"spae-yolov8n", "spae_yolov8n", "spae-yolov8", "spae_yolov8", "SPAE_YOLOv8n_int8"},
      "SPAE-YOLOv8 for Onboard Real-Time Perception: Lightweight Small UAV Detection "
      "from Air-to-Air Perspectives (Sensors 2026, DOI 10.3390/s26113424)；"
      "训练侧本库实现见 variants/SPAE-YOLOv8n/",
      "论文 CC BY 4.0；本实现 Apache-2.0",
      "Jetson Orin（DLA + FP16 推荐）/ 任意支持 TensorRT 10 的 GPU；FP16 或 INT8",
      "640x640 P2-P5；anchor=34000；NMS 在 CPU（可换插件版）", __FILE__, __LINE__);
  return true;
}();
}
