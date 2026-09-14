// spae_yolov8n.cpp —— 变体配方：SPAE-YOLOv8n
//
// 【这就是"新增一个可部署变体"的全部代码】。它不含任何推理逻辑，只回答四件事：
//   1. 用哪个 builder / 前后处理（工厂链路）；
//   2. 解码需要知道的网络结构（层数、步长、输出排布）；
//   3. 推荐硬件预设与默认阈值；
//   4. 追溯信息（论文、许可证、训练侧对应物）。
//
// 结构信息的**权威来源**始终是 Python 侧导出的部署配置（DeployConfig）。
// 这里的默认值只在没有配置文件时兜底，且会被配置覆盖 —— 见 factory.cpp 的
// merge_recipe_defaults()。
//
// 训练侧对应物：variants/SPAE-YOLOv8n/（recipe.py + variant.yaml + paper-notes.md）
//   EP1 ADown（主干 stride-2 下采样）
//   EP5 Efficient_UAVDet（分组卷积分支 stem，g = x/16）+ P2 注入
//   EP7 SIoU（只影响训练，不改变推理图）
#include "todrt/factory.hpp"

namespace {

/// SPAE-YOLOv8n 的 P2–P5 四层检测头：
///   640×640 → anchor = 160² + 80² + 40² + 20² = 25600+6400+1600+400 = 34000
///   ⚠️ 与官方 YOLOv8n（仅 P3–P5，anchor = 8400）**不同**：多出的 P2 分支正是
///      SPAE 的主要涨点来源（论文消融 +7.5pp），也是解码时必须写对 strides 的原因。
int64_t AnchorCountFor(int size, const std::vector<int>& strides) {
  int64_t total = 0;
  for (int s : strides) total += static_cast<int64_t>(size / s) * (size / s);
  return total;
}

}  // namespace

TOD_RT_DEFINE_RECIPE(SPAE_YOLOv8n, SpaeYolov8nRecipe) {
  ::todrt::ModelRecipe r;

  // ---- 工厂链路 ----
  r.builder = "yolov8-trt";  // backend/builder_trt.cpp 注册
  r.preproc = "detect_letterbox";
  r.postproc = "detect_nms";

  // ---- 解码所需的网络结构（P2–P5）----
  r.layout = ::todrt::OutputLayout::kAnchorMajorDfl;  // ultralytics 官方 Detect 导出
  r.decode.strides = {4, 8, 16, 32};                  // P2, P3, P4, P5
  r.decode.reg_max = 16;
  r.decode.num_classes = 10;  // VisDrone2019-DET
  r.decode.max_det = 300;
  r.decode.conf_threshold = 0.25f;
  r.postproc_opts.decode = r.decode;
  r.postproc_opts.iou_threshold = 0.45f;
  r.postproc_opts.max_det = 300;

  // ---- 前处理默认值（会被部署配置的 input 覆盖）----
  r.preproc_opts.input_width = 640;
  r.preproc_opts.input_height = 640;
  r.preproc_opts.pad_multiple = 32;  // Ultralytics auto=True 时为 32
  r.preproc_opts.mode = ::todrt::ResizeMode::kLetterbox;

  // ---- 推荐硬件：Orin 上优先 DLA + FP16 ----
  r.hardware_preset = "orin";

  // 校验钩子：把"配置写错但程序不报错"的那类问题提前到启动阶段
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

// 别名与元数据一起登记（来源 / 许可证 / 硬件要求 / 成本）——**强制**，与 Python 侧
// registry 同一原则：没有来源与许可证的模块不允许入库。
//
// 这里用普通函数而不是宏：别名列表里的逗号与花括号会让宏在部分编译器上展开出错，
// 而"许可证与来源"这种必须写对的东西不值得冒预处理器行为的风险。
// register_model_metadata 会把每个别名一并绑到同一个配方上（含连字符写法）。
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
}  // namespace
