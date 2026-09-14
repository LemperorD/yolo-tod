// detector_options.hpp —— 「一次部署」的完整声明
//
// 这是调用方**唯一**需要拼装的东西：
//   * 结构信息（nc / reg_max / strides / layout）来自 DeployConfig（Python 生成的 JSON）；
//   * 加上运行期选择（工厂链路、硬件、引擎路径、阈值）。
//
// 于是有两种典型用法：
//
//   1. 配置文件驱动（推荐，实机上就用这一条）：
//        auto det = todrt::Detector::CreateFromFile("configs/deploy/spae-yolov8n-orin.json");
//
//   2. 代码里拼装（调试/服务里动态改阈值）：
//        todrt::DetectorOptions o = todrt::DetectorOptions::FromFile(path);
//        o.precision = todrt::Precision::kINT8;
//        o.postproc_opts.decode.conf_threshold = 0.15f;   // 小目标宁可多召回
//        auto det = todrt::Detector::Create("SPAE-YOLOv8n", o);
#pragma once

#include <array>
#include <string>
#include <vector>

#include "todrt/modules.hpp"

namespace todrt {

struct DetectorOptions {
  // ---- 工厂查找键（与 Python 侧 variant.yaml 的 id / 注册名对齐）----
  std::string model = "SPAE-YOLOv8n";
  std::string builder = "auto";  ///< auto = 由变体配方决定
  std::string preproc = "auto";
  std::string postproc = "auto";

  // ---- 结构信息（来自 DeployConfig）----
  DeployConfig deploy;

  // ---- 运行期硬件选项 ----
  Precision precision = Precision::kFP16;
  Device device = Device::kAuto;
  int dla_core = 0;
  int dla_memory_limit_mb = 512;
  bool allow_gpu_fallback = true;
  bool dynamic_shape = false;
  bool dynamic_batch = false;
  int max_batch = 8;
  std::array<int, 2> shape_min{640, 640};
  std::array<int, 2> shape_opt{640, 640};
  std::array<int, 2> shape_max{640, 640};
  size_t workspace_mb = 1024;
  int device_id = 0;
  bool cuda_graphs = false;
  bool verbose = false;
  int bench_warmup = 10;
  int bench_iters = 100;

  // ---- 前后处理（默认由 DeployConfig + 变体配方补全）----
  PreprocessOptions preproc_opts;
  PostprocessOptions postproc_opts;

  // ---- 便捷转发：结构信息 ----
  int num_classes() const { return deploy.num_classes; }
  int reg_max() const { return deploy.reg_max; }
  OutputLayout layout() const { return deploy.layout; }
  const std::vector<int>& strides() const { return deploy.strides; }

  /// 读部署配置 JSON（含 runtime 覆盖段）。
  static DetectorOptions FromJson(const json::Value& v);
  static DetectorOptions FromFile(const std::string& path);
  json::Value ToJson() const;

  /// 把运行期选择合并成一份 BuildConfig（后端构建引擎时用）。
  BuildConfig ToBuildConfig() const;

  /// 按目标硬件套一层预设（"orin" / "dgp" / "x86" / "fp32"），再被显式字段覆盖。
  void ApplyPreset(const std::string& preset);
};

}  // namespace todrt
