#pragma once

#include <array>
#include <string>
#include <vector>

#include "todrt/modules.hpp"

namespace todrt {

struct DetectorOptions {
  std::string model = "SPAE-YOLOv8n";
  std::string builder = "auto";
  std::string preproc = "auto";
  std::string postproc = "auto";

  DeployConfig deploy;

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

  PreprocessOptions preproc_opts;
  PostprocessOptions postproc_opts;

  int num_classes() const { return deploy.num_classes; }
  int reg_max() const { return deploy.reg_max; }
  OutputLayout layout() const { return deploy.layout; }
  const std::vector<int>& strides() const { return deploy.strides; }

  static DetectorOptions FromJson(const json::Value& v);
  static DetectorOptions FromFile(const std::string& path);
  json::Value ToJson() const;

  BuildConfig ToBuildConfig() const;

  void ApplyPreset(const std::string& preset);
};

}
