#pragma once

#ifndef TODRT_HAVE_TENSORRT
#define TODRT_HAVE_TENSORRT 0
#endif

#include <memory>
#include <string>
#include <vector>

#include "todrt/modules.hpp"

namespace todrt {

class TrtEngine {
 public:
  TrtEngine();
  ~TrtEngine();
  TrtEngine(const TrtEngine&) = delete;
  TrtEngine& operator=(const TrtEngine&) = delete;

  struct PlanInfo {
    int64_t max_batch = 0;
    int nb_dla_cores = 0;
    int64_t dla_max_workspace = 0;
    bool dla_supported_build = true;
    std::string device_name;
    std::string trt_version;
    std::string compute_capability;
  };
  static PlanInfo Probe(const BuildConfig& cfg);
  static PlanInfo ProbeDlaLayers(const BuildConfig& cfg, std::vector<std::string>* gpu_layers);

  bool Build(const BuildConfig& cfg, std::string* err);
  bool Load(const std::string& engine_path, std::string* err);
  bool Serialize(const std::string& path, std::string* err) const;

  std::vector<int64_t> InputShape() const;
  std::vector<std::vector<int64_t>> OutputShapes() const;
  std::vector<DataType> OutputTypes() const;
  std::string InputName() const;
  std::vector<std::string> OutputNames() const;

  bool Infer(const void* input, size_t input_bytes, int batch, const std::vector<void*>& outputs,
             const std::vector<size_t>& output_bytes, std::string* err);

  double LastInferMs() const;

  int DlaLayerCount() const;
  int GpuLayerCount() const;
  void ClassifyLayers();
  std::vector<std::string> LayerInfo();

  void Release();
  bool valid() const;

 private:
  bool Finalize();

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}
