// engine_trt.hpp —— TensorRT 后端（编译期可选：未找到 TensorRT 时本文件不参与编译）
//
// 支持范围：
//   * TensorRT 10.x（JetPack 6.x / Orin、以及当前 dGPU 主力）——主路径，用 enqueueV3
//   * TensorRT 8.5–8.6 ——保留兼容分支（enqueueV2 + binding 索引 API）
//   * TensorRT 11.x ——DLA 已被上游移除，本文件提供显式报错而非静默忽略
//
// 硬件加速相关的开关都集中在这里：精度（FP16/INT8）、DLA（core / 回退 / 内存池）、
// CUDA Graph、异步 stream。上层只看到 DetectorOptions。
#pragma once

#ifndef TODRT_HAVE_TENSORRT
#define TODRT_HAVE_TENSORRT 0
#endif

#include <memory>
#include <string>
#include <vector>

#include "todrt/modules.hpp"

namespace todrt {

/// 引擎构建与执行（PIMPL：头文件不暴露任何 TensorRT 类型，便于在无 TRT 的机器上
/// 也能 include 这个头做工厂装配自检）。
class TrtEngine {
 public:
  TrtEngine();
  ~TrtEngine();
  TrtEngine(const TrtEngine&) = delete;
  TrtEngine& operator=(const TrtEngine&) = delete;

  /// 读取构建期信息（输入尺寸、DLA 能力等），不创建引擎。
  struct PlanInfo {
    int64_t max_batch = 0;       ///< builder->getMaxDLABatchSize / maxBatchSize（0 = 未知）
    int nb_dla_cores = 0;        ///< 当前设备的 DLA core 数（0 = 无 DLA）
    int64_t dla_max_workspace = 0;
    bool dla_supported_build = true;  ///< 本 TensorRT 版本是否还支持 DLA
    std::string device_name;
    std::string trt_version;
    std::string compute_capability;
  };
  static PlanInfo Probe(const BuildConfig& cfg);
  /// 不建引擎、只问"这个 ONNX 在 DLA 上能跑多少层"（诊断用，构建耗时较长）。
  static PlanInfo ProbeDlaLayers(const BuildConfig& cfg, std::vector<std::string>* gpu_layers);

  bool Build(const BuildConfig& cfg, std::string* err);
  bool Load(const std::string& engine_path, std::string* err);
  bool Serialize(const std::string& path, std::string* err) const;

  std::vector<int64_t> InputShape() const;
  std::vector<std::vector<int64_t>> OutputShapes() const;
  std::vector<DataType> OutputTypes() const;
  std::string InputName() const;
  std::vector<std::string> OutputNames() const;

  /// 执行一次推理。input 为主机侧 NCHW float / half；outputs 为主机侧缓冲（由调用方
  /// 按其 shape×dtype 分配）。实现内部使用固定 stream + 可选 CUDA Graph。
  bool Infer(const void* input, size_t input_bytes, int batch, const std::vector<void*>& outputs,
             const std::vector<size_t>& output_bytes, std::string* err);

  /// 最近一次 Infer 的实测耗时（CUDA event，毫秒）。
  double LastInferMs() const;

  /// 是否启用了 DLA，以及有多少层退回了 GPU（构建期统计）。
  int DlaLayerCount() const;
  int GpuLayerCount() const;
  /// 统计每层能否跑在 DLA 上（构建期调用；ONNX 解析后、构建引擎前）。
  void ClassifyLayers();
  /// 引擎逐层信息（JSON 行），排查"为什么没上 DLA / 这层是什么精度"时用。
  std::vector<std::string> LayerInfo();

  void Release();
  bool valid() const;

 private:
  /// 反序列化之后建立张量表、分配设备缓冲、创建 stream/event。
  bool Finalize();

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// 由 backend/engine_trt.cpp 提供：构造真正跑 TensorRT 的 Detector。
std::unique_ptr<Detector> CreateTrtDetector(const std::string& model_name, const DetectorOptions& opts,
                                            const ModelRecipe& recipe,
                                            const std::string& builder_name,
                                            const std::string& preproc_name,
                                            const std::string& postproc_name);

}  // namespace todrt
