// ort_factory.hpp —— ONNX Runtime 后端入口（仅在编译了 onnxruntime 时使用）
//
// 定位：**通用 CPU 后路**。典型用途是 AMD x86 主机（无 NVIDIA GPU、无 NPU）
// 以及作为 NPU/GPU 的对照基线 —— 同一份配置换个 builder 就能跑起来，
// 便于回答"加速器到底快多少、精度掉多少"。
#pragma once

#include <memory>
#include <string>

#include "todrt/factory.hpp"

namespace todrt {

/// 构造走 ONNX Runtime 的 Detector。仅当 TODRT_HAVE_ORT=1 时存在实现。
std::unique_ptr<Detector> CreateOrtDetector(const std::string& model_name,
                                            const DetectorOptions& opts,
                                            const ModelRecipe& recipe,
                                            const std::string& builder_name,
                                            const std::string& preproc_name,
                                            const std::string& postproc_name);

}  // namespace todrt
