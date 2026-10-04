// trt_factory.hpp —— TensorRT 后端入口（仅在编译了 TensorRT 时使用）
//
// 该头文件**不包含任何 TensorRT 头**，因此可以在无 TensorRT 的机器上被引用做
// 装配自检；实现在 src/backend/engine_trt.cpp（受 TODRT_HAVE_TENSORRT 保护）。
//
// 与其他三个后端（RKNN / ORT / OpenVINO）保持同一形状：**只暴露一个
// CreateXxxDetector()**；engine/runner/builder 类都是该 .cpp 的私有实现细节。
#pragma once

#include <memory>
#include <string>

#include "todrt/factory.hpp"

namespace todrt {

/// 构造走 TensorRT 的 Detector。仅当 TODRT_HAVE_TENSORRT=1 时存在实现。
std::unique_ptr<Detector> CreateTrtDetector(const std::string& model_name,
                                            const DetectorOptions& opts,
                                            const ModelRecipe& recipe,
                                            const std::string& builder_name,
                                            const std::string& preproc_name,
                                            const std::string& postproc_name);

}  // namespace todrt
