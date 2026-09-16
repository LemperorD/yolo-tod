// openvino_factory.hpp —— Intel OpenVINO 后端入口（仅在编译了 OpenVINO 时使用）
//
// 定位：与 ONNX Runtime 互补的通用 CPU 后路，且能吃到 Intel iGPU/NPU。
// 在 AMD x86 上只用 CPU 设备（OpenVINO 的 CPU 插件对 x86 有 AVX2/AVX512 深度优化），
// 在 Intel 机器上可以切 GPU/NPU —— 同一份配置换 `ov_device` 即可。
#pragma once

#include <memory>
#include <string>

#include "todrt/factory.hpp"

namespace todrt {

/// 构造走 OpenVINO 的 Detector。仅当 TODRT_HAVE_OPENVINO=1 时存在实现。
std::unique_ptr<Detector> CreateOpenVinoDetector(const std::string& model_name,
                                                 const DetectorOptions& opts,
                                                 const ModelRecipe& recipe,
                                                 const std::string& builder_name,
                                                 const std::string& preproc_name,
                                                 const std::string& postproc_name);

}  // namespace todrt
