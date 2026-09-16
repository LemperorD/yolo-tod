// rknn_factory.hpp —— Rockchip RKNN 后端入口（仅在编译了 librknnrt 时使用）
//
// 该头文件**不包含 rknn_api.h**，因此无 NPU 的机器上也能引用它做装配自检；
// 实现在 src/backend/engine_rknn.cpp（受 TODRT_HAVE_RKNN 保护）。
#pragma once

#include <memory>
#include <string>

#include "todrt/factory.hpp"

namespace todrt {

/// 构造走 RKNN（RK3588 NPU）的 Detector。仅当 TODRT_HAVE_RKNN=1 时存在实现。
std::unique_ptr<Detector> CreateRknnDetector(const std::string& model_name,
                                             const DetectorOptions& opts,
                                             const ModelRecipe& recipe,
                                             const std::string& builder_name,
                                             const std::string& preproc_name,
                                             const std::string& postproc_name);

}  // namespace todrt
