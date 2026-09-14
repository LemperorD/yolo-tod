// factories.hpp —— 内部构造助手（供 TensorRT 后端按部署配置实例化模块）
//
// 为什么需要它：公开 API（factory.hpp）只暴露「按名字取默认实现」，
// 而实际部署时前/后处理需要带参数（输入尺寸、阈值、layout…）。
// 与其给接口加一个可变的 Configure()（回去破坏"实现无状态"的约定），
// 不如让构造即注入：`make_xxx(options)`。
//
// 这些函数**不进入公共 API**：变体与业务代码只该用 `Detector::Create*`。
#pragma once

#include <memory>

#include "todrt/modules.hpp"

namespace todrt {

/// 按配置构造检测前处理器（letterbox / stretch / integer-scale）。
std::unique_ptr<IPreprocessor> make_detect_preprocessor(const PreprocessOptions& o);

/// 按配置构造 NMS 后处理器（layout / 阈值 / soft-NMS 全由 options 决定）。
std::unique_ptr<IPostprocessor> make_nms_postprocessor(const PostprocessOptions& o);

}  // namespace todrt
