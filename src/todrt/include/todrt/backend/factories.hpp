// factories.hpp —— 内部构造助手（供 TensorRT 后端按部署配置实例化模块）
//
// 为什么需要它：公开 API（factory.hpp）只暴露「按名字取默认实现」，
// 而实际部署时前/后处理需要带参数（输入尺寸、阈值、layout…）。
// 与其给接口加一个可变的 Configure()（回去破坏"实现无状态"的约定），
// 不如让构造即注入：`make_xxx(options)`。
//
// 这些函数**不进入公共 API**：变体与业务代码只该用 `Detector::Create*`。
#pragma once

#include <array>
#include <memory>

#include "todrt/detector_options.hpp"
#include "todrt/modules.hpp"

namespace todrt {

/// 按配置构造检测前处理器（letterbox / stretch / integer-scale；float 或 uint8）。
std::unique_ptr<IPreprocessor> make_detect_preprocessor(const PreprocessOptions& o);

/// 按配置构造 NMS 后处理器（layout / 阈值 / soft-NMS 全由 options 决定）。
std::unique_ptr<IPostprocessor> make_nms_postprocessor(const PostprocessOptions& o);

// ------------------------------------------------------------------ 后端共享助手
//
// 四个后端（TensorRT / RKNN / ONNX Runtime / OpenVINO）的装配流程是同一套：
// 定契约 → 建引擎 → 配前处理 → 配后处理。下面这几个函数把这段收敛成一份，
// 避免四份实现各写各的（那正是本库最想避免的"接线散落各处"）。

/// 一次推理的输入契约：由**所选引擎**决定，调用方（DetectorOptions）只能确认或纠正。
struct EngineInputSpec {
  int width = 640;
  int height = 640;
  PreprocOutput output = PreprocOutput::kFloat32;
  TensorLayout layout = TensorLayout::kNchw;
  bool dynamic_shape = false;
  std::array<int, 2> dynamic_min{640, 640};
  std::array<int, 2> dynamic_max{640, 640};
};

/// 把「部署配置 + 引擎契约」合成最终的前处理选项。
/// 优先级：引擎的输出类型/布局（RKNN 只能是 uint8 NHWC）> 配置里显式写明的值 > 默认。
/// 冲突时打 warn 并采用引擎侧，因为写错会让框全乱。
PreprocessOptions resolve_preprocess_options(const DetectorOptions& opts,
                                             const EngineInputSpec& spec);

/// 把「部署配置」里的结构信息注入工具链共享的后处理选项，并做一致性校验。
/// 返回规范化后的 PostprocessOptions（保证 decode 与 NMS 用的是同一份阈值语义）。
PostprocessOptions resolve_postprocess_options(const DetectorOptions& opts);

}  // namespace todrt
