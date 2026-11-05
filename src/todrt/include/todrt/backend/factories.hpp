#pragma once

#include <array>
#include <memory>

#include "todrt/detector_options.hpp"
#include "todrt/modules.hpp"

namespace todrt {

std::unique_ptr<IPreprocessor> make_detect_preprocessor(const PreprocessOptions& o);

std::unique_ptr<IPostprocessor> make_nms_postprocessor(const PostprocessOptions& o);

struct EngineInputSpec {
  int width = 640;
  int height = 640;
  PreprocOutput output = PreprocOutput::kFloat32;
  TensorLayout layout = TensorLayout::kNchw;
  bool dynamic_shape = false;
  std::array<int, 2> dynamic_min{640, 640};
  std::array<int, 2> dynamic_max{640, 640};
};

PreprocessOptions resolve_preprocess_options(const DetectorOptions& opts,
                                             const EngineInputSpec& spec);

PostprocessOptions resolve_postprocess_options(const DetectorOptions& opts);

}
