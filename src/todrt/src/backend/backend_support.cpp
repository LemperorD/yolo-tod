#include <array>
#include <string>

#include "todrt/backend/factories.hpp"
#include "todrt/factory.hpp"
#include "todrt/modules.hpp"

namespace todrt {

namespace {

std::string layout_name(TensorLayout l) { return to_string(l); }
std::string output_name(PreprocOutput o) {
  return o == PreprocOutput::kUint8Raw ? "uint8" : "float32";
}

}

PreprocessOptions resolve_preprocess_options(const DetectorOptions& opts,
                                             const EngineInputSpec& spec) {
  // 引擎声明决定输入类型与布局，配置冲突时采用引擎契约并记录警告。
  PreprocessOptions po = opts.preproc_opts;
  po.output = spec.output;
  po.layout = spec.layout;
  po.input_width = spec.width;
  po.input_height = spec.height;

  const PreprocessOptions& cfg = opts.preproc_opts;
  if (cfg.output != spec.output) {
    log_warn(std::string("前处理输出类型冲突：部署配置要求 ") + output_name(cfg.output) +
             "，但所选引擎需要 " + output_name(spec.output) + "。已采用引擎侧（" +
             output_name(spec.output) +
             "）。若你确实想用配置里的设置，请换对应的后端或修正配置。");
    po.output = spec.output;
  }
  if (cfg.layout != spec.layout) {
    log_warn(std::string("前处理布局冲突：部署配置要求 ") + layout_name(cfg.layout) +
             "，但所选引擎需要 " + layout_name(spec.layout) + "。已采用引擎侧（" +
             layout_name(spec.layout) + "）。");
    po.layout = spec.layout;
  }

  if (cfg.input_width > 0 && cfg.input_height > 0 &&
      cfg.input_width != opts.deploy.input[0]) {
    po.input_width = cfg.input_width;
    po.input_height = cfg.input_height;
  }
  return po;
}

PostprocessOptions resolve_postprocess_options(const DetectorOptions& opts) {
  PostprocessOptions pp = opts.postproc_opts;

  // 类别数、DFL 与 stride 来自导出配置，运行参数只覆盖阈值等推理选项。
  pp.decode.num_classes = opts.deploy.num_classes;
  pp.decode.reg_max = opts.deploy.reg_max;
  pp.decode.layout = opts.deploy.layout;
  pp.decode.strides = opts.deploy.strides;
  pp.decode.level_channels = opts.deploy.level_channels;
  pp.decode.input_width = opts.preproc_opts.input_width;
  pp.decode.input_height = opts.preproc_opts.input_height;

  if (pp.max_det <= 0) pp.max_det = pp.decode.max_det > 0 ? pp.decode.max_det : 300;
  if (pp.decode.max_det <= 0) pp.decode.max_det = pp.max_det;

  if (pp.decode.layout == OutputLayout::kPluginNms) {
    pp.max_det = std::max(pp.max_det, 1);
  }
  return pp;
}

}
