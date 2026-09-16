// backend_support.cpp —— 四个后端共享的装配助手
//
// 为什么要这一层：TensorRT / RKNN / ONNX Runtime / OpenVINO 的装配流程是同一套
// （定契约 → 建引擎 → 配前处理 → 配后处理）。若让四个后端各写一遍，就会出现
// "TRT 用 conf 0.25、RKNN 用 0.2" 这种不可追溯的错配 —— 正是本库要避免的。
//
// 本文件**不依赖任何推理框架**，可在纯 CPU 工具链下编译与自检。
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

}  // namespace

PreprocessOptions resolve_preprocess_options(const DetectorOptions& opts,
                                             const EngineInputSpec& spec) {
  // 1) 以引擎契约为基准（引擎能吃什么就吃什么，这一层不可协商）
  PreprocessOptions po = opts.preproc_opts;
  po.output = spec.output;
  po.layout = spec.layout;
  po.input_width = spec.width;
  po.input_height = spec.height;

  // 2) 配置里显式写明的值可以纠正尺寸等，但**输出类型/布局的冲突要报出来**
  const PreprocessOptions& cfg = opts.preproc_opts;
  if (cfg.output != spec.output) {
    // 唯一的例外：配置里明确要求 uint8 而引擎声明 float（或反之）——这几乎一定是
    // 配置写错（例如把 RKNN 的 uint8 配置套到 TensorRT 上），必须让用户看见。
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

  // 3) 尺寸：配置显式改过就用配置的（动态 shape 场景下用户可能想换分辨率）
  if (cfg.input_width > 0 && cfg.input_height > 0 &&
      cfg.input_width != opts.deploy.input[0]) {
    po.input_width = cfg.input_width;
    po.input_height = cfg.input_height;
  }
  return po;
}

PostprocessOptions resolve_postprocess_options(const DetectorOptions& opts) {
  PostprocessOptions pp = opts.postproc_opts;

  // 结构信息**永远**以部署配置为准：这三项写错不会报错，只会给出全错的框，
  // 所以不给"代码里硬编码"留任何机会。
  pp.decode.num_classes = opts.deploy.num_classes;
  pp.decode.reg_max = opts.deploy.reg_max;
  pp.decode.layout = opts.deploy.layout;
  pp.decode.strides = opts.deploy.strides;
  pp.decode.level_channels = opts.deploy.level_channels;
  pp.decode.input_width = opts.preproc_opts.input_width;
  pp.decode.input_height = opts.preproc_opts.input_height;

  // 阈值：配置里写了就用；否则沿用变体配方给的默认（merge_recipe_defaults 已处理）
  if (pp.max_det <= 0) pp.max_det = pp.decode.max_det > 0 ? pp.decode.max_det : 300;
  if (pp.decode.max_det <= 0) pp.decode.max_det = pp.max_det;

  // plugin-nms 布局时 max_det 由引擎决定，这里不做截断（避免和插件打架）
  if (pp.decode.layout == OutputLayout::kPluginNms) {
    pp.max_det = std::max(pp.max_det, 1);
  }
  return pp;
}

}  // namespace todrt
