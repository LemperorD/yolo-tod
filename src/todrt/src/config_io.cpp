// config_io.cpp —— 部署配置（JSON）<-> 结构体
//
// 配置的权威生成端是 Python：`tools/export_onnx.py --deploy-config xxx.json`。
// 目的很直接：**网络结构信息只写一次**。C++ 侧不再手抄 nc / reg_max / strides，
// 抄错一个数字就是「框全乱但程序不报错」这类最难查的 bug。
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <sstream>

#include "todrt/detector_options.hpp"
#include "todrt/json.hpp"

namespace todrt {

namespace {

/// 归一化配置里的枚举写法：只保留 ASCII 字母数字并转小写。
/// 这样 "anchor-major-dfl" / "anchor_major_dfl" / "AnchorMajorDfl" 都能对上，
/// 而配置是人手写的，容忍这些差异比强迫用户记住某一种写法更有价值。
///
/// 实现上刻意**只用整数比较**，不写 char 字面量，也不用 std::isalnum/std::tolower：
/// 这两者在 MSVC + /utf-8 + 含非 ASCII 注释的源文件里都被观测到会漏掉字母
/// （"anchor-major-dfl" 被归一化成 "anchormajordfl"），是极难排查的坑。
std::string lower_compact(const std::string& s) {
  std::string k;
  k.reserve(s.size());
  for (char c : s) {
    const int v = static_cast<unsigned char>(c);
    if (v >= 65 && v <= 90) {
      k.push_back(static_cast<char>(v + 32));  // A-Z -> a-z
    } else if ((v >= 97 && v <= 122) ||  // a-z
               (v >= 48 && v <= 57)) {   // 0-9
      k.push_back(static_cast<char>(v));
    }
    // 其余字符（连字符、下划线、空格、斜杠等）一律丢弃
  }
  return k;
}

std::vector<int> ints_of(const json::Value& v, const std::vector<int>& def) {
  if (!v.is_array() || v.size() == 0) return def;
  std::vector<int> out;
  out.reserve(v.size());
  for (size_t i = 0; i < v.size(); ++i) out.push_back(v.at(i).as_int(0));
  return out;
}

std::vector<std::string> strings_of(const json::Value& v) {
  std::vector<std::string> out;
  if (!v.is_array()) return out;
  out.reserve(v.size());
  for (size_t i = 0; i < v.size(); ++i) out.push_back(v.at(i).as_string());
  return out;
}

/// 形状：接受 [w,h]、{"w":..,"h":..} 或 {"width":..,"height":..}
void parse_shape(const json::Value& v, std::array<int, 2>* wh) {
  if (v.is_array() && v.size() >= 2) {
    (*wh)[0] = v.at(0).as_int((*wh)[0]);
    (*wh)[1] = v.at(1).as_int((*wh)[1]);
    return;
  }
  if (v.is_object()) {
    if (v.contains("w")) (*wh)[0] = v["w"].as_int((*wh)[0]);
    if (v.contains("h")) (*wh)[1] = v["h"].as_int((*wh)[1]);
    if (v.contains("width")) (*wh)[0] = v["width"].as_int((*wh)[0]);
    if (v.contains("height")) (*wh)[1] = v["height"].as_int((*wh)[1]);
  }
}

void parse_flags(const json::Value& v, BuildConfig* cfg) {
  if (v.is_string()) {
    if (!cfg->set_flag(v.as_string())) log_warn("未知的 builder flag：" + v.as_string());
    return;
  }
  if (!v.is_array()) return;
  for (size_t i = 0; i < v.size(); ++i) {
    const std::string name = v.at(i).as_string();
    if (!name.empty() && !cfg->set_flag(name, true)) {
      log_warn("未知的 builder flag：" + name + "（已忽略）");
    }
  }
}

/// 解析输出布局。归一化后（去掉 - _ 空格并转小写）的合法取值：
///   anchormajordfl / anchormajor / dfl / v8 / yolov8 / ultralytics
///   anchormajordfltransposed / transposed
///   featuremajordfl / featuremajor
///   pluginnms / efficientnms / nms
OutputLayout parse_layout(const std::string& s, OutputLayout def) {
  const std::string k = lower_compact(s);
  if (k.empty()) return def;
  if (k == "anchormajordfl" || k == "anchormajor" || k == "dfl" || k == "v8" ||
      k == "yolov8" || k == "ultralytics") {
    return OutputLayout::kAnchorMajorDfl;
  }
  if (k == "anchormajordfltransposed" || k == "anchormajortransposed" || k == "transposed") {
    return OutputLayout::kAnchorMajorDflTransposed;
  }
  if (k == "featuremajordfl" || k == "featuremajor") return OutputLayout::kFeatureMajorDfl;
  if (k == "pluginnms" || k == "efficientnms" || k == "nmsplugin" || k == "nms") {
    return OutputLayout::kPluginNms;
  }
  throw TritError("未知的输出布局 layout：" + s +
                  "（可选 anchor-major-dfl / anchor-major-dfl-transposed / "
                  "feature-major-dfl / plugin-nms）");
}

}  // namespace

// ------------------------------------------------------------------ BuilderFlag

bool BuildConfig::set_flag(const std::string& name, bool on) {
  const std::string k = lower_compact(name);
  BuilderFlag f;
  if (k == "fp16fallback" || k == "fp16") {
    f = BuilderFlag::kFp16Fallback;
  } else if (k == "int8fallback" || k == "int8") {
    f = BuilderFlag::kInt8Fallback;
  } else if (k == "gpufallback" || k == "dlagpufallback" || k == "allowgpufallback") {
    f = BuilderFlag::kDlaGpuFallback;
  } else if (k == "dlastandalone") {
    f = BuilderFlag::kDlaStandalone;
  } else if (k == "cudagraphs" || k == "cudagraph") {
    f = BuilderFlag::kCudaGraphs;
  } else if (k == "sparsity" || k == "sparse") {
    f = BuilderFlag::kSparsity;
  } else if (k == "profilingverbosity" || k == "verbose") {
    f = BuilderFlag::kProfilingVerbosity;
  } else if (k == "stronglytyped" || k == "strongtyping") {
    f = BuilderFlag::kStronglyTyped;
  } else {
    return false;
  }
  set(f, on);
  return true;
}

// ------------------------------------------------------------------ BuildConfig

void from_json(const json::Value& v, BuildConfig& cfg) {
  if (!v.is_object()) throw TritError("BuildConfig: 期望 JSON 对象");

  // 引擎来源既可在顶层，也可嵌在 engine 段
  const json::Value& e = v.contains("engine") ? v["engine"] : v;
  if (e.is_string()) {
    cfg.engine_path = e.as_string();
  } else if (e.is_object()) {
    if (e.contains("path")) cfg.engine_path = e["path"].as_string(cfg.engine_path);
    if (e.contains("engine_path")) cfg.engine_path = e["engine_path"].as_string(cfg.engine_path);
    if (e.contains("onnx")) cfg.onnx_path = e["onnx"].as_string(cfg.onnx_path);
    if (e.contains("onnx_path")) cfg.onnx_path = e["onnx_path"].as_string(cfg.onnx_path);
    if (e.contains("serialize_out")) cfg.serialize_out = e["serialize_out"].as_string();
    if (e.contains("input_name")) cfg.input_name = e["input_name"].as_string();
    if (e.contains("output_name")) cfg.output_name = e["output_name"].as_string();
  }
  if (v.contains("onnx")) cfg.onnx_path = v["onnx"].as_string(cfg.onnx_path);
  if (v.contains("onnx_path")) cfg.onnx_path = v["onnx_path"].as_string(cfg.onnx_path);

  // 硬件
  const json::Value& hw = v["hardware"];
  if (hw.is_object()) {
    if (hw.contains("device")) cfg.device = parse_device(hw["device"].as_string());
    if (hw.contains("precision")) cfg.precision = parse_precision(hw["precision"].as_string());
    if (hw.contains("dla_core")) cfg.dla_core = hw["dla_core"].as_int(cfg.dla_core);
    if (hw.contains("dla_memory_limit_mb")) {
      cfg.dla_memory_limit_mb = hw["dla_memory_limit_mb"].as_int(cfg.dla_memory_limit_mb);
    }
    if (hw.contains("allow_gpu_fallback")) {
      cfg.allow_gpu_fallback = hw["allow_gpu_fallback"].as_bool(cfg.allow_gpu_fallback);
    }
    if (hw.contains("dynamic_shape")) cfg.dynamic_shape = hw["dynamic_shape"].as_bool(false);
    if (hw.contains("dynamic_batch")) cfg.dynamic_batch = hw["dynamic_batch"].as_bool(false);
    if (hw.contains("max_batch")) cfg.max_batch = hw["max_batch"].as_int(cfg.max_batch);
  }
  if (v.contains("precision")) cfg.precision = parse_precision(v["precision"].as_string());
  if (v.contains("device")) cfg.device = parse_device(v["device"].as_string());

  // 资源
  const json::Value& r = v["resources"];
  if (r.is_object()) {
    if (r.contains("workspace_mb")) {
      cfg.workspace_mb = static_cast<size_t>(r["workspace_mb"].as_int(1024));
    }
    if (r.contains("dla_workspace_mb")) {
      cfg.dla_workspace_mb = static_cast<size_t>(r["dla_workspace_mb"].as_int(0));
    }
    if (r.contains("num_build_threads")) cfg.num_build_threads = r["num_build_threads"].as_int(0);
    if (r.contains("timing_cache")) cfg.timing_cache = r["timing_cache"].as_bool(true);
    if (r.contains("tactic_sources")) cfg.tactic_sources = r["tactic_sources"].as_bool(true);
    if (r.contains("hardware_compatibility")) {
      cfg.hardware_compatibility = r["hardware_compatibility"].as_bool(false);
    }
  }

  // shape
  const json::Value& sh = v["shape"];
  if (sh.is_object()) {
    if (sh.contains("dynamic")) cfg.dynamic_shape = sh["dynamic"].as_bool(cfg.dynamic_shape);
    if (sh.contains("dynamic_shape")) cfg.dynamic_shape = sh["dynamic_shape"].as_bool();
    if (sh.contains("dynamic_batch")) cfg.dynamic_batch = sh["dynamic_batch"].as_bool();
    if (sh.contains("max_batch")) cfg.max_batch = sh["max_batch"].as_int(cfg.max_batch);
    if (sh.contains("min")) parse_shape(sh["min"], &cfg.shape_min);
    if (sh.contains("opt")) parse_shape(sh["opt"], &cfg.shape_opt);
    if (sh.contains("max")) parse_shape(sh["max"], &cfg.shape_max);
    if (sh.contains("input")) {  // 固定尺寸简写
      parse_shape(sh["input"], &cfg.shape_opt);
      cfg.shape_min = cfg.shape_opt;
      cfg.shape_max = cfg.shape_opt;
    }
  }

  // 量化
  const json::Value& q = v["quantization"];
  if (q.is_object()) {
    if (q.contains("calib_cache")) cfg.int8_calib_cache = q["calib_cache"].as_string();
    if (q.contains("calib_data")) cfg.int8_calib_data = q["calib_data"].as_string();
    if (q.contains("calib_alg")) cfg.int8_calib_alg = q["calib_alg"].as_string("entropy");
    if (q.contains("calib_batches")) cfg.int8_calib_batches = q["calib_batches"].as_int(10);
  }

  // 调试
  if (v.contains("verbose")) cfg.verbose = v["verbose"].as_bool(cfg.verbose);
  if (v.contains("dump_tactics")) cfg.dump_tactics = v["dump_tactics"].as_bool();
  if (v.contains("flags")) parse_flags(v["flags"], &cfg);
  const json::Value& b = v["builder"];
  if (b.is_object() && b.contains("flags")) parse_flags(b["flags"], &cfg);

  if (cfg.device == Device::kDla && cfg.precision == Precision::kFP32) {
    log_warn("DLA 不支持 FP32，已自动回落到 FP16。");
    cfg.precision = Precision::kFP16;
  }
}

// ------------------------------------------------------------------ 子选项

void from_json(const json::Value& v, PreprocessOptions& o) {
  if (!v.is_object()) return;
  if (v.contains("width")) o.input_width = v["width"].as_int(o.input_width);
  if (v.contains("height")) o.input_height = v["height"].as_int(o.input_height);
  if (v.contains("input_width")) o.input_width = v["input_width"].as_int(o.input_width);
  if (v.contains("input_height")) o.input_height = v["input_height"].as_int(o.input_height);
  if (v.contains("pad_multiple")) o.pad_multiple = v["pad_multiple"].as_int(o.pad_multiple);
  if (v.contains("norm_scale")) o.norm_scale = v["norm_scale"].as_float(o.norm_scale);
  if (v.contains("norm_bias")) o.norm_bias = v["norm_bias"].as_float(o.norm_bias);
  if (v.contains("to_rgb")) o.to_rgb = v["to_rgb"].as_bool(o.to_rgb);

  const std::string mode = v["mode"].as_string();
  if (!mode.empty()) {
    const std::string k = lower_compact(mode);
    if (k == "letterbox") {
      o.mode = ResizeMode::kLetterbox;
    } else if (k == "stretch" || k == "resize") {
      o.mode = ResizeMode::kStretch;
    } else if (k == "integer" || k == "integerscale") {
      o.mode = ResizeMode::kIntegerScale;
    } else {
      throw TritError("未知的 resize mode：" + mode);
    }
  }
  const std::string pad = lower_compact(v["pad_value"].as_string());
  if (pad == "zero" || pad == "0" || pad == "black") {
    o.pad_value = PadValue::kZero;
  } else if (pad == "edge" || pad == "replicate") {
    o.pad_value = PadValue::kEdge;
  } else if (!pad.empty()) {
    o.pad_value = PadValue::kGray114;
  }
}

void from_json(const json::Value& v, DecodeOptions& o) {
  if (!v.is_object()) return;
  if (v.contains("nc")) o.num_classes = v["nc"].as_int(o.num_classes);
  if (v.contains("num_classes")) o.num_classes = v["num_classes"].as_int(o.num_classes);
  if (v.contains("reg_max")) o.reg_max = v["reg_max"].as_int(o.reg_max);
  if (v.contains("max_det")) o.max_det = v["max_det"].as_int(o.max_det);
  if (v.contains("conf")) o.conf_threshold = v["conf"].as_float(o.conf_threshold);
  if (v.contains("conf_threshold")) {
    o.conf_threshold = v["conf_threshold"].as_float(o.conf_threshold);
  }
  if (v.contains("strides")) o.strides = ints_of(v["strides"], o.strides);
  if (v.contains("level_channels")) o.level_channels = ints_of(v["level_channels"], {});
  if (v.contains("features")) o.level_channels = ints_of(v["features"], o.level_channels);
  if (v.contains("layout")) o.layout = parse_layout(v["layout"].as_string(), o.layout);
}

void from_json(const json::Value& v, PostprocessOptions& o) {
  if (!v.is_object()) return;
  from_json(v, o.decode);
  const std::string nms = lower_compact(v["nms"].as_string());
  if (nms == "soft" || nms == "softnms" || nms == "gaussian") o.nms = NmsKind::kSoft;
  if (v.contains("iou")) o.iou_threshold = v["iou"].as_float(o.iou_threshold);
  if (v.contains("iou_threshold")) o.iou_threshold = v["iou_threshold"].as_float(o.iou_threshold);
  if (v.contains("sigma")) o.sigma = v["sigma"].as_float(o.sigma);
  if (v.contains("soft_score_threshold")) {
    o.soft_score_threshold = v["soft_score_threshold"].as_float(o.soft_score_threshold);
  }
  if (v.contains("class_agnostic")) o.class_agnostic = v["class_agnostic"].as_bool(false);
  if (v.contains("max_det")) {
    o.max_det = v["max_det"].as_int(o.max_det);
    o.decode.max_det = o.max_det;
  }
}

// ------------------------------------------------------------------ DeployConfig

DeployConfig DeployConfig::FromJson(const json::Value& v) {
  if (!v.is_object()) throw TritError("部署配置顶层必须是 JSON 对象");
  DeployConfig d;

  // 允许 {"deploy": {...}} 包裹
  const json::Value& root = v.contains("deploy") ? v["deploy"] : v;

  if (root.contains("schema")) d.schema = root["schema"].as_string(d.schema);
  if (root.contains("model")) d.model = root["model"].as_string();
  if (root.contains("model_id")) d.model_id = root["model_id"].as_string();
  if (root.contains("id")) d.model_id = root["id"].as_string(d.model_id);
  if (d.model.empty()) d.model = d.model_id;
  if (root.contains("dataset")) d.dataset = root["dataset"].as_string();
  if (root.contains("notes")) d.notes = root["notes"].as_string();
  if (root.contains("source")) d.source = root["source"].as_string();
  if (root.contains("class_names")) d.class_names = strings_of(root["class_names"]);

  // 网络结构：既支持扁平写法，也支持 network 段
  const json::Value& net = root.contains("network") ? root["network"] : root;
  if (net.contains("nc")) d.num_classes = net["nc"].as_int(d.num_classes);
  if (net.contains("num_classes")) d.num_classes = net["num_classes"].as_int(d.num_classes);
  if (net.contains("reg_max")) d.reg_max = net["reg_max"].as_int(d.reg_max);
  if (net.contains("strides")) d.strides = ints_of(net["strides"], d.strides);
  if (net.contains("level_channels")) d.level_channels = ints_of(net["level_channels"], {});
  if (net.contains("features")) d.level_channels = ints_of(net["features"], d.level_channels);
  if (net.contains("layout")) d.layout = parse_layout(net["layout"].as_string(), d.layout);
  if (net.contains("input")) parse_shape(net["input"], &d.input);
  if (net.contains("pad_multiple")) d.pad_multiple = net["pad_multiple"].as_int(d.pad_multiple);
  if (net.contains("levels")) {
    // levels=[2,3,4,5] → strides=[4,8,16,32]（仅在 strides 未显式给出时）
    const std::vector<int> lv = ints_of(net["levels"], {});
    if (!lv.empty() && !net.contains("strides")) {
      std::vector<int> s;
      s.reserve(lv.size());
      for (int l : lv) s.push_back(1 << l);
      d.strides = s;
    }
  }

  from_json(root.contains("build") ? root["build"] : root, d.build);

  d.Validate();
  return d;
}

DeployConfig DeployConfig::FromFile(const std::string& path) {
  DeployConfig d = FromJson(json::parse_file(path));
  if (d.source.empty()) d.source = "deploy config: " + path;
  return d;
}

void DeployConfig::Validate() const {
  const std::string want = "todrt.deploy/v1";
  if (schema.rfind("todrt.", 0) == 0 && schema != want && schema != "todrt.deploy/v1.0") {
    throw TritError("部署配置 schema 不受支持：" + schema + "（本版本只认 " + want + "）");
  }
  if (num_classes <= 0) throw TritError("nc 必须为正数，实际 " + std::to_string(num_classes));
  if (reg_max <= 0 || (reg_max & (reg_max - 1)) != 0) {
    throw TritError("reg_max 必须是 2 的幂（YOLOv8 为 16），实际 " + std::to_string(reg_max));
  }
  if (strides.empty()) throw TritError("strides 不能为空");
  if (layout == OutputLayout::kFeatureMajorDfl && level_channels.empty()) {
    throw TritError(
        "layout=feature-major-dfl 时必须给出 level_channels（每层的 4*reg_max+nc），"
        "否则无法切分通道。");
  }
  if (!level_channels.empty()) {
    if (level_channels.size() != strides.size()) {
      throw TritError("level_channels 长度（" + std::to_string(level_channels.size()) +
                      "）与 strides 长度（" + std::to_string(strides.size()) + "）不一致");
    }
    const int expect = 4 * reg_max + num_classes;
    for (size_t i = 0; i < level_channels.size(); ++i) {
      if (level_channels[i] != expect) {
        throw TritError("level_channels[" + std::to_string(i) + "]=" +
                        std::to_string(level_channels[i]) + " 与 4*reg_max+nc=" +
                        std::to_string(expect) + " 不符");
      }
    }
  }
  if (input[0] <= 0 || input[1] <= 0) throw TritError("input 尺寸非法");
}

// ------------------------------------------------------------------ DetectorOptions

DetectorOptions DetectorOptions::FromJson(const json::Value& v) {
  if (!v.is_object()) throw TritError("DetectorOptions: 顶层必须是 JSON 对象");
  DetectorOptions o;

  // 1) 部署配置首先生效（结构信息以它为准）
  o.deploy = DeployConfig::FromJson(v);
  const json::Value& root = v.contains("deploy") ? v["deploy"] : v;

  // 2) 工厂选择：`model` 是注册名（工厂查找键），优先级最高；
  //    `model_id` 是训练侧变体 id，只在没有 model 时作为兜底。
  std::string chosen;
  if (root.contains("model")) chosen = root["model"].as_string();
  if (chosen.empty() && root.contains("model_id")) chosen = root["model_id"].as_string();
  if (chosen.empty()) chosen = o.deploy.model;  // DeployConfig 里已做过同样的兜底
  o.model = chosen;
  if (o.model.empty()) o.model = "SPAE-YOLOv8n";

  // 3) 硬件：先套 build 段，再套 preset，最后由 runtime 段覆盖
  if (root.contains("build")) from_json(root["build"], o.deploy.build);
  o.precision = o.deploy.build.precision;
  o.device = o.deploy.build.device;
  o.dla_core = o.deploy.build.dla_core;
  o.dla_memory_limit_mb = o.deploy.build.dla_memory_limit_mb;
  o.allow_gpu_fallback = o.deploy.build.allow_gpu_fallback;
  o.dynamic_shape = o.deploy.build.dynamic_shape;
  o.dynamic_batch = o.deploy.build.dynamic_batch;
  o.max_batch = o.deploy.build.max_batch;
  o.shape_min = o.deploy.build.shape_min;
  o.shape_opt = o.deploy.build.shape_opt;
  o.shape_max = o.deploy.build.shape_max;
  o.workspace_mb = o.deploy.build.workspace_mb;
  o.verbose = o.deploy.build.verbose;

  if (root.contains("preset")) o.ApplyPreset(root["preset"].as_string());

  // 4) 运行期覆盖段
  const json::Value& rt = root.contains("runtime") ? root["runtime"] : json::Value();
  if (rt.is_object()) {
    if (rt.contains("device")) o.device = parse_device(rt["device"].as_string());
    if (rt.contains("precision")) o.precision = parse_precision(rt["precision"].as_string());
    if (rt.contains("dla_core")) o.dla_core = rt["dla_core"].as_int(o.dla_core);
    if (rt.contains("dla_memory_limit_mb")) {
      o.dla_memory_limit_mb = rt["dla_memory_limit_mb"].as_int(o.dla_memory_limit_mb);
    }
    if (rt.contains("allow_gpu_fallback")) {
      o.allow_gpu_fallback = rt["allow_gpu_fallback"].as_bool(o.allow_gpu_fallback);
    }
    if (rt.contains("dynamic_shape")) o.dynamic_shape = rt["dynamic_shape"].as_bool(false);
    if (rt.contains("dynamic_batch")) o.dynamic_batch = rt["dynamic_batch"].as_bool(false);
    if (rt.contains("max_batch")) o.max_batch = rt["max_batch"].as_int(o.max_batch);
    if (rt.contains("workspace_mb")) {
      o.workspace_mb = static_cast<size_t>(rt["workspace_mb"].as_int(1024));
    }
    if (rt.contains("device_id")) o.device_id = rt["device_id"].as_int(o.device_id);
    if (rt.contains("cuda_graphs")) o.cuda_graphs = rt["cuda_graphs"].as_bool(false);
    if (rt.contains("verbose")) o.verbose = rt["verbose"].as_bool(false);
    if (rt.contains("bench_warmup")) o.bench_warmup = rt["bench_warmup"].as_int(o.bench_warmup);
    if (rt.contains("bench_iters")) o.bench_iters = rt["bench_iters"].as_int(o.bench_iters);
    if (rt.contains("shape")) {
      if (rt["shape"].contains("min")) parse_shape(rt["shape"]["min"], &o.shape_min);
      if (rt["shape"].contains("opt")) parse_shape(rt["shape"]["opt"], &o.shape_opt);
      if (rt["shape"].contains("max")) parse_shape(rt["shape"]["max"], &o.shape_max);
    }
  }
  if (root.contains("verbose")) o.verbose = root["verbose"].as_bool(o.verbose);

  // 5) 前后处理：默认由结构信息推导，再被显式段覆盖
  o.preproc_opts.input_width = o.deploy.input[0];
  o.preproc_opts.input_height = o.deploy.input[1];
  o.preproc_opts.pad_multiple = o.deploy.pad_multiple;

  o.postproc_opts.decode.num_classes = o.deploy.num_classes;
  o.postproc_opts.decode.reg_max = o.deploy.reg_max;
  o.postproc_opts.decode.layout = o.deploy.layout;
  o.postproc_opts.decode.strides = o.deploy.strides;
  o.postproc_opts.decode.level_channels = o.deploy.level_channels;

  if (root.contains("preprocess")) from_json(root["preprocess"], o.preproc_opts);
  if (root.contains("postprocess")) {
    from_json(root["postprocess"], o.postproc_opts);
    // 结构相关字段永远以 deploy 为准（防止配置里写偏）
    o.postproc_opts.decode.num_classes = o.deploy.num_classes;
    o.postproc_opts.decode.reg_max = o.deploy.reg_max;
    if (o.postproc_opts.decode.layout == OutputLayout::kAnchorMajorDfl) {
      o.postproc_opts.decode.layout = o.deploy.layout;
    }
    if (o.postproc_opts.decode.strides == std::vector<int>{4, 8, 16, 32}) {
      o.postproc_opts.decode.strides = o.deploy.strides;
    }
    if (o.postproc_opts.decode.level_channels.empty()) {
      o.postproc_opts.decode.level_channels = o.deploy.level_channels;
    }
  }

  if (o.device == Device::kDla && o.precision == Precision::kFP32) {
    log_warn("DLA 不支持 FP32，已自动回落到 FP16。");
    o.precision = Precision::kFP16;
  }
  return o;
}

DetectorOptions DetectorOptions::FromFile(const std::string& path) {
  return FromJson(json::parse_file(path));
}

BuildConfig DetectorOptions::ToBuildConfig() const {
  BuildConfig c = deploy.build;
  c.precision = precision;
  c.device = device;
  c.dla_core = dla_core;
  c.dla_memory_limit_mb = dla_memory_limit_mb;
  c.allow_gpu_fallback = allow_gpu_fallback;
  c.dynamic_shape = dynamic_shape;
  c.dynamic_batch = dynamic_batch;
  c.max_batch = max_batch;
  c.shape_min = shape_min;
  c.shape_opt = shape_opt;
  c.shape_max = shape_max;
  c.workspace_mb = workspace_mb;
  c.verbose = verbose;
  return c;
}

json::Value DetectorOptions::ToJson() const {
  // 该快照用于"本次运行实际生效的配置"存档：字段名与输入配置保持一致，
  // 便于和原配置直接 diff。
  json::Object engine;
  engine["onnx"] = deploy.build.onnx_path;
  engine["engine_path"] = deploy.build.engine_path;
  engine["serialize_out"] = deploy.build.serialize_out;
  engine["input_name"] = deploy.build.input_name;
  engine["output_name"] = deploy.build.output_name;

  json::Object hw;
  hw["device"] = to_string(device);
  hw["precision"] = to_string(precision);
  hw["dla_core"] = dla_core;
  hw["dla_memory_limit_mb"] = dla_memory_limit_mb;
  hw["allow_gpu_fallback"] = allow_gpu_fallback;

  json::Object shape;
  shape["dynamic"] = dynamic_shape;
  shape["dynamic_batch"] = dynamic_batch;
  shape["max_batch"] = max_batch;
  shape["min"] =
      json::Value(json::Array{json::Value(shape_min[0]), json::Value(shape_min[1])});
  shape["opt"] =
      json::Value(json::Array{json::Value(shape_opt[0]), json::Value(shape_opt[1])});
  shape["max"] =
      json::Value(json::Array{json::Value(shape_max[0]), json::Value(shape_max[1])});

  json::Object rt;
  rt["device_id"] = device_id;
  rt["cuda_graphs"] = cuda_graphs;
  rt["verbose"] = verbose;
  rt["workspace_mb"] = static_cast<int64_t>(workspace_mb);

  json::Array strides;
  for (int s : deploy.strides) strides.push_back(json::Value(s));
  json::Array levels;
  for (int c : deploy.level_channels) levels.push_back(json::Value(c));
  json::Array input{json::Value(deploy.input[0]), json::Value(deploy.input[1])};

  json::Object pre;
  pre["mode"] = (preproc_opts.mode == ResizeMode::kLetterbox)
                    ? "letterbox"
                    : (preproc_opts.mode == ResizeMode::kStretch ? "stretch" : "integer-scale");
  pre["width"] = preproc_opts.input_width;
  pre["height"] = preproc_opts.input_height;
  pre["norm_scale"] = static_cast<double>(preproc_opts.norm_scale);
  pre["norm_bias"] = static_cast<double>(preproc_opts.norm_bias);

  json::Object post;
  post["conf"] = static_cast<double>(postproc_opts.decode.conf_threshold);
  post["iou"] = static_cast<double>(postproc_opts.iou_threshold);
  post["max_det"] = postproc_opts.max_det;
  post["nms"] = (postproc_opts.nms == NmsKind::kSoft) ? "soft" : "hard";
  post["class_agnostic"] = postproc_opts.class_agnostic;

  json::Object root;
  root["schema"] = "todrt.deploy/v1";
  root["model"] = model;
  root["model_id"] = deploy.model_id;
  root["builder"] = builder;
  root["preproc"] = preproc;
  root["postproc"] = postproc;
  root["nc"] = deploy.num_classes;
  root["reg_max"] = deploy.reg_max;
  root["layout"] = to_string(deploy.layout);
  root["strides"] = json::Value(std::move(strides));
  root["level_channels"] = json::Value(std::move(levels));
  root["input"] = json::Value(std::move(input));
  root["pad_multiple"] = deploy.pad_multiple;
  root["engine"] = json::Value(std::move(engine));
  root["hardware"] = json::Value(std::move(hw));
  root["shape"] = json::Value(std::move(shape));
  root["runtime"] = json::Value(std::move(rt));
  root["preprocess"] = json::Value(std::move(pre));
  root["postprocess"] = json::Value(std::move(post));
  if (!deploy.notes.empty()) root["notes"] = deploy.notes;
  if (!deploy.dataset.empty()) root["dataset"] = deploy.dataset;
  return json::Value(std::move(root));
}

// ------------------------------------------------------------------ 硬件预设

void DetectorOptions::ApplyPreset(const std::string& preset) {
  const std::string k = lower_compact(preset);
  if (k.empty() || k == "none") return;
  if (k == "orin" || k == "jetson" || k == "dla" || k == "agxorin" || k == "orinnano") {
    // Orin：DLA + FP16，固定 shape 最稳；要 INT8 再补校准缓存。
    device = Device::kDla;
    precision = Precision::kFP16;
    dynamic_shape = false;
    dynamic_batch = false;
    allow_gpu_fallback = true;
    dla_memory_limit_mb = 512;
    cuda_graphs = true;
  } else if (k == "dgp" || k == "dgpu" || k == "desktop" || k == "x86gpu") {
    device = Device::kGpu;
    precision = Precision::kFP16;
    cuda_graphs = true;
  } else if (k == "x86") {
    device = Device::kGpu;
    precision = Precision::kFP16;
    cuda_graphs = false;
  } else if (k == "fp32" || k == "debug") {
    device = Device::kGpu;
    precision = Precision::kFP32;
    cuda_graphs = false;
  } else if (k == "int8") {
    device = Device::kGpu;
    precision = Precision::kINT8;
  } else {
    throw TritError("未知硬件预设：" + preset + "（可选 orin/dgp/x86/fp32/int8/none）");
  }
}

}  // namespace todrt
