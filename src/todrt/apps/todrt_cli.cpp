// todrt_cli.cpp —— 部署工具的单一入口
//
// 设计原则：**先用不需要 GPU 的子命令自检，再去实机跑**。
//   todrt_cli list                 工厂里有什么（不需要 GPU/TensorRT）
//   todrt_cli info   <模型>        一个变体的完整链路与预期 anchor 数
//   todrt_cli dryrun <配置.json>    只装配不建引擎——配置写错在这里就能发现
//   todrt_cli probe                这台机器到底能不能用 DLA / TensorRT 版本
//   todrt_cli catalog              自动生成注册表总表（文档用）
//   todrt_cli bench  <配置.json> <图片> [iters]
//   todrt_cli run    <配置.json> <图片> [输出.json]
//
// 图片只支持 PPM/PGM（P6/P5），避免为了读一张图就引入 OpenCV；实机上一般直接用
// Detector API 接相机帧。
#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "todrt/factory.hpp"
#include "todrt/json.hpp"

#if TODRT_HAVE_TENSORRT
#include <cuda_runtime_api.h>

#include "todrt/backend/engine_trt.hpp"
#endif

namespace {

using namespace todrt;

struct Options {
  std::string command;
  std::vector<std::string> positional;
  std::map<std::string, std::string> flags;
  bool verbose = false;

  bool has(const std::string& k) const { return flags.count(k) != 0; }
  std::string get(const std::string& k, const std::string& def = "") const {
    auto it = flags.find(k);
    return it == flags.end() ? def : it->second;
  }
  int geti(const std::string& k, int def) const {
    auto it = flags.find(k);
    if (it == flags.end() || it->second.empty()) return def;
    return std::atoi(it->second.c_str());
  }
};

void PrintUsage() {
  std::cout <<
      R"(todrt_cli —— yolo-tod 的部署端工具（TensorRT / Jetson Orin）

用法：
  todrt_cli <命令> [参数...] [--key=value ...]

命令：
  list                          列出工厂里注册的变体（不需要 GPU）
  models                        同上，但输出 JSON
  info <模型名>                 打印该变体的工厂链路与关键参数（不需要 GPU）
  dryrun <配置.json>            只做装配自检，不加载/构建引擎
  probe                         探测本机的 TensorRT / CUDA / DLA 能力
  catalog                       打印注册表总表（用于生成文档）
  bench <配置.json> <图片> [n]  实测延迟（预热 10 次，默认 100 次取均值）
  run   <配置.json> <图片> [out] 跑一次并存 JSON 结果

通用参数：
  --model=<名字>                覆盖配置里的模型（如 SPAE-YOLOv8n）
  --builder=<名字>              覆盖引擎构建器
  --preproc=<名字> / --postproc=<名字>
  --device=auto|gpu|dla         覆盖设备
  --precision=fp32|fp16|int8    覆盖精度
  --dla-core=<n>
  --conf=<float> --iou=<float>  覆盖阈值（小目标常需要更低的 conf）
  --input=<WxH>                 覆盖输入尺寸（例如 640x640）
  --preset=orin|dgp|x86|fp32
  --engine=<路径>               直接用已序列化的 engine
  --onnx=<路径>                 指定 ONNX
  --save-engine=<路径>          构建后把 engine 落盘
  --save-config=<路径>          把"实际生效的配置"落盘（便于复现与归档）
  --verbose

示例：
  # 1) 开发机上（无 GPU）先验证配置与工厂装配
  todrt_cli dryrun configs/deploy/spae-yolov8n-orin.json
  # 2) 实机上先看硬件能力
  todrt_cli probe
  # 3) Orin 上构建 FP16 + DLA 引擎并落盘
  todrt_cli dryrun cfg.json --preset=orin --save-engine=spae_orin_fp16.engine
  # 4) 实测延迟
  todrt_cli bench cfg.json sample.ppm 200
)";
}

/// 解析 `--key=value` / `--key value` / `-k v`。
Options ParseArgs(int argc, char** argv) {
  Options o;
  if (argc < 2) {
    PrintUsage();
    std::exit(argc == 1 ? 0 : 2);
  }
  o.command = argv[1];
  for (int i = 2; i < argc; ++i) {
    std::string a = argv[i];
    if (a.rfind("--", 0) == 0) {
      const std::string body = a.substr(2);
      const size_t eq = body.find('=');
      if (eq != std::string::npos) {
        o.flags[body.substr(0, eq)] = body.substr(eq + 1);
      } else if (i + 1 < argc && argv[i + 1][0] != '-') {
        o.flags[body] = argv[++i];
      } else {
        o.flags[body] = "true";
      }
    } else if (!a.empty() && a[0] == '-' && a.size() > 1) {
      const std::string k = a.substr(1);
      if (i + 1 < argc) {
        o.flags[k] = argv[++i];
      }
    } else {
      o.positional.push_back(a);
    }
  }
  o.verbose = o.has("verbose") || o.has("v");
  return o;
}

/// 把命令行覆盖项应用到 DetectorOptions 上（**只覆盖显式给出的项**）。
void ApplyOverrides(DetectorOptions* o, const Options& cli) {
  if (cli.has("model")) o->model = cli.get("model");
  if (cli.has("builder")) o->builder = cli.get("builder");
  if (cli.has("preproc")) o->preproc = cli.get("preproc");
  if (cli.has("postproc")) o->postproc = cli.get("postproc");
  if (cli.has("device")) o->device = parse_device(cli.get("device"));
  if (cli.has("precision")) o->precision = parse_precision(cli.get("precision"));
  if (cli.has("dla-core")) o->dla_core = cli.geti("dla-core", 0);
  if (cli.has("conf")) o->postproc_opts.decode.conf_threshold = std::atof(cli.get("conf").c_str());
  if (cli.has("iou")) o->postproc_opts.iou_threshold = std::atof(cli.get("iou").c_str());
  if (cli.has("max-det")) o->postproc_opts.max_det = cli.geti("max-det", 300);
  if (cli.has("preset")) o->ApplyPreset(cli.get("preset"));
  if (cli.has("engine")) o->deploy.build.engine_path = cli.get("engine");
  if (cli.has("onnx")) o->deploy.build.onnx_path = cli.get("onnx");
  if (cli.has("save-engine")) o->deploy.build.serialize_out = cli.get("save-engine");
  if (cli.has("nms")) {
    o->postproc_opts.nms = (cli.get("nms") == "soft") ? NmsKind::kSoft : NmsKind::kHard;
  }
  if (cli.has("dynamic")) {
    const bool on = cli.get("dynamic") != "false";
    o->dynamic_shape = on;
    o->dynamic_batch = on;
  }
  if (cli.has("input")) {
    const std::string v = cli.get("input");
    const size_t x = v.find_first_of("xX*");
    if (x == std::string::npos) {
      throw TritError("--input 需要 WxH 形式，例如 --input=640x640");
    }
    const int w = std::atoi(v.substr(0, x).c_str());
    const int h = std::atoi(v.substr(x + 1).c_str());
    if (w <= 0 || h <= 0) throw TritError("--input 尺寸非法：" + v);
    o->preproc_opts.input_width = w;
    o->preproc_opts.input_height = h;
    o->deploy.input = {w, h};
    o->shape_min = {w, h};
    o->shape_opt = {w, h};
    o->shape_max = {w, h};
    o->postproc_opts.decode.input_width = w;
    o->postproc_opts.decode.input_height = h;
  }
  if (cli.verbose) o->verbose = true;
}

/// 读 PPM(P6) / PGM(P5)。返回交错 RGB 数据（HWC）。
bool LoadPnm(const std::string& path, std::vector<uint8_t>* pixels, int* w, int* h, int* channels,
             std::string* err) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    *err = "无法打开图片：" + path;
    return false;
  }
  auto next_token = [&](std::string* tok) -> bool {
    *tok = "";
    char c;
    // 跳过空白与注释
    while (in.get(c)) {
      if (c == '#') {
        std::string line;
        std::getline(in, line);
      } else if (!std::isspace(static_cast<unsigned char>(c))) {
        tok->push_back(c);
        break;
      }
    }
    if (tok->empty()) return false;
    while (in.get(c) && !std::isspace(static_cast<unsigned char>(c))) tok->push_back(c);
    return true;
  };

  std::string magic;
  if (!next_token(&magic)) {
    *err = "文件为空：" + path;
    return false;
  }
  if (magic != "P6" && magic != "P5") {
    *err = "只支持 PPM(P6)/PGM(P5)：" + path + "（magic=" + magic + "）";
    return false;
  }
  std::string sw, sh, smax;
  if (!next_token(&sw) || !next_token(&sh) || !next_token(&smax)) {
    *err = "PNM 头不完整：" + path;
    return false;
  }
  *w = std::atoi(sw.c_str());
  *h = std::atoi(sh.c_str());
  *channels = (magic == "P6") ? 3 : 1;
  in.get();  // 头与数据之间恰好一个空白字符
  pixels->resize(static_cast<size_t>(*w) * static_cast<size_t>(*h) *
                 static_cast<size_t>(*channels));
  in.read(reinterpret_cast<char*>(pixels->data()),
          static_cast<std::streamsize>(pixels->size()));
  if (in.gcount() != static_cast<std::streamsize>(pixels->size())) {
    *err = "PNM 数据不完整：" + path;
    return false;
  }
  return true;
}

std::string DetectionsToJson(const std::vector<std::vector<Detection>>& per_image) {
  std::ostringstream oss;
  oss << "{\n  \"images\": [\n";
  for (size_t i = 0; i < per_image.size(); ++i) {
    oss << "    {\n      \"detections\": [\n";
    const auto& ds = per_image[i];
    for (size_t k = 0; k < ds.size(); ++k) {
      oss << "        {\"x1\": " << ds[k].box.x1 << ", \"y1\": " << ds[k].box.y1
          << ", \"x2\": " << ds[k].box.x2 << ", \"y2\": " << ds[k].box.y2
          << ", \"score\": " << ds[k].score << ", \"class_id\": " << ds[k].class_id << "}";
      oss << (k + 1 < ds.size() ? ",\n" : "\n");
    }
    oss << "      ]\n    }" << (i + 1 < per_image.size() ? ",\n" : "\n");
  }
  oss << "  ]\n}\n";
  return oss.str();
}

// ------------------------------------------------------------------ 子命令

int CmdList(bool as_json) {
  const std::vector<std::string> names = model_names();
  if (as_json) {
    json::Array arr;
    for (const auto& n : names) arr.push_back(json::Value(n));
    json::Object root;
    root["models"] = json::Value(std::move(arr));
    std::cout << json::Value(std::move(root)).dump(true) << "\n";
    return 0;
  }
  std::cout << "已注册变体（" << names.size() << "）：\n";
  for (const auto& n : names) {
    try {
      const ModelRecipe r = model_recipe(n);
      std::cout << "  - " << n;
      if (!r.notes.empty()) std::cout << "   " << r.notes;
      std::cout << "\n";
      std::cout << "      builder=" << (r.builder.empty() ? "auto" : r.builder)
                << "  preproc=" << (r.preproc.empty() ? "auto" : r.preproc)
                << "  postproc=" << (r.postproc.empty() ? "auto" : r.postproc) << "\n";
      std::cout << "      layout=" << to_string(r.layout) << "  strides=";
      for (size_t i = 0; i < r.decode.strides.size(); ++i) {
        std::cout << (i ? "," : "") << r.decode.strides[i];
      }
      std::cout << "  nc=" << r.decode.num_classes << "  reg_max=" << r.decode.reg_max << "\n";
      if (!r.source.empty()) std::cout << "      来源: " << r.source << "\n";
      if (!r.license.empty()) std::cout << "      许可证: " << r.license << "\n";
    } catch (const std::exception& e) {
      std::cout << "  - " << n << "  [错误] " << e.what() << "\n";
    }
  }
  const std::vector<std::string> missing = check_registry();
  if (!missing.empty()) {
    std::cout << "\n⚠️ 注册表依赖缺失（--list 仍可用，但装配会失败）：\n";
    for (const auto& m : missing) std::cout << "  - " << m << "\n";
  }
  return 0;
}

int CmdInfo(const std::string& name) {
  if (!has_model(name)) {
    std::cerr << "未知模型：" << name << "\n已注册：";
    for (const auto& n : model_names()) std::cerr << n << " ";
    std::cerr << "\n";
    return 2;
  }
  if (registry_has(name)) {
    const RegistryEntry& e = registry_get(name);
    std::cout << "== 元数据 ==\n";
    std::cout << "  名称    : " << e.name << "\n";
    std::cout << "  来源    : " << e.source << "\n";
    std::cout << "  许可证  : " << e.license << "\n";
    std::cout << "  硬件    : " << e.hardware << "\n";
    std::cout << "  成本    : " << e.cost << "\n";
  }
  DetectorOptions o;
  o.model = name;
  const AssemblyReport rep = PlanAssembly(name, o);
  std::cout << rep.ToText();
  return 0;
}

int CmdDryRun(const std::string& cfg_path, const Options& cli) {
  DetectorOptions o = cfg_path.empty() ? DetectorOptions{} : DetectorOptions::FromFile(cfg_path);
  ApplyOverrides(&o, cli);
  const std::string model = o.model;
  const AssemblyReport rep = PlanAssembly(model, o);
  std::cout << rep.ToText();

  if (cli.has("save-config")) {
    std::ofstream out(cli.get("save-config"));
    if (!out) {
      std::cerr << "无法写入 " << cli.get("save-config") << "\n";
      return 1;
    }
    out << o.ToJson().dump(true) << "\n";
    std::cout << "\n实际生效的配置已写入：" << cli.get("save-config") << "\n";
  }
  if (!rep.registry_missing.empty()) return 3;
  if (!rep.builder_available) {
    std::cout << "\n提示：构建器不可用（" << rep.builder_unavailable_reason
              << "）。工厂装配与配置解析仍然验证通过；请到目标机上再跑一次。\n";
  }
  return 0;
}

int CmdProbe() {
  std::cout << "== 部署环境探测 ==\n";
  std::cout << "  后端 TensorRT : ";
  std::string reason;
  const bool avail = backend_available(&reason);
  std::cout << (avail ? "可用" : "不可用") << " —— " << reason << "\n";
#if TODRT_HAVE_TENSORRT
  const TrtEngine::PlanInfo info = TrtEngine::Probe(BuildConfig{});
  std::cout << "  TensorRT 版本 : " << info.trt_version << "\n";
  std::cout << "  CUDA 设备     : " << (info.device_name.empty() ? "(无)" : info.device_name)
            << "  compute capability "
            << (info.compute_capability.empty() ? "-" : info.compute_capability) << "\n";
  std::cout << "  DLA core 数   : " << info.nb_dla_cores;
  if (!info.dla_supported_build) {
    std::cout << "  ← 当前 TensorRT 版本已移除 DLA，请用 TensorRT 10.x";
  }
  std::cout << "\n";
  std::cout << "  DLA 最大 batch: " << info.max_batch << "\n";
  if (info.nb_dla_cores > 0) {
    std::cout << "\n建议：--preset=orin（DLA + FP16，固定 shape，开启 CUDA Graph）\n";
  } else {
    std::cout << "\n建议：--preset=dgp（FP16 + CUDA Graph）；x86 无 DLA。\n";
  }
#else
  std::cout << "  （本次构建未启用 TensorRT：用 -DTODRT_WITH_TENSORRT=ON 重新构建即可）\n";
#endif
  std::cout << "\n== 已注册工厂 ==\n" << registry_catalog("");
  return avail ? 0 : 1;
}

int CmdBench(const std::string& cfg_path, const std::string& image_path, const Options& cli) {
  DetectorOptions o = DetectorOptions::FromFile(cfg_path);
  ApplyOverrides(&o, cli);

  std::vector<uint8_t> pixels;
  int w = 0, h = 0, ch = 0;
  std::string err;
  if (!LoadPnm(image_path, &pixels, &w, &h, &ch, &err)) {
    std::cerr << err << "\n";
    return 2;
  }
  ImageView img;
  img.data = pixels.data();
  img.width = w;
  img.height = h;
  img.channels = ch;
  img.bgr = (ch != 1);

  auto det = Detector::Create(o.model, o);
  std::cout << det->Describe();
  std::cout << "\n输入图片：" << image_path << " " << w << "×" << h << " ch=" << ch << "\n";

  const int warmup = cli.geti("warmup", 10);
  const int iters = cli.geti("iters", cli.positional.size() > 2
                                           ? std::atoi(cli.positional[2].c_str())
                                           : 100);
  const Detector::BenchResult b = det->Bench(img, warmup, iters);
  std::printf("\n== 实测（%d 次，预热 %d 次）==\n", b.iters, warmup);
  std::printf("  前处理 : %8.3f ms\n", b.preprocess_ms);
  std::printf("  推理   : %8.3f ms\n", b.infer_ms);
  std::printf("  后处理 : %8.3f ms\n", b.postprocess_ms);
  std::printf("  端到端 : %8.3f ms  →  %.1f FPS\n", b.end_to_end_ms, b.fps);
  std::printf("  设备/精度: %s / %s\n", b.device.c_str(), b.precision.c_str());
  std::cout << "\n注意：延迟数字必须来自**目标设备**；x86 开发机上的数字只作相对参考。\n";
  return 0;
}

int CmdRun(const std::string& cfg_path, const std::string& image_path, const std::string& out_path,
           const Options& cli) {
  DetectorOptions o = DetectorOptions::FromFile(cfg_path);
  ApplyOverrides(&o, cli);

  std::vector<uint8_t> pixels;
  int w = 0, h = 0, ch = 0;
  std::string err;
  if (!LoadPnm(image_path, &pixels, &w, &h, &ch, &err)) {
    std::cerr << err << "\n";
    return 2;
  }
  ImageView img;
  img.data = pixels.data();
  img.width = w;
  img.height = h;
  img.channels = ch;
  img.bgr = (ch != 1);

  auto det = Detector::Create(o.model, o);
  const auto t0 = std::chrono::steady_clock::now();
  const std::vector<Detection> dets = det->Run(img);
  const auto t1 = std::chrono::steady_clock::now();
  const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  std::cout << det->Describe();
  std::cout << "\n检测到 " << dets.size() << " 个目标（" << ms << " ms）\n";
  for (size_t i = 0; i < dets.size() && i < 20; ++i) {
    std::printf("  #%zu cls=%d score=%.3f box=[%.1f,%.1f,%.1f,%.1f]\n", i, dets[i].class_id,
                dets[i].score, dets[i].box.x1, dets[i].box.y1, dets[i].box.x2, dets[i].box.y2);
  }
  if (dets.size() > 20) std::cout << "  ...（仅打印前 20 条）\n";

  if (!out_path.empty()) {
    std::ofstream out(out_path);
    if (!out) {
      std::cerr << "无法写入 " << out_path << "\n";
      return 1;
    }
    out << DetectionsToJson({dets});
    std::cout << "\n结果已写入：" << out_path << "\n";
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options cli = ParseArgs(argc, argv);
    set_log_sink([&](detail::LogLevel lv, const std::string& msg) {
      if (lv == detail::LogLevel::kDebug && !cli.verbose) return;
      std::FILE* f = (lv == detail::LogLevel::kError) ? stderr : stdout;
      std::fprintf(f, "%s\n", msg.c_str());
    });

    const std::string& cmd = cli.command;
    auto need = [&](size_t n, const char* usage) -> bool {
      if (cli.positional.size() < n) {
        std::cerr << "参数不足。用法：" << usage << "\n";
        return false;
      }
      return true;
    };

    if (cmd == "list" || cmd == "models") return CmdList(cmd == "models");
    if (cmd == "catalog") {
      std::cout << registry_catalog("");
      return 0;
    }
    if (cmd == "probe") return CmdProbe();
    if (cmd == "info") {
      if (!need(1, "todrt_cli info <模型名>")) return 2;
      return CmdInfo(cli.positional[0]);
    }
    if (cmd == "dryrun" || cmd == "dry-run") {
      return CmdDryRun(cli.positional.empty() ? "" : cli.positional[0], cli);
    }
    if (cmd == "bench") {
      if (!need(2, "todrt_cli bench <配置.json> <图片.ppm> [iters]")) return 2;
      return CmdBench(cli.positional[0], cli.positional[1], cli);
    }
    if (cmd == "run") {
      if (!need(2, "todrt_cli run <配置.json> <图片.ppm> [out.json]")) return 2;
      return CmdRun(cli.positional[0], cli.positional[1],
                    cli.positional.size() > 2 ? cli.positional[2] : "", cli);
    }
    if (cmd == "help" || cmd == "-h" || cmd == "--help") {
      PrintUsage();
      return 0;
    }

    std::cerr << "未知命令：" << cmd << "\n\n";
    PrintUsage();
    return 2;
  } catch (const std::exception& e) {
    std::cerr << "\n[错误] " << e.what() << "\n";
    return 1;
  }
}
