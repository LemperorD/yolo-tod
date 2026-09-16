// core.cpp —— 日志、枚举解析、自注册表的实现
#include "todrt/core.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <set>
#include <sstream>

#include "todrt/modules.hpp"  // 枚举 <-> 字符串（DataType / OutputLayout）

namespace todrt {
namespace detail {

LogSink& log_sink() {
  static LogSink sink;
  return sink;
}

void emit(LogLevel lv, const std::string& msg) {
  LogSink& sink = log_sink();
  if (sink) {
    sink(lv, msg);
    return;
  }
  static const char* kTag[] = {"[debug]", "[info ]", "[warn ]", "[error]"};
  std::FILE* f = (lv == LogLevel::kError || lv == LogLevel::kWarn) ? stderr : stdout;
  std::fprintf(f, "%s %s\n", kTag[static_cast<int>(lv)], msg.c_str());
  std::fflush(f);
}

std::unordered_map<std::string, RegistryEntry>& registry() {
  static std::unordered_map<std::string, RegistryEntry> r;
  return r;
}

std::vector<std::string>& registry_order() {
  static std::vector<std::string> r;
  return r;
}

std::mutex& registry_mutex() {
  static std::mutex m;
  return m;
}

RegistryEntry make_entry(std::string name, std::string family, std::string source,
                         std::string license, std::string hardware, std::string cost,
                         std::string notes, std::vector<std::string> aliases,
                         std::vector<std::string> requires, const char* file, int line) {
  RegistryEntry e;
  e.name = std::move(name);
  e.family = std::move(family);
  e.source = std::move(source);
  e.license = std::move(license);
  e.hardware = std::move(hardware);
  e.cost = std::move(cost);
  e.notes = std::move(notes);
  e.aliases = std::move(aliases);
  e.requires = std::move(requires);
  e.file = file ? file : "";
  e.line = line;
  return e;
}

}  // namespace detail

void set_log_sink(detail::LogSink sink) { detail::log_sink() = std::move(sink); }

// ------------------------------------------------------------------ 枚举

const char* to_string(Device d) {
  switch (d) {
    case Device::kGpu: return "gpu";
    case Device::kDla: return "dla";
    case Device::kNpu: return "npu";
    case Device::kCpu: return "cpu";
    default: return "auto";
  }
}

const char* to_string(TensorLayout l) {
  switch (l) {
    case TensorLayout::kNhwc: return "nhwc";
    default: return "nchw";
  }
}

const char* to_string(Precision p) {
  switch (p) {
    case Precision::kFP32: return "fp32";
    case Precision::kFP16: return "fp16";
    case Precision::kINT8: return "int8";
  }
  return "fp32";
}

const char* to_string(DataType t) {
  switch (t) {
    case DataType::kF32: return "f32";
    case DataType::kF16: return "f16";
    case DataType::kI8: return "i8";
    case DataType::kU8: return "u8";
    case DataType::kI32: return "i32";
  }
  return "?";
}

size_t dtype_size(DataType t) {
  switch (t) {
    case DataType::kF32: return 4;
    case DataType::kF16: return 2;
    case DataType::kI8: return 1;
    case DataType::kU8: return 1;
    case DataType::kI32: return 4;
  }
  return 4;
}

const char* to_string(OutputLayout l) {
  switch (l) {
    case OutputLayout::kAnchorMajorDfl: return "anchor-major-dfl";
    case OutputLayout::kAnchorMajorDflTransposed: return "anchor-major-dfl-transposed";
    case OutputLayout::kFeatureMajorDfl: return "feature-major-dfl";
    case OutputLayout::kPluginNms: return "plugin-nms";
  }
  return "?";
}

namespace {
/// 归一化枚举文本：只保留 ASCII 字母数字并转小写。
/// 刻意**不用 std::tolower**：它在 MSVC + /utf-8 + 含非 ASCII 注释的源文件里
/// 被观测到会漏掉字母（见 config_io.cpp 里同款说明），而配置字段是人手写的，
/// 一个字母被吃掉就会变成"无法识别的 device"。
std::string lower(std::string s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    if (c >= 'A' && c <= 'Z') {
      out.push_back(static_cast<char>(c - 'A' + 'a'));
    } else if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')) {
      out.push_back(c);
    }
    // 连字符/下划线/空格等一律丢弃，容忍 "no-gpu-fallback"、"rk3588 npu"
  }
  return out;
}
}  // namespace

Device parse_device(const std::string& s) {
  const std::string k = lower(s);
  if (k.empty() || k == "auto" || k == "default") return Device::kAuto;
  if (k == "gpu" || k == "cuda" || k == "dgpu") return Device::kGpu;
  if (k == "dla" || k == "nvdla") return Device::kDla;
  if (k == "npu" || k == "rknn" || k == "rknnnpu" || k == "rockchip") return Device::kNpu;
  if (k == "cpu" || k == "x86" || k == "amd" || k == "arm" || k == "host") return Device::kCpu;
  throw TritError("无法识别的 device：" + s + "（可选 auto/gpu/dla/npu/cpu）");
}

Precision parse_precision(const std::string& s) {
  const std::string k = lower(s);
  if (k.empty() || k == "auto") return Precision::kFP16;
  if (k == "fp32" || k == "float" || k == "float32" || k == "tf32") return Precision::kFP32;
  if (k == "fp16" || k == "half" || k == "float16" || k == "fp16s") return Precision::kFP16;
  if (k == "int8" || k == "i8" || k == "qint8") return Precision::kINT8;
  throw TritError("无法识别的 precision：" + s + "（可选 fp32/fp16/int8）");
}

// ------------------------------------------------------------------ 注册表

bool registry_has(const std::string& name) {
  std::lock_guard<std::mutex> lk(detail::registry_mutex());
  return detail::registry().count(name) != 0;
}

const RegistryEntry& registry_get(const std::string& name) {
  std::lock_guard<std::mutex> lk(detail::registry_mutex());
  auto& reg = detail::registry();
  auto it = reg.find(name);
  if (it == reg.end()) {
    std::ostringstream oss;
    oss << "未注册的模块 " << name << "。已注册：";
    bool first = true;
    for (const auto& n : detail::registry_order()) {
      oss << (first ? "" : ", ") << n;
      first = false;
    }
    throw TritError(oss.str());
  }
  return it->second;  // 注册表生命周期 = 进程，返回引用安全
}

std::vector<std::string> registry_names(const std::string& family) {
  std::lock_guard<std::mutex> lk(detail::registry_mutex());
  std::vector<std::string> out;
  for (const auto& n : detail::registry_order()) {
    const auto& e = detail::registry()[n];
    if (family.empty() || e.family == family) out.push_back(n);
  }
  return out;
}

std::vector<RegistryEntry> registry_entries(const std::string& family) {
  std::lock_guard<std::mutex> lk(detail::registry_mutex());
  std::vector<RegistryEntry> out;
  for (const auto& n : detail::registry_order()) {
    const auto& e = detail::registry()[n];
    if (family.empty() || e.family == family) out.push_back(e);
  }
  return out;
}

std::vector<std::string> check_registry() {
  std::vector<std::string> missing;
  std::lock_guard<std::mutex> lk(detail::registry_mutex());
  for (const auto& n : detail::registry_order()) {
    const auto& e = detail::registry()[n];
    for (const auto& dep : e.requires) {
      if (!detail::registry().count(dep)) {
        std::ostringstream oss;
        oss << e.family << ":" << e.name << " 依赖 " << dep << "（未注册）";
        missing.push_back(oss.str());
      }
    }
  }
  return missing;
}

std::string registry_catalog(const std::string& family) {
  std::lock_guard<std::mutex> lk(detail::registry_mutex());
  std::ostringstream oss;
  oss << "推理端注册表（自动生成，请勿手工编辑）\n";
  const std::vector<std::string> kFamilies = {"model", "builder", "preproc", "postproc", "decoder",
                                              "nms"};
  for (const auto& fam : kFamilies) {
    if (!family.empty() && family != fam) continue;
    std::vector<const RegistryEntry*> rows;
    for (const auto& n : detail::registry_order()) {
      const auto& e = detail::registry()[n];
      if (e.family == fam && e.canonical) rows.push_back(&e);
    }
    if (rows.empty()) continue;
    oss << "\n## " << fam << "（" << rows.size() << " 项）\n";
    for (const RegistryEntry* e : rows) {
      oss << "  - " << e->name;
      if (!e->aliases.empty()) {
        oss << "  [别名: ";
        for (size_t i = 0; i < e->aliases.size(); ++i) oss << (i ? ", " : "") << e->aliases[i];
        oss << "]";
      }
      oss << "\n      来源: " << (e->source.empty() ? "-" : e->source)
          << "\n      许可证: " << (e->license.empty() ? "-" : e->license)
          << "\n      硬件/精度: " << (e->hardware.empty() ? "-" : e->hardware)
          << "\n      成本: " << (e->cost.empty() ? "-" : e->cost);
      if (!e->notes.empty()) oss << "\n      备注: " << e->notes;
      if (!e->requires.empty()) {
        oss << "\n      依赖: ";
        for (size_t i = 0; i < e->requires.size(); ++i) oss << (i ? ", " : "") << e->requires[i];
      }
      oss << "\n";
    }
  }
  return oss.str();
}

}  // namespace todrt
