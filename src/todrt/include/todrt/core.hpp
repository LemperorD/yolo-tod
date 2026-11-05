#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace todrt {

class TritError : public std::runtime_error {
 public:
  explicit TritError(const std::string& what) : std::runtime_error(what) {}
};

enum class Device { kAuto = 0, kGpu = 1, kDla = 2, kNpu = 3, kCpu = 4 };

enum class Precision { kFP32 = 0, kFP16 = 1, kINT8 = 2 };

namespace detail {
enum class LogLevel { kDebug = 0, kInfo = 1, kWarn = 2, kError = 3 };
using LogSink = std::function<void(LogLevel, const std::string&)>;
LogSink& log_sink();
void emit(LogLevel lv, const std::string& msg);
}

// 日志回调的线程安全由调用方保证。
void set_log_sink(detail::LogSink sink);

inline void log_debug(const std::string& m) { detail::emit(detail::LogLevel::kDebug, m); }
inline void log_info(const std::string& m) { detail::emit(detail::LogLevel::kInfo, m); }
inline void log_warn(const std::string& m) { detail::emit(detail::LogLevel::kWarn, m); }
inline void log_error(const std::string& m) { detail::emit(detail::LogLevel::kError, m); }

const char* to_string(Device d);
const char* to_string(Precision p);

Device parse_device(const std::string& s);
Precision parse_precision(const std::string& s);

struct RegistryEntry {
  std::string name;
  std::string family;
  bool canonical = true;
  std::string source;
  std::string license;
  std::string hardware;
  std::string cost;
  std::string notes;
  std::vector<std::string> aliases;
  std::vector<std::string> requires;
  std::string file;
  int line = 0;
};

namespace detail {

std::unordered_map<std::string, RegistryEntry>& registry();
std::vector<std::string>& registry_order();
std::mutex& registry_mutex();

void register_entry(const RegistryEntry& entry, std::vector<std::string> factories);

RegistryEntry make_entry(std::string name, std::string family, std::string source,
                         std::string license, std::string hardware, std::string cost,
                         std::string notes, std::vector<std::string> aliases,
                         std::vector<std::string> requires, const char* file, int line);

}

const RegistryEntry& registry_get(const std::string& name);

bool registry_has(const std::string& name);

std::vector<std::string> registry_names(const std::string& family = "");

std::vector<RegistryEntry> registry_entries(const std::string& family = "");

std::vector<std::string> check_registry();

std::string registry_catalog(const std::string& family = "");

}
