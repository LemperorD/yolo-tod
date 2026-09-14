// core.hpp —— 推理端基础设施：错误码 / 日志 / 自注册表（工厂模式的骨架）
//
// 设计目标（与 Python 侧 src/tod/registry.py 一一对应）：
//   * 新增一个可部署变体（或新的前后处理策略）= 一个类 + 一行 REGISTER_* 宏；
//   * 调用方永远只依赖「名字 + Options」，不依赖任何具体实现类；
//   * 元数据（来源/许可证/硬件要求）强制登记，避免部署代码变成不可追溯的散装脚本。
//
// 本文件**不依赖 TensorRT**，可以被纯 CPU 工具链编译与自检（见 tests/cpp_smoke.cpp）。
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

// ------------------------------------------------------------------ 错误与日志

/// 推理端统一异常；所有工厂/配置错误都由此抛出，便于上层一次性捕获。
class TritError : public std::runtime_error {
 public:
  explicit TritError(const std::string& what) : std::runtime_error(what) {}
};

/// 设备类型。DLA 是 Jetson/Orin 上的固定功能加速器（硬件加速的主力）。
enum class Device { kAuto = 0, kGpu = 1, kDla = 2 };

/// 数值精度。DLA 只支持 FP16 / INT8。
enum class Precision { kFP32 = 0, kFP16 = 1, kINT8 = 2 };

namespace detail {
enum class LogLevel { kDebug = 0, kInfo = 1, kWarn = 2, kError = 3 };
using LogSink = std::function<void(LogLevel, const std::string&)>;
LogSink& log_sink();
void emit(LogLevel lv, const std::string& msg);
}  // namespace detail

/// 安装日志回调（默认写 stderr）。多线程下由调用方负责保证 sink 自身线程安全。
void set_log_sink(detail::LogSink sink);

inline void log_debug(const std::string& m) { detail::emit(detail::LogLevel::kDebug, m); }
inline void log_info(const std::string& m) { detail::emit(detail::LogLevel::kInfo, m); }
inline void log_warn(const std::string& m) { detail::emit(detail::LogLevel::kWarn, m); }
inline void log_error(const std::string& m) { detail::emit(detail::LogLevel::kError, m); }

// ------------------------------------------------------------------ 枚举 <-> 字符串

const char* to_string(Device d);
const char* to_string(Precision p);

/// 宽松解析：大小写不敏感，同时接受别写法（fp16/fp16s/half、"int8"、"dla"…）。
Device parse_device(const std::string& s);
Precision parse_precision(const std::string& s);

// ------------------------------------------------------------------ 自注册表

/// 一座工厂的元数据。强制登记来源与许可证，理由同 Python 侧 registry。
struct RegistryEntry {
  std::string name;         ///< 注册名（工厂查找键）
  std::string family;       ///< 工厂族："model" / "builder" / "preproc" / "postproc" / "nms" / "decoder"
  /// 该条目是否是"规范名"（false = 它是别人的别名）。
  /// 用于让清单只列规范名，避免 CLI 列表里出现一堆同义项。
  bool canonical = true;
  std::string source;       ///< 代码来源（论文/仓库，或 "builtin"）
  std::string license;      ///< 许可证
  std::string hardware;     ///< 硬件/精度要求说明（如 "需要 DLA，仅 FP16/INT8"）
  std::string cost;         ///< 成本提示
  std::string notes;        ///< 备注
  std::vector<std::string> aliases;
  std::vector<std::string> requires;  ///< 依赖的其他注册项（可被 check_registry 校验）
  std::string file;                   ///< 注册发生的源文件（调试用）
  int line = 0;
};

namespace detail {

/// 全库唯一的注册表本体。函数内静态，避免静态初始化顺序问题。
std::unordered_map<std::string, RegistryEntry>& registry();
std::vector<std::string>& registry_order();
std::mutex& registry_mutex();

/// 注册一个工厂条目（实现见 src/factory.cpp：同时绑定工厂函数）。
void register_entry(const RegistryEntry& entry, std::vector<std::string> factories);

/// 注册宏展开后的通行证：宏内不能出现逗号以外的模板语法，因此把注册器做成函数。
RegistryEntry make_entry(std::string name, std::string family, std::string source,
                         std::string license, std::string hardware, std::string cost,
                         std::string notes, std::vector<std::string> aliases,
                         std::vector<std::string> requires, const char* file, int line);

}  // namespace detail

/// 按名字取条目；未注册直接抛错（而不是悄悄返回空）。
const RegistryEntry& registry_get(const std::string& name);

/// 名字是否已注册（含别名）。
bool registry_has(const std::string& name);

/// 列出某个工厂族的所有「规范名」（去重、稳定顺序）。family 为空时列全部。
std::vector<std::string> registry_names(const std::string& family = "");

/// 取某个工厂族的完整条目（去重、稳定顺序）。
std::vector<RegistryEntry> registry_entries(const std::string& family = "");

/// 自检：检查 `requires` 声明的依赖是否都已注册。返回缺失项列表（空 = 通过）。
std::vector<std::string> check_registry();

/// 生成人类可读的注册表总表（CLI catalog 与文档用）。family 为空时列全部。
std::string registry_catalog(const std::string& family = "");

// 注册宏（TOD_RT_DEFINE_RECIPE / TOD_RT_REGISTER_* / TOD_RT_META）定义在 factory.hpp：
// 它们要同时绑定"元数据"与"工厂函数"，把两件事写在一起才能保证不会只填一半。

}  // namespace todrt
