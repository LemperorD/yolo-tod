// json.hpp —— 极简 JSON 读写（无第三方依赖）
//
// 为什么自带而不引 nlohmann/json：实机部署（Jetson/Orin）常常离网、工具链受限，
// 一个头文件 + 一个 cpp 更容易嵌进既有工程；而且部署配置结构很简单，
// 不值得为此多一个包管理依赖。
//
// 支持：object / array / string / number / bool / null；注释（// 与 /* */）与尾随逗号
// 被容忍（工程配置文件手写时很常见）。不支持：\u 代理对以外的转义细节、NaN/Inf。
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "todrt/core.hpp"

namespace todrt::json {

class Value;
using Object = std::map<std::string, Value>;
using Array = std::vector<Value>;

enum class Type { kNull = 0, kBool = 1, kNumber = 2, kString = 3, kArray = 4, kObject = 5 };

/// 一个 JSON 值。数值统一按 double 存储（配置里都是小整数/小数）。
class Value {
 public:
  Value() = default;
  Value(std::nullptr_t) {}
  // 注意：必须同时初始化 bool_ —— as_bool() 读的是 bool_，只写 num_ 会让所有
  // JSON 布尔值静默变成 false（真实踩过的坑，见 json.cpp 的单元自检）。
  Value(bool b) : type_(Type::kBool), num_(b ? 1.0 : 0.0), bool_(b) {}
  Value(int v) : type_(Type::kNumber), num_(v) {}
  Value(int64_t v) : type_(Type::kNumber), num_(static_cast<double>(v)) {}
  Value(double v) : type_(Type::kNumber), num_(v) {}
  Value(const char* s) : type_(Type::kString), str_(s) {}
  Value(std::string s) : type_(Type::kString), str_(std::move(s)) {}
  Value(Array a) : type_(Type::kArray), arr_(std::move(a)) {}
  Value(Object o) : type_(Type::kObject), obj_(std::move(o)) {}

  Type type() const { return type_; }
  bool is_null() const { return type_ == Type::kNull; }
  bool is_object() const { return type_ == Type::kObject; }
  bool is_array() const { return type_ == Type::kArray; }
  bool is_string() const { return type_ == Type::kString; }
  bool is_number() const { return type_ == Type::kNumber; }
  bool is_bool() const { return type_ == Type::kBool; }

  /// 取值带默认值：字段缺失/类型不符时返回 def，不抛异常（配置容错）。
  bool as_bool(bool def = false) const;
  double as_double(double def = 0.0) const;
  int as_int(int def = 0) const;
  int64_t as_int64(int64_t def = 0) const;
  std::string as_string(const std::string& def = "") const;
  /// 字符串或数值都接受（配置里 "0.25" 与 0.25 写法都常见）。
  float as_float(float def = 0.f) const;

  /// 严格取值：字段必须存在且类型正确，否则抛 TritError（用于必填项）。
  double require_double(const std::string& key) const;
  int require_int(const std::string& key) const;
  std::string require_string(const std::string& key) const;

  /// 对象字段访问；不存在时返回 null 值。
  const Value& operator[](const std::string& key) const;
  const Value& at(size_t i) const;
  size_t size() const;
  bool contains(const std::string& key) const;

  const Object& object() const { return obj_; }
  const Array& array() const { return arr_; }

  /// 序列化（紧凑或缩进 2 空格）。
  std::string dump(bool pretty = false, int indent = 0) const;

 private:
  Type type_ = Type::kNull;
  double num_ = 0.0;
  bool bool_ = false;
  std::string str_;
  Array arr_;
  Object obj_;

  static const Value& null_value();
};

/// 解析文本；失败抛 TritError（含行列号）。
Value parse(const std::string& text);
/// 读文件并解析；文件不存在抛 TritError。
Value parse_file(const std::string& path);

}  // namespace todrt::json
