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

class Value {
 public:
  Value() = default;
  Value(std::nullptr_t) {}
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

  bool as_bool(bool def = false) const;
  double as_double(double def = 0.0) const;
  int as_int(int def = 0) const;
  int64_t as_int64(int64_t def = 0) const;
  std::string as_string(const std::string& def = "") const;
  float as_float(float def = 0.f) const;

  double require_double(const std::string& key) const;
  int require_int(const std::string& key) const;
  std::string require_string(const std::string& key) const;

  const Value& operator[](const std::string& key) const;
  const Value& at(size_t i) const;
  size_t size() const;
  bool contains(const std::string& key) const;

  const Object& object() const { return obj_; }
  const Array& array() const { return arr_; }

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

Value parse(const std::string& text);
Value parse_file(const std::string& path);

}
