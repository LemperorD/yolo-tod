// json.cpp —— 极简 JSON 解析/序列化（宽容注释与尾随逗号）
#include "todrt/json.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace todrt::json {

namespace {
const Value kNull;

/// 把数字转成最短往返表示（配置文件的键值要能被人读）。
std::string num_to_string(double v) {
  if (std::isfinite(v) && v == static_cast<double>(static_cast<int64_t>(v)) &&
      std::abs(v) < 1e15) {
    return std::to_string(static_cast<int64_t>(v));
  }
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.10g", v);
  return buf;
}

std::string escape(const std::string& s) {
  std::string o;
  o.reserve(s.size() + 2);
  for (unsigned char c : s) {
    switch (c) {
      case '"': o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\n': o += "\\n"; break;
      case '\r': o += "\\r"; break;
      case '\t': o += "\\t"; break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          o += buf;
        } else {
          o.push_back(static_cast<char>(c));
        }
    }
  }
  return o;
}
}  // namespace

// ------------------------------------------------------------------ 取值

bool Value::as_bool(bool def) const {
  switch (type_) {
    case Type::kBool: return bool_;
    case Type::kNumber: return num_ != 0.0;
    case Type::kString:
      if (str_ == "true" || str_ == "1" || str_ == "yes" || str_ == "on") return true;
      if (str_ == "false" || str_ == "0" || str_ == "no" || str_ == "off") return false;
      return def;
    default: return def;
  }
}

double Value::as_double(double def) const {
  if (type_ == Type::kNumber) return num_;
  if (type_ == Type::kBool) return bool_ ? 1.0 : 0.0;
  if (type_ == Type::kString) {
    // 容忍 "0.25" 这种字符串数字
    char* end = nullptr;
    const double v = std::strtod(str_.c_str(), &end);
    if (end && end != str_.c_str() && *end == '\0') return v;
  }
  return def;
}

int Value::as_int(int def) const { return static_cast<int>(as_double(def)); }
int64_t Value::as_int64(int64_t def) const { return static_cast<int64_t>(as_double(double(def))); }

std::string Value::as_string(const std::string& def) const {
  switch (type_) {
    case Type::kString: return str_;
    case Type::kNumber: return num_to_string(num_);
    case Type::kBool: return bool_ ? "true" : "false";
    case Type::kNull: return def;
    default: return def;
  }
}

float Value::as_float(float def) const { return static_cast<float>(as_double(def)); }

double Value::require_double(const std::string& key) const {
  const Value& v = (*this)[key];
  if (v.is_null()) throw TritError("配置项缺失：" + key);
  if (!v.is_number() && !v.is_string()) throw TritError("配置项类型错误（需要数值）：" + key);
  return v.as_double();
}

int Value::require_int(const std::string& key) const {
  return static_cast<int>(require_double(key));
}

std::string Value::require_string(const std::string& key) const {
  const Value& v = (*this)[key];
  if (!v.is_string()) throw TritError("配置项缺失或类型错误（需要字符串）：" + key);
  return v.str_;
}

const Value& Value::operator[](const std::string& key) const {
  if (type_ != Type::kObject) return kNull;
  auto it = obj_.find(key);
  return it == obj_.end() ? kNull : it->second;
}

const Value& Value::at(size_t i) const {
  if (type_ != Type::kArray || i >= arr_.size()) return kNull;
  return arr_[i];
}

size_t Value::size() const {
  if (type_ == Type::kArray) return arr_.size();
  if (type_ == Type::kObject) return obj_.size();
  return 0;
}

bool Value::contains(const std::string& key) const {
  return type_ == Type::kObject && obj_.count(key) != 0;
}

const Value& Value::null_value() { return kNull; }

// ------------------------------------------------------------------ 序列化

std::string Value::dump(bool pretty, int indent) const {
  const std::string pad(pretty ? static_cast<size_t>(indent) * 2 : 0, ' ');
  const std::string pad_in(pretty ? (static_cast<size_t>(indent) + 1) * 2 : 0, ' ');
  const char* nl = pretty ? "\n" : "";
  switch (type_) {
    case Type::kNull: return "null";
    case Type::kBool: return bool_ ? "true" : "false";
    case Type::kNumber: return num_to_string(num_);
    case Type::kString: return "\"" + escape(str_) + "\"";
    case Type::kArray: {
      if (arr_.empty()) return "[]";
      std::ostringstream oss;
      oss << "[" << nl;
      for (size_t i = 0; i < arr_.size(); ++i) {
        oss << pad_in << arr_[i].dump(pretty, indent + 1);
        if (i + 1 < arr_.size()) oss << ",";
        oss << nl;
      }
      oss << pad << "]";
      return oss.str();
    }
    case Type::kObject: {
      if (obj_.empty()) return "{}";
      std::ostringstream oss;
      oss << "{" << nl;
      size_t i = 0;
      for (const auto& kv : obj_) {
        oss << pad_in << "\"" << escape(kv.first) << "\": " << kv.second.dump(pretty, indent + 1);
        if (++i < obj_.size()) oss << ",";
        oss << nl;
      }
      oss << pad << "}";
      return oss.str();
    }
  }
  return "null";
}

// ------------------------------------------------------------------ 解析

namespace {

class Parser {
 public:
  explicit Parser(const std::string& text) : s_(text) {}

  Value ParseDocument() {
    SkipWs();
    Value v = ParseValue(0);
    SkipWs();
    if (pos_ != s_.size()) Fail("文档结束后仍有多余字符");
    return v;
  }

 private:
  const std::string& s_;
  size_t pos_ = 0;

  [[noreturn]] void Fail(const std::string& msg) const {
    size_t line = 1, col = 1;
    for (size_t i = 0; i < pos_ && i < s_.size(); ++i) {
      if (s_[i] == '\n') {
        ++line;
        col = 1;
      } else {
        ++col;
      }
    }
    std::ostringstream oss;
    oss << "JSON 解析失败（第 " << line << " 行第 " << col << " 列）：" << msg;
    throw TritError(oss.str());
  }

  void SkipWs() {
    while (pos_ < s_.size()) {
      const char c = s_[pos_];
      if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
        ++pos_;
      } else if (c == '/' && pos_ + 1 < s_.size() && s_[pos_ + 1] == '/') {
        while (pos_ < s_.size() && s_[pos_] != '\n') ++pos_;
      } else if (c == '/' && pos_ + 1 < s_.size() && s_[pos_ + 1] == '*') {
        pos_ += 2;
        while (pos_ + 1 < s_.size() && !(s_[pos_] == '*' && s_[pos_ + 1] == '/')) ++pos_;
        if (pos_ + 1 >= s_.size()) Fail("块注释未闭合");
        pos_ += 2;
      } else {
        break;
      }
    }
  }

  char Peek() {
    if (pos_ >= s_.size()) Fail("意外到达文件末尾");
    return s_[pos_];
  }

  void Expect(char c) {
    if (Peek() != c) Fail(std::string("期望字符 '") + c + "'");
    ++pos_;
  }

  Value ParseValue(int depth) {
    if (depth > 64) Fail("嵌套过深");
    SkipWs();
    const char c = Peek();
    switch (c) {
      case '{': return ParseObject(depth);
      case '[': return ParseArray(depth);
      case '"': return Value(ParseString());
      case 't':
        Literal("true");
        return Value(true);
      case 'f':
        Literal("false");
        return Value(false);
      case 'n':
        Literal("null");
        return Value(nullptr);
      default: return ParseNumber();
    }
  }

  void Literal(const char* lit) {
    const size_t n = std::char_traits<char>::length(lit);
    if (s_.compare(pos_, n, lit) != 0) Fail(std::string("非法字面量（期望 ") + lit + "）");
    pos_ += n;
  }

  Value ParseObject(int depth) {
    Expect('{');
    Object obj;
    SkipWs();
    if (Peek() == '}') {
      ++pos_;
      return Value(std::move(obj));
    }
    while (true) {
      SkipWs();
      if (Peek() == '}') {  // 容忍尾随逗号
        ++pos_;
        break;
      }
      if (Peek() != '"') Fail("对象的键必须是字符串");
      const std::string key = ParseString();
      SkipWs();
      Expect(':');
      obj[key] = ParseValue(depth + 1);
      SkipWs();
      const char c = Peek();
      if (c == ',') {
        ++pos_;
        continue;
      }
      if (c == '}') {
        ++pos_;
        break;
      }
      Fail("对象里期望 ',' 或 '}'");
    }
    return Value(std::move(obj));
  }

  Value ParseArray(int depth) {
    Expect('[');
    Array arr;
    SkipWs();
    if (Peek() == ']') {
      ++pos_;
      return Value(std::move(arr));
    }
    while (true) {
      SkipWs();
      if (Peek() == ']') {  // 容忍尾随逗号
        ++pos_;
        break;
      }
      arr.push_back(ParseValue(depth + 1));
      SkipWs();
      const char c = Peek();
      if (c == ',') {
        ++pos_;
        continue;
      }
      if (c == ']') {
        ++pos_;
        break;
      }
      Fail("数组里期望 ',' 或 ']'");
    }
    return Value(std::move(arr));
  }

  std::string ParseString() {
    Expect('"');
    std::string out;
    while (true) {
      if (pos_ >= s_.size()) Fail("字符串未闭合");
      const char c = s_[pos_++];
      if (c == '"') break;
      if (c != '\\') {
        out.push_back(c);
        continue;
      }
      if (pos_ >= s_.size()) Fail("转义序列未结束");
      const char e = s_[pos_++];
      switch (e) {
        case '"': out.push_back('"'); break;
        case '\\': out.push_back('\\'); break;
        case '/': out.push_back('/'); break;
        case 'b': out.push_back('\b'); break;
        case 'f': out.push_back('\f'); break;
        case 'n': out.push_back('\n'); break;
        case 'r': out.push_back('\r'); break;
        case 't': out.push_back('\t'); break;
        case 'u': {
          if (pos_ + 4 > s_.size()) Fail("\\u 转义不完整");
          unsigned code = 0;
          for (int i = 0; i < 4; ++i) {
            const char h = s_[pos_++];
            code <<= 4;
            if (h >= '0' && h <= '9') {
              code |= static_cast<unsigned>(h - '0');
            } else if (h >= 'a' && h <= 'f') {
              code |= static_cast<unsigned>(h - 'a' + 10);
            } else if (h >= 'A' && h <= 'F') {
              code |= static_cast<unsigned>(h - 'A' + 10);
            } else {
              Fail("非法十六进制字符");
            }
          }
          // UTF-8 编码（不处理代理对：配置文件里不会出现）
          if (code < 0x80) {
            out.push_back(static_cast<char>(code));
          } else if (code < 0x800) {
            out.push_back(static_cast<char>(0xC0 | (code >> 6)));
            out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
          } else {
            out.push_back(static_cast<char>(0xE0 | (code >> 12)));
            out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
          }
          break;
        }
        default: Fail("未知转义字符");
      }
    }
    return out;
  }

  Value ParseNumber() {
    const size_t start = pos_;
    if (pos_ < s_.size() && (s_[pos_] == '-' || s_[pos_] == '+')) ++pos_;
    while (pos_ < s_.size()) {
      const char c = s_[pos_];
      if ((c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-') {
        ++pos_;
      } else {
        break;
      }
    }
    if (pos_ == start) Fail("非法数值");
    const std::string tok = s_.substr(start, pos_ - start);
    char* end = nullptr;
    const double v = std::strtod(tok.c_str(), &end);
    if (end != tok.c_str() + tok.size()) Fail("非法数值：" + tok);
    return Value(v);
  }
};

}  // namespace

Value parse(const std::string& text) { return Parser(text).ParseDocument(); }

Value parse_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw TritError("无法打开配置文件：" + path);
  std::ostringstream oss;
  oss << in.rdbuf();
  try {
    return parse(oss.str());
  } catch (const TritError& e) {
    throw TritError(std::string(path) + "：" + e.what());
  }
}

}  // namespace todrt::json
