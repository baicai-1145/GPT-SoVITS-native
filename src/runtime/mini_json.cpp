// mini_json.cpp — 递归下降 JSON 解析（深度受限, 足够 convert.py 的 config 结构）
#include "mini_json.hpp"

#include <cmath>
#include <cerrno>
#include <cstdlib>
#include <stdexcept>
#include <string_view>

namespace gsv::rt::json {

namespace {

constexpr int kMaxDepth = 64;

struct Parser {
  const char* p;
  const char* end;

  [[noreturn]] static void err(const char* msg) {
    throw std::runtime_error(std::string("json: ") + msg);
  }

  void ws() {
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) ++p;
  }

  char peek() {
    if (p >= end) err("意外到达输入末尾");
    return *p;
  }

  void expect(char c) {
    if (p >= end || *p != c) err("期望字符不匹配");
    ++p;
  }

  bool lit(std::string_view s) {
    if (static_cast<size_t>(end - p) >= s.size() && std::string_view(p, s.size()) == s) {
      p += s.size();
      return true;
    }
    return false;
  }

  static void append_utf8(std::string& out, uint32_t cp) {
    if (cp <= 0x7F) {
      out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FF) {
      out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp <= 0xFFFF) {
      out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
      out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
      out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
      out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
  }

  uint32_t hex4() {
    if (end - p < 4) err("\\u 转义截断");
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i) {
      char c = *p++;
      v <<= 4;
      if (c >= '0' && c <= '9') v |= static_cast<uint32_t>(c - '0');
      else if (c >= 'a' && c <= 'f') v |= static_cast<uint32_t>(c - 'a' + 10);
      else if (c >= 'A' && c <= 'F') v |= static_cast<uint32_t>(c - 'A' + 10);
      else err("非法十六进制");
    }
    return v;
  }

  std::string parse_string() {
    expect('"');
    std::string out;
    while (true) {
      if (p >= end) err("字符串未闭合");
      char c = *p++;
      if (c == '"') break;
      if (c != '\\') {
        out.push_back(c);
        continue;
      }
      if (p >= end) err("转义截断");
      char e = *p++;
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
          uint32_t cp = hex4();
          if (cp >= 0xD800 && cp <= 0xDBFF && end - p >= 6 && p[0] == '\\' && p[1] == 'u') {
            p += 2;
            uint32_t lo = hex4();
            if (lo >= 0xDC00 && lo <= 0xDFFF)
              cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
            else
              err("非法代理对");
          }
          append_utf8(out, cp);
          break;
        }
        default: err("未知转义");
      }
    }
    return out;
  }

  JValue parse_number() {
    const char* start = p;
    if (p < end && *p == '-') ++p;
    while (p < end && *p >= '0' && *p <= '9') ++p;
    bool is_float = false;
    if (p < end && *p == '.') {
      is_float = true;
      ++p;
      while (p < end && *p >= '0' && *p <= '9') ++p;
    }
    if (p < end && (*p == 'e' || *p == 'E')) {
      is_float = true;
      ++p;
      if (p < end && (*p == '+' || *p == '-')) ++p;
      while (p < end && *p >= '0' && *p <= '9') ++p;
    }
    std::string tok(start, p);
    JValue v;
    if (!is_float) {
      errno = 0;
      char* e = nullptr;
      v.type = JType::Int;
      v.i = std::strtoll(tok.c_str(), &e, 10);
      if (e != tok.c_str() + tok.size()) err("整数解析失败");
      v.d = static_cast<double>(v.i);
    } else {
      errno = 0;
      char* e = nullptr;
      v.type = JType::Float;
      v.d = std::strtod(tok.c_str(), &e);
      if (e != tok.c_str() + tok.size()) err("浮点解析失败");
      v.i = static_cast<int64_t>(v.d);
    }
    return v;
  }

  JValue parse_value(int depth) {
    if (depth > kMaxDepth) err("嵌套过深");
    ws();
    JValue v;
    const char c = peek();
    if (c == '{') {
      v.type = JType::Object;
      ++p;
      ws();
      if (peek() == '}') {
        ++p;
        return v;
      }
      while (true) {
        ws();
        std::string key = parse_string();
        ws();
        expect(':');
        v.obj.emplace_back(std::move(key), parse_value(depth + 1));
        ws();
        const char d = peek();
        if (d == ',') {
          ++p;
          continue;
        }
        if (d == '}') {
          ++p;
          return v;
        }
        err("对象中期望 ',' 或 '}'");
      }
    }
    if (c == '[') {
      v.type = JType::Array;
      ++p;
      ws();
      if (peek() == ']') {
        ++p;
        return v;
      }
      while (true) {
        v.arr.push_back(parse_value(depth + 1));
        ws();
        const char d = peek();
        if (d == ',') {
          ++p;
          continue;
        }
        if (d == ']') {
          ++p;
          return v;
        }
        err("数组中期望 ',' 或 ']'");
      }
    }
    if (c == '"') {
      v.type = JType::String;
      v.s = parse_string();
      return v;
    }
    if (lit("true")) {
      v.type = JType::Bool;
      v.b = true;
      return v;
    }
    if (lit("false")) {
      v.type = JType::Bool;
      v.b = false;
      return v;
    }
    if (lit("null")) return v;  // Null
    if (c == '-' || (c >= '0' && c <= '9')) return parse_number();
    err("无法识别的值");
  }
};

}  // namespace

const JValue* JValue::find(std::string_view key) const {
  if (type != JType::Object) return nullptr;
  for (const auto& kv : obj)
    if (kv.first == key) return &kv.second;
  return nullptr;
}

double JValue::as_double() const {
  if (type == JType::Int || type == JType::Float) return d;
  throw std::runtime_error("json: 期望数值");
}

int64_t JValue::as_int() const {
  if (type == JType::Int || type == JType::Float) return i;
  throw std::runtime_error("json: 期望整数");
}

const std::string& JValue::as_string() const {
  if (type != JType::String) throw std::runtime_error("json: 期望字符串");
  return s;
}

JValue parse(const char* data, size_t n) {
  Parser ps{data, data + n};
  JValue v = ps.parse_value(0);
  ps.ws();
  if (ps.p != ps.end) Parser::err("文档末尾有多余内容");
  return v;
}

}  // namespace gsv::rt::json
