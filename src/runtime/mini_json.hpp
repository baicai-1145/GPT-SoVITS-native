// mini_json.hpp — 自写最小 JSON 解析器（仅覆盖 tools/convert.py 写出的 config 结构）
// 支持: object/array/string(含转义与\uXXXX)/int/float/true/false/null。
// 无第三方依赖；解析失败抛 std::runtime_error（带偏移）。
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace gsv::rt::json {

enum class JType : uint8_t { Null, Bool, Int, Float, String, Array, Object };

struct JValue {
  JType type = JType::Null;
  bool b = false;
  int64_t i = 0;
  double d = 0.0;
  std::string s;
  std::vector<JValue> arr;
  std::vector<std::pair<std::string, JValue>> obj;  // 保持出现顺序; 查找线性扫(config 很小)

  bool is(JType t) const { return type == t; }
  // 对象按 key 查找; 非对象或不存在返回 nullptr
  const JValue* find(std::string_view key) const;
  // 便捷取值(类型不符抛异常)
  double as_double() const;
  int64_t as_int() const;
  const std::string& as_string() const;
};

// 从字节缓冲解析单个 JSON 文档; 尾部仅允许空白。失败抛 std::runtime_error。
JValue parse(const char* data, size_t n);

}  // namespace gsv::rt::json
