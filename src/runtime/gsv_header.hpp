// gsv_header.hpp — GSV1 容器头部原始字段解析 (A1 最小版; A2 扩展为完整目录解析 loader)
//
// 格式规范见 tools/convert.py 模块 docstring (唯一权威):
//   '<4s' magic='GSV1' | '<I' version=1 | '<I' flags(bit0=有fp16段)
//   '<H'+utf8 名称 | '<I'+JSON config | '<I' n_tensors | 目录项×n
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace gsv::rt {

inline constexpr uint32_t GSV_VERSION = 1;
inline constexpr uint32_t GSV_FLAG_HAS_F16 = 1u << 0;

struct GsvRawHeader {
  uint32_t version = 0;
  uint32_t flags = 0;
  std::string name;
  std::string config_json;
  uint32_t n_tensors = 0;
};

// 解析文件前缀的定长头部字段。data 至少要有完整头部字节；
// 不合法（magic/version/截断）抛 std::runtime_error。
GsvRawHeader parse_gsv_raw_header(const void* data, size_t size);

// 便捷封装: 打开路径读前缀并解析。
GsvRawHeader load_gsv_raw_header(const char* path);

}  // namespace gsv::rt
