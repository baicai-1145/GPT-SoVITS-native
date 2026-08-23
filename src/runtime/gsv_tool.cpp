// gsv_tool.cpp — .gsv 文件检查 CLI（A1: 打印头部信息; A2 升级为完整目录 dump）
#include "gsv_header.hpp"

#include <cstdio>
#include <cstring>
#include <string>

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "用法: %s <file.gsv>\n", argv[0]);
    return 2;
  }
  try {
    const gsv::rt::GsvRawHeader h = gsv::rt::load_gsv_raw_header(argv[1]);
    std::printf("file:        %s\n", argv[1]);
    std::printf("magic:       GSV1\n");
    std::printf("version:     %u\n", h.version);
    std::printf("flags:       0x%x%s\n", h.flags,
                (h.flags & gsv::rt::GSV_FLAG_HAS_F16) ? " (含 fp16 段)" : "");
    std::printf("name:        %s\n", h.name.c_str());
    std::printf("n_tensors:   %u\n", h.n_tensors);
    std::printf("config_json: %zu bytes\n%s\n", h.config_json.size(), h.config_json.c_str());
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "错误: %s\n", e.what());
    return 1;
  }
}
