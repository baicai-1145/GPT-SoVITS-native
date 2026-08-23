// gsv_tool.cpp — .gsv 文件检查 CLI: 打印头部信息, --list dump 张量目录
#include "gsv_loader.hpp"

#include <cstdio>
#include <cstring>
#include <string>

int main(int argc, char** argv) {
  bool list = false;
  const char* path = nullptr;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--list") == 0) {
      list = true;
    } else if (!path) {
      path = argv[i];
    } else {
      path = nullptr;
      break;
    }
  }
  if (!path) {
    std::fprintf(stderr, "用法: %s [--list] <file.gsv>\n", argv[0]);
    return 2;
  }
  try {
    const gsv::rt::GsvFile f(path);
    const auto& h = f.header();
    std::printf("file:        %s\n", f.path().c_str());
    std::printf("size:        %llu bytes\n", static_cast<unsigned long long>(f.file_size()));
    std::printf("version:     %u\n", h.version);
    std::printf("flags:       0x%x%s\n", h.flags,
                (h.flags & gsv::rt::GSV_FLAG_HAS_F16) ? " (含 fp16 段)" : "");
    std::printf("name:        %s\n", h.name.c_str());
    std::printf("n_tensors:   %u\n", h.n_tensors);
    if (!list) {
      std::printf("config_json: %zu bytes\n%s\n", h.config_json.size(), h.config_json.c_str());
    } else {
      for (const auto& t : f.tensors()) {
        std::printf("%-64s dims=[", t.name.c_str());
        for (size_t i = 0; i < t.dims.size(); ++i)
          std::printf("%s%u", i ? "," : "", t.dims[i]);
        std::printf("] src_dtype=%s%s\n",
                    t.src_dtype == gsv::rt::DType::F16 ? "fp16" : "fp32",
                    t.has_f16() ? " +f16seg" : "");
      }
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "错误: %s\n", e.what());
    return 1;
  }
}
