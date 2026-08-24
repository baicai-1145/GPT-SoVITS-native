// gsv_tool.cpp — .gsv 文件检查 CLI: 打印头部信息, --list dump 张量目录,
// --slin 重写去冗余 (E7 加载治理: 双布局文件剥离大张量的 f32 母本段)。
#include "gsv_loader.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

// 64B 对齐常量与加载器 kAlign 一致
constexpr uint64_t kAlign = 64;

void put_u16(std::vector<uint8_t>& b, uint16_t v) {
  std::memcpy(b.data() + b.size() - 2, &v, 2);
}
void append_u16(std::vector<uint8_t>& b, uint16_t v) {
  b.insert(b.end(), 2, 0);
  put_u16(b, v);
}
void append_u32(std::vector<uint8_t>& b, uint32_t v) {
  b.insert(b.end(), 4, 0);
  std::memcpy(b.data() + b.size() - 4, &v, 4);
}
void append_u64(std::vector<uint8_t>& b, uint64_t v) {
  b.insert(b.end(), 8, 0);
  std::memcpy(b.data() + b.size() - 8, &v, 8);
}

int slim_main(const char* inPath, const char* outPath) {
  const gsv::rt::GsvFile f(inPath);
  const auto& h = f.header();

  // slim 规则: 有 f16 段且 f32 段 >64KB 的张量丢弃 f32 段。
  // 依据 (E7 画像): src_dtype=fp16 的母本恒等于 f16 升位 (bit-exact),
  // 全部引擎对大张量走 has_f16 路径; 无条件读 f32 的仅剩小偏置/标量
  // (ar biases ≤8KB, alpha 标量) 与转换器纪律的 fp32-only 张量
  // (hubert pos_conv), 均在保留侧。
  constexpr uint64_t kKeepF32Below = 65536;
  struct Keep {
    const gsv::rt::TensorView* t;
    bool keep_f32;
    bool keep_f16;
  };
  std::vector<Keep> keeps;
  keeps.reserve(f.tensors().size());
  uint64_t saved = 0;
  for (const auto& t : f.tensors()) {
    const bool keep_f32 =
        t.f32_nbytes != 0 && (!t.has_f16() || t.f32_nbytes <= kKeepF32Below);
    const bool keep_f16 = t.has_f16();
    if (!keep_f32) saved += t.f32_nbytes;
    keeps.push_back({&t, keep_f32, keep_f16});
  }

  // 布局: [0..12) magic/ver/flags; name; config; n_tensors; 目录; 数据段
  // 目录两遍构造 (第一遍求目录大小, 第二遍填最终数据偏移)
  auto entrySize = [](const gsv::rt::TensorView& t) {
    return size_t(2 + t.name.size() + 2 + t.dims.size() * 4 + 2 + 32);
  };
  size_t dirSize = 0;
  for (const auto& k : keeps) dirSize += entrySize(*k.t);

  auto buildDir = [&](uint64_t dataBase, std::vector<uint8_t>* bytes,
                      std::vector<std::array<uint64_t, 3>>* copyList) {
    std::vector<uint8_t> d;
    d.reserve(dirSize);
    uint64_t off = dataBase;
    for (const auto& k : keeps) {
      const auto& t = *k.t;
      append_u16(d, uint16_t(t.name.size()));
      d.insert(d.end(), t.name.begin(), t.name.end());
      append_u16(d, uint16_t(t.dims.size()));
      for (uint32_t dim : t.dims) append_u32(d, dim);
      append_u16(d, uint16_t(t.src_dtype));
      const uint64_t f32off = k.keep_f32 ? off : 0;
      const uint64_t f32n = k.keep_f32 ? t.f32_nbytes : 0;
      append_u64(d, f32off);
      append_u64(d, f32n);
      if (k.keep_f32) {
        copyList->push_back({off, t.f32_offset, t.f32_nbytes});
        off = (off + f32n + kAlign - 1) / kAlign * kAlign;  // 段间 64B 对齐
      }
      const uint64_t f16off = k.keep_f16 ? off : 0;
      const uint64_t f16n = k.keep_f16 ? t.f16_nbytes : 0;
      append_u64(d, f16off);
      append_u64(d, f16n);
      if (k.keep_f16) {
        copyList->push_back({off, t.f16_offset, t.f16_nbytes});
        off = (off + f16n + kAlign - 1) / kAlign * kAlign;
      }
    }
    if (bytes) *bytes = std::move(d);
  };

  // 头部前缀大小 (不含目录)
  size_t prefix = 12 + 2 + h.name.size() + 4 + h.config_json.size() + 4;
  // 第一遍: 求 copyList 总字节以校验; 目录大小已知, 数据起点即可定
  std::vector<std::array<uint64_t, 3>> copyList;
  copyList.reserve(keeps.size() * 2);
  const uint64_t dataBase =
      (prefix + dirSize + kAlign - 1) / kAlign * kAlign;
  buildDir(dataBase, nullptr, &copyList);

  std::FILE* out = std::fopen(outPath, "wb");
  if (!out) {
    std::fprintf(stderr, "错误: 无法写出 %s\n", outPath);
    return 1;
  }
  bool ok = true;
  // 头部
  std::vector<uint8_t> head;
  head.insert(head.end(), {'G', 'S', 'V', '1'});
  append_u32(head, h.version);
  append_u32(head, h.flags);
  append_u16(head, uint16_t(h.name.size()));
  head.insert(head.end(), h.name.begin(), h.name.end());
  append_u32(head, uint32_t(h.config_json.size()));
  head.insert(head.end(), h.config_json.begin(), h.config_json.end());
  append_u32(head, h.n_tensors);
  std::vector<uint8_t> dir;
  std::vector<std::array<uint64_t, 3>> finalCopies;
  buildDir(dataBase, &dir, &finalCopies);
  ok &= std::fwrite(head.data(), 1, head.size(), out) == head.size();
  ok &= std::fwrite(dir.data(), 1, dir.size(), out) == dir.size();
  // 对齐填充
  const uint64_t pad = dataBase - prefix - dir.size();
  static constexpr char zeros[64] = {0};
  if (pad) ok &= std::fwrite(zeros, 1, pad, out) == pad;
  // 数据段顺序拷贝
  const uint8_t* base = f.tensors().empty() ? nullptr : f.tensors()[0].file_base;
  {
    static constexpr char kZeros[kAlign] = {0};
    uint64_t expect = dataBase;
    for (const auto& [dstOff, srcOff, n] : finalCopies) {
      while (expect < dstOff) {  // 段间对齐填充
        const uint64_t g = std::min<uint64_t>(kAlign, dstOff - expect);
        ok &= std::fwrite(kZeros, 1, g, out) == g;
        expect += g;
      }
      ok &= std::fwrite(base + srcOff, 1, n, out) == n;
      expect = dstOff + n;
    }
  }
  if (std::fclose(out) != 0 || !ok) {
    std::fprintf(stderr, "错误: 写出失败 %s\n", outPath);
    return 1;
  }

  // 回读自检: 解析新文件并核对张量数/形状/f16 段一致
  try {
    const gsv::rt::GsvFile chk(outPath);
    if (chk.tensors().size() != f.tensors().size()) throw std::runtime_error("张量数不符");
    for (size_t i = 0; i < chk.tensors().size(); ++i) {
      const auto& a = chk.tensors()[i];
      const auto& b = f.tensors()[i];
      if (a.name != b.name || a.dims != b.dims || a.has_f16() != b.has_f16())
        throw std::runtime_error("目录不一致: " + a.name);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "错误: slim 自检失败: %s\n", e.what());
    return 1;
  }

  const uint64_t endOff = finalCopies.empty()
                              ? dataBase
                              : finalCopies.back()[0] + finalCopies.back()[2];
  std::printf("slim: %s -> %s\n", inPath, outPath);
  std::printf("  原 %.0fMB -> 新 %.0fMB (省 %.0fMB f32 冗余段)\n",
              double(f.file_size()) / 1e6, double(endOff) / 1e6,
              double(saved) / 1e6);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  bool list = false;
  const char* path = nullptr;
  const char* path2 = nullptr;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--list") == 0) {
      list = true;
    } else if (std::strcmp(argv[i], "--slim") == 0) {
      // --slim <in> <out>
      if (i + 2 >= argc) {
        std::fprintf(stderr, "用法: %s --slim <in.gsv> <out.gsv>\n", argv[0]);
        return 2;
      }
      return slim_main(argv[i + 1], argv[i + 2]);
    } else if (!path) {
      path = argv[i];
    } else if (!path2) {
      path2 = argv[i];
    } else {
      path = nullptr;
      break;
    }
  }
  if (!path) {
    std::fprintf(stderr,
                 "用法: %s [--list] <file.gsv>\n"
                 "      %s --slim <in.gsv> <out.gsv>  (去 f32 冗余段, E7)\n",
                 argv[0], argv[0]);
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
        std::printf("] src_dtype=%s%s%s\n",
               t.src_dtype == gsv::rt::DType::F16 ? "fp16" : "fp32",
               t.has_f16() ? " +f16seg" : "",
               (t.f32_nbytes == 0 && t.has_f16()) ? " (slim:无f32段)" : "");
      }
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "错误: %s\n", e.what());
    return 1;
  }
}
