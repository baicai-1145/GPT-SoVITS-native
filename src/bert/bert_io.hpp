// bert_io.hpp — B8: gsv 张量读取 + [T,C] 序列矩阵 + dump 钩子 (自足, 不依赖 src/sovits)
#pragma once
#include <sys/stat.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

#include "kern/accel.hpp"
#include "runtime/gsv_loader.hpp"

namespace gsv::bert {

// 行主 [rows=T 序列, cols=C 通道]
struct Matrix {
  size_t rows = 0, cols = 0;
  std::vector<float> d;
  void reset(size_t r, size_t c) {
    rows = r;
    cols = c;
    d.assign(r * c, 0.f);
  }
  size_t size() const { return d.size(); }
};

// fp16/fp32 任一段无损升位读取
inline void load_tensor_f32(const rt::GsvFile& f, std::string_view name,
                            std::vector<float>& dst) {
  const auto* t = f.tensor(name);
  if (!t) {
    std::fprintf(stderr, "bert: missing tensor %.*s\n", int(name.size()),
                 name.data());
    std::abort();
  }
  dst.resize(t->numel());
  if (t->has_f16()) {
    kern::accel::f16_to_f32(t->data_f16_raw(), dst.data(), t->numel());
  } else {
    const float* src = t->data_f32();
    for (size_t i = 0; i < t->numel(); ++i) dst[i] = src[i];
  }
}

inline void load_tensor_f32(const rt::GsvFile& f, std::string_view name,
                            std::vector<float>& dst,
                            std::initializer_list<size_t> expect_dims) {
#ifndef NDEBUG
  const auto* t = f.tensor(name);
  if (!t || t->dims.size() != expect_dims.size()) {
    std::fprintf(stderr, "bert: rank mismatch %.*s\n", int(name.size()),
                 name.data());
    std::abort();
  }
  size_t idx = 0;
  for (size_t e : expect_dims)
    if (t->dims[idx++] != e) {
      std::fprintf(stderr, "bert: dims mismatch %.*s\n", int(name.size()),
                   name.data());
      std::abort();
    }
#else
  (void)expect_dims;
#endif
  load_tensor_f32(f, name, dst);
}

// 调试 dump (路径为空则关闭); 与 python 导出同格式 f32 raw + .shape
class Dumper {
 public:
  Dumper() = default;
  explicit Dumper(std::string dir) : dir_(std::move(dir)) {
    if (!dir_.empty()) std::filesystem::create_directories(dir_);
  }
  void dump(std::string_view name, const Matrix& x) const {
    if (dir_.empty()) return;
    std::string base = dir_ + "/" + std::string(name);
    std::ofstream f(base + ".bin", std::ios::binary);
    f.write(reinterpret_cast<const char*>(x.d.data()),
            std::streamsize(x.d.size() * sizeof(float)));
    std::ofstream fs(base + ".shape");
    fs << x.rows << " " << x.cols;
  }

 private:
  std::string dir_;
};

}  // namespace gsv::bert
