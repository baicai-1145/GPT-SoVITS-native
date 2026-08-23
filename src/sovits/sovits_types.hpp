// sovits_types.hpp — SoVITS 引擎公共类型与权重装载辅助
// 数值纪律: 第一步全 fp32 计算; fp16 存储权重加载时无损升位; WN 融合产物已是真 fp32。
#pragma once

#include "runtime/gsv_loader.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <sys/stat.h>
#include <string>
#include <vector>

namespace gsv::sovits {

// [C, T] 行主连续 (data[c*T + t])
struct Tensor2D {
  size_t C = 0, T = 0;
  std::vector<float> d;

  Tensor2D() = default;
  Tensor2D(size_t c, size_t t) : C(c), T(t), d(c * t, 0.f) {}

  void reset(size_t c, size_t t) {
    C = c;
    T = t;
    d.assign(c * t, 0.f);
  }
  float* row(size_t c) { return d.data() + c * T; }
  const float* row(size_t c) const { return d.data() + c * T; }
};

inline void load_tensor_f32(const rt::GsvFile& f, std::string_view name,
                            std::vector<float>& dst) {
  const auto* t = f.tensor(name);
  if (!t) throw std::runtime_error("missing tensor: " + std::string(name));
  dst.resize(t->numel());
  if (t->has_f16()) {
    kern::accel::f16_to_f32(t->data_f16_raw(), dst.data(), t->numel());
  } else {
    const float* src = t->data_f32();
    for (size_t i = 0; i < t->numel(); ++i) dst[i] = src[i];
  }
}

// 严格校验: dims 逐维相等 (防 [a,b] vs [b,a] 错位)
inline void check_dims(const rt::TensorView* t, std::string_view name,
                       std::initializer_list<size_t> expect) {
  if (t->dims.size() != expect.size())
    throw std::runtime_error("tensor " + std::string(name) + ": rank mismatch");
  size_t idx = 0;
  for (size_t v : expect) {
    if (static_cast<size_t>(t->dims[idx]) != v)
      throw std::runtime_error("tensor " + std::string(name) +
                               ": dim mismatch");
    ++idx;
  }
}

inline void load_tensor_f32(const rt::GsvFile& f, std::string_view name,
                            std::vector<float>& dst,
                            std::initializer_list<size_t> expect_dims) {
  const auto* t = f.tensor(name);
  if (!t) throw std::runtime_error("missing tensor: " + std::string(name));
  check_dims(t, name, expect_dims);
  load_tensor_f32(f, name, dst);
}

// ---- debug dump: 与 tools/export_sovits_fixtures.py 同格式的 f32 raw + .shape ----
class Dumper {
 public:
  void enable(std::string dir) {
    ::mkdir(dir.c_str(), 0755);  // 已存在则忽略
    dir_ = std::move(dir);
  }
  bool enabled() const { return !dir_.empty(); }

  // 只 dump 白名单内的名字 (空白名单 = 全量)
  void allow(const std::vector<std::string>& names) { allow_ = names; }

  void dump(std::string_view name, const float* data,
            std::initializer_list<size_t> dims) const {
    if (!enabled()) return;
    if (!allow_.empty()) {
      bool hit = false;
      for (const auto& a : allow_)
        if (a == name) { hit = true; break; }
      if (!hit) return;
    }
    std::string path = dir_ + "/" + std::string(name);
    FILE* fp = fopen((path + ".bin").c_str(), "wb");
    if (!fp) return;
    size_t n = 1;
    for (size_t v : dims) n *= v;
    fwrite(data, sizeof(float), n, fp);
    fclose(fp);
    FILE* fs = fopen((path + ".shape").c_str(), "w");
    if (fs) {
      bool first = true;
      for (size_t v : dims) {
        fprintf(fs, "%s%zu", first ? "" : " ", v);
        first = false;
      }
      fprintf(fs, "\n");
      fclose(fs);
    }
  }
  void dump(std::string_view name, const Tensor2D& t) const {
    dump(name, t.d.data(), {t.C, t.T});
  }

 private:
  std::string dir_;
  std::vector<std::string> allow_;
};

}  // namespace gsv::sovits
