// sovits_types.hpp — SoVITS 引擎公共类型与权重装载辅助
// 数值纪律: 第一步全 fp32 计算; fp16 存储权重加载时无损升位; WN 融合产物已是真 fp32。
#pragma once

#include "runtime/gsv_loader.hpp"
#include "kern/accel.hpp"

#include <arm_neon.h>
#include <algorithm>
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

// ---- E5-P2: fp16 位型辅助 (w_f16 单副本 + 融合 im2col 出 f16) ----

// fp32→fp16 就近舍入单元素 (位型)
inline uint16_t f16_round(float v) {
  const __fp16 h = static_cast<__fp16>(v);
  uint16_t b;
  __builtin_memcpy(&b, &h, sizeof b);
  return b;
}

// 权重按位型读为 fp16 (源即 fp16 则零转换; fp32 源则舍入一次)
inline void load_tensor_f16(const rt::GsvFile& f, std::string_view name,
                            std::vector<uint16_t>& dst) {
  const auto* t = f.tensor(name);
  if (!t) throw std::runtime_error("missing tensor: " + std::string(name));
  dst.resize(t->numel());
  if (t->has_f16()) {
    const uint16_t* src = t->data_f16_raw();
    for (size_t i = 0; i < t->numel(); ++i) dst[i] = src[i];
  } else {
    kern::f32_to_f16(t->data_f32(), dst.data(), t->numel());
  }
}

inline void load_tensor_f16(const rt::GsvFile& f, std::string_view name,
                            std::vector<uint16_t>& dst,
                            std::initializer_list<size_t> expect_dims) {
  const auto* t = f.tensor(name);
  if (!t) throw std::runtime_error("missing tensor: " + std::string(name));
  check_dims(t, name, expect_dims);
  load_tensor_f16(f, name, dst);
}

// fp32 特征图 [C,T] → fp16 转置 [T,C] 行主 (fp16 GEMM 的 RHS B_T)
inline void cast_transpose_f16(const float* x, size_t C, size_t T,
                               std::vector<uint16_t>& out_T) {
  out_T.resize(T * C);
  thread_local std::vector<uint16_t> xf16;
  xf16.resize(C * T);
  kern::f32_to_f16(x, xf16.data(), C * T);
  for (size_t c = 0; c < C; ++c) {
    const uint16_t* xr = xf16.data() + c * T;
    for (size_t t = 0; t < T; ++t) out_T[t * C + c] = xr[t];
  }
}

// 融合 im2col + fp16 转置: 直接产出 col_T[T, C*k] fp16。
// 列序与权重 [out, in*k] 一致: p = c*k + kk。dil==1 中段 NEON 向量化;
// dil>1 无分支慢径。边界零区由 assign 清零。
inline void im2col_cast_transpose_f16(const float* x, size_t C, size_t T,
                                      size_t k, size_t dil,
                                      std::vector<uint16_t>& out_T) {
  const size_t K = C * k;
  out_T.assign(T * K, 0);
  const long pad_l = static_cast<long>((k - 1) * dil) / 2;
  uint16_t* out = out_T.data();
  for (size_t t = 0; t < T; ++t) {
    uint16_t* orow = out + t * K;
    const long s0 = static_cast<long>(t) - pad_l;
    // 有效 kk 范围: 源索引 s = s0 + kk·dil ∈ [0, T)
    const long kk_lo = (s0 >= 0) ? 0 : (-s0 + static_cast<long>(dil) - 1) /
                                           static_cast<long>(dil);
    const long span = static_cast<long>(T) - s0;
    const long kk_hi = std::min(static_cast<long>(k),
                                (span + static_cast<long>(dil) - 1) /
                                    static_cast<long>(dil));
    for (size_t c = 0; c < C; ++c) {
      const float* xr = x + c * T;
      uint16_t* dst = orow + c * k;
      if (dil == 1) {
        const float* win = xr + s0 + kk_lo;
        long n = kk_hi - kk_lo;
        long i = 0;
        if (s0 >= 0 && s0 + static_cast<long>(k) <= static_cast<long>(T)) {
          for (; i + 16 <= n; i += 16) {
            const float32x4_t l0 = vld1q_f32(win + i);
            const float32x4_t h0 = vld1q_f32(win + i + 4);
            const float32x4_t l1 = vld1q_f32(win + i + 8);
            const float32x4_t h1 = vld1q_f32(win + i + 12);
            vst1_u16(dst + i, vreinterpret_u16_f16(vcvt_f16_f32(l0)));
            vst1_u16(dst + i + 4, vreinterpret_u16_f16(vcvt_f16_f32(h0)));
            vst1_u16(dst + i + 8, vreinterpret_u16_f16(vcvt_f16_f32(l1)));
            vst1_u16(dst + i + 12, vreinterpret_u16_f16(vcvt_f16_f32(h1)));
          }
          for (; i + 8 <= n; i += 8) {
            vst1q_u16(dst + i,
                      vreinterpretq_u16_f16(
                          vcombine_f16(vcvt_f16_f32(vld1q_f32(win + i)),
                                       vcvt_f16_f32(vld1q_f32(win + i + 4)))));
          }
        } else {
          // 跨边界窗口: 有效段也可能较长, 同样向量化
          for (; i + 8 <= n; i += 8) {
            vst1q_u16(dst + kk_lo + i,
                      vreinterpretq_u16_f16(
                          vcombine_f16(vcvt_f16_f32(vld1q_f32(win + i)),
                                       vcvt_f16_f32(vld1q_f32(win + i + 4)))));
          }
        }
        for (; i < n; ++i) dst[kk_lo + i] = f16_round(win[i]);
      } else {
        for (long kk = kk_lo; kk < kk_hi; ++kk)
          dst[kk] = f16_round(xr[s0 + kk * static_cast<long>(dil)]);
      }
    }
  }
}

// E5-P2: im2col 直接写入 AMX panel 布局 (省去 col_T 中间缓冲与 pack 趟)。
// 布局与 kern::amx_pack 位级一致: nt=(T+31)/32 个 tile, tile t 的
// 元素 (行 r=t%32, 列 p=c*k+kk) 位于 dst[t*K*64 + p*64 + r*2] (f16 位型),
// 尾块补零由 assign 清零保证。
// 内层 4 行×4 列转置后整 8B 写一行内 4 个行槽 — 与 pack NEON 路径同构。
inline void im2col_to_panel_f16(const float* x, size_t C, size_t T,
                                size_t k, size_t dil,
                                std::vector<uint8_t>& out) {
  const size_t K = C * k;
  const size_t nt = (T + 31) / 32;
  out.assign(nt * K * 64 + 64, 0);
  const uintptr_t pbase = (uintptr_t)out.data();
  uint8_t* dst = out.data() + ((64 - (pbase & 63)) & 63);
  const long pad_l = static_cast<long>((k - 1) * dil) / 2;
  auto src_at = [&](size_t c, long s) {
    return f16_round(x[c * T + static_cast<size_t>(s)]);
  };
  // 标量参考路径: 单行写入 (边界/尾块)
  auto write_row_scalar = [&](size_t t) {
    uint16_t* drow = reinterpret_cast<uint16_t*>(dst + (t / 32) * K * 64 +
                                                 (t % 32) * 2);
    const long s0 = static_cast<long>(t) - pad_l;
    for (size_t c = 0; c < C; ++c) {
      uint16_t* dp = drow + c * k * 32;
      for (long kk = 0; kk < static_cast<long>(k); ++kk) {
        const long s = s0 + kk * static_cast<long>(dil);
        dp[kk * 32] =
            (s >= 0 && s < static_cast<long>(T)) ? src_at(c, s) : 0;
      }
    }
  };
  if (dil != 1) {
    for (size_t t = 0; t < T; ++t) write_row_scalar(t);
    return;
  }
  size_t t = 0;
  // 4 行一组 (仅在完整窗口且不跨 tile 边界时走快径)
  for (; t + 4 <= T; ) {
    bool ok = ((t % 32) + 4 <= 32);
    if (ok) {
      const long s_lo = static_cast<long>(t) - pad_l;
      const long s_hi = static_cast<long>(t + 3) - pad_l + static_cast<long>(k);
      ok = (s_lo >= 0) && (s_hi <= static_cast<long>(T));
    }
    if (!ok) {
      write_row_scalar(t);
      ++t;
      continue;
    }
    uint8_t* d = dst + (t / 32) * K * 64 + (t % 32) * 2;
    for (size_t c = 0; c < C; ++c) {
      // 相邻时间步窗口: 同通道内偏移 +1 (非 +T!)
      const float* w0 = x + c * T + (t - pad_l);
      const float* w1 = w0 + 1;
      const float* w2 = w0 + 2;
      const float* w3 = w0 + 3;
      uint16_t* dp = reinterpret_cast<uint16_t*>(d) + c * k * 32;
      long i = 0;
      for (; i + 4 <= static_cast<long>(k); i += 4) {
        const float16x4_t x0 = vcvt_f16_f32(vld1q_f32(w0 + i));
        const float16x4_t x1 = vcvt_f16_f32(vld1q_f32(w1 + i));
        const float16x4_t x2 = vcvt_f16_f32(vld1q_f32(w2 + i));
        const float16x4_t x3 = vcvt_f16_f32(vld1q_f32(w3 + i));
        float16x4_t o0, o1, o2, o3;
        const float16x4x2_t a01 = vtrn_f16(x0, x1);
        const float16x4x2_t a23 = vtrn_f16(x2, x3);
        const float32x2x2_t b0 = vtrn_f32(
            vreinterpret_f32_f16(a01.val[0]), vreinterpret_f32_f16(a23.val[0]));
        const float32x2x2_t b1 = vtrn_f32(
            vreinterpret_f32_f16(a01.val[1]), vreinterpret_f32_f16(a23.val[1]));
        o0 = vreinterpret_f16_f32(b0.val[0]);
        o1 = vreinterpret_f16_f32(b1.val[0]);
        o2 = vreinterpret_f16_f32(b0.val[1]);
        o3 = vreinterpret_f16_f32(b1.val[1]);
        // o_j = [row0..row3 @ col i+j] → 线 p=(c*k+i+j) 的字节 [rr*2,+8)
        vst1_f16(reinterpret_cast<__fp16*>(dp + (i + 0) * 32), o0);
        vst1_f16(reinterpret_cast<__fp16*>(dp + (i + 1) * 32), o1);
        vst1_f16(reinterpret_cast<__fp16*>(dp + (i + 2) * 32), o2);
        vst1_f16(reinterpret_cast<__fp16*>(dp + (i + 3) * 32), o3);
      }
      for (; i < static_cast<long>(k); ++i) {
        dp[(i)*32 + 0] = f16_round(w0[i]);
        dp[(i)*32 + 1] = f16_round(w1[i]);
        dp[(i)*32 + 2] = f16_round(w2[i]);
        dp[(i)*32 + 3] = f16_round(w3[i]);
      }
    }
    t += 4;
  }
  for (; t < T; ++t) write_row_scalar(t);
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
