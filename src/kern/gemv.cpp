// gemv.cpp — NEON 手写 GEMV: fp16 权重精确升位 + fp32 FMA 累加 (4-row 并行展开)
//
// 数值语义:
//   - 激活 x 保持 fp32 不做任何舍入
//   - 权重 fp16 由 vcvt_f32_f16 无损升位到 fp32
//   - vfmaq_f32 全程纯 fp32 乘加累加，累加器与输出全 fp32
//   - 与 torch "fp16存储+fp32计算" 基线数学完全等价，彻底免疫长时间自回归 (1500 步) 的累积雪崩
//
// 性能优化:
//   - 4 行同时计算 (4-way register blocking)，复用向量寄存器并最大化 ILP 流水线
//   - 大矩阵接入线程池 (parallel_for)
#include "kern.hpp"

#include "runtime/threadpool.hpp"

#include <arm_neon.h>
#include <algorithm>

namespace gsv::kern {

void gemv_f16w_f32acc_ref(const uint16_t* w, const float* x, float* y, size_t out, size_t in) {
  for (size_t r = 0; r < out; ++r) {
    const uint16_t* wr = w + r * in;
    float s = 0.0f;
    for (size_t i = 0; i < in; ++i) {
      __fp16 h;
      __builtin_memcpy(&h, &wr[i], sizeof h);
      s += static_cast<float>(h) * x[i];
    }
    y[r] = s;
  }
}

namespace {

inline void gemv_f16w_4rows_kernel(const uint16_t* w, const float* x, float* y, size_t r_begin, size_t r_end, size_t in) {
  const size_t vec_end = in & ~size_t{7};
  size_t r = r_begin;
  for (; r + 4 <= r_end; r += 4) {
    const uint16_t* wr0 = w + r * in;
    const uint16_t* wr1 = w + (r + 1) * in;
    const uint16_t* wr2 = w + (r + 2) * in;
    const uint16_t* wr3 = w + (r + 3) * in;
    const float* xr = x;
    float32x4_t acc00 = vdupq_n_f32(0.0f), acc01 = vdupq_n_f32(0.0f);
    float32x4_t acc10 = vdupq_n_f32(0.0f), acc11 = vdupq_n_f32(0.0f);
    float32x4_t acc20 = vdupq_n_f32(0.0f), acc21 = vdupq_n_f32(0.0f);
    float32x4_t acc30 = vdupq_n_f32(0.0f), acc31 = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i < vec_end; i += 8, wr0 += 8, wr1 += 8, wr2 += 8, wr3 += 8, xr += 8) {
      const float32x4_t x0 = vld1q_f32(xr);
      const float32x4_t x1 = vld1q_f32(xr + 4);
      
      const float16x8_t w0 = vld1q_f16(reinterpret_cast<const __fp16*>(wr0));
      acc00 = vfmaq_f32(acc00, vcvt_f32_f16(vget_low_f16(w0)), x0);
      acc01 = vfmaq_f32(acc01, vcvt_f32_f16(vget_high_f16(w0)), x1);

      const float16x8_t w1 = vld1q_f16(reinterpret_cast<const __fp16*>(wr1));
      acc10 = vfmaq_f32(acc10, vcvt_f32_f16(vget_low_f16(w1)), x0);
      acc11 = vfmaq_f32(acc11, vcvt_f32_f16(vget_high_f16(w1)), x1);

      const float16x8_t w2 = vld1q_f16(reinterpret_cast<const __fp16*>(wr2));
      acc20 = vfmaq_f32(acc20, vcvt_f32_f16(vget_low_f16(w2)), x0);
      acc21 = vfmaq_f32(acc21, vcvt_f32_f16(vget_high_f16(w2)), x1);

      const float16x8_t w3 = vld1q_f16(reinterpret_cast<const __fp16*>(wr3));
      acc30 = vfmaq_f32(acc30, vcvt_f32_f16(vget_low_f16(w3)), x0);
      acc31 = vfmaq_f32(acc31, vcvt_f32_f16(vget_high_f16(w3)), x1);
    }
    float s0 = vaddvq_f32(vaddq_f32(acc00, acc01));
    float s1 = vaddvq_f32(vaddq_f32(acc10, acc11));
    float s2 = vaddvq_f32(vaddq_f32(acc20, acc21));
    float s3 = vaddvq_f32(vaddq_f32(acc30, acc31));
    for (; i < in; ++i, ++wr0, ++wr1, ++wr2, ++wr3, ++xr) {
      __fp16 h0, h1, h2, h3;
      __builtin_memcpy(&h0, wr0, sizeof h0);
      __builtin_memcpy(&h1, wr1, sizeof h1);
      __builtin_memcpy(&h2, wr2, sizeof h2);
      __builtin_memcpy(&h3, wr3, sizeof h3);
      s0 += static_cast<float>(h0) * (*xr);
      s1 += static_cast<float>(h1) * (*xr);
      s2 += static_cast<float>(h2) * (*xr);
      s3 += static_cast<float>(h3) * (*xr);
    }
    y[r] = s0;
    y[r + 1] = s1;
    y[r + 2] = s2;
    y[r + 3] = s3;
  }
  for (; r < r_end; ++r) {
    const uint16_t* wr = w + r * in;
    const float* xr = x;
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i < vec_end; i += 8, wr += 8, xr += 8) {
      const float16x8_t w8 = vld1q_f16(reinterpret_cast<const __fp16*>(wr));
      acc0 = vfmaq_f32(acc0, vcvt_f32_f16(vget_low_f16(w8)), vld1q_f32(xr));
      acc1 = vfmaq_f32(acc1, vcvt_f32_f16(vget_high_f16(w8)), vld1q_f32(xr + 4));
    }
    float s = vaddvq_f32(acc0) + vaddvq_f32(acc1);
    for (; i < in; ++i, ++wr, ++xr) {
      __fp16 h;
      __builtin_memcpy(&h, wr, sizeof h);
      s += static_cast<float>(h) * (*xr);
    }
    y[r] = s;
  }
}

}  // namespace

void gemv_f16w_f32acc(const uint16_t* w, const float* x, float* y, size_t out, size_t in) {
  if (out * in < 1u << 18) {
    gemv_f16w_4rows_kernel(w, x, y, 0, out, in);
    return;
  }
  // E11-4: 全核联合派发 (P+E 双池)
  const size_t p = rt::p_core_count();
  const size_t e = rt::gemv_use_e_cores() ? rt::e_core_count() : 0;
  const size_t total_workers = p + e;
  const size_t grain = (out + total_workers - 1) / total_workers;
  rt::parallel_for_full(
      out, grain, [&](size_t b, size_t end) {
        gemv_f16w_4rows_kernel(w, x, y, b, end, in);
      });
}

}  // namespace gsv::kern
