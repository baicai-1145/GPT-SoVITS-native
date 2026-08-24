// nn_ops.hpp — SoVITS 引擎逐元素/归一化原语 (全 fp32, 语义对齐 torch)
#pragma once

#include <arm_neon.h>

#include <algorithm>
#include <cmath>

#include "sovits/sovits_types.hpp"

namespace gsv::sovits {

inline void softmax_row_inplace(float* x, size_t n) {
  float mx = x[0];
  for (size_t i = 1; i < n; ++i) mx = std::max(mx, x[i]);
  float s = 0.f;
  for (size_t i = 0; i < n; ++i) {
    x[i] = std::exp(x[i] - mx);
    s += x[i];
  }
  const float inv = 1.f / s;
  for (size_t i = 0; i < n; ++i) x[i] *= inv;
}

// torch modules.LayerNorm: 对最后一维(通道)归一化, eps=1e-5; [C,T] 行主下
// 即对每个时间帧 t 在 C 维上归一化。
inline void layernorm_ct(const Tensor2D& x, const std::vector<float>& gamma,
                         const std::vector<float>& beta, Tensor2D& y) {
  constexpr float kEps = 1e-5f;
  const size_t C = x.C, T = x.T;
  y.reset(C, T);
  for (size_t t = 0; t < T; ++t) {
    float mu = 0.f;
    for (size_t c = 0; c < C; ++c) mu += x.d[c * T + t];
    mu /= static_cast<float>(C);
    float var = 0.f;
    for (size_t c = 0; c < C; ++c) {
      const float d = x.d[c * T + t] - mu;
      var += d * d;
    }
    var /= static_cast<float>(C);
    const float inv = 1.f / std::sqrt(var + kEps);
    for (size_t c = 0; c < C; ++c)
      y.d[c * T + t] = (x.d[c * T + t] - mu) * inv * gamma[c] + beta[c];
  }
}

inline void leaky_relu_io(float* io, size_t n, float slope) {
  size_t i = 0;
  const float32x4_t vs = vdupq_n_f32(slope);
  for (; i + 4 <= n; i += 4) {
    float32x4_t v = vld1q_f32(io + i);
    const uint32x4_t m = vcltzq_f32(v);
    v = vbslq_f32(m, vmulq_f32(v, vs), v);
    vst1q_f32(io + i, v);
  }
  for (; i < n; ++i)
    if (io[i] < 0.f) io[i] *= slope;
}

// ---- E6: 向量化逐元素原语 (位型等价于标量语义) ----

// dst = src < 0 ? src·slope : src  (leaky 写入融合版)
inline void leaky_relu_write(const float* __restrict src, float* __restrict dst,
                             size_t n, float slope) {
  size_t i = 0;
  const float32x4_t vs = vdupq_n_f32(slope);
  for (; i + 4 <= n; i += 4) {
    const float32x4_t v = vld1q_f32(src + i);
    const uint32x4_t m = vcltzq_f32(v);
    vst1q_f32(dst + i, vbslq_f32(m, vmulq_f32(v, vs), v));
  }
  for (; i < n; ++i) dst[i] = src[i] < 0.f ? src[i] * slope : src[i];
}

// dst[k] += src[k]
inline void add_inplace(float* __restrict dst, const float* __restrict src,
                        size_t n) {
  size_t i = 0;
  for (; i + 4 <= n; i += 4)
    vst1q_f32(dst + i, vaddq_f32(vld1q_f32(dst + i), vld1q_f32(src + i)));
  for (; i < n; ++i) dst[i] += src[i];
}

// acc[k] += src[k]·scale
inline void add_scaled_inplace(float* __restrict acc,
                               const float* __restrict src, size_t n,
                               float scale) {
  size_t i = 0;
  const float32x4_t vs = vdupq_n_f32(scale);
  for (; i + 4 <= n; i += 4)
    vst1q_f32(acc + i,
              vmlaq_f32(vld1q_f32(acc + i), vld1q_f32(src + i), vs));
  for (; i < n; ++i) acc[i] += src[i] * scale;
}

// dst[k] = src[k]·scale
inline void mul_write(float* __restrict dst, const float* __restrict src,
                      size_t n, float scale) {
  size_t i = 0;
  const float32x4_t vs = vdupq_n_f32(scale);
  for (; i + 4 <= n; i += 4)
    vst1q_f32(dst + i, vmulq_f32(vld1q_f32(src + i), vs));
  for (; i < n; ++i) dst[i] = src[i] * scale;
}

}  // namespace gsv::sovits
