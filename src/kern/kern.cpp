// kern.cpp — 归一化/softmax/激活/RoPE 原语（统计量只用 fp32）
#include "kern.hpp"

#include <arm_neon.h>

#include <cmath>

namespace gsv::kern {

// ---------- RMSNorm ----------
void rmsnorm(const float* x, const float* g, float* y, size_t n, double eps) {
  const size_t vec_end = n & ~size_t{7};
  float32x4_t s0 = vdupq_n_f32(0.0f), s1 = vdupq_n_f32(0.0f);
  size_t i = 0;
  for (; i < vec_end; i += 8) {
    const float32x4_t a = vld1q_f32(x + i), b = vld1q_f32(x + i + 4);
    s0 = vfmaq_f32(s0, a, a);
    s1 = vfmaq_f32(s1, b, b);
  }
  float ss = vaddvq_f32(s0) + vaddvq_f32(s1);
  for (; i < n; ++i) ss += x[i] * x[i];
  const float inv_rms = static_cast<float>(1.0 / std::sqrt(ss / static_cast<double>(n) + eps));
  for (size_t j = 0; j < n; ++j) y[j] = x[j] * inv_rms * g[j];
}

// ---------- LayerNorm(有偏方差, 与 F.layer_norm 同构) ----------
void layernorm(const float* x, const float* g, const float* b, float* y, size_t n, double eps) {
  // E[x] 与 E[x²] 单遍 fp32 累加
  const size_t vec_end = n & ~size_t{7};
  float32x4_t s0 = vdupq_n_f32(0.0f), s1 = vdupq_n_f32(0.0f);
  float32x4_t q0 = vdupq_n_f32(0.0f), q1 = vdupq_n_f32(0.0f);
  size_t i = 0;
  for (; i < vec_end; i += 8) {
    const float32x4_t a = vld1q_f32(x + i), c = vld1q_f32(x + i + 4);
    s0 = vaddq_f32(s0, a);
    s1 = vaddq_f32(s1, c);
    q0 = vfmaq_f32(q0, a, a);
    q1 = vfmaq_f32(q1, c, c);
  }
  double sum = static_cast<double>(vaddvq_f32(s0) + vaddvq_f32(s1));
  double sqsum = static_cast<double>(vaddvq_f32(q0) + vaddvq_f32(q1));
  for (; i < n; ++i) {
    sum += x[i];
    sqsum += static_cast<double>(x[i]) * x[i];
  }
  const double dn = static_cast<double>(n);
  const float mean = static_cast<float>(sum / dn);
  float var = static_cast<float>(sqsum / dn - static_cast<double>(mean) * mean);
  if (!(var >= 0.0f)) {  // E[x²]-E[x]² 病态抵消兜底: 二次扫描重算
    double s2 = 0;
    for (size_t j = 0; j < n; ++j) {
      const double d = static_cast<double>(x[j]) - mean;
      s2 += d * d;
    }
    var = static_cast<float>(s2 / dn);
  }
  const float rstd = static_cast<float>(1.0 / std::sqrt(static_cast<double>(var) + eps));
  for (size_t j = 0; j < n; ++j) y[j] = (x[j] - mean) * rstd * g[j] + b[j];
}

// ---------- Softmax ----------
void softmax(const float* x, float* y, size_t n) {
  float m = x[0];
  for (size_t i = 1; i < n; ++i) m = m < x[i] ? x[i] : m;
  float sum = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    const float e = std::exp(x[i] - m);
    y[i] = e;
    sum += e;
  }
  const float inv = 1.0f / sum;
  for (size_t i = 0; i < n; ++i) y[i] *= inv;
}

void softmax_rows(const float* x, float* y, size_t rows, size_t n) {
  for (size_t r = 0; r < rows; ++r) softmax(x + r * n, y + r * n, n);
}

// ---------- 激活 ----------
void silu(const float* x, float* y, size_t n) {
  for (size_t i = 0; i < n; ++i)
    y[i] = x[i] / (1.0f + std::exp(-x[i]));  // x·sigmoid(x), fp32
}

void relu(const float* x, float* y, size_t n) {
  const float32x4_t zero = vdupq_n_f32(0.0f);
  const size_t vec_end = n & ~size_t{3};
  size_t i = 0;
  for (; i < vec_end; i += 4) vst1q_f32(y + i, vmaxq_f32(vld1q_f32(x + i), zero));
  for (; i < n; ++i) y[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

// ---------- RoPE ----------
namespace {

struct Angle {
  double cos_v;
  double sin_v;
};

inline Angle angle_at(size_t pair_idx, size_t head_dim, size_t pos, double base) {
  const double freq = std::pow(base, -2.0 * static_cast<double>(pair_idx) /
                                           static_cast<double>(head_dim));
  const double ang = static_cast<double>(pos) * freq;
  return {std::cos(ang), std::sin(ang)};
}

void rope_row(float* row, size_t heads, size_t head_dim, size_t pos, RopeStyle style,
              double base) {
  for (size_t h = 0; h < heads; ++h) {
    float* d = row + h * head_dim;
    if (style == RopeStyle::GptJ) {
      for (size_t p = 0; p < head_dim; p += 2) {
        const Angle a = angle_at(p / 2, head_dim, pos, base);
        const float c = static_cast<float>(a.cos_v), s = static_cast<float>(a.sin_v);
        const float x0 = d[p], x1 = d[p + 1];
        d[p] = x0 * c - x1 * s;
        d[p + 1] = x0 * s + x1 * c;
      }
    } else {  // NeoX
      const size_t half = head_dim / 2;
      for (size_t p = 0; p < half; ++p) {
        const Angle a = angle_at(p, head_dim, pos, base);
        const float c = static_cast<float>(a.cos_v), s = static_cast<float>(a.sin_v);
        const float x0 = d[p], x1 = d[p + half];
        d[p] = x0 * c - x1 * s;
        d[p + half] = x0 * s + x1 * c;
      }
    }
  }
}

}  // namespace

void rope_prefill(float* io, size_t seq, size_t heads, size_t head_dim,
                  size_t pos_base, RopeStyle style, double base) {
  const size_t dim = heads * head_dim;
  for (size_t t = 0; t < seq; ++t)
    rope_row(io + t * dim, heads, head_dim, pos_base + t, style, base);
}

void rope_decode(float* io, size_t heads, size_t head_dim,
                 size_t pos, RopeStyle style, double base) {
  rope_row(io, heads, head_dim, pos, style, base);
}

}  // namespace gsv::kern
