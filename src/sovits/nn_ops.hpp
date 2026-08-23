// nn_ops.hpp — SoVITS 引擎逐元素/归一化原语 (全 fp32, 语义对齐 torch)
#pragma once

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
  for (size_t i = 0; i < n; ++i)
    if (io[i] < 0.f) io[i] *= slope;
}

}  // namespace gsv::sovits
