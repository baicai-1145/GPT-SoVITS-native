// conv1d.hpp — SoVITS 用卷积原语 (全 fp32 计算)
//   Conv1dF32:      same-padding (含 dilation) 卷积, im2col → accel::sgemm
//   ConvT1dF32:     ConvTranspose1d(k, stride=u, pad=(k-u)/2), 相位分解 GEMM + scatter
// 权重布局与 torch 一致: weight[out][in][k] 连续 = [out, in*k] 行主; bias[out]。
#pragma once

#include "kern/accel.hpp"
#include "sovits/sovits_types.hpp"

#include <cmath>
#include <vector>

namespace gsv::sovits {

class Conv1dF32 {
 public:
  // k=1 的卷积即逐点 GEMM
  size_t out_c = 0, in_c = 0, k = 0, dilation = 1;
  std::vector<float> w;  // [out,in,k]
  std::vector<float> b;  // [out]

  void load(const rt::GsvFile& f, std::string_view prefix, size_t o, size_t i,
            size_t kk, size_t dil = 1) {
    out_c = o;
    in_c = i;
    k = kk;
    dilation = dil;
    std::string p(prefix);
    if (!w.empty()) return;  // 已加载
    load_tensor_f32(f, p + ".weight", w, {o, i, kk});
    const auto* tb = f.tensor(p + ".bias");
    if (tb) {
      load_tensor_f32(f, p + ".bias", b);
    } else {
      b.assign(o, 0.f);  // conv_post 无 bias (is_bias=False)
    }
  }

  // y[C_out, T] = conv(x[C_in, T]) ; same padding, 输出长度 == T
  // 允许 &x == &y (内部临时缓冲保护)
  void forward(const Tensor2D& x_in, Tensor2D& y) const {
    if (&x_in == &y) {
      Tensor2D tmp;
      forward(x_in, tmp);
      y.d.swap(tmp.d);
      y.C = tmp.C;
      y.T = tmp.T;
      return;
    }
    const Tensor2D& x = x_in;
    const size_t T = x.T;
    y.reset(out_c, T);
    if (k == 1 && dilation == 1) {
      // 逐点: y = W[out, in] · x[in, T]; 行主序 ld 均为行长度
      kern::accel::sgemm('N', 'N', static_cast<int>(out_c),
                   static_cast<int>(T), static_cast<int>(in_c), 1.f,
                   w.data(), static_cast<int>(in_c), x.d.data(),
                   static_cast<int>(T), 0.f, y.d.data(),
                   static_cast<int>(T));
    } else {
      // im2col: col[in*k, T]
      thread_local std::vector<float> col;
      col.resize(in_c * k * T);
      const int pad_l = static_cast<int>((k - 1) * dilation) / 2;
      for (size_t i = 0; i < in_c; ++i) {
        const float* xr = x.row(i);
        for (size_t t = 0; t < T; ++t) {
          for (size_t kk = 0; kk < k; ++kk) {
            long long src = static_cast<long long>(t) +
                            static_cast<long long>(kk * dilation) - pad_l;
            col[(i * k + kk) * T + t] =
                (src >= 0 && src < static_cast<long long>(T)) ? xr[src] : 0.f;
          }
        }
      }
      // y[out,T] = W[out, in*k] · col[in*k, T]
      kern::accel::sgemm('N', 'N', static_cast<int>(out_c), static_cast<int>(T),
                   static_cast<int>(in_c * k), 1.f, w.data(),
                   static_cast<int>(in_c * k), col.data(), static_cast<int>(T),
                   0.f, y.d.data(), static_cast<int>(T));
    }
    add_bias(y);
  }

  void add_bias(Tensor2D& y) const {
    for (size_t c = 0; c < out_c; ++c) {
      float* yr = y.row(c);
      const float bc = b[c];
      for (size_t t = 0; t < y.T; ++t) yr[t] += bc;
    }
  }
};

// ConvTranspose1d: out_len = T*u ; weight [in, out, k] (torch 布局!), bias[out]
// 等价散射: y[:, j*u+d-pad] += Σ_i W[i,o,d]·x[:,j], d∈[0,k)
// 实现: 对每个 d 做一次 GEMM(tmp[o,T] = Wd[o,in]·x[in,T]) 后按相位 scatter。
class ConvT1dF32 {
 public:
  size_t out_c = 0, in_c = 0, k = 0, stride_u = 1;
  std::vector<float> w;  // [in,out,k] — 注意 torch ConvTranspose1d 布局
  std::vector<float> b;

  void load(const rt::GsvFile& f, std::string_view prefix, size_t i, size_t o,
            size_t kk, size_t u) {
    out_c = o;
    in_c = i;
    k = kk;
    stride_u = u;
    std::string p(prefix);
    load_tensor_f32(f, p + ".weight", w, {i, o, kk});  // torch ConvT: [in,out,k]
    const auto* tb = f.tensor(p + ".bias");
    if (tb)
      load_tensor_f32(f, p + ".bias", b);
    else
      b.assign(o, 0.f);
  }

  // 允许 &x == &y
  void forward(const Tensor2D& x_in, Tensor2D& y, Tensor2D& scratch) const {
    if (&x_in == &y) {
      Tensor2D tmp;
      forward(x_in, tmp, scratch);
      y.d.swap(tmp.d);
      y.C = tmp.C;
      y.T = tmp.T;
      return;
    }
    const Tensor2D& x = x_in;
    const size_t T = x.T;
    const long long pad = static_cast<long long>((k - stride_u) / 2);
    const size_t Tout = T * stride_u;
    y.reset(out_c, Tout);
    scratch.reset(out_c, T);

    // w_d[o, in] : 提取 kernel 切片 d
    thread_local std::vector<float> wd;
    wd.resize(out_c * in_c);
    for (size_t d = 0; d < k; ++d) {
      for (size_t o = 0; o < out_c; ++o)
        for (size_t i = 0; i < in_c; ++i)
          wd[o * in_c + i] = w[(i * out_c + o) * k + d];
      // tmp[o,T] = wd[o,in] · x[in,T]
      kern::accel::sgemm('N', 'N', static_cast<int>(out_c), static_cast<int>(T),
                   static_cast<int>(in_c), 1.f, wd.data(), static_cast<int>(in_c),
                   x.d.data(), static_cast<int>(T), 0.f, scratch.d.data(),
                   static_cast<int>(T));
      // scatter 到 strided 输出位置
      for (size_t j = 0; j < T; ++j) {
        long long opos = static_cast<long long>(j) * stride_u +
                         static_cast<long long>(d) - pad;
        if (opos < 0 || opos >= static_cast<long long>(Tout)) continue;
        size_t op = static_cast<size_t>(opos);
        for (size_t o = 0; o < out_c; ++o) y.row(o)[op] += scratch.row(o)[j];
      }
    }
    for (size_t c = 0; c < out_c; ++c) {
      float* yr = y.row(c);
      const float bc = b[c];
      for (size_t t = 0; t < Tout; ++t) yr[t] += bc;
    }
  }
};

}  // namespace gsv::sovits
