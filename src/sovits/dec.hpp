// dec.hpp — Generator (HiFi-GAN V1, models.py Generator + ResBlock1)
//   x = conv_pre(z) + cond(ge)  [192→768 k7; cond 1024→768 k1 广播]
//   ×5: leaky_relu(0.1) → ConvT(k,u) → mean(3×ResBlock1)
//   leaky_relu(0.01 默认!) → conv_post(24→1 k7, 无 bias) → tanh
// 配置: upsample_rates [10,8,2,2,2], kernel [20,16,8,2,2], resblock k [3,7,11]
//       dilations [[1,3,5]×3], initial_channel 768。
#pragma once

#include <cmath>
#include <string>
#include <vector>

#include "sovits/conv1d.hpp"
#include "sovits/nn_ops.hpp"

namespace gsv::sovits {

class ResBlock1 {
 public:
  size_t channels = 0;
  Conv1d convs1[3];  // dilations {1,3,5}, k=3
  Conv1d convs2[3];  // dilation 1

  void load(const rt::GsvFile& f, std::string_view prefix, size_t ch,
            size_t kernel_size, const size_t (&dil)[3]) {
    channels = ch;
    std::string p(prefix);
    for (size_t j = 0; j < 3; ++j) {
      convs1[j].load(f, p + ".convs1." + std::to_string(j), ch, ch,
                     kernel_size, dil[j]);
      convs2[j].load(f, p + ".convs2." + std::to_string(j), ch, ch,
                     kernel_size, 1);
    }
  }

  // 块内计算由 Generator 驱动 (需跨块复用缓冲)
};

class Generator {
 public:
  Conv1d conv_pre;   // 192→768 k7
  Conv1d cond;       // 1024→768 k1 (ge 广播)
  static constexpr size_t kStages = 5;
  ConvT1d ups[kStages];
  ResBlock1 resblocks[kStages][3];
  static constexpr size_t kResKernels = 3;
  Conv1d conv_post;  // 24→1 k7 无 bias

  void load(const rt::GsvFile& f) {
    static const size_t kUpsU[kStages] = {10, 8, 2, 2, 2};
    static const size_t kUpsK[kStages] = {20, 16, 8, 2, 2};
    static const size_t kResDil[3] = {1, 3, 5};

    conv_pre.load(f, "dec.conv_pre", 768, 192, 7);
    cond.load(f, "dec.cond", 768, 1024, 1);
    size_t cin = 768;
    for (size_t i = 0; i < kStages; ++i) {
      const size_t cout_ = cin / 2;
      std::string up = "dec.ups." + std::to_string(i);
      ups[i].load(f, up, cin, cout_, kUpsK[i], kUpsU[i]);
      const std::string rb = "dec.resblocks." + std::to_string(i * 3);
      static const size_t kResKernelSizes[3] = {3, 7, 11};
      for (size_t j = 0; j < 3; ++j)
        resblocks[i][j].load(f, "dec.resblocks." +
                                    std::to_string(i * kResKernels + j),
                             cout_, kResKernelSizes[j], kResDil);
      cin = cout_;
    }
    conv_post.load(f, "dec.conv_post", 1, 24, 7);
  }

  // z[192,T] → wav [1, T*640], 值域 [-1,1]
  void forward(const Tensor2D& z, const Tensor2D& ge, Tensor2D& out,
               const Dumper& dm) const {
    const size_t gT = ge.T;
    StageTimers& T = sov_timers();
    const bool prof = sov_timing_enabled();
    auto tic = [&] { return prof ? now_ms_e6() : 0.0; };
    double t0;
    Tensor2D x;
    t0 = tic();
    conv_pre.forward(z, x);
    if (prof) T.dec_pre += now_ms_e6() - t0;
    dm.dump("dbg_dec_convpre", x);
    // x += cond(ge) (广播 gT==1)
    t0 = tic();
    Tensor2D cb;
    cond.forward(ge, cb);  // [768, gT]
    if (prof) T.dec_cond += now_ms_e6() - t0;
    dm.dump("dbg_dec_cond_out", cb);
    for (size_t c = 0; c < 768; ++c) {
      const float cv = cb.d[c * gT];
      for (size_t t = 0; t < x.T; ++t) x.d[c * x.T + t] += cv;
    }

    // E6: scratch 持久化 (thread_local) — 每 stage 重构造会反复 mmap/缺页零填
    static thread_local Tensor2D xu, scratch, sum;
    for (size_t i = 0; i < kStages; ++i) {
      leaky_relu_io(x.d.data(), x.d.size(), 0.1f);
      t0 = tic();
      ups[i].forward(x, xu, scratch);
      if (prof) T.dec_ups += now_ms_e6() - t0;
      x.d.swap(xu.d);
      x.C = xu.C;
      x.T = xu.T;
      {
        std::string n = "dbg_dec_up" + std::to_string(i);
        dm.dump(n.c_str(), x);
      }
      // torch: xs = Σ_j block_j(x) (块内三对串联卷积, 每对后残差相加,
      // 故 block_j 已含输入 x); x = xs / 3。
      constexpr float kInv = 1.f / static_cast<float>(kResKernels);
      t0 = tic();
      // E6: 全覆写缓冲免零填 (resize-only), 拷贝走 memcpy
      sum.C = x.C;
      sum.T = x.T;
      sum.d.resize(x.d.size());
      static thread_local Tensor2D cur, t_in, t_a, t_b;
      bool first = true;
      for (size_t j = 0; j < kResKernels; ++j) {
        const ResBlock1& rb = resblocks[i][j];
        cur.C = x.C;
        cur.T = x.T;
        cur.d.resize(x.d.size());
        std::memcpy(cur.d.data(), x.d.data(), x.d.size() * sizeof(float));
        for (size_t jj = 0; jj < 3; ++jj) {
          // torch: xt=lrelu(x); xt=c1(xt); xt=lrelu(xt); xt=c2(xt); x=xt+x
          // (lrelu 不污染残差分支 — 用独立缓冲 t_in)
          double te = tic();
          t_in.C = cur.C;
          t_in.T = cur.T;
          t_in.d.resize(cur.d.size());
          leaky_relu_write(cur.d.data(), t_in.d.data(), cur.d.size(), 0.1f);
          if (prof) T.res_ew += now_ms_e6() - te;
          t0 = tic();
          rb.convs1[jj].forward(t_in, t_a);
          if (prof) T.res_conv1 += now_ms_e6() - t0;
          te = tic();
          leaky_relu_io(t_a.d.data(), t_a.d.size(), 0.1f);
          if (prof) T.res_ew += now_ms_e6() - te;
          t0 = tic();
          rb.convs2[jj].forward(t_a, t_b);
          if (prof) T.res_conv2 += now_ms_e6() - t0;
          te = tic();
          add_inplace(cur.d.data(), t_b.d.data(), cur.d.size());
          if (prof) T.res_ew += now_ms_e6() - te;
        }
        if (first) {
          mul_write(sum.d.data(), cur.d.data(), sum.d.size(), kInv);
          first = false;
        } else {
          add_scaled_inplace(sum.d.data(), cur.d.data(), sum.d.size(), kInv);
        }
      }
      x.d.swap(sum.d);
      if (prof) T.dec_res += now_ms_e6() - t0;
      {
        std::string n = "dbg_dec_res" + std::to_string(i);
        dm.dump(n.c_str(), x);
      }
    }
    leaky_relu_io(x.d.data(), x.d.size(), 0.01f);  // F.leaky_relu 默认 slope
    t0 = tic();
    conv_post.forward(x, out);
    if (prof) T.dec_post += now_ms_e6() - t0;
    for (float& v : out.d) v = std::tanh(v);
  }
};

}  // namespace gsv::sovits
