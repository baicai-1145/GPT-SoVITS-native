// flow.hpp — ResidualCouplingBlock (仅推理 reverse 方向: z_p → z)
// 结构: 4 × [ResidualCouplingLayer(mean_only) + Flip], reverse 时逆序遍历。
// WN 条件流: cond_layer(ge[1024]) → 1536 = 2*192*4, 逐层切片。
#pragma once

#include "conv1d.hpp"
#include "kern/kern.hpp"

namespace gsv::sovits {

class FlowWN {
 public:
  size_t hidden = 0, n_layers = 0;
  // cond_layer: Conv1d(gin → 2*hidden*n_layers, k=1), weight_norm 已融合 fp32
  Conv1dF32 cond;
  std::vector<Conv1dF32> in_layers;      // hidden→2h, k=5 dil=1
  std::vector<Conv1dF32> res_skip;       // h→2h (末层 h→h), k=1

  void load(const rt::GsvFile& f, std::string_view prefix, size_t h, size_t nl,
            size_t gin, size_t kernel = 5) {
    hidden = h;
    n_layers = nl;
    std::string p(prefix);
    cond.load(f, p + ".cond_layer", 2 * h * nl, gin, 1);  // [1536,1024,1]
    in_layers.resize(nl);
    res_skip.resize(nl);
    for (size_t i = 0; i < nl; ++i) {
      in_layers[i].load(f, p + ".in_layers." + std::to_string(i), 2 * h, h,
                        kernel, 1);
      const bool last = (i == nl - 1);
      res_skip[i].load(f, p + ".res_skip_layers." + std::to_string(i),
                       last ? h : 2 * h, h, 1);
    }
  }

  // x[h,T], mask 全 1 场景仍按乘法实现; g[gin,1] 单帧条件
  // 统一核心: dbg 非空时逐层 dump (tag 前缀)
  void forward(Tensor2D& x, const Tensor2D& mask, const Tensor2D& g,
               Tensor2D& scratch, Tensor2D& cond_buf, const Dumper* dbg = nullptr,
               const char* tag = "") const {
    const size_t T = x.T;
    cond.forward(g, cond_buf);
    const size_t gT = cond_buf.T;
    auto gcol = [&](size_t ch, size_t t) -> const float& {
      return cond_buf.d[ch * gT + (gT == 1 ? 0 : t)];
    };
    std::string pfx(tag ? tag : "");
    if (dbg && !pfx.empty()) dbg->dump(pfx + "_cond", cond_buf);
    Tensor2D output(x.C, T);
    for (size_t i = 0; i < n_layers; ++i) {
      in_layers[i].forward(x, scratch);
      if (dbg && !pfx.empty())
        dbg->dump(pfx + "_" + std::to_string(i) + "_xin", scratch);
      for (size_t c = 0; c < hidden; ++c) {
        float* ar = scratch.row(c);
        float* as = ar + hidden * T;
        for (size_t t = 0; t < T; ++t)
          ar[t] = std::tanh(ar[t] + gcol(i * 2 * hidden + c, t)) *
                  (1.f / (1.f + std::exp(-(as[t] +
                      gcol(i * 2 * hidden + hidden + c, t)))));
      }
      if (dbg && !pfx.empty()) {
        Tensor2D acts_tmp(scratch);
        dbg->dump(pfx + "_" + std::to_string(i) + "_acts", acts_tmp);
      }
      res_skip[i].forward(scratch, scratch);
      if (dbg && !pfx.empty()) {
        Tensor2D rs_tmp(scratch);
        dbg->dump(pfx + "_" + std::to_string(i) + "_rs", rs_tmp);
      }
      if (i < n_layers - 1) {
        for (size_t t = 0; t < T; ++t)
          for (size_t c = 0; c < hidden; ++c)
            x.d[c * T + t] =
                (x.d[c * T + t] + scratch.row(c)[t]) * mask.d[t];
        for (size_t t = 0; t < T; ++t)
          for (size_t c = 0; c < hidden; ++c)
            output.d[c * T + t] += scratch.row(hidden + c)[t];
      } else {
        for (size_t t = 0; t < T; ++t)
          for (size_t c = 0; c < hidden; ++c)
            output.d[c * T + t] += scratch.row(c)[t];
      }
    }
    for (size_t i = 0; i < output.d.size(); ++i)
      output.d[i] *= mask.d[i % output.T];
    x = std::move(output);
  }

};

class ResidualCouplingLayer {
 public:
  size_t half = 0, hidden = 0;
  Conv1dF32 pre;   // half → hidden, k=1
  FlowWN enc;
  Conv1dF32 post;  // hidden → half (mean_only)

  void load(const rt::GsvFile& f, std::string_view prefix, size_t channels,
            size_t h, size_t gin) {
    half = channels / 2;
    hidden = h;
    std::string p(prefix);
    // torch: pre = Conv1d(half→hidden) 权重[hidden,half,1]; post 反之
    pre.load(f, p + ".pre", h, half, 1);
    enc.load(f, p + ".enc", h, 4, gin);
    post.load(f, p + ".post", half, h, 1);
  }

  // reverse=True: x1 = (x1 - m) · exp(-logs=0) = x1 - m
  void forward_reverse(Tensor2D& x, const Tensor2D& mask, const Tensor2D& g,
                       Tensor2D& scratch, Tensor2D& cond_buf,
                       Tensor2D& hbuf) const {
    const size_t T = x.T;
    // x0 = 前半通道
    Tensor2D x0(half, T);
    for (size_t c = 0; c < half; ++c)
      for (size_t t = 0; t < T; ++t) x0.d[c * T + t] = x.d[c * T + t];
    pre.forward(x0, hbuf);
    enc.forward(hbuf, mask, g, scratch, cond_buf);
    post.forward(hbuf, hbuf);
    // mean_only: logs=0 → exp(-logs)=1
    for (size_t c = 0; c < half; ++c) {
      const float* mr = hbuf.row(c);
      float* xr = x.d.data() + (half + c) * T;
      for (size_t t = 0; t < T; ++t) xr[t] -= mr[t];
    }
  }
};

class Flow {
 public:
  static constexpr size_t kFlows = 4;
  size_t channels = 0, hidden = 0, gin = 0;
  std::array<ResidualCouplingLayer, kFlows> rcl;

  void load(const rt::GsvFile& f) {
    channels = 192;
    hidden = 192;
    gin = 1024;
    for (size_t i = 0; i < kFlows; ++i)
      rcl[i].load(f, "flow.flows." + std::to_string(i * 2), channels, hidden,
                  gin);
  }

  // reverse 推理: 逆序 flows → [Flip3,RCL3,...,Flip0,RCL0]
  void forward_reverse(Tensor2D& x, const Tensor2D& mask, const Tensor2D& ge,
                       const Dumper& dm) const {
    Tensor2D scratch, cond_buf, hbuf;
    for (int i = static_cast<int>(kFlows) - 1; i >= 0; --i) {
      flip(x);
      rcl[i].forward_reverse(x, mask, ge, scratch, cond_buf, hbuf);
      dm.dump("h_flow_rcl" + std::to_string(i), x);
    }
  }

 public:
  static void flip_for_debug(Tensor2D& x) { flip(x); }

 private:
  // torch Flip: x = torch.flip(x, [1]) — 对 [B,C,T] 翻转通道维!
  static void flip(Tensor2D& x) {
    const size_t C = x.C, T = x.T;
    for (size_t c = 0; c < C / 2; ++c) {
      float* a = x.d.data() + c * T;
      float* b = x.d.data() + (C - 1 - c) * T;
      for (size_t t = 0; t < T; ++t) std::swap(a[t], b[t]);
    }
  }
};

}  // namespace gsv::sovits
