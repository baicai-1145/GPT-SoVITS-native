// dec.hpp — Generator (HiFi-GAN V1, models.py Generator + ResBlock1)
//   x = conv_pre(z) + cond(ge)  [192→768 k7; cond 1024→768 k1 广播]
//   ×5: leaky_relu(0.1) → ConvT(k,u) → mean(3×ResBlock1)
//   leaky_relu(0.01 默认!) → conv_post(24→1 k7, 无 bias) → tanh
// 配置: upsample_rates [10,8,2,2,2], kernel [20,16,8,2,2], resblock k [3,7,11]
//       dilations [[1,3,5]×3], initial_channel 768。
#pragma once

#include <cmath>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "sovits/conv1d.hpp"
#include "sovits/nn_ops.hpp"
#if defined(GSV_AMX_GEMM)
#include "kern/gemm_f16_amx.hpp"
#endif

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

 private:
#if defined(GSV_AMX_GEMM)
  // E10-K2: resblock stage 按-tile DAG 调度 — 3链×6卷积×nmb×nnb 个 tile。
  // amx_tile_chain_run 内部将同 (c,d,mb) 内连续 nb 合并为 chunk (调度节点),
  // 以 ~32*healthy tile/chunk 为粒度。依赖模型 (chunk 级):
  //   跨深度 (c,d,chunk) → (c,d-1,chunk)
  //   同行列 (c,d,chunk) → (c,d,chunk-1)
  //   左边界 (c,d,chunk) → (c,d-1,chunk-1)  (im2col 窗口跨 chunk 边界)
  // prepare 钩子仅处理 tile 行区间 [row_b, row_e) — 池内 im2col 切片
  // 与同 chain 后续 chunk 数学重叠, 消除 ~25ms 全局 prepare 串行。
  // 数值序与 E10-K amx_chain_run 一致: 链内算子序列不变, sum 累加按 j 序。
  struct ResBatchScratch {
    Tensor2D cur[3], tin[3], tA[3][2], tB[3][2];
    kern::AmxPanel pb[3][6];
  };

  // batch_ms 输出本次批量派发耗时 (含 prepare+gemm)。
  // 返回 false = 不满足批量条件 (调用方回退串行循环)。
  bool res_forward_batched(size_t stage, const Tensor2D& x,
                           Tensor2D& sum, double* batch_ms) const {
    if (!kern::amx_gemm_available()) return false;
    const size_t C = x.C, Tn = x.T;
    const size_t nn = C * Tn;
    if (Tn < 64 || nn == 0) return false;
    for (size_t j = 0; j < kResKernels; ++j)
      for (size_t m = 0; m < 3; ++m)
        if (!resblocks[stage][j].convs1[m].amx_ready() ||
            !resblocks[stage][j].convs2[m].amx_ready())
          return false;

    // 单飞安全: SoVITS stage 为单线程 FIFO, amx_tile_chain_run 阻塞至完成;
    // 静态存活使 pb/张量容量跨调用复用 (免 prepare 内分配)。
    static ResBatchScratch s;
    auto ensure = [](Tensor2D& t, size_t c, size_t tt) {
      t.C = c; t.T = tt; t.d.resize(c * tt);
    };
    for (size_t j = 0; j < kResKernels; ++j) {
      ensure(s.cur[j], C, Tn);
      ensure(s.tin[j], C, Tn);
      for (int h = 0; h < 2; ++h) {
        ensure(s.tA[j][h], C, Tn);
        ensure(s.tB[j][h], C, Tn);
      }
    }

    const float* xn = x.d.data();
    const float kInv = 1.f / static_cast<float>(kResKernels);
    // tile 网格
    const size_t nmb = (C + 31) / 32, nnb = (Tn + 31) / 32;
    std::vector<kern::AmxTileChainLink> nodes;
    nodes.reserve(kResKernels * 6 * nmb * nnb);
    for (size_t j = 0; j < kResKernels; ++j) {
      const ResBlock1& rb = resblocks[stage][j];
      for (size_t p = 0; p < 6; ++p) {
        const size_t m = p / 2;
        const bool even = (p % 2) == 0;
        const Conv1d& cv = even ? rb.convs1[m] : rb.convs2[m];
        kern::AmxPanel& pb = s.pb[j][p];
        pb.rows = Tn;
        pb.K = cv.in_c * cv.k + 1;
        float* outp = even ? s.tA[j][m & 1].d.data()
                           : s.tB[j][m & 1].d.data();
        for (size_t mb = 0; mb < nmb; ++mb) {
          for (size_t nb = 0; nb < nnb; ++nb) {
            kern::AmxTileChainLink nd;
            nd.chain_id = static_cast<int>(j);
            nd.depth = static_cast<int>(p);
            nd.tile_mb = mb;
            nd.tile_nb = nb;
            nd.pa = &cv.amx_panel();
            nd.pb = &pb;
            nd.c = outp;
            nd.M = cv.out_c;
            nd.N = Tn;
            // prepare 钩子: 处理 tile 行区间 [row_b, row_e)
            nd.prepare = [j, p, m, even, Tn, C, xn, cvl = &cv](size_t row_b,
                                                           size_t row_e) {
              const size_t tile_len = row_e - row_b;
              kern::AmxPanel& pbl = s.pb[j][p];
              if (even) {
                if (m == 0) {
                  for (size_t c = 0; c < C; ++c)
                    std::memcpy(s.cur[j].d.data() + c * Tn + row_b,
                                xn + c * Tn + row_b, tile_len * sizeof(float));
                } else {
                  for (size_t c = 0; c < C; ++c)
                    add_inplace(s.cur[j].d.data() + c * Tn + row_b,
                                s.tB[j][(m - 1) & 1].d.data() + c * Tn + row_b,
                                tile_len);
                }
                for (size_t c = 0; c < C; ++c)
                  leaky_relu_write(s.cur[j].d.data() + c * Tn + row_b,
                                   s.tin[j].d.data() + c * Tn + row_b,
                                   tile_len, 0.1f);
                im2col_to_panel_f16_tile(s.tin[j].d.data(), C, Tn, cvl->k,
                                          cvl->dilation, row_b, row_e, pbl.buf,
                                          true);
              } else {
                for (size_t c = 0; c < C; ++c)
                  leaky_relu_io(s.tA[j][m & 1].d.data() + c * Tn + row_b,
                                tile_len, 0.1f);
                im2col_to_panel_f16_tile(s.tA[j][m & 1].d.data(), C, Tn,
                                          cvl->k, cvl->dilation, row_b, row_e,
                                          pbl.buf, true);
              }
            };
            nodes.push_back(std::move(nd));
          }
        }
      }
    }

    const double tb = now_ms_e6();
    kern::amx_tile_chain_run(nodes.data(), nodes.size());
    *batch_ms = now_ms_e6() - tb;

    // 三链 sum 累加 (主线程): cur += convs2[2] 输出; sum += cur/3
    for (size_t j = 0; j < kResKernels; ++j) {
      add_inplace(s.cur[j].d.data(), s.tB[j][0].d.data(), nn);
      if (j == 0) {
        mul_write(sum.d.data(), s.cur[j].d.data(), nn, kInv);
      } else {
        add_scaled_inplace(sum.d.data(), s.cur[j].d.data(), nn, kInv);
      }
    }
    return true;
  }
#else
  bool res_forward_batched(size_t, const Tensor2D&, Tensor2D&, double*) const {
    return false;
  }
#endif

 public:
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
      // E6: 全覆写缓冲免零填 (resize-only), 拷贝走 memcpy
      sum.C = x.C;
      sum.T = x.T;
      sum.d.resize(x.d.size());
      // E10-K: resblock stage 按-chain DAG 调度 — 3链×6卷积=18 节点,
      // chain_id=j/depth=p, 同 chain 串行异 chain 并行 (无 phase barrier)。
      // im2col/lrelu/residual-add 放进 prepare 钩子, 与同 chain 数学重叠。
      double batch_ms = 0.0;
      t0 = tic();
      if (res_forward_batched(i, x, sum, &batch_ms)) {
        if (prof) T.dec_res += batch_ms;
      } else {
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
        if (prof) T.dec_res += now_ms_e6() - t0;
      }
      x.d.swap(sum.d);
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
