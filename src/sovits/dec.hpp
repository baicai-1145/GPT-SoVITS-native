// dec.hpp — Generator (HiFi-GAN V1, models.py Generator + ResBlock1)
//   x = conv_pre(z) + cond(ge)  [192→768 k7; cond 1024→768 k1 广播]
//   ×5: leaky_relu(0.1) → ConvT(k,u) → mean(3×ResBlock1)
//   leaky_relu(0.01 默认!) → conv_post(24→1 k7, 无 bias) → tanh
// 配置: upsample_rates [10,8,2,2,2], kernel [20,16,8,2,2], resblock k [3,7,11]
//       dilations [[1,3,5]×3], initial_channel 768。
#pragma once

#include <cmath>
#include <cstdio>
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
  // E10-K: resblock stage 按-chain DAG 调度 —— 3链×6卷积 = 18 节点。
  // 依赖仅限同 chain 内 depth-1→depth (跨链无 barrier)。im2col/lrelu/残差加
  // 放进 prepare 钩子在池 worker 内执行, 与同 chain 其他节点的数学重叠。
  // 三链 sum 累加留主线程 (单飞, 阻塞式 amx_chain_run)。
  // 数值序与串行路径完全一致: 链内算子序列不变, sum 累加按 j 序。
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

    // E10-MEM: 预算守卫。ResBatchScratch 静态复用 s.cur/tin/tA/tB (18×C*Tn
    // floats = 72·C·Tn 字节) + 18 个 panel (nt·K·64+64)。 10s 段 S2 (C=192,
    // T=102400) 总占用 ~3GB+ → 16GB 机器上多段叠加 phys_footprint 失控。 超过
    // 预算回退串行 (后者 thread_local + 临时分配, 峰值小)。 默认 3GB
    // (ResBatchScratch 主贡献; Conv1d::forward / textfront / encoder 另计
    // 在外), 环境变量 GSV_SOVITS_BUDGET_MB 覆盖。
    static const size_t kBudgetBytes = []() {
      long mb = 3L * 1024L * 1024L * 1024L;  // 3 GiB 默认
      if (const char* e = std::getenv("GSV_SOVITS_BUDGET_MB")) {
        char* end = nullptr;
        long v = std::strtol(e, &end, 10);
        if (end && *end == '\0' && v > 0) mb = v * 1024L * 1024L;
      }
      return static_cast<size_t>(mb);
    }();
    // 预算精账: 18 panel (取本 stage 各 (j, k) 形状计算) + 18 张量。
    // 注: panel size 取阶段所有 j 中最大 K, 偏保守但能准确盖住 (j*p 形状
    // 不同, 但所有 panel 同时间活在 scratch)。实际 K_j = stage_ch·k_j + 1。
    const size_t nt = (Tn + 31) / 32;
    size_t panel_total = 0;
    for (size_t j = 0; j < kResKernels; ++j) {
      const ResBlock1& rb = resblocks[stage][j];
      // convs1 与 convs2 共享 in_c/out_c, 取最大 k 即可覆盖两种 panel
      const size_t k_max = std::max(rb.convs1[0].k,
                              std::max(rb.convs1[1].k,
                                       std::max(rb.convs1[2].k,
                                                std::max(rb.convs2[0].k,
                                                  std::max(rb.convs2[1].k,
                                                           rb.convs2[2].k)))));
      const size_t K = rb.convs1[0].in_c * k_max + 1;
      // 每 j 有 6 panel (2 convs × 3 m), shape 相同
      panel_total += 6 * (nt * K * 64 + 64);
    }
    const size_t tensor_total = 18 * C * Tn * sizeof(float);
    // 单飞安全: SoVITS stage 为单线程 FIFO, amx_batch_run 阻塞至完成;
    // 静态存活使 pb/张量容量跨调用复用 (免 prepare 内分配)。
    // E10-MEM: 避免 shrink 触发 realloc churn; 容量足仅 size 调, 容量
    // 不够才 resize (capacity 翻倍策略, 后续调用走 size-only 免费)。
    // 静态 s 提前到本块顶部声明, 使预算守卫能访问并主动释放超容的 s
    // (避免大 stage 调大后 OS 仍记 1 页高水位, 后续小 stage 调不到
    // 较紧 footprint)。
    static ResBatchScratch s;
    auto ensure = [](Tensor2D& t, size_t c, size_t tt) {
      const size_t need = c * tt;
      if (t.d.capacity() < need) t.d.resize(need);
      else if (t.d.size() < need) t.d.resize(need);
      t.C = c; t.T = tt;
    };
    if (panel_total + tensor_total > kBudgetBytes) {
      if (sov_timing_enabled()) {
        std::fprintf(stderr,
            "[sovits-dec-budget] stage=%zu T=%zu C=%zu panel=%zumiB "
            "tensor=%zumiB total=%zumiB > budget=%zumiB → fallback serial\n",
            stage, Tn, C, panel_total >> 20, tensor_total >> 20,
            (panel_total + tensor_total) >> 20, kBudgetBytes >> 20);
      }
      // 主动释放 s: 避免上一调大后 OS 仍记 1 页高水位。仅释放一次, 后续
      // 小 stage 调不到预算也不会重新触发。
      static thread_local bool released = false;
      if (!released) {
        for (size_t j = 0; j < kResKernels; ++j) {
          for (size_t p = 0; p < 6; ++p) {
            s.pb[j][p].buf.clear();
            s.pb[j][p].buf.shrink_to_fit();
          }
          // 18 张量也缩回 0 容量, 让 OS 回收 (cur/tin/tA/tB 共 18 个)。
          for (auto* t : {&s.cur[j], &s.tin[j], &s.tA[j][0], &s.tA[j][1],
                          &s.tB[j][0], &s.tB[j][1]}) {
            std::vector<float>().swap(t->d);
            t->C = t->T = 0;
          }
        }
        released = true;
      }
      return false;
    }
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
    std::vector<kern::AmxChainLink> nodes;
    nodes.reserve(kResKernels * 6);
    for (size_t j = 0; j < kResKernels; ++j) {
      const ResBlock1& rb = resblocks[stage][j];
      for (size_t p = 0; p < 6; ++p) {
        const size_t m = p / 2;
        const bool even = (p % 2) == 0;
        const Conv1d& cv = even ? rb.convs1[m] : rb.convs2[m];
        // MUST set rows/K before node creation: amx_chain_run validates
        // pb->rows==N && pa->K==pb->K before prepare hook runs.
        kern::AmxPanel& pb = s.pb[j][p];
        pb.rows = Tn;
        pb.K = cv.in_c * cv.k + 1;
        float* outp = even ? s.tA[j][m & 1].d.data()
                           : s.tB[j][m & 1].d.data();
        kern::AmxChainLink nd;
        nd.chain_id = static_cast<int>(j);
        nd.depth = static_cast<int>(p);
        nd.pa = &cv.amx_panel();
        nd.pb = &pb;
        nd.c = outp;
        nd.M = cv.out_c;
        nd.N = Tn;
        nd.prepare = [j, p, m, even, Tn, nn, xn, cvl = &cv]() {
          kern::AmxPanel& pbl = s.pb[j][p];
          if (even) {
            // cur 初始化/累加, 然后 lrelu(cur) → tin, im2col → panel
            if (m == 0) {
              std::memcpy(s.cur[j].d.data(), xn, nn * sizeof(float));
            } else {
              add_inplace(s.cur[j].d.data(),
                          s.tB[j][(m - 1) & 1].d.data(), nn);
            }
            leaky_relu_write(s.cur[j].d.data(), s.tin[j].d.data(), nn, 0.1f);
            im2col_to_panel_f16(s.tin[j].d.data(), cvl->in_c, Tn,
                                cvl->k, cvl->dilation, pbl.buf, true);
          } else {
            // tA 就地 lrelu 后产 panel (与串行序一致)
            leaky_relu_io(s.tA[j][m & 1].d.data(), nn, 0.1f);
            im2col_to_panel_f16(s.tA[j][m & 1].d.data(), cvl->in_c, Tn,
                                cvl->k, cvl->dilation, pbl.buf, true);
          }
        };
        nodes.push_back(std::move(nd));
      }
    }

    const double tb = now_ms_e6();
    kern::amx_chain_run(nodes.data(), nodes.size());
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
    // x += cond(ge) (广播 gT==1)
    double t_cond_ew = 0;
    { double te = tic();
      for (size_t c = 0; c < 768; ++c) {
        const float cv = cb.d[c * gT];
        for (size_t t = 0; t < x.T; ++t) x.d[c * x.T + t] += cv;
      }
      if (prof) t_cond_ew = now_ms_e6() - te;
    }

    // E10: per-stage timing (lightweight, only when prof)
    double stage_ups[kStages] = {};
    double stage_pre_ew[kStages] = {};
    double stage_swap[kStages] = {};
    double stage_res_t[kStages] = {};
    double stage_res_pool[kStages] = {};
    // E6: scratch 持久化 (thread_local) — 每 stage 重构造会反复 mmap/缺页零填
    static thread_local Tensor2D xu, scratch, sum;
    for (size_t i = 0; i < kStages; ++i) {
      { double te = tic();
        leaky_relu_io(x.d.data(), x.d.size(), 0.1f);
        if (prof) stage_pre_ew[i] = now_ms_e6() - te;
      }
      t0 = tic();
      ups[i].forward(x, xu, scratch);
      if (prof) { T.dec_ups += now_ms_e6() - t0; stage_ups[i] = now_ms_e6() - t0; }
      { double ts = tic();
        x.d.swap(xu.d);
        x.C = xu.C;
        x.T = xu.T;
        {
          std::string n = "dbg_dec_up" + std::to_string(i);
          dm.dump(n.c_str(), x);
        }
        if (prof) stage_swap[i] = now_ms_e6() - ts;
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
      double t_res = tic();
      t0 = tic();
      if (res_forward_batched(i, x, sum, &batch_ms)) {
        if (prof) T.dec_res += batch_ms;
      } else {
        static thread_local Tensor2D cur, t_in, t_a, t_b;
        // E10-MEM: 静态 thread_local 跨调用复用; 只增长 capacity, 避免小
        // 调用收缩触发 realloc churn (与 res_forward_batched 同样的策略)
        auto ensure_tl = [](Tensor2D& t, size_t c, size_t tt) {
          const size_t need = c * tt;
          if (t.d.capacity() < need) t.d.resize(need);
          else if (t.d.size() < need) t.d.resize(need);
          t.C = c; t.T = tt;
        };
        bool first = true;
        for (size_t j = 0; j < kResKernels; ++j) {
          const ResBlock1& rb = resblocks[i][j];
          ensure_tl(cur, x.C, x.T);
          std::memcpy(cur.d.data(), x.d.data(), x.d.size() * sizeof(float));
          for (size_t jj = 0; jj < 3; ++jj) {
            // torch: xt=lrelu(x); xt=c1(xt); xt=lrelu(xt); xt=c2(xt); x=xt+x
            // (lrelu 不污染残差分支 — 用独立缓冲 t_in)
            double te = tic();
            ensure_tl(t_in, cur.C, cur.T);
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
      if (prof) { stage_res_t[i] = now_ms_e6() - t_res; stage_res_pool[i] = batch_ms; }
      x.d.swap(sum.d);
      {
        std::string n = "dbg_dec_res" + std::to_string(i);
        dm.dump(n.c_str(), x);
      }
    }
    double t_final_ew = 0, t_tanh = 0;
    { double te = tic();
      leaky_relu_io(x.d.data(), x.d.size(), 0.01f);  // F.leaky_relu 默认 slope
      if (prof) t_final_ew = now_ms_e6() - te;
    }
    t0 = tic();
    conv_post.forward(x, out);
    if (prof) T.dec_post += now_ms_e6() - t0;
    { double te = tic();
      for (float& v : out.d) v = std::tanh(v);
      if (prof) t_tanh = now_ms_e6() - te;
    }
    if (prof) {
      double total = T.dec_pre + T.dec_cond + t_cond_ew + T.dec_ups +
                     T.dec_post + t_final_ew + t_tanh;
      for (size_t i = 0; i < kStages; ++i) total += stage_res_t[i];
      const auto pct = [&](double v) { return total > 0 ? v / total * 100.0 : 0; };
      std::fprintf(stderr,
          "[sovits-dec-timing] total=%.1fms (prof=%s)\n"
          "  conv_pre       %8.2fms (%.1f%%)\n"
          "  cond+broadcast %8.2fms (%.1f%%)\n",
          total, prof ? "on" : "off",
          T.dec_pre, pct(T.dec_pre),
          T.dec_cond + t_cond_ew, pct(T.dec_cond + t_cond_ew));
      for (size_t i = 0; i < kStages; ++i) {
        const double main_work = stage_res_t[i] - stage_res_pool[i];
        std::fprintf(stderr,
            "  stage %d: ups=%6.2fms pre_ew=%5.2fms res=%6.2fms "
            "pool=%6.2fms main_work=%5.2fms swap=%5.2fms\n",
            (int)i, stage_ups[i], stage_pre_ew[i], stage_res_t[i],
            stage_res_pool[i], main_work, stage_swap[i]);
      }
      std::fprintf(stderr,
          "  final_lrelu+tanh %8.2fms (%.1f%%)\n"
          "  conv_post      %8.2fms (%.1f%%)\n",
          t_final_ew + t_tanh, pct(t_final_ew + t_tanh),
          T.dec_post, pct(T.dec_post));
    }
  }
};

}  // namespace gsv::sovits
