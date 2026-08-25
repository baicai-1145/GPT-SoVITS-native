// amx_batch_bench.cpp — E9 微基准: 逐节点派发 vs 批量单次派发
//
// 场景取自 SoVITS dec 真实结构:
//   A) 大块链式: 每 stage 3 个独立 resblock × 链深 6 conv = 18 GEMM,
//      形状 (M=384/192/96, T=2000, K=1152/2688/4224)
//   B) 小块多派发: 深层 stage (M=96, T=2000→8000, K=672), 同步开销主导
//
// 对照:
//   seq   = 逐 gemm_f16_amx_pp (现状, 每 GEMM 一次池往返)
//   batch = amx_batch_run 按 phase 分桶 (链深=phase)
// 正确性: 两路结果必须逐位一致 (同一内核同 tile 划分语义)。
#include "kern/gemm_f16_amx.hpp"

#include <algorithm>
#include <functional>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

double now_ms() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

struct Shape {
  size_t M, N, K;
};

}  // namespace

int main() {
  if (!gsv::kern::amx_gemm_available()) {
    std::printf("AMX unavailable\n");
    return 1;
  }
  std::mt19937 rng(20260825);
  auto mkpanel = [&](size_t rows, size_t K) {
    std::vector<uint16_t> w(rows * K);
    for (auto& v : w) {
      __fp16 h((float)((int)(rng() % 4096) - 2048) * 0.001953125f);
      std::memcpy(&v, &h, 2);
    }
    return gsv::kern::amx_pack(w.data(), rows, K);
  };

  // 场景 A: stage0 三链 × 链深 6
  const Shape sa[3] = {{384, 2000, 1152}, {384, 2000, 2688}, {384, 2000, 4224}};
  // 场景 B: 深层小块
  const Shape sb[3] = {{96, 8000, 672}, {96, 8000, 672}, {96, 8000, 672}};

  for (int sc = 0; sc < 3; ++sc) {
    const bool with_prep = sc == 2;
    const Shape* sh = (sc == 1 || sc == 2) ? sb : sa;
    const int CHAIN = 6;  // 每 resblock 6 conv (convs1×3 + convs2×3)
    // 准备 panels: 3 链各自权重 panel + 共享激活 panel (每相位独立缓冲以隔离)
    std::vector<gsv::kern::AmxPanel> pas(3), pbs(3 * CHAIN);
    for (int j = 0; j < 3; ++j) pas[j] = mkpanel(sh[j].M, sh[j].K);
    // 场景 C: 原始 f16 激活 [N,K], prepare 时池内打包 (模拟 im2col 直产)
    std::vector<std::vector<uint16_t>> braw(3 * CHAIN);
    for (int j = 0; j < 3; ++j)
      for (int d = 0; d < CHAIN; ++d) {
        if (!with_prep) {
          pbs[j * CHAIN + d] = mkpanel(sh[j].N, sh[j].K);
        } else {
          braw[j * CHAIN + d].resize(sh[j].N * sh[j].K);
          for (auto& v : braw[j * CHAIN + d]) {
            __fp16 h((float)((int)(rng() % 4096) - 2048) * 0.001953125f);
            std::memcpy(&v, &h, 2);
          }
        }
      }
    std::vector<std::vector<float>> cs(3 * CHAIN);
    for (int j = 0; j < 3; ++j)
      for (int d = 0; d < CHAIN; ++d)
        cs[j * CHAIN + d].assign(sh[j].M * sh[j].N, 0.f);
    constexpr int ROUNDS = 10;

    double t_seq = 1e30, t_bat = 1e30;
    for (int r = 0; r < ROUNDS; ++r) {
      auto t0 = now_ms();
      for (int j = 0; j < 3; ++j)
        for (int d = 0; d < CHAIN; ++d) {
          const size_t id = j * CHAIN + d;
          if (with_prep) {  // 现状口径: caller 串行 prep → 提交 → 等
            gsv::kern::amx_pack_into(braw[id].data(), sh[j].N, sh[j].K,
                                     pbs[id].buf);
            pbs[id].rows = sh[j].N;
            pbs[id].K = sh[j].K;
          }
          gsv::kern::gemm_f16_amx_pp(pas[j], pbs[id], cs[id].data(), sh[j].M,
                                     sh[j].N);
        }
      t_seq = std::min(t_seq, now_ms() - t0);

      // batch: prepare 在池内执行 (与同相位其他节点数学重叠)
      std::vector<gsv::kern::AmxBatchNode> nodes;
      nodes.reserve(3 * CHAIN);
      std::vector<std::function<void()>> preps(3 * CHAIN);
      for (int d = 0; d < CHAIN; ++d)
        for (int j = 0; j < 3; ++j) {
          gsv::kern::AmxBatchNode nd;
          nd.phase = d;
          nd.pa = &pas[j];
          nd.pb = &pbs[j * CHAIN + d];
          nd.c = cs[j * CHAIN + d].data();
          nd.M = sh[j].M;
          nd.N = sh[j].N;
          if (with_prep) {
            const size_t id = j * CHAIN + d;
            preps[id] = [&braw, &pbs, id, N = sh[j].N, K = sh[j].K] {
              gsv::kern::amx_pack_into(braw[id].data(), N, K, pbs[id].buf);
            };
            nd.prepare = preps[id];
          }
          nodes.push_back(std::move(nd));
        }
      t0 = now_ms();
      gsv::kern::amx_batch_run(nodes.data(), nodes.size());
      t_bat = std::min(t_bat, now_ms() - t0);
    }

    // 一致性: batch 路径与 seq 路径结果逐位一致
    size_t diff = 0;
    std::vector<std::vector<float>> ref(3 * CHAIN);
    for (int j = 0; j < 3; ++j)
      for (int d = 0; d < CHAIN; ++d) {
        if (with_prep) {  // 确保 panel 就绪后再取参考
          gsv::kern::amx_pack_into(braw[j * CHAIN + d].data(), sh[j].N,
                                   sh[j].K, pbs[j * CHAIN + d].buf);
          pbs[j * CHAIN + d].rows = sh[j].N;
          pbs[j * CHAIN + d].K = sh[j].K;
        }
        ref[j * CHAIN + d] = cs[j * CHAIN + d];
        gsv::kern::gemm_f16_amx_pp(pas[j], pbs[j * CHAIN + d],
                                   cs[j * CHAIN + d].data(), sh[j].M, sh[j].N);
        const auto& a = ref[j * CHAIN + d];
        const auto& b = cs[j * CHAIN + d];
        for (size_t i = 0; i < a.size(); ++i)
          if (std::memcmp(&a[i], &b[i], 4)) ++diff;
      }

    static const char* kNames[3] = {
        "A大块链式(384x2000,K1152/2688/4224)x18",
        "B深层小块(96x8000,K672)x18            ",
        "C小块+prep入池流水(96x8000,K672)x18   "};
    std::printf("%s: seq=%7.3fms batch=%7.3fms  加速=%.2fx  节省=%5.1f%%  bitwise_diff=%zu\n",
                kNames[sc], t_seq, t_bat, t_seq / t_bat,
                100.0 * (t_seq - t_bat) / t_seq, diff);
  }
  return 0;
}
