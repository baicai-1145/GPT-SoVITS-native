// amx_batch_bench.cpp — E9/E10-K 微基准: 逐节点 vs phase-屏障 vs chain-DAG
//
// 场景取自 SoVITS dec 真实结构:
//   A) 大块链式: 每 stage 3 个独立 resblock × 链深 6 conv = 18 GEMM,
//      形状 (M=384/192/96, T=2000, K=1152/2688/4224)
//   B) 小块多派发: 深层 stage (M=96, T=2000→8000, K=672), 同步开销主导
//   C) 小块 + prepare 入池流水: 与 B 同形, prepare(im2col) 在池 worker 内
//
// 对照:
//   seq   = 逐 gemm_f16_amx_pp (现状, 每 GEMM 一次池往返)
//   batch = amx_batch_run 按 phase 分桶 (链深=phase, 全局 barrier)
//   chain = amx_chain_run 按 chain_id/depth DAG (同链串行, 跨链并行, 无 barrier)
// 正确性: 三路结果必须逐位一致 (同一内核同 tile 划分语义)。
#include "kern/gemm_f16_amx.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
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
  // 场景 B/C: 深层小块
  const Shape sb[3] = {{96, 8000, 672}, {96, 8000, 672}, {96, 8000, 672}};

  for (int sc = 0; sc < 3; ++sc) {
    const bool with_prep = sc == 2;
    const Shape* sh = (sc == 1 || sc == 2) ? sb : sa;
    const int CHAIN = 6;  // 每 resblock 6 conv (convs1×3 + convs2×3)
    const int NCHAIN = 3;
    // 准备 panels: 3 链各自权重 panel
    std::vector<gsv::kern::AmxPanel> pas(NCHAIN);
    for (int j = 0; j < NCHAIN; ++j) pas[j] = mkpanel(sh[j].M, sh[j].K);
    // 场景 C: 原始 f16 激活 [N,K], prepare 时池内打包 (模拟 im2col 直产)
    std::vector<std::vector<uint16_t>> braw(NCHAIN * CHAIN);
    std::vector<gsv::kern::AmxPanel> pbs(NCHAIN * CHAIN);
    for (int j = 0; j < NCHAIN; ++j)
      for (int d = 0; d < CHAIN; ++d) {
        const size_t id = j * CHAIN + d;
        if (!with_prep) {
          pbs[id] = mkpanel(sh[j].N, sh[j].K);
        } else {
          braw[id].resize(sh[j].N * sh[j].K);
          for (auto& v : braw[id]) {
            __fp16 h((float)((int)(rng() % 4096) - 2048) * 0.001953125f);
            std::memcpy(&v, &h, 2);
          }
        }
      }
    const size_t total_nodes = NCHAIN * CHAIN;
    std::vector<std::vector<float>> cs(total_nodes), cs_chain(total_nodes);
    for (size_t i = 0; i < total_nodes; ++i) {
      const size_t idx = i / CHAIN;
      cs[i].assign(sh[idx].M * sh[idx].N, 0.f);
      cs_chain[i].assign(sh[idx].M * sh[idx].N, 0.f);
    }
    constexpr int ROUNDS = 10;

    double t_seq = 1e30, t_bat = 1e30, t_chain = 1e30;

    // 预建 chain 节点 (prepare 钩子在运行期刷新 pbs buf, 无需重建节点)
    std::vector<gsv::kern::AmxChainLink> cnodes(total_nodes);
    std::vector<std::function<void()>> cpreps(total_nodes);
    for (int d = 0; d < CHAIN; ++d)
      for (int j = 0; j < NCHAIN; ++j) {
        const size_t id = j * CHAIN + d;
        cnodes[id].chain_id = j;
        cnodes[id].depth = d;
        cnodes[id].pa = &pas[j];
        cnodes[id].pb = &pbs[id];
        cnodes[id].c = cs_chain[id].data();
        cnodes[id].M = sh[j].M;
        cnodes[id].N = sh[j].N;
        if (with_prep) {
          cpreps[id] = [&braw, &pbs, id, N = sh[j].N, K = sh[j].K] {
            gsv::kern::amx_pack_into(braw[id].data(), N, K, pbs[id].buf);
            pbs[id].rows = N;
            pbs[id].K = K;
          };
          cnodes[id].prepare = cpreps[id];
        }
      }

    for (int r = 0; r < ROUNDS; ++r) {
      // seq: 逐 gemm_f16_amx_pp
      auto t0 = now_ms();
      for (int j = 0; j < NCHAIN; ++j)
        for (int d = 0; d < CHAIN; ++d) {
          const size_t id = j * CHAIN + d;
          if (with_prep) {
            gsv::kern::amx_pack_into(braw[id].data(), sh[j].N, sh[j].K,
                                     pbs[id].buf);
            pbs[id].rows = sh[j].N;
            pbs[id].K = sh[j].K;
          }
          gsv::kern::gemm_f16_amx_pp(pas[j], pbs[id], cs[id].data(),
                                     sh[j].M, sh[j].N);
        }
      t_seq = std::min(t_seq, now_ms() - t0);

      // batch: phase-屏障 (现状)
      std::vector<gsv::kern::AmxBatchNode> nodes;
      nodes.reserve(total_nodes);
      std::vector<std::function<void()>> preps(total_nodes);
      for (int d = 0; d < CHAIN; ++d)
        for (int j = 0; j < NCHAIN; ++j) {
          const size_t id = j * CHAIN + d;
          gsv::kern::AmxBatchNode nd;
          nd.phase = d;
          nd.pa = &pas[j];
          nd.pb = &pbs[id];
          nd.c = cs[id].data();
          nd.M = sh[j].M;
          nd.N = sh[j].N;
          if (with_prep) {
            preps[id] = [&braw, &pbs, id, N = sh[j].N, K = sh[j].K] {
              gsv::kern::amx_pack_into(braw[id].data(), N, K, pbs[id].buf);
              pbs[id].rows = N;
              pbs[id].K = K;
            };
            nd.prepare = preps[id];
          }
          nodes.push_back(std::move(nd));
        }
      t0 = now_ms();
      gsv::kern::amx_batch_run(nodes.data(), nodes.size());
      t_bat = std::min(t_bat, now_ms() - t0);

      // chain: 按-chain DAG (E10-K, 无 barrier)
      t0 = now_ms();
      gsv::kern::amx_chain_run(cnodes.data(), cnodes.size());
      t_chain = std::min(t_chain, now_ms() - t0);
    }

    // 一致性: batch(cs) 与 chain(cs_chain) 逐位一致
    size_t diff_bc = 0;
    for (size_t i = 0; i < total_nodes; ++i)
      for (size_t k = 0; k < cs[i].size(); ++k)
        if (std::memcmp(&cs[i][k], &cs_chain[i][k], 4) != 0) ++diff_bc;

    static const char* kNames[3] = {
        "A大块链式(384x2000)x18       ",
        "B深层小块(96x8000)x18       ",
        "C小块+prep入池流水(96x8000)x18"};
    std::printf(
        "%s: seq=%7.3fms batch=%7.3fms chain=%7.3fms  "
        "chain_seq=%.2fx chain_bat=%.2fx batch_bat=%.2fx  "
        "diff_batch_chain=%zu\n",
        kNames[sc], t_seq, t_bat, t_chain, t_seq / t_chain,
        t_bat / t_chain, t_seq / t_bat, diff_bc);
  }
  return 0;
}
