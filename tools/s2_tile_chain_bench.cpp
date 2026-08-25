// s2_tile_chain_bench.cpp — E10-K2 微基准: per-tile DAG vs per-node DAG
//
// 场景取自 SoVITS dec S2 真实结构: 18 节点 (3 chain × 6 depth), C=96, T=43840,
// kernel {3,7,11}, dilation {1,3,5}。每个节点分 nmb*nnb 个 32×32 tile,
// per-tile 路径下展开为 73,980 个 AmxTileChainLink 节点 + 1 个 DAG。
//
// 对照:
//   chain       = amx_chain_run (per-node DAG, 18 节点, 当前 E10-K 现状)
//   tile_chain  = amx_tile_chain_run (per-tile DAG, 73,980 节点, E10-K2)
// 正确性: 两路结果必须逐位一致 (同内核同 tile 划分, 仅调度拓扑更细)。
// 目标: tile_chain ≤110ms (当前 chain 约 157ms, 理论地板 60ms)。
#include "kern/gemm_f16_amx.hpp"
#include "sovits/sovits_types.hpp"
#include "sovits/nn_ops.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <utility>
#include <vector>

namespace {

double now_ms() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

}  // namespace

int main() {
  if (!gsv::kern::amx_gemm_available()) {
    std::printf("AMX unavailable\n");
    return 1;
  }
  const size_t C = 96, T = 43840;
  const size_t ks[3] = {3, 7, 11};
  const size_t Ks[3] = {289, 673, 1057};
  const size_t dils[3] = {1, 3, 5};
  const size_t CHAIN = 6, NCHAIN = 3;
  std::mt19937 rng(20260825);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  // 随机激活 [C,T]
  std::vector<float> x(C * T);
  for (auto& v : x) v = dist(rng);

  // 18 个权重 panel + 18 个激活 panel (per-node 路径: 一个 panel 一节点)
  const size_t total = NCHAIN * CHAIN;
  std::vector<gsv::kern::AmxPanel> pas(total);
  std::vector<gsv::kern::AmxPanel> pbs(total);
  std::vector<std::vector<uint16_t>> wraw(total);
  for (size_t j = 0; j < NCHAIN; ++j) {
    for (size_t p = 0; p < CHAIN; ++p) {
      const size_t kk = ks[j];
      const size_t idx = j * CHAIN + p;
      wraw[idx].resize(C * kk);
      for (auto& v : wraw[idx])
        v = static_cast<uint16_t>(__fp16(dist(rng)));
      pas[idx] = gsv::kern::amx_pack(wraw[idx].data(), C, kk);
      pas[idx].rows = C;
      pas[idx].K = Ks[j];
    }
  }
  // 18 个 panel 缓冲 (per-node 准备完整 panel)
  for (size_t j = 0; j < NCHAIN; ++j) {
    for (size_t p = 0; p < CHAIN; ++p) {
      const size_t K = Ks[j];
      const size_t idx = j * CHAIN + p;
      pbs[idx].rows = T;
      pbs[idx].K = K;
      pbs[idx].buf.resize(((T + 31) / 32) * K * 64 + 64);
    }
  }
  // 2 个输出缓冲 (chain 流水用, 偶/奇交替)
  std::vector<std::vector<float>> out(2, std::vector<float>(C * T, 0.0f));
  std::vector<std::vector<float>> out_tile(2, std::vector<float>(C * T, 0.0f));

  // ---- 准备 S2 的 prepare 钩子 (模拟 dec resblock) ----
  // 用链式状态: cur[j] 累加 tB[m-1], tin[j] = lrelu(cur), tA[j][m&1] = lrelu 数学输出
  // 为简化, 用随机张量模拟各步中间结果 (bitwise 验证不要求语义匹配, 只要求
  // per-tile 与 per-node 的 im2col+math 结果一致)
  std::vector<std::vector<float>> cur(NCHAIN, std::vector<float>(C * T));
  std::vector<std::vector<float>> tin(NCHAIN, std::vector<float>(C * T));
  std::vector<std::vector<float>> tA(NCHAIN, std::vector<float>(C * T));
  std::vector<std::vector<float>> tB(NCHAIN, std::vector<float>(C * T));
  for (size_t j = 0; j < NCHAIN; ++j) {
    for (auto& v : cur[j]) v = dist(rng);
    for (auto& v : tin[j]) v = dist(rng);
    for (auto& v : tA[j]) v = dist(rng);
    for (auto& v : tB[j]) v = dist(rng);
  }

  auto make_prep_node = [&](size_t j, size_t p, gsv::kern::AmxPanel& pb) {
    const size_t m = p / 2;
    const bool even = (p % 2) == 0;
    const size_t kk = ks[j];
    const size_t dil = even ? dils[m] : 1;
    return [j, m, even, kk, dil, &pb, &cur, &tin, &tA, &tB, &x]() {
      const size_t C = 96, T = 43840;
      if (even) {
        if (m == 0) {
          std::memcpy(cur[j].data(), x.data(), C * T * sizeof(float));
        } else {
          gsv::sovits::add_inplace(cur[j].data(), tB[j].data(), C * T);
        }
        gsv::sovits::leaky_relu_write(cur[j].data(), tin[j].data(), C * T, 0.1f);
        gsv::sovits::im2col_to_panel_f16(tin[j].data(), C, T, kk, dil, pb.buf,
                                          true);
      } else {
        gsv::sovits::leaky_relu_io(tA[j].data(), C * T, 0.1f);
        gsv::sovits::im2col_to_panel_f16(tA[j].data(), C, T, kk, dil, pb.buf,
                                          true);
      }
    };
  };

  // ---- per-node 路径: 18 个 AmxChainLink ----
  std::vector<gsv::kern::AmxChainLink> c_nodes(total);
  for (size_t j = 0; j < NCHAIN; ++j) {
    for (size_t p = 0; p < CHAIN; ++p) {
      const size_t idx = j * CHAIN + p;
      c_nodes[idx].chain_id = static_cast<int>(j);
      c_nodes[idx].depth = static_cast<int>(p);
      c_nodes[idx].pa = &pas[idx];
      c_nodes[idx].pb = &pbs[idx];
      c_nodes[idx].c = out[idx % 2].data();
      c_nodes[idx].M = C;
      c_nodes[idx].N = T;
      c_nodes[idx].prepare = make_prep_node(j, p, pbs[idx]);
    }
  }

  // ---- per-tile 路径: 18 * nmb * nnb 个 AmxTileChainLink ----
  const size_t nmb = (C + 31) / 32, nnb = (T + 31) / 32;
  const size_t ntiles = nmb * nnb;
  const size_t total_tiles = total * ntiles;
  std::vector<gsv::kern::AmxTileChainLink> t_nodes;
  t_nodes.reserve(total_tiles);
  // 准备 im2col 源数据 (提前 fill, 避免 per-tile 重复 fill 干扰计时)
  // 准备阶段: 对每节点 0..j*p-1, 模拟上游链的 prepare+math 已完成
  // (为公平对比, per-node 路径也走相同准备)
  for (size_t j = 0; j < NCHAIN; ++j) {
    for (size_t p = 0; p < CHAIN; ++p) {
      const size_t idx = j * CHAIN + p;
      const size_t m = p / 2;
      const bool even = (p % 2) == 0;
      const size_t kk = ks[j];
      const size_t dil = even ? dils[m] : 1;
      for (size_t mb = 0; mb < nmb; ++mb) {
        for (size_t nb = 0; nb < nnb; ++nb) {
          gsv::kern::AmxTileChainLink nd;
          nd.chain_id = static_cast<int>(j);
          nd.depth = static_cast<int>(p);
          nd.tile_mb = mb;
          nd.tile_nb = nb;
          nd.pa = &pas[idx];
          nd.pb = &pbs[idx];
          nd.c = out_tile[idx % 2].data();
          nd.M = C;
          nd.N = T;
          const size_t row_b = nb * 32;
          const size_t row_e = std::min(row_b + 32, T);
          // prepare: 仅处理 tile 行区间, 写 panel 对应段
          nd.prepare = [j, m, even, kk, dil, &pbs = pbs[idx], &cur, &tin,
                         &tA, &tB, &x, row_b, row_e](size_t rb, size_t re) {
            (void)rb; (void)re;  // tile_mb/nb 已编码在闭包, 区间一致
            constexpr size_t C = 96;
            constexpr size_t T = 43840;
            const size_t tile_len = row_e - row_b;
            if (even) {
              if (m == 0) {
                for (size_t c = 0; c < C; ++c)
                  std::memcpy(cur[j].data() + c * T + row_b,
                              x.data() + c * T + row_b, tile_len * sizeof(float));
              } else {
                for (size_t c = 0; c < C; ++c)
                  gsv::sovits::add_inplace(cur[j].data() + c * T + row_b,
                                           tB[j].data() + c * T + row_b,
                                           tile_len);
              }
              for (size_t c = 0; c < C; ++c)
                gsv::sovits::leaky_relu_write(cur[j].data() + c * T + row_b,
                                             tin[j].data() + c * T + row_b,
                                             tile_len, 0.1f);
              gsv::sovits::im2col_to_panel_f16_tile(tin[j].data(), C, T, kk, dil,
                                                    row_b, row_e, pbs.buf, true);
            } else {
              for (size_t c = 0; c < C; ++c)
                gsv::sovits::leaky_relu_io(tA[j].data() + c * T + row_b, tile_len,
                                           0.1f);
              gsv::sovits::im2col_to_panel_f16_tile(tA[j].data(), C, T, kk, dil,
                                                    row_b, row_e, pbs.buf, true);
            }
          };
          t_nodes.push_back(std::move(nd));
        }
      }
    }
  }

  // warm up
  gsv::kern::amx_chain_run(c_nodes.data(), c_nodes.size());
  std::fill(out[0].begin(), out[0].end(), 0.f);
  std::fill(out[1].begin(), out[1].end(), 0.f);
  gsv::kern::amx_tile_chain_run(t_nodes.data(), t_nodes.size());

  // ---- 计时 ----
  constexpr int ROUNDS = 5;
  double best_chain = 1e30, best_tile = 1e30;
  for (int r = 0; r < ROUNDS; ++r) {
    // chain 路径
    for (auto& v : out[0]) v = 0.f;
    for (auto& v : out[1]) v = 0.f;
    auto t0 = now_ms();
    gsv::kern::amx_chain_run(c_nodes.data(), c_nodes.size());
    best_chain = std::min(best_chain, now_ms() - t0);
    // tile_chain 路径
    for (auto& v : out_tile[0]) v = 0.f;
    for (auto& v : out_tile[1]) v = 0.f;
    t0 = now_ms();
    gsv::kern::amx_tile_chain_run(t_nodes.data(), t_nodes.size());
    best_tile = std::min(best_tile, now_ms() - t0);
  }

  // ---- 逐位一致性验证 ----
  size_t diff = 0;
  for (size_t h = 0; h < 2; ++h)
    for (size_t i = 0; i < out[h].size(); ++i)
      if (std::memcmp(&out[h][i], &out_tile[h][i], 4) != 0) ++diff;

  std::printf("=== S2 per-tile vs per-node (C=%zu T=%zu, %zu tile-nodes) ===\n",
              C, T, total_tiles);
  std::printf("  chain      (per-node):  best=%.2fms\n", best_chain);
  std::printf("  tile_chain (per-tile):  best=%.2fms\n", best_tile);
  std::printf("  speedup:     %.2fx\n", best_chain / best_tile);
  std::printf("  diff:        %zu  (0=bitwise identical)\n", diff);
  std::printf("  target:      <=110ms %s\n",
              best_tile <= 110.0 ? "PASS" : "FAIL");
  std::printf("  floor:       60ms, ratio=%.2fx\n", best_tile / 60.0);
  return diff == 0 ? 0 : 1;
}
