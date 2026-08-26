// prefill_bench.cpp — E11-2 microbench: prefill (24 layers) at fixed S
// Usage: build/prefill_bench <S>
#include "ar/t2s_engine.hpp"
#include "runtime/gsv_loader.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char** argv) {
  size_t S = (argc > 1) ? std::atoi(argv[1]) : 256;
  gsv::rt::GsvFile f("weights/ar_s1v3.gsv");
  gsv::ar::T2SEngine eng(f);
  const auto& dm = eng.dims();
  const size_t D = dm.d_model, FF = dm.ffn, NL = dm.n_layers;
  std::printf("D=%zu H=%zu HD=%zu FF=%zu NL=%zu\n", D, dm.n_heads, D/dm.n_heads, FF, NL);

  // Build S×D input (sin PE rows, like test_ar_unit does)
  std::vector<float> x(S * D);
  std::vector<float> pe(D);
  for (size_t t = 0; t < S; ++t) {
    eng.pe_row(pe.data(), t);
    for (size_t d = 0; d < D; ++d) x[t * D + d] = pe[d];
  }
  // KV caches
  const size_t cap = S + 32;
  std::vector<std::vector<float>> kc(NL), vc(NL);
  for (auto& v : kc) v.assign(cap * D, 0.f);
  for (auto& v : vc) v.assign(cap * D, 0.f);

  // Warmup
  for (int i = 0; i < 3; ++i) {
    auto xc = x;
    for (size_t l = 0; l < NL; ++l)
      eng.block_prefill(l, xc.data(), S, 0, 0, kc[l].data(), vc[l].data());
  }
  // Time
  const int N_ITER = 10;
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < N_ITER; ++i) {
    auto xc = x;
    for (size_t l = 0; l < NL; ++l)
      eng.block_prefill(l, xc.data(), S, 0, 0, kc[l].data(), vc[l].data());
  }
  auto t1 = std::chrono::steady_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / N_ITER;
  std::printf("S=%zu  prefill=%.2f ms (24层, 0.8ms/S_estimate 仅供参考)\n", S, ms);
  return 0;
}
