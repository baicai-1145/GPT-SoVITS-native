// bench_bw.cpp — 测量 GEMV 实际内存带宽
#include "kern/gemv_fmlal.hpp"
#include "runtime/threadpool.hpp"
#include <arm_neon.h>
#include <chrono>
#include <cstdio>
#include <vector>
#include <cstring>

int main() {
  // 模拟 wqkv 形状
  const size_t out = 1536, in = 512;
  std::vector<uint16_t> w(out * in);
  std::vector<uint16_t> xh(in);
  std::vector<float> y(out);
  srand(42);
  for (auto& v : w) {
    __fp16 h = (float(rand()) / RAND_MAX - 0.5f) * 0.1f;
    __builtin_memcpy(&v, &h, 2);
  }
  for (auto& v : xh) {
    __fp16 h = (float(rand()) / RAND_MAX - 0.5f) * 0.1f;
    __builtin_memcpy(&v, &h, 2);
  }

  // 数据量: 每次 GEMV 读 out*in*2 = 1.5MB, 写 out*4 = 6KB, 读 xh*2 = 1KB
  // 有效带宽 = 1.5MB / 1us (假设 1us/GEMV) = 1.5TB/s (L2/L1 缓存命中)
  // 真正衡量要看 hot/cold cache —— GEMV 调用间隔 < 1us 时 L2 命中
  // 现实: 120 GEMV/token, 每 token 4ms 间隔, L2 命中率高
  const int N = 200;
  setenv("GSV_GEMV_E_DISABLE", "1", 1);
  for (int i = 0; i < 5; ++i)
    gsv::kern::gemv_f16x_fmlal(w.data(), xh.data(), y.data(), out, in);
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < N; ++i)
    gsv::kern::gemv_f16x_fmlal(w.data(), xh.data(), y.data(), out, in);
  auto t1 = std::chrono::steady_clock::now();
  double p_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N;
  unsetenv("GSV_GEMV_E_DISABLE");
  for (int i = 0; i < 5; ++i)
    gsv::kern::gemv_f16x_fmlal(w.data(), xh.data(), y.data(), out, in);
  t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < N; ++i)
    gsv::kern::gemv_f16x_fmlal(w.data(), xh.data(), y.data(), out, in);
  t1 = std::chrono::steady_clock::now();
  double pe_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N;
  // 估算有效带宽 (假设 1.5MB 读 + 6KB 写 + 1KB 读 xh, 实际 L2 命中下不可达 1.5MB)
  const double bytes_per_call = 1.5e6 + 6e3 + 1e3;
  double p_bw = bytes_per_call / p_us / 1e3;  // GB/s
  double pe_bw = bytes_per_call / pe_us / 1e3;
  std::printf("GEMV P-only: %.2f us/call  est_bw=%.1f GB/s\n", p_us, p_bw);
  std::printf("GEMV P+E:    %.2f us/call  est_bw=%.1f GB/s\n", pe_us, pe_bw);
  return 0;
}
