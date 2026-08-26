// bench_gemv.cpp — 测试 GEMV 实际耗时
#include "kern/gemv_fmlal.hpp"
#include "runtime/threadpool.hpp"
#include <arm_neon.h>
#include <chrono>
#include <cstdio>
#include <vector>
#include <cstring>

int main(int argc, char** argv) {
  // Try multiple GEMV shapes used in AR decode
  struct Shape { const char* name; size_t out, in; };
  Shape shapes[] = {
    {"wqkv", 1536, 512},
    {"wout", 512, 512},
    {"w1",   2048, 512},
    {"w2",   512, 2048},
  };
  for (auto& s : shapes) {
    const size_t out = s.out, in = s.in;
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
    // Warmup
    for (int i = 0; i < 5; ++i) gsv::kern::gemv_f16x_fmlal(w.data(), xh.data(), y.data(), out, in);
    const int N = 200;
    setenv("GSV_GEMV_E_DISABLE", "1", 1);
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < N; ++i)
      gsv::kern::gemv_f16x_fmlal(w.data(), xh.data(), y.data(), out, in);
    auto t1 = std::chrono::steady_clock::now();
    double p_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N;
    unsetenv("GSV_GEMV_E_DISABLE");
    t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < N; ++i)
      gsv::kern::gemv_f16x_fmlal(w.data(), xh.data(), y.data(), out, in);
    t1 = std::chrono::steady_clock::now();
    double pe_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N;
    std::printf("%-6s  P-only=%.2f us  P+E=%.2f us  speedup=%.2fx\n",
                s.name, p_us, pe_us, p_us / pe_us);
  }
  return 0;
}
