// bench_pool2.cpp — 测试 pool dispatch 开销 (空任务)
#include "runtime/threadpool.hpp"
#include <chrono>
#include <cstdio>

int main() {
  using clk = std::chrono::steady_clock;
  // Warmup
  for (int i = 0; i < 100; ++i) {
    gsv::rt::parallel_for(4, 1, [](size_t, size_t) {}, gsv::rt::Qos::UserInitiated);
  }
  // Time P-only 4 no-op tasks
  auto t0 = clk::now();
  for (int i = 0; i < 10000; ++i) {
    gsv::rt::parallel_for(4, 1, [](size_t, size_t) {}, gsv::rt::Qos::UserInitiated);
  }
  auto t1 = clk::now();
  double p_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / 10000;
  std::printf("P-only 4 no-op: %.2f us/call\n", p_us);

  // Time gemv_pool 10 no-op tasks
  t0 = clk::now();
  for (int i = 0; i < 10000; ++i) {
    gsv::rt::parallel_for_full(10, 1, [](size_t, size_t) {});
  }
  t1 = clk::now();
  double g_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / 10000;
  std::printf("GEMV pool 10 no-op: %.2f us/call\n", g_us);

  // Time gemv_pool 40 no-op tasks
  t0 = clk::now();
  for (int i = 0; i < 10000; ++i) {
    gsv::rt::parallel_for_full(40, 1, [](size_t, size_t) {});
  }
  t1 = clk::now();
  double g40_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / 10000;
  std::printf("GEMV pool 40 no-op: %.2f us/call\n", g40_us);

  return 0;
}
