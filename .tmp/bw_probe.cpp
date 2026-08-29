// E13 探针基础设施: 流式内存带宽微基准(只读扫描)
// 用法: ./bw_probe [线程数]  — 输出各缓冲规模下的单线程/多线程 GB/s
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

static double now_ms() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

static double scan_bw(const std::vector<float>& buf, int reps) {
  const size_t n = buf.size();
  const float* p = buf.data();
  float sink = 0.f;
  for (int r = 0; r < 2; ++r)  // warm
    for (size_t i = 0; i < n; ++i) sink += p[i];
  double best = 1e18;
  for (int r = 0; r < reps; ++r) {
    const double t0 = now_ms();
    float s = 0.f;
    for (size_t i = 0; i < n; ++i) s += p[i];
    const double dt = now_ms() - t0;
    if (s == 1.2345f) printf(" ");  // 防 DCE
    if (dt < best) best = dt;
  }
  return (double(n) * 4.0 / (best * 1e-3)) / 1e9;  // GB/s
}

int main(int argc, char** argv) {
  const int threads = argc > 1 ? atoi(argv[1]) : 1;
  for (size_t mb : {16, 32, 64, 128, 256}) {
    const size_t n = mb * 1024 * 1024 / 4;
    std::vector<float> buf(n);
    for (size_t i = 0; i < n; ++i) buf[i] = float(i & 255);
    if (threads <= 1) {
      printf("buf=%zuMB threads=1  bw=%.2f GB/s\n", mb,
             scan_bw(buf, /*reps*/ 5));
    } else {
      std::vector<std::thread> ts;
      std::vector<double> out(threads, 0.);
      const size_t chunk = n / threads;
      for (int t = 0; t < threads; ++t)
        ts.emplace_back([&, t] {
          std::vector<float> part(buf.begin() + ptrdiff_t(t) * ptrdiff_t(chunk),
                                  buf.begin() + ptrdiff_t(t + 1) * ptrdiff_t(chunk));
          out[t] = scan_bw(part, 3);
        });
      for (auto& th : ts) th.join();
      double tot = 0;
      for (double v : out) tot += v;
      printf("buf=%zuMB threads=%d bw_total=%.2f GB/s\n", mb, threads, tot);
    }
  }
  return 0;
}
