// test_accel.cpp — A4 单测: Accelerate 封装数值 + QoS 分簇线程池行为
#include "test_util.h"

#include "kern/accel.hpp"
#include "kern/kern.hpp"
#include "runtime/threadpool.hpp"

#include <pthread.h>
#include <sys/qos.h>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <set>
#include <vector>

namespace {

struct Rng {
  uint64_t s = 0x123456789abcdefull;
  float next() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return static_cast<float>(static_cast<int64_t>(s >> 40) / static_cast<double>(1ull << 23)) -
           1.0f;
  }
};

// 与 test_kern 相同格式的自描述二进制加载([u32 rank][u32 dims][data])
struct Bin {
  std::vector<size_t> dims;
  std::vector<float> f32;
  std::vector<uint16_t> f16;
};

Bin load_bin_fwd(const std::string& name, bool is_f16) {
#ifndef GSV_KERN_GOLDEN_DIR
#define GSV_KERN_GOLDEN_DIR "../tests/kern_golden"
#endif
  const std::string path = std::string(GSV_KERN_GOLDEN_DIR) + "/" + name +
                           (is_f16 ? ".f16" : ".f32");
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) std::exit(1);
  Bin b;
  uint32_t rank = 0;
  if (std::fread(&rank, 4, 1, f) != 1) std::exit(1);
  b.dims.resize(rank);
  if (rank) {
    std::vector<uint32_t> d32(rank);
    if (std::fread(d32.data(), 4, rank, f) != rank) std::exit(1);
    for (size_t i = 0; i < rank; ++i) b.dims[i] = d32[i];
  }
  size_t n = 1;
  for (size_t d : b.dims) n *= d;
  if (is_f16) {
    b.f16.resize(n);
    if (n && std::fread(b.f16.data(), 2, n, f) != n) std::exit(1);
  } else {
    b.f32.resize(n);
    if (n && std::fread(b.f32.data(), 4, n, f) != n) std::exit(1);
  }
  std::fclose(f);
  return b;
}

}  // namespace

GSV_TEST(f16_to_f32_special_values_exact) {
  // 特殊位型必须无损升位
  const uint16_t bits[] = {0x0000, 0x8000, 0x3c00, 0xc000, 0x7bff, 0x0400, 0x3555, 0xabcd};
  std::vector<float> got(std::size(bits));
  gsv::kern::accel::f16_to_f32(bits, got.data(), std::size(bits));
  for (size_t i = 0; i < std::size(bits); ++i) {
    __fp16 h;
    __builtin_memcpy(&h, &bits[i], 2);
    CHECK_EQ(got[i], static_cast<float>(h));  // 与编译器语义一致即正确
  }
  CHECK_EQ(got[0], 0.0f);
  CHECK_EQ(got[2], 1.0f);
  CHECK_EQ(got[3], -2.0f);
  CHECK_EQ(got[4], 65504.0f);
}

GSV_TEST(sgemm_matches_naive_loop) {
  Rng rng;
  const int M = 5, N = 7, K = 3;  // 小且奇形
  std::vector<float> A(M * K), B(K * N), C(M * N), Cref(M * N);
  for (auto& v : A) v = rng.next();
  for (auto& v : B) v = rng.next();

  // 'N','N': C[M,N] = A[M,K]·B[K,N], α=1 β=0 (C 预先放 NaN → 必须被忽略)
  // 数学定义直接对存储写, 不引入转置歧义
  for (auto& v : C) v = std::nanf("");
  gsv::kern::accel::sgemm('N', 'N', M, N, K, 1.f, A.data(), K, B.data(), N, 0.f, C.data(), N);
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      double acc = 0;
      for (int k = 0; k < K; ++k) acc += A[i * K + k] * B[k * N + j];
      Cref[i * N + j] = static_cast<float>(acc);
    }
  for (int i = 0; i < M * N; ++i) CHECK_NEAR(C[i], Cref[i], 1e-5);

  // 'T','T' α=2 β=1: P=传入的[K,M]存储, Q=传入的[N,K]存储 →
  //   C[i,j] = 2·Σ_k P[k,i]·Q[j,k] + 1·C[i,j]
  std::vector<float> P(K * M), Q(N * K);
  for (auto& v : P) v = rng.next();
  for (auto& v : Q) v = rng.next();
  for (int i = 0; i < M * N; ++i) C[i] = 0.5f;
  gsv::kern::accel::sgemm('T', 'T', M, N, K, 2.f, P.data(), M, Q.data(), K, 1.f, C.data(), N);
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      double acc = 0;
      for (int k = 0; k < K; ++k) acc += P[k * M + i] * Q[j * K + k];
      Cref[i * N + j] = static_cast<float>(2.0 * acc + 0.5);
    }
  for (int i = 0; i < M * N; ++i) CHECK_NEAR(C[i], Cref[i], 1e-5);
}

GSV_TEST(gemm_nt_f16w_and_dense_match_gemv_rows) {
  // 用 kern golden 的 fp16 权重 [128,96]; x 取 17 行随机
  const Bin w = load_bin_fwd("gemv_a_w", true);
  const size_t out_ = w.dims[0], in_ = w.dims[1];
  Rng rng;
  const size_t T = 17;
  std::vector<float> x(T * in_);
  for (auto& v : x) v = rng.next();
  std::vector<float> y(T * out_), yrow(out_);
  gsv::kern::accel::gemm_nt_f16w(x.data(), T, in_, w.f16.data(), out_, y.data());
  gsv::kern::accel::DenseF16 dense(w.f16.data(), out_, in_);
  std::vector<float> yd(T * out_);
  dense.forward(x.data(), T, yd.data());

  for (size_t t = 0; t < T; ++t) {
    gsv::kern::gemv_f16w_f32acc(w.f16.data(), x.data() + t * in_, yrow.data(), out_, in_);
    float peak = 0;
    for (size_t o = 0; o < out_; ++o) peak = std::max(peak, std::fabs(yrow[o]));
    for (size_t o = 0; o < out_; ++o) {
      CHECK_NEAR(y[t * out_ + o], yrow[o], 1e-5 * peak);
      CHECK_NEAR(yd[t * out_ + o], yrow[o], 1e-5 * peak);
    }
  }
}

GSV_TEST(parallel_for_correct_and_qos_bound) {
  constexpr size_t N = 1000;
  std::vector<long long> v(N, -1);
  for (gsv::rt::Qos qos : {gsv::rt::Qos::UserInitiated, gsv::rt::Qos::Utility}) {
    gsv::rt::parallel_for(N, 64,
                          [&](size_t b, size_t e) {
                            for (size_t i = b; i < e; ++i) v[i] = static_cast<long long>(i) * 3 + 7;
                          },
                          qos);
    bool ok = true;
    for (size_t i = 0; i < N; ++i)
      if (v[i] != static_cast<long long>(i) * 3 + 7) ok = false;
    CHECK(ok);

    // QoS 绑簇验证: worker 内自报类别, 观察到的非主线程类别必须是请求的那一个
    std::atomic<int> seen_user{0}, seen_util{0};
    gsv::rt::parallel_for(N, 8,
                          [&](size_t b, size_t e) {
                            (void)b;
                            (void)e;
                            qos_class_t cl = QOS_CLASS_UNSPECIFIED;
                            int rel = 0;
                            if (::pthread_get_qos_class_np(::pthread_self(), &cl, &rel) == 0) {
                              if (cl == QOS_CLASS_USER_INITIATED) ++seen_user;
                              if (cl == QOS_CLASS_UTILITY) ++seen_util;
                            }
                          },
                          qos);
    const int u = seen_user.load(), t = seen_util.load();
    if (qos == gsv::rt::Qos::UserInitiated) {
      CHECK_MSG(u > 0 && t == 0, "UserInitiated 任务未绑 P 核 QoS");
    } else {
      CHECK_MSG(t > 0 && u == 0, "Utility 任务未绑 E 核 QoS");
    }
  }

  // 内联路径(n<=grain)与零长
  gsv::rt::parallel_for(10, 64, [&](size_t b, size_t e) {
    for (size_t i = b; i < e; ++i) v[i] = 42;
  }, gsv::rt::Qos::UserInitiated);
  CHECK(v[5] == 42);
  gsv::rt::parallel_for(0, 8, [](size_t, size_t) { CHECK(false); }, gsv::rt::Qos::Utility);
  CHECK(gsv::rt::p_core_count() > 0);
}

GSV_TEST_MAIN()
