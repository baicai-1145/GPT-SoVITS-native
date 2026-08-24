// amx_bench.cpp — E5 验收基准: 三方 GEMM 对照
//   (1) Accelerate sgemm (fp32, 内部多线程 — 现网基线)
//   (2) kern::gemm_f16x_fmlal (fp16 FMLAL NEON + P 核池, 现网 fp16 路径)
//   (3) kern::gemm_f16_amx (E5 AMX MATFP, tile 并行)   [需 GSV_AMX_GEMM]
//
// 形状: SoVITS dec conv im2col GEMM (y[Co,S]=W[Co,K]·col[S,K]ᵀ, K=in*k) 与
// encoder 全连接形。T 取 Tq 与 upsample 后真实长度 (v2ProPlus: Tq=2T_lat,
// dec 末端 ×640)。
// 正确性: amx vs fmlal 逐元素全等 (同 fp16 输入同 fp32 累加语义);
//         fmlal vs sgemm 记录量化 relerr (只作参考)。
//
// 用法: ./amx_bench [--reps N]
#include "kern/accel.hpp"
#include "kern/gemv_fmlal.hpp"
#if defined(GSV_AMX_GEMM)
#include "kern/gemm_f16_amx.hpp"
#define HAVE_AMX 1
#else
#define HAVE_AMX 0
#endif

#include <Accelerate/Accelerate.h>
#include <chrono>
#include <cmath>
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
  const char* name;
  size_t M, N, K;  // C[M,N] = A[M,K]·B[N,K]ᵀ
};

}  // namespace

int main(int argc, char** argv) {
  int reps = 5;
  for (int i = 1; i < argc; ++i)
    if (!std::strcmp(argv[i], "--reps") && i + 1 < argc) reps = std::atoi(argv[++i]);

#if HAVE_AMX
  const bool amx_ok = gsv::kern::amx_gemm_available();
  std::printf("AMX backend: %s\n", amx_ok ? "AVAILABLE" : "UNAVAILABLE (fallback)");
#else
  std::printf("AMX backend: compiled OUT (build with -DGSV_AMX_GEMM=ON)\n");
#endif

  // dec (v2ProPlus): conv_pre 768←192 k7 → K=1344; res k3/k7/k11 各通道;
  // upsample 相位 GEMM; conv_post 1←24 k7。T_lat=100 → Tq=200, dec 各级
  // 长度: 200(u0前)→…×10/8/2/2/2→64000 采样, 取各级代表长度。
  std::vector<Shape> shapes = {
      {"dec.conv_pre   Co768 K1344 T200", 768, 200, 1344},
      {"dec.ups0.ph    Co384 K768  T200", 384, 200, 768},
      {"dec.res0.k3    Co384 K1152 T2000", 384, 2000, 1152},
      {"dec.res0.k7    Co384 K2688 T2000", 384, 2000, 2688},
      {"dec.res0.k11   Co384 K4224 T2000", 384, 2000, 4224},
      {"dec.res2.k7    Co96  K672  T8000", 96, 8000, 672},
      {"dec.conv_post  Co1   K168  T12800", 1, 12800, 168},
      {"enc_p.ssl_proj Co768 K512  T300", 768, 300, 512},
      {"enc_p.ffn      Co2048 K768 T300", 2048, 300, 768},
      {"bert.ffn(24L)  Co4096 K1024 T64", 4096, 64, 1024},
  };

  std::mt19937 rng(20260824);
  std::printf("%-30s %8s %8s %8s %8s %7s %7s %7s %s\n", "shape",
              "sgemm", "fmlal", "amx", "amxpp", "sg/pp", "fl/pp", "ax/pp",
              "accuracy");
  const int ROUNDS = 8;  // 交错轮数; 每列取 min (抗热降频/干扰)
  for (const auto& s : shapes) {
    const size_t M = s.M, N = s.N, K = s.K;
    // 本基准用 A·Bᵀ 布局: A=[M,K], B=[N,K] (与 kern 接口同构)
    std::vector<float> Af(M * K), Bf(N * K), Cf(M * N);
    std::vector<uint16_t> Ah(M * K), Bh(N * K);
    for (auto& v : Af) v = (float)((int)(rng() % 4096) - 2048) * 0.001953125f;  // ±4 精确 f16
    for (auto& v : Bf) v = (float)((int)(rng() % 4096) - 2048) * 0.001953125f;
    gsv::kern::f32_to_f16(Af.data(), Ah.data(), Af.size());
    gsv::kern::f32_to_f16(Bf.data(), Bh.data(), Bf.size());
    std::vector<float> Cfl(M * N);
    gsv::kern::gemm_f16x_fmlal(Ah.data(), Bh.data(), Cfl.data(), M, N, K);

#if HAVE_AMX
    std::vector<float> Ca(M * N);
    gsv::kern::gemm_f16_amx(Ah.data(), Bh.data(), Ca.data(), M, N, K);
    gsv::kern::AmxPanel pA = gsv::kern::amx_pack(Ah.data(), M, K);
    gsv::kern::AmxPanel pB = gsv::kern::amx_pack(Bh.data(), N, K);
    std::vector<float> Cpp(M * N);
    gsv::kern::gemm_f16_amx_pp(pA, pB, Cpp.data(), M, N);

    double sg = 1e30, fl = 1e30, ax = 1e30, pp = 1e30, pr = 1e30;
    for (int rd = 0; rd < ROUNDS; ++rd) {
      auto t0 = now_ms();
      for (int r = 0; r < reps; ++r)
        gsv::kern::accel::sgemm('N', 'T', (int)M, (int)N, (int)K, 1.f, Af.data(),
                                (int)K, Bf.data(), (int)K, 0.f, Cf.data(), (int)N);
      auto t1 = now_ms();
      sg = std::min(sg, (t1 - t0) / reps);

      t0 = now_ms();
      for (int r = 0; r < reps; ++r)
        gsv::kern::gemm_f16x_fmlal(Ah.data(), Bh.data(), Cfl.data(), M, N, K);
      t1 = now_ms();
      fl = std::min(fl, (t1 - t0) / reps);

      t0 = now_ms();
      for (int r = 0; r < reps; ++r)
        gsv::kern::gemm_f16_amx(Ah.data(), Bh.data(), Ca.data(), M, N, K);
      t1 = now_ms();
      ax = std::min(ax, (t1 - t0) / reps);

      t0 = now_ms();
      for (int r = 0; r < reps; ++r)
        gsv::kern::gemm_f16_amx_pp(pA, pB, Cpp.data(), M, N);
      t1 = now_ms();
      pp = std::min(pp, (t1 - t0) / reps);

      // 生产稳态: A 预打包 (conv 权重), B 每调用打包 (激活) + 内核
      t0 = now_ms();
      for (int r = 0; r < reps; ++r) {
        pB = gsv::kern::amx_pack(Bh.data(), N, K);
        gsv::kern::gemm_f16_amx_pp(pA, pB, Cpp.data(), M, N);
      }
      t1 = now_ms();
      pr = std::min(pr, (t1 - t0) / reps);
    }
#else
    double sg = 0, fl = 0;
    for (int r = 0; r < reps; ++r)
      gsv::kern::accel::sgemm('N', 'T', (int)M, (int)N, (int)K, 1.f, Af.data(),
                              (int)K, Bf.data(), (int)K, 0.f, Cf.data(), (int)N);
    gsv::kern::gemm_f16x_fmlal(Ah.data(), Bh.data(), Cfl.data(), M, N, K);
    auto t0 = now_ms();
    for (int r = 0; r < reps; ++r)
      gsv::kern::accel::sgemm('N', 'T', (int)M, (int)N, (int)K, 1.f, Af.data(),
                              (int)K, Bf.data(), (int)K, 0.f, Cf.data(), (int)N);
    auto t1 = now_ms();
    sg = (t1 - t0) / reps;
    t0 = now_ms();
    for (int r = 0; r < reps; ++r)
      gsv::kern::gemm_f16x_fmlal(Ah.data(), Bh.data(), Cfl.data(), M, N, K);
    t1 = now_ms();
    fl = (t1 - t0) / reps;
    std::printf("%-30s %8.3f %8.3f %8s %8s\n", s.name, sg, fl, "-", "-");
    continue;
#endif
#if HAVE_AMX
    // 数值: amx vs fmlal 同输入同 fp32 累加语义 → 累加序不同非逐位等。
    // 口径同 G1 (check_b12): cos≥0.9999 且 rel = max|a-b|/max|b| ≤ 1e-3
    double maxabs = 0, maxref = 0, dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < M * N; ++i) {
      const double x = Ca[i], y = Cfl[i];
      dot += x * y; na += x * x; nb += y * y;
      maxabs = std::max(maxabs, std::fabs(x - y));
      maxref = std::max(maxref, std::fabs(y));
    }
    const double cos = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
    const double rel = maxabs / (maxref + 1e-30); (void)0;
    // pp 结果必须与即时路径一致 (同 tiles 同内核)
    double pp_diff = 0;
    for (size_t i = 0; i < M * N; ++i)
      pp_diff = std::max(pp_diff, std::fabs((double)Cpp[i] - Ca[i]));
    const bool ok = cos >= 0.9999 && rel <= 1e-3 && pp_diff == 0;
    std::printf("%-32s %8.3f %8.3f %8.3f %8.3f %6.2fx %6.2fx %6.2fx c=%.6f r=%.0e %s\n",
                s.name, sg, fl, ax, pp, sg / pp, fl / pp, ax / pp, cos, rel,
                ok ? "PASS" : "FAIL");
#else
    std::printf("%-32s %10.3f %10.3f %10s %9s %9s %10s\n", s.name, sg, fl,
                "-", "-", "-", "-");
#endif
    std::fflush(stdout);
  }
  return 0;
}
