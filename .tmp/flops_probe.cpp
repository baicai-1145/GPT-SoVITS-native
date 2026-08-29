// E13 探针基础设施: 参考编码链各层形状的 GEMM 微基准(amxpp 实测吞吐)
// 形状即报告 §逐层表: C[M,N] = A[M,K]·B[N,K]^T, 与 kern::gemm_f16_amx_pp 同约定。
// 用法: ./flops_probe [reps]
#include "kern/gemv_fmlal.hpp"
#include "kern/gemm_f16_amx.hpp"

#include <Accelerate/Accelerate.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static double now_ms() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

struct Layer {
  const char* name;
  size_t M, N, K;  // sv/hubert conv im2col 形状用名义值(见报告)
};

using namespace gsv;

int main(int argc, char** argv) {
  const int reps = argc > 1 ? atoi(argv[1]) : 30;
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> uf(-1.f, 1.f);

  std::vector<Layer> layers = {
      {"sv_L2out_1024", 1024, 3700, 384},   // k=1x1 convs: out[1024,S3700]
      {"sv_S2out_2048", 2048, 930, 1024},   // stride2 旁路 1x1
      {"sv_S3out_1024x", 1024, 930, 2048},
      {"hub_cnnL1_512", 512, 12788, 1536},  // k3 s2: K=512*3
      {"hub_cnnL2_512", 512, 6393, 1536},
      {"hub_qkv_T399", 1197, 768, 768},     // QKV 三联单批(T*3 行)
      {"hub_ffn_T399_f1", 399, 3072, 768},
      {"hub_ffn_T399_f2", 399, 768, 3072},
      {"hub_proj_T399", 399, 768, 512},
      {"attn_score_K64", 399, 399, 64},   // SDPA QKT 单头批量视图(全局拼接)
      {"attn_pv_K64", 399, 64, 399},      // SDPA P·V 同理
      {"rvq_dist_T199", 199, 1024, 768},  // RVQ 分块化后可化为的 GEMM 形状
  };
  printf("AMX backend: %s\n",
         kern::amx_gemm_available() ? "AVAILABLE" : "UNAVAILABLE");
  printf("%-17s %9s %9s %10s\n", "shape", "amxpp_ms", "sgemm_ms", "GFLOPS_pp");
  for (const auto& L : layers) {
    std::vector<float> A(L.M * L.K), Bm(L.N * L.K), C(L.M * L.N);
    std::vector<uint16_t> Ah(A.size()), Bh(Bm.size());
    for (size_t i = 0; i < A.size(); ++i) {
      A[i] = uf(rng);
      Ah[i] = kern::f32_to_f16_scalar(A[i]);
    }
    for (size_t i = 0; i < Bm.size(); ++i) {
      Bm[i] = uf(rng);
      Bh[i] = kern::f32_to_f16_scalar(Bm[i]);
    }
    // pack
    kern::AmxPanel pa, pb;
    kern::amx_pack_into(Ah.data(), L.M, L.K, pa.buf);
    pa.rows = L.M;
    pa.K = L.K;
    kern::amx_pack_into(Bh.data(), L.N, L.K, pb.buf);
    pb.rows = L.N;
    pb.K = L.K;
    double t_pp = 1e18, t_sg = 1e18;
    for (int r = 0; r < reps; ++r) {
      const double t0 = now_ms();
      kern::gemm_f16_amx_pp(pa, pb, C.data(), L.M, L.N);
      t_pp = std::min(t_pp, now_ms() - t0);
    }
    for (int r = 0; r < reps; ++r) {
      const double t0 = now_ms();
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, int(L.M), int(L.N),
                  int(L.K), 1.0f, A.data(), int(L.K), Bm.data(), int(L.K), 0.0f,
                  C.data(), int(L.N));
      t_sg = std::min(t_sg, now_ms() - t0);
    }
    const double gflops = 2.0 * double(L.M) * L.N * L.K / (t_pp * 1e6);
    printf("%-17s %9.3f %9.3f %10.1f\n", L.name, t_pp, t_sg, gflops);
  }
  return 0;
}
