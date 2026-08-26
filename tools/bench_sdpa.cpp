// bench_sdpa.cpp — E11-5: 模拟 prefill SDPA 单头 attention, 比较 NEON 循环 vs AMX GEMM
#include "kern/gemm_f16_amx.hpp"
#include <arm_neon.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

// NEON 4-lane tree-reduce dot (同 E11-1 ar_neon::dot_f32)
static float dot_f32_neon(const float* qv, const float* kv, size_t HD) {
  float32x4_t a0 = vdupq_n_f32(0.f), a1 = vdupq_n_f32(0.f);
  float32x4_t a2 = vdupq_n_f32(0.f), a3 = vdupq_n_f32(0.f);
  size_t e = 0;
  for (; e + 16 <= HD; e += 16) {
    a0 = vfmaq_f32(a0, vld1q_f32(qv + e + 0),  vld1q_f32(kv + e + 0));
    a1 = vfmaq_f32(a1, vld1q_f32(qv + e + 4),  vld1q_f32(kv + e + 4));
    a2 = vfmaq_f32(a2, vld1q_f32(qv + e + 8),  vld1q_f32(kv + e + 8));
    a3 = vfmaq_f32(a3, vld1q_f32(qv + e + 12), vld1q_f32(kv + e + 12));
  }
  float32x4_t s01 = vaddq_f32(a0, a1);
  float32x4_t s23 = vaddq_f32(a2, a3);
  float32x4_t s = vaddq_f32(s01, s23);
  float32x2_t lo = vget_low_f32(s);
  float32x2_t hi = vget_high_f32(s);
  return vget_lane_f32(vpadd_f32(vadd_f32(lo, hi), vadd_f32(lo, hi)), 0);
}

static void accum_f32_neon(float* ov, const float* vv, float p, size_t HD) {
  float32x4_t p4 = vdupq_n_f32(p);
  size_t e = 0;
  for (; e + 16 <= HD; e += 16) {
    float32x4_t a0 = vmulq_f32(p4, vld1q_f32(vv + e + 0));
    float32x4_t a1 = vmulq_f32(p4, vld1q_f32(vv + e + 4));
    float32x4_t a2 = vmulq_f32(p4, vld1q_f32(vv + e + 8));
    float32x4_t a3 = vmulq_f32(p4, vld1q_f32(vv + e + 12));
    vst1q_f32(ov + e + 0,  vaddq_f32(vld1q_f32(ov + e + 0),  a0));
    vst1q_f32(ov + e + 4,  vaddq_f32(vld1q_f32(ov + e + 4),  a1));
    vst1q_f32(ov + e + 8,  vaddq_f32(vld1q_f32(ov + e + 8),  a2));
    vst1q_f32(ov + e + 12, vaddq_f32(vld1q_f32(ov + e + 12), a3));
  }
}

// Q·K^T 模拟: Q [S, HD], K [S, HD] → scores [S, S]
// NEON: per query row, per key col, dot
static void sdpa_neon(const float* Q, const float* K, const float* V,
                      size_t S, size_t HD, float* scores, float* out) {
  const float scale = 1.0f / std::sqrt(float(HD));
  for (size_t q = 0; q < S; ++q) {
    for (size_t k = 0; k < S; ++k) {
      scores[k] = dot_f32_neon(Q + q*HD, K + k*HD, HD) * scale;
    }
    // softmax
    float mx = scores[0];
    for (size_t k = 1; k < S; ++k) mx = std::max(mx, scores[k]);
    float sum = 0;
    for (size_t k = 0; k < S; ++k) { scores[k] = std::exp(scores[k] - mx); sum += scores[k]; }
    const float inv = 1.f / sum;
    for (size_t k = 0; k < S; ++k) scores[k] *= inv;
    // out[q] += scores[k] * V[k]
    std::memset(out + q*HD, 0, HD * sizeof(float));
    for (size_t k = 0; k < S; ++k) {
      accum_f32_neon(out + q*HD, V + k*HD, scores[k], HD);
    }
  }
}

// AMX version: Q·K^T 全矩阵 + softmax + P·V 全矩阵
// Q 是 S×HD, K 是 S×HD
// scores = Q · K^T 是 S×S  (M=S, N=S, K=HD)
// AMX panel: pa = Q 形状 [S,HD], pb = K 形状 [S,HD], C = scores [S,S]
// 注意: gemm_f16_amx_pp 语义 C[M,N] = pa·pb^T, 即 pa=[M,K], pb=[N,K]
//   pa = Q (M=S, K=HD), pb = K (N=S, K=HD)
static void sdpa_amx(const float* Q, const float* K, const float* V,
                     size_t S, size_t HD, float* scores, float* out) {
  const float scale = 1.0f / std::sqrt(float(HD));
  // 1) Q·K^T  via AMX
  std::vector<uint16_t> Q_f16(S * HD), K_f16(S * HD);
  for (size_t i = 0; i < S * HD; ++i) {
    __fp16 hq = static_cast<__fp16>(Q[i]);
    __fp16 hk = static_cast<__fp16>(K[i]);
    __builtin_memcpy(&Q_f16[i], &hq, 2);
    __builtin_memcpy(&K_f16[i], &hk, 2);
  }
  auto pa = gsv::kern::amx_pack(Q_f16.data(), S, HD);
  auto pb = gsv::kern::amx_pack(K_f16.data(), S, HD);
  gsv::kern::gemm_f16_amx_pp(pa, pb, scores, S, S);
  // scale
  for (size_t i = 0; i < S * S; ++i) scores[i] *= scale;
  // 2) softmax (per row)
  for (size_t q = 0; q < S; ++q) {
    float* row = scores + q * S;
    float mx = row[0];
    for (size_t k = 1; k < S; ++k) mx = std::max(mx, row[k]);
    float sum = 0;
    for (size_t k = 0; k < S; ++k) { row[k] = std::exp(row[k] - mx); sum += row[k]; }
    const float inv = 1.f / sum;
    for (size_t k = 0; k < S; ++k) row[k] *= inv;
  }
  // 3) P·V via AMX: out = P · V (M=S, N=HD, K=S)
  std::vector<uint16_t> P_f16(S * S);
  for (size_t i = 0; i < S * S; ++i) {
    __fp16 h = static_cast<__fp16>(scores[i]);
    __builtin_memcpy(&P_f16[i], &h, 2);
  }
  std::vector<uint16_t> V_f16(S * HD);
  for (size_t i = 0; i < S * HD; ++i) {
    __fp16 h = static_cast<__fp16>(V[i]);
    __builtin_memcpy(&V_f16[i], &h, 2);
  }
  auto pa2 = gsv::kern::amx_pack(P_f16.data(), S, S);
  auto pb2 = gsv::kern::amx_pack(V_f16.data(), S, HD);
  gsv::kern::gemm_f16_amx_pp(pa2, pb2, out, S, HD);
}

int main(int argc, char** argv) {
  size_t S = (argc > 1) ? std::atoi(argv[1]) : 200;
  size_t HD = 32;
  std::vector<float> Q(S * HD), K(S * HD), V(S * HD);
  std::vector<float> scores(S * S), out(S * HD);
  srand(42);
  for (auto& v : Q) v = (float(rand()) / RAND_MAX - 0.5f) * 0.1f;
  for (auto& v : K) v = (float(rand()) / RAND_MAX - 0.5f) * 0.1f;
  for (auto& v : V) v = (float(rand()) / RAND_MAX - 0.5f) * 0.1f;

  // Warmup
  for (int i = 0; i < 5; ++i) sdpa_neon(Q.data(), K.data(), V.data(), S, HD, scores.data(), out.data());
  for (int i = 0; i < 5; ++i) sdpa_amx(Q.data(), K.data(), V.data(), S, HD, scores.data(), out.data());

  // Verify similarity
  std::vector<float> scores2(S * S), out2(S * HD);
  std::vector<float> scores_neon_pre(S * S);
  for (size_t q = 0; q < S; ++q) {
    for (size_t k = 0; k < S; ++k) {
      scores_neon_pre[q*S+k] = dot_f32_neon(Q.data() + q*HD, K.data() + k*HD, HD) * (1.0f / std::sqrt(float(HD)));
    }
  }
  std::vector<float> scores_amx_pre(S * S);
  // Capture AMX pre-softmax by running AMX Q·K^T and scaling
  {
    std::vector<uint16_t> Q_f16(S * HD), K_f16(S * HD);
    for (size_t i = 0; i < S * HD; ++i) {
      __fp16 hq = static_cast<__fp16>(Q[i]);
      __fp16 hk = static_cast<__fp16>(K[i]);
      __builtin_memcpy(&Q_f16[i], &hq, 2);
      __builtin_memcpy(&K_f16[i], &hk, 2);
    }
    auto pa = gsv::kern::amx_pack(Q_f16.data(), S, HD);
    auto pb = gsv::kern::amx_pack(K_f16.data(), S, HD);
    gsv::kern::gemm_f16_amx_pp(pa, pb, scores_amx_pre.data(), S, S);
    const float scale = 1.0f / std::sqrt(float(HD));
    for (size_t i = 0; i < S * S; ++i) scores_amx_pre[i] *= scale;
  }
  // Pre-softmax cos
  double dp = 0, na = 0, nb = 0;
  for (size_t i = 0; i < S * S; ++i) {
    dp += scores_amx_pre[i] * scores_neon_pre[i];
    na += scores_amx_pre[i] * scores_amx_pre[i];
    nb += scores_neon_pre[i] * scores_neon_pre[i];
  }
  double cos_pre = dp / std::sqrt(na * nb);
  std::printf("S=%zu  cos_pre_softmax=%.10f\n", S, cos_pre);

  sdpa_neon(Q.data(), K.data(), V.data(), S, HD, scores2.data(), out2.data());
  sdpa_amx(Q.data(), K.data(), V.data(), S, HD, scores.data(), out.data());
  double max_d_out = 0, max_d_prob = 0;
  for (size_t i = 0; i < S * HD; ++i)
    max_d_out = std::max(max_d_out, (double)std::fabs(out[i] - out2[i]));
  for (size_t i = 0; i < S * S; ++i)
    max_d_prob = std::max(max_d_prob, (double)std::fabs(scores[i] - scores2[i]));
  // cos-sim of post-softmax
  dp = 0; na = 0; nb = 0;
  for (size_t i = 0; i < S * S; ++i) {
    dp += scores[i] * scores2[i];
    na += scores[i] * scores[i];
    nb += scores2[i] * scores2[i];
  }
  double cos = dp / std::sqrt(na * nb);
  std::printf("S=%zu  max_d_out=%.3e  max_d_prob=%.3e  cos_post=%.10f\n",
              S, max_d_out, max_d_prob, cos);

  const int N = 100;
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < N; ++i) sdpa_neon(Q.data(), K.data(), V.data(), S, HD, scores.data(), out.data());
  auto t1 = std::chrono::steady_clock::now();
  double ms_n = std::chrono::duration<double, std::milli>(t1 - t0).count() / N;

  t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < N; ++i) sdpa_amx(Q.data(), K.data(), V.data(), S, HD, scores.data(), out.data());
  t1 = std::chrono::steady_clock::now();
  double ms_a = std::chrono::duration<double, std::milli>(t1 - t0).count() / N;

  std::printf("S=%zu  NEON=%.3f ms  AMX=%.3f ms  speedup=%.2fx\n", S, ms_n, ms_a, ms_n / ms_a);
  return 0;
}
