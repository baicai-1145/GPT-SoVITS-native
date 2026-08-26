// sdpa_bench.cpp — E11-1 microbench: scalar vs NEON SDPA decode hot loop
//
// Simulates a single block_decode SDPA: H=16 heads, HD=32, len=key cache length
// Measures only the dot product + softmax + value accumulation (the L294-322 area).
// Run: build/sdpa_bench <len>
#include <arm_neon.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static constexpr size_t H = 16;
static constexpr size_t HD = 32;
static constexpr size_t D = H * HD;
static constexpr size_t NL = 24;

// Scalar reference
static void sdpa_scalar(const float* qkv, const float* kf32, const float* vf32,
                        size_t len, float* attn) {
  const float scale = 1.0f / std::sqrt(static_cast<float>(HD));
  std::vector<float> scores(len);
  std::memset(attn, 0, D * sizeof(float));
  for (size_t h = 0; h < H; ++h) {
    const float* qv = qkv + h * HD;
    for (size_t k = 0; k < len; ++k) {
      const float* kv = kf32 + k * D + h * HD;
      float dot = 0.f;
      for (size_t e = 0; e < HD; ++e) dot += qv[e] * kv[e];
      scores[k] = dot * scale;
    }
    // softmax (numerically stable, in-place)
    float mx = scores[0];
    for (size_t k = 1; k < len; ++k) mx = std::max(mx, scores[k]);
    float sum = 0.f;
    for (size_t k = 0; k < len; ++k) {
      scores[k] = std::exp(scores[k] - mx);
      sum += scores[k];
    }
    const float inv = 1.f / sum;
    for (size_t k = 0; k < len; ++k) scores[k] *= inv;
    float* ov = attn + h * HD;
    for (size_t k = 0; k < len; ++k) {
      const float p = scores[k];
      const float* vv = vf32 + k * D + h * HD;
      for (size_t e = 0; e < HD; ++e) ov[e] += p * vv[e];
    }
  }
}

// NEON inline (same as ar_neon:: in t2s_engine.cpp)
namespace neon {
inline float dot_f32(const float* qv, const float* kv, size_t HD) {
  float32x4_t a0 = vdupq_n_f32(0.f);
  float32x4_t a1 = vdupq_n_f32(0.f);
  float32x4_t a2 = vdupq_n_f32(0.f);
  float32x4_t a3 = vdupq_n_f32(0.f);
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
  float32x2_t sum2 = vadd_f32(lo, hi);
  float32x2_t sum1 = vpadd_f32(sum2, sum2);
  return vget_lane_f32(sum1, 0);
}
inline void accum_f32(float* ov, const float* vv, float p, size_t HD) {
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
}

static void sdpa_neon(const float* qkv, const float* kf32, const float* vf32,
                      size_t len, float* attn) {
  const float scale = 1.0f / std::sqrt(static_cast<float>(HD));
  std::vector<float> scores(len);
  std::memset(attn, 0, D * sizeof(float));
  for (size_t h = 0; h < H; ++h) {
    const float* qv = qkv + h * HD;
    for (size_t k = 0; k < len; ++k) {
      float dot = neon::dot_f32(qv, kf32 + k * D + h * HD, HD);
      scores[k] = dot * scale;
    }
    float mx = scores[0];
    for (size_t k = 1; k < len; ++k) mx = std::max(mx, scores[k]);
    float sum = 0.f;
    for (size_t k = 0; k < len; ++k) {
      scores[k] = std::exp(scores[k] - mx);
      sum += scores[k];
    }
    const float inv = 1.f / sum;
    for (size_t k = 0; k < len; ++k) scores[k] *= inv;
    float* ov = attn + h * HD;
    for (size_t k = 0; k < len; ++k) {
      neon::accum_f32(ov, vf32 + k * D + h * HD, scores[k], HD);
    }
  }
}

int main(int argc, char** argv) {
  size_t len = (argc > 1) ? std::atoi(argv[1]) : 256;
  // Allocate: qkv[3D], kv cache[len][2D] (k+v stacked), attn[D]
  std::vector<float> qkv(3 * D);
  std::vector<float> kv(2 * len * D);
  std::vector<float> attn_s(D), attn_n(D);
  // Pseudo-random init (deterministic)
  srand(42);
  for (auto& v : qkv) v = (float(rand()) / RAND_MAX - 0.5f) * 0.1f;
  for (auto& v : kv) v = (float(rand()) / RAND_MAX - 0.5f) * 0.1f;
  const float* kf32 = kv.data();
  const float* vf32 = kv.data() + len * D;

  // Verify bitwise similarity (tolerance for reordering)
  sdpa_scalar(qkv.data(), kf32, vf32, len, attn_s.data());
  sdpa_neon(qkv.data(), kf32, vf32, len, attn_n.data());
  double max_diff = 0;
  for (size_t i = 0; i < D; ++i)
    max_diff = std::max(max_diff, (double)std::fabs(attn_s[i] - attn_n[i]));
  std::printf("[正确性] len=%zu  H=%zu HD=%zu  max|attn_s - attn_n|=%.3e\n",
              len, H, HD, max_diff);

  // Time
  const int N_ITER = 200;
  // warmup
  for (int i = 0; i < 10; ++i) sdpa_scalar(qkv.data(), kf32, vf32, len, attn_s.data());
  for (int i = 0; i < 10; ++i) sdpa_neon(qkv.data(), kf32, vf32, len, attn_n.data());

  // measure scalar
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < N_ITER; ++i) sdpa_scalar(qkv.data(), kf32, vf32, len, attn_s.data());
  auto t1 = std::chrono::steady_clock::now();
  double ms_scalar = std::chrono::duration<double, std::milli>(t1 - t0).count() / N_ITER;

  auto t2 = std::chrono::steady_clock::now();
  for (int i = 0; i < N_ITER; ++i) sdpa_neon(qkv.data(), kf32, vf32, len, attn_n.data());
  auto t3 = std::chrono::steady_clock::now();
  double ms_neon = std::chrono::duration<double, std::milli>(t3 - t2).count() / N_ITER;

  std::printf("[性能] len=%zu  scalar=%.4f ms/call  neon=%.4f ms/call  speedup=%.2fx\n",
              len, ms_scalar, ms_neon, ms_scalar / ms_neon);
  std::printf("[外推] 24 层 × 上述:  scalar=%.3f ms/tok  neon=%.3f ms/tok  (per token)\n",
              ms_scalar * NL, ms_neon * NL);
  return 0;
}
