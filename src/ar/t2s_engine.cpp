// t2s_engine.cpp — AR/T2S 引擎实现 (B1 prefill + B2 decode)
//
// 数值对照要点(均与 torch 同构, 见头文件注释):
//   - 大矩阵乘 prefill 走 accel::sgemm(DenseF16 加载时升位缓存);
//     decode 单 token 走 kern::gemv_f16w_f32acc(fp16 权重 NEON 升位 + fp32 FMA 累加)
//   - LayerNorm 有偏方差 eps=1e-5, 统计量 fp32 (kern::layernorm)
//   - 注意力 scale = 1/sqrt(head_dim); softmax 行内稳定 fp32
//   - 无 RoPE(T2S 用正弦位置编码加在输入上, 与 kern 的 rope 无关)
#include "ar/t2s_engine.hpp"

#include "kern/gemv_fmlal.hpp"
#include "runtime/threadpool.hpp"

#include <arm_neon.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <pthread.h>
#include <stdexcept>
#include <string>
#include <sys/qos.h>
#include <sys/resource.h>
#include <thread>
#include <vector>

// ---- E11-1: SDPA NEON helpers (decode + prefill 共享) ----
// 数值纪律:
//   fp32 路径: 4-lane 独立累加 → 树形归约 (a0+a1)+(a2+a3) → 水平求和
//   (等价于 8 个 vdot 等价 FMA 但顺序可比, 保留 G1/G2 数值容差)
//   fp16 KV 路径: vld1q_f16×2 升位 fp32 后 FMA, 避免标量升位瓶颈
// 性能纪律: HD=32 (AR 默认 512/16) 时全部走 4-lane×8vec 主路径, 无尾循环
namespace {
namespace ar_neon {

// fp32 dot: qv[0..HD) · kv[0..HD), 4-lane tree reduce
inline float dot_f32(const float* qv, const float* kv, size_t HD) {
  float32x4_t a0 = vdupq_n_f32(0.f);
  float32x4_t a1 = vdupq_n_f32(0.f);
  float32x4_t a2 = vdupq_n_f32(0.f);
  float32x4_t a3 = vdupq_n_f32(0.f);
  size_t e = 0;
  // 主路径: 16 元素一组 (4 vec × 4 lanes) — AR 默认 HD=32 时一轮吃完
  for (; e + 16 <= HD; e += 16) {
    a0 = vfmaq_f32(a0, vld1q_f32(qv + e + 0),  vld1q_f32(kv + e + 0));
    a1 = vfmaq_f32(a1, vld1q_f32(qv + e + 4),  vld1q_f32(kv + e + 4));
    a2 = vfmaq_f32(a2, vld1q_f32(qv + e + 8),  vld1q_f32(kv + e + 8));
    a3 = vfmaq_f32(a3, vld1q_f32(qv + e + 12), vld1q_f32(kv + e + 12));
  }
  // 4-lane 树形归约: (a0+a1)+(a2+a3)
  float32x4_t s01 = vaddq_f32(a0, a1);
  float32x4_t s23 = vaddq_f32(a2, a3);
  float32x4_t s = vaddq_f32(s01, s23);
  // 水平求和: pairwise add
  float32x2_t lo = vget_low_f32(s);
  float32x2_t hi = vget_high_f32(s);
  float32x2_t sum2 = vadd_f32(lo, hi);
  float32x2_t sum1 = vpadd_f32(sum2, sum2);
  float dot = vget_lane_f32(sum1, 0);
  // 尾元素 (HD 非 16 倍数时; AR 实际 HD=32/64 走不到这里)
  for (; e < HD; ++e) dot += qv[e] * kv[e];
  return dot;
}

// fp16 KV dot: qv[fp32] · k16[fp16] 升位累加
inline float dot_f16kv(const float* qv, const uint16_t* k16, size_t HD) {
  float32x4_t a0 = vdupq_n_f32(0.f);
  float32x4_t a1 = vdupq_n_f32(0.f);
  float32x4_t a2 = vdupq_n_f32(0.f);
  float32x4_t a3 = vdupq_n_f32(0.f);
  size_t e = 0;
  for (; e + 16 <= HD; e += 16) {
    // 每 8 个 fp16 装一 NEON half 向量; 一组 16 fp16 = 2 个 half vec
    float16x8_t h0 = vld1q_f16(reinterpret_cast<const __fp16*>(k16 + e));
    float16x8_t h1 = vld1q_f16(reinterpret_cast<const __fp16*>(k16 + e + 8));
    a0 = vfmaq_f32(a0, vld1q_f32(qv + e + 0),
                   vcvt_f32_f16(vget_low_f16(h0)));
    a1 = vfmaq_f32(a1, vld1q_f32(qv + e + 4),
                   vcvt_f32_f16(vget_high_f16(h0)));
    a2 = vfmaq_f32(a2, vld1q_f32(qv + e + 8),
                   vcvt_f32_f16(vget_low_f16(h1)));
    a3 = vfmaq_f32(a3, vld1q_f32(qv + e + 12),
                   vcvt_f32_f16(vget_high_f16(h1)));
  }
  float32x4_t s01 = vaddq_f32(a0, a1);
  float32x4_t s23 = vaddq_f32(a2, a3);
  float32x4_t s = vaddq_f32(s01, s23);
  float32x2_t lo = vget_low_f32(s);
  float32x2_t hi = vget_high_f32(s);
  float32x2_t sum2 = vadd_f32(lo, hi);
  float32x2_t sum1 = vpadd_f32(sum2, sum2);
  float dot = vget_lane_f32(sum1, 0);
  for (; e < HD; ++e) {
    __fp16 h;
    __builtin_memcpy(&h, k16 + e, 2);
    dot += qv[e] * static_cast<float>(h);
  }
  return dot;
}

// ov[0..HD) += p * vv[0..HD)  (fp32)
inline void accum_f32(float* ov, const float* vv, float p, size_t HD) {
  float32x4_t p4 = vdupq_n_f32(p);
  // 4-lane 独立累加 → 加到 ov
  // 注意: vv 是只读, ov 是读写; 4-lane 分组保持 4×4 网格, 末尾归并到 ov
  // 简化: 16 元素一组, 累加后立即归并 (HD=32 时仅一轮)
  size_t e = 0;
  for (; e + 16 <= HD; e += 16) {
    float32x4_t a0 = vmulq_f32(p4, vld1q_f32(vv + e + 0));
    float32x4_t a1 = vmulq_f32(p4, vld1q_f32(vv + e + 4));
    float32x4_t a2 = vmulq_f32(p4, vld1q_f32(vv + e + 8));
    float32x4_t a3 = vmulq_f32(p4, vld1q_f32(vv + e + 12));
    // ov 加载 + 累加
    vst1q_f32(ov + e + 0,  vaddq_f32(vld1q_f32(ov + e + 0),  a0));
    vst1q_f32(ov + e + 4,  vaddq_f32(vld1q_f32(ov + e + 4),  a1));
    vst1q_f32(ov + e + 8,  vaddq_f32(vld1q_f32(ov + e + 8),  a2));
    vst1q_f32(ov + e + 12, vaddq_f32(vld1q_f32(ov + e + 12), a3));
  }
  for (; e < HD; ++e) ov[e] += p * vv[e];
}

// ov[0..HD) += p * vv16[0..HD)  (fp16 KV)
inline void accum_f16kv(float* ov, const uint16_t* v16, float p, size_t HD) {
  float32x4_t p4 = vdupq_n_f32(p);
  size_t e = 0;
  for (; e + 16 <= HD; e += 16) {
    float16x8_t h0 = vld1q_f16(reinterpret_cast<const __fp16*>(v16 + e));
    float16x8_t h1 = vld1q_f16(reinterpret_cast<const __fp16*>(v16 + e + 8));
    float32x4_t a0 = vmulq_f32(p4, vcvt_f32_f16(vget_low_f16(h0)));
    float32x4_t a1 = vmulq_f32(p4, vcvt_f32_f16(vget_high_f16(h0)));
    float32x4_t a2 = vmulq_f32(p4, vcvt_f32_f16(vget_low_f16(h1)));
    float32x4_t a3 = vmulq_f32(p4, vcvt_f32_f16(vget_high_f16(h1)));
    vst1q_f32(ov + e + 0,  vaddq_f32(vld1q_f32(ov + e + 0),  a0));
    vst1q_f32(ov + e + 4,  vaddq_f32(vld1q_f32(ov + e + 4),  a1));
    vst1q_f32(ov + e + 8,  vaddq_f32(vld1q_f32(ov + e + 8),  a2));
    vst1q_f32(ov + e + 12, vaddq_f32(vld1q_f32(ov + e + 12), a3));
  }
  for (; e < HD; ++e) {
    __fp16 h;
    __builtin_memcpy(&h, v16 + e, 2);
    ov[e] += p * static_cast<float>(h);
  }
}

// ---------------------------------------------------------------------------
// E18: Split-K GEMV Kernels & Barrier Persistent Pool
// ---------------------------------------------------------------------------
inline void gemv_slice_f16w_f32acc_4rows(const uint16_t* w, const float* x,
                                         float* y_part, size_t out, size_t in,
                                         size_t k_start, size_t k_end) {
  const size_t k_chunk = k_end - k_start;
  const size_t vec_end = k_chunk & ~size_t{7};
  size_t r = 0;
  for (; r + 4 <= out; r += 4) {
    const uint16_t* wr0 = w + r * in + k_start;
    const uint16_t* wr1 = w + (r + 1) * in + k_start;
    const uint16_t* wr2 = w + (r + 2) * in + k_start;
    const uint16_t* wr3 = w + (r + 3) * in + k_start;
    const float* xr = x + k_start;
    float32x4_t acc00 = vdupq_n_f32(0.0f), acc01 = vdupq_n_f32(0.0f);
    float32x4_t acc10 = vdupq_n_f32(0.0f), acc11 = vdupq_n_f32(0.0f);
    float32x4_t acc20 = vdupq_n_f32(0.0f), acc21 = vdupq_n_f32(0.0f);
    float32x4_t acc30 = vdupq_n_f32(0.0f), acc31 = vdupq_n_f32(0.0f);
    size_t k = 0;
    for (; k < vec_end; k += 8, wr0 += 8, wr1 += 8, wr2 += 8, wr3 += 8, xr += 8) {
      const float32x4_t x0 = vld1q_f32(xr);
      const float32x4_t x1 = vld1q_f32(xr + 4);

      const float16x8_t w0 = vld1q_f16(reinterpret_cast<const __fp16*>(wr0));
      acc00 = vfmaq_f32(acc00, vcvt_f32_f16(vget_low_f16(w0)), x0);
      acc01 = vfmaq_f32(acc01, vcvt_f32_f16(vget_high_f16(w0)), x1);

      const float16x8_t w1 = vld1q_f16(reinterpret_cast<const __fp16*>(wr1));
      acc10 = vfmaq_f32(acc10, vcvt_f32_f16(vget_low_f16(w1)), x0);
      acc11 = vfmaq_f32(acc11, vcvt_f32_f16(vget_high_f16(w1)), x1);

      const float16x8_t w2 = vld1q_f16(reinterpret_cast<const __fp16*>(wr2));
      acc20 = vfmaq_f32(acc20, vcvt_f32_f16(vget_low_f16(w2)), x0);
      acc21 = vfmaq_f32(acc21, vcvt_f32_f16(vget_high_f16(w2)), x1);

      const float16x8_t w3 = vld1q_f16(reinterpret_cast<const __fp16*>(wr3));
      acc30 = vfmaq_f32(acc30, vcvt_f32_f16(vget_low_f16(w3)), x0);
      acc31 = vfmaq_f32(acc31, vcvt_f32_f16(vget_high_f16(w3)), x1);
    }
    float s0 = vaddvq_f32(vaddq_f32(acc00, acc01));
    float s1 = vaddvq_f32(vaddq_f32(acc10, acc11));
    float s2 = vaddvq_f32(vaddq_f32(acc20, acc21));
    float s3 = vaddvq_f32(vaddq_f32(acc30, acc31));
    for (; k < k_chunk; ++k, ++wr0, ++wr1, ++wr2, ++wr3, ++xr) {
      __fp16 h0, h1, h2, h3;
      std::memcpy(&h0, wr0, sizeof h0);
      std::memcpy(&h1, wr1, sizeof h1);
      std::memcpy(&h2, wr2, sizeof h2);
      std::memcpy(&h3, wr3, sizeof h3);
      s0 += static_cast<float>(h0) * (*xr);
      s1 += static_cast<float>(h1) * (*xr);
      s2 += static_cast<float>(h2) * (*xr);
      s3 += static_cast<float>(h3) * (*xr);
    }
    y_part[r] = s0;
    y_part[r + 1] = s1;
    y_part[r + 2] = s2;
    y_part[r + 3] = s3;
  }
  for (; r < out; ++r) {
    const uint16_t* wr = w + r * in + k_start;
    const float* xr = x + k_start;
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    size_t k = 0;
    for (; k < vec_end; k += 8, wr += 8, xr += 8) {
      const float16x8_t w8 = vld1q_f16(reinterpret_cast<const __fp16*>(wr));
      acc0 = vfmaq_f32(acc0, vcvt_f32_f16(vget_low_f16(w8)), vld1q_f32(xr));
      acc1 = vfmaq_f32(acc1, vcvt_f32_f16(vget_high_f16(w8)), vld1q_f32(xr + 4));
    }
    float s = vaddvq_f32(acc0) + vaddvq_f32(acc1);
    for (; k < k_chunk; ++k, ++wr, ++xr) {
      __fp16 h;
      std::memcpy(&h, wr, sizeof h);
      s += static_cast<float>(h) * (*xr);
    }
    y_part[r] = s;
  }
}

inline void gemv_slice_f16x_fmlal_4rows(const uint16_t* w, const uint16_t* xh,
                                        float* y_part, size_t out, size_t in,
                                        size_t k_start, size_t k_end) {
  const size_t k_chunk = k_end - k_start;
  const size_t vec_end = k_chunk & ~size_t{15};
  size_t r = 0;
  for (; r + 4 <= out; r += 4) {
    const uint16_t* wr0 = w + r * in + k_start;
    const uint16_t* wr1 = w + (r + 1) * in + k_start;
    const uint16_t* wr2 = w + (r + 2) * in + k_start;
    const uint16_t* wr3 = w + (r + 3) * in + k_start;
    const uint16_t* xr = xh + k_start;
    float32x4_t acc00 = vdupq_n_f32(0.0f), acc01 = vdupq_n_f32(0.0f);
    float32x4_t acc10 = vdupq_n_f32(0.0f), acc11 = vdupq_n_f32(0.0f);
    float32x4_t acc20 = vdupq_n_f32(0.0f), acc21 = vdupq_n_f32(0.0f);
    float32x4_t acc30 = vdupq_n_f32(0.0f), acc31 = vdupq_n_f32(0.0f);
    size_t k = 0;
    for (; k < vec_end; k += 16, wr0 += 16, wr1 += 16, wr2 += 16, wr3 += 16, xr += 16) {
      const float16x8_t x0 = vld1q_f16(reinterpret_cast<const __fp16*>(xr));
      const float16x8_t x1 = vld1q_f16(reinterpret_cast<const __fp16*>(xr + 8));

      const float16x8_t w0_0 = vld1q_f16(reinterpret_cast<const __fp16*>(wr0));
      const float16x8_t w0_1 = vld1q_f16(reinterpret_cast<const __fp16*>(wr0 + 8));
      acc00 = vfmlalq_low_f16(acc00, w0_0, x0);
      acc00 = vfmlalq_high_f16(acc00, w0_0, x0);
      acc01 = vfmlalq_low_f16(acc01, w0_1, x1);
      acc01 = vfmlalq_high_f16(acc01, w0_1, x1);

      const float16x8_t w1_0 = vld1q_f16(reinterpret_cast<const __fp16*>(wr1));
      const float16x8_t w1_1 = vld1q_f16(reinterpret_cast<const __fp16*>(wr1 + 8));
      acc10 = vfmlalq_low_f16(acc10, w1_0, x0);
      acc10 = vfmlalq_high_f16(acc10, w1_0, x0);
      acc11 = vfmlalq_low_f16(acc11, w1_1, x1);
      acc11 = vfmlalq_high_f16(acc11, w1_1, x1);

      const float16x8_t w2_0 = vld1q_f16(reinterpret_cast<const __fp16*>(wr2));
      const float16x8_t w2_1 = vld1q_f16(reinterpret_cast<const __fp16*>(wr2 + 8));
      acc20 = vfmlalq_low_f16(acc20, w2_0, x0);
      acc20 = vfmlalq_high_f16(acc20, w2_0, x0);
      acc21 = vfmlalq_low_f16(acc21, w2_1, x1);
      acc21 = vfmlalq_high_f16(acc21, w2_1, x1);

      const float16x8_t w3_0 = vld1q_f16(reinterpret_cast<const __fp16*>(wr3));
      const float16x8_t w3_1 = vld1q_f16(reinterpret_cast<const __fp16*>(wr3 + 8));
      acc30 = vfmlalq_low_f16(acc30, w3_0, x0);
      acc30 = vfmlalq_high_f16(acc30, w3_0, x0);
      acc31 = vfmlalq_low_f16(acc31, w3_1, x1);
      acc31 = vfmlalq_high_f16(acc31, w3_1, x1);
    }
    float s0 = vaddvq_f32(vaddq_f32(acc00, acc01));
    float s1 = vaddvq_f32(vaddq_f32(acc10, acc11));
    float s2 = vaddvq_f32(vaddq_f32(acc20, acc21));
    float s3 = vaddvq_f32(vaddq_f32(acc30, acc31));
    for (; k < k_chunk; ++k, ++wr0, ++wr1, ++wr2, ++wr3, ++xr) {
      __fp16 wh0, wh1, wh2, wh3, xhh;
      std::memcpy(&wh0, wr0, sizeof wh0);
      std::memcpy(&wh1, wr1, sizeof wh1);
      std::memcpy(&wh2, wr2, sizeof wh2);
      std::memcpy(&wh3, wr3, sizeof wh3);
      std::memcpy(&xhh, xr, sizeof xhh);
      s0 += static_cast<float>(wh0) * static_cast<float>(xhh);
      s1 += static_cast<float>(wh1) * static_cast<float>(xhh);
      s2 += static_cast<float>(wh2) * static_cast<float>(xhh);
      s3 += static_cast<float>(wh3) * static_cast<float>(xhh);
    }
    y_part[r] = s0;
    y_part[r + 1] = s1;
    y_part[r + 2] = s2;
    y_part[r + 3] = s3;
  }
  for (; r < out; ++r) {
    const uint16_t* wr = w + r * in + k_start;
    const uint16_t* xr = xh + k_start;
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    size_t k = 0;
    const size_t vec_end8 = k_chunk & ~size_t{7};
    for (; k < vec_end8; k += 8, wr += 8, xr += 8) {
      const float16x8_t w8 = vld1q_f16(reinterpret_cast<const __fp16*>(wr));
      const float16x8_t x8 = vld1q_f16(reinterpret_cast<const __fp16*>(xr));
      acc0 = vfmlalq_low_f16(acc0, w8, x8);
      acc1 = vfmlalq_high_f16(acc1, w8, x8);
    }
    float s = vaddvq_f32(acc0) + vaddvq_f32(acc1);
    for (; k < k_chunk; ++k, ++wr, ++xr) {
      __fp16 wh, xhh;
      std::memcpy(&wh, wr, sizeof wh);
      std::memcpy(&xhh, xr, sizeof xhh);
      s += static_cast<float>(wh) * static_cast<float>(xhh);
    }
    y_part[r] = s;
  }
}

inline void reduce_slice(const float* const* parts, size_t n_parts, float* y, size_t r_start, size_t r_end) {
  const size_t out = r_end - r_start;
  float* dst = y + r_start;
  if (n_parts == 1) {
    if (dst != parts[0] + r_start) std::memcpy(dst, parts[0] + r_start, out * sizeof(float));
    return;
  }
  if (n_parts == 2) {
    const float* p0 = parts[0] + r_start;
    const float* p1 = parts[1] + r_start;
    size_t r = 0;
    for (; r + 4 <= out; r += 4) {
      vst1q_f32(dst + r, vaddq_f32(vld1q_f32(p0 + r), vld1q_f32(p1 + r)));
    }
    for (; r < out; ++r) dst[r] = p0[r] + p1[r];
    return;
  }
  if (n_parts == 4) {
    const float* p0 = parts[0] + r_start;
    const float* p1 = parts[1] + r_start;
    const float* p2 = parts[2] + r_start;
    const float* p3 = parts[3] + r_start;
    size_t r = 0;
    for (; r + 4 <= out; r += 4) {
      float32x4_t a01 = vaddq_f32(vld1q_f32(p0 + r), vld1q_f32(p1 + r));
      float32x4_t a23 = vaddq_f32(vld1q_f32(p2 + r), vld1q_f32(p3 + r));
      vst1q_f32(dst + r, vaddq_f32(a01, a23));
    }
    for (; r < out; ++r) dst[r] = p0[r] + p1[r] + p2[r] + p3[r];
    return;
  }
  std::memcpy(dst, parts[0] + r_start, out * sizeof(float));
  for (size_t p = 1; p < n_parts; ++p) {
    const float* src = parts[p] + r_start;
    size_t r = 0;
    for (; r + 4 <= out; r += 4) {
      vst1q_f32(dst + r, vaddq_f32(vld1q_f32(dst + r), vld1q_f32(src + r)));
    }
    for (; r < out; ++r) dst[r] += src[r];
  }
}

inline void reduce_parts(const float* const* parts, size_t n_parts, float* y, size_t out) {
  reduce_slice(parts, n_parts, y, 0, out);
}

}  // namespace ar_neon
}  // namespace

namespace gsv::ar {

using DenseF16 = kern::accel::DenseF16;
using rt::GsvFile;
using rt::TensorView;

// ---------------------------------------------------------------------------
// ArSplitKPool: 4 P-Core Persistent Barrier Thread Pool for Split-K Decode
// ---------------------------------------------------------------------------
class T2SEngine::ArSplitKPool {
 public:
  static constexpr size_t kMaxThreads = 4;
  struct alignas(64) WorkerSlot {
    std::atomic<uint32_t> task_seq{0};
    std::atomic<uint32_t> done_seq{0};
  };

  class GenerationBarrier {
   public:
    explicit GenerationBarrier(size_t total = 4) : total_(total) {
      count_[0].store(0, std::memory_order_relaxed);
      count_[1].store(0, std::memory_order_relaxed);
      gen_[0].store(0, std::memory_order_relaxed);
      gen_[1].store(0, std::memory_order_relaxed);
    }

    void sync(size_t phase) {
      if (total_ <= 1) return;
      const size_t slot = phase & 1;
      const uint32_t cur_gen = gen_[slot].load(std::memory_order_relaxed);
      if (count_[slot].fetch_add(1, std::memory_order_acq_rel) + 1 == total_) {
        count_[slot].store(0, std::memory_order_relaxed);
        gen_[slot].store(cur_gen + 1, std::memory_order_release);
      } else {
        while (gen_[slot].load(std::memory_order_acquire) == cur_gen) {
          __builtin_arm_yield();
        }
      }
    }

   private:
    size_t total_{4};
    std::atomic<uint32_t> count_[2];
    std::atomic<uint32_t> gen_[2];
  };

  explicit ArSplitKPool(size_t n_threads)
      : n_threads_(std::min(n_threads == 0 ? size_t{1} : n_threads, kMaxThreads)),
        stop_(false),
        in_session_(false),
        barrier_(std::min(n_threads == 0 ? size_t{1} : n_threads, kMaxThreads)) {
    if (n_threads_ > 1) {
      workers_.reserve(n_threads_ - 1);
      for (size_t i = 1; i < n_threads_; ++i) {
        workers_.emplace_back([this, i] {
          ::pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);
          worker_loop(i);
        });
      }
    }
  }

  ~ArSplitKPool() {
    if (!workers_.empty()) {
      {
        std::lock_guard<std::mutex> lk(mu_);
        stop_.store(true, std::memory_order_release);
        in_session_.store(false, std::memory_order_release);
        for (size_t i = 1; i < n_threads_; ++i) {
          slots_[i].task_seq.fetch_add(1, std::memory_order_release);
        }
      }
      cv_.notify_all();
      for (auto& t : workers_) t.join();
    }
  }

  void begin_session() {
    if (n_threads_ <= 1) return;
    {
      std::lock_guard<std::mutex> lk(mu_);
      in_session_.store(true, std::memory_order_release);
    }
    cv_.notify_all();
  }

  void end_session() {
    if (n_threads_ <= 1) return;
    std::lock_guard<std::mutex> lk(mu_);
    in_session_.store(false, std::memory_order_release);
  }

  template <typename Fn>
  void run_layer(Fn&& fn) {
    if (n_threads_ <= 1) {
      fn(0, 1, barrier_);
      return;
    }
    struct Runner {
      static void invoke(void* ctx, size_t tid, size_t nth,
                         GenerationBarrier& bar) {
        (*static_cast<std::decay_t<Fn>*>(ctx))(tid, nth, bar);
      }
    };
    task_fn_ = &Runner::invoke;
    task_ctx_ = (void*)&fn;
    const uint32_t seq = cur_seq_ + 1;
    cur_seq_ = seq;
    if (!in_session_.load(std::memory_order_relaxed)) {
      {
        std::lock_guard<std::mutex> lk(mu_);
        for (size_t i = 1; i < n_threads_; ++i) {
          slots_[i].task_seq.store(seq, std::memory_order_release);
        }
      }
      cv_.notify_all();
    } else {
      for (size_t i = 1; i < n_threads_; ++i) {
        slots_[i].task_seq.store(seq, std::memory_order_release);
      }
    }
    fn(0, n_threads_, barrier_);
    for (size_t i = 1; i < n_threads_; ++i) {
      while (slots_[i].done_seq.load(std::memory_order_acquire) != seq) {
        __builtin_arm_yield();
      }
    }
  }

  size_t threads() const { return n_threads_; }

 private:
  void worker_loop(size_t tid) {
    uint32_t last_seq = 0;
    while (true) {
      while (!stop_.load(std::memory_order_relaxed)) {
        if (slots_[tid].task_seq.load(std::memory_order_acquire) != last_seq) break;
        if (!in_session_.load(std::memory_order_relaxed)) {
          std::unique_lock<std::mutex> lk(mu_);
          if (slots_[tid].task_seq.load(std::memory_order_relaxed) != last_seq ||
              stop_.load(std::memory_order_relaxed)) break;
          cv_.wait(lk, [&] {
            return stop_.load(std::memory_order_relaxed) ||
                   in_session_.load(std::memory_order_relaxed) ||
                   slots_[tid].task_seq.load(std::memory_order_relaxed) != last_seq;
          });
          break;
        }
        __builtin_arm_yield();
      }
      if (stop_.load(std::memory_order_acquire)) break;
      if (slots_[tid].task_seq.load(std::memory_order_acquire) == last_seq) continue;

      last_seq = slots_[tid].task_seq.load(std::memory_order_acquire);
      task_fn_(task_ctx_, tid, n_threads_, barrier_);
      slots_[tid].done_seq.store(last_seq, std::memory_order_release);
    }
  }

  size_t n_threads_;
  std::atomic<bool> stop_;
  std::atomic<bool> in_session_;
  uint32_t cur_seq_{0};
  WorkerSlot slots_[kMaxThreads];
  GenerationBarrier barrier_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::vector<std::thread> workers_;
  void (*task_fn_)(void*, size_t, size_t, GenerationBarrier&){nullptr};
  void* task_ctx_{nullptr};
};

T2SEngine::T2SEngine(const GsvFile& f) {
  // ---- 维度: 全部来自 ckpt 内嵌 config(convert.py 存入 .gsv header) ----
  const auto* model_cfg = f.config().find("model_config");
  const auto* m = model_cfg ? model_cfg->find("model") : nullptr;
  if (!m) throw std::runtime_error("ar .gsv 缺 model_config.model");
  dims_.d_model = static_cast<size_t>(m->find("hidden_dim")->as_int());
  dims_.n_heads = static_cast<size_t>(m->find("head")->as_int());
  dims_.ffn = static_cast<size_t>(m->find("linear_units")->as_int());
  dims_.n_layers = static_cast<size_t>(m->find("n_layer")->as_int());
  dims_.vocab = static_cast<size_t>(m->find("vocab_size")->as_int());
  dims_.phone_vocab = static_cast<size_t>(m->find("phoneme_vocab_size")->as_int());
  dims_.bert_dim = 1024; // bert_proj 固定输入维(roberta hidden), config 不含 → 常量
  dims_.eos = static_cast<int>(m->find("EOS")->as_int());

  const size_t D = dims_.d_model;
  if (dims_.d_model != static_cast<size_t>(m->find("embedding_dim")->as_int()))
    throw std::runtime_error("embedding_dim != hidden_dim: 本引擎按相等设计");
  if (dims_.eos != static_cast<int>(dims_.vocab) - 1)
    throw std::runtime_error("EOS != vocab-1");

  auto vec_of = [&](const char* name) {
    const TensorView& t = need(f, name);
    return std::vector<float>(t.data_f32(), t.data_f32() + t.numel());
  };
  auto dense16 = [&](const char* name, size_t rows, size_t cols) {
    const TensorView& t = need(f, name);
    if (t.numel() != rows * cols || !t.has_f16())
      throw std::runtime_error(std::string(name) + ": 形状/f16 段不符");
    return DenseF16(t.data_f16_raw(), rows, cols);
  };

  // ---- 词表嵌入: 整表升位一次(共 ~3.6MB fp32, 查表 O(1) 且免逐行转换) ----
  {
    const TensorView& te = need(f, "ar_text_embedding.word_embeddings.weight");
    if (te.numel() != dims_.phone_vocab * D || !te.has_f16())
      throw std::runtime_error("ar_text_embedding 形状不符");
    text_emb_.resize(te.numel());
    accel::f16_to_f32(te.data_f16_raw(), text_emb_.data(), te.numel());
    const TensorView& ae = need(f, "ar_audio_embedding.word_embeddings.weight");
    if (ae.numel() != dims_.vocab * D || !ae.has_f16())
      throw std::runtime_error("ar_audio_embedding 形状不符");
    audio_emb_.resize(ae.numel());
    accel::f16_to_f32(ae.data_f16_raw(), audio_emb_.data(), ae.numel());
  }

  bert_b_ = vec_of("bert_proj.bias");
  bert_proj_ = dense16("bert_proj.weight", D, dims_.bert_dim);
  {
    const TensorView& t = need(f, "ar_predict_layer.weight");
    if (t.numel() != dims_.vocab * D || !t.has_f16())
      throw std::runtime_error("ar_predict_layer 形状/f16 段不符");
    wp_ = DenseF16(t.data_f16_raw(), dims_.vocab, D);
    wp16_.assign(t.data_f16_raw(), t.data_f16_raw() + t.numel());
  }
  alpha_text_ = need(f, "ar_text_position.alpha").data_f32()[0];
  alpha_audio_ = need(f, "ar_audio_position.alpha").data_f32()[0];

  layers_.resize(dims_.n_layers);
  for (size_t l = 0; l < dims_.n_layers; ++l) {
    Layer& L = layers_[l];
    const std::string p = "h.layers." + std::to_string(l) + ".";
    const std::string ps = p + "self_attn.";
    auto dense16r = [&](const char* name, size_t rows, size_t cols,
                        std::vector<uint16_t>* raw) {
      const TensorView& t = need(f, name);
      if (t.numel() != rows * cols || !t.has_f16())
        throw std::runtime_error(std::string(name) + ": 形状/f16 段不符");
      if (raw)
        raw->assign(t.data_f16_raw(), t.data_f16_raw() + t.numel());
      return DenseF16(t.data_f16_raw(), rows, cols);
    };
    L.wqkv = dense16r((ps + "in_proj_weight").c_str(), 3 * D, D, &L.wqkv16);
    L.bqkv = vec_of((ps + "in_proj_bias").c_str());
    L.wout = dense16r((ps + "out_proj.weight").c_str(), D, D, &L.wout16);
    L.bout = vec_of((ps + "out_proj.bias").c_str());
    L.w1 = dense16r((p + "linear1.weight").c_str(), dims_.ffn, D, &L.w116);
    L.b1 = vec_of((p + "linear1.bias").c_str());
    L.w2 = dense16r((p + "linear2.weight").c_str(), D, dims_.ffn, &L.w216);
    L.b2 = vec_of((p + "linear2.bias").c_str());
    L.n1g = vec_of((p + "norm1.weight").c_str());
    L.n1b = vec_of((p + "norm1.bias").c_str());
    L.n2g = vec_of((p + "norm2.weight").c_str());
    L.n2b = vec_of((p + "norm2.bias").c_str());
#ifdef GSV_AMX_GEMM
    // E11-2: prefill AMX 预打包面板(包后即可在 block_prefill_impl 走 gemm_f16_amx_pp)
    if (kern::amx_gemm_available()) {
      L.wqkv_pa = kern::amx_pack(L.wqkv16.data(), 3 * D, D);
      L.w1_pa   = kern::amx_pack(L.w116.data(),   dims_.ffn, D);
      L.w2_pa   = kern::amx_pack(L.w216.data(),   D, dims_.ffn);
    }
#endif
  }
#ifdef GSV_AMX_GEMM
  prefill_amx_in_use_ = kern::amx_gemm_available();
#endif

  // E18: 初始化 split-K scratch 缓冲
  sk_part_qkv_.assign(4 * 3 * D, 0.f);
  sk_part_wout_.assign(4 * D, 0.f);
  sk_part_w1_.assign(4 * dims_.ffn, 0.f);
  sk_part_w2_.assign(4 * D, 0.f);
  sk_qkv_.assign(3 * D, 0.f);
  sk_attn_.assign(D, 0.f);
  sk_xb_.assign(D, 0.f);
  sk_ff_.assign(dims_.ffn, 0.f);
  sk_hbuf_.assign(D, 0.f);
  sk_scores_.assign(4 * (kMaxDecodeSteps + 500), 0.f);
  sk_xh512_.assign(D, 0);
  sk_attn16_.assign(D, 0);
  sk_xb16_.assign(D, 0);
  sk_ff16_.assign(dims_.ffn, 0);

  for (size_t i = 0; i < 4; ++i) {
    sk_qkv_ptrs_[i] = sk_part_qkv_.data() + i * 3 * D;
    sk_wout_ptrs_[i] = sk_part_wout_.data() + i * D;
    sk_w1_ptrs_[i] = sk_part_w1_.data() + i * dims_.ffn;
    sk_w2_ptrs_[i] = sk_part_w2_.data() + i * D;
  }
}

T2SEngine::~T2SEngine() = default;

void T2SEngine::set_splitk(bool enable) {
  ar_splitk_ = enable;
  if (enable && !splitk_pool_) {
    size_t pool_size = rt::p_core_count();
    if (pool_size > 4) pool_size = 4;
    if (pool_size == 0) pool_size = 1;
    if (const char* e = std::getenv("GSV_AMX_THREADS")) {
      long v = std::atol(e);
      if (v > 0) pool_size = static_cast<size_t>(v);
    }
    if (pool_size > ArSplitKPool::kMaxThreads) {
      pool_size = ArSplitKPool::kMaxThreads;
    }
    splitk_pool_ = std::make_unique<ArSplitKPool>(pool_size);
  }
}

const TensorView& T2SEngine::need(const GsvFile& f, const char* name) const {
  const auto* t = f.tensor(name);
  if (!t) throw std::runtime_error(std::string("缺张量: ") + name);
  return *t;
}

// ---- M1-fp16 开关与直读路径 ----
void T2SEngine::set_fp16(const Fp16Options& o) { fp16_ = o; }

// y[out] = W16·xh: 激活舍入到 fp16 后走 FMLAL 扩展精度累加 GEMV。
// xh_ 复用为暂存(调用方串行使用, 无重入)。
// y[out] = W16·xh: 激活舍入到 fp16 后走 FMLAL 扩展精度累加 GEMV。
// xh_ 复用为暂存(调用方串行使用, 无重入)。
void T2SEngine::gemv_fmlal(const std::vector<uint16_t>& w16, const float* x,
                           size_t out, size_t in, float* y) {
  xh_.resize(in);
  kern::f32_to_f16(x, xh_.data(), in);
  kern::gemv_f16x_fmlal(w16.data(), xh_.data(), y, out, in);
}

// generate() 内部使用的 logits 投影(尊重 fp16.gemv)
void T2SEngine::predict_layer_fp(const float* x, float* y) {
  // Logits 投影层保持 fp32 精度 (DenseF16 升位 sgemm) 以免在 top-1/top-2 微小差异时引起 argmax 翻转
  wp_.forward(x, 1, y);
}

// SinePositionalEmbedding.extend_pe 单行同构:
// div_term[i]=exp(2i·(-ln(10000)/D)); pe[2i]=sin(pos·div), pe[2i+1]=cos(pos·div)
void T2SEngine::pe_row(float* pe, size_t pos) {
  const size_t D = dims_.d_model;
  const float log_inc = static_cast<float>(-std::log(10000.0) / static_cast<double>(D));
  const float p = static_cast<float>(pos);
  for (size_t d = 0; d < D; d += 2) {
    const float div = std::exp(static_cast<float>(d) * log_inc);
    pe[d] = std::sin(p * div);
    pe[d + 1] = std::cos(p * div);
  }
}

// ---- prefill 单层: 与 T2SBlock.process_prompt 同构(post-LN) ----
// KV16: k/v 以 fp16 位型写入 cache, 注意力读出升位 fp32 计算(存储舍入一次)。
#ifdef GSV_AMX_GEMM
// E11-5: prefill SDPA 走 AMX (Q·K^T + P·V 两次 GEMM per head, 16 头批量派发)
//   设计: 从 qkv_ 抽取 per-head Q/K/V (各 [S, HD] 头主) 进临时 f16 缓冲,
//   pack 进 AmxPanel, 16 头 Q·K^T 一次性 amx_batch_run 提交 (phase 0),
//   应用 -inf 掩码 → 行 softmax → P·V (phase 1) 一次性 提交。
//   数值纪律: K=HD=32 薄 K 形状, AMX 归约序与 FMLAL 4-lane 树形可能差异较 E11-2 大
//   (E11-2 K=512/2048 形状下 fp16 源等价, 这里 fp32 → fp16 转换 + GEMM 归约序双重漂移)。
//   G1 验收 cos8≥0.9999 + B12 全过 为位级可比验证。
//   熔断: amx_gemm_available() 失败 或 HD 非 32 倍数 (rare) → 走 NEON 紧致路径。
template <bool KV16>
void T2SEngine::sdpa_amx_prefill(size_t H, size_t S, size_t HD, float scale,
                                 size_t text_len, const float* qkv_buf,
                                 const float* kf32, const uint16_t* k16,
                                 const float* vf32, const uint16_t* v16,
                                 float* attn_out) {
  const size_t D = H * HD;
  const size_t need = S * S > S * HD ? S * S : S * HD;
  const size_t nth = std::min(size_t{4}, kern::amx_pool_healthy_threads());
  const size_t use_nth = nth > 0 ? nth : 1;

  if (sdpa_threads_.size() < use_nth) {
    sdpa_threads_.resize(use_nth);
  }

  for (size_t tid = 0; tid < use_nth; ++tid) {
    auto& scr = sdpa_threads_[tid];
    if (scr.cap < need) {
      scr.xh.resize(need);
      scr.cap = need;
    }
    if (scr.scores.size() < S * S) scr.scores.assign(S * S, 0.f);
    if (scr.probs.size() < S * S) scr.probs.assign(S * S, 0.f);
    if (scr.Q_f16.size() < S * HD) scr.Q_f16.resize(S * HD);
    if (scr.K_f16.size() < S * HD) scr.K_f16.resize(S * HD);
    if (scr.V_f16.size() < S * HD) scr.V_f16.resize(S * HD);
    if (scr.VT_f16.size() < HD * S) scr.VT_f16.resize(HD * S);
    if (scr.attn_tmp.size() < S * HD) scr.attn_tmp.resize(S * HD);
    scr.pa_Q.rows = S; scr.pa_Q.K = HD;
    scr.pa_K.rows = S; scr.pa_K.K = HD;
    scr.pa_V.rows = HD; scr.pa_V.K = S;
    scr.pa_P.rows = S; scr.pa_P.K = S;
  }

  auto run_head = [&](size_t h, SdpaThreadScratch& scr) {
    // 1) 从 qkv_ 抽取 Q_h: qkv_ 布局 [S, 3D] (D=H*HD), 每 token Q 段起点 = q*3D
    //    head h 段起点 = q*3D + h*HD; Q_h[t, e] = qkv_[t*3D + h*HD + e]
    for (size_t t = 0; t < S; ++t) {
      const float* src = qkv_buf + t * 3 * D + h * HD;
      for (size_t e = 0; e < HD; ++e) {
        __fp16 hh = static_cast<__fp16>(src[e]);
        __builtin_memcpy(&scr.Q_f16[t * HD + e], &hh, 2);
      }
    }
    // 2) 抽取 K_h: KV cache 布局 [S, D] 行主, head h 段起点 = t*D + h*HD
    for (size_t t = 0; t < S; ++t) {
      if constexpr (KV16) {
        const uint16_t* src = k16 + t * D + h * HD;
        std::memcpy(&scr.K_f16[t * HD], src, HD * sizeof(uint16_t));
      } else {
        const float* src = kf32 + t * D + h * HD;
        for (size_t e = 0; e < HD; ++e) {
          __fp16 hh = static_cast<__fp16>(src[e]);
          __builtin_memcpy(&scr.K_f16[t * HD + e], &hh, 2);
        }
      }
    }
    // 3) 抽取 V_h
    for (size_t t = 0; t < S; ++t) {
      if constexpr (KV16) {
        const uint16_t* src = v16 + t * D + h * HD;
        std::memcpy(&scr.V_f16[t * HD], src, HD * sizeof(uint16_t));
      } else {
        const float* src = vf32 + t * D + h * HD;
        for (size_t e = 0; e < HD; ++e) {
          __fp16 hh = static_cast<__fp16>(src[e]);
          __builtin_memcpy(&scr.V_f16[t * HD + e], &hh, 2);
        }
      }
    }

    // 4) Pack Q/K/V 进 AmxPanel
    for (size_t t = 0; t < S; ++t) {
      for (size_t e = 0; e < HD; ++e) {
        scr.VT_f16[e * S + t] = scr.V_f16[t * HD + e];
      }
    }
    kern::amx_pack_into(scr.Q_f16.data(), S, HD, scr.pa_Q.buf);
    kern::amx_pack_into(scr.K_f16.data(), S, HD, scr.pa_K.buf);
    kern::amx_pack_into(scr.VT_f16.data(), HD, S, scr.pa_V.buf);

    // 5) Q·K^T via AMX: pa_Q [S,HD] (M=S), pa_K [S,HD] (N=S) → scores [S, S]
    float* scores = scr.scores.data();
    kern::gemm_f16_amx_pp(scr.pa_Q, scr.pa_K, scores, S, S);
    // scale
    for (size_t i = 0; i < S * S; ++i) scores[i] *= scale;

    // 6) 应用掩码: k < text_len || k <= q → 保持; 否则 -inf
    for (size_t q = 0; q < S; ++q) {
      for (size_t k = 0; k < S; ++k) {
        const bool allowed = (k < text_len) || (k <= q);
        if (!allowed) scores[q * S + k] = -std::numeric_limits<float>::infinity();
      }
    }
    // 7) 行 softmax: 复制到 probs, in-place 稳定 softmax
    float* probs = scr.probs.data();
    for (size_t q = 0; q < S; ++q) {
      float* row = scores + q * S;
      float mx = row[0];
      for (size_t k = 1; k < S; ++k) mx = std::max(mx, row[k]);
      float sum = 0.f;
      for (size_t k = 0; k < S; ++k) {
        const float e = std::exp(row[k] - mx);
        probs[q * S + k] = e;
        sum += e;
      }
      const float inv = sum > 0 ? 1.f / sum : 0.f;
      for (size_t k = 0; k < S; ++k) probs[q * S + k] *= inv;
    }
    // 8) Pack P 进 AmxPanel
    if (scr.cap < S * S) {
      scr.xh.resize(S * S);
      scr.cap = S * S;
    }
    for (size_t i = 0; i < S * S; ++i) {
      __fp16 hh = static_cast<__fp16>(probs[i]);
      __builtin_memcpy(&scr.xh[i], &hh, 2);
    }
    kern::amx_pack_into(scr.xh.data(), S, S, scr.pa_P.buf);

    // 9) P·V via AMX: pa_P [S, S] (M=S), pa_V [HD, S] (N=HD, K=S)
    kern::gemm_f16_amx_pp(scr.pa_P, scr.pa_V, scr.attn_tmp.data(), S, HD);
    // scatter: attn_[q*D + h*HD + e] = attn_tmp[q*HD + e]  (D 上面已定义)
    for (size_t q = 0; q < S; ++q) {
      std::memcpy(attn_out + q * D + h * HD, scr.attn_tmp.data() + q * HD,
                  HD * sizeof(float));
    }
  };

  if (use_nth <= 1) {
    for (size_t h = 0; h < H; ++h) run_head(h, sdpa_threads_[0]);
  } else {
    std::vector<std::function<void()>> batch;
    batch.reserve(use_nth);
    for (size_t tid = 0; tid < use_nth; ++tid) {
      const size_t h0 = tid * H / use_nth;
      const size_t h1 = (tid + 1) * H / use_nth;
      batch.emplace_back([&, tid, h0, h1] {
        for (size_t h = h0; h < h1; ++h) {
          run_head(h, sdpa_threads_[tid]);
        }
      });
    }
    kern::amx_run_batch(std::move(batch));
  }
}
#endif  // GSV_AMX_GEMM
template <bool KV16>
void T2SEngine::block_prefill_impl(size_t l, float* x, size_t S, size_t pos,
                                   size_t text_len, float* kf32, uint16_t* k16,
                                   float* vf32, uint16_t* v16) {
  const size_t D = dims_.d_model, H = dims_.n_heads, HD = D / H;
  const size_t FF = dims_.ffn;
  Layer& L = layers_[l];
  const float scale = 1.0f / std::sqrt(static_cast<float>(HD));

  const auto t_start_wqkv = std::chrono::steady_clock::now();
  // fused QKV: qkv[S,3D] = x·Wqkvᵀ+b
  qkv_.resize(S * 3 * D);
#ifdef GSV_AMX_GEMM
  if (prefill_amx_in_use_) {
    if (prefill_cap_ < S * D) {
      prefill_xh_.resize(S * D);
      prefill_cap_ = S * D;
    }
    kern::f32_to_f16(x, prefill_xh_.data(), S * D);
    prefill_pa_act_.rows = S; prefill_pa_act_.K = D;
    kern::amx_pack_into(prefill_xh_.data(), S, D, prefill_pa_act_.buf);
    // L.wqkv_pa 预打包为 [3D][D] = pb 侧 (N=3D)
    kern::gemm_f16_amx_pp(prefill_pa_act_, L.wqkv_pa, qkv_.data(), S, 3 * D);
  } else
#endif
  {
    L.wqkv.forward(x, S, qkv_.data());
  }
  for (size_t i = 0; i < S; ++i)
    for (size_t j = 0; j < 3 * D; ++j) qkv_[i * 3 * D + j] += L.bqkv[j];

  const auto t_start_sdpa = std::chrono::steady_clock::now();
  pf_wqkv_ms_ += std::chrono::duration<double, std::milli>(t_start_sdpa - t_start_wqkv).count();

  // k/v 写 cache 第 pos..pos+S-1 槽
  for (size_t t = 0; t < S; ++t) {
    const float* krow = qkv_.data() + t * 3 * D + D;
    const float* vrow = qkv_.data() + t * 3 * D + 2 * D;
    if constexpr (KV16) {
      kern::f32_to_f16(krow, k16 + (pos + t) * D, D);
      kern::f32_to_f16(vrow, v16 + (pos + t) * D, D);
    } else {
      std::memcpy(kf32 + (pos + t) * D, krow, D * sizeof(float));
      std::memcpy(vf32 + (pos + t) * D, vrow, D * sizeof(float));
    }
  }

  scores_.resize(S);
  probs_.resize(S);
  attn_.assign(S * D, 0.f);
  if constexpr (KV16) kvrow_.resize(D);

#ifdef GSV_AMX_GEMM
  if (prefill_amx_in_use_ && HD % 32 == 0) {  // AMX tile 是 32 行, HD=32 占一行, 保证对齐
    if constexpr (KV16)
      sdpa_amx_prefill<true>(H, S, HD, scale, text_len, qkv_.data(), kf32, k16, vf32, v16,
                             attn_.data());
    else
      sdpa_amx_prefill<false>(H, S, HD, scale, text_len, qkv_.data(), kf32, k16, vf32, v16,
                              attn_.data());
  } else
#endif
  {
    for (size_t h = 0; h < H; ++h) {
      for (size_t q = 0; q < S; ++q) {
        const float* qv = qkv_.data() + q * 3 * D + h * HD;
        const size_t n_max = (text_len > q + 1 ? text_len : q + 1);
        size_t n = 0;
        for (size_t k = 0; k < n_max; ++k) {
          if (!(k < text_len || k <= q)) continue;
          float dot;
          if constexpr (KV16) {
            dot = ar_neon::dot_f16kv(qv, k16 + (pos + k) * D + h * HD, HD);
          } else {
            dot = ar_neon::dot_f32(qv, kf32 + (pos + k) * D + h * HD, HD);
          }
          scores_[n++] = dot * scale;
        }
        gsv::kern::softmax(scores_.data(), probs_.data(), n);
        float* ov = attn_.data() + q * D + h * HD;
        size_t idx = 0;
        for (size_t k = 0; k < n_max; ++k) {
          if (!(k < text_len || k <= q)) continue;
          const float p = probs_[idx++];
          if constexpr (KV16) {
            ar_neon::accum_f16kv(ov, v16 + (pos + k) * D + h * HD, p, HD);
          } else {
            ar_neon::accum_f32(ov, vf32 + (pos + k) * D + h * HD, p, HD);
          }
        }
      }
    }
  }

  const auto t_start_wout = std::chrono::steady_clock::now();
  pf_sdpa_ms_ += std::chrono::duration<double, std::milli>(t_start_wout - t_start_sdpa).count();

  // out proj + 残差 + post-LN(norm1)
  tmp_.resize(S * D);
  L.wout.forward(attn_.data(), S, tmp_.data());
  for (size_t i = 0; i < S * D; ++i) x[i] += tmp_[i] + L.bout[i % D];

  const auto t_start_ln1 = std::chrono::steady_clock::now();
  pf_wout_ms_ += std::chrono::duration<double, std::milli>(t_start_ln1 - t_start_wout).count();

  for (size_t t = 0; t < S; ++t)
    gsv::kern::layernorm(x + t * D, L.n1g.data(), L.n1b.data(), x + t * D, D,
                         dims_.ln_eps);

  const auto t_start_w1 = std::chrono::steady_clock::now();
  pf_ln_ms_ += std::chrono::duration<double, std::milli>(t_start_w1 - t_start_ln1).count();

  // FFN(ReLU) + 残差 + post-LN(norm2)
  ff_.resize(S * FF);
  L.w1.forward(x, S, ff_.data());
  for (size_t i = 0; i < S * FF; ++i) ff_[i] += L.b1[i % FF];
  gsv::kern::relu(ff_.data(), ff_.data(), S * FF);

  const auto t_start_w2 = std::chrono::steady_clock::now();
  pf_w1_ms_ += std::chrono::duration<double, std::milli>(t_start_w2 - t_start_w1).count();

  // E11-2: W2 走 AMX (FFN 下降 S×FF → S×D)
#ifdef GSV_AMX_GEMM
  if (prefill_amx_in_use_) {
    if (prefill_cap_ < S * FF) {
      prefill_xh_.resize(S * FF);
      prefill_cap_ = S * FF;
    }
    kern::f32_to_f16(ff_.data(), prefill_xh_.data(), S * FF);
    prefill_pa_act_.rows = S; prefill_pa_act_.K = dims_.ffn;
    kern::amx_pack_into(prefill_xh_.data(), S, dims_.ffn, prefill_pa_act_.buf);
    // L.w2_pa 预打包为 [D][FF] = pb 侧 (N=D)
    kern::gemm_f16_amx_pp(prefill_pa_act_, L.w2_pa, tmp_.data(), S, D);
  } else
#endif
  {
    L.w2.forward(ff_.data(), S, tmp_.data());
  }
  for (size_t i = 0; i < S * D; ++i) x[i] += tmp_[i] + L.b2[i % D];

  const auto t_start_ln2 = std::chrono::steady_clock::now();
  pf_w2_ms_ += std::chrono::duration<double, std::milli>(t_start_ln2 - t_start_w2).count();

  for (size_t t = 0; t < S; ++t)
    gsv::kern::layernorm(x + t * D, L.n2g.data(), L.n2b.data(), x + t * D, D,
                         dims_.ln_eps);

  const auto t_end_layer = std::chrono::steady_clock::now();
  pf_ln_ms_ += std::chrono::duration<double, std::milli>(t_end_layer - t_start_ln2).count();
}  // block_prefill_impl<KV16>

// ---- decode 单层: 与 T2SBlock.decode_next_token 同构 ----
// KV cache 布局: token-major 行主 cache[tok*D + head*HD + e](见头文件),
// 每 token 写一行, 注意力对 [0,len) 全可见(decode 阶段无掩码)。
// KV16: 存储 fp16/读出计算 fp32; GEMV16: 全部权重 GEMV 走 FMLAL 直读。
template <bool KV16, bool GEMV16>
void T2SEngine::block_decode_impl(size_t l, float* x, size_t pos, size_t len,
                                  float* kf32, uint16_t* k16, float* vf32,
                                  uint16_t* v16) {
  const size_t D = dims_.d_model, H = dims_.n_heads, HD = D / H;
  Layer& L = layers_[l];
  const float scale = 1.0f / std::sqrt(static_cast<float>(HD));

  // fused QKV 单行 GEMV: qkv[3D]
  dec_qkv_.resize(3 * D);
  float* qkv = dec_qkv_.data();
  if constexpr (GEMV16)
    gemv_fmlal(L.wqkv16, x, 3 * D, D, qkv);
  else
    L.wqkv.forward(x, 1, qkv);
  for (size_t j = 0; j < 3 * D; ++j) qkv[j] += L.bqkv[j];

  // 本 token 的 k/v 追加到第 pos 槽(len == pos+1, 见 generate 调用点)
  if constexpr (KV16) {
    kern::f32_to_f16(qkv + D, k16 + pos * D, D);
    kern::f32_to_f16(qkv + 2 * D, v16 + pos * D, D);
  } else {
    std::memcpy(kf32 + pos * D, qkv + D, D * sizeof(float));
    std::memcpy(vf32 + pos * D, qkv + 2 * D, D * sizeof(float));
  }

  // SDPA 单 query 对 len 个 key — E11-1: 标量 → NEON 4-lane 树形
  scores_.resize(len);
  attn_.assign(D, 0.f);
  if constexpr (KV16) kvrow_.resize(D);
  for (size_t h = 0; h < H; ++h) {
    const float* qv = qkv + h * HD;
    for (size_t k = 0; k < len; ++k) {
      float dot;
      if constexpr (KV16) {
        // 原地升位 + dot (避免中间缓冲跳跳; 升位为 inline 热路径)
        dot = ar_neon::dot_f16kv(qv, k16 + k * D + h * HD, HD);
      } else {
        dot = ar_neon::dot_f32(qv, kf32 + k * D + h * HD, HD);
      }
      scores_[k] = dot * scale;
    }
    gsv::kern::softmax(scores_.data(), scores_.data(), len);  // 就地稳定 softmax
    float* ov = attn_.data() + h * HD;
    for (size_t k = 0; k < len; ++k) {
      const float p = scores_[k];
      if constexpr (KV16) {
        ar_neon::accum_f16kv(ov, v16 + k * D + h * HD, p, HD);
      } else {
        ar_neon::accum_f32(ov, vf32 + k * D + h * HD, p, HD);
      }
    }
  }

  // out proj + 残差 + post-LN(norm1)
  dec_xb_.resize(D);
  float* xb = dec_xb_.data();
  if constexpr (GEMV16)
    gemv_fmlal(L.wout16, attn_.data(), D, D, xb);
  else
    L.wout.forward(attn_.data(), 1, xb);
  for (size_t i = 0; i < D; ++i) xb[i] += L.bout[i];
  for (size_t i = 0; i < D; ++i) xb[i] += x[i];
  gsv::kern::layernorm(xb, L.n1g.data(), L.n1b.data(), xb, D, dims_.ln_eps);

  // FFN(ReLU) + 残差 + post-LN(norm2)
  // 注意 post-LN 第二残差基是 norm1 输出 h(即当前 xb), 不是层输入 ——
  // 对应 torch `x = x + self.mlp.forward(x)`(此处 x 已是 norm1 之后)。
  std::vector<float>& hbuf = dec_h_;
  hbuf.resize(D);
  std::memcpy(hbuf.data(), xb, D * sizeof(float));
  ff_.resize(dims_.ffn);
  if constexpr (GEMV16)
    gemv_fmlal(L.w116, xb, dims_.ffn, D, ff_.data());
  else
    L.w1.forward(xb, 1, ff_.data());
  for (size_t i = 0; i < dims_.ffn; ++i) ff_[i] += L.b1[i];
  gsv::kern::relu(ff_.data(), ff_.data(), dims_.ffn);
  if constexpr (GEMV16)
    L.w2.forward(ff_.data(), 1, xb);  // W2 走 fp32 sgemm/升位
  else
    L.w2.forward(ff_.data(), 1, xb);
  for (size_t i = 0; i < D; ++i) xb[i] += L.b2[i];
  for (size_t i = 0; i < D; ++i) xb[i] += hbuf[i];
  gsv::kern::layernorm(xb, L.n2g.data(), L.n2b.data(), x, D, dims_.ln_eps);
}  // block_decode_impl<KV16,GEMV16>

template <bool KV16, bool GEMV16>
void T2SEngine::block_decode_splitk_impl(size_t l, float* x, size_t pos,
                                         size_t len, float* kf32, uint16_t* k16,
                                         float* vf32, uint16_t* v16) {
  if (!splitk_pool_) set_splitk(true);
  const size_t D = dims_.d_model, H = dims_.n_heads, HD = D / H, FF = dims_.ffn;
  Layer& L = layers_[l];
  const float scale = 1.0f / std::sqrt(static_cast<float>(HD));
  const size_t max_scores_stride = kMaxDecodeSteps + 500;

  splitk_pool_->run_layer([&](size_t tid, size_t nth, auto& barrier) {
    size_t phase = 0;
    const size_t k0_d = tid * (D / nth);
    const size_t k1_d = (tid + 1) * (D / nth);
    float* qkv_part = sk_part_qkv_.data() + tid * 3 * D;

    if constexpr (GEMV16) {
      kern::f32_to_f16(x + k0_d, sk_xh512_.data() + k0_d, k1_d - k0_d);
      ar_neon::gemv_slice_f16x_fmlal_4rows(L.wqkv16.data(), sk_xh512_.data(),
                                           qkv_part, 3 * D, D, k0_d, k1_d);
    } else {
      ar_neon::gemv_slice_f16w_f32acc_4rows(L.wqkv16.data(), x, qkv_part,
                                            3 * D, D, k0_d, k1_d);
    }

    barrier.sync(phase++);

    // --- E19: SDPA Head-Parallel + QKV Reduction + KV Write in Parallel ---
    const size_t h0 = tid * H / nth;
    const size_t h1 = (tid + 1) * H / nth;
    const size_t q_off = h0 * HD;
    const size_t q_len = (h1 - h0) * HD;
    const size_t k_off = D + q_off;
    const size_t v_off = 2 * D + q_off;

    ar_neon::reduce_slice(sk_qkv_ptrs_, nth, sk_qkv_.data(), q_off, q_off + q_len);
    for (size_t j = q_off; j < q_off + q_len; ++j) sk_qkv_[j] += L.bqkv[j];

    ar_neon::reduce_slice(sk_qkv_ptrs_, nth, sk_qkv_.data(), k_off, k_off + q_len);
    for (size_t j = k_off; j < k_off + q_len; ++j) sk_qkv_[j] += L.bqkv[j];

    ar_neon::reduce_slice(sk_qkv_ptrs_, nth, sk_qkv_.data(), v_off, v_off + q_len);
    for (size_t j = v_off; j < v_off + q_len; ++j) sk_qkv_[j] += L.bqkv[j];

    if constexpr (KV16) {
      kern::f32_to_f16(sk_qkv_.data() + k_off, k16 + pos * D + q_off, q_len);
      kern::f32_to_f16(sk_qkv_.data() + v_off, v16 + pos * D + q_off, q_len);
    } else {
      std::memcpy(kf32 + pos * D + q_off, sk_qkv_.data() + k_off, q_len * sizeof(float));
      std::memcpy(vf32 + pos * D + q_off, sk_qkv_.data() + v_off, q_len * sizeof(float));
    }

    float* scores = sk_scores_.data() + tid * max_scores_stride;
    for (size_t h = h0; h < h1; ++h) {
      const float* qv = sk_qkv_.data() + h * HD;
      for (size_t k = 0; k < len; ++k) {
        float dot;
        if constexpr (KV16) {
          dot = ar_neon::dot_f16kv(qv, k16 + k * D + h * HD, HD);
        } else {
          dot = ar_neon::dot_f32(qv, kf32 + k * D + h * HD, HD);
        }
        scores[k] = dot * scale;
      }
      gsv::kern::softmax(scores, scores, len);
      float* ov = sk_attn_.data() + h * HD;
      std::memset(ov, 0, HD * sizeof(float));
      for (size_t k = 0; k < len; ++k) {
        const float p = scores[k];
        if constexpr (KV16) {
          ar_neon::accum_f16kv(ov, v16 + k * D + h * HD, p, HD);
        } else {
          ar_neon::accum_f32(ov, vf32 + k * D + h * HD, p, HD);
        }
      }
    }

    if constexpr (GEMV16) {
      kern::f32_to_f16(sk_attn_.data() + q_off, sk_attn16_.data() + q_off, q_len);
    }

    barrier.sync(phase++);

    float* wout_part = sk_part_wout_.data() + tid * D;
    if constexpr (GEMV16) {
      ar_neon::gemv_slice_f16x_fmlal_4rows(L.wout16.data(), sk_attn16_.data(),
                                           wout_part, D, D, k0_d, k1_d);
    } else {
      ar_neon::gemv_slice_f16w_f32acc_4rows(L.wout16.data(), sk_attn_.data(),
                                            wout_part, D, D, k0_d, k1_d);
    }

    barrier.sync(phase++);

    if (tid == 0) {
      float* xb = sk_xb_.data();
      ar_neon::reduce_parts(sk_wout_ptrs_, nth, xb, D);
      for (size_t i = 0; i < D; ++i) xb[i] += L.bout[i] + x[i];
      gsv::kern::layernorm(xb, L.n1g.data(), L.n1b.data(), xb, D, dims_.ln_eps);
      std::memcpy(sk_hbuf_.data(), xb, D * sizeof(float));
      if constexpr (GEMV16) {
        kern::f32_to_f16(xb, sk_xb16_.data(), D);
      }
    }

    barrier.sync(phase++);

    float* w1_part = sk_part_w1_.data() + tid * FF;
    if constexpr (GEMV16) {
      ar_neon::gemv_slice_f16x_fmlal_4rows(L.w116.data(), sk_xb16_.data(),
                                           w1_part, FF, D, k0_d, k1_d);
    } else {
      ar_neon::gemv_slice_f16w_f32acc_4rows(L.w116.data(), sk_xb_.data(),
                                            w1_part, FF, D, k0_d, k1_d);
    }

    barrier.sync(phase++);

    // --- E19: W1 Reduction + Bias + ReLU + F16 in Parallel ---
    const size_t ff0 = tid * FF / nth;
    const size_t ff1 = (tid + 1) * FF / nth;
    const size_t ff_len = ff1 - ff0;
    ar_neon::reduce_slice(sk_w1_ptrs_, nth, sk_ff_.data(), ff0, ff1);
    for (size_t i = ff0; i < ff1; ++i) sk_ff_[i] += L.b1[i];
    gsv::kern::relu(sk_ff_.data() + ff0, sk_ff_.data() + ff0, ff_len);
    if constexpr (GEMV16) {
      kern::f32_to_f16(sk_ff_.data() + ff0, sk_ff16_.data() + ff0, ff_len);
    }

    barrier.sync(phase++);

    const size_t k0_ff = tid * (FF / nth);
    const size_t k1_ff = (tid + 1) * (FF / nth);
    float* w2_part = sk_part_w2_.data() + tid * D;
    if constexpr (GEMV16) {
      ar_neon::gemv_slice_f16x_fmlal_4rows(L.w216.data(), sk_ff16_.data(),
                                           w2_part, D, FF, k0_ff, k1_ff);
    } else {
      ar_neon::gemv_slice_f16w_f32acc_4rows(L.w216.data(), sk_ff_.data(),
                                            w2_part, D, FF, k0_ff, k1_ff);
    }

    barrier.sync(phase++);

    if (tid == 0) {
      ar_neon::reduce_parts(sk_w2_ptrs_, nth, x, D);
      for (size_t i = 0; i < D; ++i) x[i] += L.b2[i] + sk_hbuf_[i];
      gsv::kern::layernorm(x, L.n2g.data(), L.n2b.data(), x, D, dims_.ln_eps);
    }
  });
}

template <bool KV16, bool GEMV16>
void T2SEngine::decode_step_splitk_impl(float* x, size_t pos, size_t len,
                                        float* logits_out) {
  if (!splitk_pool_) set_splitk(true);
  const size_t D = dims_.d_model, H = dims_.n_heads, HD = D / H, FF = dims_.ffn, NL = dims_.n_layers;
  const float scale = 1.0f / std::sqrt(static_cast<float>(HD));
  const size_t max_scores_stride = kMaxDecodeSteps + 500;

  splitk_pool_->run_layer([&](size_t tid, size_t nth, auto& barrier) {
    size_t phase = 0;
    const size_t k0_d = tid * (D / nth);
    const size_t k1_d = (tid + 1) * (D / nth);
    const size_t k0_ff = tid * (FF / nth);
    const size_t k1_ff = (tid + 1) * (FF / nth);
    const size_t ff0 = k0_ff;
    const size_t ff1 = k1_ff;
    const size_t ff_len = ff1 - ff0;

    const size_t h0 = tid * H / nth;
    const size_t h1 = (tid + 1) * H / nth;
    const size_t q_off = h0 * HD;
    const size_t q_len = (h1 - h0) * HD;
    const size_t k_off = D + q_off;
    const size_t v_off = 2 * D + q_off;

    float* qkv_part = sk_part_qkv_.data() + tid * 3 * D;
    float* wout_part = sk_part_wout_.data() + tid * D;
    float* w1_part = sk_part_w1_.data() + tid * FF;
    float* w2_part = sk_part_w2_.data() + tid * D;
    float* scores = sk_scores_.data() + tid * max_scores_stride;

    for (size_t l = 0; l < NL; ++l) {
      Layer& L = layers_[l];
      uint16_t* k16 = KV16 ? kc16_[l].data() : nullptr;
      uint16_t* v16 = KV16 ? vc16_[l].data() : nullptr;
      float* kf32 = !KV16 ? kc_[l].data() : nullptr;
      float* vf32 = !KV16 ? vc_[l].data() : nullptr;

      if constexpr (GEMV16) {
        kern::f32_to_f16(x + k0_d, sk_xh512_.data() + k0_d, k1_d - k0_d);
        ar_neon::gemv_slice_f16x_fmlal_4rows(L.wqkv16.data(), sk_xh512_.data(),
                                             qkv_part, 3 * D, D, k0_d, k1_d);
      } else {
        ar_neon::gemv_slice_f16w_f32acc_4rows(L.wqkv16.data(), x, qkv_part,
                                              3 * D, D, k0_d, k1_d);
      }

      barrier.sync(phase++);

      ar_neon::reduce_slice(sk_qkv_ptrs_, nth, sk_qkv_.data(), q_off, q_off + q_len);
      for (size_t j = q_off; j < q_off + q_len; ++j) sk_qkv_[j] += L.bqkv[j];

      ar_neon::reduce_slice(sk_qkv_ptrs_, nth, sk_qkv_.data(), k_off, k_off + q_len);
      for (size_t j = k_off; j < k_off + q_len; ++j) sk_qkv_[j] += L.bqkv[j];

      ar_neon::reduce_slice(sk_qkv_ptrs_, nth, sk_qkv_.data(), v_off, v_off + q_len);
      for (size_t j = v_off; j < v_off + q_len; ++j) sk_qkv_[j] += L.bqkv[j];

      if constexpr (KV16) {
        kern::f32_to_f16(sk_qkv_.data() + k_off, k16 + pos * D + q_off, q_len);
        kern::f32_to_f16(sk_qkv_.data() + v_off, v16 + pos * D + q_off, q_len);
      } else {
        std::memcpy(kf32 + pos * D + q_off, sk_qkv_.data() + k_off, q_len * sizeof(float));
        std::memcpy(vf32 + pos * D + q_off, sk_qkv_.data() + v_off, q_len * sizeof(float));
      }

      if (HD == 32) {
        for (size_t h = h0; h < h1; ++h) {
          const float* qv = sk_qkv_.data() + h * HD;
          const float32x4_t q0 = vld1q_f32(qv + 0);
          const float32x4_t q1 = vld1q_f32(qv + 4);
          const float32x4_t q2 = vld1q_f32(qv + 8);
          const float32x4_t q3 = vld1q_f32(qv + 12);
          const float32x4_t q4 = vld1q_f32(qv + 16);
          const float32x4_t q5 = vld1q_f32(qv + 20);
          const float32x4_t q6 = vld1q_f32(qv + 24);
          const float32x4_t q7 = vld1q_f32(qv + 28);

          for (size_t k = 0; k < len; ++k) {
            if constexpr (KV16) {
              const uint16_t* k_ptr = k16 + k * D + h * HD;
              float16x8_t kh0 = vld1q_f16(reinterpret_cast<const __fp16*>(k_ptr + 0));
              float16x8_t kh1 = vld1q_f16(reinterpret_cast<const __fp16*>(k_ptr + 8));
              float16x8_t kh2 = vld1q_f16(reinterpret_cast<const __fp16*>(k_ptr + 16));
              float16x8_t kh3 = vld1q_f16(reinterpret_cast<const __fp16*>(k_ptr + 24));

              float32x4_t a0 = vmulq_f32(q0, vcvt_f32_f16(vget_low_f16(kh0)));
              float32x4_t a1 = vmulq_f32(q1, vcvt_f32_f16(vget_high_f16(kh0)));
              float32x4_t a2 = vmulq_f32(q2, vcvt_f32_f16(vget_low_f16(kh1)));
              float32x4_t a3 = vmulq_f32(q3, vcvt_f32_f16(vget_high_f16(kh1)));
              float32x4_t a4 = vmulq_f32(q4, vcvt_f32_f16(vget_low_f16(kh2)));
              float32x4_t a5 = vmulq_f32(q5, vcvt_f32_f16(vget_high_f16(kh2)));
              float32x4_t a6 = vmulq_f32(q6, vcvt_f32_f16(vget_low_f16(kh3)));
              float32x4_t a7 = vmulq_f32(q7, vcvt_f32_f16(vget_high_f16(kh3)));

              float32x4_t s01 = vaddq_f32(a0, a1);
              float32x4_t s23 = vaddq_f32(a2, a3);
              float32x4_t s45 = vaddq_f32(a4, a5);
              float32x4_t s67 = vaddq_f32(a6, a7);
              float32x4_t s = vaddq_f32(vaddq_f32(s01, s23), vaddq_f32(s45, s67));
              float32x2_t lo = vget_low_f32(s);
              float32x2_t hi = vget_high_f32(s);
              float32x2_t sum2 = vadd_f32(lo, hi);
              float32x2_t sum1 = vpadd_f32(sum2, sum2);
              float dot = vget_lane_f32(sum1, 0);
              scores[k] = dot * scale;
            } else {
              const float* k_ptr = kf32 + k * D + h * HD;
              float32x4_t a0 = vmulq_f32(q0, vld1q_f32(k_ptr + 0));
              float32x4_t a1 = vmulq_f32(q1, vld1q_f32(k_ptr + 4));
              float32x4_t a2 = vmulq_f32(q2, vld1q_f32(k_ptr + 8));
              float32x4_t a3 = vmulq_f32(q3, vld1q_f32(k_ptr + 12));
              float32x4_t a4 = vmulq_f32(q4, vld1q_f32(k_ptr + 16));
              float32x4_t a5 = vmulq_f32(q5, vld1q_f32(k_ptr + 20));
              float32x4_t a6 = vmulq_f32(q6, vld1q_f32(k_ptr + 24));
              float32x4_t a7 = vmulq_f32(q7, vld1q_f32(k_ptr + 28));

              float32x4_t s01 = vaddq_f32(a0, a1);
              float32x4_t s23 = vaddq_f32(a2, a3);
              float32x4_t s45 = vaddq_f32(a4, a5);
              float32x4_t s67 = vaddq_f32(a6, a7);
              float32x4_t s = vaddq_f32(vaddq_f32(s01, s23), vaddq_f32(s45, s67));
              float32x2_t lo = vget_low_f32(s);
              float32x2_t hi = vget_high_f32(s);
              float32x2_t sum2 = vadd_f32(lo, hi);
              float32x2_t sum1 = vpadd_f32(sum2, sum2);
              float dot = vget_lane_f32(sum1, 0);
              scores[k] = dot * scale;
            }
          }

          gsv::kern::softmax(scores, scores, len);

          float32x4_t o0 = vdupq_n_f32(0.f), o1 = vdupq_n_f32(0.f);
          float32x4_t o2 = vdupq_n_f32(0.f), o3 = vdupq_n_f32(0.f);
          float32x4_t o4 = vdupq_n_f32(0.f), o5 = vdupq_n_f32(0.f);
          float32x4_t o6 = vdupq_n_f32(0.f), o7 = vdupq_n_f32(0.f);

          for (size_t k = 0; k < len; ++k) {
            const float32x4_t p4 = vdupq_n_f32(scores[k]);
            if constexpr (KV16) {
              const uint16_t* v_ptr = v16 + k * D + h * HD;
              float16x8_t vh0 = vld1q_f16(reinterpret_cast<const __fp16*>(v_ptr + 0));
              float16x8_t vh1 = vld1q_f16(reinterpret_cast<const __fp16*>(v_ptr + 8));
              float16x8_t vh2 = vld1q_f16(reinterpret_cast<const __fp16*>(v_ptr + 16));
              float16x8_t vh3 = vld1q_f16(reinterpret_cast<const __fp16*>(v_ptr + 24));

              o0 = vfmaq_f32(o0, p4, vcvt_f32_f16(vget_low_f16(vh0)));
              o1 = vfmaq_f32(o1, p4, vcvt_f32_f16(vget_high_f16(vh0)));
              o2 = vfmaq_f32(o2, p4, vcvt_f32_f16(vget_low_f16(vh1)));
              o3 = vfmaq_f32(o3, p4, vcvt_f32_f16(vget_high_f16(vh1)));
              o4 = vfmaq_f32(o4, p4, vcvt_f32_f16(vget_low_f16(vh2)));
              o5 = vfmaq_f32(o5, p4, vcvt_f32_f16(vget_high_f16(vh2)));
              o6 = vfmaq_f32(o6, p4, vcvt_f32_f16(vget_low_f16(vh3)));
              o7 = vfmaq_f32(o7, p4, vcvt_f32_f16(vget_high_f16(vh3)));
            } else {
              const float* v_ptr = vf32 + k * D + h * HD;
              o0 = vfmaq_f32(o0, p4, vld1q_f32(v_ptr + 0));
              o1 = vfmaq_f32(o1, p4, vld1q_f32(v_ptr + 4));
              o2 = vfmaq_f32(o2, p4, vld1q_f32(v_ptr + 8));
              o3 = vfmaq_f32(o3, p4, vld1q_f32(v_ptr + 12));
              o4 = vfmaq_f32(o4, p4, vld1q_f32(v_ptr + 16));
              o5 = vfmaq_f32(o5, p4, vld1q_f32(v_ptr + 20));
              o6 = vfmaq_f32(o6, p4, vld1q_f32(v_ptr + 24));
              o7 = vfmaq_f32(o7, p4, vld1q_f32(v_ptr + 28));
            }
          }

          float* ov = sk_attn_.data() + h * HD;
          vst1q_f32(ov + 0, o0);
          vst1q_f32(ov + 4, o1);
          vst1q_f32(ov + 8, o2);
          vst1q_f32(ov + 12, o3);
          vst1q_f32(ov + 16, o4);
          vst1q_f32(ov + 20, o5);
          vst1q_f32(ov + 24, o6);
          vst1q_f32(ov + 28, o7);

          if constexpr (GEMV16) {
            uint16_t* ov16 = sk_attn16_.data() + h * HD;
            float16x8_t h01 = vcombine_f16(vcvt_f16_f32(o0), vcvt_f16_f32(o1));
            float16x8_t h23 = vcombine_f16(vcvt_f16_f32(o2), vcvt_f16_f32(o3));
            float16x8_t h45 = vcombine_f16(vcvt_f16_f32(o4), vcvt_f16_f32(o5));
            float16x8_t h67 = vcombine_f16(vcvt_f16_f32(o6), vcvt_f16_f32(o7));
            vst1q_u16(ov16 + 0, vreinterpretq_u16_f16(h01));
            vst1q_u16(ov16 + 8, vreinterpretq_u16_f16(h23));
            vst1q_u16(ov16 + 16, vreinterpretq_u16_f16(h45));
            vst1q_u16(ov16 + 24, vreinterpretq_u16_f16(h67));
          }
        }
      } else {
        for (size_t h = h0; h < h1; ++h) {
          const float* qv = sk_qkv_.data() + h * HD;
          for (size_t k = 0; k < len; ++k) {
            float dot;
            if constexpr (KV16) {
              dot = ar_neon::dot_f16kv(qv, k16 + k * D + h * HD, HD);
            } else {
              dot = ar_neon::dot_f32(qv, kf32 + k * D + h * HD, HD);
            }
            scores[k] = dot * scale;
          }
          gsv::kern::softmax(scores, scores, len);
          float* ov = sk_attn_.data() + h * HD;
          std::memset(ov, 0, HD * sizeof(float));
          for (size_t k = 0; k < len; ++k) {
            const float p = scores[k];
            if constexpr (KV16) {
              ar_neon::accum_f16kv(ov, v16 + k * D + h * HD, p, HD);
            } else {
              ar_neon::accum_f32(ov, vf32 + k * D + h * HD, p, HD);
            }
          }
        }

        if constexpr (GEMV16) {
          kern::f32_to_f16(sk_attn_.data() + q_off, sk_attn16_.data() + q_off, q_len);
        }
      }

      barrier.sync(phase++);

      if constexpr (GEMV16) {
        ar_neon::gemv_slice_f16x_fmlal_4rows(L.wout16.data(), sk_attn16_.data(),
                                             wout_part, D, D, k0_d, k1_d);
      } else {
        ar_neon::gemv_slice_f16w_f32acc_4rows(L.wout16.data(), sk_attn_.data(),
                                              wout_part, D, D, k0_d, k1_d);
      }

      barrier.sync(phase++);

      if (tid == 0) {
        float* xb = sk_xb_.data();
        ar_neon::reduce_parts(sk_wout_ptrs_, nth, xb, D);
        for (size_t i = 0; i < D; ++i) xb[i] += L.bout[i] + x[i];
        gsv::kern::layernorm(xb, L.n1g.data(), L.n1b.data(), xb, D, dims_.ln_eps);
        std::memcpy(sk_hbuf_.data(), xb, D * sizeof(float));
        if constexpr (GEMV16) {
          kern::f32_to_f16(xb, sk_xb16_.data(), D);
        }
      }

      barrier.sync(phase++);

      if constexpr (GEMV16) {
        ar_neon::gemv_slice_f16x_fmlal_4rows(L.w116.data(), sk_xb16_.data(),
                                             w1_part, FF, D, k0_d, k1_d);
      } else {
        ar_neon::gemv_slice_f16w_f32acc_4rows(L.w116.data(), sk_xb_.data(),
                                              w1_part, FF, D, k0_d, k1_d);
      }

      barrier.sync(phase++);

      ar_neon::reduce_slice(sk_w1_ptrs_, nth, sk_ff_.data(), ff0, ff1);
      for (size_t i = ff0; i < ff1; ++i) sk_ff_[i] += L.b1[i];
      gsv::kern::relu(sk_ff_.data() + ff0, sk_ff_.data() + ff0, ff_len);
      if constexpr (GEMV16) {
        kern::f32_to_f16(sk_ff_.data() + ff0, sk_ff16_.data() + ff0, ff_len);
      }

      barrier.sync(phase++);

      if constexpr (GEMV16) {
        ar_neon::gemv_slice_f16x_fmlal_4rows(L.w216.data(), sk_ff16_.data(),
                                             w2_part, D, FF, k0_ff, k1_ff);
      } else {
        ar_neon::gemv_slice_f16w_f32acc_4rows(L.w216.data(), sk_ff_.data(),
                                              w2_part, D, FF, k0_ff, k1_ff);
      }

      barrier.sync(phase++);

      if (tid == 0) {
        ar_neon::reduce_parts(sk_w2_ptrs_, nth, x, D);
        for (size_t i = 0; i < D; ++i) x[i] += L.b2[i] + sk_hbuf_[i];
        gsv::kern::layernorm(x, L.n2g.data(), L.n2b.data(), x, D, dims_.ln_eps);
      }

      if (l + 1 < NL) {
        barrier.sync(phase++);
      }
    }
  });

  predict_layer_fp(x, logits_out);
}

template void T2SEngine::block_prefill_impl<false>(size_t, float*, size_t,
                                                   size_t, size_t, float*,
                                                   uint16_t*, float*, uint16_t*);
template void T2SEngine::block_prefill_impl<true>(size_t, float*, size_t,
                                                  size_t, size_t, float*,
                                                  uint16_t*, float*, uint16_t*);
template void T2SEngine::block_decode_impl<false, false>(size_t, float*,
                                                         size_t, size_t,
                                                         float*, uint16_t*,
                                                         float*, uint16_t*);
template void T2SEngine::block_decode_impl<true, true>(size_t, float*, size_t,
                                                       size_t, float*,
                                                       uint16_t*, float*,
                                                       uint16_t*);
template void T2SEngine::block_decode_splitk_impl<false, false>(size_t, float*,
                                                                size_t, size_t,
                                                                float*, uint16_t*,
                                                                float*, uint16_t*);
template void T2SEngine::block_decode_splitk_impl<true, true>(size_t, float*,
                                                              size_t, size_t,
                                                              float*, uint16_t*,
                                                              float*, uint16_t*);
template void T2SEngine::decode_step_splitk_impl<false, false>(float*, size_t, size_t, float*);
template void T2SEngine::decode_step_splitk_impl<true, true>(float*, size_t, size_t, float*);

// ---- 贪心采样(infer_panel_naive sample() top_k=1 同构) ----
// 关键语义: torch 的 logits_to_probs 用 scatter_ 就地修改传入的 logits 张量,
// 而 golden 导出 hook 持有的是同一存储的 detached 视图 ⇒ .pt 里存的是"惩罚后"
// 的 logits。故本函数对 logits_io 就地施压, raw_argmax 与记录值都取惩罚后状态:
//   - eos_allowed=false(idx<11): torch 先切掉 EOS 列再 scatter_, EOS 列保持原始值;
//     raw_argmax 仍是全词表 argmax(golden `tokens` 口径)
int T2SEngine::greedy_sample(float* logits_io,
                             const std::vector<int32_t>& history,
                             bool eos_allowed, int* raw_argmax_out) {
  const int V = static_cast<int>(dims_.vocab);
  const int limit = eos_allowed ? V : V - 1;  // idx<11 时 EOS 列不受惩罚
  // repetition_penalty=1.35, 正值除/负值乘。
  // 关键语义: torch 先 gather 全部历史出现次数的"原始值"再统一 scatter_ 写回,
  // 重复 token 的多次写互相覆盖 ⇒ 每个"去重后"的 token 只施压一次(非逐次累乘)。
  if (pen_mark_.size() != dims_.vocab) {
    pen_mark_.assign(dims_.vocab, 0);
    pen_stamp_ = 0;
  }
  ++pen_stamp_;
  for (const int32_t c : history) {
    if (static_cast<int>(c) >= limit) continue;
    if (pen_mark_[static_cast<size_t>(c)] == pen_stamp_) continue;
    pen_mark_[static_cast<size_t>(c)] = pen_stamp_;
    float& v = logits_io[static_cast<size_t>(c)];
    v = v < 0.f ? v * kRepPenalty : v / kRepPenalty;
  }
  // 全词表 argmax(golden `tokens` 口径, 含保持原始值的 EOS 列)
  if (raw_argmax_out) {
    int best = 0;
    float bv = logits_io[0];
    for (int c = 1; c < V; ++c) {
      if (logits_io[c] > bv) { bv = logits_io[c]; best = c; }
    }
    *raw_argmax_out = best;
  }
  // 采样 argmax(top_k=1 ⇒ one-hot; idx<11 时剔除 EOS 列 —— torch logits[:, :-1])
  int best = 0;
  float bv = logits_io[0];
  for (int c = 1; c < limit; ++c) {
    if (logits_io[c] > bv) { bv = logits_io[c]; best = c; }
  }
  return best;
}

// ---- E4: topk_sample —— 对齐 python infer_panel_naive sample() ----
// 链顺序: repetition_penalty → top_k(15) → top_p(1.0) → temperature(1.0) → multinomial
// 关键语义:
//  - 惩罚对"去重后"history 各施压一次(torch scatter_ 覆盖语义, 同 greedy_sample)
//  - idx<11 ⇒ EOS 列保持原始值且不参与采样(torch logits[:, :-1] 前置裁切)
//  - temperature=1.0 时 softmax 等价直接归一化; 非 1.0 时 logits /= temperature
//  - multinomial 用浮点 CDF 累加 + uniform 比较(1025 词表无需别名表, < 5us 一次)
//  - seed 来自 SamplingParams(0 = std::random_device 真随机)
int T2SEngine::topk_sample(float* logits_io,
                           const std::vector<int32_t>& history,
                           bool eos_allowed, int* raw_argmax_out,
                           const SamplingParams& sp, std::mt19937_64& rng) {
  const int V = static_cast<int>(dims_.vocab);
  const int limit = eos_allowed ? V : V - 1;
  // 1) repetition_penalty (同 greedy_sample, 标量化 rep_penalty 入参)
  if (pen_mark_.size() != dims_.vocab) {
    pen_mark_.assign(dims_.vocab, 0);
    pen_stamp_ = 0;
  }
  ++pen_stamp_;
  for (const int32_t c : history) {
    if (static_cast<int>(c) >= limit) continue;
    if (pen_mark_[static_cast<size_t>(c)] == pen_stamp_) continue;
    pen_mark_[static_cast<size_t>(c)] = pen_stamp_;
    float& v = logits_io[static_cast<size_t>(c)];
    v = v < 0.f ? v * sp.rep_penalty : v / sp.rep_penalty;
  }
  // 2) raw_argmax: 全词表 argmax(含 EOS, 保持原始值)
  if (raw_argmax_out) {
    int best = 0;
    float bv = logits_io[0];
    for (int c = 1; c < V; ++c) {
      if (logits_io[c] > bv) { bv = logits_io[c]; best = c; }
    }
    *raw_argmax_out = best;
  }
  // 3) top_k: 在 [0, limit) 范围中保留 sp.top_k 个最高(此为样本有效域)
  //    策略: 在 limit 范围内收集 top_k (insertion-select, k≤15 1025 上 O(k·V) 仍 < 20us)
  const int k = sp.top_k > static_cast<size_t>(limit)
                    ? limit
                    : static_cast<int>(sp.top_k);
  // 简化: 使用部分排序后的索引表 (V=1025, 1025*4B = 4KB, 贴进 L1)
  // 为避免动态分配, 复用 per-instance topk_idx_/topk_val_ (在 generate() 期用, 尺寸 V)
  int* idxs = topk_idx_.data();
  float* vals = topk_val_.data();
  for (int c = 0; c < limit; ++c) { idxs[c] = c; vals[c] = logits_io[c]; }
  // partial sort top-k (selection sort k 轮, k≤15, V=1025 ⇒ 15*1025 = 15k 比较)
  for (int i = 0; i < k; ++i) {
    int max_j = i;
    float max_v = vals[i];
    for (int j = i + 1; j < limit; ++j) {
      if (vals[j] > max_v) { max_v = vals[j]; max_j = j; }
    }
    if (max_j != i) {
      std::swap(vals[i], vals[max_j]);
      std::swap(idxs[i], idxs[max_j]);
    }
  }
  // 4) temperature 应用到 top-k logits(若 != 1.0)
  //    temperature=0 ⇒ 退贪心
  if (sp.temperature <= 0.f) {
    int best = idxs[0];
    float bv = vals[0];
    for (int i = 1; i < k; ++i) {
      if (vals[i] > bv) { bv = vals[i]; best = idxs[i]; }
    }
    return best;
  }
  if (sp.temperature != 1.0f) {
    for (int i = 0; i < k; ++i) vals[i] = vals[i] / sp.temperature;
  }
  // 5) softmax over top-k (数值稳定: 减 max)
  float max_logit = vals[0];
  for (int i = 1; i < k; ++i) {
    if (vals[i] > max_logit) max_logit = vals[i];
  }
  float sum_exp = 0.f;
  for (int i = 0; i < k; ++i) {
    vals[i] = std::exp(vals[i] - max_logit);
    sum_exp += vals[i];
  }
  // 6) top_p (nucleus): 从高到低累加, 保留累积 ≥ (1 - top_p) 比例的最小前缀
  int keep = k;
  if (sp.top_p < 1.0f) {
    const float threshold = (1.f - sp.top_p) * sum_exp;
    float cum = 0;
    for (int i = k - 1; i >= 0; --i) {
      cum += vals[i];
      if (cum >= threshold) { keep = i + 1; break; }
    }
    // 重算 sum_exp (裁掉后部分)
    if (keep < k) {
      sum_exp = 0;
      for (int i = 0; i < keep; ++i) sum_exp += vals[i];
    }
  }
  // 7) multinomial: 浮点 CDF + uniform 比较
  //    1025 词表, keep≤15, 线性扫描 < 100ns
  std::uniform_real_distribution<float> uni(0.f, 1.f);
  const float u = uni(rng) * sum_exp;
  float cdf = 0.f;
  for (int i = 0; i < keep; ++i) {
    cdf += vals[i];
    if (u <= cdf) return idxs[i];
  }
  return idxs[keep - 1];  // 数值末位兜底
}

GenResult T2SEngine::generate(const int64_t* phones, size_t T,
                              const int64_t* prompt, size_t P,
                              const float* bert1024, size_t max_steps,
                              GenDebug* dbg, const SamplingParams* sampling) {
  const auto t_enter = std::chrono::steady_clock::now();
  const size_t D = dims_.d_model;
  if (T == 0 || P == 0) throw std::runtime_error("T/P 必须非零(golden 口径)");
  const size_t S = T + P;

  // ---- scratch/cache 容量(跨调用复用; 模式切换或容量不足时重建) ----
  if (cap_ < S + max_steps || fp16_.kv != kv_mode_active_) {
    cap_ = S + max_steps;
    kv_mode_active_ = fp16_.kv;
    kc_.assign(dims_.n_layers, {});
    vc_.assign(dims_.n_layers, {});
    kc16_.assign(dims_.n_layers, {});
    vc16_.assign(dims_.n_layers, {});
    for (size_t l = 0; l < dims_.n_layers; ++l) {
      if (fp16_.kv) {
        kc16_[l].resize(cap_ * D);
        vc16_[l].resize(cap_ * D);
      } else {
        kc_[l].resize(cap_ * D);
        vc_[l].resize(cap_ * D);
      }
    }
  }

  GenResult r;

  // ---- 输入重建 xy[S,D]: 文本 emb+bert_proj+PE | 音频 prompt emb+PE ----
  xy_.resize(S * D);
  pe_.resize(D);
  bert_proj_.forward(bert1024, T, xy_.data());  // bert_proj(bert_feat) 先落 xy
  for (size_t t = 0; t < T; ++t) {
    const float* erow = text_emb_.data() + static_cast<size_t>(phones[t]) * D;
    float* xr = xy_.data() + t * D;
    for (size_t d = 0; d < D; ++d) xr[d] += erow[d] + bert_b_[d];
    pe_row(pe_.data(), t);
    for (size_t d = 0; d < D; ++d) xr[d] += alpha_text_ * pe_[d];
  }
  for (size_t t = 0; t < P; ++t) {
    const float* erow = audio_emb_.data() + static_cast<size_t>(prompt[t]) * D;
    float* xr = xy_.data() + (T + t) * D;
    pe_row(pe_.data(), t);
    for (size_t d = 0; d < D; ++d) xr[d] = erow[d] + alpha_audio_ * pe_[d];
  }
  const auto t_prep_done = std::chrono::steady_clock::now();

  // ---- B1: prefill(24 层, 大矩阵乘固定 Accelerate sgemm) ----
  if (dbg) dbg->on_input(xy_.data(), S);
  pf_wqkv_ms_ = 0; pf_sdpa_ms_ = 0; pf_wout_ms_ = 0;
  pf_w1_ms_ = 0; pf_w2_ms_ = 0; pf_ln_ms_ = 0;
  for (size_t l = 0; l < dims_.n_layers; ++l) {
    if (fp16_.kv)
      block_prefill_impl<true>(l, xy_.data(), S, /*pos=*/0, /*text_len=*/T,
                               nullptr, kc16_[l].data(), nullptr,
                               vc16_[l].data());
    else
      block_prefill_impl<false>(l, xy_.data(), S, /*pos=*/0, /*text_len=*/T,
                                kc_[l].data(), nullptr, vc_[l].data(),
                                nullptr);
    if (dbg) dbg->on_layer(l, xy_.data(), S);
  }

  // 首 logits 来自最后一个音频 prompt 位置(xy_dec[:, -1])
  logits_.resize(dims_.vocab);
  predict_layer_fp(xy_.data() + (S - 1) * D, logits_.data());
  const auto t_prefill_done = std::chrono::steady_clock::now();

  // ---- B2: decode 循环(GEMV + KV cache fp32 + 贪心) ----
  // E4: 采样路径 —— sampling.mode=TopK 时走 topk_sample (复现 python 默认 15 采样, 根治复读)
  // 默认 Greedy 不变(位级一致保证 B12 golden G1/G2)
  const bool use_topk = sampling && sampling->mode == SamplingParams::Mode::TopK;
  std::mt19937_64 rng = [&]() {
    if (sampling && sampling->seed != 0) return std::mt19937_64(sampling->seed);
    std::random_device rd;
    return std::mt19937_64(rd());
  }();
  if (use_topk) {
    topk_idx_.assign(dims_.vocab, 0);
    topk_val_.assign(dims_.vocab, 0.f);
  }
  std::vector<int32_t> history;
  history.reserve(P + max_steps);
  for (size_t t = 0; t < P; ++t) history.push_back(static_cast<int32_t>(prompt[t]));
  r.logits_first8.reserve(8 * dims_.vocab);
  r.sampled.reserve(max_steps);
  r.raw_argmax.reserve(max_steps);

  size_t cur_len = S;  // 已写入 cache 的 token 数
  bool stop = false;
  size_t idx = 0;
  if (ar_splitk_) {
    if (!splitk_pool_) set_splitk(true);
    if (splitk_pool_) splitk_pool_->begin_session();
  }
  struct rusage ru0{}, ru1{};
  ::getrusage(RUSAGE_SELF, &ru0);
  for (; idx < max_steps && !stop; ++idx) {
    float* lg = logits_.data();
    int raw = -1;
    const bool eos_allowed = idx >= kEosMaskSteps;
    // 采样 + 就地惩罚(torch scatter_ 语义): 惩罚后 logits_ 才是 golden 捕获口径
    int sample;
    if (use_topk) {
      sample = topk_sample(lg, history, eos_allowed, &raw, *sampling, rng);
    } else {
      sample = greedy_sample(lg, history, eos_allowed, &raw);
    }
    r.raw_argmax.push_back(raw);
    if (r.logits_first8.size() < 8 * dims_.vocab)
      r.logits_first8.insert(r.logits_first8.end(), lg, lg + dims_.vocab);

    if (eos_allowed && sample == dims_.eos) {  // EOS 触发停止且不入序列
      stop = true;
      r.hit_eos = true;
    } else {
      r.sampled.push_back(sample);
      history.push_back(sample);
    }
    if (idx + 1 == max_steps) stop = true;  // 步数保护(torch idx==MAX-1)

    if (stop) break;

    // 下一步输入 = emb_audio(sample) + alpha_audio·PE(P+idx)
    x1_.resize(D);
    const float* erow = audio_emb_.data() + static_cast<size_t>(sample) * D;
    pe_row(pe_.data(), P + idx);
    for (size_t d = 0; d < D; ++d) x1_[d] = erow[d] + alpha_audio_ * pe_[d];

    if (ar_splitk_) {
      if (fp16_.kv && fp16_.gemv) {
        decode_step_splitk_impl<true, true>(x1_.data(), cur_len, cur_len + 1,
                                            logits_.data());
      } else {
        decode_step_splitk_impl<false, false>(x1_.data(), cur_len, cur_len + 1,
                                              logits_.data());
      }
    } else {
      for (size_t l = 0; l < dims_.n_layers; ++l) {
        if (fp16_.kv && fp16_.gemv) {
          block_decode_impl<true, true>(l, x1_.data(), cur_len, cur_len + 1,
                                        nullptr, kc16_[l].data(), nullptr,
                                        vc16_[l].data());
        } else {
          block_decode_impl<false, false>(l, x1_.data(), cur_len, cur_len + 1,
                                          kc_[l].data(), nullptr,
                                          vc_[l].data(), nullptr);
        }
      }
      predict_layer_fp(x1_.data(), logits_.data());
    }
    ++cur_len;
  }
  if (ar_splitk_ && splitk_pool_) splitk_pool_->end_session();
  const auto t2 = std::chrono::steady_clock::now();
  ::getrusage(RUSAGE_SELF, &ru1);

  r.steps = idx + 1;  // 含触发停止的当前步(hook 计数口径)
  r.logits_last.assign(logits_.data(), logits_.data() + dims_.vocab);
  r.prep_ms = std::chrono::duration<double, std::milli>(t_prep_done - t_enter).count();
  r.prefill_ms = std::chrono::duration<double, std::milli>(t_prefill_done - t_prep_done).count();
  r.decode_ms = std::chrono::duration<double, std::milli>(t2 - t_prefill_done).count();
  last_prefill_ms_ = r.prefill_ms;
  last_decode_ms_ = r.decode_ms;
  // E11-2/E18 GSV_AR_TIMING=1 探针
  if (std::getenv("GSV_AR_TIMING")) {
    const double ms_per_tok =
        last_decode_ms_ / static_cast<double>(r.steps);
    const double user_s = (ru1.ru_utime.tv_sec - ru0.ru_utime.tv_sec) +
                          (ru1.ru_utime.tv_usec - ru0.ru_utime.tv_usec) * 1e-6;
    const double sys_s = (ru1.ru_stime.tv_sec - ru0.ru_stime.tv_sec) +
                         (ru1.ru_stime.tv_usec - ru0.ru_stime.tv_usec) * 1e-6;
    const double cpu_s = user_s + sys_s;
    const double wall_s = last_decode_ms_ * 1e-3;
    const double avg_cores = wall_s > 0 ? (cpu_s / wall_s) : 0.0;
    std::fprintf(stderr,
                 "[ar-timing] T=%zu P=%zu S=%zu steps=%zu prefill=%.2fms (wqkv=%.1f sdpa=%.1f wout=%.1f w1=%.1f w2=%.1f ln=%.1f) "
                 "decode=%.2fms (%.3f ms/tok, CPU=%.2fs/%.2fs wall, avg %.1f cores, splitk=%d)\n",
                 T, P, S, r.steps, last_prefill_ms_, pf_wqkv_ms_, pf_sdpa_ms_, pf_wout_ms_,
                 pf_w1_ms_, pf_w2_ms_, pf_ln_ms_,
                 last_decode_ms_, ms_per_tok, cpu_s, wall_s, avg_cores,
                 ar_splitk_ ? 1 : 0);
  }
  return r;
}

}  // namespace gsv::ar
