// gemv_fmlal.cpp — M1-fp16: FMLAL 扩展精度累加 GEMV(NEON FEAT_FMLAL, 多核)
//
// 数值语义:
//   - 激活 x 由调用方一次性舍入到 fp16(xh), 权重本就是 fp16 位型
//   - vfmlalq_low/high_f16: 8 个 fp16×fp16 积直接融合进 fp32 累加器,
//     单积无中间舍入(fp16×fp16 精确落在 fp32), 累加链为纯 fp32 长链
//   - 输出 y fp32
// 与 cvt+FMA 变体(kern::gemv_f16w_f32acc)的差异仅「激活 fp16 舍入」一处。
//
// 性能设计(decode GEMV 带宽受限):
//   - 行间并行(parallel_for P 核池), 行内 8×f16x8 累加器 ILP + 提前取下一行
//   - fp16 直读权重流量减半 → 目标吞吐 ~2× DenseF16(sgemm 升位) 路径
#include "kern/gemv_fmlal.hpp"

#include "runtime/threadpool.hpp"

#include <arm_neon.h>
#include <algorithm>
#include <cstring>

namespace gsv::kern {

void f32_to_f16(const float* src, uint16_t* dst, size_t n) {
  size_t i = 0;
  for (; i + 8 <= n; i += 8, src += 8, dst += 8) {
    const float32x4_t lo = vld1q_f32(src);
    const float32x4_t hi = vld1q_f32(src + 4);
    vst1q_f16(reinterpret_cast<__fp16*>(dst),
              vcombine_f16(vcvt_f16_f32(lo), vcvt_f16_f32(hi)));
  }
  for (; i < n; ++i, ++src, ++dst) {
    const __fp16 h = static_cast<__fp16>(*src);
    std::memcpy(dst, &h, sizeof h);
  }
}

namespace {

// 单行 FMLAL 点积: 返回 sum(w[row]·xh)。主循环 64 元素/迭代 × 8 累加器。
inline float row_dot_fmlal(const uint16_t* wr, const uint16_t* xh,
                           size_t in) {
  const size_t vec_end = in & ~size_t{63};
  float32x4_t a0 = vdupq_n_f32(0.f), a1 = vdupq_n_f32(0.f);
  float32x4_t a2 = vdupq_n_f32(0.f), a3 = vdupq_n_f32(0.f);
  float32x4_t a4 = vdupq_n_f32(0.f), a5 = vdupq_n_f32(0.f);
  float32x4_t a6 = vdupq_n_f32(0.f), a7 = vdupq_n_f32(0.f);
  size_t i = 0;
  const uint16_t* next = wr + in;  // 预取下一行首(软件流水提示)
  __builtin_prefetch(next, 0, 0);
  for (; i < vec_end; i += 64, wr += 64) {
    __builtin_prefetch(wr + 128, 0, 0);
    const float16x8_t x0 = vld1q_f16(reinterpret_cast<const __fp16*>(xh + i));
    const float16x8_t x1 =
        vld1q_f16(reinterpret_cast<const __fp16*>(xh + i + 8));
    const float16x8_t x2 =
        vld1q_f16(reinterpret_cast<const __fp16*>(xh + i + 16));
    const float16x8_t x3 =
        vld1q_f16(reinterpret_cast<const __fp16*>(xh + i + 24));
    const float16x8_t x4 =
        vld1q_f16(reinterpret_cast<const __fp16*>(xh + i + 32));
    const float16x8_t x5 =
        vld1q_f16(reinterpret_cast<const __fp16*>(xh + i + 40));
    const float16x8_t x6 =
        vld1q_f16(reinterpret_cast<const __fp16*>(xh + i + 48));
    const float16x8_t x7 =
        vld1q_f16(reinterpret_cast<const __fp16*>(xh + i + 56));
    const float16x8_t w0 = vld1q_f16(reinterpret_cast<const __fp16*>(wr));
    const float16x8_t w1 =
        vld1q_f16(reinterpret_cast<const __fp16*>(wr + 8));
    const float16x8_t w2 =
        vld1q_f16(reinterpret_cast<const __fp16*>(wr + 16));
    const float16x8_t w3 =
        vld1q_f16(reinterpret_cast<const __fp16*>(wr + 24));
    const float16x8_t w4 =
        vld1q_f16(reinterpret_cast<const __fp16*>(wr + 32));
    const float16x8_t w5 =
        vld1q_f16(reinterpret_cast<const __fp16*>(wr + 40));
    const float16x8_t w6 =
        vld1q_f16(reinterpret_cast<const __fp16*>(wr + 48));
    const float16x8_t w7 =
        vld1q_f16(reinterpret_cast<const __fp16*>(wr + 56));
    a0 = vfmlalq_low_f16(a0, w0, x0);
    a0 = vfmlalq_high_f16(a0, w0, x0);
    a1 = vfmlalq_low_f16(a1, w1, x1);
    a1 = vfmlalq_high_f16(a1, w1, x1);
    a2 = vfmlalq_low_f16(a2, w2, x2);
    a2 = vfmlalq_high_f16(a2, w2, x2);
    a3 = vfmlalq_low_f16(a3, w3, x3);
    a3 = vfmlalq_high_f16(a3, w3, x3);
    a4 = vfmlalq_low_f16(a4, w4, x4);
    a4 = vfmlalq_high_f16(a4, w4, x4);
    a5 = vfmlalq_low_f16(a5, w5, x5);
    a5 = vfmlalq_high_f16(a5, w5, x5);
    a6 = vfmlalq_low_f16(a6, w6, x6);
    a6 = vfmlalq_high_f16(a6, w6, x6);
    a7 = vfmlalq_low_f16(a7, w7, x7);
    a7 = vfmlalq_high_f16(a7, w7, x7);
  }
  float s = vaddvq_f32(a0) + vaddvq_f32(a1) + vaddvq_f32(a2) +
            vaddvq_f32(a3) + vaddvq_f32(a4) + vaddvq_f32(a5) +
            vaddvq_f32(a6) + vaddvq_f32(a7);
  for (; i < in; ++i, ++wr) {  // 尾部标量(同语义: fp16×fp16→fp32 精确)
    __fp16 wh, xhh;
    std::memcpy(&wh, wr, sizeof wh);
    std::memcpy(&xhh, xh + i, sizeof xhh);
    s += static_cast<float>(wh) * static_cast<float>(xhh);
  }
  return s;
}

}  // namespace

void gemv_f16x_fmlal(const uint16_t* w, const uint16_t* xh, float* y,
                     size_t out, size_t in) {
  // 小矩阵单线程免池开销; 大矩阵按行块并行(P 核池, 实时链路 QoS)
  if (out * in < 1u << 18) {
    for (size_t r = 0; r < out; ++r)
      y[r] = row_dot_fmlal(w + r * in, xh, in);
    return;
  }
  // 行块数 ≈ P 核数(粗粒度, 免逐块派发开销); 带宽受限负载无需细粒度均衡
  const size_t workers = std::max(rt::p_core_count(), size_t{1});
  const size_t grain = (out + workers - 1) / workers;
  rt::parallel_for(
      out, grain, [&](size_t b, size_t e) {
        for (size_t r = b; r < e; ++r)
          y[r] = row_dot_fmlal(w + r * in, xh, in);
      }, rt::Qos::UserInitiated);
}

}  // namespace gsv::kern
