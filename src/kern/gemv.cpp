// gemv.cpp — NEON 手写 GEMV: fp16 权重精确升位 + fp32 FMA 累加
//
// 路线选择(任务卡允许 FMLAL 或 dequant 二选一): 采用 **dequant 路径** ——
// FMLAL 要求两个操作数都是 fp16, 激活必须先舍入到 fp16, 违反"激活保持 fp32"纪律;
// vcvt_f32_f16(fp16→fp32) 是无损升位, 之后 vfmaq_f32 全程 fp32, 数值语义与 torch
// "fp16存储+fp32计算" 基线完全同构。
#include "kern.hpp"

#include <arm_neon.h>

namespace gsv::kern {

void gemv_f16w_f32acc_ref(const uint16_t* w, const float* x, float* y, size_t out, size_t in) {
  for (size_t r = 0; r < out; ++r) {
    const uint16_t* wr = w + r * in;
    float s = 0.0f;
    for (size_t i = 0; i < in; ++i) {
      __fp16 h;
      __builtin_memcpy(&h, &wr[i], sizeof h);
      s += static_cast<float>(h) * x[i];
    }
    y[r] = s;
  }
}

void gemv_f16w_f32acc(const uint16_t* w, const float* x, float* y, size_t out, size_t in) {
  const size_t vec_end = in & ~size_t{7};  // 8 元素一组(16B fp16 / 32B fp32)
  for (size_t r = 0; r < out; ++r) {
    const uint16_t* wr = w + r * in;
    const float* xr = x;
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i < vec_end; i += 8, wr += 8, xr += 8) {
      const float16x8_t w8 = vld1q_f16(reinterpret_cast<const __fp16*>(wr));
      const float32x4_t wlo = vcvt_f32_f16(vget_low_f16(w8));   // 无损升位
      const float32x4_t whi = vcvt_f32_f16(vget_high_f16(w8));
      acc0 = vfmaq_f32(acc0, wlo, vld1q_f32(xr));
      acc1 = vfmaq_f32(acc1, whi, vld1q_f32(xr + 4));
    }
    float s = vaddvq_f32(acc0) + vaddvq_f32(acc1);
    for (; i < in; ++i, ++wr, ++xr) {  // 尾部标量(同语义)
      __fp16 h;
      __builtin_memcpy(&h, wr, sizeof h);
      s += static_cast<float>(h) * (*xr);
    }
    y[r] = s;
  }
}

}  // namespace gsv::kern
