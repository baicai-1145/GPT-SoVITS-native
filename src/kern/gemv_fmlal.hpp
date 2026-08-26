// gemv_fmlal.hpp — M1-fp16 新增内核变体声明(task 卡授权的 src/kern/ 新文件)
//
// FMLAL 扩展精度累加 GEMV: 权重与激活均为 fp16 位型, 乘积经 FMLAL 直接
// 融合进 fp32 累加器(单点积无中间舍入; fp16×fp16 积在 fp32 内精确)。
// 与 gemv_f16w_f32acc 的数值差异仅在于「激活先舍入到 fp16」—— 这是
// ARCHITECTURE §3 两步走第二步的设计噪声来源。
#pragma once

#include <arm_neon.h>
#include <cstddef>
#include <stdint.h>

namespace gsv::kern {

// fp32 → fp16 位型批量转换(NEON vcvth_f16, 就近舍入)
void f32_to_f16(const float* src, uint16_t* dst, size_t n);

// E8: 单点 fp32 → fp16 位型转换(vcvth_f16, 就近舍入; 与批量版同位级)
inline uint16_t f32_to_f16_scalar(float v) {
  const float16x4_t h = vcvt_f16_f32(vdupq_n_f32(v));
  return vget_lane_u16(vreinterpret_u16_f16(h), 0);
}

// 热路径 GEMV: y[out] = W[out,in]·xh[in]; W/xh 均 fp16 raw(row-major),
// FMLAL(fp16×fp16→fp32) 累加, 累加器与输出 fp32。in 建议为 8 的倍数(尾部标量兜底)。
void gemv_f16x_fmlal(const uint16_t* w, const uint16_t* xh, float* y,
                     size_t out, size_t in);

// ---- E2-ENC: 矩阵版 FMLAL GEMM ----
// C[M,N] = Σ_k A[m,k]·B[n,k]  (即 C = A·Bᵀ; A/B 均 [行,K] fp16 行主, C fp32 行主)
// 覆盖两类引擎惯用形:
//   全连接:  y[T,out] = x[T,in]·W[out,in]ᵀ   (A=x, B=W)
//   卷积:    y[Co,S] = W[Co,K]·cols[S,K]ᵀ    (A=W, B=im2col)
// 数值语义与 gemv_f16x_fmlal 同构: 激活/权重均 fp16 位型, 单积 fp16×fp16 精确
// 落 fp32(FMLAL 无中间舍入), 累加链纯 fp32。4×4 tile 寄存器累加 + P 核行带并行。
void gemm_f16x_fmlal(const uint16_t* a, const uint16_t* b, float* c,
                     size_t M, size_t N, size_t K);

// SV 通道主激活 [R,S] fp32 → 转置+量化为 [S,R] fp16(旓nn 路径适配 nt 内核用)
void f32_trans_to_f16(const float* src, uint16_t* dst, size_t R, size_t S);

}  // namespace gsv::kern
