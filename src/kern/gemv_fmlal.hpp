// gemv_fmlal.hpp — M1-fp16 新增内核变体声明(task 卡授权的 src/kern/ 新文件)
//
// FMLAL 扩展精度累加 GEMV: 权重与激活均为 fp16 位型, 乘积经 FMLAL 直接
// 融合进 fp32 累加器(单点积无中间舍入; fp16×fp16 积在 fp32 内精确)。
// 与 gemv_f16w_f32acc 的数值差异仅在于「激活先舍入到 fp16」—— 这是
// ARCHITECTURE §3 两步走第二步的设计噪声来源。
#pragma once

#include <cstddef>
#include <stdint.h>

namespace gsv::kern {

// fp32 → fp16 位型批量转换(NEON vcvth_f16, 就近舍入)
void f32_to_f16(const float* src, uint16_t* dst, size_t n);

// 热路径 GEMV: y[out] = W[out,in]·xh[in]; W/xh 均 fp16 raw(row-major),
// FMLAL(fp16×fp16→fp32) 累加, 累加器与输出 fp32。in 建议为 8 的倍数(尾部标量兜底)。
void gemv_f16x_fmlal(const uint16_t* w, const uint16_t* xh, float* y,
                     size_t out, size_t in);

}  // namespace gsv::kern
