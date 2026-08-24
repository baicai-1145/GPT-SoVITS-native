// amx.h — Apple AMX 矩阵协处理器指令内联编码 (E5)
//
// 指令编码出处: corsix/amx 逆向工程 (github.com/corsix/amx), M4 上验证。
// 全部为 A64 保留空间 (0x201000..) 的未文档化指令; Apple 未公开承诺兼容,
// 故整个后端默认编译期关闭 (GSV_AMX_GEMM), 运行时另有 SIGILL 探测兜底。
//
// 编程模型 (仅列本后端用到的子集):
//   * X/Y 寄存器池: 8 × 64B (512B 可寻址, 指令操作数内 9-bit 字节偏移)
//   * Z 寄存器池: 64 × 64B
//   * LDX/LDY: 64B 内存 → X/Y 寄存器 (操作数高 8-bit 编码寄存器索引)
//   * LDZ/STZ: Z 行 ↔ 64B 内存
//   * MATFP mode3 (操作数 bit42=3): f16 X ⊗ f16 Y 外积 → f32 Z 累加
//       z[j][i] += x[i] * y[j]   (i=0..31 X lane, j=0..31 Y lane)
//     Z 布局 (混合 lane 宽度, corsix/amx RegisterFile.md):
//       C[m][n] ← Z[(n<<1)|(m&1)].f32[m>>1]
//   * AMX_SET: 每线程使能+清零全部 AMX 寄存器; 已 SET 再 SET → SIGILL
//   * AMX_CLR: 释放状态
//
// ⚠ 同线程互斥: Accelerate BLAS 内部也用 AMX。同一线程"我们的 SET 状态"与
//   BLAS 的 SET/状态管理冲突 (实测: 先 SET 再 cblas → SIGILL; 先 cblas 再
//   我们的路径 → 结果全零)。纪律: AMX 只在自有 worker 线程使用, 这些线程
//   禁止调用任何 Accelerate 函数 (本仓库线程池满足: cblas 只在主线程跑)。
#pragma once

#include <stdint.h>

#define AMX_OP_GPR(op, gpr)                                                   \
  __asm(".word (0x201000 + (%0 << 5) + 0%1 - ((0%1 >> 4) * 6))"               \
        :                                                                     \
        : "i"(op), "r"((uint64_t)(gpr))                                       \
        : "memory")

#define AMX_NOP_OP_IMM5(op, imm5)                                             \
  __asm("nop\nnop\nnop\n.word (0x201000 + (%0 << 5) + %1)"                    \
        :                                                                     \
        : "i"(op), "i"(imm5)                                                  \
        : "memory")

#define AMX_LDX(gpr) AMX_OP_GPR(0, gpr)
#define AMX_LDY(gpr) AMX_OP_GPR(1, gpr)
#define AMX_STX(gpr) AMX_OP_GPR(2, gpr)
#define AMX_STY(gpr) AMX_OP_GPR(3, gpr)
#define AMX_LDZ(gpr) AMX_OP_GPR(4, gpr)
#define AMX_STZ(gpr) AMX_OP_GPR(5, gpr)
#define AMX_SET() AMX_NOP_OP_IMM5(17, 0)
#define AMX_CLR() AMX_NOP_OP_IMM5(17, 1)
#define AMX_MATFP(gpr) AMX_OP_GPR(21, gpr)

namespace gsv::kern::amx {

// MATFP 操作数: mode3 = f16 xy / f32 z 外积累加 (alumode 0 = z + x*y,
// 全 lane 使能)。x_off/y_off 为 X/Y 寄存器池内字节偏移 (0..511)。
inline uint64_t matfp_f16f32(uint64_t x_off, uint64_t y_off) {
  return (3ull << 42) | ((x_off & 0x1FFull) << 10) | (y_off & 0x1FFull);
}

// LDX/LDY/LDZ/STZ 操作数: 64B 内存指针 + 高 8-bit 寄存器索引。
inline uint64_t ld_op(const void* p, uint64_t reg_idx) {
  return (uint64_t)(p) | (reg_idx << 56);
}

// 混合宽度 Z 读回: 32×32 f32 tile 中 C[m][n] 的地址
// (调用方需已把 64 行 Z STZ 到连续 [64][64B] 缓冲)。
inline float z_read(const uint8_t* zbuf, int m, int n) {
  return *reinterpret_cast<const float*>(zbuf + (size_t)(n << 1 | (m & 1)) * 64 +
                                         (size_t)(m >> 1) * 4);
}

}  // namespace gsv::kern::amx
