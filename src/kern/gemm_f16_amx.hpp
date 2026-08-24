// gemm_f16_amx.hpp — E5: AMX MATFP fp16×fp16→fp32 GEMM 后端 (实验)
//
// 接口与 gemm_f16x_fmlal 同构:
//   C[M,N] = Σ_k A[m,k]·B[n,k]   (A [M,K] f16 行主, B [N,K] f16 行主, C f32 行主)
// 覆盖 SoVITS conv im2col (y[Co,S]=W[Co,K]·col[S,K]ᵀ) 与 encoder 全连接形。
//
// 数值语义: 与 gemm_f16x_fmlal 完全一致 —— fp16 输入精确落 f32 (f16×f16 积
// 在 fp32 内无舍入), 累加链纯 fp32。对拍参考即 FMLAL 实现, golden 门禁同口径
// (cos≥0.9999, rel≤1e-3)。
//
// 调度: AMX 专用线程池 (不与 Accelerate/BLAS 共线程, 见 amx.h 互斥注记),
// tile (mb,nb) 粒度分发; worker 首任务前完成 SIGILL 探测 + AMX_SET 长驻,
// 探测失败全局降级回退 fmlal。
//
// 预打包 API: 权重侧 (conv 的 W, 全连接的 W) 为静态数据, pack 一次后反复
// 使用 —— gemm 用 packed panel 直跑内核, 省每调用打包成本。
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace gsv::kern {

// 编译期总开关: 定义 GSV_AMX_GEMM 才编入 (默认关, CMake -DGSV_AMX_GEMM=ON)。
// 运行时开关: amx_gemm_available() (AMX_SET+MATFP 探测 + 专用池 worker 使能)。
#if defined(GSV_AMX_GEMM)

// 探测结论 (首次调用惰性启动 AMX 专用池并探测)。
bool amx_gemm_available();

// 预打包 panel: [rows,K] f16 行主 → AMX X/Y 直读布局 (对齐, 尾行补零)。
// 非多线程安全: 不同线程请各自持 panel (或外部同步)。
struct AmxPanel {
  size_t rows = 0, K = 0;             // 逻辑形状
  std::vector<uint8_t> buf;           // 对齐基址在 buf 内偏移
  const uint8_t* data() const {       // 64B 对齐 panel 首地址
    const uintptr_t p = (uintptr_t)buf.data();
    return buf.data() + ((64 - (p & 63)) & 63);
  }
};

AmxPanel amx_pack(const uint16_t* w, size_t rows, size_t K);

// 预打包版 GEMM: C[M,N] = pa·pbᵀ (pa: M×K, pb: N×K)。
// 需 pa.rows==M && pb.rows==N && pa.K==pb.K, 否则回退即时打包路径。
// AMX 不可用时回退 gemm_f16x_fmlal (自动从 panel 解不出原矩阵, 故要求
// 调用方仅在 available()==true 时走此路径, 内部不二次校验)。
void gemm_f16_amx_pp(const AmxPanel& pa, const AmxPanel& pb, float* c,
                     size_t M, size_t N);

// C[M,N] = A·Bᵀ; 不可用时内部回退 gemm_f16x_fmlal。 (即时打包, 通用路径)
void gemm_f16_amx(const uint16_t* a, const uint16_t* b, float* c,
                  size_t M, size_t N, size_t K);

#else

struct AmxPanel {};  // 空壳: 非 AMX 构建下可引用类型, 不可调用函数

inline bool amx_gemm_available() { return false; }

#endif  // GSV_AMX_GEMM

}  // namespace gsv::kern
