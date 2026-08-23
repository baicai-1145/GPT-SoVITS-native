// accel.hpp — Accelerate (vecLib/cblas) 薄封装
//
// 本机调研结论(2024 SDK, Apple clang 17):
//   - cblas_hgemm 不存在(cblas.h 无声明, libBLAS 无该导出符号)
//   - BNNS 有 Float16 数据类型, 但仅 FullyConnected filter/graph API, 无可直接
//     对标的 hgemm C 接口
//   → 统一接口走回退路径: fp16 权重无损升位(fp16→fp32 零舍入) + cblas_sgemm,
//     调用方不感知实现差异; 未来若出现原生 hgemm 仅改本文件即可切换。
#pragma once

#include <cstddef>
#include <stdint.h>
#include <vector>

namespace gsv::kern::accel {

// ---- 底层: 行主序 fp32 GEMM, C[M,N] = α·op(A)·op(B) + β·C ----
// transa/transb: 'N' 或 'T'。lda/ldb/ldc 为对应矩阵的行主序 leading dim。
// β==0 时按 cblas 语义不读旧 C(允许 C 含 NaN)。
void sgemm(char transa, char transb, int M, int N, int K,
           float alpha, const float* A, int lda,
           const float* B, int ldb,
           float beta, float* C, int ldc);

// ---- fp16 位型批量无损升位 ----
void f16_to_f32(const uint16_t* src, float* dst, size_t n);

// ---- 引擎惯用形: y[T,out] = x[T,in] · W[out,in]ᵀ (W 为 fp16 存储, 行主) ----
// 即每个输出维度一个点积; 与 kern::gemv_f16w_f32acc 单行语义一致(T=1 时同义)。
// 内部把 W 升位到临时缓冲后走 sgemm; 高频调用请用 DenseF16 缓存升位结果。
void gemm_nt_f16w(const float* x, size_t T, size_t in,
                  const uint16_t* w, size_t out,
                  float* y);

// ---- 全连接层权重持有者: 加载时一次性升位, 前向零转换开销 ----
class DenseF16 {
 public:
  DenseF16() = default;
  DenseF16(const uint16_t* w16, size_t out, size_t in);  // 拷贝并升位
  DenseF16(const float* w32_src, size_t out, size_t in);  // 已是 fp32 的权重(WN 融合产物)

  size_t rows() const { return out_; }  // out
  size_t cols() const { return in_; }   // in

  // y[T,out] = x[T,in]·Wᵀ; y 不清零直接覆写
  void forward(const float* x, size_t T, float* y) const;

 private:
  size_t out_ = 0, in_ = 0;
  std::vector<float> w_;  // [out,in] 行主 fp32
};

}  // namespace gsv::kern::accel
