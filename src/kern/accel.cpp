// accel.cpp — Accelerate 封装实现(回退路径: fp16 升位 + cblas_sgemm)
#include "accel.hpp"

#include <arm_neon.h>

#include <Accelerate/Accelerate.h>

namespace gsv::kern::accel {

void f16_to_f32(const uint16_t* src, float* dst, size_t n) {
  size_t i = 0;
  const size_t vec_end = n & ~size_t{7};
  for (; i < vec_end; i += 8, src += 8, dst += 8) {
    const float16x8_t h = vld1q_f16(reinterpret_cast<const __fp16*>(src));
    vst1q_f32(dst, vcvt_f32_f16(vget_low_f16(h)));      // 无损升位
    vst1q_f32(dst + 4, vcvt_f32_f16(vget_high_f16(h)));
  }
  for (; i < n; ++i, ++src, ++dst) {
    __fp16 h;
    __builtin_memcpy(&h, src, sizeof h);
    *dst = static_cast<float>(h);
  }
}

void sgemm(char transa, char transb, int M, int N, int K,
           float alpha, const float* A, int lda,
           const float* B, int ldb,
           float beta, float* C, int ldc) {
  const CBLAS_ORDER order = CblasRowMajor;
  const CBLAS_TRANSPOSE ta = (transa == 'T' || transa == 't') ? CblasTrans : CblasNoTrans;
  const CBLAS_TRANSPOSE tb = (transb == 'T' || transb == 't') ? CblasTrans : CblasNoTrans;
  cblas_sgemm(order, ta, tb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

void gemm_nt_f16w(const float* x, size_t T, size_t in,
                  const uint16_t* w, size_t out,
                  float* y) {
  // 行主 y[T,out] = x[T,in]·Wᵀ ≡ cblas_sgemm(RowMajor, NoTrans, Trans,
  //                                        T, out, in, 1, x, in, Wf, in, 0, y, out)
  std::vector<float> wf(out * in);
  f16_to_f32(w, wf.data(), wf.size());
  sgemm('N', 'T', static_cast<int>(T), static_cast<int>(out), static_cast<int>(in),
        1.0f, x, static_cast<int>(in), wf.data(), static_cast<int>(in),
        0.0f, y, static_cast<int>(out));
}

// ---------- DenseF16 ----------
DenseF16::DenseF16(const uint16_t* w16, size_t out, size_t in) : out_(out), in_(in) {
  w_.resize(out_ * in_);
  f16_to_f32(w16, w_.data(), w_.size());
}

DenseF16::DenseF16(const float* w32_src, size_t out, size_t in) : out_(out), in_(in) {
  w_.assign(w32_src, w32_src + out_ * in_);
}

void DenseF16::forward(const float* x, size_t T, float* y) const {
  sgemm('N', 'T', static_cast<int>(T), static_cast<int>(out_), static_cast<int>(in_),
        1.0f, x, static_cast<int>(in_), w_.data(), static_cast<int>(in_),
        0.0f, y, static_cast<int>(out_));
}

}  // namespace gsv::kern::accel
