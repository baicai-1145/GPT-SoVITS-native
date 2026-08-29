// amx_bench.cpp — E5 验收基准: 三方 GEMM 对照
//   (1) Accelerate sgemm (fp32, 内部多线程 — 现网基线)
//   (2) kern::gemm_f16x_fmlal (fp16 FMLAL NEON + P 核池, 现网 fp16 路径)
//   (3) kern::gemm_f16_amx (E5 AMX MATFP, tile 并行)   [需 GSV_AMX_GEMM]
//
// 形状: SoVITS dec conv im2col GEMM (y[Co,S]=W[Co,K]·col[S,K]ᵀ, K=in*k) 与
// encoder 全连接形。T 取 Tq 与 upsample 后真实长度 (v2ProPlus: Tq=2T_lat,
// dec 末端 ×640)。
// 正确性: amx vs fmlal 逐元素全等 (同 fp16 输入同 fp32 累加语义);
//         fmlal vs sgemm 记录量化 relerr (只作参考)。
//
// 用法: ./amx_bench [--reps N]
#include "kern/accel.hpp"
#include "kern/gemv_fmlal.hpp"
#include "kern/kern.hpp"
#include "runtime/threadpool.hpp"
#if defined(GSV_AMX_GEMM)
#include "kern/gemm_f16_amx.hpp"
#define HAVE_AMX 1
#else
#define HAVE_AMX 0
#endif

#include <Accelerate/Accelerate.h>
#include <arm_neon.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <pthread.h>
#include <random>
#include <sys/qos.h>
#include <thread>
#include <vector>

namespace {

double now_ms() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

double now_us() {
  return std::chrono::duration<double, std::micro>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

struct Shape {
  const char* name;
  size_t M, N, K;  // C[M,N] = A[M,K]·B[N,K]ᵀ
};

// ---------------------------------------------------------------------------
// Split-K GEMV kernels
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

inline void reduce_parts(const float* const* parts, size_t n_parts, float* y, size_t out) {
  if (n_parts == 1) {
    if (y != parts[0]) std::memcpy(y, parts[0], out * sizeof(float));
    return;
  }
  if (n_parts == 2) {
    const float* p0 = parts[0];
    const float* p1 = parts[1];
    size_t r = 0;
    for (; r + 4 <= out; r += 4) {
      vst1q_f32(y + r, vaddq_f32(vld1q_f32(p0 + r), vld1q_f32(p1 + r)));
    }
    for (; r < out; ++r) y[r] = p0[r] + p1[r];
    return;
  }
  if (n_parts == 4) {
    const float* p0 = parts[0];
    const float* p1 = parts[1];
    const float* p2 = parts[2];
    const float* p3 = parts[3];
    size_t r = 0;
    for (; r + 4 <= out; r += 4) {
      float32x4_t a01 = vaddq_f32(vld1q_f32(p0 + r), vld1q_f32(p1 + r));
      float32x4_t a23 = vaddq_f32(vld1q_f32(p2 + r), vld1q_f32(p3 + r));
      vst1q_f32(y + r, vaddq_f32(a01, a23));
    }
    for (; r < out; ++r) y[r] = p0[r] + p1[r] + p2[r] + p3[r];
    return;
  }
  // Generic fallback
  std::memcpy(y, parts[0], out * sizeof(float));
  for (size_t p = 1; p < n_parts; ++p) {
    const float* src = parts[p];
    size_t r = 0;
    for (; r + 4 <= out; r += 4) {
      vst1q_f32(y + r, vaddq_f32(vld1q_f32(y + r), vld1q_f32(src + r)));
    }
    for (; r < out; ++r) y[r] += src[r];
  }
}

// ---------------------------------------------------------------------------
// Split-N (Split Rows) GEMV kernels
// ---------------------------------------------------------------------------
inline void gemv_slice_f16x_fmlal_rows(const uint16_t* w, const uint16_t* xh,
                                       float* y, size_t /*out*/, size_t in,
                                       size_t r_start, size_t r_end) {
  const size_t vec_end = in & ~size_t{15};
  size_t r = r_start;
  for (; r + 4 <= r_end; r += 4) {
    const uint16_t* wr0 = w + r * in;
    const uint16_t* wr1 = w + (r + 1) * in;
    const uint16_t* wr2 = w + (r + 2) * in;
    const uint16_t* wr3 = w + (r + 3) * in;
    const uint16_t* xr = xh;
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
    for (; k < in; ++k, ++wr0, ++wr1, ++wr2, ++wr3, ++xr) {
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
    y[r] = s0;
    y[r + 1] = s1;
    y[r + 2] = s2;
    y[r + 3] = s3;
  }
  for (; r < r_end; ++r) {
    const uint16_t* wr = w + r * in;
    const uint16_t* xr = xh;
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    size_t k = 0;
    const size_t vec_end8 = in & ~size_t{7};
    for (; k < vec_end8; k += 8, wr += 8, xr += 8) {
      const float16x8_t w8 = vld1q_f16(reinterpret_cast<const __fp16*>(wr));
      const float16x8_t x8 = vld1q_f16(reinterpret_cast<const __fp16*>(xr));
      acc0 = vfmlalq_low_f16(acc0, w8, x8);
      acc1 = vfmlalq_high_f16(acc1, w8, x8);
    }
    float s = vaddvq_f32(acc0) + vaddvq_f32(acc1);
    for (; k < in; ++k, ++wr, ++xr) {
      __fp16 wh, xhh;
      std::memcpy(&wh, wr, sizeof wh);
      std::memcpy(&xhh, xr, sizeof xhh);
      s += static_cast<float>(wh) * static_cast<float>(xhh);
    }
    y[r] = s;
  }
}

// ---------------------------------------------------------------------------
// Fast Spin-Wait Persistent Thread Pool (QoS UserInitiated, P cores only)
// ---------------------------------------------------------------------------
class FastPool {
 public:
  struct alignas(64) Slot {
    std::atomic<uint32_t> task_seq{0};
    std::atomic<uint32_t> done_seq{0};
  };

  explicit FastPool(size_t n_threads) : n_threads_(n_threads), stop_(false) {
    if (n_threads_ > 1) {
      slots_ = new Slot[n_threads_];
      workers_.reserve(n_threads_ - 1);
      for (size_t i = 1; i < n_threads_; ++i) {
        workers_.emplace_back([this, i] {
          ::pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);
          worker_loop(i);
        });
      }
    }
  }

  ~FastPool() {
    if (!workers_.empty()) {
      stop_.store(true, std::memory_order_release);
      for (size_t i = 1; i < n_threads_; ++i) {
        slots_[i].task_seq.fetch_add(1, std::memory_order_release);
      }
      for (auto& t : workers_) t.join();
      delete[] slots_;
    }
  }

  template <typename Fn>
  void parallel_run(Fn&& fn) {
    if (n_threads_ <= 1) {
      fn(0, 1);
      return;
    }
    struct Caller {
      static void invoke(void* ctx, size_t tid, size_t n_th) {
        (*static_cast<std::decay_t<Fn>*>(ctx))(tid, n_th);
      }
    };
    task_fn_ = &Caller::invoke;
    task_ctx_ = (void*)&fn;
    const uint32_t seq = cur_seq_ + 1;
    cur_seq_ = seq;
    // Signal workers
    for (size_t i = 1; i < n_threads_; ++i) {
      slots_[i].task_seq.store(seq, std::memory_order_release);
    }
    // Worker 0 executes on caller thread
    fn(0, n_threads_);
    // Spin-wait for workers
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
        __builtin_arm_yield();
      }
      if (stop_.load(std::memory_order_acquire)) break;
      last_seq = slots_[tid].task_seq.load(std::memory_order_acquire);
      task_fn_(task_ctx_, tid, n_threads_);
      slots_[tid].done_seq.store(last_seq, std::memory_order_release);
    }
  }

  size_t n_threads_;
  std::atomic<bool> stop_;
  uint32_t cur_seq_{0};
  Slot* slots_{nullptr};
  std::vector<std::thread> workers_;
  void (*task_fn_)(void*, size_t, size_t){nullptr};
  void* task_ctx_{nullptr};
};

}  // namespace

int main(int argc, char** argv) {
  int reps = 5;
  for (int i = 1; i < argc; ++i)
    if (!std::strcmp(argv[i], "--reps") && i + 1 < argc) reps = std::atoi(argv[++i]);

#if HAVE_AMX
  const bool amx_ok = gsv::kern::amx_gemm_available();
  std::printf("AMX backend: %s\n", amx_ok ? "AVAILABLE" : "UNAVAILABLE (fallback)");
#else
  std::printf("AMX backend: compiled OUT (build with -DGSV_AMX_GEMM=ON)\n");
#endif

  // dec (v2ProPlus): conv_pre 768←192 k7 → K=1344; res k3/k7/k11 各通道;
  // upsample 相位 GEMM; conv_post 1←24 k7。T_lat=100 → Tq=200, dec 各级
  // 长度: 200(u0前)→…×10/8/2/2/2→64000 采样, 取各级代表长度。
  std::vector<Shape> shapes = {
      {"dec.conv_pre   Co768 K1344 T200", 768, 200, 1344},
      {"dec.ups0.ph    Co384 K768  T200", 384, 200, 768},
      {"dec.res0.k3    Co384 K1152 T2000", 384, 2000, 1152},
      {"dec.res0.k7    Co384 K2688 T2000", 384, 2000, 2688},
      {"dec.res0.k11   Co384 K4224 T2000", 384, 2000, 4224},
      {"dec.res2.k7    Co96  K672  T8000", 96, 8000, 672},
      {"dec.conv_post  Co1   K168  T12800", 1, 12800, 168},
      {"enc_p.ssl_proj Co768 K512  T300", 768, 300, 512},
      {"enc_p.ffn      Co2048 K768 T300", 2048, 300, 768},
      {"bert.ffn(24L)  Co4096 K1024 T64", 4096, 64, 1024},
      // E17: AR prefill & decode shapes at S=280 / T=1
      {"ar.prefill_qkv Co1536 K512  S280", 1536, 280, 512},
      {"ar.prefill_w1  Co2048 K512  S280", 2048, 280, 512},
      {"ar.prefill_w2  Co512  K2048 S280", 512, 280, 2048},
      {"ar.sdpa_qk     M280   N280  K32",  280, 280, 32},
      {"ar.decode_qkv  Co1536 K512  T1",   1536, 1, 512},
      {"ar.decode_w1   Co2048 K512  T1",   2048, 1, 512},
      {"ar.decode_w2   Co512  K2048 T1",   512, 1, 2048},
      {"ar.decode_wout Co512  K512  T1",   512, 1, 512},
      {"ar.decode_wp   Co1025 K512  T1",   1025, 1, 512},
  };

  std::mt19937 rng(20260824);
  std::printf("%-30s %8s %8s %8s %8s %7s %7s %7s %s\n", "shape",
              "sgemm", "fmlal", "amx", "amxpp", "sg/pp", "fl/pp", "ax/pp",
              "accuracy");
  const int ROUNDS = 8;  // 交错轮数; 每列取 min (抗热降频/干扰)
  for (const auto& s : shapes) {
    const size_t M = s.M, N = s.N, K = s.K;
    // 本基准用 A·Bᵀ 布局: A=[M,K], B=[N,K] (与 kern 接口同构)
    std::vector<float> Af(M * K), Bf(N * K), Cf(M * N);
    std::vector<uint16_t> Ah(M * K), Bh(N * K);
    for (auto& v : Af) v = (float)((int)(rng() % 4096) - 2048) * 0.001953125f;  // ±4 精确 f16
    for (auto& v : Bf) v = (float)((int)(rng() % 4096) - 2048) * 0.001953125f;
    gsv::kern::f32_to_f16(Af.data(), Ah.data(), Af.size());
    gsv::kern::f32_to_f16(Bf.data(), Bh.data(), Bf.size());
    std::vector<float> Cfl(M * N);
    gsv::kern::gemm_f16x_fmlal(Ah.data(), Bh.data(), Cfl.data(), M, N, K);

#if HAVE_AMX
    std::vector<float> Ca(M * N);
    gsv::kern::gemm_f16_amx(Ah.data(), Bh.data(), Ca.data(), M, N, K);
    gsv::kern::AmxPanel pA = gsv::kern::amx_pack(Ah.data(), M, K);
    gsv::kern::AmxPanel pB = gsv::kern::amx_pack(Bh.data(), N, K);
    std::vector<float> Cpp(M * N);
    gsv::kern::gemm_f16_amx_pp(pA, pB, Cpp.data(), M, N);

    double sg = 1e30, fl = 1e30, ax = 1e30, pp = 1e30, pr = 1e30;
    for (int rd = 0; rd < ROUNDS; ++rd) {
      auto t0 = now_ms();
      for (int r = 0; r < reps; ++r)
        gsv::kern::accel::sgemm('N', 'T', (int)M, (int)N, (int)K, 1.f, Af.data(),
                                (int)K, Bf.data(), (int)K, 0.f, Cf.data(), (int)N);
      auto t1 = now_ms();
      sg = std::min(sg, (t1 - t0) / reps);

      t0 = now_ms();
      for (int r = 0; r < reps; ++r)
        gsv::kern::gemm_f16x_fmlal(Ah.data(), Bh.data(), Cfl.data(), M, N, K);
      t1 = now_ms();
      fl = std::min(fl, (t1 - t0) / reps);

      t0 = now_ms();
      for (int r = 0; r < reps; ++r)
        gsv::kern::gemm_f16_amx(Ah.data(), Bh.data(), Ca.data(), M, N, K);
      t1 = now_ms();
      ax = std::min(ax, (t1 - t0) / reps);

      t0 = now_ms();
      for (int r = 0; r < reps; ++r)
        gsv::kern::gemm_f16_amx_pp(pA, pB, Cpp.data(), M, N);
      t1 = now_ms();
      pp = std::min(pp, (t1 - t0) / reps);

      // 生产稳态: A 预打包 (conv 权重), B 每调用打包 (激活) + 内核
      t0 = now_ms();
      for (int r = 0; r < reps; ++r) {
        pB = gsv::kern::amx_pack(Bh.data(), N, K);
        gsv::kern::gemm_f16_amx_pp(pA, pB, Cpp.data(), M, N);
      }
      t1 = now_ms();
      pr = std::min(pr, (t1 - t0) / reps);
    }
#else
    double sg = 0, fl = 0;
    for (int r = 0; r < reps; ++r)
      gsv::kern::accel::sgemm('N', 'T', (int)M, (int)N, (int)K, 1.f, Af.data(),
                              (int)K, Bf.data(), (int)K, 0.f, Cf.data(), (int)N);
    gsv::kern::gemm_f16x_fmlal(Ah.data(), Bh.data(), Cfl.data(), M, N, K);
    auto t0 = now_ms();
    for (int r = 0; r < reps; ++r)
      gsv::kern::accel::sgemm('N', 'T', (int)M, (int)N, (int)K, 1.f, Af.data(),
                              (int)K, Bf.data(), (int)K, 0.f, Cf.data(), (int)N);
    auto t1 = now_ms();
    sg = (t1 - t0) / reps;
    t0 = now_ms();
    for (int r = 0; r < reps; ++r)
      gsv::kern::gemm_f16x_fmlal(Ah.data(), Bh.data(), Cfl.data(), M, N, K);
    t1 = now_ms();
    fl = (t1 - t0) / reps;
    std::printf("%-30s %8.3f %8.3f %8s %8s\n", s.name, sg, fl, "-", "-");
    continue;
#endif
#if HAVE_AMX
    // 数值: amx vs fmlal 同输入同 fp32 累加语义 → 累加序不同非逐位等。
    // 口径同 G1 (check_b12): cos≥0.9999 且 rel = max|a-b|/max|b| ≤ 1e-3
    double maxabs = 0, maxref = 0, dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < M * N; ++i) {
      const double x = Ca[i], y = Cfl[i];
      dot += x * y; na += x * x; nb += y * y;
      maxabs = std::max(maxabs, std::fabs(x - y));
      maxref = std::max(maxref, std::fabs(y));
    }
    const double cos = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
    const double rel = maxabs / (maxref + 1e-30); (void)0;
    // pp 结果必须与即时路径一致 (同 tiles 同内核)
    double pp_diff = 0;
    for (size_t i = 0; i < M * N; ++i)
      pp_diff = std::max(pp_diff, std::fabs((double)Cpp[i] - Ca[i]));
    const bool ok = cos >= 0.9999 && rel <= 1e-3 && pp_diff == 0;
    std::printf("%-32s %8.3f %8.3f %8.3f %8.3f %6.2fx %6.2fx %6.2fx c=%.6f r=%.0e %s\n",
                s.name, sg, fl, ax, pp, sg / pp, fl / pp, ax / pp, cos, rel,
                ok ? "PASS" : "FAIL");
#else
    std::printf("%-32s %10.3f %10.3f %10s %9s %9s %10s\n", s.name, sg, fl,
                "-", "-", "-", "-");
#endif
    std::fflush(stdout);
  }

  // -------------------------------------------------------------------------
  // P0: Split-K GEMV Microbenchmark (1T / 2T / 4T persistent pool)
  // -------------------------------------------------------------------------
  std::printf("\n========================================================================================\n");
  std::printf("P0: Split-K GEMV Microbenchmark (1T / 2T / 4T Persistent P-Core Pool)\n");
  std::printf("========================================================================================\n");

  FastPool pool1(1);
  FastPool pool2(2);
  FastPool pool4(4);

  struct GemvShape {
    const char* name;
    size_t out;
    size_t in;
  };

  std::vector<GemvShape> gemv_shapes = {
      {"decode_w1  [2048, 512]", 2048, 512},
      {"decode_qkv [1536, 512]", 1536, 512},
      {"decode_w2  [ 512,2048]",  512, 2048},
      {"decode_wp  [1025, 512]", 1025, 512},
  };

  const int GEMV_ROUNDS = 20;
  const int GEMV_ITERS = 200;

  std::printf("\n--- Mode 1: f16 weights + fp32 acc (gemv_f16w_f32acc) ---\n");
  std::printf("%-24s %8s %10s %10s %10s %8s %8s %7s %7s %8s\n",
              "Shape", "Size(KB)", "1T (us)", "2T (us)", "4T (us)",
              "BW-1T", "BW-4T", "2T/1T", "4T/1T", "Gate>=2.5x");

  for (const auto& gs : gemv_shapes) {
    const size_t out = gs.out;
    const size_t in = gs.in;
    const size_t w_bytes = out * in * sizeof(uint16_t);

    std::vector<uint16_t> w(out * in);
    std::vector<float> x(in);
    std::vector<float> y(out);
    for (size_t i = 0; i < out * in; ++i) w[i] = static_cast<uint16_t>((rng() % 4096));
    for (size_t i = 0; i < in; ++i) x[i] = (static_cast<float>(rng() % 1000)) * 0.001f;

    std::vector<float> part_buf(4 * out);
    const float* parts[4] = {
        part_buf.data() + 0 * out,
        part_buf.data() + 1 * out,
        part_buf.data() + 2 * out,
        part_buf.data() + 3 * out,
    };

    auto run_bench = [&](FastPool& pool, size_t n_threads) -> double {
      double best_us = 1e30;
      for (int rd = 0; rd < GEMV_ROUNDS; ++rd) {
        auto t0 = now_us();
        for (int it = 0; it < GEMV_ITERS; ++it) {
          pool.parallel_run([&](size_t tid, size_t num_threads) {
            const size_t k_start = tid * (in / num_threads);
            const size_t k_end = (tid + 1) * (in / num_threads);
            gemv_slice_f16w_f32acc_4rows(w.data(), x.data(),
                                         part_buf.data() + tid * out,
                                         out, in, k_start, k_end);
          });
          reduce_parts(parts, n_threads, y.data(), out);
        }
        auto t1 = now_us();
        best_us = std::min(best_us, (t1 - t0) / GEMV_ITERS);
      }
      return best_us;
    };

    double t1_us = run_bench(pool1, 1);
    double t2_us = run_bench(pool2, 2);
    double t4_us = run_bench(pool4, 4);

    double bw1 = (static_cast<double>(w_bytes) / (t1_us * 1e-6)) / 1e9;
    double bw4 = (static_cast<double>(w_bytes) / (t4_us * 1e-6)) / 1e9;
    double sp2 = t1_us / t2_us;
    double sp4 = t1_us / t4_us;
    bool pass = sp4 >= 2.5;

    std::printf("%-24s %8.1f %10.2f %10.2f %10.2f %7.1fG %7.1fG %6.2fx %6.2fx %8s\n",
                gs.name, static_cast<double>(w_bytes) / 1024.0,
                t1_us, t2_us, t4_us, bw1, bw4, sp2, sp4, pass ? "PASS" : "FAIL");
  }

  std::printf("\n--- Mode 2: f16 weights + fp16 x + FMLAL acc (gemv_f16x_fmlal) ---\n");
  std::printf("%-24s %8s %10s %10s %10s %8s %8s %7s %7s %8s\n",
              "Shape", "Size(KB)", "1T (us)", "2T (us)", "4T (us)",
              "BW-1T", "BW-4T", "2T/1T", "4T/1T", "Gate>=2.5x");

  for (const auto& gs : gemv_shapes) {
    const size_t out = gs.out;
    const size_t in = gs.in;
    const size_t w_bytes = out * in * sizeof(uint16_t);

    std::vector<uint16_t> w(out * in);
    std::vector<uint16_t> xh(in);
    std::vector<float> y(out);
    for (size_t i = 0; i < out * in; ++i) w[i] = static_cast<uint16_t>((rng() % 4096));
    for (size_t i = 0; i < in; ++i) xh[i] = static_cast<uint16_t>((rng() % 4096));

    std::vector<float> part_buf(4 * out);
    const float* parts[4] = {
        part_buf.data() + 0 * out,
        part_buf.data() + 1 * out,
        part_buf.data() + 2 * out,
        part_buf.data() + 3 * out,
    };

    auto run_bench = [&](FastPool& pool, size_t n_threads) -> double {
      double best_us = 1e30;
      for (int rd = 0; rd < GEMV_ROUNDS; ++rd) {
        auto t0 = now_us();
        for (int it = 0; it < GEMV_ITERS; ++it) {
          pool.parallel_run([&](size_t tid, size_t num_threads) {
            const size_t k_start = tid * (in / num_threads);
            const size_t k_end = (tid + 1) * (in / num_threads);
            gemv_slice_f16x_fmlal_4rows(w.data(), xh.data(),
                                        part_buf.data() + tid * out,
                                        out, in, k_start, k_end);
          });
          reduce_parts(parts, n_threads, y.data(), out);
        }
        auto t1 = now_us();
        best_us = std::min(best_us, (t1 - t0) / GEMV_ITERS);
      }
      return best_us;
    };

    double t1_us = run_bench(pool1, 1);
    double t2_us = run_bench(pool2, 2);
    double t4_us = run_bench(pool4, 4);

    double bw1 = (static_cast<double>(w_bytes) / (t1_us * 1e-6)) / 1e9;
    double bw4 = (static_cast<double>(w_bytes) / (t4_us * 1e-6)) / 1e9;
    double sp2 = t1_us / t2_us;
    double sp4 = t1_us / t4_us;
    bool pass = sp4 >= 2.5;

    std::printf("%-24s %8.1f %10.2f %10.2f %10.2f %7.1fG %7.1fG %6.2fx %6.2fx %8s\n",
                gs.name, static_cast<double>(w_bytes) / 1024.0,
                t1_us, t2_us, t4_us, bw1, bw4, sp2, sp4, pass ? "PASS" : "FAIL");
  }

  // -------------------------------------------------------------------------
  // Mode 3: 24-Layer Full AR Decode DRAM Stream Simulation (75.5 MB fp16 weights)
  // -------------------------------------------------------------------------
  std::printf("\n--- Mode 3: 24-Layer AR Decode Stream (75.5 MB DRAM Stream, 24 Layers x 4 GEMVs) ---\n");
  {
    struct LayerWeights {
      std::vector<uint16_t> w_qkv; // [1536, 512]
      std::vector<uint16_t> w_out; // [ 512, 512]
      std::vector<uint16_t> w_w1;  // [2048, 512]
      std::vector<uint16_t> w_w2;  // [ 512,2048]
    };
    std::vector<LayerWeights> layers(24);
    size_t total_stream_bytes = 0;
    for (size_t l = 0; l < 24; ++l) {
      layers[l].w_qkv.assign(1536 * 512, 1);
      layers[l].w_out.assign(512 * 512, 1);
      layers[l].w_w1.assign(2048 * 512, 1);
      layers[l].w_w2.assign(512 * 2048, 1);
      total_stream_bytes += (1536 * 512 + 512 * 512 + 2048 * 512 + 512 * 2048) * 2;
    }

    std::vector<uint16_t> xh512(512, 1);
    std::vector<uint16_t> xh2048(2048, 1);
    std::vector<float> y1536(1536), y512(512), y2048(2048);
    std::vector<float> part1536(4 * 1536), part512(4 * 512), part2048(4 * 2048);
    const float* parts1536[4] = {part1536.data(), part1536.data() + 1536, part1536.data() + 3072, part1536.data() + 4608};
    const float* parts512[4]  = {part512.data(), part512.data() + 512, part512.data() + 1024, part512.data() + 1536};
    const float* parts2048[4] = {part2048.data(), part2048.data() + 2048, part2048.data() + 4096, part2048.data() + 6144};

    auto run_stream_splitk = [&](FastPool& pool, size_t n_threads) -> double {
      double best_ms = 1e30;
      for (int rd = 0; rd < 10; ++rd) {
        auto t0 = now_ms();
        for (int it = 0; it < 5; ++it) {
          for (size_t l = 0; l < 24; ++l) {
            // QKV
            pool.parallel_run([&](size_t tid, size_t num_threads) {
              const size_t k0 = tid * (512 / num_threads);
              const size_t k1 = (tid + 1) * (512 / num_threads);
              gemv_slice_f16x_fmlal_4rows(layers[l].w_qkv.data(), xh512.data(),
                                          part1536.data() + tid * 1536, 1536, 512, k0, k1);
            });
            reduce_parts(parts1536, n_threads, y1536.data(), 1536);

            // W_out
            pool.parallel_run([&](size_t tid, size_t num_threads) {
              const size_t k0 = tid * (512 / num_threads);
              const size_t k1 = (tid + 1) * (512 / num_threads);
              gemv_slice_f16x_fmlal_4rows(layers[l].w_out.data(), xh512.data(),
                                          part512.data() + tid * 512, 512, 512, k0, k1);
            });
            reduce_parts(parts512, n_threads, y512.data(), 512);

            // W1
            pool.parallel_run([&](size_t tid, size_t num_threads) {
              const size_t k0 = tid * (512 / num_threads);
              const size_t k1 = (tid + 1) * (512 / num_threads);
              gemv_slice_f16x_fmlal_4rows(layers[l].w_w1.data(), xh512.data(),
                                          part2048.data() + tid * 2048, 2048, 512, k0, k1);
            });
            reduce_parts(parts2048, n_threads, y2048.data(), 2048);

            // W2
            pool.parallel_run([&](size_t tid, size_t num_threads) {
              const size_t k0 = tid * (2048 / num_threads);
              const size_t k1 = (tid + 1) * (2048 / num_threads);
              gemv_slice_f16x_fmlal_4rows(layers[l].w_w2.data(), xh2048.data(),
                                          part512.data() + tid * 512, 512, 2048, k0, k1);
            });
            reduce_parts(parts512, n_threads, y512.data(), 512);
          }
        }
        auto t1 = now_ms();
        best_ms = std::min(best_ms, (t1 - t0) / 5.0);
      }
      return best_ms;
    };

    auto run_stream_splitn = [&](FastPool& pool, size_t /*n_threads*/) -> double {
      double best_ms = 1e30;
      for (int rd = 0; rd < 10; ++rd) {
        auto t0 = now_ms();
        for (int it = 0; it < 5; ++it) {
          for (size_t l = 0; l < 24; ++l) {
            // QKV [1536, 512]
            pool.parallel_run([&](size_t tid, size_t num_threads) {
              const size_t r0 = tid * (1536 / num_threads);
              const size_t r1 = (tid + 1) * (1536 / num_threads);
              gemv_slice_f16x_fmlal_rows(layers[l].w_qkv.data(), xh512.data(),
                                         y1536.data(), 1536, 512, r0, r1);
            });

            // W_out [512, 512]
            pool.parallel_run([&](size_t tid, size_t num_threads) {
              const size_t r0 = tid * (512 / num_threads);
              const size_t r1 = (tid + 1) * (512 / num_threads);
              gemv_slice_f16x_fmlal_rows(layers[l].w_out.data(), xh512.data(),
                                         y512.data(), 512, 512, r0, r1);
            });

            // W1 [2048, 512]
            pool.parallel_run([&](size_t tid, size_t num_threads) {
              const size_t r0 = tid * (2048 / num_threads);
              const size_t r1 = (tid + 1) * (2048 / num_threads);
              gemv_slice_f16x_fmlal_rows(layers[l].w_w1.data(), xh512.data(),
                                         y2048.data(), 2048, 512, r0, r1);
            });

            // W2 [512, 2048]
            pool.parallel_run([&](size_t tid, size_t num_threads) {
              const size_t r0 = tid * (512 / num_threads);
              const size_t r1 = (tid + 1) * (512 / num_threads);
              gemv_slice_f16x_fmlal_rows(layers[l].w_w2.data(), xh2048.data(),
                                         y512.data(), 512, 2048, r0, r1);
            });
          }
        }
        auto t1 = now_ms();
        best_ms = std::min(best_ms, (t1 - t0) / 5.0);
      }
      return best_ms;
    };

    double sk1 = run_stream_splitk(pool1, 1);
    double sk2 = run_stream_splitk(pool2, 2);
    double sk4 = run_stream_splitk(pool4, 4);

    double sn1 = run_stream_splitn(pool1, 1);
    double sn2 = run_stream_splitn(pool2, 2);
    double sn4 = run_stream_splitn(pool4, 4);

    std::printf("24-Layer Stream (75.5MB) -- Split-K:\n");
    std::printf("  1T: %6.3f ms (BW: %6.1f GB/s)\n", sk1, (total_stream_bytes / (sk1 * 1e-3)) / 1e9);
    std::printf("  2T: %6.3f ms (BW: %6.1f GB/s, Speedup: %.2fx)\n", sk2, (total_stream_bytes / (sk2 * 1e-3)) / 1e9, sk1 / sk2);
    std::printf("  4T: %6.3f ms (BW: %6.1f GB/s, Speedup: %.2fx)\n", sk4, (total_stream_bytes / (sk4 * 1e-3)) / 1e9, sk1 / sk4);

    std::printf("24-Layer Stream (75.5MB) -- Split-N (Split Rows):\n");
    std::printf("  1T: %6.3f ms (BW: %6.1f GB/s)\n", sn1, (total_stream_bytes / (sn1 * 1e-3)) / 1e9);
    std::printf("  2T: %6.3f ms (BW: %6.1f GB/s, Speedup: %.2fx)\n", sn2, (total_stream_bytes / (sn2 * 1e-3)) / 1e9, sn1 / sn2);
    std::printf("  4T: %6.3f ms (BW: %6.1f GB/s, Speedup: %.2fx)\n", sn4, (total_stream_bytes / (sn4 * 1e-3)) / 1e9, sn1 / sn4);
  }

  // -------------------------------------------------------------------------
  // P0: SDPA Head-Parallel Microbenchmark (E19: 1T / 2T / 4T persistent pool)
  // -------------------------------------------------------------------------
  std::printf("\n========================================================================================\n");
  std::printf("P0: SDPA Head-Parallel Microbenchmark (E19: 1T / 2T / 4T Persistent P-Core Pool)\n");
  std::printf("========================================================================================\n");

  auto dot_f32_bench = [](const float* qv, const float* kv, size_t HD) -> float {
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
    float dot = vget_lane_f32(sum1, 0);
    for (; e < HD; ++e) dot += qv[e] * kv[e];
    return dot;
  };

  auto dot_f16kv_bench = [](const float* qv, const uint16_t* k16, size_t HD) -> float {
    float32x4_t a0 = vdupq_n_f32(0.f);
    float32x4_t a1 = vdupq_n_f32(0.f);
    float32x4_t a2 = vdupq_n_f32(0.f);
    float32x4_t a3 = vdupq_n_f32(0.f);
    size_t e = 0;
    for (; e + 16 <= HD; e += 16) {
      float16x8_t h0 = vld1q_f16(reinterpret_cast<const __fp16*>(k16 + e));
      float16x8_t h1 = vld1q_f16(reinterpret_cast<const __fp16*>(k16 + e + 8));
      a0 = vfmaq_f32(a0, vld1q_f32(qv + e + 0),  vcvt_f32_f16(vget_low_f16(h0)));
      a1 = vfmaq_f32(a1, vld1q_f32(qv + e + 4),  vcvt_f32_f16(vget_high_f16(h0)));
      a2 = vfmaq_f32(a2, vld1q_f32(qv + e + 8),  vcvt_f32_f16(vget_low_f16(h1)));
      a3 = vfmaq_f32(a3, vld1q_f32(qv + e + 12), vcvt_f32_f16(vget_high_f16(h1)));
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
  };

  auto accum_f32_bench = [](float* ov, const float* vv, float p, size_t HD) {
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
    for (; e < HD; ++e) ov[e] += p * vv[e];
  };

  auto accum_f16kv_bench = [](float* ov, const uint16_t* v16, float p, size_t HD) {
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
  };

  struct SdpaBenchCase {
    const char* name;
    size_t H;
    size_t HD;
    size_t S;
  };

  std::vector<SdpaBenchCase> sdpa_cases = {
      {"sdpa_h16_s280", 16, 32, 280},
      {"sdpa_h16_s560", 16, 32, 560},
      {"sdpa_h24_s280", 24, 32, 280},
      {"sdpa_h24_s560", 24, 32, 560},
  };

  const int SDPA_ROUNDS = 20;
  const int SDPA_ITERS = 200;

  std::printf("\n--- Mode 1: FP32 KV Cache (Single Layer SDPA) ---\n");
  std::printf("%-20s %6s %6s %10s %10s %10s %8s %8s %7s %7s %8s\n",
              "Shape", "H", "S", "1T (us)", "2T (us)", "4T (us)",
              "BW-1T", "BW-4T", "2T/1T", "4T/1T", "Gate>=2.2x");

  for (const auto& sc : sdpa_cases) {
    const size_t H = sc.H, HD = sc.HD, S = sc.S, D = H * HD;
    const float scale = 1.0f / std::sqrt(static_cast<float>(HD));
    const size_t kv_bytes = 2 * S * D * sizeof(float); // K + V read

    std::vector<float> q(D);
    std::vector<float> kcache(S * D);
    std::vector<float> vcache(S * D);
    std::vector<float> attn(D);
    for (size_t i = 0; i < D; ++i) q[i] = (static_cast<float>(rng() % 1000) - 500.f) * 0.001f;
    for (size_t i = 0; i < S * D; ++i) kcache[i] = (static_cast<float>(rng() % 1000) - 500.f) * 0.001f;
    for (size_t i = 0; i < S * D; ++i) vcache[i] = (static_cast<float>(rng() % 1000) - 500.f) * 0.001f;

    std::vector<std::vector<float>> thread_scores(4, std::vector<float>(S));

    auto run_sdpa = [&](FastPool& pool, size_t /*n_threads*/) -> double {
      double best_us = 1e30;
      for (int rd = 0; rd < SDPA_ROUNDS; ++rd) {
        auto t0 = now_us();
        for (int it = 0; it < SDPA_ITERS; ++it) {
          pool.parallel_run([&](size_t tid, size_t num_threads) {
            const size_t h0 = tid * (H / num_threads);
            const size_t h1 = (tid + 1) * (H / num_threads);
            float* scores = thread_scores[tid].data();
            for (size_t h = h0; h < h1; ++h) {
              const float* qv = q.data() + h * HD;
              for (size_t k = 0; k < S; ++k) {
                float dot = dot_f32_bench(qv, kcache.data() + k * D + h * HD, HD);
                scores[k] = dot * scale;
              }
              gsv::kern::softmax(scores, scores, S);
              float* ov = attn.data() + h * HD;
              std::memset(ov, 0, HD * sizeof(float));
              for (size_t k = 0; k < S; ++k) {
                accum_f32_bench(ov, vcache.data() + k * D + h * HD, scores[k], HD);
              }
            }
          });
        }
        auto t1 = now_us();
        best_us = std::min(best_us, (t1 - t0) / SDPA_ITERS);
      }
      return best_us;
    };

    double t1_us = run_sdpa(pool1, 1);
    double t2_us = run_sdpa(pool2, 2);
    double t4_us = run_sdpa(pool4, 4);

    double bw1 = (static_cast<double>(kv_bytes) / (t1_us * 1e-6)) / 1e9;
    double bw4 = (static_cast<double>(kv_bytes) / (t4_us * 1e-6)) / 1e9;
    double sp2 = t1_us / t2_us;
    double sp4 = t1_us / t4_us;
    bool pass = sp4 >= 2.2;

    std::printf("%-20s %6zu %6zu %10.2f %10.2f %10.2f %7.1fG %7.1fG %6.2fx %6.2fx %8s\n",
                sc.name, H, S, t1_us, t2_us, t4_us, bw1, bw4, sp2, sp4, pass ? "PASS" : "FAIL");
  }

  std::printf("\n--- Mode 2: FP16 KV Cache (Single Layer SDPA) ---\n");
  std::printf("%-20s %6s %6s %10s %10s %10s %8s %8s %7s %7s %8s\n",
              "Shape", "H", "S", "1T (us)", "2T (us)", "4T (us)",
              "BW-1T", "BW-4T", "2T/1T", "4T/1T", "Gate>=2.2x");

  for (const auto& sc : sdpa_cases) {
    const size_t H = sc.H, HD = sc.HD, S = sc.S, D = H * HD;
    const float scale = 1.0f / std::sqrt(static_cast<float>(HD));
    const size_t kv_bytes = 2 * S * D * sizeof(uint16_t); // K + V read fp16

    std::vector<float> q(D);
    std::vector<uint16_t> kcache16(S * D);
    std::vector<uint16_t> vcache16(S * D);
    std::vector<float> attn(D);
    for (size_t i = 0; i < D; ++i) q[i] = (static_cast<float>(rng() % 1000) - 500.f) * 0.001f;
    for (size_t i = 0; i < S * D; ++i) kcache16[i] = static_cast<uint16_t>(rng() % 4096);
    for (size_t i = 0; i < S * D; ++i) vcache16[i] = static_cast<uint16_t>(rng() % 4096);

    std::vector<std::vector<float>> thread_scores(4, std::vector<float>(S));

    auto run_sdpa16 = [&](FastPool& pool, size_t /*n_threads*/) -> double {
      double best_us = 1e30;
      for (int rd = 0; rd < SDPA_ROUNDS; ++rd) {
        auto t0 = now_us();
        for (int it = 0; it < SDPA_ITERS; ++it) {
          pool.parallel_run([&](size_t tid, size_t num_threads) {
            const size_t h0 = tid * (H / num_threads);
            const size_t h1 = (tid + 1) * (H / num_threads);
            float* scores = thread_scores[tid].data();
            for (size_t h = h0; h < h1; ++h) {
              const float* qv = q.data() + h * HD;
              for (size_t k = 0; k < S; ++k) {
                float dot = dot_f16kv_bench(qv, kcache16.data() + k * D + h * HD, HD);
                scores[k] = dot * scale;
              }
              gsv::kern::softmax(scores, scores, S);
              float* ov = attn.data() + h * HD;
              std::memset(ov, 0, HD * sizeof(float));
              for (size_t k = 0; k < S; ++k) {
                accum_f16kv_bench(ov, vcache16.data() + k * D + h * HD, scores[k], HD);
              }
            }
          });
        }
        auto t1 = now_us();
        best_us = std::min(best_us, (t1 - t0) / SDPA_ITERS);
      }
      return best_us;
    };

    double t1_us = run_sdpa16(pool1, 1);
    double t2_us = run_sdpa16(pool2, 2);
    double t4_us = run_sdpa16(pool4, 4);

    double bw1 = (static_cast<double>(kv_bytes) / (t1_us * 1e-6)) / 1e9;
    double bw4 = (static_cast<double>(kv_bytes) / (t4_us * 1e-6)) / 1e9;
    double sp2 = t1_us / t2_us;
    double sp4 = t1_us / t4_us;
    bool pass = sp4 >= 2.2;

    std::printf("%-20s %6zu %6zu %10.2f %10.2f %10.2f %7.1fG %7.1fG %6.2fx %6.2fx %8s\n",
                sc.name, H, S, t1_us, t2_us, t4_us, bw1, bw4, sp2, sp4, pass ? "PASS" : "FAIL");
  }

  // 24-layer Full AR Decode SDPA Extrapolation
  std::printf("\n--- Mode 3: 24-Layer Full AR Decode SDPA (24 Layers x SDPA) ---\n");
  for (const auto& sc : sdpa_cases) {
    if (sc.H != 16) continue;
    const size_t H = sc.H, HD = sc.HD, S = sc.S, D = H * HD;
    const float scale = 1.0f / std::sqrt(static_cast<float>(HD));
    std::vector<float> q(D);
    std::vector<std::vector<uint16_t>> kcache16(24, std::vector<uint16_t>(S * D));
    std::vector<std::vector<uint16_t>> vcache16(24, std::vector<uint16_t>(S * D));
    std::vector<float> attn(D);
    std::vector<std::vector<float>> thread_scores(4, std::vector<float>(S));

    auto run_sdpa24 = [&](FastPool& pool, size_t /*n_threads*/) -> double {
      double best_ms = 1e30;
      for (int rd = 0; rd < 10; ++rd) {
        auto t0 = now_ms();
        for (int it = 0; it < 50; ++it) {
          for (size_t l = 0; l < 24; ++l) {
            pool.parallel_run([&](size_t tid, size_t num_threads) {
              const size_t h0 = tid * (H / num_threads);
              const size_t h1 = (tid + 1) * (H / num_threads);
              float* scores = thread_scores[tid].data();
              for (size_t h = h0; h < h1; ++h) {
                const float* qv = q.data() + h * HD;
                for (size_t k = 0; k < S; ++k) {
                  float dot = dot_f16kv_bench(qv, kcache16[l].data() + k * D + h * HD, HD);
                  scores[k] = dot * scale;
                }
                gsv::kern::softmax(scores, scores, S);
                float* ov = attn.data() + h * HD;
                std::memset(ov, 0, HD * sizeof(float));
                for (size_t k = 0; k < S; ++k) {
                  accum_f16kv_bench(ov, vcache16[l].data() + k * D + h * HD, scores[k], HD);
                }
              }
            });
          }
        }
        auto t1 = now_ms();
        best_ms = std::min(best_ms, (t1 - t0) / 50.0);
      }
      return best_ms;
    };

    double t1_ms = run_sdpa24(pool1, 1);
    double t2_ms = run_sdpa24(pool2, 2);
    double t4_ms = run_sdpa24(pool4, 4);

    std::printf("24-Layer SDPA (FP16 KV, S=%zu): 1T=%.3f ms  2T=%.3f ms (%.2fx)  4T=%.3f ms (%.2fx)  [Gate: %s]\n",
                S, t1_ms, t2_ms, t1_ms / t2_ms, t4_ms, t1_ms / t4_ms, (t1_ms / t4_ms >= 2.2) ? "PASS" : "FAIL");
  }
  std::printf("========================================================================================\n\n");

  return 0;
}
