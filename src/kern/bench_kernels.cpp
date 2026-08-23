// bench_kernels.cpp — see bench_kernels.hpp.
//
// All parallel kernels take a `reps` argument and hoist it into a single
// parallelForRep: one condvar dispatch covers the whole timing batch. The
// plain (reps==1) path is identical to a normal kernel invocation.
//
// NEON notes:
//  * gemvFp16w: 16-wide fp16 load -> vcvt -> VFMA into four independent
//    float32x4 accumulator chains; horizontal add at row end.
//  * gemvFp16wFmlal: vfmlalq_low/high_f16 fuse the widen+muladd (FMLAL).
#include "bench_kernels.hpp"

#include <Accelerate/Accelerate.h>
#include <arm_neon.h>
#include <cmath>

#include "bench_harness.hpp"

namespace gsv {
namespace bench {

static inline float hsum(float32x4_t v) {
    v = vaddq_f32(v, vextq_f32(v, v, 2));
    v = vaddq_f32(v, vextq_f32(v, v, 1));
    return vgetq_lane_f32(v, 0);
}

static inline float f16bitsToFloat(F16 h) {
    __fp16 hf;
    std::memcpy(&hf, &h, sizeof(hf));
    return static_cast<float>(hf);
}

template <typename RowFn>
static void gemvRows(int n, int threads, int reps, RowFn&& rowFn) {
    if (threads <= 1) {
        for (int r = 0; r < reps; ++r)
            for (int i = 0; i < n; ++i) rowFn(i);
        return;
    }
    parallelForRep(0, n, threads, reps,
                   [&](int lo, int hi) {  // NOLINT
                       for (int i = lo; i < hi; ++i) rowFn(i);
                   });
}

void gemvFp16w(const F16* w, const float* x, float* y, int n, int k,
               int threads, int reps) {
    gemvRows(n, threads, reps, [&](int r) {
        const F16* wr = w + static_cast<int64_t>(r) * k;
        // four independent accumulator chains hide cvt+fma latency
        float32x4_t a0 = vdupq_n_f32(0.f), a1 = vdupq_n_f32(0.f);
        float32x4_t b0 = vdupq_n_f32(0.f), b1 = vdupq_n_f32(0.f);
        int kk = 0;
        for (; kk + 16 <= k; kk += 16) {
            float16x8_t wv0 = *reinterpret_cast<const float16x8_t*>(wr + kk);
            float16x8_t wv1 =
                *reinterpret_cast<const float16x8_t*>(wr + kk + 8);
            a0 = vfmaq_f32(a0, vcvt_f32_f16(vget_low_f16(wv0)),
                           vld1q_f32(x + kk));
            a1 = vfmaq_f32(a1, vcvt_f32_f16(vget_high_f16(wv0)),
                           vld1q_f32(x + kk + 4));
            b0 = vfmaq_f32(b0, vcvt_f32_f16(vget_low_f16(wv1)),
                           vld1q_f32(x + kk + 8));
            b1 = vfmaq_f32(b1, vcvt_f32_f16(vget_high_f16(wv1)),
                           vld1q_f32(x + kk + 12));
        }
        for (; kk + 8 <= k; kk += 8) {
            float16x8_t wv = *reinterpret_cast<const float16x8_t*>(wr + kk);
            a0 = vfmaq_f32(a0, vcvt_f32_f16(vget_low_f16(wv)),
                           vld1q_f32(x + kk));
            a1 = vfmaq_f32(a1, vcvt_f32_f16(vget_high_f16(wv)),
                           vld1q_f32(x + kk + 4));
        }
        float acc =
            hsum(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(b0, b1)));
        for (; kk < k; ++kk)
            acc += f16bitsToFloat(wr[kk]) * x[kk];
        y[r] = acc;
    });
}

void gemvFp16wFmlal(const F16* w, const F16* xh, float* y, int n, int k,
                    int threads, int reps) {
    gemvRows(n, threads, reps, [&](int r) {
        const F16* wr = w + static_cast<int64_t>(r) * k;
        float32x4_t a0 = vdupq_n_f32(0.f), a1 = vdupq_n_f32(0.f);
        int kk = 0;
        for (; kk + 8 <= k; kk += 8) {
            float16x8_t wv = *reinterpret_cast<const float16x8_t*>(wr + kk);
            float16x8_t xv = *reinterpret_cast<const float16x8_t*>(xh + kk);
            a0 = vfmlalq_low_f16(a0, wv, xv);
            a1 = vfmlalq_high_f16(a1, wv, xv);
        }
        float acc = hsum(vaddq_f32(a0, a1));
        for (; kk < k; ++kk)
            acc += f16bitsToFloat(wr[kk]) * f16bitsToFloat(xh[kk]);
        y[r] = acc;
    });
}

void sgemm(const float* a, const float* b, float* c, int m, int n, int k) {
    // Row-major C[M x N] = A[M x K] * B[K x N] via vDSP (classic cblas
    // entries are deprecated since macOS 13.3; A4's wrapper should make the
    // same choice). Accelerate owns its internal threading — the threads
    // knob does not apply here.
    vDSP_mmul(a, 1, b, 1, c, 1, static_cast<vDSP_Length>(m),
              static_cast<vDSP_Length>(n), static_cast<vDSP_Length>(k));
}

void rmsnorm(const float* x, const float* g, float* y, int n, int threads,
             int reps) {
    auto once = [&] {
        float32x4_t s = vdupq_n_f32(0.f);
        int i = 0;
        for (; i + 4 <= n; i += 4) {
            float32x4_t v = vld1q_f32(x + i);
            s = vfmaq_f32(s, v, v);
        }
        float ss = hsum(s);
        for (; i < n; ++i) ss += x[i] * x[i];
        float inv = 1.f / std::sqrt(ss / n + 1e-5f);
        float32x4_t iv = vdupq_n_f32(inv);
        for (i = 0; i + 4 <= n; i += 4)
            vst1q_f32(y + i,
                      vmulq_f32(vld1q_f32(g + i),
                                vmulq_f32(vld1q_f32(x + i), iv)));
        for (; i < n; ++i) y[i] = g[i] * x[i] * inv;
    };
    // Row is small enough that cross-thread reduction overhead exceeds any
    // gain at hidden-size scale; threaded variant belongs to A3's fused
    // multi-row kernels. Measured single-threaded on purpose (documented in
    // report): scaling study uses rope, whose per-element work is uniform.
    (void)threads;
    for (int r = 0; r < reps; ++r) once();
}

void softmax(const float* x, float* y, int n, int threads, int reps) {
    // Softmax over one row is latency-bound; threading splits max/sum
    // reductions but libm exp dominates. Kept scalar-exp + NEON reduce so
    // numbers stay comparable across toolchains. See rmsnorm note: this is
    // a deliberate single-thread reference measurement.
    (void)threads;
    for (int r = 0; r < reps; ++r) {
        float m = x[0];
        for (int i = 1; i < n; ++i) m = m > x[i] ? m : x[i];
        float32x4_t s = vdupq_n_f32(0.f);
        int i = 0;
        for (; i + 4 <= n; i += 4) {
            float32x4_t e = {std::exp(x[i] - m), std::exp(x[i + 1] - m),
                             std::exp(x[i + 2] - m),
                             std::exp(x[i + 3] - m)};
            vst1q_f32(y + i, e);
            s = vaddq_f32(s, e);
        }
        float sum = hsum(s);
        for (; i < n; ++i) {
            y[i] = std::exp(x[i] - m);
            sum += y[i];
        }
        float32x4_t iv = vdupq_n_f32(1.f / sum);
        for (i = 0; i + 4 <= n; i += 4)
            vst1q_f32(y + i, vmulq_f32(vld1q_f32(y + i), iv));
        for (; i < n; ++i) y[i] /= sum;
    }
}

void rope(const float* x, const float* cosT, const float* sinT, float* y,
          int n, int threads, int reps) {
    if (threads <= 1) {
        for (int r = 0; r < reps; ++r) {
            for (int i = 0; i < n / 2; ++i) {
                float c = cosT[i], s = sinT[i];
                float a = x[2 * i], b = x[2 * i + 1];
                y[2 * i] = a * c - b * s;
                y[2 * i + 1] = a * s + b * c;
            }
        }
        return;
    }
    parallelForRep(0, n / 2, threads, reps, [&](int lo, int hi) {
        for (int i = lo; i < hi; ++i) {
            float c = cosT[i], s = sinT[i];
            float a = x[2 * i], b = x[2 * i + 1];
            y[2 * i] = a * c - b * s;
            y[2 * i + 1] = a * s + b * c;
        }
    });
}

}  // namespace bench
}  // namespace gsv
