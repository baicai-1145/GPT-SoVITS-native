// bench_kernels.hpp — self-contained kernels under measurement (D1-part1).
//
// These are bench-owned reference implementations, NOT the future A3/A4
// production kernels. They exist so the perf infrastructure is usable before
// Phase A lands; when src/kern production API exists we re-point the
// benchmarks at it and keep these as the baseline column.
//
// All multi-row/element kernels take `reps`: benchmarking runs them in
// steady-state batches where ONE thread-pool dispatch covers the whole
// batch (see ThreadPool::parallelForRep). This removes ~20us/jitter of
// dispatch cost from microsecond kernels and is what makes cross-run
// reproducibility achievable on a fanless part.
#pragma once

#include <cstdint>

namespace gsv {
namespace bench {

using F16 = uint16_t;  // IEEE half bits; read via float16x8_t views

// y[n] = W[n,k] @ x[k], weights stored fp16, x fp32, accumulation fp32.
void gemvFp16w(const F16* w, const float* x, float* y, int n, int k,
               int threads, int reps = 1);

// Same shape but x also stored as fp16, accumulated via FMLAL
// (vfmlalq low/high) — the upper-bound variant A3 may adopt.
void gemvFp16wFmlal(const F16* w, const F16* xh, float* y, int n, int k,
                    int threads, int reps = 1);

// Row-major C[m,n] = A[m,k] * B[k,n]; Accelerate vDSP_mmul (AMX-backed,
// internally threaded — the threads knob does not apply).
void sgemm(const float* a, const float* b, float* c, int m, int n, int k);

// y[i] = x[i] * rsqrt(mean(x^2)+eps) * g[i], fp32 statistics.
// Single-threaded by design at hidden-size scale (cross-thread reduction
// overhead dominates one row); scaling study lives in rope.
void rmsnorm(const float* x, const float* g, float* y, int n, int threads,
             int reps = 1);

// Softmax over one row, fp32 max/sum, scalar exp + NEON reductions.
void softmax(const float* x, float* y, int n, int threads, int reps = 1);

// RoPE over interleaved pairs with precomputed cos/sin tables.
void rope(const float* x, const float* cosT, const float* sinT, float* y,
          int n, int threads, int reps = 1);

}  // namespace bench
}  // namespace gsv
