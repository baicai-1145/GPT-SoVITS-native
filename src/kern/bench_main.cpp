// bench_main.cpp — D1-part1 micro-benchmark driver.
//
// Usage:
//   kern_bench [--suite all|gemv|sgemm|elementwise] [--threads 1,2,4]
//              [--reps 7] [--warmup 2] [--qos user_initiated|utility|...]
//              [--out bench.csv] [--p-ghz 4.41]
//
// Suites:
//   gemv       fp16-weight GEMV, K in {512,1024,1536,2048} x N in
//              {1025,1536,2048}, both cvt+FMA and FMLAL variants.
//   sgemm      Accelerate cblas_sgemm at AR-prefill / decode shapes.
//   elementwise rmsnorm/softmax/rope scaling with thread count.
//
// Utilization accounting (also printed as CSV columns gflops; the report
// generator computes ratios): peak model documented in
// tools/bench/README.md — P-core fp32 NEON = freq * 32 flop/cycle.
#include "bench_kernels.hpp"
#include "bench_harness.hpp"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <Accelerate/Accelerate.h>

using namespace gsv::bench;

struct Args {
    std::string suite = "all";
    std::vector<int> threads{1, 4};
    int reps = 9;
    int warmup = 2;
    std::string qos;
    std::string out;
    double pGhz = 4.41;
};

static Args parseArgs(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        auto next = [&]() -> const char* { return argv[++i]; };
        if (!std::strcmp(argv[i], "--suite")) a.suite = next();
        else if (!std::strcmp(argv[i], "--threads")) {
            a.threads.clear();
            std::string s = next();
            size_t p = 0;
            while (p < s.size()) {
                size_t c = s.find(',', p);
                if (c == std::string::npos) c = s.size();
                a.threads.push_back(std::atoi(s.substr(p, c - p).c_str()));
                p = c + 1;
            }
        } else if (!std::strcmp(argv[i], "--reps")) a.reps = std::atoi(next());
        else if (!std::strcmp(argv[i], "--warmup"))
            a.warmup = std::atoi(next());
        else if (!std::strcmp(argv[i], "--qos")) a.qos = next();
        else if (!std::strcmp(argv[i], "--out")) a.out = next();
        else if (!std::strcmp(argv[i], "--p-ghz")) a.pGhz = std::atof(next());
    }
    return a;
}

// Deterministic pseudo-random fill so runs are reproducible bit-for-bit.
static uint64_t lcg(uint64_t& s) {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return s >> 33;
}
static void fillF32(std::vector<float>& v, uint64_t seed) {
    uint64_t st = seed;
    for (auto& x : v) x = static_cast<float>(lcg(st) % 2000 - 1000) / 997.f;
}
static void fillF16(std::vector<F16>& v, uint64_t seed) {
    uint64_t st = seed;
    for (auto& h : v) {
        float f = static_cast<float>(lcg(st) % 2000 - 1000) / 997.f;
        _Float16 hf = static_cast<_Float16>(f);
        std::memcpy(&h, &hf, sizeof(F16));
    }
}

int main(int argc, char** argv) {
    Args args = parseArgs(argc, argv);
    if (!setQos(args.qos)) {
        std::fprintf(stderr, "warn: qos '%s' not applied\n",
                     args.qos.c_str());
    }
    MachineProfile mp;
    mp.pFreqGhz = args.pGhz;

    CsvWriter csv(args.out);
    if (!csv.ok()) {
        std::fprintf(stderr, "cannot open output\n");
        return 1;
    }

    // ------------------------------------------------------------------
    // GEMV fp16-weight
    // ------------------------------------------------------------------
    if (args.suite == "all" || args.suite == "gemv") {
        const int Ks[] = {512, 1024, 1536, 2048};
        const int Ns[] = {1025, 1536, 2048};  // 1025: unaligned tail row
        for (int kDim : Ks) {
            for (int nDim : Ns) {
                std::vector<F16> w(static_cast<size_t>(nDim) * kDim);
                std::vector<float> x(static_cast<size_t>(kDim));
                std::vector<F16> xh(static_cast<size_t>(kDim));
                std::vector<float> y(static_cast<size_t>(nDim));
                fillF16(w, 42);
                fillF32(x, 43);
                fillF16(xh, 44);
                double flops = 2.0 * nDim * kDim;
                double bytes =
                    (static_cast<double>(nDim) * kDim * 2 +
                     static_cast<double>(kDim) * 4 + nDim * 4);
                for (int th : args.threads) {
                    Sample s = measure(
                        args.warmup, args.reps, flops, bytes,
                        [&](int) { std::memset(y.data(), 0, y.size() * 4); },
                        [&](int count) {
                            gemvFp16w(w.data(), x.data(), y.data(), nDim,
                                      kDim, th, count);
                        });
                    csv.add({"gemv", "fp16w_cvt_fma",
                             "K=" + std::to_string(kDim) +
                                 ",N=" + std::to_string(nDim),
                             th, "", s.msMedian, s.msMin, s.cvPct, s.gflops, s.gbps});
                    if (th != args.threads.front()) continue;
                    Sample sf = measure(
                        args.warmup, args.reps, flops, bytes,
                        [&](int) { std::memset(y.data(), 0, y.size() * 4); },
                        [&](int count) {
                            gemvFp16wFmlal(w.data(), xh.data(), y.data(),
                                           nDim, kDim, th, count);
                        });
                    csv.add({"gemv", "fp16w_fmlal",
                             "K=" + std::to_string(kDim) +
                                 ",N=" + std::to_string(nDim),
                             th, "", sf.msMedian, sf.msMin, sf.cvPct,
                             sf.gflops, sf.gbps});
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Accelerate sgemm at pipeline-representative shapes
    // ------------------------------------------------------------------
    if (args.suite == "all" || args.suite == "sgemm") {
        struct Shape {
            int m, n, k;
            const char* tag;
        };
        const Shape shapes[] = {
            // AR prefill: token-batch GEMMs through qkv/proj/mlp
            {128, 1024, 1024, "prefill_qkv_128"},
            {128, 4096, 1024, "prefill_mlp_up_128"},
            {64, 1024, 1024, "prefill_qkv_64"},
            // decode batch=1
            {1, 1024, 1024, "dec_qkv_1"},
            {1, 4096, 1024, "dec_mlp_up_1"},
            {1, 1025, 1024, "dec_head_1"},   // vocab head incl. BOS bias dim
            // SoVITS dec im2col-ish mid shapes
            {512, 512, 256, "sovits_mid"},
            {2048, 512, 512, "sovits_dec"},
        };
        for (const Shape& sh : shapes) {
            std::vector<float> a(static_cast<size_t>(sh.m) * sh.k);
            std::vector<float> b(static_cast<size_t>(sh.k) * sh.n);
            std::vector<float> c(static_cast<size_t>(sh.m) * sh.n);
            fillF32(a, 7);
            fillF32(b, 8);
            double flops = 2.0 * sh.m * sh.n * sh.k;
            double bytes = (static_cast<double>(sh.m) * sh.k +
                            static_cast<double>(sh.k) * sh.n +
                            static_cast<double>(sh.m) * sh.n) * 4;
            Sample s = measure(
                args.warmup, args.reps, flops, bytes,
                [&](int) {},
                [&](int count) {
                    for (int i = 0; i < count; ++i)
                        sgemm(a.data(), b.data(), c.data(), sh.m, sh.n,
                              sh.k);
                });
            csv.add({"sgemm", "cblas_sgemm", sh.tag, 0 /*Accelerate-owned*/,
                     "", s.msMedian, s.msMin, s.cvPct, s.gflops, s.gbps});
        }
    }

    // ------------------------------------------------------------------
    // Elementwise kernels vs thread count (scaling study)
    // ------------------------------------------------------------------
    if (args.suite == "all" || args.suite == "elementwise") {
        const int len = 2048;  // hidden size per row
        std::vector<float> x(len), g(len), y(len), cosT(len / 2),
            sinT(len / 2);
        fillF32(x, 11);
        fillF32(g, 12);
        for (int i = 0; i < len / 2; ++i) {
            cosT[static_cast<size_t>(i)] = std::cos(i * 1e-4f);
            sinT[static_cast<size_t>(i)] = std::sin(i * 1e-4f);
        }
        double bytesRms =
            3.0 * len * 4;  // read x,g write y (weights cached)
        double bytesSoftmax = 3.0 * len * 4;
        double bytesRope = 4.0 * len * 4;  // x,y pairs + tables
        for (int th : args.threads) {
            Sample r = measure(
                args.warmup, args.reps, 0, bytesRms,
                [&](int) {},
                [&](int count) {
                    rmsnorm(x.data(), g.data(), y.data(), len, th, count);
                });
            csv.add({"elementwise", "rmsnorm", "len=" + std::to_string(len),
                     th, "", r.msMedian, r.msMin, r.cvPct, r.gflops,
                     r.gbps});
            Sample sm = measure(
                args.warmup, args.reps, 2.0 * len, bytesSoftmax,
                [&](int) {},
                [&](int count) {
                    softmax(x.data(), y.data(), len, th, count);
                });
            csv.add({"elementwise", "softmax", "len=" + std::to_string(len),
                     th, "", sm.msMedian, sm.msMin, sm.cvPct, sm.gflops,
                     sm.gbps});
            Sample rp = measure(
                args.warmup, args.reps, 0, bytesRope,
                [&](int) {},
                [&](int count) {
                    rope(x.data(), cosT.data(), sinT.data(), y.data(), len,
                         th, count);
                });
            csv.add({"elementwise", "rope", "len=" + std::to_string(len), th,
                     "", rp.msMedian, rp.msMin, rp.cvPct, rp.gflops,
                     rp.gbps});
        }
    }

    // Footer row carrying machine profile for the report generator.
    Row meta;
    meta.suite = "meta";
    meta.kernel = "machine";
    char buf[160];
    // semicolon-separated: commas would break the CSV row
    std::snprintf(buf, sizeof(buf),
                  "p_cores=%g;e_cores=%g;p_ghz=%.2f;e_ghz=%.2f;"
                  "peak_single_p_gflops=%.1f;peak_all_p_gflops=%.1f;qos=%s",
                  mp.pCores, mp.eCores, mp.pFreqGhz, mp.eFreqGhz,
                  mp.peakSingleP(), mp.peakAllP(), currentQosName());
    meta.params = buf;
    csv.add(meta);

    std::fprintf(stderr, "done: suite=%s qos=%s out=%s\n",
                 args.suite.c_str(), currentQosName(),
                 args.out.empty() ? "<stdout>" : args.out.c_str());
    return 0;
}
