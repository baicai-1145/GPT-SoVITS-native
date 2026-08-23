// bench_harness.hpp — micro-benchmark infrastructure for D1-part1.
//
// Self-contained by design: Phase A kernels do not exist yet, so the
// harness ships its own persistent thread pool and timing utilities. When
// A3/A4 land, bench_kernels switches to the real kern API without touching
// this file (the pool semantics here are the reference for A4's).
//
// Conventions:
//  * Every measurement warms up first (caches/TLB/predictor), then runs N
//    reps and reports the MEDIAN plus coefficient of variation.
//  * Reproducibility gate: CV < 5% across reps for a sample to count as
//    stable; run_kern_bench.sh additionally diffs two full invocations.
//  * Timings use std::chrono::steady_clock (CLOCK_MONOTONIC_RAW-ish).
#pragma once

#include <algorithm>
#include <condition_variable>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <pthread.h>
#include <sys/qos.h>

namespace gsv {
namespace bench {

// ---------------------------------------------------------------------------
// QoS helpers (ARCHITECTURE.md §4: user-initiated -> P cores, utility ->
// E cores). Applied to the calling thread; pool workers spawned afterwards
// inherit the QoS on Darwin.
// ---------------------------------------------------------------------------
inline bool setQos(const std::string& name) {
    qos_class_t q;
    if (name.empty() || name == "default") return true;
    if (name == "user_initiated") q = QOS_CLASS_USER_INITIATED;
    else if (name == "user_interactive") q = QOS_CLASS_USER_INTERACTIVE;
    else if (name == "utility") q = QOS_CLASS_UTILITY;
    else if (name == "background") q = QOS_CLASS_BACKGROUND;
    else return false;
    return pthread_set_qos_class_self_np(q, 0) == 0;
}

inline const char* currentQosName() {
    qos_class_t q = QOS_CLASS_UNSPECIFIED;
    int rel = 0;
    pthread_get_qos_class_np(pthread_self(), &q, &rel);
    switch (q) {
        case QOS_CLASS_USER_INTERACTIVE: return "user_interactive";
        case QOS_CLASS_USER_INITIATED: return "user_initiated";
        case QOS_CLASS_UTILITY: return "utility";
        case QOS_CLASS_BACKGROUND: return "background";
        default: return "unspecified";
    }
}

// ---------------------------------------------------------------------------
// ThreadPool — persistent fork/join over [begin,end).
//
// Why persistent: per-call pthread_create costs ~50-100us on macOS which
// swamps sub-millisecond kernels (measured: GEMV rows went 2-3x SLOWER at
// threads=4 with spawn-per-call). Workers park on a condvar between calls,
// inherit the creating thread's QoS, and each owns a contiguous chunk.
// The calling thread participates as rank 0, so threads==1 is a plain
// inline call with zero synchronization overhead.
// ---------------------------------------------------------------------------
class ThreadPool {
public:
    explicit ThreadPool(int totalParticipants) : n_(totalParticipants) {}
    ~ThreadPool() {
        if (!tids_.empty()) {
            {
                std::lock_guard<std::mutex> lk(m_);
                stop_ = true;
                ++seq_;
            }
            cv_.notify_all();
            for (pthread_t t : tids_) pthread_join(t, nullptr);
        }
    }

    // Single-dispatch fork/join where every participant (workers AND the
    // caller) executes body on its chunk `reps` times. This is what lets
    // the benchmark amortize one condvar round-trip across a whole timing
    // batch — per-call dispatch costs ~20us and jitter here would dominate
    // microsecond kernels.
    template <typename Fn>
    void parallelForRep(int begin, int end, int reps, Fn&& body) {
        if (n_ <= 1 || end - begin <= 1) {
            for (int i = 0; i < reps; ++i) body(begin, end);
            return;
        }
        ensureWorkers();
        Task t{&trampoline<std::remove_reference_t<Fn>>, &body, begin, end};
        {
            std::lock_guard<std::mutex> lk(m_);
            cur_ = t;
            curReps_ = reps > 0 ? reps : 1;
            ++seq_;
            pending_ = static_cast<int>(tids_.size());
        }
        cv_.notify_all();
        const int span = end - begin;
        const int step = (span + n_ - 1) / n_;
        const int lo = begin, hi = std::min(end, begin + step);
        for (int i = 0; i < curReps_; ++i) body(lo, hi);  // rank 0 chunk
        std::unique_lock<std::mutex> lk(m_);
        doneCv_.wait(lk, [this] { return pending_ == 0; });
    }

    template <typename Fn>
    void parallelFor(int begin, int end, Fn&& body) {
        parallelForRep(begin, end, 1, std::forward<Fn>(body));
    }

private:
    using FnT = void (*)(void*, int, int);
    struct Task {
        FnT fn;
        void* arg;
        int begin, end;
    };
    struct Boot {
        ThreadPool* self;
        int rank;
    };
    template <typename Fn>
    static void trampoline(void* arg, int lo, int hi) {
        (*static_cast<Fn*>(arg))(lo, hi);
    }

    void ensureWorkers() {
        if (!tids_.empty()) return;
        tids_.resize(static_cast<size_t>(n_ - 1));
        boots_.resize(tids_.size());
        auto entry = +[](void* p) -> void* {
            auto* b = static_cast<Boot*>(p);
            ThreadPool* self = b->self;
            const int rank = b->rank + 1;  // main thread owns rank 0
            // seen starts BELOW the first dispatch's seq (seq starts at 1):
            // a late-starting worker still wakes for the in-flight task,
            // which is what closes the lost-wakeup/init race.
            long seen = 0;
            for (;;) {
                Task t{};
                bool run = false;
                {
                    std::unique_lock<std::mutex> lk(self->m_);
                    self->cv_.wait(
                        lk, [&] { return self->seq_ > seen || self->stop_; });
                    if (self->stop_) break;
                    seen = self->seq_;
                    t = self->cur_;
                    run = true;
                }
                int reps = 0;
                if (run) {
                    reps = self->curReps_;
                    std::lock_guard<std::mutex> lk(self->m_);
                    // snapshot under lock is unnecessary for compute; just
                    // take the count and release while working
                }
                if (run) {
                    const int span = t.end - t.begin;
                    const int step = (span + self->n_ - 1) / self->n_;
                    const int lo = t.begin + step * rank;
                    const int hi = std::min(t.end, lo + step);
                    for (int i = 0; i < reps; ++i)
                        if (lo < hi) t.fn(t.arg, lo, hi);
                    std::lock_guard<std::mutex> lk(self->m_);
                    if (--self->pending_ == 0) self->doneCv_.notify_all();
                }
            }
            return nullptr;
        };
        for (size_t i = 0; i < tids_.size(); ++i) {
            boots_[i] = Boot{this, static_cast<int>(i)};
            pthread_create(&tids_[i], nullptr, entry, &boots_[i]);
        }
    }

    int n_;                       // total participants incl. caller
    std::vector<pthread_t> tids_;
    std::vector<Boot> boots_;
    std::mutex m_;
    std::condition_variable cv_, doneCv_;
    Task cur_{};
    int curReps_ = 1;
    long seq_ = 0;  // bumped under m_ per dispatch/stop
    int pending_ = 0;
    bool stop_ = false;
};

// Pool registry keyed by participant count so suites can interleave thread
// settings without tearing down workers.
inline ThreadPool& poolFor(int threads) {
    static std::unordered_map<int, ThreadPool*> pools;
    static std::mutex m;
    std::lock_guard<std::mutex> lk(m);
    auto it = pools.find(threads);
    if (it == pools.end()) {
        auto* p = new ThreadPool(threads);
        pools[threads] = p;
        return *p;
    }
    return *it->second;
}

// Free-function parallel_for used by kernels: dispatches through the pooled
// instance for threads>1, inline otherwise. The `reps` variant repeats the
// body on each chunk without re-dispatching (throughput-mode measurement).
template <typename Fn>
void parallelForRep(int begin, int end, int threads, int reps, Fn&& body) {
    if (threads <= 1 || end - begin <= 1) {
        for (int i = 0; i < reps; ++i) body(begin, end);
        return;
    }
    poolFor(threads).parallelForRep(begin, end, reps, body);
}

template <typename Fn>
void parallelFor(int begin, int end, int threads, Fn&& body) {
    parallelForRep(begin, end, threads, 1, std::forward<Fn>(body));
}

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------
class Stopwatch {
public:
    void start() { t0 = clock::now(); }
    double stopMs() {
        auto t1 = clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

private:
    using clock = std::chrono::steady_clock;
    clock::time_point t0{};
};

struct Sample {
    double msMedian = 0.0;
    double msMin = 0.0;  // best batch: timing noise is one-sided (adds
                         // time), so min is the stable cross-run statistic
    double cvPct = 0.0;  // stddev / mean * 100 across batches
    double gflops = 0.0; // computed from msMin
    double gbps = 0.0;   // effective traffic if provided
};

// `fn(count)` runs the kernel count times; kernels that own a dispatch
// receive the count and hoist it into a single parallelForRep so one
// condvar round-trip covers the whole batch. Per-sample time is divided
// by count, i.e. reported numbers are steady-state per-invocation.
template <typename Setup, typename Fn>
Sample measure(int warmup, int reps, double flopsPerCall, double bytesPerCall,
               Setup&& setup, Fn&& fn) {
    constexpr double kTargetBatchMs = 3.0;
    auto runBatch = [&](int count) {
        Stopwatch sw;
        sw.start();
        fn(count);
        return sw.stopMs();
    };
    // thermal prime: long enough for the multi-core governor to settle
    // into sustained clocks; duration overridable for experiments.
    double primeMs = 400.0;
    if (const char* e = std::getenv("BENCH_PRIME_MS")) {
        char* end = nullptr;
        double v = std::strtod(e, &end);
        if (end && *end == '\0' && v >= 0) primeMs = v;
    }
    {
        Stopwatch sw;
        while (sw.stopMs() < primeMs) {
            setup(0);
            runBatch(8);
        }
    }
    int inner = 1;
    for (;;) {
        setup(0);
        double ms = runBatch(inner);
        if (ms >= kTargetBatchMs || inner >= (1 << 14)) break;
        inner = ms < kTargetBatchMs / 4 ? inner * 8 : inner * 3;
        if (inner < 1) inner = 1;
    }
    std::vector<double> xs;
    xs.reserve(static_cast<size_t>(reps));
    for (int i = 0; i < warmup; ++i) {
        setup(i);
        runBatch(inner);
    }
    for (int i = 0; i < reps; ++i) {
        setup(i);
        xs.push_back(runBatch(inner) / inner);
    }
    std::sort(xs.begin(), xs.end());
    double median = xs[xs.size() / 2];
    if (xs.size() % 2 == 0)
        median = 0.5 * (xs[xs.size() / 2 - 1] + xs[xs.size() / 2]);
    double mean = 0.0;
    for (double x : xs) mean += x;
    mean /= static_cast<double>(xs.size());
    double var = 0.0;
    for (double x : xs) var += (x - mean) * (x - mean);
    var /= static_cast<double>(xs.size());
    Sample s;
    s.msMedian = median;
    s.msMin = xs.front();
    s.cvPct = mean > 0 ? 100.0 * std::sqrt(var) / mean : 0.0;
    s.gflops = flopsPerCall / (s.msMin * 1e-3) / 1e9;
    s.gbps = bytesPerCall / (s.msMin * 1e-3) / 1e9;
    return s;
}

// ---------------------------------------------------------------------------
// CSV output: one row per measurement.
// ---------------------------------------------------------------------------
struct Row {
    std::string suite;
    std::string kernel;
    std::string params;
    int threads = 1;
    std::string qos;
    double ms = 0.0;      // median across timing batches
    double msBest = 0.0;  // min across timing batches (stable statistic)
    double cvPct = 0.0;
    double gflops = 0.0;  // computed from msBest
    double gbps = 0.0;
};

class CsvWriter {
public:
    explicit CsvWriter(const std::string& path) {
        f_ = path.empty() ? stdout : std::fopen(path.c_str(), "w");
        if (f_)
            std::fprintf(f_,
                         "suite,kernel,params,threads,qos,ms_median,ms_min,"
                         "cv_pct,gflops,gbps\n");
    }
    ~CsvWriter() {
        if (f_ && f_ != stdout) std::fclose(f_);
    }
    bool ok() const { return f_ != nullptr; }
    void add(Row r) {
        r.qos = r.qos.empty() ? currentQosName() : r.qos;
        std::fprintf(f_, "%s,%s,%s,%d,%s,%.4f,%.4f,%.2f,%.3f,%.3f\n",
                     csv(r.suite).c_str(), csv(r.kernel).c_str(),
                     csv(r.params).c_str(), r.threads,
                     csv(r.qos).c_str(), r.ms, r.msBest, r.cvPct, r.gflops,
                     r.gbps);
    }

private:
    static std::string csv(const std::string& v) {
        if (v.find_first_of(",\"\n") == std::string::npos) return v;
        std::string q = "\"";
        for (char c : v) {
            q += c;
            if (c == '"') q += '"';
        }
        q += '"';
        return q;
    }
    std::FILE* f_ = nullptr;
};

// ---------------------------------------------------------------------------
// Machine description used for peak-FLOPS accounting.
//
// Peak model assumptions (documented in tools/bench/README.md):
//   * Apple M4 P-core issue width for fp32 NEON FMA: 4 pipes x (4 lanes x
//     2 flop) = 32 flop/cycle/core; E-core counted at half rate.
//   * Frequency is NOT assumed constant: callers may pass a measured GHz
//     (median of the powermetrics curve); default is the boost value.
//   * The honest utilization metric is measured / Accelerate-sgemm-same-
//     scale, which report.py also prints alongside theory ratios.
// ---------------------------------------------------------------------------
struct MachineProfile {
    double pCores = 4;
    double eCores = 6;
    double pFreqGhz = 4.41;  // override with measured curve median
    double eFreqGhz = 2.60;

    // Theoretical fp32 NEON peaks (GFLOPS).
    double peakSingleP() const { return pFreqGhz * 32.0; }
    double peakAllP() const { return peakSingleP() * pCores; }
    double peakAllE() const { return eFreqGhz * 16.0 * eCores; }
};

}  // namespace bench
}  // namespace gsv
