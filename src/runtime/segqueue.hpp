// segqueue.hpp — segment-level double-buffered pipeline skeleton (D2-prep).
//
// M5 prerequisite: "AR produces segment N+1 while VITS consumes segment N"
// (ARCHITECTURE.md §2). Pure skeleton — no model code; C2 integration will
// plug the real three stages in.
//
// Components:
//   * SegQueue<T>     bounded MPMC queue, condvar-based by design (we are
//                     not chasing lock-free: a segment is milliseconds of
//                     work, a wakeup is microseconds). Capacity defaults to
//                     2 = the double buffer. Producers backpressure when
//                     full, consumers block when empty, close() wakes both.
//   * SegTiming       per-segment timing hooks (t_textfront/t_ar/t_sov/
//                     t_wait). The envelope carries one shared SegTiming
//                     through all hops; the pipeline fills each field and
//                     accumulates queue wait automatically — user stage
//                     functions never touch clocks.
//   * SegmentPipeline three-stage worker chain (textfront -> ar -> sov),
//                     one QoS label per stage, graceful shutdown, exception
//                     transport via std::exception_ptr on the envelope.
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>

#include <pthread.h>
#include <sys/qos.h>

namespace gsv {
namespace runtime {

using SteadyClock = std::chrono::steady_clock;
using TimePoint = SteadyClock::time_point;

// Per-segment timing hooks for D1-part2 full-link reporting. One instance
// travels with a segment from submit() to drain(); t_wait accumulates the
// time spent sitting in every intermediate queue.
struct SegTiming {
    double t_textfrontMs = 0.0;
    double t_arMs = 0.0;
    double t_sovMs = 0.0;
    double t_waitMs = 0.0;

    void add(const SegTiming& o) {
        t_textfrontMs += o.t_textfrontMs;
        t_arMs += o.t_arMs;
        t_sovMs += o.t_sovMs;
        t_waitMs += o.t_waitMs;
    }
};

namespace detail {

// QoS label -> calling thread. Workers call this at entry so each stage can
// land on its intended cluster (§4: user_initiated -> P cores, utility ->
// E cores). Same mapping as src/kern/bench_harness.hpp. Defined in
// segqueue.cpp (non-template, single definition point).
bool applyQos(const std::string& name);

}  // namespace detail

// ---------------------------------------------------------------------------
// SegQueue<T> — bounded blocking MPMC queue of segments.
//
// Semantics:
//   * push(Item): blocks while full (backpressure); false once closed.
//   * pop(Item&): blocks while empty; false only after close()+drain.
//   * close(): idempotent, wakes all waiters; subsequent pushes fail.
//   * *_for variants bound the wait for tests/polling; they never alter the
//     item on failure.
// ---------------------------------------------------------------------------
template <typename T>
class SegQueue {
public:
    struct Item {
        T data{};
        std::exception_ptr exc{nullptr};
        std::shared_ptr<SegTiming> timing{nullptr};  // pipeline-internal
        TimePoint enqueuedAt{};                      // set by push()
    };

    explicit SegQueue(size_t capacity) : cap_(capacity < 1 ? 1 : capacity) {}
    SegQueue(const SegQueue&) = delete;
    SegQueue& operator=(const SegQueue&) = delete;

    size_t capacity() const { return cap_; }

    bool push(Item&& it) {
        it.enqueuedAt = SteadyClock::now();
        {
            std::unique_lock<std::mutex> lk(m_);
            fullCv_.wait(lk, [this] { return q_.size() < cap_ || closed_; });
            if (closed_) return false;
            q_.push_back(std::move(it));
        }
        itemCv_.notify_one();
        return true;
    }

    bool pop(Item& out) {
        {
            std::unique_lock<std::mutex> lk(m_);
            itemCv_.wait(lk, [this] { return !q_.empty() || closed_; });
            if (q_.empty()) return false;  // closed && drained
            out = std::move(q_.front());
            q_.pop_front();
        }
        fullCv_.notify_one();
        return true;
    }

    // Bounded-wait variants: true on success, false on timeout (queue state
    // unchanged from the caller's perspective).
    bool pushFor(Item&& it, std::chrono::milliseconds timeout) {
        it.enqueuedAt = SteadyClock::now();
        {
            std::unique_lock<std::mutex> lk(m_);
            if (!fullCv_.wait_for(lk, timeout, [this] {
                    return q_.size() < cap_ || closed_;
                }))
                return false;
            if (closed_) return false;
            q_.push_back(std::move(it));
        }
        itemCv_.notify_one();
        return true;
    }

    bool popFor(Item& out, std::chrono::milliseconds timeout) {
        {
            std::unique_lock<std::mutex> lk(m_);
            if (!itemCv_.wait_for(lk, timeout, [this] {
                    return !q_.empty() || closed_;
                }))
                return false;
            if (q_.empty()) return false;
            out = std::move(q_.front());
            q_.pop_front();
        }
        fullCv_.notify_one();
        return true;
    }

    void close() {
        {
            std::lock_guard<std::mutex> lk(m_);
            closed_ = true;
        }
        itemCv_.notify_all();
        fullCv_.notify_all();
    }

    bool closed() const {
        std::lock_guard<std::mutex> lk(m_);
        return closed_;
    }

    size_t sizeUnsafe() const {
        std::lock_guard<std::mutex> lk(m_);
        return q_.size();
    }

private:
    mutable std::mutex m_;
    std::condition_variable itemCv_;
    std::condition_variable fullCv_;
    std::deque<Item> q_;
    size_t cap_;
    bool closed_ = false;
};

// ---------------------------------------------------------------------------
// SegmentPipeline — textfront -> ar -> sov over two SegQueues (cap 2 = the
// "produce N+1 / consume N" double buffer; the output queue adds a third
// slot so the consumer can lag one finished segment without stalling AR).
//
// Types: TText --s1--> TAr --s2--> TSov --s3--> TWav.
// Stage functions: TWavOut fn(const In&, SegTiming&); throw to poison the
// segment — the exception rides downstream in-order and is rethrown at
// drain(); later stages skip poisoned payloads.
//
// Shutdown: shutdownInput() closes the entry queue; every stage then drains
// and retires, closing its output queue in turn; drain() returns false when
// the wav side is exhausted. cancel() drops in-flight work immediately.
// ---------------------------------------------------------------------------
template <typename TText, typename TAr, typename TSov, typename TWav>
class SegmentPipeline {
public:
    using TextQ = SegQueue<TText>;
    using ArQ = SegQueue<TAr>;
    using SovQ = SegQueue<TSov>;
    using WavQ = SegQueue<TWav>;
    using WavItem = typename WavQ::Item;

    using Stage1Fn = std::function<TAr(const TText&, SegTiming&)>;
    using Stage2Fn = std::function<TSov(const TAr&, SegTiming&)>;
    using Stage3Fn = std::function<TWav(const TSov&, SegTiming&)>;

    SegmentPipeline(Stage1Fn s1, Stage2Fn s2, Stage3Fn s3,
                    std::string qos1, std::string qos2, std::string qos3,
                    size_t queueCap = 2)
        : s1_(std::move(s1)), s2_(std::move(s2)), s3_(std::move(s3)),
          qos1_(std::move(qos1)), qos2_(std::move(qos2)),
          qos3_(std::move(qos3)), qIn_(queueCap), qMid1_(queueCap),
          qMid2_(queueCap), qOut_(queueCap) {}

    SegmentPipeline(const SegmentPipeline&) = delete;
    SegmentPipeline& operator=(const SegmentPipeline&) = delete;

    ~SegmentPipeline() {
        shutdownInput();
        cancel();
        joinAll();
    }

    void start() {
        if (running_.exchange(true)) return;
        t1_ = std::thread([this] { runStage(1, qIn_, qMid1_, qos1_, s1_); });
        t2_ = std::thread([this] { runStage(2, qMid1_, qMid2_, qos2_, s2_); });
        t3_ = std::thread([this] { runStage(3, qMid2_, qOut_, qos3_, s3_); });
    }

    // Backpressures while the entry queue is full; false after input close.
    bool submit(TText seg) {
        typename TextQ::Item it;
        it.data = std::move(seg);
        it.timing = std::make_shared<SegTiming>();
        return qIn_.push(std::move(it));
    }

    bool submitFor(TText seg, std::chrono::milliseconds timeout) {
        typename TextQ::Item it;
        it.data = std::move(seg);
        it.timing = std::make_shared<SegTiming>();
        return qIn_.pushFor(std::move(it), timeout);
    }

    // Fetch one finished segment. False once fully drained after input
    // close. Stage exceptions are DELIVERED, not rethrown: a poisoned
    // segment arrives with item.exc set (data undefined) and the consumer
    // decides what to do — rethrow for fail-fast, log-and-continue for
    // batch resilience. erroredCount() counts poisoned deliveries.
    bool drain(WavItem& out) {
        if (!qOut_.pop(out)) return false;
        ++completed_;
        if (out.exc) ++errored_;
        return true;
    }

    bool drainFor(WavItem& out, std::chrono::milliseconds timeout) {
        if (!qOut_.popFor(out, timeout)) return false;
        ++completed_;
        if (out.exc) ++errored_;
        return true;
    }

    // Graceful producer-side stop: in-flight segments still complete.
    void shutdownInput() { qIn_.close(); }

    // Drop everything queued upstream so workers exit promptly (fatal paths,
    // destructor safety). Idempotent.
    void cancel() {
        qIn_.close();
        qMid1_.close();
        qMid2_.close();
        qOut_.close();
    }

    void joinAll() {
        if (t1_.joinable()) t1_.join();
        if (t2_.joinable()) t2_.join();
        if (t3_.joinable()) t3_.join();
    }

    uint64_t completedCount() const { return completed_.load(); }
    uint64_t erroredCount() const { return errored_.load(); }
    size_t inflight() const {
        return qIn_.sizeUnsafe() + qMid1_.sizeUnsafe() + qMid2_.sizeUnsafe() +
               qOut_.sizeUnsafe();
    }

    struct TimingReport {
        SegTiming mean;
        uint64_t samples = 0;
        std::deque<SegTiming> recent;  // last kTimingRing individual timings
    };

    // Aggregate timing snapshot for D1-part2 wiring.
    TimingReport timingReport() const {
        std::lock_guard<std::mutex> lk(timingMtx_);
        TimingReport r;
        r.samples = timingCount_;
        r.recent = lastTimings_;
        if (timingCount_) {
            r.mean = sum_;
            double n = static_cast<double>(timingCount_);
            r.mean.t_textfrontMs /= n;
            r.mean.t_arMs /= n;
            r.mean.t_sovMs /= n;
            r.mean.t_waitMs /= n;
        }
        return r;
    }

private:
    template <typename InQ, typename OutQ, typename Fn>
    void runStage(int stageIdx, InQ& in, OutQ& out, const std::string& qos,
                  Fn& fn) {
        detail::applyQos(qos);
        typename InQ::Item it;
        while (in.pop(it)) {
            if (!it.timing) it.timing = std::make_shared<SegTiming>();
            SegTiming& tm = *it.timing;
            // queue wait accumulated BEFORE processing so the field reflects
            // pure waiting regardless of what the stage does to clocks
            tm.t_waitMs +=
                std::chrono::duration<double, std::milli>(SteadyClock::now() -
                                                          it.enqueuedAt)
                    .count();
            typename OutQ::Item o;
            o.timing = it.timing;
            if (!it.exc) {
                TimePoint t0 = SteadyClock::now();
                try {
                    o.data = fn(it.data, tm);
                } catch (...) {
                    o.exc = std::current_exception();
                }
                double dt = std::chrono::duration<double, std::milli>(
                                SteadyClock::now() - t0)
                                .count();
                if (stageIdx == 1) tm.t_textfrontMs += dt;
                else if (stageIdx == 2) tm.t_arMs += dt;
                else tm.t_sovMs += dt;
            } else {
                o.exc = it.exc;  // poison flows downstream untouched
            }
            if (stageIdx == 3) recordTiming(tm);
            if (!out.push(std::move(o))) break;  // downstream cancelled
        }
        out.close();  // retire: next stage drains then exits likewise
    }

    void recordTiming(const SegTiming& tm) {
        std::lock_guard<std::mutex> lk(timingMtx_);
        lastTimings_.push_back(tm);
        if (lastTimings_.size() > kTimingRing) lastTimings_.pop_front();
        sum_.add(tm);
        ++timingCount_;
    }

    static constexpr size_t kTimingRing = 64;

    Stage1Fn s1_;
    Stage2Fn s2_;
    Stage3Fn s3_;
    std::string qos1_, qos2_, qos3_;

    TextQ qIn_;
    ArQ qMid1_;
    SovQ qMid2_;
    WavQ qOut_;

    std::thread t1_, t2_, t3_;
    std::atomic<bool> running_{false};
    std::atomic<uint64_t> completed_{0}, errored_{0};

    mutable std::mutex timingMtx_;
    std::deque<SegTiming> lastTimings_;
    SegTiming sum_{};
    uint64_t timingCount_ = 0;
};

}  // namespace runtime
}  // namespace gsv
