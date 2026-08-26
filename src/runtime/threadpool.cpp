// threadpool.cpp — QoS 分簇持久线程池实现（mutex+condvar 简单队列, 够用为先）
#include "threadpool.hpp"

#include <pthread.h>
#include <sys/qos.h>
#include <sys/sysctl.h>

#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <thread>
#include <mutex>
#include <thread>
#include <vector>

namespace gsv::rt {

namespace {

size_t sysctl_size(const char* name, size_t fallback) {
  size_t v = 0;
  size_t len = sizeof v;
  if (::sysctlbyname(name, &v, &len, nullptr, 0) != 0 || v == 0) return fallback;
  return v;
}

struct Task {
  std::function<void()> fn;
};

class Pool {
 public:
  explicit Pool(size_t n_threads, qos_class_t qos) : qos_(qos), stop_(false) {
    threads_.reserve(n_threads);
    for (size_t i = 0; i < n_threads; ++i)
      threads_.emplace_back([this] { worker_loop(); });
  }

  ~Pool() {
    {
      std::lock_guard<std::mutex> lk(mu_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto& t : threads_) t.join();
  }

  void submit(std::function<void()> fn) {
    {
      std::lock_guard<std::mutex> lk(mu_);
      tasks_.push_back(std::move(fn));
    }
    cv_.notify_one();
  }

  // 提交一批任务并等待全部完成(代际计数, 避免逐任务唤醒开销)
  void run_batch(std::vector<std::function<void()>> batch) {
    const size_t n = batch.size();
    if (n == 0) return;
    std::unique_lock<std::mutex> lk(mu_);
    for (auto& t : batch) tasks_.push_back(std::move(t));
    pending_ += n;
    cv_.notify_all();
    done_cv_.wait(lk, [&] { return pending_ == 0; });
  }

 private:
  void worker_loop() {
    ::pthread_set_qos_class_self_np(qos_, 0);
    std::unique_lock<std::mutex> lk(mu_);
    while (true) {
      cv_.wait(lk, [&] { return stop_ || !tasks_.empty(); });
      if (stop_ && tasks_.empty()) return;
      Task t{std::move(tasks_.front())};
      tasks_.pop_front();
      lk.unlock();
      t.fn();
      lk.lock();
      if (--pending_ == 0 && tasks_.empty()) done_cv_.notify_all();
    }
  }

  qos_class_t qos_;
  bool stop_;
  size_t pending_ = 0;
  std::mutex mu_;
  std::condition_variable cv_;
  std::condition_variable done_cv_;
  std::deque<std::function<void()>> tasks_;
  std::vector<std::thread> threads_;
};

Pool& pool_for(Qos q) {
  static Pool p_user(p_core_count(), QOS_CLASS_USER_INITIATED);
  static Pool p_util(e_core_count() > 0 ? e_core_count() : 1, QOS_CLASS_UTILITY);
  return q == Qos::UserInitiated ? p_user : p_util;
}

// E11-4: GEMV 专用全核池 —— P + E 合一个队列, QoS 为 UserInitiated (GEMV 可占 P 核);
// 10 worker 永久等 condvar, run_batch 一次派发 10 task, 免 std::thread 边边成本。
// 适用于纯 NEON GEMV/GEMM 路径(AMX 已限定走 P 池, 见 amx_pool)。
// E11-4: GEMV 调度走 GemvDualDispatcher.

// E11-4: 全核联合派发调度器 —— 持久 helper 线程(绑 P 核 QoS UserInitiated)
// 接收 E 池任务信号并在 E 核上跑 E 池 run_batch; 同时调用线程跑 P 池 run_batch。
//   避免 std::thread 每次创建的 ~50us 边边成本; 改为 1~2us 的 condvar 信号。
//   helper 自身占一个 P 核(P 池仍 4 核, 但实际可用 3 核并发), 换得 E 核全阵
//   参与 GEMV/GEMM(内存带宽型负载) —— 预期 decode GEMV 从 70→100 GB/s。
class GemvDualDispatcher {
 public:
  GemvDualDispatcher() : has_work_(false), stop_(false) {
    helper_ = std::thread([this] { helper_loop(); });
  }
  ~GemvDualDispatcher() {
    {
      std::lock_guard<std::mutex> lk(mu_);
      stop_ = true;
    }
    cv_.notify_all();
    if (helper_.joinable()) helper_.join();
  }
  // 调用线程: 同时驱 P 池(在自己) + E 池(在 helper 线程上)并行, join 同步
  void run(std::vector<std::function<void()>> p_batch,
           std::vector<std::function<void()>> e_batch) {
    if (e_batch.empty()) {
      pool_for(Qos::UserInitiated).run_batch(std::move(p_batch));
      return;
    }
    if (p_batch.empty()) {
      pool_for(Qos::Utility).run_batch(std::move(e_batch));
      return;
    }
    {
      std::lock_guard<std::mutex> lk(mu_);
      pending_e_ = std::move(e_batch);
      has_work_ = true;
    }
    cv_.notify_one();
    pool_for(Qos::UserInitiated).run_batch(std::move(p_batch));
    // 等 E 池完成
    std::unique_lock<std::mutex> lk(mu_);
    e_done_cv_.wait(lk, [this] { return !has_work_; });
  }

 private:
  void helper_loop() {
    // helper 线程本身绑 P 核(能同时与 P 池 worker 抢核), 但驱动的 E 池 worker 在 E 核。
    // helper 自身不抢业务 task, 只 调 E 池 run_batch。
    ::pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);
    std::unique_lock<std::mutex> lk(mu_);
    while (true) {
      cv_.wait(lk, [this] { return has_work_ || stop_; });
      if (stop_) return;
      std::vector<std::function<void()>> e_batch = std::move(pending_e_);
      lk.unlock();
      pool_for(Qos::Utility).run_batch(std::move(e_batch));
      lk.lock();
      has_work_ = false;
      e_done_cv_.notify_one();
    }
  }
  std::mutex mu_;
  std::condition_variable cv_;
  std::condition_variable e_done_cv_;
  std::vector<std::function<void()>> pending_e_;
  bool has_work_;
  bool stop_;
  std::thread helper_;
};

GemvDualDispatcher& gemv_dispatcher() {
  static GemvDualDispatcher d;
  return d;
}

}  // namespace

size_t p_core_count() { return sysctl_size("hw.perflevel0.logicalcpu", std::thread::hardware_concurrency()); }
size_t e_core_count() { return sysctl_size("hw.perflevel1.logicalcpu", 0); }

void parallel_for(size_t n, size_t grain,
                  const std::function<void(size_t, size_t)>& fn, Qos qos) {
  if (n == 0) return;
  const size_t g = grain == 0 ? 1 : grain;
  if (n <= g) {  // 内联路径
    fn(0, n);
    return;
  }
  const size_t nchunks = (n + g - 1) / g;
  std::vector<std::function<void()>> batch;
  batch.reserve(nchunks);
  for (size_t c = 0; c < nchunks; ++c) {
    const size_t b = c * g, e = b + g < n ? b + g : n;
    batch.emplace_back([&fn, b, e] { fn(b, e); });
  }
  pool_for(qos).run_batch(std::move(batch));
}

bool gemv_use_e_cores() {
  // E11-4 验收裁决: 端到端无收益(5.76 vs 5.82 不可区分, L2复用已超4核流式带宽),
  // 默认关闭; GSV_GEMV_E_ENABLE=1 实验开启
  static const bool b = []() {
    const char* e = std::getenv("GSV_GEMV_E_ENABLE");
    return e != nullptr && e[0] == '1';
  }();
  return b;
}

void parallel_for_full(size_t n, size_t grain,
                       const std::function<void(size_t, size_t)>& fn) {
  if (n == 0) return;
  const size_t g = grain == 0 ? 1 : grain;
  // E11-4: 工作量小/E 核不可用/被探针关掉 → 退到 P-only (避免多池合并开销)
  const size_t ec = gemv_use_e_cores() ? e_core_count() : 0;
  if (n <= g || ec == 0) {
    fn(0, n);
    return;
  }
  // E11-4: 任务量 = min(全核数 * K, 实际量/K) —— K=2 是低开销上限。
  // 每 worker 多个 task 防尾慢(E 核频率低, P 核先抢完会回头帮抢; 4 P 核 × 2 task × 38 rows
  // ≈ P-only 4 任务的等价 work, 但额外得 6 E 核全数抢剩余 work)。
  const size_t pc = p_core_count();
  // 一轮: 任务数 = total_workers, 每核 1 task (最小开销)
  // 负载均衡: P 拿 1.5x 任务大小 (pe_ratio 2 含义是 per_core_p/per_core_e, 即 P 任务大小 / E 任务大小)
  static const size_t pe_ratio = []() {
    const char* e = std::getenv("GSV_GEMV_PE_RATIO");
    if (e == nullptr) return size_t{2};
    long v = std::atol(e);
    return v > 0 ? static_cast<size_t>(v) : size_t{2};
  }();
  // total P_work = pc * chunk_p, total E_work = ec * chunk_e
  // chunk_p / chunk_e = pe_ratio → chunk_p = pe_ratio * chunk_e
  // pc * pe_ratio * chunk_e + ec * chunk_e = n
  // chunk_e = n / (pc * pe_ratio + ec)
  const size_t chunk_e = (n + pc * pe_ratio + ec - 1) / (pc * pe_ratio + ec);
  const size_t chunk_p = chunk_e * pe_ratio;
  // 构造 P 池批( pc 个任务)
  std::vector<std::function<void()>> p_batch;
  p_batch.reserve(pc);
  for (size_t i = 0; i < pc; ++i) {
    const size_t b = i * chunk_p;
    const size_t end = std::min(b + chunk_p, n);
    if (b >= end) break;
    p_batch.emplace_back([&fn, b, end] { fn(b, end); });
  }
  // 构造 E 池批( ec 个任务)
  const size_t e_offset = pc * chunk_p;
  std::vector<std::function<void()>> e_batch;
  e_batch.reserve(ec);
  for (size_t i = 0; i < ec; ++i) {
    const size_t b = e_offset + i * chunk_e;
    const size_t end = std::min(b + chunk_e, n);
    if (b >= end) break;
    e_batch.emplace_back([&fn, b, end] { fn(b, end); });
  }
  // 双池并行: 调用线程 P 池, helper 线程 E 池 —— 真正占 P 核(3) + E 核(6)全数
  gemv_dispatcher().run(std::move(p_batch), std::move(e_batch));
}

}  // namespace gsv::rt
