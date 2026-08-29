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
  return false;
}

void parallel_for_full(size_t n, size_t /*grain*/,
                       const std::function<void(size_t, size_t)>& fn) {
  if (n == 0) return;
  fn(0, n);
}

}  // namespace gsv::rt
