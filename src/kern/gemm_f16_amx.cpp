// gemm_f16_amx.cpp — E5 实现 (见 gemm_f16_amx.hpp 契约)
//
// Phase 1 内核设计:
//   tile 32×32 (X lanes = M, Y lanes = N, K 全展开逐 k 外积)
//   每 tile: K×(LDX+LDY+MATFP), 64×LDZ 清零, 64×STZ 读回散布到 C
//   panel 预打包: A[mb] 块转置 [K][32] f16 (X 直读布局); B 同构
//
// 线程模型 (与任务卡"线程池 worker 各自 AMX_SET"的偏差声明):
//   不能复用 rt 现有 P 核池 —— 实测 Accelerate BLAS (sgemm) 在调用线程
//   同样驱动 AMX, 与我们长驻的 AMX_SET 状态互斥 (先 SET 再 cblas → SIGILL;
//   先 cblas 再 SET → 结果全零)。现有 P 池 worker 会承接 cblas 任务,
//   因此 AMX 使用专用私有线程池 (AMX-only, 永不触碰 Accelerate), 每个
//   worker 启动即探测 + AMX_SET 长驻。功能等价且隔离安全。
#include "kern/gemm_f16_amx.hpp"

#if defined(GSV_AMX_GEMM)

#include "kern/amx.h"
#include "kern/gemv_fmlal.hpp"

#include <arm_neon.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <functional>
#include <mutex>
#include <setjmp.h>
#include <signal.h>
#include <sys/sysctl.h>
#include <thread>
#include <vector>

namespace gsv::kern {

namespace {

static thread_local bool t_is_amx_worker = false;

// ---------------- SIGILL 探测 (并发安全) ----------------
// 处理器全程长驻安装 (首次探测时装, 不再恢复): handler 只服务探测,
// 且 buf/in_probe 均为 thread_local —— 任意线程 SIGILL 只会跳回自己的
// 探测点。真 SIGILL (探测外) 不再转发: 保持进程可被默认终止, 便于定位。
// 所有探测串行化 (全局互斥), 杜绝 install/restore 交叠竞态。
static thread_local sigjmp_buf* t_probe_jmp = nullptr;  // 当前线程在探测中的 buf
static std::atomic<bool> g_handler_installed{false};
static std::mutex g_probe_mu;

void probe_sigill_handler(int) {
  if (t_probe_jmp) siglongjmp(*t_probe_jmp, 1);
  // 探测之外的 SIGILL: 交回默认处理
  struct sigaction sa {};
  sa.sa_handler = SIG_DFL;
  sigaction(SIGILL, &sa, nullptr);
}

// 当前线程探测全套用到的 AMX 指令; 成功后保持 SET 长驻并返回 true。
bool probe_and_set_amx() {
  std::lock_guard<std::mutex> lk(g_probe_mu);
  if (!g_handler_installed.load(std::memory_order_acquire)) {
    struct sigaction sa {};
    sa.sa_handler = probe_sigill_handler;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGILL, &sa, nullptr);
    g_handler_installed.store(true, std::memory_order_release);
  }
  sigjmp_buf jmp;
  bool ok = false;
  if (sigsetjmp(jmp, 1) == 0) {
    t_probe_jmp = &jmp;
    __attribute__((aligned(64))) uint8_t buf[64] = {0};
    AMX_SET();
    AMX_LDX(amx::ld_op(buf, 0));
    AMX_LDY(amx::ld_op(buf, 0));
    AMX_LDZ(amx::ld_op(buf, 0));
    AMX_STZ(amx::ld_op(buf, 0));
    AMX_MATFP(amx::matfp_f16f32(0, 0));
    ok = true;  // 探测序列自带 SET, 保持长驻; 探测写入的 Z 零/脏值无碍
                // —— 业务 tile 前每 tile 都会 LDZ 重置全部 64 行
  }
  t_probe_jmp = nullptr;
  return ok;
}

// ---------------- AMX 专用线程池 ----------------
struct AmxPool {
  std::mutex mu;
  std::mutex dispatch_mu;  // E15: 派发互斥 — 保护 run_phased/submit_graph/submit_dag 并发派发安全
  std::condition_variable cv, done_cv;
  struct Task {
    int phase;
    std::function<void()> fn;
    std::function<void()> on_done;  // mu 持有下于完成后执行 (可入队后续任务)
  };
  std::deque<Task> ready;
  std::vector<std::deque<Task>> future_phases;  // 未解锁相位 (任务自带相位)
  std::vector<size_t> phase_left;                                 // 各相位未完数
  size_t pending = 0;  // 全部未完成任务数 (含未解锁相位)
  bool stop = false;
  std::vector<std::thread> threads;
  std::atomic<int> healthy{0};   // 探测成功的 worker 数
  std::atomic<int> probed{0};    // 已完成探测的 worker 数 (含失败)

  // E10-K: 按-chain DAG 调度状态 (amx_chain_run 使用)
  // dag_mode 开启时, finish_task 用 dag_remaining/dag_succs 代替
  // phase_left/future_phases, 实现节点就绪立即执行 (无全局 barrier)。
  bool dag_mode = false;
  std::vector<int> dag_deps;           // 每节点剩余前驱未完成数
  std::vector<int> dag_remaining;      // 每节点剩余任务 chunk 数
  std::vector<std::vector<int>> dag_succs;  // 每节点的后继节点索引列表
  std::vector<std::deque<Task>> dag_init_tasks;  // 节点解锁时入队的初始任务

  explicit AmxPool(size_t n) {
    threads.reserve(n);
    for (size_t i = 0; i < n; ++i)
      threads.emplace_back([this] { worker(); });
    // E9: 等全部 worker 完成探测再返回, 保证 available()/派发时结论就绪
    while (probed.load(std::memory_order_acquire) < int(n))
      std::this_thread::yield();
  }
  ~AmxPool() {
    {
      std::lock_guard<std::mutex> lk(mu);
      stop = true;
    }
    cv.notify_all();
    for (auto& t : threads) t.join();
  }

 private:
  // mu 已持: 入队就绪任务
  void push_ready(int phase, std::function<void()> fn,
                  std::function<void()> on_done = nullptr) {
    ready.push_back(Task{phase, std::move(fn), std::move(on_done)});
  }

  // mu 已持: 任务完成记账; 相位清空则解锁下一相位
  void finish_task(int phase) {
    if (phase < 0) return;  // 简单独立批任务: on_done 已就地更新调用方局部计数
    if (dag_mode) {
      // E10-K: 按-chain DAG 调度 —— 仅当本节点所有 chunks 完成时,
      // 才解锁其后继节点 (而非等待整个 phase). idx = node_idx。
      const size_t idx = size_t(phase);
      if (idx < dag_remaining.size() && --dag_remaining[idx] == 0) {
        for (int succ : dag_succs[idx]) {
          if (succ >= 0 && size_t(succ) < dag_deps.size() &&
              --dag_deps[succ] == 0) {
            auto& q = dag_init_tasks[succ];
            while (!q.empty()) {
              ready.push_back(std::move(q.front()));
              q.pop_front();
            }
          }
        }
        cv.notify_all();
      }
    } else {
      const size_t p = size_t(phase);
      if (p < phase_left.size() && --phase_left[p] == 0 &&
          p + 1 < future_phases.size()) {
        auto& nx = future_phases[p + 1];
        while (!nx.empty()) {
          ready.push_back(std::move(nx.front()));
          nx.pop_front();
        }
        cv.notify_all();
      }
    }
    if (--pending == 0) done_cv.notify_all();
  }

 public:
  void worker() {
    t_is_amx_worker = true;
    if (!probe_and_set_amx()) {  // 本线程无 AMX → 不接活
      probed.fetch_add(1, std::memory_order_release);
      return;
    }
    healthy.fetch_add(1, std::memory_order_release);
    probed.fetch_add(1, std::memory_order_release);
    std::unique_lock<std::mutex> lk(mu);
    while (true) {
      cv.wait(lk, [this] { return stop || !ready.empty(); });
      if (stop && ready.empty()) return;
      Task t = std::move(ready.front());
      ready.pop_front();
      lk.unlock();
      t.fn();
      lk.lock();
      if (t.on_done) t.on_done();  // 可能向 ready 入队后续任务
      finish_task(t.phase);
    }
  }
  // E9: 多相位批执行。phases[p] 的任务仅在相位 p-1 全部完成后入队;
  // 同相位任务被全部 worker 抢占并行。调用方阻塞一次至全完成。
  void run_phased(std::vector<std::vector<std::function<void()>>> phases) {
    size_t n = 0;
    for (auto& p : phases) n += p.size();
    if (n == 0) return;
    std::lock_guard<std::mutex> dlk(dispatch_mu);
    {
      std::lock_guard<std::mutex> lk(mu);
      future_phases.clear();
      future_phases.resize(phases.size());
      phase_left.assign(phases.size(), 0);
      size_t total = 0;
      for (size_t p = 0; p < phases.size(); ++p) {
        phase_left[p] = phases[p].size();
        total += phases[p].size();
        for (auto& f : phases[p]) future_phases[p].push_back(Task{int(p), std::move(f), nullptr});
      }
      pending += total;
      auto& ph0 = future_phases[0];
      while (!ph0.empty()) {
        ready.push_back(std::move(ph0.front()));
        ph0.pop_front();
      }
    }
    cv.notify_all();
    std::unique_lock<std::mutex> lk(mu);
    done_cv.wait(lk, [this] { return pending == 0; });
  }
  // E21: 无锁非阻塞独立批任务 (不占 dispatch_mu, 与 DAG/Phased 并发共存)
  void run_batch(std::vector<std::function<void()>> batch) {
    if (batch.empty()) return;
    const size_t total = batch.size();
    size_t left = total;
    std::condition_variable local_cv;
    {
      std::lock_guard<std::mutex> lk(mu);
      for (auto& f : batch) {
        ready.push_back(Task{
            -1,
            std::move(f),
            [&left, &local_cv] {
              if (--left == 0) {
                local_cv.notify_one();
              }
            }});
      }
      cv.notify_all();
    }
    std::unique_lock<std::mutex> lk(mu);
    local_cv.wait(lk, [&] { return left == 0; });
  }
  // E9: 相位化图提交。byphase[p] 为相位 p 的任务队列 (head 等); 含延迟
  // 入队者 (由某任务 on_done 推入) 也必须计入 phase_totals[p]。调用方
  // 保证所有计数任务最终完成。
  void submit_graph(size_t n_phases, const size_t* phase_totals,
                    std::vector<std::deque<Task>>& byphase) {
    size_t total = 0;
    for (size_t p = 0; p < n_phases; ++p) total += phase_totals[p];
    if (total == 0) return;
    std::lock_guard<std::mutex> dlk(dispatch_mu);
    {
      std::lock_guard<std::mutex> lk(mu);
      future_phases = std::move(byphase);
      future_phases.resize(n_phases);
      phase_left.assign(phase_totals, phase_totals + n_phases);
      pending += total;
      auto& ph0 = future_phases[0];
      while (!ph0.empty()) {
        ready.push_back(std::move(ph0.front()));
        ph0.pop_front();
      }
      cv.notify_all();
    }
    std::unique_lock<std::mutex> lk(mu);
    done_cv.wait(lk, [this] { return pending == 0; });
  }
  // E10-K: 按-chain DAG 图提交。deps[i] = 节点 i 的前驱未完成数;
  // succs[i] = 节点 i 的后继列表; remaining[i] = 节点 i 的总任务块数;
  // init_tasks[i] = 节点 i 解锁时入队的初始任务 (prepare 或 tiles)。
  // 与 submit_graph 异同: 不按 phase 屏障, 而按节点依赖图就绪即入队。
  void submit_dag(const std::vector<int>& deps,
                  const std::vector<std::vector<int>>& succs,
                  const std::vector<int>& remaining,
                  std::vector<std::deque<Task>> init_tasks) {
    size_t n = deps.size();
    size_t total = 0;
    for (size_t i = 0; i < n; ++i) total += remaining[i];
    if (total == 0) return;
    std::lock_guard<std::mutex> dlk(dispatch_mu);
    {
      std::lock_guard<std::mutex> lk(mu);
      dag_mode = true;
      dag_deps = deps;
      dag_succs = succs;
      dag_remaining = remaining;
      dag_init_tasks = std::move(init_tasks);
      pending += total;
      // 初始解锁: deps==0 的节点, 入队其 init_tasks
      for (size_t i = 0; i < n; ++i) {
        if (dag_deps[i] == 0) {
          auto& q = dag_init_tasks[i];
          while (!q.empty()) {
            ready.push_back(std::move(q.front()));
            q.pop_front();
          }
        }
      }
      cv.notify_all();
    }
    std::unique_lock<std::mutex> lk(mu);
    done_cv.wait(lk, [this] { return pending == 0; });
    dag_mode = false;
    dag_init_tasks.clear();
    dag_deps.clear();
    dag_succs.clear();
    dag_remaining.clear();
  }
};

size_t amx_pool_size() {
  if (const char* e = ::getenv("GSV_AMX_THREADS")) {
    const long v = std::atol(e);
    if (v > 0) return (size_t)v;
  }
  size_t v = 0, len = sizeof v;
  if (::sysctlbyname("hw.perflevel0.logicalcpu", &v, &len, nullptr, 0) != 0 ||
      v == 0) {
    v = std::thread::hardware_concurrency();
  }
  return v ? v : 1;
}

AmxPool& amx_pool() {
  static AmxPool pool(amx_pool_size());
  return pool;
}

// ---------------- panel 打包 ----------------
// pack[nt][K][32]: 第 t 块 tile 的 X/Y 直读布局 (每 k 一个 64B 行),
// 尾块行补零。src [rows,K] f16 行主。
// 4×4 f16 转置 (NEON): 输入 4 行×4 列, 输出 4 个列向量
// 依赖 armv8.2 fp16 (项目已要求 apple-m4)
inline void transpose4x4_f16(float16x4_t x0, float16x4_t x1,
                             float16x4_t x2, float16x4_t x3,
                             float16x4_t& o0, float16x4_t& o1,
                             float16x4_t& o2, float16x4_t& o3) {
  const float16x4x2_t a01 = vtrn_f16(x0, x1);
  const float16x4x2_t a23 = vtrn_f16(x2, x3);
  const float32x2x2_t b0 = vtrn_f32(
      vreinterpret_f32_f16(a01.val[0]), vreinterpret_f32_f16(a23.val[0]));
  const float32x2x2_t b1 = vtrn_f32(
      vreinterpret_f32_f16(a01.val[1]), vreinterpret_f32_f16(a23.val[1]));
  o0 = vreinterpret_f16_f32(b0.val[0]);
  o1 = vreinterpret_f16_f32(b1.val[0]);
  o2 = vreinterpret_f16_f32(b0.val[1]);
  o3 = vreinterpret_f16_f32(b1.val[1]);
}

// src [rows,K] f16 行主 → out: nt 个 [K][64B] panel (x[k*64+m*2]=src[r0+m][k])
// NEON: 4 行 × 8k 一块, 两个 4×4 转置, 8B 列写
void pack_panel(const uint16_t* src, size_t rows, size_t K,
                std::vector<uint8_t>& out) {
  const size_t nt = (rows + 31) / 32;
  out.assign(nt * K * 64 + 64, 0);
  const uintptr_t p = (uintptr_t)out.data();
  const size_t base = (64 - (p & 63)) & 63;
  uint8_t* dst = out.data() + base;
  for (size_t t = 0; t < nt; ++t) {
    const size_t r0 = t * 32;
    const size_t tr = std::min<size_t>(32, rows - r0);
    uint8_t* d = dst + t * K * 64;
    if (tr == 32) {  // 满 tile: 纯 NEON 快路径
      for (size_t r = 0; r < 32; r += 4) {
        const uint16_t* s0 = src + (r0 + r) * K;
        const uint16_t* s1 = s0 + K, *s2 = s0 + 2 * K, *s3 = s0 + 3 * K;
        size_t k = 0;
        for (; k + 4 <= K; k += 4) {
          float16x4_t o0, o1, o2, o3;
          transpose4x4_f16(vld1_f16((const __fp16*)(s0 + k)),
                           vld1_f16((const __fp16*)(s1 + k)),
                           vld1_f16((const __fp16*)(s2 + k)),
                           vld1_f16((const __fp16*)(s3 + k)), o0, o1, o2, o3);
          uint8_t* c = d + k * 64 + r * 2;
          vst1_f16((__fp16*)(c), o0);
          vst1_f16((__fp16*)(c + 64), o1);
          vst1_f16((__fp16*)(c + 128), o2);
          vst1_f16((__fp16*)(c + 192), o3);
        }
        for (; k < K; ++k) {  // K 尾
          uint8_t* c = d + k * 64 + r * 2;
          *(uint16_t*)(c) = s0[k];
          *(uint16_t*)(c + 2) = s1[k];
          *(uint16_t*)(c + 4) = s2[k];
          *(uint16_t*)(c + 6) = s3[k];
        }
      }
    } else {  // 尾 tile: 标量 (一次性成本, 少见)
      for (size_t r = 0; r < 32; r += 4) {
        const size_t nr = r < tr ? std::min<size_t>(4, tr - r) : 0;
        const uint16_t* s0 = nr > 0 ? src + (r0 + r) * K : nullptr;
        const uint16_t* s1 = nr > 1 ? s0 + K : nullptr;
        const uint16_t* s2 = nr > 2 ? s0 + 2 * K : nullptr;
        const uint16_t* s3 = nr > 3 ? s0 + 3 * K : nullptr;
        for (size_t k = 0; k < K; ++k) {
          uint8_t* col = d + k * 64 + r * 2;
          if (s0) std::memcpy(col, s0 + k, 2); else std::memset(col, 0, 2);
          if (s1) std::memcpy(col + 2, s1 + k, 2); else std::memset(col + 2, 0, 2);
          if (s2) std::memcpy(col + 4, s2 + k, 2); else std::memset(col + 4, 0, 2);
          if (s3) std::memcpy(col + 6, s3 + k, 2); else std::memset(col + 6, 0, 2);
        }
      }
    }
  }
}

// ---------------- tile 内核 (worker 线程, 已 SET) ----------------
void amx_tile(const uint8_t* Xbase, const uint8_t* Ybase, float* C,
              size_t M, size_t N, size_t K, size_t mb, size_t nb) {
  static thread_local __attribute__((aligned(64))) uint8_t z64[64] = {0};
  static thread_local __attribute__((aligned(64))) uint8_t zb[64][64];
  const size_t tm = std::min<size_t>(32, M - mb * 32);
  const size_t tn = std::min<size_t>(32, N - nb * 32);
  const uint8_t* X = Xbase + (mb * K) * 64;  // 块内 panel: [K][64B]
  const uint8_t* Y = Ybase + (nb * K) * 64;
  for (int z = 0; z < 64; ++z) AMX_LDZ(amx::ld_op(z64, (uint64_t)z));
  const uint64_t op = amx::matfp_f16f32(0, 0);
  for (size_t k = 0; k < K; ++k) {
    AMX_LDX((uint64_t)(X + k * 64));
    AMX_LDY((uint64_t)(Y + k * 64));
    AMX_MATFP(op);
  }
  for (int z = 0; z < 64; ++z) AMX_STZ(amx::ld_op(zb[z], (uint64_t)z));
  float* Crow = C + (mb * 32) * N + nb * 32;
  for (size_t n = 0; n < tn; ++n) {
    const uint8_t* fe = zb[n * 2];
    const uint8_t* fo = zb[n * 2 + 1];
    for (size_t m2 = 0; m2 < 16; ++m2) {
      const size_t m = m2 * 2;
      if (m < tm) std::memcpy(&Crow[m * N + n], fe + m2 * 4, 4);
      if (m + 1 < tm) std::memcpy(&Crow[(m + 1) * N + n], fo + m2 * 4, 4);
    }
  }
}

// ---------------- 回退 (fmlal 全量) ----------------
// panel 缓冲复用 (调用方线程; assign 保留容量免反复分配)
static thread_local std::vector<uint8_t> t_araw, t_braw;

// 预打包版核心: pa/pb 已就绪, 直接分发 tiles。
// 前置条件: AMX 已可用 (调用方保证), 输入形状已校验。
void gemm_pp_dispatch(const uint8_t* X, const uint8_t* Y, float* c,
                      size_t M, size_t N, size_t K) {
  const size_t nmb = (M + 31) / 32, nnb = (N + 31) / 32;
  const size_t ntiles = nmb * nnb;
  if (ntiles == 0) return;

  // 若当前已在 AMX worker 线程内执行, 直接内联运行 tiles (零重入死锁, 零调度开销)
  if (t_is_amx_worker) {
    for (size_t mb = 0; mb < nmb; ++mb) {
      for (size_t nb = 0; nb < nnb; ++nb) {
        amx_tile(X, Y, c, M, N, K, mb, nb);
      }
    }
    return;
  }

  AmxPool& pool = amx_pool();
  const int healthy = pool.healthy.load(std::memory_order_acquire);
  const size_t grain = healthy > 0 ? (ntiles + (size_t)healthy - 1) / (size_t)healthy : ntiles;
  std::vector<std::function<void()>> batch;
  for (size_t t = 0; t < ntiles; t += grain) {
    const size_t e = std::min(ntiles, t + grain);
    batch.emplace_back([&, t, e] {
      for (size_t i = t; i < e; ++i) {
        const size_t mb = i / nnb, nb = i % nnb;
        amx_tile(X, Y, c, M, N, K, mb, nb);
      }
    });
  }
  pool.run_batch(std::move(batch));
}

void gemm_f16_amx_impl(const uint16_t* a, const uint16_t* b, float* c,
                       size_t M, size_t N, size_t K) {
  if (M == 0 || N == 0) return;
  if (K == 0) {
    std::memset(c, 0, M * N * 4);
    return;
  }
  // 小规模: FMLAL 单次分派更省 (打包+池往返不划算)
  if ((M < 64 && N < 64) || M * N * K < (1u << 18)) {
    gemm_f16x_fmlal(a, b, c, M, N, K);
    return;
  }

  static std::atomic<bool> g_amx_disabled{false};  // 池全部 worker 探测失败
  AmxPool& pool = amx_pool();
  const int healthy = pool.healthy.load(std::memory_order_acquire);
  if (healthy <= 0 || g_amx_disabled.load()) {
    gemm_f16x_fmlal(a, b, c, M, N, K);  // 平台无 AMX → 全量回退
    g_amx_disabled.store(true);
    return;
  }

  pack_panel(a, M, K, t_araw);
  pack_panel(b, N, K, t_braw);
  const uint8_t* X =
      t_araw.data() + ((64 - ((uintptr_t)t_araw.data() & 63)) & 63);
  const uint8_t* Y =
      t_braw.data() + ((64 - ((uintptr_t)t_braw.data() & 63)) & 63);
  gemm_pp_dispatch(X, Y, c, M, N, K);
}

}  // namespace

bool amx_gemm_available() {
  // 惰性触发池创建; 池 worker 探测在各自线程完成
  AmxPool& pool = amx_pool();
  return pool.healthy.load(std::memory_order_acquire) > 0;
}

size_t amx_pool_healthy_threads() {
  AmxPool& pool = amx_pool();
  int h = pool.healthy.load(std::memory_order_acquire);
  return h > 0 ? (size_t)h : 0;
}

void amx_run_batch(std::vector<std::function<void()>> tasks) {
  if (tasks.empty()) return;
  AmxPool& pool = amx_pool();
  if (pool.healthy.load(std::memory_order_acquire) <= 1) {
    for (auto& t : tasks) t();
    return;
  }
  pool.run_batch(std::move(tasks));
}

AmxPanel amx_pack(const uint16_t* w, size_t rows, size_t K) {
  AmxPanel p;
  p.rows = rows;
  p.K = K;
  pack_panel(w, rows, K, p.buf);
  return p;
}

void amx_pack_into(const uint16_t* w, size_t rows, size_t K,
                   std::vector<uint8_t>& out) {
  pack_panel(w, rows, K, out);
}

void gemm_f16_amx_pp(const AmxPanel& pa, const AmxPanel& pb, float* c,
                     size_t M, size_t N) {
  if (pa.rows != M || pb.rows != N || pa.K != pb.K || M == 0 || N == 0) {
    return;  // 契约违规: 静默不写 (调用方仅 available() 后使用, 形状自证)
  }
  gemm_pp_dispatch(pa.data(), pb.data(), c, M, N, pa.K);
}

void gemm_f16_amx(const uint16_t* a, const uint16_t* b, float* c,
                  size_t M, size_t N, size_t K) {
  gemm_f16_amx_impl(a, b, c, M, N, K);
}

// ---------------- E9: 批量多 GEMM 单次派发 ----------------
void amx_batch_run(const AmxBatchNode* nodes, size_t n) {
  if (nodes == nullptr || n == 0) return;
  AmxPool& pool = amx_pool();
  const int healthy = pool.healthy.load(std::memory_order_acquire);
  if (healthy <= 0) {
    for (size_t i = 0; i < n; ++i)
      gemm_f16_amx_pp(*nodes[i].pa, *nodes[i].pb, nodes[i].c, nodes[i].M,
                      nodes[i].N);
    return;
  }

  int max_phase = 0;
  for (size_t i = 0; i < n; ++i) {
    if (nodes[i].phase > max_phase) max_phase = nodes[i].phase;
  }
  const size_t np = size_t(max_phase) + 1;

  // 节点执行态: tile 块预构建; 有 prepare 的节点先跑 head, 完成后才入块
  struct NodeExec {
    std::deque<AmxPool::Task> chunks;
    int heads_left = 0;
    bool launched = false;
  };
  std::vector<NodeExec> ex(n);
  std::vector<size_t> totals(np, 0);
  std::vector<std::deque<AmxPool::Task>> byphase(np);
  std::vector<size_t> nodes_per_phase(np, 0);
  {
    auto valid = [&np](const AmxBatchNode& nd) {
      return nd.phase >= 0 && size_t(nd.phase) < np && nd.pa && nd.pb &&
             nd.c && nd.M > 0 && nd.N > 0 && nd.pa->rows == nd.M &&
             nd.pb->rows == nd.N && nd.pa->K == nd.pb->K;
    };
    for (size_t i = 0; i < n; ++i)
      if (valid(nodes[i])) ++nodes_per_phase[size_t(nodes[i].phase)];
  }

  auto launch_node = [&](size_t i) {  // mu 已持
    auto& e = ex[i];
    if (e.launched || --e.heads_left > 0) return;
    e.launched = true;
    while (!e.chunks.empty()) {
      pool.ready.push_back(std::move(e.chunks.front()));
      e.chunks.pop_front();
    }
    pool.cv.notify_all();
  };

  for (size_t i = 0; i < n; ++i) {
    const AmxBatchNode& nd = nodes[i];
    if (nd.phase < 0 || nd.phase > max_phase || !nd.pa || !nd.pb || !nd.c ||
        nd.M == 0 || nd.N == 0 || nd.pa->rows != nd.M || nd.pb->rows != nd.N ||
        nd.pa->K != nd.pb->K)
      continue;  // 非法节点静默跳过 (与单发口径一致)
    const size_t p = size_t(nd.phase);

    // 节点内 tile 均分切块: 相位内总块数目标 ~2×worker (各相位独占全池)
    const size_t tiles = ((nd.M + 31) / 32) * ((nd.N + 31) / 32);
    const size_t nchunk = std::min(
        tiles, std::max<size_t>(1, size_t(healthy) * 2 /
                                       std::max<size_t>(nodes_per_phase[size_t(nd.phase)], 1)));
    const size_t chunk = (tiles + nchunk - 1) / nchunk;
    float* c = nd.c;
    const size_t M = nd.M, N = nd.N, K = nd.pa->K;
    const size_t nnb = (N + 31) / 32;
    // 注意: pa/pb 的 data() 在任务执行时才取 —— prepare 可能刚填完 pb.buf
    // (assign 后容量稳定, 但 data() 基址须以执行时为准)
    for (size_t t0 = 0; t0 < tiles; t0 += chunk) {
      const size_t t1 = std::min(tiles, t0 + chunk);
      ex[i].chunks.push_back(AmxPool::Task{
          nd.phase,
          [&nd, c, M, N, K, nnb, t0, t1] {
            const uint8_t* pa_data = nd.pa->data();
            const uint8_t* pb_data = nd.pb->data();
            for (size_t t = t0; t < t1; ++t)
              amx_tile(pa_data, pb_data, c, M, N, K, t / nnb, t % nnb);
          },
          nullptr});
      totals[p] += 1;
    }

    if (nd.prepare) {  // head: 池内前置 → 完成后放行本节点 tile 块
      ex[i].heads_left = 1;
      auto* pp = &nd.prepare;
      AmxPool::Task head{
          nd.phase,
          [pp]() { (*pp)(); },
          [launch_node, i] { launch_node(i); }};  // finish_task 持锁路径调用
      byphase[p].push_back(std::move(head));
      totals[p] += 1;
    } else {
      // 无前置: 块直接挂到相位队列, 由相位解锁自然放行
      while (!ex[i].chunks.empty()) {
        byphase[p].push_back(std::move(ex[i].chunks.front()));
        ex[i].chunks.pop_front();
      }
    }
  }
  pool.submit_graph(np, totals.data(), byphase);
}

// ---------------- E10-K: 按-chain DAG 调度消 barrier ----------------
// 与 amx_batch_run 同构, 差异仅在依赖模型:
//   amx_batch_run: 同 phase (depth) 节点并行, 异 phase 串行 (全局 barrier);
//   amx_chain_run: 同 chain_id 内 depth 串行, 异 chain_id 并行 (无 barrier)。
//     —— node (c,d) 仅依赖 node (c,d-1), 不等待其他链。
// 数值序完全一致 (同内核、同 tile 划分、同 prepare 钩子顺序), 故逐位相同。
// profiling 目标: 消除 phase barrier 浪费 (math 96ms + prepare 32ms → ≤100ms)。
void amx_chain_run(const AmxChainLink* nodes, size_t n) {
  if (nodes == nullptr || n == 0) return;
  AmxPool& pool = amx_pool();
  const int healthy = pool.healthy.load(std::memory_order_acquire);
  if (healthy <= 0) {
    // 无 AMX: 退化为串行 amx_batch_run (回退 fmlal)
    for (size_t i = 0; i < n; ++i)
      gemm_f16_amx_pp(*nodes[i].pa, *nodes[i].pb, nodes[i].c, nodes[i].M,
                      nodes[i].N);
    return;
  }

  // ---- 构建 chain 依赖图: node (c,d) 依赖同链 (c,d-1) ----
  // O(n²) 枚举 (n≤18, 忽略不计); 支持多前驱(同链多节点同 depth).
  std::vector<int> deps(n, 0);
  std::vector<std::vector<int>> succs(n);
  for (size_t i = 0; i < n; ++i)
    for (size_t j = 0; j < n; ++j)
      if (i != j && nodes[j].chain_id == nodes[i].chain_id &&
          nodes[j].depth + 1 == nodes[i].depth) {
        ++deps[i];
        succs[j].push_back(static_cast<int>(i));
      }

  // ---- 逐节点构建 tile chunk + prepare hook (沿用 amx_batch_run 模式) ----
  struct NodeExec {
    std::deque<AmxPool::Task> chunks;
    int heads_left = 0;
    bool launched = false;
  };
  std::vector<NodeExec> ex(n);
  std::vector<int> remaining(n, 0);       // 每节点总任务数 (prepare + tiles)
  std::vector<std::deque<AmxPool::Task>> init_tasks(n);  // 解锁时入队

  auto launch_node = [&](size_t i) {  // mu 已持
    auto& e = ex[i];
    if (e.launched || --e.heads_left > 0) return;
    e.launched = true;
    while (!e.chunks.empty()) {
      pool.ready.push_back(std::move(e.chunks.front()));
      e.chunks.pop_front();
    }
    pool.cv.notify_all();
  };

  for (size_t i = 0; i < n; ++i) {
    const AmxChainLink& nd = nodes[i];
    if (!nd.pa || !nd.pb || !nd.c || nd.M == 0 || nd.N == 0 ||
        nd.pa->rows != nd.M || nd.pb->rows != nd.N || nd.pa->K != nd.pb->K)
      continue;  // 非法节点静默跳过
    const size_t M = nd.M, N = nd.N, K = nd.pa->K;
    const size_t tiles = ((M + 31) / 32) * ((N + 31) / 32);
    // 单节点内 tile 均分: 目标 ~2×worker
    const size_t nnb = (N + 31) / 32;
    const size_t nchunk = std::min(
        tiles, std::max<size_t>(1, size_t(healthy) * 2));
    const size_t chunk = tiles == 0 ? 1 : (tiles + nchunk - 1) / nchunk;
    const int idx = static_cast<int>(i);
    float* c = nd.c;
    // 注意: pa/pb data() 在任务执行时才取 —— prepare 可能刚填完 pb.buf
    for (size_t t0 = 0; t0 < tiles; t0 += chunk) {
      const size_t t1 = std::min(tiles, t0 + chunk);
      ex[i].chunks.push_back(AmxPool::Task{
          idx,
          [&nd, c, M, N, K, nnb, t0, t1] {
            const uint8_t* pa_data = nd.pa->data();
            const uint8_t* pb_data = nd.pb->data();
            for (size_t t = t0; t < t1; ++t)
              amx_tile(pa_data, pb_data, c, M, N, K, t / nnb, t % nnb);
          },
          nullptr});
      ++remaining[i];
    }

    if (nd.prepare) {  // head: prepare 完成后放行本节点 tile 块
      ex[i].heads_left = 1;
      auto* pp = const_cast<std::function<void()>*>(&nd.prepare);
      init_tasks[i].push_back(AmxPool::Task{
          idx,
          [pp]() { (*pp)(); },
          [launch_node, i]() { launch_node(i); }});
      ++remaining[i];
    } else {
      // 无前置: tile 块直接作为 init tasks (解锁时入队)
      while (!ex[i].chunks.empty()) {
        init_tasks[i].push_back(std::move(ex[i].chunks.front()));
        ex[i].chunks.pop_front();
      }
    }
  }

  pool.submit_dag(deps, succs, remaining, std::move(init_tasks));
}

// ---------------- E10-K2: per-tile prepare 依赖解耦调度 ----------------
// 在 amx_chain_run 基础上把 prepare 拆到 tile 粒度。AmxTileChainLink 代表
// 一个 32×32 AMX tile; 为避免 73,980 个 std::function 堆分配, 内部将同
// (c,d,mb) 内的连续 nb 合并为 chunk (~healthy 个 tile/chunk), 以 chunk
// 为 DAG 节点调度, 上层 API 仍为 per-tile。
//
// 依赖模型 (chunk 级):
//   * 跨深度 (c,d,chunk) → (c,d-1,chunk): prepare 需同 chunk 上一深度数学
//   * 同行列 (c,d,chunk) → (c,d,chunk-1): 同一深度内 chunk 顺序
//   * 边界 (c,d,chunk) → (c,d-1,chunk-1): im2col 窗口跨 chunk, 需上一
//     深度上一 chunk 数学也完成才能取到边界源数据
//
// profiling 目标: S2 ≤110ms (per-tile prepare 跨链重叠消除 ~25ms im2col
// 全局串行)。数值序与 amx_chain_run 一致 (同内核同 tile 划分, 仅调度更细)。
void amx_tile_chain_run(const AmxTileChainLink* nodes, size_t n) {
  if (nodes == nullptr || n == 0) return;
  AmxPool& pool = amx_pool();
  const int healthy = pool.healthy.load(std::memory_order_acquire);
  if (healthy <= 0) {
    for (size_t i = 0; i < n; ++i) {
      const auto& nd = nodes[i];
      if (!nd.pa || !nd.pb || !nd.c) continue;
      gemm_f16_amx_pp(*nd.pa, *nd.pb, nd.c, nd.M, nd.N);
    }
    return;
  }

  // ---- 阶段 1: 按 (chain_id, depth, tile_mb, tile_nb) 排序并分 chunk ----
  std::vector<size_t> ord(n);
  for (size_t i = 0; i < n; ++i) ord[i] = i;
  std::sort(ord.begin(), ord.end(), [&](size_t a, size_t b) {
    if (nodes[a].chain_id != nodes[b].chain_id)
      return nodes[a].chain_id < nodes[b].chain_id;
    if (nodes[a].depth != nodes[b].depth)
      return nodes[a].depth < nodes[b].depth;
    if (nodes[a].tile_mb != nodes[b].tile_mb)
      return nodes[a].tile_mb < nodes[b].tile_mb;
    return nodes[a].tile_nb < nodes[b].tile_nb;
  });

  // chunk 元信息: 每个 chunk 覆盖 (c,d,mb) 下一段连续 nb
  struct ChunkInfo {
    int chain_id, depth;
    size_t tile_mb;
    size_t nb_begin, nb_end;  // tile_nb 范围 [b,e)
  };
  std::vector<ChunkInfo> chunks;
  for (size_t p = 0; p < n; ) {
    const AmxTileChainLink& hd = nodes[ord[p]];
    size_t q = p;
    while (q < n) {
      const AmxTileChainLink& cur = nodes[ord[q]];
      if (cur.chain_id != hd.chain_id || cur.depth != hd.depth ||
          cur.tile_mb != hd.tile_mb) break;
      ++q;
    }
    const size_t tot = q - p;
    // chunk 大小: 32*healthy tiles/chunk (与 E9 batch_run 同口径)
    // — 足够多 tile 抩平调度开销, 又不会过度打包造成并行不足
    const size_t cs = std::max<size_t>(32, size_t(healthy) * 32);
    const size_t nchunk = std::max<size_t>(1, (tot + cs - 1) / cs);
    const size_t step = (tot + nchunk - 1) / nchunk;
    for (size_t s = 0; s < tot; s += step) {
      const size_t e = std::min(tot, s + step);
      ChunkInfo ci;
      ci.chain_id = hd.chain_id;
      ci.depth = hd.depth;
      ci.tile_mb = hd.tile_mb;
      ci.nb_begin = nodes[ord[p + s]].tile_nb;
      ci.nb_end = nodes[ord[p + e - 1]].tile_nb + 1;
      chunks.push_back(ci);
    }
    p = q;
  }
  const int nch = static_cast<int>(chunks.size());

  // ---- 阶段 2: 构建 chunk-level DAG ----
  struct ChunkKey {
    int chain_id, depth;
    size_t tile_mb;
    bool operator==(const ChunkKey& o) const {
      return chain_id == o.chain_id && depth == o.depth &&
             tile_mb == o.tile_mb;
    }
  };
  struct ChunkKeyHash {
    size_t operator()(const ChunkKey& k) const {
      return std::hash<int>()(k.chain_id) ^ std::hash<int>()(k.depth) ^
             std::hash<size_t>()(k.tile_mb);
    }
  };
  std::unordered_map<ChunkKey, std::vector<int>, ChunkKeyHash> by_key;
  for (int ci = 0; ci < nch; ++ci) {
    const auto& c = chunks[ci];
    by_key[{c.chain_id, c.depth, c.tile_mb}].push_back(ci);
  }
  std::vector<int> deps(nch, 0);
  std::vector<std::vector<int>> succs(nch);
  for (int ci = 0; ci < nch; ++ci) {
    const auto& c = chunks[ci];
    const auto& lst = by_key[{c.chain_id, c.depth, c.tile_mb}];
    int local_idx = 0;
    for (size_t k = 0; k < lst.size(); ++k) {
      if (lst[k] == ci) { local_idx = static_cast<int>(k); break; }
    }
    if (c.depth > 0) {  // 跨深度同 chunk
      const auto& prev_lst = by_key[{c.chain_id, c.depth - 1, c.tile_mb}];
      if (local_idx < int(prev_lst.size())) {
        ++deps[ci];
        succs[prev_lst[local_idx]].push_back(ci);
      }
    }
    if (local_idx > 0) {  // 同行列上一 chunk
      ++deps[ci];
      succs[lst[local_idx - 1]].push_back(ci);
    }
    if (c.depth > 0 && local_idx > 0) {  // 边界依赖
      const auto& prev_lst = by_key[{c.chain_id, c.depth - 1, c.tile_mb}];
      if (local_idx - 1 < int(prev_lst.size())) {
        ++deps[ci];
        succs[prev_lst[local_idx - 1]].push_back(ci);
      }
    }
  }

  // ---- 阶段 3: 逐 chunk 构建 task: prepare + 所有 tile math ----
  // chunk 内 tile 共享 pa/pb/c (调用方约定: 同 c/d/mb 的 tile 共享 panel 与输出)
  struct TileRef {
    size_t mb, nb;
    float* c;
    const AmxPanel* pa;
    const AmxPanel* pb;
    std::function<void(size_t, size_t)>* prep;
    size_t M, N, K;
  };
  std::vector<int> remaining(nch, 0);
  std::vector<std::deque<AmxPool::Task>> init_tasks(nch);
  for (int ci = 0; ci < nch; ++ci) {
    const auto& ch = chunks[ci];
    std::vector<TileRef> tiles;
    size_t row_b = 0, row_e = 0;
    for (size_t p2 = 0; p2 < n; ++p2) {
      const AmxTileChainLink& nd = nodes[ord[p2]];
      if (int(nd.chain_id) != ch.chain_id) continue;
      if (int(nd.depth) != ch.depth) continue;
      if (nd.tile_mb != ch.tile_mb) continue;
      if (nd.tile_nb < ch.nb_begin || nd.tile_nb >= ch.nb_end) continue;
      if (!nd.pa || !nd.pb || !nd.c || nd.M == 0 || nd.N == 0 ||
          nd.pa->rows != nd.M || nd.pb->rows != nd.N || nd.pa->K != nd.pb->K)
        continue;
      TileRef tr;
      tr.mb = nd.tile_mb;
      tr.nb = nd.tile_nb;
      tr.c = nd.c;
      tr.pa = nd.pa;
      tr.pb = nd.pb;
      tr.prep = const_cast<std::function<void(size_t, size_t)>*>(&nd.prepare);
      tr.M = nd.M; tr.N = nd.N; tr.K = nd.pa->K;
      if (tiles.empty()) {
        row_b = ch.nb_begin * 32;
        row_e = std::min(ch.nb_end * 32, nd.N);
      }
      tiles.push_back(tr);
    }
    if (tiles.empty()) continue;
    init_tasks[ci].push_back(AmxPool::Task{
        ci,
        [tiles, row_b, row_e]() {
          if (tiles[0].prep && *tiles[0].prep) (*tiles[0].prep)(row_b, row_e);
          const uint8_t* pa_data = tiles[0].pa->data();
          const uint8_t* pb_data = tiles[0].pb->data();
          float* c0 = tiles[0].c;
          for (const auto& t : tiles) {
            amx_tile(pa_data, pb_data, c0, t.M, t.N, t.K, t.mb, t.nb);
          }
        },
        nullptr});
    ++remaining[ci];
  }

  pool.submit_dag(deps, succs, remaining, std::move(init_tasks));
}

}  // namespace gsv::kern

#endif  // GSV_AMX_GEMM
