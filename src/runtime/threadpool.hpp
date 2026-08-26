// threadpool.hpp — QoS 分簇线程池（吃满 CPU 的基础件）
//
// 两个持久池:
//   Qos::UserInitiated → 绑 P 核数量(hw.perflevel0.logicalcpu) —— 实时链路
//   Qos::Utility       → 绑 E 核数量(hw.perflevel1.logicalcpu) —— 后台编码
// worker 启动即 pthread_set_qos_class_self_np 固定簇; 池惰性创建, 进程退出回收。
#pragma once

#include <cstddef>
#include <functional>

namespace gsv::rt {

enum class Qos : uint8_t { UserInitiated, Utility };

size_t p_core_count();   // hw.perflevel0.logicalcpu, 缺失回退 hw.ncpu
size_t e_core_count();   // hw.perflevel1.logicalcpu, 缺失回退 0(单池方案)

// 并行 for: [0,n) 切成 ≥grain 的块并行执行 fn(begin,end)。
// 工作量 ≤ grain 时在调用线程内联执行。阻塞至全部完成。
void parallel_for(size_t n, size_t grain,
                  const std::function<void(size_t, size_t)>& fn, Qos qos);

// E11-4: 全核联合派发 —— P 池拿 p_weight/(p_weight+e_weight) 的总工作量,
//        E 池拿余下; 默认权重 P:E = 2:1(授 P 核频率更高，可多领任务)。
//        E 核拿不到工作则 fallback P-only。禁止用于 AMX/cblas(它们绑 P 核)。
//        E11-4 边界: 纯 NEON FMLAL GEMV/GEMM 路径才允许(项目当前唯一全核可走点)。
void parallel_for_full(size_t n, size_t grain,
                       const std::function<void(size_t, size_t)>& fn);

// 查询当前 GSV_GEMV_E_DISABLE 环境开关: 1=退到 P-only, 0=全核(默认 0)。
// 本探针供 decode GEMV 路径探判选用; 全核路径与 P-only 数值位级一致 (纯调度)。
bool gemv_use_e_cores();

}  // namespace gsv::rt
