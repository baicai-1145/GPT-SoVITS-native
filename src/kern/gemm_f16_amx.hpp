// gemm_f16_amx.hpp — E5: AMX MATFP fp16×fp16→fp32 GEMM 后端 (实验)
//
// 接口与 gemm_f16x_fmlal 同构:
//   C[M,N] = Σ_k A[m,k]·B[n,k]   (A [M,K] f16 行主, B [N,K] f16 行主, C f32 行主)
// 覆盖 SoVITS conv im2col (y[Co,S]=W[Co,K]·col[S,K]ᵀ) 与 encoder 全连接形。
//
// 数值语义: 与 gemm_f16x_fmlal 完全一致 —— fp16 输入精确落 f32 (f16×f16 积
// 在 fp32 内无舍入), 累加链纯 fp32。对拍参考即 FMLAL 实现, golden 门禁同口径
// (cos≥0.9999, rel≤1e-3)。
//
// 调度: AMX 专用线程池 (不与 Accelerate/BLAS 共线程, 见 amx.h 互斥注记),
// tile (mb,nb) 粒度分发; worker 首任务前完成 SIGILL 探测 + AMX_SET 长驻,
// 探测失败全局降级回退 fmlal。
//
// 预打包 API: 权重侧 (conv 的 W, 全连接的 W) 为静态数据, pack 一次后反复
// 使用 —— gemm 用 packed panel 直跑内核, 省每调用打包成本。
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace gsv::kern {

// 编译期总开关: 定义 GSV_AMX_GEMM 才编入 (默认关, CMake -DGSV_AMX_GEMM=ON)。
// 运行时开关: amx_gemm_available() (AMX_SET+MATFP 探测 + 专用池 worker 使能)。
#if defined(GSV_AMX_GEMM)

// 探测结论 (首次调用惰性启动 AMX 专用池并探测)。
bool amx_gemm_available();

// 预打包 panel: [rows,K] f16 行主 → AMX X/Y 直读布局 (对齐, 尾行补零)。
// 非多线程安全: 不同线程请各自持 panel (或外部同步)。
struct AmxPanel {
  size_t rows = 0, K = 0;             // 逻辑形状
  std::vector<uint8_t> buf;           // 对齐基址在 buf 内偏移
  const uint8_t* data() const {       // 64B 对齐 panel 首地址
    const uintptr_t p = (uintptr_t)buf.data();
    return buf.data() + ((64 - (p & 63)) & 63);
  }
};

AmxPanel amx_pack(const uint16_t* w, size_t rows, size_t K);

// E9: 就地打包 (复用调用方缓冲容量; 与 amx_batch_run 的节点 prepare 钩子
// 配套 —— 池内 worker 调用本函数把激活打包成 Y panel, 与其他节点数学重叠)
void amx_pack_into(const uint16_t* w, size_t rows, size_t K,
                   std::vector<uint8_t>& out);

// 预打包版 GEMM: C[M,N] = pa·pbᵀ (pa: M×K, pb: N×K)。
// 需 pa.rows==M && pb.rows==N && pa.K==pb.K, 否则回退即时打包路径。
// AMX 不可用时回退 gemm_f16x_fmlal (自动从 panel 解不出原矩阵, 故要求
// 调用方仅在 available()==true 时走此路径, 内部不二次校验)。
void gemm_f16_amx_pp(const AmxPanel& pa, const AmxPanel& pb, float* c,
                     size_t M, size_t N);

// ---- E9: 批量多 GEMM 单次派发 ----
// 一批相互独立的预打包 GEMM, 按 phase 分桶调度: 同 phase 节点并行,
// 高 phase 节点等低相位全部完成后开始; 整批仅一次池往返 (替代逐节点
// 提交的 打包→等→打包→等 串行互等)。典型用法: SoVITS dec 每个 stage
// 的 3 个独立 resblock 链按链深分 phase, 18 次派发→6 次。
struct AmxBatchNode {
  int phase = 0;              // ≥0; 同相位并行, 高相位等低相位全部完成
  const AmxPanel* pa = nullptr;  // 权重侧 [M,K]
  const AmxPanel* pb = nullptr;  // 激活侧 [N,K] (sovits im2col 直产)
  float* c = nullptr;            // 输出 [M,N]; 各节点输出不得重叠
  size_t M = 0, N = 0;
  // 可选前置工作 (如 im2col 直产 panel): 在池内执行, 完成后才放行本节点
  // 的 tile 计算 —— 与同相位其他节点的数学流水重叠。空 = 无前置。
  std::function<void()> prepare;
};

// 阻塞至全部节点完成。不可用时逐节点回退 gemm_f16_amx_pp (含 fmlal 回退),
// 语义不变。形状不合法的节点静默跳过 (与 pp 单发一致)。缓冲生命周期:
// 调用返回前 pa/pb/c 必须保持有效。
void amx_batch_run(const AmxBatchNode* nodes, size_t n);

// ---- E10-K: 按链调度 DAG 派发 (消除 phase barrier) ----
// 取代 amx_batch_run 的 phase-屏障模型: 节点间依赖仅限同一 chain_id
// 内 depth-1 → depth (单条链内串行, 跨链并行), 节点就绪(前驱完成)即刻
// 入队执行, 不等待同深度的其他链 —— 消除 amx_batch_run 的全局 barrier
// 浪费 ( profiling: math 96ms / prepare 32ms / barrier 94ms )。
//
// AmxChainLink 与 AmxBatchNode 同构 (pa/pb/c/M/N/prepare), 以 chain_id+depth
// 取代 phase。语义: 同 chain_id 者按 depth 顺序串行 (depth d 依赖 d-1 完毕),
// 不同 chain_id 者无依赖。数值序与 amx_batch_run 完全一致 (同内核同 tile 划分)。
// 不可用时回退 amx_batch_run (含 fmlal 回退), 语义不变。
struct AmxChainLink {
  int chain_id = 0;      // 链ID: 同链内串行, 0≤depth<max_depth
  int depth = 0;         // 深度: 依赖同链 depth-1 节点
  const AmxPanel* pa = nullptr;
  const AmxPanel* pb = nullptr;
  float* c = nullptr;
  size_t M = 0, N = 0;
  std::function<void()> prepare;
};

void amx_chain_run(const AmxChainLink* nodes, size_t n);

// ---- E10-K2: per-tile (32×32) prepare 依赖消除 prepare↔math 全局互锁 ----
// 在 amx_chain_run 的 per-node prepare 依赖之上, 把 prepare 步拆到 tile
// 粒度: (c,d,tile) 的 prepare 只需等 (c,d-1,tile) 的数学完成, 即可在
// 池内与 (c,d-1,tile+1) 的数学重叠执行, 把 ~25ms 的 im2col 从全局串行
// 解放为 tile-by-tile 流水。AmxTileChainLink 与 AmxChainLink 同构, 差异
// 仅在 prepare 签名: prepare 接收 tile 行区间 [row_b, row_e), 内部只产
// 激活 panel 的对应 tile 段 (dec 端 im2col 按 32 步切片)。与 amx_chain_run
// 的数值序完全一致: 同内核同 tile 划分, 仅调度拓扑更细。
struct AmxTileChainLink {
  int chain_id = 0;     // 链ID: 同链内串行, 0≤depth<max_depth
  int depth = 0;        // 深度: 依赖同链 depth-1 节点
  size_t tile_mb = 0;   // tile 行号 (M 维 32 步)
  size_t tile_nb = 0;   // tile 列号 (N 维 32 步)
  const AmxPanel* pa = nullptr;
  const AmxPanel* pb = nullptr;
  float* c = nullptr;
  size_t M = 0, N = 0;
  // prepare 接收 [row_b, row_e) 时间步区间, 内部只产该区间对应 panel tile。
  // 调用方保证 row_b%32==0 且 row_e<=N, row_e-row_b<=32 (单个 32 步 tile)。
  std::function<void(size_t row_b, size_t row_e)> prepare;
};

void amx_tile_chain_run(const AmxTileChainLink* nodes, size_t n);

// C[M,N] = A·Bᵀ; 不可用时内部回退 gemm_f16x_fmlal。 (即时打包, 通用路径)
void gemm_f16_amx(const uint16_t* a, const uint16_t* b, float* c,
                  size_t M, size_t N, size_t K);

#else

struct AmxPanel {};  // 空壳: 非 AMX 构建下可引用类型, 不可调用函数

inline bool amx_gemm_available() { return false; }

#endif  // GSV_AMX_GEMM

}  // namespace gsv::kern
