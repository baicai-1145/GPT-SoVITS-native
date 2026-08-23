// kern.hpp — 计算内核原语 v1（数值纪律: 权重 fp16 存储、计算/累加/统计量全 fp32；禁 bf16）
//
// 约定:
//   - 所有指针要求合理对齐(NEON 加载用 vld1q 未对齐安全, 无需手工对齐)
//   - eps 类参数用 double 传入避免调用方舍入歧义
#pragma once

#include <cstddef>
#include <stdint.h>

namespace gsv::kern {

// ---- GEMV: y[out] = W[out,in] · x[in]; W 以 fp16 位型存储(row-major), 逐元素
// 精确升位到 fp32 后 FMA, 累加器与输出 fp32。(见 gemv.cpp 的 NEON 实现)
void gemv_f16w_f32acc(const uint16_t* w, const float* x, float* y, size_t out, size_t in);

// 参考实现(标量循环, 供对拍/兜底): 数值语义与 NEON 版一致
void gemv_f16w_f32acc_ref(const uint16_t* w, const float* x, float* y, size_t out, size_t in);

// ---- RMSNorm: y = x * g / sqrt(mean(x²) + eps)；统计量 fp32
void rmsnorm(const float* x, const float* g, float* y, size_t n, double eps);

// ---- LayerNorm(带仿射): y = (x-μ)/sqrt(σ²+eps) * g + b；μ/σ² 为有偏方差(fp32)
//      与 torch F.layer_norm 同构 —— AR/BERT 层栈使用
void layernorm(const float* x, const float* g, const float* b, float* y, size_t n, double eps);

// ---- Softmax(行内稳定版): max/exp/sum 全部 fp32
void softmax(const float* x, float* y, size_t n);
void softmax_rows(const float* x, float* y, size_t rows, size_t n);

// ---- 激活: silu(x)=x·sigmoid(x), relu(x)=max(x,0)
void silu(const float* x, float* y, size_t n);
void relu(const float* x, float* y, size_t n);

// ---- RoPE 旋转位置编码（两种配对约定）
//   GptJ: 相邻配对  (x[2i], x[2i+1]) 旋转角 θ_pos·freq_i
//   NeoX : 半分配对  (x[i], x[i+D/2]) 旋转角 θ_pos·freq_i, i < D/2
//   freq_i = base^(-2i/D)。向量按 [heads, head_dim] 解释, 对每个 head 独立旋转。
enum class RopeStyle : uint8_t { GptJ, NeoX };

// prefill 批量: io 为 [seq, heads*head_dim] 行连续, 位置从 pos_base 起
void rope_prefill(float* io, size_t seq, size_t heads, size_t head_dim,
                  size_t pos_base, RopeStyle style, double base = 10000.0);
// decode 单 token: io 为 [heads*head_dim], 位置 pos
void rope_decode(float* io, size_t heads, size_t head_dim,
                 size_t pos, RopeStyle style, double base = 10000.0);

}  // namespace gsv::kern
