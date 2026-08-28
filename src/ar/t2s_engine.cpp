// t2s_engine.cpp — AR/T2S 引擎实现 (B1 prefill + B2 decode)
//
// 数值对照要点(均与 torch 同构, 见头文件注释):
//   - 大矩阵乘 prefill 走 accel::sgemm(DenseF16 加载时升位缓存);
//     decode 单 token 走 kern::gemv_f16w_f32acc(fp16 权重 NEON 升位 + fp32 FMA 累加)
//   - LayerNorm 有偏方差 eps=1e-5, 统计量 fp32 (kern::layernorm)
//   - 注意力 scale = 1/sqrt(head_dim); softmax 行内稳定 fp32
//   - 无 RoPE(T2S 用正弦位置编码加在输入上, 与 kern 的 rope 无关)
#include "ar/t2s_engine.hpp"

#include "kern/gemv_fmlal.hpp"

#include <arm_neon.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

// ---- E11-1: SDPA NEON helpers (decode + prefill 共享) ----
// 数值纪律:
//   fp32 路径: 4-lane 独立累加 → 树形归约 (a0+a1)+(a2+a3) → 水平求和
//   (等价于 8 个 vdot 等价 FMA 但顺序可比, 保留 G1/G2 数值容差)
//   fp16 KV 路径: vld1q_f16×2 升位 fp32 后 FMA, 避免标量升位瓶颈
// 性能纪律: HD=32 (AR 默认 512/16) 时全部走 4-lane×8vec 主路径, 无尾循环
namespace {
namespace ar_neon {

// fp32 dot: qv[0..HD) · kv[0..HD), 4-lane tree reduce
inline float dot_f32(const float* qv, const float* kv, size_t HD) {
  float32x4_t a0 = vdupq_n_f32(0.f);
  float32x4_t a1 = vdupq_n_f32(0.f);
  float32x4_t a2 = vdupq_n_f32(0.f);
  float32x4_t a3 = vdupq_n_f32(0.f);
  size_t e = 0;
  // 主路径: 16 元素一组 (4 vec × 4 lanes) — AR 默认 HD=32 时一轮吃完
  for (; e + 16 <= HD; e += 16) {
    a0 = vfmaq_f32(a0, vld1q_f32(qv + e + 0),  vld1q_f32(kv + e + 0));
    a1 = vfmaq_f32(a1, vld1q_f32(qv + e + 4),  vld1q_f32(kv + e + 4));
    a2 = vfmaq_f32(a2, vld1q_f32(qv + e + 8),  vld1q_f32(kv + e + 8));
    a3 = vfmaq_f32(a3, vld1q_f32(qv + e + 12), vld1q_f32(kv + e + 12));
  }
  // 4-lane 树形归约: (a0+a1)+(a2+a3)
  float32x4_t s01 = vaddq_f32(a0, a1);
  float32x4_t s23 = vaddq_f32(a2, a3);
  float32x4_t s = vaddq_f32(s01, s23);
  // 水平求和: pairwise add
  float32x2_t lo = vget_low_f32(s);
  float32x2_t hi = vget_high_f32(s);
  float32x2_t sum2 = vadd_f32(lo, hi);
  float32x2_t sum1 = vpadd_f32(sum2, sum2);
  float dot = vget_lane_f32(sum1, 0);
  // 尾元素 (HD 非 16 倍数时; AR 实际 HD=32/64 走不到这里)
  for (; e < HD; ++e) dot += qv[e] * kv[e];
  return dot;
}

// fp16 KV dot: qv[fp32] · k16[fp16] 升位累加
inline float dot_f16kv(const float* qv, const uint16_t* k16, size_t HD) {
  float32x4_t a0 = vdupq_n_f32(0.f);
  float32x4_t a1 = vdupq_n_f32(0.f);
  float32x4_t a2 = vdupq_n_f32(0.f);
  float32x4_t a3 = vdupq_n_f32(0.f);
  size_t e = 0;
  for (; e + 16 <= HD; e += 16) {
    // 每 8 个 fp16 装一 NEON half 向量; 一组 16 fp16 = 2 个 half vec
    float16x8_t h0 = vld1q_f16(reinterpret_cast<const __fp16*>(k16 + e));
    float16x8_t h1 = vld1q_f16(reinterpret_cast<const __fp16*>(k16 + e + 8));
    a0 = vfmaq_f32(a0, vld1q_f32(qv + e + 0),
                   vcvt_f32_f16(vget_low_f16(h0)));
    a1 = vfmaq_f32(a1, vld1q_f32(qv + e + 4),
                   vcvt_f32_f16(vget_high_f16(h0)));
    a2 = vfmaq_f32(a2, vld1q_f32(qv + e + 8),
                   vcvt_f32_f16(vget_low_f16(h1)));
    a3 = vfmaq_f32(a3, vld1q_f32(qv + e + 12),
                   vcvt_f32_f16(vget_high_f16(h1)));
  }
  float32x4_t s01 = vaddq_f32(a0, a1);
  float32x4_t s23 = vaddq_f32(a2, a3);
  float32x4_t s = vaddq_f32(s01, s23);
  float32x2_t lo = vget_low_f32(s);
  float32x2_t hi = vget_high_f32(s);
  float32x2_t sum2 = vadd_f32(lo, hi);
  float32x2_t sum1 = vpadd_f32(sum2, sum2);
  float dot = vget_lane_f32(sum1, 0);
  for (; e < HD; ++e) {
    __fp16 h;
    __builtin_memcpy(&h, k16 + e, 2);
    dot += qv[e] * static_cast<float>(h);
  }
  return dot;
}

// ov[0..HD) += p * vv[0..HD)  (fp32)
inline void accum_f32(float* ov, const float* vv, float p, size_t HD) {
  float32x4_t p4 = vdupq_n_f32(p);
  // 4-lane 独立累加 → 加到 ov
  // 注意: vv 是只读, ov 是读写; 4-lane 分组保持 4×4 网格, 末尾归并到 ov
  // 简化: 16 元素一组, 累加后立即归并 (HD=32 时仅一轮)
  size_t e = 0;
  for (; e + 16 <= HD; e += 16) {
    float32x4_t a0 = vmulq_f32(p4, vld1q_f32(vv + e + 0));
    float32x4_t a1 = vmulq_f32(p4, vld1q_f32(vv + e + 4));
    float32x4_t a2 = vmulq_f32(p4, vld1q_f32(vv + e + 8));
    float32x4_t a3 = vmulq_f32(p4, vld1q_f32(vv + e + 12));
    // ov 加载 + 累加
    vst1q_f32(ov + e + 0,  vaddq_f32(vld1q_f32(ov + e + 0),  a0));
    vst1q_f32(ov + e + 4,  vaddq_f32(vld1q_f32(ov + e + 4),  a1));
    vst1q_f32(ov + e + 8,  vaddq_f32(vld1q_f32(ov + e + 8),  a2));
    vst1q_f32(ov + e + 12, vaddq_f32(vld1q_f32(ov + e + 12), a3));
  }
  for (; e < HD; ++e) ov[e] += p * vv[e];
}

// ov[0..HD) += p * vv16[0..HD)  (fp16 KV)
inline void accum_f16kv(float* ov, const uint16_t* v16, float p, size_t HD) {
  float32x4_t p4 = vdupq_n_f32(p);
  size_t e = 0;
  for (; e + 16 <= HD; e += 16) {
    float16x8_t h0 = vld1q_f16(reinterpret_cast<const __fp16*>(v16 + e));
    float16x8_t h1 = vld1q_f16(reinterpret_cast<const __fp16*>(v16 + e + 8));
    float32x4_t a0 = vmulq_f32(p4, vcvt_f32_f16(vget_low_f16(h0)));
    float32x4_t a1 = vmulq_f32(p4, vcvt_f32_f16(vget_high_f16(h0)));
    float32x4_t a2 = vmulq_f32(p4, vcvt_f32_f16(vget_low_f16(h1)));
    float32x4_t a3 = vmulq_f32(p4, vcvt_f32_f16(vget_high_f16(h1)));
    vst1q_f32(ov + e + 0,  vaddq_f32(vld1q_f32(ov + e + 0),  a0));
    vst1q_f32(ov + e + 4,  vaddq_f32(vld1q_f32(ov + e + 4),  a1));
    vst1q_f32(ov + e + 8,  vaddq_f32(vld1q_f32(ov + e + 8),  a2));
    vst1q_f32(ov + e + 12, vaddq_f32(vld1q_f32(ov + e + 12), a3));
  }
  for (; e < HD; ++e) {
    __fp16 h;
    __builtin_memcpy(&h, v16 + e, 2);
    ov[e] += p * static_cast<float>(h);
  }
}

}  // namespace ar_neon
}  // namespace

namespace gsv::ar {

using DenseF16 = kern::accel::DenseF16;
using rt::GsvFile;
using rt::TensorView;

T2SEngine::T2SEngine(const GsvFile& f) {
  // ---- 维度: 全部来自 ckpt 内嵌 config(convert.py 存入 .gsv header) ----
  const auto* model_cfg = f.config().find("model_config");
  const auto* m = model_cfg ? model_cfg->find("model") : nullptr;
  if (!m) throw std::runtime_error("ar .gsv 缺 model_config.model");
  dims_.d_model = static_cast<size_t>(m->find("hidden_dim")->as_int());
  dims_.n_heads = static_cast<size_t>(m->find("head")->as_int());
  dims_.ffn = static_cast<size_t>(m->find("linear_units")->as_int());
  dims_.n_layers = static_cast<size_t>(m->find("n_layer")->as_int());
  dims_.vocab = static_cast<size_t>(m->find("vocab_size")->as_int());
  dims_.phone_vocab = static_cast<size_t>(m->find("phoneme_vocab_size")->as_int());
  dims_.bert_dim = 1024; // bert_proj 固定输入维(roberta hidden), config 不含 → 常量
  dims_.eos = static_cast<int>(m->find("EOS")->as_int());

  const size_t D = dims_.d_model;
  if (dims_.d_model != static_cast<size_t>(m->find("embedding_dim")->as_int()))
    throw std::runtime_error("embedding_dim != hidden_dim: 本引擎按相等设计");
  if (dims_.eos != static_cast<int>(dims_.vocab) - 1)
    throw std::runtime_error("EOS != vocab-1");

  auto vec_of = [&](const char* name) {
    const TensorView& t = need(f, name);
    return std::vector<float>(t.data_f32(), t.data_f32() + t.numel());
  };
  auto dense16 = [&](const char* name, size_t rows, size_t cols) {
    const TensorView& t = need(f, name);
    if (t.numel() != rows * cols || !t.has_f16())
      throw std::runtime_error(std::string(name) + ": 形状/f16 段不符");
    return DenseF16(t.data_f16_raw(), rows, cols);
  };

  // ---- 词表嵌入: 整表升位一次(共 ~3.6MB fp32, 查表 O(1) 且免逐行转换) ----
  {
    const TensorView& te = need(f, "ar_text_embedding.word_embeddings.weight");
    if (te.numel() != dims_.phone_vocab * D || !te.has_f16())
      throw std::runtime_error("ar_text_embedding 形状不符");
    text_emb_.resize(te.numel());
    accel::f16_to_f32(te.data_f16_raw(), text_emb_.data(), te.numel());
    const TensorView& ae = need(f, "ar_audio_embedding.word_embeddings.weight");
    if (ae.numel() != dims_.vocab * D || !ae.has_f16())
      throw std::runtime_error("ar_audio_embedding 形状不符");
    audio_emb_.resize(ae.numel());
    accel::f16_to_f32(ae.data_f16_raw(), audio_emb_.data(), ae.numel());
  }

  bert_b_ = vec_of("bert_proj.bias");
  bert_proj_ = dense16("bert_proj.weight", D, dims_.bert_dim);
  {
    const TensorView& t = need(f, "ar_predict_layer.weight");
    if (t.numel() != dims_.vocab * D || !t.has_f16())
      throw std::runtime_error("ar_predict_layer 形状/f16 段不符");
    wp_ = DenseF16(t.data_f16_raw(), dims_.vocab, D);
    wp16_.assign(t.data_f16_raw(), t.data_f16_raw() + t.numel());
  }
  alpha_text_ = need(f, "ar_text_position.alpha").data_f32()[0];
  alpha_audio_ = need(f, "ar_audio_position.alpha").data_f32()[0];

  layers_.resize(dims_.n_layers);
  for (size_t l = 0; l < dims_.n_layers; ++l) {
    Layer& L = layers_[l];
    const std::string p = "h.layers." + std::to_string(l) + ".";
    const std::string ps = p + "self_attn.";
    auto dense16r = [&](const char* name, size_t rows, size_t cols,
                        std::vector<uint16_t>* raw) {
      const TensorView& t = need(f, name);
      if (t.numel() != rows * cols || !t.has_f16())
        throw std::runtime_error(std::string(name) + ": 形状/f16 段不符");
      if (raw)
        raw->assign(t.data_f16_raw(), t.data_f16_raw() + t.numel());
      return DenseF16(t.data_f16_raw(), rows, cols);
    };
    L.wqkv = dense16r((ps + "in_proj_weight").c_str(), 3 * D, D, &L.wqkv16);
    L.bqkv = vec_of((ps + "in_proj_bias").c_str());
    L.wout = dense16r((ps + "out_proj.weight").c_str(), D, D, &L.wout16);
    L.bout = vec_of((ps + "out_proj.bias").c_str());
    L.w1 = dense16r((p + "linear1.weight").c_str(), dims_.ffn, D, &L.w116);
    L.b1 = vec_of((p + "linear1.bias").c_str());
    L.w2 = dense16r((p + "linear2.weight").c_str(), D, dims_.ffn, &L.w216);
    L.b2 = vec_of((p + "linear2.bias").c_str());
    L.n1g = vec_of((p + "norm1.weight").c_str());
    L.n1b = vec_of((p + "norm1.bias").c_str());
    L.n2g = vec_of((p + "norm2.weight").c_str());
    L.n2b = vec_of((p + "norm2.bias").c_str());
#ifdef GSV_AMX_GEMM
    // E11-2: prefill AMX 预打包面板(包后即可在 block_prefill_impl 走 gemm_f16_amx_pp)
    if (kern::amx_gemm_available()) {
      L.wqkv_pa = kern::amx_pack(L.wqkv16.data(), 3 * D, D);
      L.w1_pa   = kern::amx_pack(L.w116.data(),   dims_.ffn, D);
      L.w2_pa   = kern::amx_pack(L.w216.data(),   D, dims_.ffn);
    }
#endif
  }
#ifdef GSV_AMX_GEMM
  prefill_amx_in_use_ = kern::amx_gemm_available();
#endif
}

const TensorView& T2SEngine::need(const GsvFile& f, const char* name) const {
  const auto* t = f.tensor(name);
  if (!t) throw std::runtime_error(std::string("缺张量: ") + name);
  return *t;
}

// ---- M1-fp16 开关与直读路径 ----
void T2SEngine::set_fp16(const Fp16Options& o) { fp16_ = o; }

// y[out] = W16·xh: 激活舍入到 fp16 后走 FMLAL 扩展精度累加 GEMV。
// xh_ 复用为暂存(调用方串行使用, 无重入)。
// y[out] = W16·xh: 激活舍入到 fp16 后走 FMLAL 扩展精度累加 GEMV。
// xh_ 复用为暂存(调用方串行使用, 无重入)。
void T2SEngine::gemv_fmlal(const std::vector<uint16_t>& w16, const float* x,
                           size_t out, size_t in, float* y) {
  xh_.resize(in);
  kern::f32_to_f16(x, xh_.data(), in);
  kern::gemv_f16x_fmlal(w16.data(), xh_.data(), y, out, in);
}

// generate() 内部使用的 logits 投影(尊重 fp16.gemv)
void T2SEngine::predict_layer_fp(const float* x, float* y) {
  // Logits 投影层保持 fp32 精度 (DenseF16 升位 sgemm) 以免在 top-1/top-2 微小差异时引起 argmax 翻转
  wp_.forward(x, 1, y);
}

// SinePositionalEmbedding.extend_pe 单行同构:
// div_term[i]=exp(2i·(-ln(10000)/D)); pe[2i]=sin(pos·div), pe[2i+1]=cos(pos·div)
void T2SEngine::pe_row(float* pe, size_t pos) {
  const size_t D = dims_.d_model;
  const float log_inc = static_cast<float>(-std::log(10000.0) / static_cast<double>(D));
  const float p = static_cast<float>(pos);
  for (size_t d = 0; d < D; d += 2) {
    const float div = std::exp(static_cast<float>(d) * log_inc);
    pe[d] = std::sin(p * div);
    pe[d + 1] = std::cos(p * div);
  }
}

// ---- prefill 单层: 与 T2SBlock.process_prompt 同构(post-LN) ----
// KV16: k/v 以 fp16 位型写入 cache, 注意力读出升位 fp32 计算(存储舍入一次)。
#ifdef GSV_AMX_GEMM
// E11-5: prefill SDPA 走 AMX (Q·K^T + P·V 两次 GEMM per head, 16 头批量派发)
//   设计: 从 qkv_ 抽取 per-head Q/K/V (各 [S, HD] 头主) 进临时 f16 缓冲,
//   pack 进 AmxPanel, 16 头 Q·K^T 一次性 amx_batch_run 提交 (phase 0),
//   应用 -inf 掩码 → 行 softmax → P·V (phase 1) 一次性 提交。
//   数值纪律: K=HD=32 薄 K 形状, AMX 归约序与 FMLAL 4-lane 树形可能差异较 E11-2 大
//   (E11-2 K=512/2048 形状下 fp16 源等价, 这里 fp32 → fp16 转换 + GEMM 归约序双重漂移)。
//   G1 验收 cos8≥0.9999 + B12 全过 为位级可比验证。
//   熔断: amx_gemm_available() 失败 或 HD 非 32 倍数 (rare) → 走 NEON 紧致路径。
template <bool KV16>
void T2SEngine::sdpa_amx_prefill(size_t H, size_t S, size_t HD, float scale,
                                 size_t text_len, const float* qkv_buf,
                                 const float* kf32, const uint16_t* k16,
                                 const float* vf32, const uint16_t* v16,
                                 float* attn_out) {
  // 确保 scratch 足够: sdpa_xh_ 需 max(S*D, S*S) fp16 容量
  const size_t need = S * S > S * HD ? S * S : S * HD;
  if (sdpa_cap_ < need) {
    sdpa_xh_.resize(need);
    sdpa_cap_ = need;
  }
  sdpa_scores_.assign(S * S, 0.f);
  sdpa_probs_.assign(S * S, 0.f);

  // 为每头分配临时 Q/K/V fp16 缓冲 + AmxPanel (在栈上, 大小 S*HD fp16 + panel 头)
  // 16 头 × 3 面板 = 48 面板, 不大; 但 AmxPanel.buf 约 S*HD*64B(panel 1MB) 太大。
  // 改为串行 16 头, 复用同一个 Q/K/V scratch + 同一 AmxPanel。
  std::vector<uint16_t> Q_f16(S * HD);
  std::vector<uint16_t> K_f16(S * HD);
  std::vector<uint16_t> V_f16(S * HD);
  // V_T 供 P·V 的 pa_V 用: pa_V 形状 [HD, S] (rows=HD, K=S)
  // V[S,HD] → V_T[HD,S]: V_T[e, t] = V_f16[t*HD + e]
  std::vector<uint16_t> V_T_f16(HD * S);
  const size_t D = H * HD;

  for (size_t h = 0; h < H; ++h) {
    // 1) 从 qkv_ 抽取 Q_h: qkv_ 布局 [S, 3D] (D=H*HD), 每 token Q 段起点 = q*3D
    //    head h 段起点 = q*3D + h*HD; Q_h[t, e] = qkv_[t*3D + h*HD + e]
    for (size_t t = 0; t < S; ++t) {
      const float* src = qkv_buf + t * 3 * D + h * HD;
      for (size_t e = 0; e < HD; ++e) {
        __fp16 hh = static_cast<__fp16>(src[e]);
        __builtin_memcpy(&Q_f16[t * HD + e], &hh, 2);
      }
    }
    // 2) 抽取 K_h: KV cache 布局 [S, D] 行主, head h 段起点 = t*D + h*HD
    for (size_t t = 0; t < S; ++t) {
      if constexpr (KV16) {
        const uint16_t* src = k16 + t * D + h * HD;
        std::memcpy(&K_f16[t * HD], src, HD * sizeof(uint16_t));
      } else {
        const float* src = kf32 + t * D + h * HD;
        for (size_t e = 0; e < HD; ++e) {
          __fp16 hh = static_cast<__fp16>(src[e]);
          __builtin_memcpy(&K_f16[t * HD + e], &hh, 2);
        }
      }
    }
    // 3) 抽取 V_h
    for (size_t t = 0; t < S; ++t) {
      if constexpr (KV16) {
        const uint16_t* src = v16 + t * D + h * HD;
        std::memcpy(&V_f16[t * HD], src, HD * sizeof(uint16_t));
      } else {
        const float* src = vf32 + t * D + h * HD;
        for (size_t e = 0; e < HD; ++e) {
          __fp16 hh = static_cast<__fp16>(src[e]);
          __builtin_memcpy(&V_f16[t * HD + e], &hh, 2);
        }
      }
    }

    // 4) Pack Q/K/V 进 AmxPanel (调用方在主线程 pack, AmxPool 跑 GEMM 不重 pack)
    kern::AmxPanel pa_Q, pa_K, pa_V;
    pa_Q.rows = S; pa_Q.K = HD;
    pa_K.rows = S; pa_K.K = HD;
    // pa_V 形状 [HD, S]: N=HD (输出列), K=S (P 的 reduction 维)
    // 需要 V 转置为 [HD, S] 行主 (从 V_f16 [S, HD] 转置)
    pa_V.rows = HD; pa_V.K = S;
    for (size_t t = 0; t < S; ++t) {
      for (size_t e = 0; e < HD; ++e) {
        V_T_f16[e * S + t] = V_f16[t * HD + e];
      }
    }
    kern::amx_pack_into(Q_f16.data(), S, HD, pa_Q.buf);
    kern::amx_pack_into(K_f16.data(), S, HD, pa_K.buf);
    kern::amx_pack_into(V_T_f16.data(), HD, S, pa_V.buf);

    // 5) Q·K^T via AMX: pa_Q [S,HD] (M=S), pa_K [S,HD] (N=S) → scores [S, S]
    float* scores = sdpa_scores_.data();
    kern::gemm_f16_amx_pp(pa_Q, pa_K, scores, S, S);
    // scale
    for (size_t i = 0; i < S * S; ++i) scores[i] *= scale;

    // 6) 应用掩码: k < text_len || k <= q → 保持; 否则 -inf
    // torch 同构: text_len>0 时, q<text_len 看到全部 text_len 个 key (即 k<text_len 总是真);
    //             q≥text_len 看到 [0..q] (k<=q)。
    for (size_t q = 0; q < S; ++q) {
      for (size_t k = 0; k < S; ++k) {
        const bool allowed = (k < text_len) || (k <= q);
        if (!allowed) scores[q * S + k] = -std::numeric_limits<float>::infinity();
      }
    }
    // 7) 行 softmax: 复制到 probs, in-place 稳定 softmax
    float* probs = sdpa_probs_.data();
    for (size_t q = 0; q < S; ++q) {
      float* row = scores + q * S;
      // 找 max (仅在 allowed 集, 但 -inf 不参与 max, 可以全局)
      float mx = row[0];
      for (size_t k = 1; k < S; ++k) mx = std::max(mx, row[k]);
      // exp + sum (inf → 0, 正常)
      float sum = 0.f;
      for (size_t k = 0; k < S; ++k) {
        const float e = std::exp(row[k] - mx);
        probs[q * S + k] = e;
        sum += e;
      }
      const float inv = sum > 0 ? 1.f / sum : 0.f;
      for (size_t k = 0; k < S; ++k) probs[q * S + k] *= inv;
    }
    // 8) Pack P 进 AmxPanel
    // P 是 fp32 → fp16 转换 + pack
    kern::AmxPanel pa_P;
    pa_P.rows = S; pa_P.K = S;
    // 用 sdpa_xh_ 暂存 P fp16
    if (sdpa_cap_ < S * S) {
      sdpa_xh_.resize(S * S);
      sdpa_cap_ = S * S;
    }
    for (size_t i = 0; i < S * S; ++i) {
      __fp16 hh = static_cast<__fp16>(probs[i]);
      __builtin_memcpy(&sdpa_xh_[i], &hh, 2);
    }
    kern::amx_pack_into(sdpa_xh_.data(), S, S, pa_P.buf);

    // 9) P·V via AMX: pa_P [S, S] (M=S), pa_V [HD, S] (N=HD, K=S)
    //    pa_V 是 V 转置 [HD, S] (从 V_f16 [S, HD] 转置), 否则 pa_P.K!=pa_V.K 被契约拒挥。
    //    输出是 [S, HD] 行主连续 (gemm_f16_amx_pp 标准输出布局),
    //    而 attn_ 是 [S, D] 行主, head h 段以 D 步距 — 先写临时 attn_tmp 再 scatter。
    std::vector<float> attn_tmp(S * HD, 0.f);
    kern::gemm_f16_amx_pp(pa_P, pa_V, attn_tmp.data(), S, HD);
    // scatter: attn_[q*D + h*HD + e] = attn_tmp[q*HD + e]  (D 上面已定义)
    for (size_t q = 0; q < S; ++q) {
      std::memcpy(attn_out + q * D + h * HD, attn_tmp.data() + q * HD,
                  HD * sizeof(float));
    }
  }
}
#endif  // GSV_AMX_GEMM
template <bool KV16>
void T2SEngine::block_prefill_impl(size_t l, float* x, size_t S, size_t pos,
                                   size_t text_len, float* kf32, uint16_t* k16,
                                   float* vf32, uint16_t* v16) {
  const size_t D = dims_.d_model, H = dims_.n_heads, HD = D / H;
  const size_t FF = dims_.ffn;
  Layer& L = layers_[l];
  const float scale = 1.0f / std::sqrt(static_cast<float>(HD));

  // fused QKV: qkv[S,3D] = x·Wqkvᵀ+b
  // E11-2: prefill 优先走 AMX(预打包面板) → gemm_f16_amx_pp;
  // 不可用时回退 DenseF16::forward(FMLAL) —— 数值同源(fp16 存储 + FMLAL 扩展精度累加),
  // 差异仅在中位 bit 舍入顺序。
  // gemm_f16_amx_pp 语义: C[M,N] = pa·pbᵀ, pa 形态 = M×K (激活侧), pb 形态 = N×K (权重侧)
  qkv_.resize(S * 3 * D);
#ifdef GSV_AMX_GEMM
  if (prefill_amx_in_use_) {
    if (prefill_cap_ < S * D) {
      prefill_xh_.resize(S * D);
      prefill_cap_ = S * D;
    }
    kern::f32_to_f16(x, prefill_xh_.data(), S * D);
    kern::AmxPanel pa_act;  // 激活面板: [S][D] = pa 侧 (M=S)
    pa_act.rows = S; pa_act.K = D;
    kern::amx_pack_into(prefill_xh_.data(), S, D, pa_act.buf);
    // L.wqkv_pa 预打包为 [3D][D] = pb 侧 (N=3D)
    kern::gemm_f16_amx_pp(pa_act, L.wqkv_pa, qkv_.data(), S, 3 * D);
  } else
#endif
  {
    L.wqkv.forward(x, S, qkv_.data());
  }
  for (size_t i = 0; i < S; ++i)
    for (size_t j = 0; j < 3 * D; ++j) qkv_[i * 3 * D + j] += L.bqkv[j];

  // k/v 写 cache 第 pos..pos+S-1 槽(token-major 行主, 头内连续 —— 与 qkv 行布局一致,
  // 直接按行拷贝 [token, D] 中 k 段/D 段)
  for (size_t t = 0; t < S; ++t) {
    const float* krow = qkv_.data() + t * 3 * D + D;
    const float* vrow = qkv_.data() + t * 3 * D + 2 * D;
    if constexpr (KV16) {
      kern::f32_to_f16(krow, k16 + (pos + t) * D, D);
      kern::f32_to_f16(vrow, v16 + (pos + t) * D, D);
    } else {
      std::memcpy(kf32 + (pos + t) * D, krow, D * sizeof(float));
      std::memcpy(vf32 + (pos + t) * D, vrow, D * sizeof(float));
    }
  }

  // SDPA: 掩码 allowed(q,k) = k<text_len || k<=q —— 文本块双向互见(text_len>0 时
  // 对 q<text_len 自动覆盖全部文本 key), 音频行因果。text_len==0 ⇒ 纯因果。
  // torch 侧为全行 softmax(-inf 掩蔽); 此处紧致遍历 allowed 集, 数学等价。
  // E11-1: 标量 → NEON 4-lane 树形 (同 decode)
  // E11-5: AMX 路径 —— 全 S×S 一次 GEMM (M=S, N=S, K=HD), 后 -inf 掩蔽 → softmax → P·V 全 GEMM
  //   K=HD=32 薄 K 形状 AMX 吞吐打折 (微基准: S=200 时 1.5x, S=300 1.8x), 收益实事求是;
  //   fp16 转换 + 归约序差可能影响 G1 软门 (实测 cos8≥0.9999 验证)。
  scores_.resize(S);
  probs_.resize(S);
  attn_.assign(S * D, 0.f);
  if constexpr (KV16) kvrow_.resize(D);

#ifdef GSV_AMX_GEMM
  if (prefill_amx_in_use_ && HD % 32 == 0) {  // AMX tile 是 32 行, HD=32 占一行, 保证对齐
    if constexpr (KV16)
      sdpa_amx_prefill<true>(H, S, HD, scale, text_len, qkv_.data(), kf32, k16, vf32, v16,
                             attn_.data());
    else
      sdpa_amx_prefill<false>(H, S, HD, scale, text_len, qkv_.data(), kf32, k16, vf32, v16,
                              attn_.data());
  } else
#endif
  {
    for (size_t h = 0; h < H; ++h) {
      for (size_t q = 0; q < S; ++q) {
        const float* qv = qkv_.data() + q * 3 * D + h * HD;
        const size_t n_max = (text_len > q + 1 ? text_len : q + 1);
        size_t n = 0;
        for (size_t k = 0; k < n_max; ++k) {
          if (!(k < text_len || k <= q)) continue;
          float dot;
          if constexpr (KV16) {
            dot = ar_neon::dot_f16kv(qv, k16 + (pos + k) * D + h * HD, HD);
          } else {
            dot = ar_neon::dot_f32(qv, kf32 + (pos + k) * D + h * HD, HD);
          }
          scores_[n++] = dot * scale;
        }
        gsv::kern::softmax(scores_.data(), probs_.data(), n);
        float* ov = attn_.data() + q * D + h * HD;
        size_t idx = 0;
        for (size_t k = 0; k < n_max; ++k) {
          if (!(k < text_len || k <= q)) continue;
          const float p = probs_[idx++];
          if constexpr (KV16) {
            ar_neon::accum_f16kv(ov, v16 + (pos + k) * D + h * HD, p, HD);
          } else {
            ar_neon::accum_f32(ov, vf32 + (pos + k) * D + h * HD, p, HD);
          }
        }
      }
    }
  }

  // out proj + 残差 + post-LN(norm1)
  tmp_.resize(S * D);
  L.wout.forward(attn_.data(), S, tmp_.data());
  for (size_t i = 0; i < S * D; ++i) x[i] += tmp_[i] + L.bout[i % D];
  for (size_t t = 0; t < S; ++t)
    gsv::kern::layernorm(x + t * D, L.n1g.data(), L.n1b.data(), x + t * D, D,
                         dims_.ln_eps);

  // FFN(ReLU) + 残差 + post-LN(norm2)
  ff_.resize(S * FF);
  // E11-2: W1 暂留 FMLAL —— ff_ 在 S~200 区间内 FMLAL ~ 1ms, AMX ~0.5ms 但漂移源多;
  // 数值稳定性优先(避免与 QKV/W2 同时改归约序导致雪崩对漂移)
  L.w1.forward(x, S, ff_.data());
  for (size_t i = 0; i < S * FF; ++i) ff_[i] += L.b1[i % FF];
  gsv::kern::relu(ff_.data(), ff_.data(), S * FF);

  // E11-2: W2 走 AMX (FFN 下降 S×FF → S×D)
#ifdef GSV_AMX_GEMM
  if (prefill_amx_in_use_) {
    if (prefill_cap_ < S * FF) {
      prefill_xh_.resize(S * FF);
      prefill_cap_ = S * FF;
    }
    kern::f32_to_f16(ff_.data(), prefill_xh_.data(), S * FF);
    kern::AmxPanel pa_act;  // 激活面板: [S][FF] = pa 侧 (M=S)
    pa_act.rows = S; pa_act.K = dims_.ffn;
    kern::amx_pack_into(prefill_xh_.data(), S, dims_.ffn, pa_act.buf);
    // L.w2_pa 预打包为 [D][FF] = pb 侧 (N=D)
    kern::gemm_f16_amx_pp(pa_act, L.w2_pa, tmp_.data(), S, D);
  } else
#endif
  {
    L.w2.forward(ff_.data(), S, tmp_.data());
  }
  for (size_t i = 0; i < S * D; ++i) x[i] += tmp_[i] + L.b2[i % D];
  for (size_t t = 0; t < S; ++t)
    gsv::kern::layernorm(x + t * D, L.n2g.data(), L.n2b.data(), x + t * D, D,
                         dims_.ln_eps);
}  // block_prefill_impl<KV16>

// ---- decode 单层: 与 T2SBlock.decode_next_token 同构 ----
// KV cache 布局: token-major 行主 cache[tok*D + head*HD + e](见头文件),
// 每 token 写一行, 注意力对 [0,len) 全可见(decode 阶段无掩码)。
// KV16: 存储 fp16/读出计算 fp32; GEMV16: 全部权重 GEMV 走 FMLAL 直读。
template <bool KV16, bool GEMV16>
void T2SEngine::block_decode_impl(size_t l, float* x, size_t pos, size_t len,
                                  float* kf32, uint16_t* k16, float* vf32,
                                  uint16_t* v16) {
  const size_t D = dims_.d_model, H = dims_.n_heads, HD = D / H;
  Layer& L = layers_[l];
  const float scale = 1.0f / std::sqrt(static_cast<float>(HD));

  // fused QKV 单行 GEMV: qkv[3D]
  dec_qkv_.resize(3 * D);
  float* qkv = dec_qkv_.data();
  if constexpr (GEMV16)
    gemv_fmlal(L.wqkv16, x, 3 * D, D, qkv);
  else
    L.wqkv.forward(x, 1, qkv);
  for (size_t j = 0; j < 3 * D; ++j) qkv[j] += L.bqkv[j];

  // 本 token 的 k/v 追加到第 pos 槽(len == pos+1, 见 generate 调用点)
  if constexpr (KV16) {
    kern::f32_to_f16(qkv + D, k16 + pos * D, D);
    kern::f32_to_f16(qkv + 2 * D, v16 + pos * D, D);
  } else {
    std::memcpy(kf32 + pos * D, qkv + D, D * sizeof(float));
    std::memcpy(vf32 + pos * D, qkv + 2 * D, D * sizeof(float));
  }

  // SDPA 单 query 对 len 个 key — E11-1: 标量 → NEON 4-lane 树形
  scores_.resize(len);
  attn_.assign(D, 0.f);
  if constexpr (KV16) kvrow_.resize(D);
  for (size_t h = 0; h < H; ++h) {
    const float* qv = qkv + h * HD;
    for (size_t k = 0; k < len; ++k) {
      float dot;
      if constexpr (KV16) {
        // 原地升位 + dot (避免中间缓冲跳跳; 升位为 inline 热路径)
        dot = ar_neon::dot_f16kv(qv, k16 + k * D + h * HD, HD);
      } else {
        dot = ar_neon::dot_f32(qv, kf32 + k * D + h * HD, HD);
      }
      scores_[k] = dot * scale;
    }
    gsv::kern::softmax(scores_.data(), scores_.data(), len);  // 就地稳定 softmax
    float* ov = attn_.data() + h * HD;
    for (size_t k = 0; k < len; ++k) {
      const float p = scores_[k];
      if constexpr (KV16) {
        ar_neon::accum_f16kv(ov, v16 + k * D + h * HD, p, HD);
      } else {
        ar_neon::accum_f32(ov, vf32 + k * D + h * HD, p, HD);
      }
    }
  }

  // out proj + 残差 + post-LN(norm1)
  dec_xb_.resize(D);
  float* xb = dec_xb_.data();
  if constexpr (GEMV16)
    gemv_fmlal(L.wout16, attn_.data(), D, D, xb);
  else
    L.wout.forward(attn_.data(), 1, xb);
  for (size_t i = 0; i < D; ++i) xb[i] += L.bout[i];
  for (size_t i = 0; i < D; ++i) xb[i] += x[i];
  gsv::kern::layernorm(xb, L.n1g.data(), L.n1b.data(), xb, D, dims_.ln_eps);

  // FFN(ReLU) + 残差 + post-LN(norm2)
  // 注意 post-LN 第二残差基是 norm1 输出 h(即当前 xb), 不是层输入 ——
  // 对应 torch `x = x + self.mlp.forward(x)`(此处 x 已是 norm1 之后)。
  std::vector<float>& hbuf = dec_h_;
  hbuf.resize(D);
  std::memcpy(hbuf.data(), xb, D * sizeof(float));
  ff_.resize(dims_.ffn);
  if constexpr (GEMV16)
    gemv_fmlal(L.w116, xb, dims_.ffn, D, ff_.data());
  else
    L.w1.forward(xb, 1, ff_.data());
  for (size_t i = 0; i < dims_.ffn; ++i) ff_[i] += L.b1[i];
  gsv::kern::relu(ff_.data(), ff_.data(), dims_.ffn);
  if constexpr (GEMV16)
    L.w2.forward(ff_.data(), 1, xb);  // W2 走 fp32 sgemm/升位
  else
    L.w2.forward(ff_.data(), 1, xb);
  for (size_t i = 0; i < D; ++i) xb[i] += L.b2[i];
  for (size_t i = 0; i < D; ++i) xb[i] += hbuf[i];
  gsv::kern::layernorm(xb, L.n2g.data(), L.n2b.data(), x, D, dims_.ln_eps);
}  // block_decode_impl<KV16,GEMV16>

template void T2SEngine::block_prefill_impl<false>(size_t, float*, size_t,
                                                   size_t, size_t, float*,
                                                   uint16_t*, float*, uint16_t*);
template void T2SEngine::block_prefill_impl<true>(size_t, float*, size_t,
                                                  size_t, size_t, float*,
                                                  uint16_t*, float*, uint16_t*);
template void T2SEngine::block_decode_impl<false, false>(size_t, float*,
                                                         size_t, size_t,
                                                         float*, uint16_t*,
                                                         float*, uint16_t*);
template void T2SEngine::block_decode_impl<true, false>(size_t, float*, size_t,
                                                        size_t, float*,
                                                        uint16_t*, float*,
                                                        uint16_t*);
template void T2SEngine::block_decode_impl<true, true>(size_t, float*, size_t,
                                                       size_t, float*,
                                                       uint16_t*, float*,
                                                       uint16_t*);

// ---- 贪心采样(infer_panel_naive sample() top_k=1 同构) ----
// 关键语义: torch 的 logits_to_probs 用 scatter_ 就地修改传入的 logits 张量,
// 而 golden 导出 hook 持有的是同一存储的 detached 视图 ⇒ .pt 里存的是"惩罚后"
// 的 logits。故本函数对 logits_io 就地施压, raw_argmax 与记录值都取惩罚后状态:
//   - eos_allowed=false(idx<11): torch 先切掉 EOS 列再 scatter_, EOS 列保持原始值;
//     raw_argmax 仍是全词表 argmax(golden `tokens` 口径)
int T2SEngine::greedy_sample(float* logits_io,
                             const std::vector<int32_t>& history,
                             bool eos_allowed, int* raw_argmax_out) {
  const int V = static_cast<int>(dims_.vocab);
  const int limit = eos_allowed ? V : V - 1;  // idx<11 时 EOS 列不受惩罚
  // repetition_penalty=1.35, 正值除/负值乘。
  // 关键语义: torch 先 gather 全部历史出现次数的"原始值"再统一 scatter_ 写回,
  // 重复 token 的多次写互相覆盖 ⇒ 每个"去重后"的 token 只施压一次(非逐次累乘)。
  if (pen_mark_.size() != dims_.vocab) {
    pen_mark_.assign(dims_.vocab, 0);
    pen_stamp_ = 0;
  }
  ++pen_stamp_;
  for (const int32_t c : history) {
    if (static_cast<int>(c) >= limit) continue;
    if (pen_mark_[static_cast<size_t>(c)] == pen_stamp_) continue;
    pen_mark_[static_cast<size_t>(c)] = pen_stamp_;
    float& v = logits_io[static_cast<size_t>(c)];
    v = v < 0.f ? v * kRepPenalty : v / kRepPenalty;
  }
  // 全词表 argmax(golden `tokens` 口径, 含保持原始值的 EOS 列)
  if (raw_argmax_out) {
    int best = 0;
    float bv = logits_io[0];
    for (int c = 1; c < V; ++c) {
      if (logits_io[c] > bv) { bv = logits_io[c]; best = c; }
    }
    *raw_argmax_out = best;
  }
  // 采样 argmax(top_k=1 ⇒ one-hot; idx<11 时剔除 EOS 列 —— torch logits[:, :-1])
  int best = 0;
  float bv = logits_io[0];
  for (int c = 1; c < limit; ++c) {
    if (logits_io[c] > bv) { bv = logits_io[c]; best = c; }
  }
  return best;
}

// ---- E4: topk_sample —— 对齐 python infer_panel_naive sample() ----
// 链顺序: repetition_penalty → top_k(15) → top_p(1.0) → temperature(1.0) → multinomial
// 关键语义:
//  - 惩罚对"去重后"history 各施压一次(torch scatter_ 覆盖语义, 同 greedy_sample)
//  - idx<11 ⇒ EOS 列保持原始值且不参与采样(torch logits[:, :-1] 前置裁切)
//  - temperature=1.0 时 softmax 等价直接归一化; 非 1.0 时 logits /= temperature
//  - multinomial 用浮点 CDF 累加 + uniform 比较(1025 词表无需别名表, < 5us 一次)
//  - seed 来自 SamplingParams(0 = std::random_device 真随机)
int T2SEngine::topk_sample(float* logits_io,
                           const std::vector<int32_t>& history,
                           bool eos_allowed, int* raw_argmax_out,
                           const SamplingParams& sp, std::mt19937_64& rng) {
  const int V = static_cast<int>(dims_.vocab);
  const int limit = eos_allowed ? V : V - 1;
  // 1) repetition_penalty (同 greedy_sample, 标量化 rep_penalty 入参)
  if (pen_mark_.size() != dims_.vocab) {
    pen_mark_.assign(dims_.vocab, 0);
    pen_stamp_ = 0;
  }
  ++pen_stamp_;
  for (const int32_t c : history) {
    if (static_cast<int>(c) >= limit) continue;
    if (pen_mark_[static_cast<size_t>(c)] == pen_stamp_) continue;
    pen_mark_[static_cast<size_t>(c)] = pen_stamp_;
    float& v = logits_io[static_cast<size_t>(c)];
    v = v < 0.f ? v * sp.rep_penalty : v / sp.rep_penalty;
  }
  // 2) raw_argmax: 全词表 argmax(含 EOS, 保持原始值)
  if (raw_argmax_out) {
    int best = 0;
    float bv = logits_io[0];
    for (int c = 1; c < V; ++c) {
      if (logits_io[c] > bv) { bv = logits_io[c]; best = c; }
    }
    *raw_argmax_out = best;
  }
  // 3) top_k: 在 [0, limit) 范围中保留 sp.top_k 个最高(此为样本有效域)
  //    策略: 在 limit 范围内收集 top_k (insertion-select, k≤15 1025 上 O(k·V) 仍 < 20us)
  const int k = sp.top_k > static_cast<size_t>(limit)
                    ? limit
                    : static_cast<int>(sp.top_k);
  // 简化: 使用部分排序后的索引表 (V=1025, 1025*4B = 4KB, 贴进 L1)
  // 为避免动态分配, 复用 per-instance topk_idx_/topk_val_ (在 generate() 期用, 尺寸 V)
  int* idxs = topk_idx_.data();
  float* vals = topk_val_.data();
  for (int c = 0; c < limit; ++c) { idxs[c] = c; vals[c] = logits_io[c]; }
  // partial sort top-k (selection sort k 轮, k≤15, V=1025 ⇒ 15*1025 = 15k 比较)
  for (int i = 0; i < k; ++i) {
    int max_j = i;
    float max_v = vals[i];
    for (int j = i + 1; j < limit; ++j) {
      if (vals[j] > max_v) { max_v = vals[j]; max_j = j; }
    }
    if (max_j != i) {
      std::swap(vals[i], vals[max_j]);
      std::swap(idxs[i], idxs[max_j]);
    }
  }
  // 4) temperature 应用到 top-k logits(若 != 1.0)
  //    temperature=0 ⇒ 退贪心
  if (sp.temperature <= 0.f) {
    int best = idxs[0];
    float bv = vals[0];
    for (int i = 1; i < k; ++i) {
      if (vals[i] > bv) { bv = vals[i]; best = idxs[i]; }
    }
    return best;
  }
  if (sp.temperature != 1.0f) {
    for (int i = 0; i < k; ++i) vals[i] = vals[i] / sp.temperature;
  }
  // 5) softmax over top-k (数值稳定: 减 max)
  float max_logit = vals[0];
  for (int i = 1; i < k; ++i) {
    if (vals[i] > max_logit) max_logit = vals[i];
  }
  float sum_exp = 0.f;
  for (int i = 0; i < k; ++i) {
    vals[i] = std::exp(vals[i] - max_logit);
    sum_exp += vals[i];
  }
  // 6) top_p (nucleus): 从高到低累加, 保留累积 ≥ (1 - top_p) 比例的最小前缀
  int keep = k;
  if (sp.top_p < 1.0f) {
    const float threshold = (1.f - sp.top_p) * sum_exp;
    float cum = 0;
    for (int i = k - 1; i >= 0; --i) {
      cum += vals[i];
      if (cum >= threshold) { keep = i + 1; break; }
    }
    // 重算 sum_exp (裁掉后部分)
    if (keep < k) {
      sum_exp = 0;
      for (int i = 0; i < keep; ++i) sum_exp += vals[i];
    }
  }
  // 7) multinomial: 浮点 CDF + uniform 比较
  //    1025 词表, keep≤15, 线性扫描 < 100ns
  std::uniform_real_distribution<float> uni(0.f, 1.f);
  const float u = uni(rng) * sum_exp;
  float cdf = 0.f;
  for (int i = 0; i < keep; ++i) {
    cdf += vals[i];
    if (u <= cdf) return idxs[i];
  }
  return idxs[keep - 1];  // 数值末位兜底
}

GenResult T2SEngine::generate(const int64_t* phones, size_t T,
                              const int64_t* prompt, size_t P,
                              const float* bert1024, size_t max_steps,
                              GenDebug* dbg, const SamplingParams* sampling) {
  const size_t D = dims_.d_model;
  if (T == 0 || P == 0) throw std::runtime_error("T/P 必须非零(golden 口径)");
  const size_t S = T + P;

  // ---- scratch/cache 容量(跨调用复用; 模式切换或容量不足时重建) ----
  if (cap_ < S + max_steps || fp16_.kv != kv_mode_active_) {
    cap_ = S + max_steps;
    kv_mode_active_ = fp16_.kv;
    kc_.assign(dims_.n_layers, {});
    vc_.assign(dims_.n_layers, {});
    kc16_.assign(dims_.n_layers, {});
    vc16_.assign(dims_.n_layers, {});
    for (size_t l = 0; l < dims_.n_layers; ++l) {
      if (fp16_.kv) {
        kc16_[l].resize(cap_ * D);
        vc16_[l].resize(cap_ * D);
      } else {
        kc_[l].resize(cap_ * D);
        vc_[l].resize(cap_ * D);
      }
    }
  }

  GenResult r;

  // ---- 输入重建 xy[S,D]: 文本 emb+bert_proj+PE | 音频 prompt emb+PE ----
  xy_.resize(S * D);
  pe_.resize(D);
  bert_proj_.forward(bert1024, T, xy_.data());  // bert_proj(bert_feat) 先落 xy
  for (size_t t = 0; t < T; ++t) {
    const float* erow = text_emb_.data() + static_cast<size_t>(phones[t]) * D;
    float* xr = xy_.data() + t * D;
    for (size_t d = 0; d < D; ++d) xr[d] += erow[d] + bert_b_[d];
    pe_row(pe_.data(), t);
    for (size_t d = 0; d < D; ++d) xr[d] += alpha_text_ * pe_[d];
  }
  for (size_t t = 0; t < P; ++t) {
    const float* erow = audio_emb_.data() + static_cast<size_t>(prompt[t]) * D;
    float* xr = xy_.data() + (T + t) * D;
    pe_row(pe_.data(), t);
    for (size_t d = 0; d < D; ++d) xr[d] = erow[d] + alpha_audio_ * pe_[d];
  }

  // ---- B1: prefill(24 层, 大矩阵乘固定 Accelerate sgemm) ----
  if (dbg) dbg->on_input(xy_.data(), S);
  const auto t0 = std::chrono::steady_clock::now();
  bool hit = false;
  if (kv_reuse_ && prompt_snapshot_.valid &&
      prompt_snapshot_.is_fp16 == fp16_.kv &&
      prompt_snapshot_.T == T && prompt_snapshot_.P == P &&
      std::memcmp(prompt_snapshot_.phones.data(), phones, T * sizeof(int64_t)) == 0 &&
      std::memcmp(prompt_snapshot_.prompt.data(), prompt, P * sizeof(int64_t)) == 0 &&
      std::memcmp(prompt_snapshot_.bert1024.data(), bert1024, T * dims_.bert_dim * sizeof(float)) == 0) {
    hit = true;
    for (size_t l = 0; l < dims_.n_layers; ++l) {
      if (fp16_.kv) {
        std::memcpy(kc16_[l].data(), prompt_snapshot_.kc16[l].data(), S * D * sizeof(uint16_t));
        std::memcpy(vc16_[l].data(), prompt_snapshot_.vc16[l].data(), S * D * sizeof(uint16_t));
      } else {
        std::memcpy(kc_[l].data(), prompt_snapshot_.kc[l].data(), S * D * sizeof(float));
        std::memcpy(vc_[l].data(), prompt_snapshot_.vc[l].data(), S * D * sizeof(float));
      }
    }
  } else {
    for (size_t l = 0; l < dims_.n_layers; ++l) {
      if (fp16_.kv)
        block_prefill_impl<true>(l, xy_.data(), S, /*pos=*/0, /*text_len=*/T,
                                 nullptr, kc16_[l].data(), nullptr,
                                 vc16_[l].data());
      else
        block_prefill_impl<false>(l, xy_.data(), S, /*pos=*/0, /*text_len=*/T,
                                  kc_[l].data(), nullptr, vc_[l].data(),
                                  nullptr);
      if (dbg) dbg->on_layer(l, xy_.data(), S);
    }
    if (kv_reuse_) {
      prompt_snapshot_.phones.assign(phones, phones + T);
      prompt_snapshot_.prompt.assign(prompt, prompt + P);
      prompt_snapshot_.bert1024.assign(bert1024, bert1024 + T * dims_.bert_dim);
      prompt_snapshot_.S = S;
      prompt_snapshot_.T = T;
      prompt_snapshot_.P = P;
      prompt_snapshot_.is_fp16 = fp16_.kv;
      prompt_snapshot_.last_xy_row.assign(xy_.data() + (S - 1) * D, xy_.data() + S * D);
      prompt_snapshot_.kc.assign(dims_.n_layers, {});
      prompt_snapshot_.vc.assign(dims_.n_layers, {});
      prompt_snapshot_.kc16.assign(dims_.n_layers, {});
      prompt_snapshot_.vc16.assign(dims_.n_layers, {});
      for (size_t l = 0; l < dims_.n_layers; ++l) {
        if (fp16_.kv) {
          prompt_snapshot_.kc16[l].assign(kc16_[l].begin(), kc16_[l].begin() + S * D);
          prompt_snapshot_.vc16[l].assign(vc16_[l].begin(), vc16_[l].begin() + S * D);
        } else {
          prompt_snapshot_.kc[l].assign(kc_[l].begin(), kc_[l].begin() + S * D);
          prompt_snapshot_.vc[l].assign(vc_[l].begin(), vc_[l].begin() + S * D);
        }
      }
      prompt_snapshot_.valid = true;
    }
  }
  const auto t1 = std::chrono::steady_clock::now();
  last_prefill_hit_ = hit;

  // 首 logits 来自最后一个音频 prompt 位置(xy_dec[:, -1])
  logits_.resize(dims_.vocab);
  if (hit) {
    predict_layer_fp(prompt_snapshot_.last_xy_row.data(), logits_.data());
  } else {
    predict_layer_fp(xy_.data() + (S - 1) * D, logits_.data());
  }

  // ---- B2: decode 循环(GEMV + KV cache fp32 + 贪心) ----
  // E4: 采样路径 —— sampling.mode=TopK 时走 topk_sample (复现 python 默认 15 采样, 根治复读)
  // 默认 Greedy 不变(位级一致保证 B12 golden G1/G2)
  const bool use_topk = sampling && sampling->mode == SamplingParams::Mode::TopK;
  std::mt19937_64 rng = [&]() {
    if (sampling && sampling->seed != 0) return std::mt19937_64(sampling->seed);
    std::random_device rd;
    return std::mt19937_64(rd());
  }();
  if (use_topk) {
    topk_idx_.assign(dims_.vocab, 0);
    topk_val_.assign(dims_.vocab, 0.f);
  }
  std::vector<int32_t> history;
  history.reserve(P + max_steps);
  for (size_t t = 0; t < P; ++t) history.push_back(static_cast<int32_t>(prompt[t]));
  r.logits_first8.reserve(8 * dims_.vocab);
  r.sampled.reserve(max_steps);
  r.raw_argmax.reserve(max_steps);

  size_t cur_len = S;  // 已写入 cache 的 token 数
  bool stop = false;
  size_t idx = 0;
  for (; idx < max_steps && !stop; ++idx) {
    float* lg = logits_.data();
    int raw = -1;
    const bool eos_allowed = idx >= kEosMaskSteps;
    // 采样 + 就地惩罚(torch scatter_ 语义): 惩罚后 logits_ 才是 golden 捕获口径
    int sample;
    if (use_topk) {
      sample = topk_sample(lg, history, eos_allowed, &raw, *sampling, rng);
    } else {
      sample = greedy_sample(lg, history, eos_allowed, &raw);
    }
    r.raw_argmax.push_back(raw);
    if (r.logits_first8.size() < 8 * dims_.vocab)
      r.logits_first8.insert(r.logits_first8.end(), lg, lg + dims_.vocab);

    if (eos_allowed && sample == dims_.eos) {  // EOS 触发停止且不入序列
      stop = true;
      r.hit_eos = true;
    } else {
      r.sampled.push_back(sample);
      history.push_back(sample);
    }
    if (idx + 1 == max_steps) stop = true;  // 步数保护(torch idx==MAX-1)

    if (stop) break;

    // 下一步输入 = emb_audio(sample) + alpha_audio·PE(P+idx)
    x1_.resize(D);
    const float* erow = audio_emb_.data() + static_cast<size_t>(sample) * D;
    pe_row(pe_.data(), P + idx);
    for (size_t d = 0; d < D; ++d) x1_[d] = erow[d] + alpha_audio_ * pe_[d];

    for (size_t l = 0; l < dims_.n_layers; ++l) {
      if (fp16_.kv) {
        if (fp16_.gemv)
          block_decode_impl<true, true>(l, x1_.data(), cur_len, cur_len + 1,
                                        nullptr, kc16_[l].data(), nullptr,
                                        vc16_[l].data());
        else
          block_decode_impl<true, false>(l, x1_.data(), cur_len,
                                         cur_len + 1, nullptr,
                                         kc16_[l].data(), nullptr,
                                         vc16_[l].data());
      } else {
        block_decode_impl<false, false>(l, x1_.data(), cur_len, cur_len + 1,
                                        kc_[l].data(), nullptr,
                                        vc_[l].data(), nullptr);
      }
    }
    ++cur_len;
    predict_layer_fp(x1_.data(), logits_.data());
  }
  const auto t2 = std::chrono::steady_clock::now();

  r.steps = idx + 1;  // 含触发停止的当前步(hook 计数口径)
  r.logits_last.assign(logits_.data(), logits_.data() + dims_.vocab);
  last_prefill_ms_ = std::chrono::duration<double, std::milli>(t1 - t0).count();
  last_decode_ms_ = std::chrono::duration<double, std::milli>(t2 - t1).count();
  // E11-2 起加 GSV_AR_TIMING=1 探针, stderr 打印 prefill/decode/全程耗时(用与 E11-1/E11-2 验收报表)
  if (std::getenv("GSV_AR_TIMING")) {
    const double ms_per_tok =
        last_decode_ms_ / static_cast<double>(r.steps);
    std::fprintf(stderr,
                 "[ar-timing] T=%zu P=%zu S=%zu steps=%zu prefill=%.2fms (hit=%d) "
                 "decode=%.2fms (%.3f ms/tok)\n",
                 T, P, S, r.steps, last_prefill_ms_, last_prefill_hit_ ? 1 : 0,
                 last_decode_ms_, ms_per_tok);
  }
  return r;
}

}  // namespace gsv::ar
