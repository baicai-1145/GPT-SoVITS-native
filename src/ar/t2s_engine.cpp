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

#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

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
  }
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
template <bool KV16>
void T2SEngine::block_prefill_impl(size_t l, float* x, size_t S, size_t pos,
                                   size_t text_len, float* kf32, uint16_t* k16,
                                   float* vf32, uint16_t* v16) {
  const size_t D = dims_.d_model, H = dims_.n_heads, HD = D / H;
  const size_t FF = dims_.ffn;
  Layer& L = layers_[l];
  const float scale = 1.0f / std::sqrt(static_cast<float>(HD));

  // fused QKV: qkv[S,3D] = x·Wqkvᵀ+b (prefill 大矩阵乘固定 Accelerate sgemm)
  qkv_.resize(S * 3 * D);
  L.wqkv.forward(x, S, qkv_.data());
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
  // KV16 读出: 每 key 行先升位到 kvrow_(fp32), 计算全 fp32。
  scores_.resize(S);
  probs_.resize(S);
  attn_.assign(S * D, 0.f);
  if constexpr (KV16) kvrow_.resize(D);
  for (size_t h = 0; h < H; ++h) {
    for (size_t q = 0; q < S; ++q) {
      const float* qv = qkv_.data() + q * 3 * D + h * HD;
      const size_t n_max = (text_len > q + 1 ? text_len : q + 1);
      size_t n = 0;
      for (size_t k = 0; k < n_max; ++k) {
        if (!(k < text_len || k <= q)) continue;
        const float* kv;
        if constexpr (KV16) {
          accel::f16_to_f32(k16 + (pos + k) * D + h * HD, kvrow_.data(), HD);
          kv = kvrow_.data();
        } else {
          kv = kf32 + (pos + k) * D + h * HD;
        }
        float dot = 0.f;
        for (size_t e = 0; e < HD; ++e) dot += qv[e] * kv[e];
        scores_[n++] = dot * scale;
      }
      gsv::kern::softmax(scores_.data(), probs_.data(), n);
      float* ov = attn_.data() + q * D + h * HD;
      size_t idx = 0;
      for (size_t k = 0; k < n_max; ++k) {
        if (!(k < text_len || k <= q)) continue;
        const float p = probs_[idx++];
        const float* vv;
        if constexpr (KV16) {
          accel::f16_to_f32(v16 + (pos + k) * D + h * HD, kvrow_.data(), HD);
          vv = kvrow_.data();
        } else {
          vv = vf32 + (pos + k) * D + h * HD;
        }
        for (size_t e = 0; e < HD; ++e) ov[e] += p * vv[e];
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
  L.w1.forward(x, S, ff_.data());
  for (size_t i = 0; i < S * FF; ++i) ff_[i] += L.b1[i % FF];
  gsv::kern::relu(ff_.data(), ff_.data(), S * FF);
  L.w2.forward(ff_.data(), S, tmp_.data());
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

  // SDPA 单 query 对 len 个 key(KV16: 每 key 升位 fp32 后计算)
  scores_.resize(len);
  attn_.assign(D, 0.f);
  if constexpr (KV16) kvrow_.resize(D);
  for (size_t h = 0; h < H; ++h) {
    const float* qv = qkv + h * HD;
    for (size_t k = 0; k < len; ++k) {
      const float* kv;
      if constexpr (KV16) {
        accel::f16_to_f32(k16 + k * D + h * HD, kvrow_.data(), HD);
        kv = kvrow_.data();
      } else {
        kv = kf32 + k * D + h * HD;
      }
      float dot = 0.f;
      for (size_t e = 0; e < HD; ++e) dot += qv[e] * kv[e];
      scores_[k] = dot * scale;
    }
    gsv::kern::softmax(scores_.data(), scores_.data(), len);  // 就地稳定 softmax
    float* ov = attn_.data() + h * HD;
    for (size_t k = 0; k < len; ++k) {
      const float p = scores_[k];
      const float* vv;
      if constexpr (KV16) {
        accel::f16_to_f32(v16 + k * D + h * HD, kvrow_.data(), HD);
        vv = kvrow_.data();
      } else {
        vv = vf32 + k * D + h * HD;
      }
      for (size_t e = 0; e < HD; ++e) ov[e] += p * vv[e];
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

GenResult T2SEngine::generate(const int64_t* phones, size_t T,
                              const int64_t* prompt, size_t P,
                              const float* bert1024, size_t max_steps,
                              GenDebug* dbg) {
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
  const auto t1 = std::chrono::steady_clock::now();

  // 首 logits 来自最后一个音频 prompt 位置(xy_dec[:, -1])
  logits_.resize(dims_.vocab);
  predict_layer_fp(xy_.data() + (S - 1) * D, logits_.data());

  // ---- B2: decode 循环(GEMV + KV cache fp32 + 贪心) ----
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
    const int sample = greedy_sample(lg, history, eos_allowed, &raw);
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
  return r;
}

}  // namespace gsv::ar
