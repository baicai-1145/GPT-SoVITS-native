// mha.hpp — MultiHeadAttention (attentions.py), 可选相对位置编码 window_size=4
// 数值语义与 torch 同构:
//   scores = (q/√dk)·kᵀ (+ rel_k 经 _relative_position_to_absolute_position)
//   masked_fill(mask==0, -1e4) → softmax(dim=-1) → p·v
//   (+ p 经 _absolute_position_to_relative_position · rel_v)
#pragma once

#include <cmath>
#include <string>
#include <vector>

#include "conv1d.hpp"
#include "sovits/nn_ops.hpp"

namespace gsv::sovits {

class MultiHeadAttention {
 public:
  size_t channels = 0, heads = 0, dk = 0;
  size_t win_table = 0;  // 2W+1; 0 表示无相对位置编码
  Conv1d conv_q, conv_k, conv_v, conv_o;
  std::vector<float> emb_rel_k, emb_rel_v;  // [win_table, dk]

  void load(const rt::GsvFile& f, std::string_view prefix, size_t ch,
            size_t nh, size_t table) {
    channels = ch;
    heads = nh;
    dk = ch / nh;
    win_table = table;
    std::string p(prefix);
    conv_q.load(f, p + ".conv_q", ch, ch, 1);
    conv_k.load(f, p + ".conv_k", ch, ch, 1);
    conv_v.load(f, p + ".conv_v", ch, ch, 1);
    conv_o.load(f, p + ".conv_o", ch, ch, 1);
    if (table) {
      load_tensor_f32(f, p + ".emb_rel_k", emb_rel_k,
                      {static_cast<size_t>(1), table, dk});
      load_tensor_f32(f, p + ".emb_rel_v", emb_rel_v,
                      {static_cast<size_t>(1), table, dk});
    }
  }

  bool has_rel() const { return win_table != 0; }

  // torch _get_relative_embeddings: [win_table,dk] → used[2L-1, dk]
  static void relative_slice(const std::vector<float>& emb, size_t win_table,
                             size_t dk, size_t L, float* used) {
    const size_t half_w = win_table / 2;  // W
    const size_t pad_len = (L > half_w + 1) ? L - (half_w + 1) : 0;
    const size_t start = (half_w + 1 > L) ? (half_w + 1 - L) : 0;
    for (size_t m = 0; m < 2 * L - 1; ++m) {
      const long long src =
          static_cast<long long>(m + start) - static_cast<long long>(pad_len);
      const bool ok = src >= 0 && src < static_cast<long long>(win_table);
      for (size_t d = 0; d < dk; ++d)
        used[m * dk + d] = ok ? emb[static_cast<size_t>(src) * dk + d] : 0.f;
    }
  }

  // q_in/kv_in [C,T]; mask[Tq*Tk] (行主 Tq×Tk, 1=有效); out [C,Tq]
  // dbg: 可选中间量导出 (前缀 tag, 仅 self-attn 调试用)
  void forward(const Tensor2D& q_in, const Tensor2D& kv_in,
               const std::vector<uint8_t>& mask, Tensor2D& out, Tensor2D& qb,
               Tensor2D& kb, Tensor2D& vb, Tensor2D& scores,
               const Dumper* dbg = nullptr) const {
    const size_t Tq = q_in.T, Tk = kv_in.T;
    conv_q.forward(q_in, qb);
    conv_k.forward(kv_in, kb);
    conv_v.forward(kv_in, vb);

    const float scale = 1.f / std::sqrt(static_cast<float>(dk));
    scores.reset(heads * Tq, Tk);
    // E10-mha-sgemm: scores[h] = scale * Q[h]^T · K[h] via Accelerate sgemm.
    // Q/K 布局 [h*dk, T] 行主 → Q[h]^T 是 [Tq, dk] (ldA=Tq), K[h] 是 [dk, Tk] (ldB=Tk)。
    // 逐头一次 sgemm (h≤4, 远低于批门限; BLAS 调优已自洽)。
    for (size_t h = 0; h < heads; ++h) {
      kern::accel::sgemm('T', 'N', static_cast<int>(Tq), static_cast<int>(Tk),
                         static_cast<int>(dk), scale,
                         &qb.d[(h * dk) * Tq], static_cast<int>(Tq),
                         &kb.d[(h * dk) * Tk], static_cast<int>(Tk),
                         0.0f, &scores.d[(h * Tq) * Tk], static_cast<int>(Tk));
    }
    if (dbg) {
      dbg->dump("mha_q", qb);
      dbg->dump("mha_k", kb);
      dbg->dump("mha_v", vb);
      dbg->dump("mha_scores_core", scores);
    }

    thread_local std::vector<float> usedk_buf, usedv_buf;
    if (has_rel()) {
      const size_t L = Tq;
      usedk_buf.resize((2 * L - 1) * dk);
      relative_slice(emb_rel_k, win_table, dk, L, usedk_buf.data());
      usedv_buf.resize((2 * L - 1) * dk);
      relative_slice(emb_rel_v, win_table, dk, L, usedv_buf.data());
      // rel_logits[h,t,m]
      thread_local std::vector<float> rel_logits;
      rel_logits.assign(heads * L * (2 * L - 1), 0.f);
      // E10-mha-sgemm: rel_logits[h] = scale * Q[h]^T · usedk_buf^T
      //   Q[h] = [dk, Tq] (ldA=Tq), usedk_buf = [2L-1, dk] (ldB=dk)
      //   rel_logits[h] = [Tq, 2L-1] (ldC=2L-1)
      for (size_t h = 0; h < heads; ++h) {
        kern::accel::sgemm('T', 'T', static_cast<int>(L),
                           static_cast<int>(2 * L - 1), static_cast<int>(dk),
                           scale,
                           &qb.d[(h * dk) * Tq], static_cast<int>(Tq),
                           usedk_buf.data(), static_cast<int>(dk),
                           0.0f, &rel_logits[(h * L) * (2 * L - 1)],
                           static_cast<int>(2 * L - 1));
      }
      if (dbg) {
        // torch rel_logits [b,h,L,2L-1]: 内存序 [h][L][m]
        Tensor2D rl_tmp(heads * L, 2 * L - 1);
        for (size_t i = 0; i < rl_tmp.d.size(); ++i)
          rl_tmp.d[i] = rel_logits[i];
        dbg->dump("mha_rel_logits", rl_tmp);
      }
      // rel→abs 加进 scores (_relative_position_to_absolute_position 索引同构):
      //   n=i*(2L-1)+L-1+j; a=n/(2L), b=n%(2L); val = b==2L-1 ? 0 : rl[a,b]
      for (size_t h = 0; h < heads; ++h)
        for (size_t i = 0; i < L; ++i)
          for (size_t j = 0; j < L; ++j) {
            const size_t n = i * (2 * L - 1) + j + L - 1;
            float v = 0.f;
            if (n < 2 * L * L) {
              const size_t a = n / (2 * L), b = n % (2 * L);
              if (b != 2 * L - 1)
                v = rel_logits[(h * L + a) * (2 * L - 1) + b];
            }
            scores.d[(h * L + i) * Tk + j] += v;
          }
      if (dbg) {
        Tensor2D ra_tmp(heads * L, L);
        for (size_t h2 = 0; h2 < heads; ++h2)
          for (size_t i = 0; i < L; ++i)
            for (size_t j = 0; j < L; ++j) {
              // 重算 rel_abs 与上方同式
              const size_t n = i * (2 * L - 1) + j + L - 1;
              float vv = 0.f;
              if (n < 2 * L * L) {
                const size_t a = n / (2 * L), b = n % (2 * L);
                if (b != 2 * L - 1)
                  vv = rel_logits[(h2 * L + a) * (2 * L - 1) + b];
              }
              ra_tmp.d[(h2 * L + i) * L + j] = vv;
            }
        dbg->dump("mha_rel_abs", ra_tmp);
      }
    }

    // mask → -1e4, softmax over s
    for (size_t h = 0; h < heads; ++h)
      for (size_t t = 0; t < Tq; ++t) {
        float* sr = &scores.d[(h * Tq + t) * Tk];
        for (size_t s = 0; s < Tk; ++s)
          if (!mask[t * Tk + s]) sr[s] = -1e4f;
        softmax_row_inplace(sr, Tk);
      }
    if (dbg) dbg->dump("mha_p_attn", scores);

    // out = Σ_s p·v
    out.reset(channels, Tq);
    for (size_t h = 0; h < heads; ++h)
      for (size_t t = 0; t < Tq; ++t) {
        float* op = &out.d[(h * dk) * Tq + t];
        const float* pr = &scores.d[(h * Tq + t) * Tk];
        for (size_t s = 0; s < Tk; ++s) {
          const float p = pr[s];
          if (p == 0.f) continue;
          for (size_t d = 0; d < dk; ++d)
            op[d * Tq] += p * vb.d[(h * dk + d) * Tk + s];
        }
      }
    if (has_rel()) {
      const size_t L = Tq;
      // p_rel[h,i,mrel]: _absolute_position_to_relative_position 索引同构:
      //   n=i*2L+mrel+1; n<L → 0; 否则 a=(n-L)/(2L-1), b=(n-L)%(2L-1),
      //   w = b<L ? p[a,b] : 0 (b∈[L,2L-2] 是 pad 零区)
      for (size_t h = 0; h < heads; ++h)
        for (size_t i = 0; i < L; ++i)
          for (size_t mrel = 0; mrel < 2 * L - 1; ++mrel) {
            const size_t n = i * 2 * L + mrel + 1;
            float w = 0.f;
            if (n >= L) {
              const size_t mm = n - L;
              const size_t a = mm / (2 * L - 1), b = mm % (2 * L - 1);
              w = (b < L) ? scores.d[(h * L + a) * L + b] : 0.f;
            }
            if (w == 0.f) continue;
            for (size_t d = 0; d < dk; ++d)
              out.d[(h * dk + d) * Tq + i] += w * usedv_buf[mrel * dk + d];
          }
      if (dbg) {
        Tensor2D ao_tmp(heads * dk, Tq);
        for (size_t i2 = 0; i2 < out.d.size(); ++i2)
          ao_tmp.d[i2] = out.d[i2];
        dbg->dump("mha_attn_out", ao_tmp);
      }
    }

    conv_o.forward(out, out);
  }
};

}  // namespace gsv::sovits
