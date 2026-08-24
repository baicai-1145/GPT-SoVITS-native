// enc_p.hpp — TextEncoder (v2ProPlus decode 路径, models.py TextEncoder)
//   y = ssl_proj(q·mask)·mask → Encoder×3
//   text = text_embedding(phones)ᵀ → Encoder×6
//   y = MRTE(y, text, ge_text)  [512 维空间 + ge_text 广播]
//   y = Encoder×3 → proj(y)·mask → split(m_p, logs_p)
// Encoder 层 = rel-MHA(window4, heads2) + LN(1e-5) + FFN(k3,relu) + LN。
#pragma once

#include <cmath>
#include <string>
#include <vector>

#include "sovits/mha.hpp"

namespace gsv::sovits {

class FFN {
 public:
  Conv1d conv_1, conv_2;
  // torch: conv_1 = Conv1d(in_ch, filt, k) 权重[filt,in,k]; conv_2 反之
  void load(const rt::GsvFile& f, std::string_view prefix, size_t in_ch,
            size_t filt, size_t k) {
    std::string p(prefix);
    conv_1.load(f, p + ".conv_1", filt, in_ch, k);
    conv_2.load(f, p + ".conv_2", in_ch, filt, k);
  }
  void forward(const Tensor2D& x, Tensor2D& h, Tensor2D& y) const {
    conv_1.forward(x, h);
    for (float& v : h.d) v = v > 0.f ? v : 0.f;  // relu
    conv_2.forward(h, y);
  }
};

class Encoder {
 public:
  struct Layer {
    MultiHeadAttention attn;
    std::vector<float> ln1_g, ln1_b, ln2_g, ln2_b;
    FFN ffn;
  };
  size_t channels = 0, n_layers = 0;
  std::vector<Layer> layers;

  void load(const rt::GsvFile& f, std::string_view prefix, size_t ch,
            size_t heads, size_t nl, size_t filt, size_t k, size_t win_table) {
    // .gsv 命名: {prefix}.attn_layers.{i}.* / {prefix}.norm_layers_1.{i}.* /
    //           {prefix}.ffn_layers.{i}.* / {prefix}.norm_layers_2.{i}.*
    channels = ch;
    n_layers = nl;
    layers.resize(nl);
    for (size_t i = 0; i < nl; ++i) {
      const std::string idx = "." + std::to_string(i);
      const std::string p(prefix);
      Layer& L = layers[i];
      L.attn.load(f, p + ".attn_layers" + idx, ch, heads, win_table);
      load_tensor_f32(f, p + ".norm_layers_1" + idx + ".gamma", L.ln1_g);
      load_tensor_f32(f, p + ".norm_layers_1" + idx + ".beta", L.ln1_b);
      load_tensor_f32(f, p + ".norm_layers_2" + idx + ".gamma", L.ln2_g);
      load_tensor_f32(f, p + ".norm_layers_2" + idx + ".beta", L.ln2_b);
      L.ffn.load(f, p + ".ffn_layers" + idx, ch, filt, k);
    }
  }

  // x: 输入输出共用 (结果写回); mask[T*T] 全有效场景由调用方填 1
  // dbg_tag 非空时逐层 dump {tag}_L{i}_out
  void forward(Tensor2D& x, const std::vector<uint8_t>& mask,
               const Dumper* dbg = nullptr, const char* dbg_tag = nullptr) const {
    const size_t C = channels, T = x.T;
    Tensor2D y(C, T), s(C, T), ff_h, ff_out;
    for (size_t li = 0; li < n_layers; ++li) {
      const Layer& L = layers[li];
      L.attn.forward(x, x, mask, y, buf_q_, buf_k_, buf_v_, buf_sc_);
      for (size_t i = 0; i < C * T; ++i) s.d[i] = x.d[i] + y.d[i];
      layernorm_ct(s, L.ln1_g, L.ln1_b, x);
      L.ffn.forward(x, ff_h, ff_out);
      for (size_t i = 0; i < C * T; ++i) s.d[i] = x.d[i] + ff_out.d[i];
      layernorm_ct(s, L.ln2_g, L.ln2_b, x);
      if (dbg != nullptr && dbg_tag != nullptr) {
        std::string name = std::string(dbg_tag) + "_L" +
                           std::to_string(li) + "_out";
        dbg->dump(name.c_str(), x);
      }
    }
  }

 private:
  // 线程安全不追求 — 引擎单线程驱动这些缓冲 (mutable 复用减少分配)
  mutable Tensor2D buf_q_, buf_k_, buf_v_, buf_sc_;
};

class MRTE {
 public:
  MultiHeadAttention cross_attention;  // hidden 512, 4 heads, 无 rel-pos
  Conv1d c_pre, text_pre, c_post;

  void load(const rt::GsvFile& f, std::string_view prefix) {
    std::string p(prefix);
    cross_attention.load(f, p + ".cross_attention", 512, 4, 0);
    // torch MRTE: c_pre=Conv1d(192,512), text_pre=同, c_post=Conv1d(512,192)
    c_pre.load(f, p + ".c_pre", 512, 192, 1);
    text_pre.load(f, p + ".text_pre", 512, 192, 1);
    c_post.load(f, p + ".c_post", 192, 512, 1);  // torch Conv1d(512,192): [out=192,in=512]
  }

  // ssl_enc[192,Ts], text[192,Tt], ge_text[512,gT] gT==1 时广播
  void forward(const Tensor2D& ssl_enc, const Tensor2D& text_enc,
               const Tensor2D& ge_text, Tensor2D& out,
               const Dumper* dbg = nullptr) const {
    const size_t Ts = ssl_enc.T, Tt = text_enc.T;
    Tensor2D c512, t512, attn_out;
    c_pre.forward(ssl_enc, c512);
    text_pre.forward(text_enc, t512);
    if (dbg) { dbg->dump("mte_ssl512", c512); dbg->dump("mte_text512", t512); }
    std::vector<uint8_t> mask(Ts * Tt, 1);  // attn_mask[Tq=Ts,Tk=Tt] 全 1
    Tensor2D qb, kb, vb, sc;
    cross_attention.forward(c512, t512, mask, attn_out, qb, kb, vb, sc);
    if (dbg) dbg->dump("mte_cross", attn_out);
    // x = attn + ssl_enc(c_pre 后) + ge_text (广播)
    out.reset(512, Ts);
    const size_t gT = ge_text.T;
    for (size_t c = 0; c < 512; ++c) {
      const float gv = ge_text.d[c * gT];
      for (size_t t = 0; t < Ts; ++t)
        out.d[c * Ts + t] =
            attn_out.d[c * Ts + t] + c512.d[c * Ts + t] + gv;
    }
    if (dbg) dbg->dump("mte_sum", out);
    c_post.forward(out, out);
  }
};

}  // namespace gsv::sovits
