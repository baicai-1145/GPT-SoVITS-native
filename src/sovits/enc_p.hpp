// enc_p.hpp — TextEncoder 顶层组装 (B3)
// forward 输入: quantized[768,Tq](已 ×2 上采样), phones[Tt], ge_text[512,gT]
// 输出: m_p/logs_p 各 [192,Tq]
#pragma once

#include <string>
#include <vector>

#include "sovits/enc_p_parts.hpp"

namespace gsv::sovits {

class TextEncoder {
 public:
  Conv1d ssl_proj;      // 768 → 192 (k1)
  Encoder encoder_ssl;     // 192, heads2, 3 层
  Encoder encoder_text;    // 192, heads2, 6 层
  Encoder encoder2;        // 192, heads2, 3 层
  std::vector<float> text_embedding;  // [732, 192]
  size_t n_symbols = 0;
  MRTE mrte;
  Conv1d proj;          // 192 → 384 (k1)

  void load(const rt::GsvFile& f) {
    ssl_proj.load(f, "enc_p.ssl_proj", 192, 768, 1);
    encoder_ssl.load(f, "enc_p.encoder_ssl", /*ch*/ 192, /*heads*/ 2,
                     /*nl*/ 3, /*filt*/ 768, /*k*/ 3, /*win*/ 9);
    encoder_text.load(f, "enc_p.encoder_text", 192, 2, 6, 768, 3, 9);
    encoder2.load(f, "enc_p.encoder2", 192, 2, 3, 768, 3, 9);

    const auto* te = f.tensor("enc_p.text_embedding.weight");
    if (!te) throw std::runtime_error("missing text_embedding");
    n_symbols = te->dims[0];
    load_tensor_f32(f, "enc_p.text_embedding.weight", text_embedding);

    mrte.load(f, "enc_p.mrte");
    proj.load(f, "enc_p.proj", 384, 192, 1);
  }

  // quantized[768,Tq]; phones int64[Tt]; 输出 stats 写入 m_p/logs_p [192,Tq]
  void forward(const Tensor2D& quantized, const int64_t* phones, size_t Tt,
               const Tensor2D& ge_text, Tensor2D& m_p, Tensor2D& logs_p,
               const Dumper& dm) const {
    const size_t Tq = quantized.T;
    std::vector<uint8_t> mask_q(Tq * Tq, 1), mask_t(Tt * Tt, 1);

    // y = ssl_proj(q)
    Tensor2D y;
    ssl_proj.forward(quantized, y);
    dm.dump("dbg_ssl_proj_out", y);

    // encoder_ssl
    encoder_ssl.forward(y, mask_q, &dm, "dbg_es");
    dm.dump("dbg_encoder_ssl_out", y);

    // text embedding → [192, Tt]
    Tensor2D temb(192, Tt);
    for (size_t t = 0; t < Tt; ++t) {
      const int64_t id = phones[t];
      if (id < 0 || static_cast<size_t>(id) >= n_symbols)
        throw std::runtime_error("phone id out of range");
      const float* e = &text_embedding[static_cast<size_t>(id) * 192];
      for (size_t c = 0; c < 192; ++c) temb.d[c * Tt + t] = e[c];
    }
    dm.dump("dbg_text_emb_t", temb);

    encoder_text.forward(temb, mask_t);
    dm.dump("dbg_encoder_text_out", temb);
    dm.dump("dbg_encoder_text_out", temb);

    // MRTE(y, text, ge_text)
    Tensor2D m;
    mrte.forward(y, temb, ge_text, m);
    dm.dump("dbg_mrte_out", m);

    // encoder2 → proj
    encoder2.forward(m, mask_q);
    dm.dump("dbg_encoder2_out", m);
    dm.dump("dbg_encp_x", m);
    Tensor2D stats;
    proj.forward(m, stats);
    dm.dump("h_encp_proj", stats);

    // split
    m_p.reset(192, Tq);
    logs_p.reset(192, Tq);
    for (size_t c = 0; c < 192; ++c)
      for (size_t t = 0; t < Tq; ++t) {
        m_p.d[c * Tq + t] = stats.d[c * Tq + t];
        logs_p.d[c * Tq + t] = stats.d[(192 + c) * Tq + t];
      }
  }
};

}  // namespace gsv::sovits
