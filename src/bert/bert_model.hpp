// bert_model.hpp — B8: 完整 BertModel (embeddings + N 层编码栈)
// 命名与 .gsv 一致: bert.embeddings.* / bert.encoder.layer.{i}.*
#pragma once
#include <string>
#include <vector>

#include "bert/bert_ops.hpp"
#include "bert/bert_io.hpp"

namespace gsv::bert {

struct BertConfig {
  size_t vocab = 21128, max_pos = 512, type_vocab = 2;
  size_t hidden = 1024, heads = 16, layers = 24, inter = 4096;
  float ln_eps = 1e-12f;      // roberta-large: 1e-12; g2pw: 1e-5
  float mask_neg = kFinfoMin; // roberta: finfo.min; g2pw: -10000.f
};

struct BertModel {
  BertConfig cfg;
  Linear word_emb, pos_emb, type_emb;  // 仅用 w (查表)
  LayerNorm1D emb_ln;
  std::vector<BertLayer> stack;

  void load(const rt::GsvFile& f, const std::string& pfx = "bert") {
    const size_t C = cfg.hidden;
    word_emb.load(f, pfx + ".embeddings.word_embeddings", cfg.vocab, C, false);
    pos_emb.load(f, pfx + ".embeddings.position_embeddings", cfg.max_pos, C,
                 false);
    type_emb.load(f, pfx + ".embeddings.token_type_embeddings",
                  cfg.type_vocab, C, false);
    emb_ln.load(f, pfx + ".embeddings.LayerNorm", C, cfg.ln_eps);
    stack.resize(cfg.layers);
    for (size_t i = 0; i < cfg.layers; ++i)
      stack[i].load(f, pfx + ".encoder.layer." + std::to_string(i), C,
                    cfg.heads, cfg.inter, cfg.ln_eps);
  }

  // ids/tt/amask 等长 T; out = last_hidden_state
  void forward(const std::vector<int64_t>& ids, const std::vector<int64_t>& tt,
               const std::vector<int64_t>& amask, Matrix& out,
               const Dumper& dm) const {
    const size_t T = ids.size(), C = cfg.hidden;
    // extended mask: (1-amask[j])*neg
    std::vector<float> ext(T);
    for (size_t j = 0; j < T; ++j) ext[j] = (1.f - float(amask[j])) * cfg.mask_neg;

    Matrix x;
    x.reset(T, C);
    for (size_t t = 0; t < T; ++t)
      for (size_t c = 0; c < C; ++c)
        x.d[t * C + c] = word_emb.w[size_t(ids[t]) * C + c] +
                         pos_emb.w[t * C + c] +
                         type_emb.w[size_t(tt[t]) * C + c];
    emb_ln.forward(x);
    dm.dump("bert_emb_out", x);

    Matrix y, scr, ctxh;
    for (size_t i = 0; i < cfg.layers; ++i) {
      stack[i].forward(x, ext, y, scr, ctxh);
      x.d.swap(y.d);
      if (i == 0) dm.dump("bert_layer0_out", x);
      if (i == cfg.layers - 3) dm.dump("bert_layer_m3_out", x);  // hidden_states[-3]
    }
    dm.dump("bert_last_out", x);  // 末层 LN 输出 = last_hidden_state
    out.d.swap(x.d);
    out.rows = T;
    out.cols = C;
  }
};

}  // namespace gsv::bert
