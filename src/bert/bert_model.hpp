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
  // Embedding 查表(非 matmul): fp32 常驻(word 21128×1024=86MB; pos/type 极小)
  std::vector<float> word_emb_w, pos_emb_w, type_emb_w;
  LayerNorm1D emb_ln;
  std::vector<BertLayer> stack;

  void load(const rt::GsvFile& f, const std::string& pfx = "bert") {
    const size_t C = cfg.hidden;
    load_tensor_f32(f, pfx + ".embeddings.word_embeddings.weight", word_emb_w,
                    {cfg.vocab, C});
    load_tensor_f32(f, pfx + ".embeddings.position_embeddings.weight", pos_emb_w,
                    {cfg.max_pos, C});
    load_tensor_f32(f, pfx + ".embeddings.token_type_embeddings.weight", type_emb_w,
                    {cfg.type_vocab, C});
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
        x.d[t * C + c] = word_emb_w[size_t(ids[t]) * C + c] +
                         pos_emb_w[t * C + c] +
                         type_emb_w[size_t(tt[t]) * C + c];
    emb_ln.forward(x);
    dm.dump("bert_emb_out", x);

    Matrix y, scr, ctxh;
    std::vector<uint16_t> xh;  // fp16 激活暂存(跨层复用)
    // E8: per-layer timing 重置
    if (bert_layer_timing_on()) {
      auto& lt = last_layer_timing();
      lt.qkv = lt.head = lt.wout_ln = lt.ffn = 0;
    }
    for (size_t i = 0; i < cfg.layers; ++i) {
      stack[i].forward(x, ext, y, scr, ctxh, xh);
      x.d.swap(y.d);
      if (i == 0) dm.dump("bert_layer0_out", x);
      if (i == cfg.layers - 3) dm.dump("bert_layer_m3_out", x);  // hidden_states[-3]
    }
    if (bert_layer_timing_on()) {
      const auto& lt = last_layer_timing();
      std::fprintf(stderr,
                   "[bert-layer-timing] layers=%zu qkv=%.1f head=%.1f "
                   "wout+ffn=%.1f (ms)\n",
                   cfg.layers, lt.qkv, lt.head, lt.ffn);
    }
    dm.dump("bert_last_out", x);  // 末层 LN 输出 = last_hidden_state
    out.d.swap(x.d);
    out.rows = T;
    out.cols = C;
  }
};

}  // namespace gsv::bert
