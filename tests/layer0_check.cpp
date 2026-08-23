// layer0_check.cpp — A3 出口条件: 用 A2+A3 手工跑通 AR 第一个 transformer 层
//
// 输入重建(对照 CPUFast infer_panel_naive, eval 模式 dropout=identity):
//   x_text = ar_text_embedding(phones) + bert_proj(bert_feat_1024)
//   x_text = x_text * 1 + alpha_text · PE_text        (SinePositionalEmbedding)
//   y_audio = ar_audio_embedding(prompt); 同加 alpha_audio·PE_audio
//   xy_pos = concat[x_text, y_audio]                  (文本在前, 共 193 行)
//
// 层0(T2SBlock.post-norm, eps=1e-5):
//   qkv = xy·Wqkvᵀ+b → 16头 SDPA(前缀全可见+音频段因果) → out proj
//   h = LN(x+attn); h = h + relu(h·W1ᵀ+b1)·W2ᵀ+b2; 输出 = LN(h)
#include "kern/accel.hpp"
#include "kern/kern.hpp"
#include "runtime/gsv_loader.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using gsv::kern::accel::DenseF16;
using gsv::rt::GsvFile;
using gsv::rt::TensorView;

namespace {

struct Args {
  const char* weights;
  const char* phones;
  const char* prompt;
  const char* bert_in;
  const char* out;
  const char* dump_dir = nullptr;  // 可选: 导出中间量供对拍
  int text_len = 0;
  int prompt_len = 0;
};

Args parse_args(int argc, char** argv) {
  Args a{};
  for (int i = 1; i < argc; ++i) {
    auto next = [&]() { return argv[++i]; };
    if (!std::strcmp(argv[i], "--weights")) a.weights = next();
    else if (!std::strcmp(argv[i], "--phones")) a.phones = next();
    else if (!std::strcmp(argv[i], "--prompt")) a.prompt = next();
    else if (!std::strcmp(argv[i], "--bert-in")) a.bert_in = next();
    else if (!std::strcmp(argv[i], "--out")) a.out = next();
    else if (!std::strcmp(argv[i], "--dump")) a.dump_dir = next();
    else if (!std::strcmp(argv[i], "--text-len")) a.text_len = std::atoi(next());
    else if (!std::strcmp(argv[i], "--prompt-len")) a.prompt_len = std::atoi(next());
  }
  return a;
}

std::vector<float> read_f32(const char* path, size_t expect_n) {
  FILE* f = std::fopen(path, "rb");
  if (!f) std::exit(2);
  std::vector<float> v(expect_n);
  if (std::fread(v.data(), 4, expect_n, f) != expect_n) std::exit(2);
  std::fclose(f);
  return v;
}

std::vector<int64_t> read_i64(const char* path, size_t expect_n) {
  FILE* f = std::fopen(path, "rb");
  if (!f) std::exit(2);
  std::vector<int64_t> v(expect_n);
  if (std::fread(v.data(), 8, expect_n, f) != expect_n) std::exit(2);
  std::fclose(f);
  return v;
}

// SinePositionalEmbedding 的 pe 行(fp32 计算, 与 torch 逐算子同构):
// div[i]=exp(2i·-ln(10000)/D); pe[2i]=sin(pos·div), pe[2i+1]=cos(pos·div)
void sine_pe_row(float* pe, int pos, size_t D) {
  const float log_inc = -std::log(10000.0f) / static_cast<float>(D);
  for (size_t i = 0; i < D; i += 2) {
    const float div = std::exp(static_cast<float>(i) * log_inc);
    pe[i] = std::sin(static_cast<float>(pos) * div);
    pe[i + 1] = std::cos(static_cast<float>(pos) * div);
  }
}

}  // namespace

int main(int argc, char** argv) {
  const Args a = parse_args(argc, argv);
  try {
    const GsvFile f(a.weights);
    auto need = [&](const char* name) -> const TensorView& {
      const auto* t = f.tensor(name);
      if (!t) {
        std::fprintf(stderr, "缺张量: %s\n", name);
        std::exit(3);
      }
      return *t;
    };

    const size_t D = 512, H = 16, HD = D / H, FF = 2048;
    const size_t T = static_cast<size_t>(a.text_len), L = static_cast<size_t>(a.prompt_len);
    const size_t S = T + L;

    // ---- 权重加载(全部经 DenseF16 升位缓存; 小张量直接读 fp32 段) ----
    const auto& text_emb_w = need("ar_text_embedding.word_embeddings.weight");  // [732,512]
    const auto& audio_emb_w = need("ar_audio_embedding.word_embeddings.weight");
    DenseF16 bert_proj(need("bert_proj.weight").data_f16_raw(), D, 1024);
    const auto& bert_b = need("bert_proj.bias");

    DenseF16 wqkv(need("h.layers.0.self_attn.in_proj_weight").data_f16_raw(), 3 * D, D);
    const auto& bqkv = need("h.layers.0.self_attn.in_proj_bias");
    DenseF16 wout(need("h.layers.0.self_attn.out_proj.weight").data_f16_raw(), D, D);
    const auto& bout = need("h.layers.0.self_attn.out_proj.bias");
    DenseF16 w1(need("h.layers.0.linear1.weight").data_f16_raw(), FF, D);
    const auto& b1 = need("h.layers.0.linear1.bias");
    DenseF16 w2(need("h.layers.0.linear2.weight").data_f16_raw(), D, FF);
    const auto& b2 = need("h.layers.0.linear2.bias");
    const auto& n1g = need("h.layers.0.norm1.weight");
    const auto& n1b = need("h.layers.0.norm1.bias");
    const auto& n2g = need("h.layers.0.norm2.weight");
    const auto& n2b = need("h.layers.0.norm2.bias");

    const float alpha_text = need("ar_text_position.alpha").data_f32()[0];
    const float alpha_audio = need("ar_audio_position.alpha").data_f32()[0];

    // ---- 输入 ----
    const auto phones = read_i64(a.phones, T);
    const auto prompt = read_i64(a.prompt, L);
    const auto bert_in = read_f32(a.bert_in, T * 1024);

    // ---- 输入重建 ----
    std::vector<float> xy(S * D, 0.f), tmp(T * D);
    for (size_t t = 0; t < T; ++t)  // 文本 embedding 行查表(fp16→fp32 无损)
      for (size_t d = 0; d < D; ++d) {
        __fp16 h;
        __builtin_memcpy(&h, &text_emb_w.data_f16_raw()[static_cast<size_t>(phones[t]) * D + d], 2);
        xy[t * D + d] = static_cast<float>(h);
      }
    bert_proj.forward(bert_in.data(), T, tmp.data());  // [T,512] = bert_in·Wᵀ
    for (size_t i = 0; i < T * D; ++i) xy[i] += tmp[i] + bert_b.data_f32()[i % D];
    {  // bert_proj 复算 vs golden bert_feat 一致性诊断由 python 侧对 layers_prefill 判断,
       // 这里顺带把 bert_feat golden 若存在则打印 cos —— 保持简单: 略。
    }
    for (size_t t = 0; t < L; ++t)
      for (size_t d = 0; d < D; ++d) {
        __fp16 h;
        __builtin_memcpy(&h, &audio_emb_w.data_f16_raw()[static_cast<size_t>(prompt[t]) * D + d], 2);
        xy[(T + t) * D + d] = static_cast<float>(h);
      }
    {  // 正弦位置编码: x + alpha·pe；文本/音频是两个独立 PE 模块, 各自从 0 计位置
      std::vector<float> pe(D);
      for (size_t t = 0; t < S; ++t) {
        const int pos = static_cast<int>(t < T ? t : t - T);
        sine_pe_row(pe.data(), pos, D);
        const float alpha = t < T ? alpha_text : alpha_audio;
        for (size_t d = 0; d < D; ++d) xy[t * D + d] += alpha * pe[d];
      }
    }

    // ---- 层0 ----
    std::vector<float> qkv(S * 3 * D);
    wqkv.forward(xy.data(), S, qkv.data());  // [S,1536]
    for (size_t i = 0; i < S; ++i)
      for (size_t j = 0; j < 3 * D; ++j) qkv[i * 3 * D + j] += bqkv.data_f32()[j];

    const float scale = 1.0f / std::sqrt(static_cast<float>(HD));
    std::vector<float> attn_out(S * D, 0.f);
    std::vector<float> scores(S), probs(S);
    for (size_t h = 0; h < H; ++h) {
      for (size_t q = 0; q < S; ++q) {
        const float* qv = &qkv[q * 3 * D + h * HD];               // q 段
        size_t n_allowed = 0;
        for (size_t s = 0; s < S; ++s) {
          const bool allowed = (s < T) || (s <= q);  // 前缀全可见 + 音频因果
          if (!allowed) continue;
          const float* kv = &qkv[s * 3 * D + D + h * HD];          // k 段
          float dot = 0.f;
          for (size_t e = 0; e < HD; ++e) dot += qv[e] * kv[e];
          scores[n_allowed++] = dot * scale;
        }
        gsv::kern::softmax(scores.data(), probs.data(), n_allowed);
        float* ov = &attn_out[q * D + h * HD];
        size_t idx = 0;
        for (size_t s = 0; s < S; ++s) {
          if (!(s < T || s <= q)) continue;
          const float p = probs[idx++];
          const float* vv = &qkv[s * 3 * D + 2 * D + h * HD];      // v 段
          for (size_t e = 0; e < HD; ++e) ov[e] += p * vv[e];
        }
      }
    }

    std::vector<float> proj(S * D);
    wout.forward(attn_out.data(), S, proj.data());
    for (size_t i = 0; i < S * D; ++i) proj[i] += bout.data_f32()[i % D];
    std::vector<float> h1(S * D);
    for (size_t i = 0; i < S * D; ++i) h1[i] = xy[i] + proj[i];

    auto dump = [&](const char* nm, const std::vector<float>& v) {
      if (!a.dump_dir) return;
      const std::string path = std::string(a.dump_dir) + "/" + nm;
      FILE* f = std::fopen(path.c_str(), "wb");
      if (f) {
        std::fwrite(v.data(), 4, v.size(), f);
        std::fclose(f);
      }
    };
    dump("xy.f32", xy);
    dump("qkv.f32", qkv);
    dump("proj.f32", proj);
    dump("attn_res.f32", h1);

    std::vector<float> hn(S * D), ff(S * FF), mlp(S * D), out(S * D);
    for (size_t t = 0; t < S; ++t)
      gsv::kern::layernorm(&h1[t * D], n1g.data_f32(), n1b.data_f32(), &hn[t * D], D, 1e-5);
    w1.forward(hn.data(), S, ff.data());
    for (size_t i = 0; i < S * FF; ++i) ff[i] += b1.data_f32()[i % FF];
    gsv::kern::relu(ff.data(), ff.data(), S * FF);
    w2.forward(ff.data(), S, mlp.data());
    for (size_t i = 0; i < S * D; ++i) mlp[i] += b2.data_f32()[i % D];
    for (size_t i = 0; i < S * D; ++i) h1[i] = hn[i] + mlp[i];
    for (size_t t = 0; t < S; ++t)
      gsv::kern::layernorm(&h1[t * D], n2g.data_f32(), n2b.data_f32(), &out[t * D], D, 1e-5);
    dump("hn.f32", hn);
    dump("mlp_res.f32", h1);

    FILE* fo = std::fopen(a.out, "wb");
    if (!fo) return 2;
    std::fwrite(out.data(), 4, out.size(), fo);
    std::fclose(fo);
    std::printf("[layer0] S=%zu D=%zu 输出已写 %s\n", S, D, a.out);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "错误: %s\n", e.what());
    return 1;
  }
}
