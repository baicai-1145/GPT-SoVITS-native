// e2_enc_dump.cpp — E2-ENC 验收 harness: 加载 roberta/hubert/sv, 喂固定输入,
// dump 特征到目录供跨分支(main=fp32基线 vs task/E2-ENC=fp16) cos 对照。
//
// 用法: ./e2_enc_dump <weightsDir> <bertRefDir> <refWav16k> <outDir>
//   bertRefDir: 含 s{0,1,2}_{ids,tt,am}.bin (int64, 由 tokenizer 离线导出)
//   refWav16k:  16k wav (走 wav 解码; 引擎吃 float 波形)
// 输出:
//   bert_s{i}_feat.f32   [T,1024] roberta hidden[-3](即 AR 消费的 bert 特征源)
//   hub_last.f32         [T,768] HuBERT last_hidden_state (3s wav16k+0.3s 零)
//   hub_l0.f32           [T,768]
//   sv_emb.f32           [20480]
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "bert/bert_model.hpp"
#include "encoder/hubert.hpp"
#include "encoder/sv.hpp"
#include "encoder/fbank.hpp"
#include "runtime/gsv_loader.hpp"
#include "runtime/wav.hpp"

namespace fs = std::filesystem;
using namespace gsv;  // bert:: / rt:: / encoder:: 快捷

static std::vector<int64_t> read_i64(const std::string& p) {
  std::ifstream in(p, std::ios::binary);
  if (!in) { std::fprintf(stderr, "缺 %s\n", p.c_str()); std::exit(2); }
  const std::string b{std::istreambuf_iterator<char>(in), {}};
  std::vector<int64_t> v(b.size() / 8);
  std::memcpy(v.data(), b.data(), v.size() * 8);
  return v;
}
static void dump_f32(const std::string& p, const float* d, size_t n) {
  std::ofstream o(p, std::ios::binary);
  o.write(reinterpret_cast<const char*>(d), std::streamsize(n * 4));
}

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr, "usage: %s <weightsDir> <bertRefDir> <refWav16k> <outDir>\n",
                 argv[0]);
    return 2;
  }
  const std::string wdir = argv[1], bdir = argv[2], wavPath = argv[3],
                    outDir = argv[4];
  fs::create_directories(outDir);
  const bert::Dumper dm("");

  // ---- 1) RoBERTa (bert 特征: hidden[-3]) ----
  {
    rt::GsvFile f(wdir + "/roberta_wwm_ext_large.gsv");
    bert::BertModel bm;
    bm.cfg = bert::BertConfig{};  // roberta-large 默认
    bm.load(f, "bert");
    for (int i = 0; i < 3; ++i) {
      auto ids = read_i64(bdir + "/s" + std::to_string(i) + "_ids.bin");
      auto tt = read_i64(bdir + "/s" + std::to_string(i) + "_tt.bin");
      auto am = read_i64(bdir + "/s" + std::to_string(i) + "_am.bin");
      // featurize 口径: 前进到 layers-3 (hidden_states[-3]) —— 与 pipeline 一致
      const size_t T = ids.size(), C = bm.cfg.hidden;
      bert::Matrix x;
      x.reset(T, C);
      for (size_t t = 0; t < T; ++t)
        for (size_t c = 0; c < C; ++c)
          x.d[t * C + c] = bm.word_emb_w[size_t(ids[t]) * C + c] +
                           bm.pos_emb_w[t * C + c] +
                           bm.type_emb_w[size_t(tt[t]) * C + c];
      bm.emb_ln.forward(x);
      std::vector<float> ext(T);
      for (size_t j = 0; j < T; ++j)
        ext[j] = (1.f - float(am[j])) * bm.cfg.mask_neg;
      bert::Matrix y, scr, ctxh;
      std::vector<uint16_t> xh;
      const size_t stopAt = bm.cfg.layers - 3;
      for (size_t L = 0; L <= stopAt; ++L) {
        bm.stack[L].forward(x, ext, y, scr, ctxh, xh);
        x.d.swap(y.d);
      }
      dump_f32(outDir + "/bert_s" + std::to_string(i) + "_feat.f32", x.d.data(),
               x.d.size());
      std::printf("bert s%d: T=%zu dumped\n", i, T);
    }
  }  // f/bm 释放

  // ---- 2) HuBERT + SV (3s ref wav; 串行作用域) ----
  {
    rt::GsvFile f(wdir + "/hubert_base.gsv");
    encoder::HubertEngine hub(f);
    rt::wav::WavFile w = rt::wav::load_wav(wavPath);
    std::vector<float> wav16k = w.samples;  // 假定 16k 源
    // TTS 口径: 尾接 0.3s@32k 的零段历史口径 = 9600 个零
    wav16k.insert(wav16k.end(), 9600, 0.f);
    const size_t T = hub.run(wav16k.data(), wav16k.size());
    dump_f32(outDir + "/hub_last.f32", hub.out().data(), T * 768);
    dump_f32(outDir + "/hub_l0.f32", hub.l0_out().data(), T * 768);
    std::printf("hubert: T=%zu dumped\n", T);
  }
  {
    rt::GsvFile f(wdir + "/eres2netv2_sv.gsv");
    encoder::SvEngine sv(f);
    // fbank 输入: 用同 wav16k(未补零) 的前 3s → kaldi_fbank_80
    rt::wav::WavFile w = rt::wav::load_wav(wavPath);
    size_t frames = 0;
    const std::vector<float> fb =
        encoder::kaldi_fbank_80(w.samples.data(), w.samples.size(), &frames);
    sv.forward3(fb.data(), frames);
    dump_f32(outDir + "/sv_emb.f32", sv.emb_out().data(), sv.emb_out().size());
    std::printf("sv: emb=%zu dumped\n", sv.emb_out().size());
  }
  std::printf("ALL DUMPED -> %s\n", outDir.c_str());
  return 0;
}
