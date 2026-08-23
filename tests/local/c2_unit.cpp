// c2_unit.cpp — C2 本地单测 (验收后由决策者决定是否入库 tests/):
//   1. BertTokenizer 对照 B8 fixtures 的 input_ids (3 句)
//   2. ConditionBuilder spectrogram + 完整条件链 对照 export_c2_fixtures.py
//   3. Resampler 自检 (恒等比率 + 正弦频率保持)
//
// 用法: ./c2_unit <bert_fixture_dir> <cond_fixture_dir> <sovits.gsv> <vocab.txt>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "runtime/pipeline_audio.hpp"
#include "runtime/pipeline_condition.hpp"
#include "runtime/pipeline_tokenizer.hpp"
#include "runtime/gsv_loader.hpp"

namespace {

std::vector<float> loadF32(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) return {};
  f.seekg(0, std::ios::end);
  const size_t n = size_t(f.tellg()) / 4;
  f.seekg(0);
  std::vector<float> v(n);
  f.read(reinterpret_cast<char*>(v.data()), std::streamsize(n * 4));
  return v;
}

std::vector<long long> loadI64(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) return {};
  f.seekg(0, std::ios::end);
  const size_t n = size_t(f.tellg()) / 8;
  f.seekg(0);
  std::vector<long long> v(n);
  f.read(reinterpret_cast<char*>(v.data()), std::streamsize(n * 8));
  return v;
}

int fails = 0;
void check(bool ok, const std::string& what) {
  std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
  if (!ok) ++fails;
}

double cosSim(const std::vector<float>& a, const std::vector<float>& b) {
  double dot = 0, na = 0, nb = 0;
  const size_t n = a.size() < b.size() ? a.size() : b.size();
  for (size_t i = 0; i < n; ++i) {
    dot += double(a[i]) * double(b[i]);
    na += double(a[i]) * double(a[i]);
    nb += double(b[i]) * double(b[i]);
  }
  return dot / (std::sqrt(na) * std::sqrt(nb));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr,
                 "usage: %s <bert_fixture_dir> <cond_fixture_dir> "
                 "<sovits.gsv> <vocab.txt>\n",
                 argv[0]);
    return 2;
  }
  const std::string bertDir = argv[1], condDir = argv[2];

  // ---------- 1. tokenizer vs B8 input_ids ----------
  {
    gsv::rt::pipeline::BertTokenizer tok;
    std::string err;
    check(tok.load(argv[4], &err), "tokenizer.load");
    for (const char* tag : {"hello_world", "hotpot", "bank_river"}) {
      const auto golden = loadI64(bertDir + "/" + tag +
                                  "/inputs/input_ids.bin");
      std::ifstream mf(bertDir + "/" + tag + "/meta.json");
      const std::string meta((std::istreambuf_iterator<char>(mf)),
                             std::istreambuf_iterator<char>());
      const size_t p = meta.find("\"sentence\": \"") + 13;
      const size_t e = meta.find('"', p);
      const std::string sent = meta.substr(p, e - p);
      std::vector<int64_t> ids, tt, mask;
      tok.encode(sent, &ids, &tt, &mask);
      bool same = ids.size() == golden.size();
      for (size_t i = 0; same && i < ids.size(); ++i)
        same = ids[i] == golden[i];
      char buf[160];
      std::snprintf(buf, sizeof(buf), "tokenizer[%s] len=%zu golden=%zu %s",
                    tag, ids.size(), golden.size(), sent.c_str());
      check(same, buf);
    }
  }

  // ---------- 3. Resampler 自检 ----------
  {
    using gsv::rt::pipeline::Resampler;
    // 恒等: 同采样率直通
    std::vector<float> x(1000);
    for (size_t i = 0; i < x.size(); ++i) x[i] = float(i) * 0.001f;
    Resampler rid;
    rid.init(32000, 32000);
    std::vector<float> y;
    rid.process(x.data(), x.size(), y);
    double mx = 0;
    for (size_t i = 10; i + 10 < y.size(); ++i)
      mx = std::max(mx, double(std::abs(y[i] - x[i])));
    char buf[96];
    std::snprintf(buf, sizeof(buf), "resampler 恒等 maxerr=%.2e", mx);
    check(mx < 1e-5, buf);

    // 48k→16k 正弦 220Hz 频率保持 (过零计数近似)
    std::vector<float> s(48000);
    for (size_t i = 0; i < s.size(); ++i)
      s[i] = std::sin(2.0 * M_PI * 220.0 * double(i) / 48000.0);
    Resampler r48;
    r48.init(48000, 16000);
    std::vector<float> d;
    r48.process(s.data(), s.size(), d);
    int zc = 0;
    for (size_t i = 1; i < d.size(); ++i)
      if (d[i - 1] < 0 && d[i] >= 0) ++zc;
    // 220Hz × 1s → 220 个正过零
    std::snprintf(buf, sizeof(buf), "resampler 48k→16k 过零=%d (期望≈220)",
                  zc);
    check(std::abs(zc - 220) <= 3, buf);
  }

  // ---------- 2. 条件链 vs torch fixture ----------
  {
    using namespace gsv::rt::pipeline;
    gsv::rt::GsvFile sovF(argv[3]);
    ConditionBuilder cb;
    cb.load(sovF);

    const std::vector<float> audio32k = loadF32(condDir + "/audio32k.bin");
    const std::vector<float> specG = loadF32(condDir + "/spec.bin");
    const std::vector<float> geG = loadF32(condDir + "/ge.bin");
    const std::vector<float> geTG = loadF32(condDir + "/ge_text.bin");
    const std::vector<float> wG = loadF32(condDir + "/ref_enc_out.bin");
    const std::vector<float> svEmb = loadF32(condDir + "/sv_emb.bin");

    std::vector<float> specO;
    size_t frames = 0;
    ConditionBuilder::spectrogram(audio32k.data(), audio32k.size(), specO,
                                  frames);
    check(specG.size() == specO.size(), "spec 尺寸一致");
    double maxAbs = 0, relSum = 0, refSum = 0;
    for (size_t i = 0; i < specO.size() && i < specG.size(); ++i) {
      maxAbs = std::max(maxAbs,
                        std::abs(double(specO[i]) - double(specG[i])));
      relSum += std::abs(double(specO[i]) - double(specG[i]));
      refSum += std::abs(double(specG[i]));
    }
    char buf[192];
    std::snprintf(buf, sizeof(buf), "spec maxabs=%.3e rel=%.3e", maxAbs,
                  relSum / refSum);
    check(maxAbs < 2e-4, buf);

    if (svEmb.empty()) {
      std::printf("[SKIP] 条件链对拍缺 sv_emb.bin (exporter 补写)\n");
    } else {
      DecodeCondition cond;
      cb.compute(audio32k.data(), audio32k.size(), svEmb.data(), &cond);
      const double cosGe = cosSim(cond.ge, geG);
      const double cosGT = cosSim(cond.ge_text, geTG);
      std::snprintf(buf, sizeof(buf), "ge cos=%.7f ge_text cos=%.7f", cosGe,
                    cosGT);
      check(cosGe > 0.9999 && cosGT > 0.9999, buf);

      if (!wG.empty()) {
        // ref_enc 主干单独对拍 (sv 相加前): compute 内部不可拆 → 用
        // refEnc 接口直接喂 fixture 的 [T][704]
        std::vector<float> condIn(frames * 704);
        for (size_t t = 0; t < frames; ++t)
          for (size_t b = 0; b < 704; ++b)
            condIn[t * 704 + b] = specO[b * frames + t];
        std::vector<float> pooled;
        cb.refEnc(condIn.data(), frames, pooled);
        const double cosW = cosSim(pooled, wG);
        std::snprintf(buf, sizeof(buf), "ref_enc 主干 cos=%.7f", cosW);
        check(cosW > 0.9999, buf);
      }
    }
  }

  std::printf("== done, %d fail(s) ==\n", fails);
  return fails ? 1 : 0;
}
