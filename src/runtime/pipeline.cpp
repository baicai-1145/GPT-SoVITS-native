// pipeline.cpp — C2: 全链路编排实现 (口径见 pipeline.hpp)
#include "runtime/pipeline.hpp"

#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <filesystem>
#include <initializer_list>
#include <random>

#include "runtime/threadpool.hpp"
#include "runtime/wav.hpp"
#include "sovits/wav_writer.hpp"

namespace gsv::rt::pipeline {

namespace {

std::string joinPath(const std::string& a, const std::string& b) {
  return (std::filesystem::path(a) / b).string();
}

double nowMs() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// UTF-8 → codepoint 数
size_t cpCount(const std::string& s) {
  size_t n = 0;
  for (size_t i = 0; i < s.size();) {
    unsigned char b = static_cast<unsigned char>(s[i]);
    i += b >= 0xF0 ? 4 : b >= 0xE0 ? 3 : b >= 0xC0 ? 2 : 1;
    ++n;
  }
  return n;
}

// UTF-8 ↔ U32 (textfront 内部 helper 未导出, 本地实现同语义)
std::u32string utf8ToU32(const std::string& s) {
  std::u32string o;
  for (size_t i = 0; i < s.size();) {
    unsigned char b = static_cast<unsigned char>(s[i]);
    if (b < 0x80) {
      o.push_back(b);
      ++i;
      continue;
    }
    uint32_t cp = 0;
    size_t extra = 0;
    if ((b & 0xE0) == 0xC0) {
      cp = b & 0x1F;
      extra = 1;
    } else if ((b & 0xF0) == 0xE0) {
      cp = b & 0x0F;
      extra = 2;
    } else {
      cp = b & 0x07;
      extra = 3;
    }
    for (size_t k = 1; k <= extra && i + k < s.size(); ++k)
      cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3F);
    i += extra + 1;
    o.push_back(cp);
  }
  return o;
}

std::string u32ToUtf8(const std::u32string& u) {
  std::string o;
  o.reserve(u.size() * 3);
  char buf[4];
  for (char32_t c : u) {
    int n = 0;
    if (c < 0x80) {
      buf[n++] = char(c);
    } else if (c < 0x800) {
      buf[n++] = char(0xC0 | (c >> 6));
      buf[n++] = char(0x80 | (c & 0x3F));
    } else if (c < 0x10000) {
      buf[n++] = char(0xE0 | (c >> 12));
      buf[n++] = char(0x80 | ((c >> 6) & 0x3F));
      buf[n++] = char(0x80 | (c & 0x3F));
    } else {
      buf[n++] = char(0xF0 | (c >> 18));
      buf[n++] = char(0x80 | ((c >> 12) & 0x3F));
      buf[n++] = char(0x80 | ((c >> 6) & 0x3F));
      buf[n++] = char(0x80 | (c & 0x3F));
    }
    o.append(buf, size_t(n));
  }
  return o;
}

// 峰值归一 (_get_ref_spec 口径): maxx>1 时 audio /= min(2, maxx)
void normalizeRef(std::vector<float>* audio) {
  double mx = 0.0;
  for (float v : *audio) mx = std::max(mx, double(std::abs(v)));
  if (mx > 1.0) {
    const float div = float(std::min(2.0, mx));
    for (float& v : *audio) v /= div;
  }
}

}  // namespace

bool Pipeline::load(const std::string& weightsDir, const std::string& dataDir,
                    const PipelineOptions& opt, std::string* err) {
  opt_ = opt;
  weightsDir_ = weightsDir;
  // 文本前端数据优先在 dataDir 直取; 兼容仓库默认布局 (jieba/pinyin 在
  // src/textfront/data/, 词表在 src/runtime/data/)。
  auto firstExisting = [](std::initializer_list<std::string> cands,
                          std::string* out) {
    for (const auto& c : cands)
      if (std::filesystem::exists(c)) {
        *out = c;
        return true;
      }
    return false;
  };
  std::string trieP, pinyinP;
  if (!firstExisting({joinPath(dataDir, "jieba_trie.bin"),
                      joinPath(dataDir + "/../..", "textfront/data/jieba_trie.bin")},
                     &trieP)) {
    if (err) *err = "找不到 jieba_trie.bin (于 " + dataDir + ")";
    return false;
  }
  if (!firstExisting({joinPath(dataDir, "pinyin.bin"),
                      joinPath(dataDir + "/../..", "textfront/data/pinyin.bin")},
                     &pinyinP)) {
    if (err) *err = "找不到 pinyin.bin (于 " + dataDir + ")";
    return false;
  }
  try {
    if (!tf_.load(trieP, pinyinP, err)) return false;
    if (!tok_.load(joinPath(dataDir, "roberta_vocab.txt"), err)) return false;

    bert_.cfg = bert::BertConfig{};  // roberta-wwm-ext-large 默认即此
    bert_.load(rt::GsvFile(joinPath(weightsDir, "roberta_wwm_ext_large.gsv")),
               "bert");

    ar_ = std::make_unique<ar::T2SEngine>(
        rt::GsvFile(joinPath(weightsDir, "ar_s1v3.gsv")));
    sovits_ = std::make_unique<sovits::SovitsEngine>();
    sovits_->load(joinPath(weightsDir, "sovits_v2ProPlus.gsv"));
    {
      rt::GsvFile sovF(joinPath(weightsDir, "sovits_v2ProPlus.gsv"));
      cond_.load(sovF);
      hubert_ = std::make_unique<encoder::HubertEngine>(
          rt::GsvFile(joinPath(weightsDir, "hubert_base.gsv")));
      sv_ = std::make_unique<encoder::SvEngine>(
          rt::GsvFile(joinPath(weightsDir, "eres2netv2_sv.gsv")));
    }
  } catch (const std::exception& e) {
    if (err) *err = e.what();
    return false;
  }
  return true;
}

bool Pipeline::buildReference(const std::string& refWavPath, SynthResult* out,
                              std::string* err) {
  // ---- 磁盘缓存命中: 直接拿 prompt_semantic + ge + ge_text ----
  if (opt_.use_ref_cache) {
    // 缓存条目复用 RefEntry: hubert 字段改存 prompt_semantic(以 f32 载体),
    // sv 字段改存 [ge(1024)|ge_text(512)]。tag 区分格式。
    encoder::RefEntry e;
    if (refCache_.load(refWavPath, e)) {
      out->prompt_semantic.resize(e.hubert.size());
      for (size_t i = 0; i < e.hubert.size(); ++i)
        out->prompt_semantic[i] = int64_t(e.hubert[i]);
      out->cond.ge.assign(e.sv.begin(), e.sv.begin() + 1024);
      out->cond.ge_text.assign(e.sv.begin() + 1024, e.sv.end());
      out->ref_from_cache = true;
      return true;
    }
  }

  try {
    // ---- 解码源文件 (mono float @原采样率) ----
    rt::wav::WavFile w = rt::wav::load_wav(refWavPath);
    if (w.samples.empty()) throw std::runtime_error("空音频: " + refWavPath);

    Resampler to32k;
    to32k.init(int(w.sample_rate), kSr);
    std::vector<float> a32;
    if (int(w.sample_rate) == int(kSr))
      a32 = w.samples;
    else
      to32k.process(w.samples.data(), w.samples.size(), a32);
    normalizeRef(&a32);

    // sv_emb 输入 = 归一后 32k → 16k
    Resampler to16;
    to16.init(kSr, 16000);
    std::vector<float> a16cond;
    to16.process(a32.data(), a32.size(), a16cond);
    std::vector<float> fbank =
        encoder::kaldi_fbank_80(a16cond.data(), a16cond.size(), nullptr);
    size_t frames80 = fbank.size() / 80;
    sv_->forward3(fbank.data(), frames80);
    std::vector<float> svEmb(sv_->emb_out());  // [20480]

    // prompt_semantic: 原始音频直接 → 16k (+9600 零) → HuBERT → ssl_proj → RVQ
    Resampler to16raw;
    to16raw.init(int(w.sample_rate), 16000);
    std::vector<float> wav16k;
    if (int(w.sample_rate) == 16000)
      wav16k = w.samples;
    else
      to16raw.process(w.samples.data(), w.samples.size(), wav16k);
    if (wav16k.size() < 48000 || wav16k.size() > 160000)
      throw std::runtime_error(
          "参考音频长度须为 3~10 秒 (当前约 " +
          std::to_string(double(wav16k.size()) / 16000.0) + " 秒): " +
          refWavPath);

    // 条件链 (spec → ref_enc → ge/ge_text)
    cond_.compute(a32.data(), a32.size(), svEmb.data(), &out->cond);

    // HuBERT: 零尾填充与 CPUFast 一致 (cat[wav16k, zeros(9600)])
    std::vector<float> hubIn(wav16k);
    hubIn.insert(hubIn.end(), kSilence, 0.f);  // 0.3s@32k 的零段, CPUFast 同款
    const size_t T = hubert_->run(hubIn.data(), hubIn.size());
    const std::vector<float>& hidden = hubert_->out();  // [T,768]

    // extract_latent: v2ProPlus semantic_frame_rate="25hz" ⇒
    //   ssl_proj = Conv1d(768,768,k=2,stride=2) 帧率减半, 再最近码字。
    rt::GsvFile sovF(joinPath(weightsDir_, "sovits_v2ProPlus.gsv"));
    std::vector<float> sslW, sslB;  // W[768,768,2](out,in,tap), b[768]
    sovits::load_tensor_f32(sovF, "ssl_proj.weight", sslW);
    sovits::load_tensor_f32(sovF, "ssl_proj.bias", sslB);
    const size_t Tq = (T >= 2) ? ((T - 2) / 2 + 1) : 0;  // floor((L-k)/s)+1
    std::vector<float> proj(Tq * 768);
    {  // im2col: 相邻两输入帧拼 [Tq,1536]; 权重按 tap 展平 [768,1536]
      std::vector<float> w2(size_t(768) * 1536);
      for (size_t c = 0; c < 768; ++c)
        for (size_t d = 0; d < 768; ++d) {
          w2[c * 1536 + d] = sslW[(c * 768 + d) * 2 + 0];
          w2[c * 1536 + d + 768] = sslW[(c * 768 + d) * 2 + 1];
        }
      kern::accel::sgemm('N', 'T', int(Tq), 768, 1536, 1.0f, hidden.data(),
                         1536, w2.data(), 1536, 0.0f, proj.data(), 768);
      for (size_t t = 0; t < Tq; ++t)
        for (size_t c = 0; c < 768; ++c) proj[t * 768 + c] += sslB[c];
    }
    std::vector<float> embed;  // [1024,768]
    sovits::load_tensor_f32(sovF, "quantizer.vq.layers.0._codebook.embed",
                            embed, {1024, 768});
    out->prompt_semantic.resize(Tq);
    for (size_t t = 0; t < Tq; ++t) {
      int64_t best = 0;
      double bd = 1e30;
      for (int64_t cb = 0; cb < 1024; ++cb) {
        const float* e = embed.data() + size_t(cb) * 768;
        double d = 0.0;
        for (size_t c = 0; c < 768; ++c) {
          const double diff = double(proj[t * 768 + c]) - double(e[c]);
          d += diff * diff;
        }
        if (d < bd) {
          bd = d;
          best = cb;
        }
      }
      out->prompt_semantic[t] = best;
    }

    if (opt_.use_ref_cache) {
      encoder::RefEntry e;
      e.hubert.reserve(out->prompt_semantic.size());
      for (int64_t v : out->prompt_semantic) e.hubert.push_back(float(v));
      e.sv = out->cond.ge;
      e.sv.insert(e.sv.end(), out->cond.ge_text.begin(),
                  out->cond.ge_text.end());
      refCache_.store(refWavPath, e);
    }
  } catch (const std::exception& e) {
    if (err) *err = e.what();
    return false;
  }
  return true;
}

namespace {

// 文本 → (phones, word2ph, normU8, bert 特征 [n,1024])。
// 口径: TextFrontend 单段处理(cut0) + textNormalize 作 BERT 输入 +
//       roberta hidden_states[-3][1:-1] 按字符索引行 × word2ph 重复
//       (= chinese_bert.get_bert_feature, 含 ASCII 词时的静默对齐语义)。
struct Feat {
  std::vector<int64_t> phones;
  std::vector<int> word2ph;
  std::string normU8;
  std::vector<float> bert;  // [phones.size()*1024]
};

Feat featurize(textfront::TextFrontend* tf, textfront::ChineseG2p* g2pNorm,
               const BertTokenizer* tok, const bert::BertModel& bm,
               const std::string& utf8, std::string* /*err*/) {
  textfront::TextFrontend::Result one;
  if (!tf->process(utf8, &one, /*cutMethod=*/0))
    throw std::runtime_error(one.error);
  Feat f;
  f.phones.assign(one.phones.begin(), one.phones.end());
  f.word2ph = one.word2ph;
  if (f.phones.empty()) return f;
  if (f.word2ph.empty()) throw std::runtime_error("word2ph 为空: " + utf8);

  f.normU8 = u32ToUtf8(g2pNorm->textNormalize(utf8ToU32(utf8)));
  std::vector<int64_t> ids, tt, amask;
  tok->encode(f.normU8, &ids, &tt, &amask);
  // roberta forward 至 layer21 输出 (hidden_states[-3]); 权重只读复用。
  const size_t Ln = ids.size(), C = bm.cfg.hidden;
  bert::Matrix x;
  x.reset(Ln, C);
  for (size_t t = 0; t < Ln; ++t)
    for (size_t c = 0; c < C; ++c)
      x.d[t * C + c] = bm.word_emb.w[size_t(ids[t]) * C + c] +
                       bm.pos_emb.w[t * C + c] +
                       bm.type_emb.w[size_t(tt[t]) * C + c];
  bm.emb_ln.forward(x);
  std::vector<float> ext(Ln);
  for (size_t j = 0; j < Ln; ++j)
    ext[j] = (1.f - float(amask[j])) * bm.cfg.mask_neg;
  bert::Matrix y, scr, ctxh;
  const size_t stopAt = bm.cfg.layers - 3;
  for (size_t i = 0; i <= stopAt; ++i) {
    bm.stack[i].forward(x, ext, y, scr, ctxh);
    x.d.swap(y.d);
  }

  const size_t nPhones = f.phones.size();
  f.bert.assign(nPhones * 1024, 0.f);
  const size_t charN = cpCount(f.normU8);
  if (charN != f.word2ph.size())
    throw std::runtime_error("word2ph 与 norm_text 长度不一致 (" +
                             std::to_string(f.word2ph.size()) + " vs " +
                             std::to_string(charN) + "): " + f.normU8);
  if (Ln >= charN + 2) {
    size_t o = 0;
    for (size_t ch = 0; ch < charN; ++ch) {
      const float* row = x.d.data() + (1 + ch) * C;
      for (int r = 0; r < f.word2ph[ch]; ++r, ++o)
        std::memcpy(f.bert.data() + o * 1024, row, sizeof(float) * 1024);
    }
    if (o != nPhones)
      throw std::runtime_error("word2ph 总和 != phones 数: " + utf8);
  } else {
    std::fprintf(stderr, "[pipeline] 警告: token 数不足, BERT 特征置零 (%s)\n",
                 f.normU8.c_str());
  }
  return f;
}

}  // namespace

bool Pipeline::buildPrompt(PromptCond* out, std::string* err) {
  try {
    // CPUFast: strip 换行; 尾字符非 SPLITS 时补 "。"
    std::string pt = opt_.prompt_text;
    while (!pt.empty() && pt.back() == '\n') pt.pop_back();
    if (!pt.empty()) {
      // UTF-8 下这些拆分符均为 3 字节
      static const char* kSplits[] = {"。", "！", "？", "；", "，"};
      bool endsWithSplit = false;
      for (const char* sp : kSplits)
        if (pt.size() >= 3 &&
            pt.compare(pt.size() - 3, 3, sp) == 0)
          endsWithSplit = true;
      if (!endsWithSplit) pt += "。";
    }
    if (pt.empty()) {
      out->ready = true;  // no_prompt_text 口径
      return true;
    }
    Feat f = featurize(&tf_, &g2pNorm_, &tok_, bert_, pt, err);
    out->phones = std::move(f.phones);
    out->bert = std::move(f.bert);
    out->ready = true;
    return true;
  } catch (const std::exception& e) {
    if (err) *err = std::string("提示文本处理失败: ") + e.what();
    return false;
  }
}

bool Pipeline::synthesize(const std::string& utf8Text,
                          const std::string& refWavPath, SynthResult* out,
                          std::string* err) {
  try {
    // 1. 参考条件 (缓存/编码器)
    if (!buildReference(refWavPath, out, err)) return false;

    // 2. 文本前端: 先整体切段, 再逐段独立处理 (CPUFast 每段独立 clean_text)
    textfront::TextFrontend::Result full;
    if (!tf_.process(utf8Text, &full, opt_.cut_method)) {
      if (err) *err = full.error;
      return false;
    }
    if (full.sentences.empty()) {
      if (err) *err = "文本前端未产出任何合成段";
      return false;
    }

    // 3. SoVITS 噪声 RNG (自定种子语义, 与 torch 非位等价)
    std::mt19937_64 rng(opt_.seed);
    std::normal_distribution<float> nd(0.f, 1.f);

    // 3. 提示文本条件 (一次构建, 全段复用)
    if (!prompt_.ready && !buildPrompt(&prompt_, err)) return false;

    std::vector<float> allAudio;
    for (const auto& segStr : full.sentences) {
      SegmentResult seg;
      seg.sentence = segStr;

      // 3a. 单句前端 + BERT 特征 (cut0: 不再重切)
      Feat f = featurize(&tf_, &g2pNorm_, &tok_, bert_, segStr, err);
      if (f.phones.empty()) continue;  // 纯标点段跳过
      seg.norm_text = f.normU8;
      seg.phones = f.phones;  // SoVITS 文本输入 = 段自身 phones (无提示前缀)

      // 3b. AR 输入 = prompt_phones ⊕ 段 phones (CPUFast all_phones 口径),
      //     bert 同序拼接; 无提示文本时退化为段自身。
      std::vector<int64_t> phonesAll;
      std::vector<float> bertAll;
      if (prompt_.ready && !prompt_.phones.empty()) {
        phonesAll = prompt_.phones;
        phonesAll.insert(phonesAll.end(), f.phones.begin(), f.phones.end());
        bertAll = prompt_.bert;
        bertAll.insert(bertAll.end(), f.bert.begin(), f.bert.end());
      } else {
        phonesAll = f.phones;
        bertAll = f.bert;
      }

      // 3d. AR 贪心生成
      const double t0 = nowMs();
      ar::GenResult gen = ar_->generate(
          phonesAll.data(), phonesAll.size(), out->prompt_semantic.data(),
          out->prompt_semantic.size(), bertAll.data());
      seg.ar_ms = nowMs() - t0;
      seg.tokens = gen.sampled;
      seg.raw_argmax = gen.raw_argmax;  // ↔ pairs golden tokens 口径
      seg.hit_eos = gen.hit_eos;
      if (seg.tokens.empty()) {
        std::fprintf(stderr, "[pipeline] 警告: AR 未产出 token (%s)\n",
                     segStr.c_str());
        continue;
      }

      // 3e. SoVITS decode (噪声外部注入)
      const size_t Tq = seg.tokens.size() * 2;
      std::vector<float> noise(192 * Tq);
      for (float& v : noise) v = nd(rng);
      std::vector<int64_t> codes64(seg.tokens.begin(), seg.tokens.end());
      sovits::SovitsEngine::Inputs in;
      in.codes = codes64.data();
      in.n_codes = seg.tokens.size();
      in.phones = seg.phones.data();
      in.n_phones = seg.phones.size();
      in.ge = out->cond.ge.data();
      in.ge_text = out->cond.ge_text.data();
      in.noise = noise.data();
      in.Tq_expected = Tq;
      const double t1 = nowMs();
      sovits::Tensor2D wavOut;
      static sovits::Dumper noopDumper;  // 未 enable → 不落盘
      sovits_->run(in, wavOut, noopDumper);
      seg.voc_ms = nowMs() - t1;

      // 峰值归一 (>1 才除) 后入列
      seg.audio_frames = wavOut.T;
      double mx = 0.0;
      for (float v : wavOut.d) mx = std::max(mx, double(std::abs(v)));
      const bool needDiv = mx > 1.0;
      allAudio.reserve(allAudio.size() + seg.audio_frames + kSilence);
      for (size_t i = 0; i < seg.audio_frames; ++i)
        allAudio.push_back(needDiv ? wavOut.d[i] / float(mx) : wavOut.d[i]);
      // fragment_interval=0.3s 静音尾 (每段都加, 与 audio_postprocess 一致)
      allAudio.insert(allAudio.end(), kSilence, 0.f);
      out->segments.push_back(std::move(seg));
    }

    out->audio = std::move(allAudio);
    out->sr = kSr;
    return true;
  } catch (const std::exception& e) {
    if (err) *err = e.what();
    return false;
  }
}

}  // namespace gsv::rt::pipeline
