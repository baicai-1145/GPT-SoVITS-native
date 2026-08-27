// pipeline.cpp — C2: 全链路编排实现 (口径见 pipeline.hpp)
#include "runtime/pipeline.hpp"
#include "runtime/sovits_io.hpp"

#include "runtime/segqueue.hpp"

#include <atomic>
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
#if defined(GSV_AMX_GEMM)
#include "encoder/hubert.hpp"  // E12: amx_hubert_enabled() 装载期接线
#include "encoder/sv.hpp"     // E12: amx_sv_enabled() 装载期接线
#endif

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
  std::string trieP, pinyinP, cmuP;
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
  // cmudict 可选: 存在则启用英文段路径(B10), 缺失时纯中文不受影响
  firstExisting({joinPath(dataDir, "cmudict.bin"),
                 joinPath(dataDir + "/../..", "textfront/data/cmudict.bin")},
                &cmuP);

  // B6: G2PW 可选加载 (存在 g2pw_bert.gsv + g2pw_assets.bin + bert_vocab.txt 时自动启用)
  textfront::TextFrontend::G2pwOptions g2pwOpt;
  std::string g2pwGsv, g2pwAssets, g2pwVocab, g2pwOverrides;
  const bool haveG2pw =
      firstExisting({joinPath(weightsDir, "g2pw_bert.gsv"),
                     joinPath(weightsDir + "/..", "weights/g2pw_bert.gsv")},
                    &g2pwGsv) &&
      firstExisting({joinPath(dataDir, "g2pw_assets.bin"),
                     joinPath(dataDir + "/../..", "textfront/data/g2pw_assets.bin")},
                    &g2pwAssets) &&
      firstExisting({joinPath(dataDir, "bert_vocab.txt"),
                     joinPath(dataDir + "/../..", "textfront/data/bert_vocab.txt")},
                    &g2pwVocab);
  if (haveG2pw) {
    g2pwOpt.gsvPath = g2pwGsv;
    g2pwOpt.assetsBin = g2pwAssets;
    g2pwOpt.vocabPath = g2pwVocab;
    if (firstExisting({joinPath(dataDir, "polyphone_overrides.bin"),
                       joinPath(dataDir + "/../..",
                                "textfront/data/polyphone_overrides.bin")},
                      &g2pwOverrides)) {
      g2pwOpt.overridesBin = g2pwOverrides;
    }
  }

  // ---- E7: load 段分阶段画像 (GSV_LOAD_PROFILE=1 时 stderr 输出各阶段耗时) ----
  const bool profileLoad = ::getenv("GSV_LOAD_PROFILE") != nullptr;
  auto profMark = [profileLoad](const char* stage, double* t) {
    const double now = nowMs();
    if (profileLoad)
      std::fprintf(stderr, "[load-profile] %-22s %8.1f ms\n", stage, now - *t);
    *t = now;
  };
  double tStage = nowMs();

  try {
    // ---- E7 画像结论: 权重在 USB 外接盘 (~73MB/s) ----
    // 跨文件同时预取实测有害 (USB 寻道抖动, 冷态反而 16s→77s), 已撤销。
    // 保留逐文件 F_RDADVISE 能力 (GsvFile::prefetch / prefetch_file) 供
    // 单文件流内聚簇使用; 真正的冷态治理见 gsv_tool --slim (去 f32 冗余段,
    // 文件体积 -67%, 页缓存压力同比降)。
    if (profileLoad) profMark("prefetch_issue(异步,不阻塞)", &tStage);

    // FE-AUTO-1: 语种切分数据(lid.176.bin + budoux json)可选; 按环境变量
    // GSV_LID_BIN > dataDir > CPUFast 权威路径 的顺序探测。缺失时 auto
    // 模式在 process 报错, all_zh 不受影响。
    std::string lidP;
    if (const char* envLid = ::getenv("GSV_LID_BIN"); envLid && *envLid) {
      lidP = envLid;
    } else if (!firstExisting(
                   {joinPath(dataDir, "lid.176.bin"),
                    joinPath(dataDir + "/../..", "pretrained_models/lid.176.bin")},
                   &lidP)) {
      const char* cpuFast = "/Volumes/2T/GPT-SoVITS-CPUFast/GPT_SoVITS/"
                            "pretrained_models/fast_langdetect/lid.176.bin";
      if (std::filesystem::exists(cpuFast)) lidP = cpuFast;
    }
    std::string budouxD;
    if (!firstExisting({joinPath(dataDir, "budoux"),
                        joinPath(dataDir + "/../..", "textfront/data/budoux")},
                       &budouxD))
      budouxD.clear();

    if (!tf_.load(trieP, pinyinP, err, cmuP, haveG2pw ? &g2pwOpt : nullptr,
                  lidP, budouxD))
      return false;
    profMark("textfront(含G2PW)", &tStage);
    if (!tok_.load(joinPath(dataDir, "roberta_vocab.txt"), err)) return false;
    profMark("tokenizer_vocab", &tStage);

    bert_.cfg = bert::BertConfig{};  // roberta-wwm-ext-large 默认即此
    fBert_ = std::make_unique<rt::GsvFile>(
        joinPath(weightsDir, "roberta_wwm_ext_large.gsv"));
    fBert_->prefetch();  // GsvFile 内映射续期 WILLNEED (幂等)
    bert_.load(*fBert_, "bert");
    profMark("bert_load(1.9GB)", &tStage);

    fAr_ = std::make_unique<rt::GsvFile>(joinPath(weightsDir, "ar_s1v3.gsv"));
    ar_ = std::make_unique<ar::T2SEngine>(*fAr_);
    if (opt_.ar_fp16_kv || opt_.ar_fp16_gemv) {
      ar::T2SEngine::Fp16Options fo;
      fo.kv = opt_.ar_fp16_kv;
      fo.gemv = opt_.ar_fp16_gemv;
      ar_->set_fp16(fo);
      std::fprintf(stderr, "AR fp16: kv=%d gemv=%d (实验开关)\n", fo.kv,
                   fo.gemv);
    }
    profMark("ar_load(444MB)", &tStage);
    sovits_ = std::make_unique<sovits::SovitsEngine>();
    fSov_ = std::make_unique<rt::GsvFile>(joinPath(weightsDir, "sovits_v2ProPlus.gsv"));
    sovits_->load(fSov_->path(), opt.sovits_amx);
    profMark("sovits_load(460MB+pack)", &tStage);
    cond_.load(*fSov_);
#if defined(GSV_AMX_GEMM)
    // T8: cond 批量化 ref_enc 收进 --amx-enc 门控(默认 false=基线标量位级口径);
    //     T6 优化只在旗开时生效(mel 门口径验收)。
    cond_.setAmxEnc(opt_.enc_amx);
#endif
    profMark("cond_load", &tStage);
#if defined(GSV_AMX_GEMM)
    // E12: SV 装载前使能 — SvEngine 构造期据此预打包 conv panel。
    // GSV_AMX_ENC_SUB=nosv 可关掉 SV(隔离验收用)
    const char* sub2 = std::getenv("GSV_AMX_ENC_SUB");
    const bool no_sv = sub2 && std::string(sub2).find("nosv") != std::string::npos;
    encoder::amx_sv_enabled() = opt_.enc_amx && !no_sv;
#endif
    fHub_ = std::make_unique<rt::GsvFile>(joinPath(weightsDir, "hubert_base.gsv"));
#if defined(GSV_AMX_GEMM)
    // E12: 装载前使能 — HubertEngine 构造期据此预打包 dense panel。
    // GSV_AMX_ENC_SUB=nohub 可关掉 HuBERT(隔离验收用)
    const char* sub = std::getenv("GSV_AMX_ENC_SUB");
    const bool no_hub = sub && std::string(sub).find("nohub") != std::string::npos;
    encoder::amx_hubert_enabled() = opt_.enc_amx && !no_hub;
#endif
    hubert_ = std::make_unique<encoder::HubertEngine>(*fHub_);
    profMark("hubert_load(531MB)", &tStage);
    fSv_ = std::make_unique<rt::GsvFile>(
        joinPath(weightsDir, "eres2netv2_sv.gsv"));
    sv_ = std::make_unique<encoder::SvEngine>(*fSv_);
    profMark("sv_load(306MB)", &tStage);
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
    // E12: GSV_REF_TIMING 探针 — 编码侧分段耗时(--no-cache 口径 stderr 输出)。
    const bool refTim = std::getenv("GSV_REF_TIMING") != nullptr;
    const double tRef0 = nowMs();
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
    const double tSv0 = nowMs();
    sv_->forward3(fbank.data(), frames80);
    const double tSv1 = nowMs();
    std::vector<float> svEmb(sv_->emb_out());  // [20480]
    // E13-SV 探针: GSV_SV_EMB_DUMP=<path> 时落盘 svEmb(验证侧工具, 零行为变更)
    if (const char* dp = std::getenv("GSV_SV_EMB_DUMP"); dp && *dp) {
      if (FILE* fp = std::fopen(dp, "wb")) {
        std::fwrite(svEmb.data(), sizeof(float), svEmb.size(), fp);
        std::fclose(fp);
        std::fprintf(stderr, "[sv-dump] %s (%zu floats)\n", dp, svEmb.size());
      } else {
        std::fprintf(stderr, "[sv-dump] 写失败: %s\n", dp);
    }
    }

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
    const double tCond = nowMs();

    // HuBERT: 零尾填充与 CPUFast 一致 (cat[wav16k, zeros(9600)])
    std::vector<float> hubIn(wav16k);
    hubIn.insert(hubIn.end(), kSilence, 0.f);  // 0.3s@32k 的零段, CPUFast 同款
    const size_t T = hubert_->run(hubIn.data(), hubIn.size());
    const double tHub = nowMs();
    if (refTim) {
      std::fprintf(stderr,
                   "[ref-timing] sv.forward3=%.0fms(%zu帧) cond=%.0fms "
                   "hubert.run=%.0fms(T=%zu) 前处理(resample+fbank+load)=%.0fms "
                   "合计=%.0fms\n",
                   tSv1 - tSv0, frames80, tCond - tSv1, tHub - tCond, T,
                   tSv0 - tRef0, tHub - tRef0);
    }
    const std::vector<float>& hidden = hubert_->out();  // [T,768]

    // extract_latent: v2ProPlus semantic_frame_rate="25hz" ⇒
    //   ssl_proj = Conv1d(768,768,k=2,stride=2) 帧率减半, 再最近码字。
    const double tLat0 = refTim ? nowMs() : 0.0;  // E13 探针: extract_latent 段计时
    rt::GsvFile sovF(joinPath(weightsDir_, "sovits_v2ProPlus.gsv"));
    std::vector<float> sslW, sslB;  // W[768,768,2](out,in,tap), b[768]
    sovits::load_tensor_f32(sovF, "ssl_proj.weight", sslW);
    sovits::load_tensor_f32(sovF, "ssl_proj.bias", sslB);
    const size_t Tq = (T >= 2) ? ((T - 2) / 2 + 1) : 0;  // floor((L-k)/s)+1
    const double tLatOpen = refTim ? nowMs() : 0.0;  // E13: gsv 重开+读权重
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
    const double tLatProj = refTim ? nowMs() : 0.0;  // E13: im2col+sgemm+bias
    std::vector<float> embed;  // [1024,768]
    sovits::load_tensor_f32(sovF, "quantizer.vq.layers.0._codebook.embed",
                            embed, {1024, 768});
    out->prompt_semantic.resize(Tq);
    // T5: 最近码字块化 — dist[t,cb] = ‖proj_t − e_cb‖²
    //   = ‖proj_t‖²(行常量,argmin 无关) − 2·proj·e + ‖e_cb‖²(逐码不 可略)
    //   ⇒ 先 sgemm 算 −2·proj·embedᵀ, 再逐列加 norm_e[1024]。
    //   标量三重循环(80-87ms) → 单次 sgemm + 归一化逐行 argmin。
    //   数值护栏: fp32 块化累加序变化引入的绝对误差实测 ≤0.045(见
    //   .tmp/evidence-T5.md), 而 fp32 行 top-2 差实测 ≥3.0 ⇒ 默认 τ=1.0:
    //   差距 < GSV_RVQ_TIE_EPS 的帧回退原始双精度逐码精确复算(与基线判定
    //   位级同构); GSV_RVQ_AUDIT=1 打印回退/捕获数。
    {
      const char* epsEnv = std::getenv("GSV_RVQ_TIE_EPS");
      const double tieEps = epsEnv ? std::atof(epsEnv) : 1.0;
      const bool rvqAudit = std::getenv("GSV_RVQ_AUDIT") != nullptr;
      double normE[1024];
      for (int64_t cb = 0; cb < 1024; ++cb) {
        const float* e = embed.data() + size_t(cb) * 768;
        double s = 0.0;
        for (size_t c = 0; c < 768; ++c)
          s += double(e[c]) * double(e[c]);
        normE[cb] = s;
      }
      std::vector<float> dist(size_t(Tq) * 1024);
      kern::accel::sgemm('N', 'T', int(Tq), 1024, 768, -2.0f, proj.data(), 768,
                         embed.data(), 768, 0.0f, dist.data(), 1024);
      for (size_t t = 0; t < Tq; ++t) {
        float* row = dist.data() + t * 1024;
        for (int64_t cb = 0; cb < 1024; ++cb)
          row[cb] += float(normE[cb]);
      }
      size_t nFallback = 0, nFlip = 0;
      for (size_t t = 0; t < Tq; ++t) {
        const float* row = dist.data() + t * 1024;
        size_t b0 = 0;
        float v0, v1;
        if (row[0] <= row[1]) {
          v0 = row[0]; b0 = 0; v1 = row[1];
        } else {
          v0 = row[1]; b0 = 1; v1 = row[0];
        }
        for (size_t cb = 2; cb < 1024; ++cb) {
          const float d = row[cb];
          if (d < v0) {
            v1 = v0; v0 = d; b0 = cb;
          } else if (d < v1) {
            v1 = d;
          }
        }
        int64_t best = int64_t(b0);
        if (double(v1) - double(v0) < tieEps) {
          // 位级护栏分支: 双精度精确复算整帧(argmin 判定与基线同构:
          // cb 升序扫 + 严格小于刷新)
          ++nFallback;
          double bd = 1e30;
          int64_t bExact = 0;
          for (int64_t cb = 0; cb < 1024; ++cb) {
            const float* e = embed.data() + size_t(cb) * 768;
            double d = 0.0;
            for (size_t c = 0; c < 768; ++c) {
              const double diff = double(proj[t * 768 + c]) - double(e[c]);
              d += diff * diff;
            }
            if (d < bd) {
              bd = d;
              bExact = cb;
            }
          }
          if (bExact != best) ++nFlip;
          best = bExact;
        }
        out->prompt_semantic[t] = best;
      }
      if (rvqAudit)
        std::fprintf(stderr,
                     "[rvq-batch] Tq=%zu fallback=%zu flip_caught=%zu "
                     "tie_eps=%g\n",
                     Tq, nFallback, nFlip, tieEps);
    }
    if (refTim)  // E13 探针: extract_latent 三段(open/proj/rvq)
      std::fprintf(stderr,
                   "[ref-timing] extract_latent=%.0fms(Tq=%zu) = open%.0f + "
                   "proj%.0f + rvq%.0f\n",
                   nowMs() - tLat0, Tq, tLatOpen - tLat0, tLatProj - tLatOpen,
                   nowMs() - tLatProj);
    // E13-MIX/T5 探针: prompt_semantic codes + ssl_proj 输出落盘
    // (GSV_RVQ_DUMP=<path>; 两路径位级比对用, 零行为变更)
    if (const char* dumpPath = std::getenv("GSV_RVQ_DUMP")) {
      FILE* df = fopen(dumpPath, "wb");
      if (df) {
        const bool okD = fwrite(out->prompt_semantic.data(), sizeof(int64_t),
                                out->prompt_semantic.size(), df) ==
                            out->prompt_semantic.size() &&
                        fwrite(proj.data(), sizeof(float), proj.size(), df) ==
                            proj.size();
        fclose(df);
        if (!okD)
          std::fprintf(stderr, "警告: GSV_RVQ_DUMP 写入不完整: %s\n", dumpPath);
      } else {
        std::fprintf(stderr, "警告: GSV_RVQ_DUMP 无法写入: %s\n", dumpPath);
      }
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

Feat featurize(const textfront::TextFrontend* tf,
               const textfront::ChineseG2p* g2pNorm, const BertTokenizer* tok,
               const bert::BertModel& bm, const std::string& utf8,
               std::string* /*err*/,
               textfront::TextLangMode langMode = textfront::TextLangMode::Auto) {
  const double tf0 = nowMs();
  textfront::TextFrontend::Result one;
  if (!tf->process(utf8, &one, /*cutMethod=*/0, langMode))
    throw std::runtime_error(one.error);
  if (std::getenv("GSV_FRONT_TIMING"))
    std::fprintf(stderr, "[front] tf.process(jieba+g2pw规则+g2pwBERT)=%.1fms\n", nowMs()-tf0);
  Feat f;
  f.phones.assign(one.phones.begin(), one.phones.end());
  f.word2ph = one.word2ph;
  if (f.phones.empty()) return f;
  if (f.word2ph.empty()) throw std::runtime_error("word2ph 为空: " + utf8);

  // BERT 输入文本必须与 word2ph 同基准: process(cut0) 会按 splitSentences
  // 规则补前置 。(短句/非 SPLITS 开头), 该规则发生在 clean_text 之前 ⇒
  // 归一文本要对"有效段"(one.sentences[0]) 取, 不能对原始 utf8。
  if (one.sentences.empty()) return f;
  f.normU8 = u32ToUtf8(g2pNorm->textNormalize(utf8ToU32(one.sentences[0])));
  std::vector<int64_t> ids, tt, amask;
  tok->encode(f.normU8, &ids, &tt, &amask);
  // roberta forward 至 layer21 输出 (hidden_states[-3]); 权重只读复用。
  const double bt0 = nowMs();
  const size_t Ln = ids.size(), C = bm.cfg.hidden;
  bert::Matrix x;
  x.reset(Ln, C);
  for (size_t t = 0; t < Ln; ++t)
    for (size_t c = 0; c < C; ++c)
      x.d[t * C + c] = bm.word_emb_w[size_t(ids[t]) * C + c] +
                       bm.pos_emb_w[t * C + c] +
                       bm.type_emb_w[size_t(tt[t]) * C + c];
  bm.emb_ln.forward(x);
  std::vector<float> ext(Ln);
  for (size_t j = 0; j < Ln; ++j)
    ext[j] = (1.f - float(amask[j])) * bm.cfg.mask_neg;
  bert::Matrix y, scr, ctxh;
  struct FrontBertTiming { double v = 0; };
  static thread_local double s_bert_ms = 0; (void)s_bert_ms;
  const size_t stopAt = bm.cfg.layers - 3;
  for (size_t i = 0; i <= stopAt; ++i) {
    bm.stack[i].forward(x, ext, y, scr, ctxh);
    x.d.swap(y.d);
  }
  if (std::getenv("GSV_FRONT_TIMING"))
    std::fprintf(stderr, "[front] roberta 21层 forward=%.1fms (L=%zu)\n", nowMs()-bt0, Ln);

  const size_t nPhones = f.phones.size();
  f.bert.assign(nPhones * 1024, 0.f);
  const size_t charN = cpCount(f.normU8);
  if (charN != f.word2ph.size()) {
    // 混合中英段: B10 的 EN 片词 word2ph 是"每词 token"粒度(总和仍=phones),
    // 与 get_bert_feature 的按字符索引不兼容 —— CPUFast 对非 zh 语言本就置零
    // (get_bert_inf)。此处同样置零并告警; 契约细化待决策者/B6 裁定。
    std::fprintf(stderr,
                 "[pipeline] 警告: 混合/英文段 word2ph(%zu)!=字符数(%zu), "
                 "该段 BERT 特征置零 (%s)\n",
                 f.word2ph.size(), charN, f.normU8.c_str());
    return f;
  }
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
    textfront::TextLangMode lm = opt_.lang_mode == "all_zh"
                                     ? textfront::TextLangMode::AllZh
                                     : textfront::TextLangMode::Auto;
    Feat f = featurize(&tf_, &g2pNorm_, &tok_, bert_, pt, err, lm);
    out->phones = std::move(f.phones);
    out->bert = std::move(f.bert);
    out->ready = true;
    return true;
  } catch (const std::exception& e) {
    if (err) *err = std::string("提示文本处理失败: ") + e.what();
    return false;
  }
}

// ---- D2: 三阶段实现 (串行/重叠两模式调用完全相同的函数) ----
Pipeline::SegArIn Pipeline::stageFeaturize(const SegText& t) const {
  textfront::TextLangMode lm = opt_.lang_mode == "all_zh"
                                   ? textfront::TextLangMode::AllZh
                                   : textfront::TextLangMode::Auto;
  Feat f = featurize(&tf_, &g2pNorm_, &tok_, bert_, t.sentence, nullptr, lm);
  SegArIn r;
  r.phonesSeg = std::move(f.phones);
  r.normText = std::move(f.normU8);
  if (r.phonesSeg.empty()) return r;  // 纯标点段: 下游直通
  // AR 输入 = prompt_phones ⊕ 段 phones (CPUFast all_phones 口径)
  if (prompt_.ready && !prompt_.phones.empty()) {
    r.phonesAll = prompt_.phones;
    r.phonesAll.insert(r.phonesAll.end(), r.phonesSeg.begin(),
                       r.phonesSeg.end());
    r.bertAll = prompt_.bert;
    r.bertAll.insert(r.bertAll.end(), f.bert.begin(), f.bert.end());
  } else {
    r.phonesAll = r.phonesSeg;
    r.bertAll = std::move(f.bert);
  }
  return r;
}

Pipeline::SegSovIn Pipeline::stageAr(const SegArIn& in,
                                     const std::vector<int64_t>& promptSem) {
  SegSovIn o;
  o.empty = in.phonesSeg.empty();
  if (o.empty) return o;  // 纯标点段直通, 不触碰 RNG (与 C2 串行语义一致)

  const double t0 = nowMs();
  ar::GenResult gen =
      opt_.ar_sample_on
          ? ar_->generate(in.phonesAll.data(), in.phonesAll.size(),
                          promptSem.data(), promptSem.size(),
                          in.bertAll.data(),
                          ar::T2SEngine::kMaxDecodeSteps, nullptr,
                          &opt_.ar_sampling)
          : ar_->generate(in.phonesAll.data(), in.phonesAll.size(),
                          promptSem.data(), promptSem.size(),
                          in.bertAll.data());
  o.arMs = nowMs() - t0;
  o.hitEos = gen.hit_eos;
  o.codes.assign(gen.sampled.begin(), gen.sampled.end());
  o.rawArgmax = gen.raw_argmax;
  o.phonesSeg = in.phonesSeg;
  o.normText = in.normText;
  o.empty = gen.sampled.empty();
  if (o.empty) {
    std::fprintf(stderr, "[pipeline] 警告: AR 未产出 token (%s)\n",
                 in.normText.c_str());
    return o;
  }

  // 噪声按段序抽取: overlap 模式下本函数只在 stage2 线程 FIFO 执行 ⇒
  // 与串行模式的抽取序列逐位一致
  const size_t Tq = o.codes.size() * 2;
  o.noise.resize(192 * Tq);
  if (!rngSeeded_) {
    rng_.seed(opt_.seed);
    rngSeeded_ = true;
  }
  std::normal_distribution<float> nd(0.f, 1.f);
  for (float& v : o.noise) v = nd(rng_);
  return o;
}

void Pipeline::stageVoc(const SegSovIn& in, SegmentResult* seg,
                        const DecodeCondition& cond) const {
  seg->norm_text = in.normText;
  seg->phones = in.phonesSeg;
  seg->tokens.assign(in.codes.begin(), in.codes.end());
  seg->raw_argmax = in.rawArgmax;
  seg->ar_ms = in.arMs;
  seg->hit_eos = in.hitEos;
  if (in.empty) return;

  sovits::SovitsEngine::Inputs si;
  si.codes = in.codes.data();
  si.n_codes = in.codes.size();
  si.phones = in.phonesSeg.data();
  si.n_phones = in.phonesSeg.size();
  si.ge = cond.ge.data();
  si.ge_text = cond.ge_text.data();
  si.noise = in.noise.data();
  si.Tq_expected = in.codes.size() * 2;
  const double t1 = nowMs();
  sovits::Tensor2D wavOut;
  static sovits::Dumper noopDumper;  // 未 enable → 不落盘
  sovits_->run(si, wavOut, noopDumper);
  seg->voc_ms = nowMs() - t1;
  seg->audio_frames = wavOut.T;

  // 峰值归一 (>1 才除), 存归一后波形供拼接
  double mx = 0.0;
  for (float v : wavOut.d) mx = std::max(mx, double(std::abs(v)));
  const bool needDiv = mx > 1.0;
  seg->audio.resize(wavOut.T);
  for (size_t i = 0; i < wavOut.T; ++i)
    seg->audio[i] = needDiv ? wavOut.d[i] / float(mx) : wavOut.d[i];
}

static void writeTimingCsv(const std::string& path,
                           const std::vector<SegmentResult>& segs) {
  FILE* fp = fopen(path.c_str(), "w");
  if (!fp) {
    std::fprintf(stderr, "[pipeline] 警告: 无法写计时 CSV %s\n", path.c_str());
    return;
  }
  std::fprintf(fp,
               "idx,norm_text,phones,tokens,t_textfront_ms,t_ar_ms,t_sov_ms,"
               "t_wait_ms,audio_frames,hit_eos\n");
  for (size_t i = 0; i < segs.size(); ++i) {
    const auto& g = segs[i];
    std::fprintf(fp, "%zu,\"%s\",%zu,%zu,%.3f,%.3f,%.3f,%.3f,%zu,%d\n", i,
                 g.norm_text.c_str(), g.phones.size(), g.tokens.size(), g.tf_ms,
                 g.ar_ms, g.voc_ms, g.wait_ms, g.audio_frames,
                 int(g.hit_eos));
  }
  fclose(fp);
}

bool Pipeline::synthesize(const std::string& utf8Text,
                          const std::string& refWavPath, SynthResult* out,
                          std::string* err) {
  const double tSynth = nowMs();
  try {
    // 1. 参考条件 (缓存/编码器)
    if (!buildReference(refWavPath, out, err)) return false;

    // 2. 文本前端切段 (CPUFast 口径); 提示文本条件一次构建
    textfront::TextFrontend::Result full;
    if (!tf_.process(utf8Text, &full, opt_.cut_method)) {
      if (err) *err = full.error;
      return false;
    }
    if (full.sentences.empty()) {
      if (err) *err = "文本前端未产出任何合成段";
      return false;
    }
    if (!prompt_.ready && !buildPrompt(&prompt_, err)) return false;

    std::vector<float> allAudio;
    auto appendSeg = [&](SegmentResult&& seg) {
      if (seg.audio.empty()) return;  // 空段(纯标点/无 token)不进结果
      allAudio.reserve(allAudio.size() + seg.audio.size() + kSilence);
      allAudio.insert(allAudio.end(), seg.audio.begin(), seg.audio.end());
      // fragment_interval=0.3s 静音尾 (每段都加, 与 audio_postprocess 一致)
      allAudio.insert(allAudio.end(), kSilence, 0.f);
      out->segments.push_back(std::move(seg));
    };

    if (!opt_.overlap) {
      // ---- 串行模式: 同一阶段函数顺序内联 (数值路径与重叠态一致) ----
      rngSeeded_ = false;
      for (const auto& segStr : full.sentences) {
        SegmentResult seg;
        seg.sentence = segStr;
        const double t0 = nowMs();
        SegArIn ain = stageFeaturize({segStr});
        seg.tf_ms = nowMs() - t0;
        seg.norm_text = ain.normText;
        seg.phones = ain.phonesSeg;
            SegSovIn sin = stageAr(ain, out->prompt_semantic);
        if (!opt_.sovits_in_dump.empty()) sovInSink_.push_back(sin);
        stageVoc(sin, &seg, out->cond);
            if (!seg.audio.empty()) noteFirstPacket(nowMs() - tSynth);
        appendSeg(std::move(seg));
      }
    } else {
      // ---- 重叠模式: SegQueue 双缓冲, AR(N+1) ‖ SoVITS(N) ----
      // QoS 落位(§4): textfront/BERT→utility(E 核); AR/VITS→P 核。
      using gsv::runtime::SegTiming;
      using Pipe = gsv::runtime::SegmentPipeline<SegText, SegArIn, SegSovIn,
                                                 SegmentResult>;
      Pipe pipe(
          [this](const SegText& t, SegTiming&) {
            return stageFeaturize(t);
          },
          [this, &out](const SegArIn& a, SegTiming&) {
            return stageAr(a, out->prompt_semantic);
          },
          [this, &tSynth, &out](const SegSovIn& v, SegTiming&) {
            SegmentResult seg;
            if (!opt_.sovits_in_dump.empty()) sovInSink_.push_back(v);  // stage3 FIFO 单线程
            stageVoc(v, &seg, out->cond);
            if (!seg.audio.empty()) noteFirstPacket(nowMs() - tSynth);
            return seg;
          },
          /*qos1=*/"utility", /*qos2=*/"user_initiated",
          /*qos3=*/"user_initiated", /*queueCap=*/2);
      pipe.start();

      // 异常传输: stage 抛错经 envelope 送达 drain, 此处重抛给调用方
      size_t submitted = 0, drained = 0;
      try {
        for (; submitted < full.sentences.size(); ++submitted)
          pipe.submit({full.sentences[submitted]});
        pipe.shutdownInput();
        while (drained < full.sentences.size()) {
          Pipe::WavItem it;
          if (!pipe.drain(it)) break;
          SegmentResult seg = std::move(it.data);
          if (it.exc) std::rethrow_exception(it.exc);
          if (it.timing) {
            seg.tf_ms = it.timing->t_textfrontMs;
            seg.wait_ms = it.timing->t_waitMs;
          }
          seg.sentence = full.sentences[drained];
          appendSeg(std::move(seg));
          ++drained;
        }
      } catch (...) {
        pipe.cancel();
        throw;
      }
    }

    out->audio = std::move(allAudio);
    out->sr = kSr;
    out->first_packet_ms = firstPacketMs();
    if (!opt_.timing_csv.empty())
      writeTimingCsv(opt_.timing_csv, out->segments);
    if (!opt_.sovits_in_dump.empty()) {
      std::string derr;
      if (!dumpSovitsIn(opt_.sovits_in_dump, out->cond, sovInSink_, &derr))
        fprintf(stderr, "警告: %s\n", derr.c_str());
      else
        fprintf(stderr, "SoVITS输入快照已存: %s (%zu 段)\n",
                opt_.sovits_in_dump.c_str(), sovInSink_.size());
      sovInSink_.clear();
    }
    return true;
  } catch (const std::exception& e) {
    if (err) *err = e.what();
    return false;
  }
}

// SoVITS 专攻重放: 跳过 textfront/AR/BERT/编码器, 直接逐段 stageVoc。
// 输入含 noise 快照 ⇒ 与原链位级同输入; cond 来自快照(不依赖 ref-wav/缓存)。
bool Pipeline::synthesizeFromSovitsIn(const std::string& sovitsInPath,
                                      SynthResult* out, std::string* err) {
  try {
    DecodeCondition cond;
    std::vector<SegSovIn> segs;
    if (!loadSovitsIn(sovitsInPath, &cond, &segs, err)) return false;

    std::vector<float> allAudio;
    for (auto& sin : segs) {
      SegmentResult seg;
      seg.sentence = sin.normText;  // 重放时无原文, 用 normText 标识段
      stageVoc(sin, &seg, cond);
      seg.tf_ms = 0;  // 未执行 textfront
      seg.wait_ms = 0;
      if (!seg.audio.empty()) {
        allAudio.reserve(allAudio.size() + seg.audio.size() + kSilence);
        allAudio.insert(allAudio.end(), seg.audio.begin(), seg.audio.end());
        allAudio.insert(allAudio.end(), kSilence, 0.f);
      }
      out->segments.push_back(std::move(seg));
    }
    out->audio = std::move(allAudio);
    out->sr = kSr;
    if (!opt_.timing_csv.empty())
      writeTimingCsv(opt_.timing_csv, out->segments);
    return true;
  } catch (const std::exception& e) {
    if (err) *err = e.what();
    return false;
  }
}

}  // namespace gsv::rt::pipeline
