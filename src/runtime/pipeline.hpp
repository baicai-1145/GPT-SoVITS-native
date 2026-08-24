// pipeline.hpp — C2: gsv-native 全链路编排层。
//
//   TextFrontend(B9) → roberta BERT(B8) → AR/T2S(B12) → SoVITS(B34)
//        ↑ 参考音频路径: wav→16k→HuBERT(C1)→RVQ encode(prompt_semantic);
//                       wav→32k 归一→spec→ref_enc + sv(16k)→ge/ge_text
//
// 口径对齐 CPUFast TTS.py (text_split_method=cutN, batch=1, speed=1):
//   - 每句独立推理; phones/word2ph/bert 均按句计算;
//   - prompt_text 不参与 v2ProPlus 输入(no_prompt_text);
//   - 段间静音 fragment_interval=0.3s(9600 样本@32k), 逐段峰值归一(>1 才除);
//   - 最终 int16 = float×32768 截断(wav_writer)。
// 本文件为任务卡授权的编排域; 各引擎头文件只读使用。
#pragma once

#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "ar/t2s_engine.hpp"
#include "bert/bert_io.hpp"
#include "bert/bert_model.hpp"
#include "encoder/fbank.hpp"
#include "encoder/hubert.hpp"
#include "encoder/ref_cache.hpp"
#include "encoder/sv.hpp"
#include "runtime/pipeline_audio.hpp"
#include "runtime/pipeline_condition.hpp"
#include "runtime/pipeline_tokenizer.hpp"
#include "sovits/sovits_engine.hpp"
#include "textfront/chinese_g2p.h"
#include "textfront/textfront.h"

namespace gsv::rt::pipeline {

struct PipelineOptions {
  int threads = 0;         // 0 = 默认(P+E 全核)
  uint64_t seed = 42;      // SoVITS 噪声种子(自定 RNG; 与 torch 非位等价)
  int cut_method = 1;      // cut0..cut5, 与 CPUFast text_split_method 对应
  bool use_ref_cache = true;
  std::string prompt_text = "原来你也玩原神。";  // 参考文本(AR 条件前缀);
                            // 空串 = 无提示文本(CPUFast no_prompt_text 口径)
  bool overlap = false;    // D2: AR(N+1) ‖ SoVITS(N) 流水重叠(纯调度, 数值同串行)
  std::string timing_csv;  // D2: per-segment 三阶段耗时 CSV 路径 (空=不写)
};

struct SegmentResult {
  std::string sentence;             // 前端切出的合成段(UTF-8)
  std::string norm_text;            // BERT 实际输入文本(textNormalize 后)
  std::vector<int64_t> phones;      // symbols2 ids (AR 文本输入)
  std::vector<int32_t> tokens;      // AR 实际生成序列(不含 EOS)
  std::vector<int32_t> raw_argmax;  // 每步惩罚后全词表 argmax ↔ pairs tokens
  size_t audio_frames = 0;          // 本段样本数(@32k, 未加静音)
  std::vector<float> audio;         // 本段归一后波形 (峰值>1 已除)
  double ar_ms = 0.0, voc_ms = 0.0;
  double tf_ms = 0.0, wait_ms = 0.0;  // D2: 前端耗时/队列等待 (CSV 用)
  bool hit_eos = false;
};

struct SynthResult {
  std::vector<float> audio;         // 全部段拼接+段间静音 @32000
  double first_packet_ms = 0.0;     // 合成开始→首段音频就绪 (生产侧口径)
  uint32_t sr = 32000;
  std::vector<SegmentResult> segments;
  DecodeCondition cond;             // ge / ge_text (调试与验收)
  std::vector<int64_t> prompt_semantic;
  bool ref_from_cache = false;
};

class Pipeline {
 public:
  // weightsDir 下应有 ar_s1v3.gsv 等; dataDir 下有 jieba_trie.bin/
  // pinyin.bin/roberta_vocab.txt。加载失败返回 false 并写 err。
  bool load(const std::string& weightsDir, const std::string& dataDir,
            const PipelineOptions& opt, std::string* err);

  // 全链路合成。任何失败抛/写 err 并返回 false(错误信息面向 CLI 用户)。
  bool synthesize(const std::string& utf8Text, const std::string& refWavPath,
                  SynthResult* out, std::string* err);

  static constexpr uint32_t kSr = 32000;
  static constexpr size_t kSilence = 9600;  // fragment_interval=0.3s@32k

 private:
  // 提示文本条件 (CPUFast prompt_cache["phones"/"bert_features"] 对应物):
  // 每个 AR 输入 = prompt_phones + 段 phones, bert 同序拼接。
  struct PromptCond {
    bool ready = false;
    std::vector<int64_t> phones;
    std::vector<float> bert;  // [len(phones),1024]
  };
  bool buildPrompt(PromptCond* out, std::string* err);

  bool buildReference(const std::string& refWavPath, SynthResult* out,
                      std::string* err);

  // ---- D2: 三阶段载荷 (envelope 在两模式间共享同一数值路径) ----
  struct SegText {
    std::string sentence;
  };
  struct SegArIn {
    // featurize 产物 + 提示前缀拼接结果 (stage2 直接可用)
    std::vector<int64_t> phonesSeg;   // 段自身 phones (SoVITS 文本输入)
    std::vector<int64_t> phonesAll;   // 提示 ⊕ 段 (AR 输入)
    std::vector<float> bertAll;       // [len(phonesAll),1024]
    std::string normText;
  };
  struct SegSovIn {
    std::vector<int64_t> codes;       // AR 生成 token (= sampled)
    std::vector<int32_t> rawArgmax;   // ↔ pairs golden tokens 口径
    std::vector<int64_t> phonesSeg;
    std::vector<float> noise;         // stage2 内按段序抽取 (RNG 序不变)
    std::string normText;             // BERT 输入文本 (CSV/日志用)
    double arMs = 0.0;
    bool hitEos = false;
    bool empty = false;               // 纯标点段直通
  };

  // 阶段函数: 两模式调用完全相同 ⇒ 纯调度差异不触数值。
  // stageFeaturize 前端失败以异常上报(overlap 下经 envelope 顺序送达)。
  SegArIn stageFeaturize(const SegText& t) const;
  SegSovIn stageAr(const SegArIn& in, const std::vector<int64_t>& promptSem);
  void stageVoc(const SegSovIn& in, SegmentResult* seg,
                const DecodeCondition& cond) const;

  // RNG: 仅 stage2/串行主线程按段序触碰 (overlap 下 FIFO 单线程独占)
  std::mt19937_64 rng_{42};
  bool rngSeeded_ = false;

 public:
  // D2: 重叠模式下由 stage3 在"首段音频产出"时刻写入(生产侧口径,
  // 与 drain 解耦); 串行模式在 appendSeg 处写。0 = 未产出一包。
  std::atomic<int64_t> firstPacketMsX100{0};
  double firstPacketMs() const {
    return double(firstPacketMsX100.load(std::memory_order_relaxed)) / 100.0;
  }
  void noteFirstPacket(double msFromSynthStart) {
    int64_t expect = 0;
    const int64_t v = int64_t(msFromSynthStart * 100.0);
    firstPacketMsX100.compare_exchange_strong(expect, v);
  }

 private:

  PipelineOptions opt_;
  textfront::TextFrontend tf_;
  textfront::ChineseG2p g2pNorm_;  // 仅用 textNormalize (无需词典数据)
  BertTokenizer tok_;
  bert::BertModel bert_;
  std::unique_ptr<ar::T2SEngine> ar_;
  std::unique_ptr<sovits::SovitsEngine> sovits_;
  std::unique_ptr<encoder::HubertEngine> hubert_;
  std::unique_ptr<encoder::SvEngine> sv_;
  // E2-ENC: fp16 直读视图(Linear/DenseF16/conv w16 指向 mmap)要求 GsvFile
  // 生命周期覆盖引擎 —— 由 Pipeline 持有(析构顺序: 引擎先于文件, 成员倒序✓)
  std::unique_ptr<rt::GsvFile> fBert_, fAr_, fSov_, fHub_, fSv_;
  ConditionBuilder cond_;
  encoder::RefCache refCache_{"c2cond-v2"};  // v2: ssl_proj stride-2 口径
  std::string weightsDir_;
  PromptCond prompt_;
};

}  // namespace gsv::rt::pipeline
