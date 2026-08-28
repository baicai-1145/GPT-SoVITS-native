// t2s_engine.hpp — AR/T2S 引擎 (M1 fp32 步): prefill(Accelerate GEMM) + decode(NEON GEMV)
//
// 结构对照 CPUFast: GPT_SoVITS/AR/models/t2s_model.py 的 Text2SemanticDecoder +
// T2SBlock.process_prompt / decode_next_token / infer_panel_naive (eval, dropout=identity)。
// 数值纪律: 权重 fp16 存储(.gsv f16 段)逐元素无损升位、计算/累加/统计量全 fp32 —— 与
// torch fp32 基线同构 (tests/golden/CALIBRATION.md 事实 1)。
//
// 推理循环语义 (infer_panel_naive, batch=1):
//   1. x_text = emb_text(phones) + bert_proj(bert_feat_1024); 再加 alpha_text·PE(0..T-1)
//   2. y = prompt(P 个音频语义 token); y_pos = emb_audio(y) + alpha_audio·PE(0..P-1)
//   3. prefill: xy=[x_text;y_pos] 过 24 层 post-LN block;
//      注意力掩码 = 文本行只见文本 + 音频行见文本与因果音频 (torch xy_attn_mask 同构)
//   4. 循环 idx=0..max_steps-1:
//        logits = ar_predict_layer(最后位置输出)          ← 无 bias Linear(512→vocab)
//        raw_argmax[idx] = argmax(logits)                 ← golden `tokens` 字段口径(hook 在惩罚前捕获)
//        penalty: 对历史 y(prompt+已生成) 中每个 token c:
//            logits[c] = logits[c]<0 ? logits[c]*1.35 : logits[c]/1.35
//        idx<11 时采样与停止检查剔除 EOS 列 ("至少预测出10个token不然不给停止")
//        sample = argmax(penalty 后 logits); sample==EOS ⇒ 停止(EOS 不入序列)
//   5. 下一步输入 = emb_audio(sample) + alpha_audio·PE(P+idx)
#pragma once

#include "kern/accel.hpp"
#include "kern/gemm_f16_amx.hpp"  // E11-2: prefill AMX 预打包面板; E11-5: prefill SDPA AMX
#include "kern/kern.hpp"
#include "runtime/gsv_loader.hpp"

#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

namespace gsv::ar {

namespace accel = kern::accel;  // 引擎内简写

struct T2SDims {
  size_t d_model = 0;     // hidden_dim == embedding_dim (512)
  size_t n_heads = 0;     // head (16)
  size_t ffn = 0;         // linear_units (2048)
  size_t n_layers = 0;    // n_layer (24)
  size_t vocab = 0;       // 音频语义词表含 EOS (1025)
  size_t phone_vocab = 0; // 文本音素词表 (732)
  size_t bert_dim = 0;    // bert_proj 输入维 (1024)
  int eos = 0;            // EOS id (1024 == vocab-1)
  double ln_eps = 1e-5;   // layer_norm_eps
};

// 一次贪心生成的产物。字段口径与 tests/golden/pairs/*.pt 一一对应。
// 注意: golden 的 logits/tokens 均为"repetition penalty 施压后"的状态(导出 hook
// 持有被 scatter_ 就地修改的存储), 本引擎按同一口径记录。
struct GenResult {
  std::vector<int32_t> raw_argmax;   // 每步惩罚后全词表 argmax ↔ golden `tokens`
  std::vector<int32_t> sampled;      // 实际生成序列(prompt 之后; 不含触发停止的 EOS)
  std::vector<float> logits_first8;  // 前 8 步(惩罚后)logits ↔ logits_first8
  std::vector<float> logits_last;    // 最后一步(惩罚后)logits [vocab] ↔ logits_last
  size_t steps = 0;                  // 执行的 decode 步数(含触发停止的最后一步) ↔ n_ar_steps
  bool hit_eos = false;              // true=EOS 正常终止; false=max_steps 用尽
};

// E4: 采样参数(对齐 python infer_panel_naive LogitsProcessor 链)。
// 默认 mode=greedy → 与原贪心路径位级一致(保持 B12 golden G1/G2)。
// mode=topk → 复现 python sample(k=15, p=1, temp=1, pen=1.35) 自然 EOS,
//           根治长单段复读环。产品级修复。
struct SamplingParams {
  enum class Mode { Greedy, TopK };
  Mode mode = Mode::Greedy;
  size_t top_k = 15;                // python 默认 15, topk 启用时生效
  float top_p = 1.0f;               // 默认 1.0 不裁减; <1 启用 nucleus
  float temperature = 1.0f;         // 1.0 不缩放; 0 退贪心
  float rep_penalty = 1.35f;        // 与 kRepPenalty 一致; 1.0 不施压
  uint64_t seed = 0;                // 0 = 末固定 (用 std::random_device)
};

// 调试对拍钩子(prefill 阶段逐层导出, 仅验收工具用)
struct GenDebug {
  virtual ~GenDebug() = default;
  virtual void on_input(const float* /*xy*/, size_t /*S*/) {}
  virtual void on_layer(size_t /*l*/, const float* /*x*/, size_t /*S*/) {}
};

class T2SEngine {
 public:
  // 采样常量(CPUFast infer_panel_naive 默认值, golden 以此导出)
  static constexpr float kRepPenalty = 1.35f;
  static constexpr size_t kEosMaskSteps = 11;     // idx<11 ⇒ 剔除 EOS 列
  static constexpr size_t kMaxDecodeSteps = 1500; // MAX_AR_DECODE_STEPS
                                                  // (early_stop_num=50*57=2850>1500, 不激活)

  // M1-fp16: 第二步开关(默认全关, 保持 fp32 步数值路径不变)。
  //   kv   : KV cache 以 fp16 位型存储(写入舍入), 读出升位 fp32 计算
  //   gemv : decode 全部 GEMV 走 kern::gemv_f16x_fmlal(激活先舍入 fp16,
  //          FMLAL fp16×fp16→fp32 无中间舍入累加)
  struct Fp16Options {
    bool kv = false;
    bool gemv = false;
  };
  void set_fp16(const Fp16Options& o);
  const Fp16Options& fp16() const { return fp16_; }

  // 从 .gsv 加载全部权重; 维度取自 config JSON(版本锁纪律, 不硬编码形状)。
  explicit T2SEngine(const rt::GsvFile& f);
  const T2SDims& dims() const { return dims_; }

  // 全链路: prefill + 贪心 decode。phones[T] 文本音素, prompt[P] 音频语义,
  // bert1024 为 [T,bert_dim] 行主(即 pairs 的 bert_feat_1024 摊平)。
  // sampling = nullopt → 贪心(位级一致默认, B12 golden 口径)
  // sampling = SamplingParams{Mode::TopK,...} → python 口径采样(根治复读)
  GenResult generate(const int64_t* phones, size_t T,
                     const int64_t* prompt, size_t P,
                     const float* bert1024,
                     size_t max_steps = kMaxDecodeSteps,
                     GenDebug* dbg = nullptr,
                     const SamplingParams* sampling = nullptr);

  // K1: KV cache 复用快照管理
  void set_kv_reuse(bool enable) { kv_reuse_ = enable; }
  bool kv_reuse() const { return kv_reuse_; }
  bool last_prefill_hit() const { return last_prefill_hit_; }
  void reset_kv_cache() { prompt_snapshot_.valid = false; }

  double last_prefill_ms() const { return last_prefill_ms_; }
  double last_decode_ms() const { return last_decode_ms_; }

  // ---- 单层前向原语(公开供单测做 prefill/decode 一致性对拍) ----
  // KV cache 布局(自定, 见 README 注释): 每层两个独立缓冲 kcache/vcache,
  // 行主 token-major: cache[tok*D + head*HD + e], HD=D/heads。
  // fp32 模式存 float; fp16.kv 模式存 uint16_t(f16 位型), 接口指针仍按
  // 实际启用的缓冲传入(见 set_fp16/generate 的分派)。
  //
  // prefill: x[S,D] 就地更新为层输出; 该层 k/v 追加写入 cache 第 pos..pos+S-1 槽。
  // causal_prefix>0 时前 causal_prefix 个 query 只允许看见同数量的 key(文本前缀段),
  // 其余 query 标准因果 —— 等价 torch xy_attn_mask [文本行只见文本; 音频行因果]。
  void block_prefill(size_t l, float* x, size_t S, size_t pos,
                     size_t causal_prefix, float* kcache, float* vcache) {
    if (fp16_.kv)
      throw std::runtime_error(
          "fp16.kv 模式下请走 generate() 内部路径(缓冲类型不同)");
    block_prefill_impl<false>(l, x, S, pos, causal_prefix, kcache, nullptr,
                              vcache, nullptr);
  }
  // decode: x[D] 单 token, 位置 pos(k/v 写第 pos 槽), 可见 key 数 len(含自身)。
  // 注: fp16.gemv 开关只影响 generate() 内部路径; 本公开入口保持 fp32 权重路径
  // (DenseF16 升位 sgemm), 供单测做两路一致性对拍。
  void block_decode(size_t l, float* x, size_t pos, size_t len,
                    float* kcache, float* vcache) {
    if (fp16_.kv)
      throw std::runtime_error(
          "fp16.kv 模式下请走 generate() 内部路径(缓冲类型不同)");
    block_decode_impl<false, false>(l, x, pos, len, kcache, nullptr, vcache,
                                    nullptr);
  }

  // logits 投影(无 bias): y[vocab] = W·x —— 固定 fp32 路径(DenseF16 升位 sgemm)。
  // generate() 内部走 predict_layer_fp() 以尊重 fp16.gemv 开关。
  void predict_layer(const float* x, float* y) { wp_.forward(x, 1, y); }

  // 贪心采样(就地施压 repetition penalty —— golden 捕获口径为惩罚后状态, 见实现注释)。
  // history=prompt+已生成; eos_allowed=false ⇒ EOS 列不受惩罚且采样时剔除(idx<11)。
  // raw_argmax_out 非空则写入惩罚后全词表 argmax(golden tokens 口径)。返回选中 token。
  int greedy_sample(float* logits_io, const std::vector<int32_t>& history,
                    bool eos_allowed, int* raw_argmax_out);

  // E4: 复现 python infer_panel_naive sample() — repetition_penalty → top_k(15) →
  //   top_p → temperature → multinomial。rng 为外部传入, 须在多次调用间持同一个
  //   实例以保证 state 连续。返回选中 token; 贪心/采样路径 logits_ 都被就地修改
  //   (与 greedy_sample 一致: hook 抓取的是惩罚后状态)。
  int topk_sample(float* logits_io, const std::vector<int32_t>& history,
                  bool eos_allowed, int* raw_argmax_out,
                  const SamplingParams& sp, std::mt19937_64& rng);

  // E4: 公开 scratch 初始化 (正常由 generate() 自动调用; bench/单测直调 topk_sample 前需先调一次)
  void init_topk_scratch() {
    topk_idx_.assign(dims_.vocab, 0);
    topk_val_.assign(dims_.vocab, 0.f);
  }

  // 正弦位置编码单行: d 偶=sin(pos·div), 奇=cos(pos·div),
  // div[i]=exp(i·(-ln(10000)/D)) i 取偶数索引 —— 与 SinePositionalEmbedding.extend_pe 同构。
  void pe_row(float* pe, size_t pos);

 private:
  struct Layer {
    accel::DenseF16 wqkv, wout, w1, w2;
    std::vector<float> bqkv, bout, b1, b2;
    std::vector<float> n1g, n1b, n2g, n2b;
    // M1-fp16: 原始 f16 位型副本(fp16 直读 GEMV 路径用; 与升位缓冲并存)
    std::vector<uint16_t> wqkv16, wout16, w116, w216;
    // E11-2: prefill 专用的 AMX 预打包面板(wqkv/w1/w2; wout 保留 FMLAL)。
    // 非 AMX 构建下为空表(隐式 false 检查), 供 block_prefill_impl 路径选择。
    kern::AmxPanel wqkv_pa, w1_pa, w2_pa;
  };

  const rt::TensorView& need(const rt::GsvFile& f, const char* name) const;

  // 模板实现(KV16: KV cache fp16 位型存储; GEMV16: FMLAL 直读 GEMV)。
  // 未启用模式的缓冲指针传 nullptr。
  template <bool KV16>
  void block_prefill_impl(size_t l, float* x, size_t S, size_t pos,
                          size_t causal_prefix, float* kf32, uint16_t* k16,
                          float* vf32, uint16_t* v16);
  template <bool KV16, bool GEMV16>
  void block_decode_impl(size_t l, float* x, size_t pos, size_t len,
                         float* kf32, uint16_t* k16, float* vf32,
                         uint16_t* v16);
#ifdef GSV_AMX_GEMM
  // E11-5: prefill SDPA 走 AMX (Q·K^T + P·V 两次 GEMM per head, 16 头串行复用 scratch)
  // attn_out 布局 [S, D] 行主, head h 段不连续 (stride D), 内部分散写入。
  template <bool KV16>
  void sdpa_amx_prefill(size_t H, size_t S, size_t HD, float scale,
                        size_t text_len, const float* qkv_buf,
                        const float* kf32, const uint16_t* k16,
                        const float* vf32, const uint16_t* v16,
                        float* attn_out);
#endif

  // generate() 内部使用的 logits 投影(尊重 fp16.gemv)
  void predict_layer_fp(const float* x, float* y);
  // 激活舍入到 fp16 后走 FMLAL GEMV(y = W16·xh)
  void gemv_fmlal(const std::vector<uint16_t>& w16, const float* x, size_t out,
                  size_t in, float* y);

  T2SDims dims_{};
  std::vector<Layer> layers_;
  std::vector<uint16_t> text_emb_up_, audio_emb_up_; // 词表升位缓存([V*D] fp32)
  std::vector<float> text_emb_, audio_emb_;
  std::vector<float> bert_b_;
  float alpha_text_ = 1.f, alpha_audio_ = 1.f;
  accel::DenseF16 bert_proj_, wp_;
  std::vector<uint16_t> wp16_;  // ar_predict_layer 原始 f16 位型

  // M1-fp16 状态与 scratch
  Fp16Options fp16_{};
  std::vector<uint16_t> xh_, ffh_;          // 激活 fp16 化暂存([D]/[FF])
  std::vector<std::vector<uint16_t>> kc16_, vc16_;  // KV cache fp16 位型 [cap*D]
  std::vector<float> kvrow_;                // KV fp16 读出升位行暂存([D])

  // E11-2: prefill AMX 路径复用 scratch(按最大 S 一次性分配, 避免逐层重分配)
  std::vector<uint16_t> prefill_xh_;        // 激活 f16 暂存 [cap_prefill_max*D]
  size_t prefill_cap_ = 0;                  // 已分配 cap(按 S 重分配)
  bool prefill_amx_in_use_ = false;          // 构造期 amx_gemm_available() 快照

  // E11-5: prefill SDPA AMX 路径 scratch — per-head 拼接缓冲与 panel 复用
  // sdpa_scores_/sdpa_probs_: [S, S] 行主, 16 头串行复用同一片
  // sdpa_xh_: [max(S,D)] fp16 复用 (供 P fp16 转换 + pack)
  std::vector<float> sdpa_scores_;
  std::vector<float> sdpa_probs_;
  std::vector<uint16_t> sdpa_xh_;
  size_t sdpa_cap_ = 0;

  // E4: topk_sample scratch — 1025 词表, idx+val 各占 4KB, 贴 L1
  // 在 generate() 入口按 V 一次性 resize, 之后复用
  std::vector<int> topk_idx_;
  std::vector<float> topk_val_;

  // K1: prompt KV cache 复用快照
  struct PromptSnapshot {
    std::vector<int64_t> phones;
    std::vector<int64_t> prompt;
    std::vector<float> bert1024;
    size_t S = 0;
    size_t T = 0;
    size_t P = 0;
    std::vector<float> last_xy_row;  // [D]
    std::vector<std::vector<float>> kc, vc;
    std::vector<std::vector<uint16_t>> kc16, vc16;
    bool is_fp16 = false;
    bool valid = false;
  };
  PromptSnapshot prompt_snapshot_;
  bool kv_reuse_ = false;
  bool last_prefill_hit_ = false;

  // scratch(generate 内复用)
  std::vector<std::vector<float>> kc_, vc_; // 每层 KV cache [cap*D] (fp32 模式)
  size_t cap_ = 0;
  bool kv_mode_active_ = false;  // 当前缓冲所属模式(fp32/fp16 切换时重建)
  std::vector<float> xy_, qkv_, attn_, tmp_, ff_, pe_;
  std::vector<float> scores_, probs_, logits_;
  std::vector<uint32_t> pen_mark_;             // 惩罚去重标记(与 pen_stamp_ 配合)
  uint32_t pen_stamp_ = 0;
  std::vector<float> dec_qkv_, dec_xb_, dec_h_, x1_;  // decode 单 token scratch

  double last_prefill_ms_ = 0, last_decode_ms_ = 0;
};

}  // namespace gsv::ar
