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
#include "kern/kern.hpp"
#include "runtime/gsv_loader.hpp"

#include <cstddef>
#include <cstdint>
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

  // 从 .gsv 加载全部权重; 维度取自 config JSON(版本锁纪律, 不硬编码形状)。
  explicit T2SEngine(const rt::GsvFile& f);
  const T2SDims& dims() const { return dims_; }

  // 全链路: prefill + 贪心 decode。phones[T] 文本音素, prompt[P] 音频语义,
  // bert1024 为 [T,bert_dim] 行主(即 pairs 的 bert_feat_1024 摊平)。
  GenResult generate(const int64_t* phones, size_t T,
                     const int64_t* prompt, size_t P,
                     const float* bert1024,
                     size_t max_steps = kMaxDecodeSteps,
                     GenDebug* dbg = nullptr);

  double last_prefill_ms() const { return last_prefill_ms_; }
  double last_decode_ms() const { return last_decode_ms_; }

  // ---- 单层前向原语(公开供单测做 prefill/decode 一致性对拍) ----
  // KV cache 布局(自定, 见 README 注释): 每层两个独立缓冲 kcache/vcache,
  // 行主 token-major: cache[tok*D + head*HD + e], HD=D/heads。fp32 存储。
  //
  // prefill: x[S,D] 就地更新为层输出; 该层 k/v 追加写入 cache 第 pos..pos+S-1 槽。
  // causal_prefix>0 时前 causal_prefix 个 query 只允许看见同数量的 key(文本前缀段),
  // 其余 query 标准因果 —— 等价 torch xy_attn_mask [文本行只见文本; 音频行因果]。
  void block_prefill(size_t l, float* x, size_t S, size_t pos,
                     size_t causal_prefix, float* kcache, float* vcache);
  // decode: x[D] 单 token, 位置 pos(k/v 写第 pos 槽), 可见 key 数 len(含自身)。
  void block_decode(size_t l, float* x, size_t pos, size_t len,
                    float* kcache, float* vcache);

  // logits 投影(无 bias): y[vocab] = W·x
  void predict_layer(const float* x, float* y) { wp_.forward(x, 1, y); }

  // 贪心采样(就地施压 repetition penalty —— golden 捕获口径为惩罚后状态, 见实现注释)。
  // history=prompt+已生成; eos_allowed=false ⇒ EOS 列不受惩罚且采样时剔除(idx<11)。
  // raw_argmax_out 非空则写入惩罚后全词表 argmax(golden tokens 口径)。返回选中 token。
  int greedy_sample(float* logits_io, const std::vector<int32_t>& history,
                    bool eos_allowed, int* raw_argmax_out);

  // 正弦位置编码单行: d 偶=sin(pos·div), 奇=cos(pos·div),
  // div[i]=exp(i·(-ln(10000)/D)) i 取偶数索引 —— 与 SinePositionalEmbedding.extend_pe 同构。
  void pe_row(float* pe, size_t pos);

 private:
  struct Layer {
    accel::DenseF16 wqkv, wout, w1, w2;
    std::vector<float> bqkv, bout, b1, b2;
    std::vector<float> n1g, n1b, n2g, n2b;
  };

  const rt::TensorView& need(const rt::GsvFile& f, const char* name) const;

  T2SDims dims_{};
  std::vector<Layer> layers_;
  std::vector<uint16_t> text_emb_up_, audio_emb_up_; // 词表升位缓存([V*D] fp32)
  std::vector<float> text_emb_, audio_emb_;
  std::vector<float> bert_b_;
  float alpha_text_ = 1.f, alpha_audio_ = 1.f;
  accel::DenseF16 bert_proj_, wp_;

  // scratch(generate 内复用)
  std::vector<std::vector<float>> kc_, vc_; // 每层 KV cache [cap*D]
  size_t cap_ = 0;
  std::vector<float> xy_, qkv_, attn_, tmp_, ff_, pe_;
  std::vector<float> scores_, probs_, logits_;
  std::vector<uint32_t> pen_mark_;             // 惩罚去重标记(与 pen_stamp_ 配合)
  uint32_t pen_stamp_ = 0;
  std::vector<float> dec_qkv_, dec_xb_, dec_h_, x1_;  // decode 单 token scratch

  double last_prefill_ms_ = 0, last_decode_ms_ = 0;
};

}  // namespace gsv::ar
