// pipeline_condition.hpp — C2: v2ProPlus decode-condition 构建 (ge / ge_text)。
//
// 复刻 CPUFast 口径:
//   _get_ref_spec:      48k→32k 重采样(调用方) → 峰值归一(audio /= min(2,maxx))
//                       → spectrogram_torch(n_fft=2048, hop=640, win=2048,
//                         center=False, 反射填充 704×2) → sqrt(re²+im²+1e-8)
//   build_decode_condition:
//     ge     = prelu(ref_enc(spec[:, :704]) + sv_emb(sv_emb_20480))
//     ge_text= ge_to512(ge^T)^T            (Linear 1024→512)
//   其中 ref_enc = MelStyleEncoder(704, hidden=128, out=1024, k=5, head=2):
//     spectral: Linear(704→128)+Mish → Linear(128→128)+Mish   (Dropout=恒等)
//     temporal: 2 × Conv1dGLU(k=5, pad=2, bias=True):
//               Conv(128→256) → split(128,128) → x1·σ(x2) + 残差
//     slf_attn: MultiHeadAttention(head=2, d=128, temperature=√d_model)
//     fc:       Linear(128→1024) → 时间维平均池化
//
// 本层属于任务卡授权的 pipeline 编排域; 若后续要挪进 src/sovits/ 需决策者裁决。
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "runtime/gsv_loader.hpp"

namespace gsv::rt::pipeline {

struct DecodeCondition {
  std::vector<float> ge;        // [1024]
  std::vector<float> ge_text;   // [512]
};

class ConditionBuilder {
 public:
  // 从 sovits .gsv 读 ref_enc.* / sv_emb / ge_to512 / prelu 权重
  void load(const rt::GsvFile& f);

  static constexpr size_t kSpecBins = 1025;
  static constexpr size_t kCondIn = 704;
  static constexpr size_t kHidden = 128;
  static constexpr size_t kGeDim = 1024;
  static constexpr size_t kGeTextDim = 512;
  static constexpr int kFft = 2048, kHop = 640, kWin = 2048;

  // 线性谱: audio32k 已是归一后单声道 32 kHz PCM。返回 [1025][T] 行主。
  static void spectrogram(const float* audio, size_t n,
                          std::vector<float>& spec, size_t& frames);

  // 完整条件链。svEmb20480 来自 SvEngine::forward3 的 16 kHz 归一音频。
  void compute(const float* audio32k, size_t n, const float* svEmb20480,
               DecodeCondition* out) const;

  // T8: 批量化 ref_enc 开关(--amx-enc 门控, Pipeline 初始化时接线)。
  //   默认 false = 原标量实现逐位口径(E13-MIX 前基线); true = T6 批量 sgemm 版。
  void setAmxEnc(bool on) { amxEnc_ = on; }

  // 仅 ref_enc 主干 (供单测对 golden ref_enc 输出)
  void refEnc(const float* condIn, size_t T,
              std::vector<float>& pooled /*[1024]*/) const;

 private:
  // LinearNorm: W[out,in] 行主 + bias
  struct Lin {
    std::vector<float> W, b;
    size_t in = 0, out = 0;
    void run(const float* x, float* y) const;  // y = W·x + b
  };

  Lin sp0_, sp3_;                          // spectral 两层
  std::vector<float> tc_[2], tb_[2];       // temporal conv 权重 [256,128,5]/bias
  Lin attWq_, attWk_, attWv_, attFc_;      // slf_attn (bias 可能存在)
  Lin fc_;                                 // [1024,128]
  Lin svProj_, geTo512_;                   // sv_emb [1024,20480]; ge_to512 [512,1024]
  std::vector<float> preluSlope_;          // [1024]

  static inline float mish(float x);
  void conv1dGlu(const float* in /*[128,T]*/, size_t T, int layer,
                 float* out /*[128,T]*/) const;

  // T8: 双实现共存 — refEnc 按 amxEnc_ 派发。
  //   refEncScalar/conv1dGlu: E13-MIX 之前基线原样代码(默认路径位级红线);
  //   refEncBatched/conv1dGluBatched: T6 批量化(--amx-enc 时走此路径, mel 门验议)。
  void refEncScalar(const float* condIn, size_t T,
                    std::vector<float>& pooled) const;
  void refEncBatched(const float* condIn, size_t T,
                     std::vector<float>& pooled) const;
  void conv1dGluBatched(const float* in /*[128,T]*/, size_t T, int layer,
                        float* out /*[128,T]*/) const;
  bool amxEnc_ = false;
};

}  // namespace gsv::rt::pipeline
