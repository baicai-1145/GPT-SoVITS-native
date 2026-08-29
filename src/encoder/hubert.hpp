// hubert.hpp — chinese-hubert-base 推理引擎(C1), 对照 CPUFast feature_extractor/cnhubert.py
// (= transformers HubertModel, feat_extract_norm="group", do_stable_layer_norm=false)
//
// 前向路径:
//   raw wav [N] → CNN 7 层(k10s5 + k3s2×3 + k2s2×2, 无 bias; 首层 GroupNorm(=逐通道)
//   +GELU, 其余仅 GELU) → [T',512](帧主) → LayerNorm+Linear(512→768) →
//   x += GELU(pos_conv(x))(groups16,k128,pad64,weight_norm 已融合为 fp32 常权,
//   SamePad 去尾帧) → LayerNorm → 12 × post-LN 层(QKV/OUT 全带 bias,
//   scale=head_dim^-0.5; FFN 3072 GELU) → last_hidden_state [T,768]
//
// 运行时输入配方(TTS.py::_set_prompt_semantic): wav16k 后拼接 9600 个零
// (zero_wav = sampling_rate(32000)·0.3 的历史遗留口径), 由调用方完成; 引擎只吃给定波形。
// eval 语义: dropout 全 identity, 不做 spec_augment/masking。
//
// 中间量对拍: run() 结束后经 cnn_out()/proj_out()/l0_out()/out() 取只读视图
// (成员缓冲持有, 下次 run 前有效)。
#pragma once

#include "kern/accel.hpp"
#include "kern/gemv_fmlal.hpp"
#if defined(GSV_AMX_GEMM)
#include "kern/gemm_f16_amx.hpp"
#endif

#include <cstddef>
#include <vector>

namespace gsv::rt {
class GsvFile;
}

namespace gsv::encoder {

namespace accel = kern::accel;

// E12: 进程级 HuBERT AMX 使能(--amx-enc; 装载前设置)。分流模式同 E8:
// all = QKV/OUT/W1/W2 全切; ffn = 仅 FFN; none = 全 FMLAL。
bool& amx_hubert_enabled();

enum class AmxEncMode { kAll, kFfnOnly, kNone };
AmxEncMode& amx_hubert_mode();

class HubertEngine {
 public:
  explicit HubertEngine(const rt::GsvFile& f);

  // 波形 → last_hidden_state 写入内部缓冲; 返回帧数 T。
  // 结果视图: out()=[T,768], 另有 proj_out()/l0_out()/cnn_out()。
  size_t run(const float* waveform, size_t n);

  const std::vector<float>& out() const { return last_; }        // [T,768]
  const std::vector<float>& l0_out() const { return l0_; }       // [T,768]
  const std::vector<float>& proj_out() const { return proj_o_; } // [T,768]
  const std::vector<float>& cnn_out() const { return cnn_; }     // [512,T'] 通道主
  size_t cnn_frames() const { return cnn_t_; }

  // 细粒度对拍视图(与 CPUFast hook 同语义; 仅在 run 后有效)
  const std::vector<float>& pos_out_view() const { return cap_pos_; }    // [T,768] GELU+去尾后
  const std::vector<float>& encln_out_view() const { return cap_encln_; }
  const std::vector<float>& l0_attn_out_view() const { return cap_l0attn_; }
  const std::vector<float>& l0_ln1_out_view() const { return cap_l0ln1_; }
  const std::vector<float>& l0_ffn_out_view() const { return cap_l0ffn_; }
  const std::vector<float>& l0_ln2_out_view() const { return cap_l0ln2_; }

 private:
  static void gelu(float* x, size_t n);
#if defined(GSV_AMX_GEMM)
  // T12: AMX 旗后 SDPA 批量化(batched sgemm; 数值 = E12/cos+codes 谱系)
  void sdpa_amx_sgemm(float* qp, float* kp, float* vp, size_t T, size_t hd,
                      float scale, float* att_out);
#endif
  void conv_layer(int li, const std::vector<float>& in, int in_c, size_t in_len,
                  std::vector<float>& out, int& out_c, size_t& out_len);

  // E12: 进程级 AMX 使能(--amx-enc 开关; 装载前设置), 分流门槛同 E8 bert 口径。
  friend bool& amx_hubert_enabled();

  struct Dense {
    accel::DenseF16 w;
    std::vector<float> b;
#if defined(GSV_AMX_GEMM)
    kern::AmxPanel w_panel;      // E12: 装载期预打包(生命周期 = GsvFile 同域)
    bool w_panel_ready = false;
#endif
  };

  struct Layer {
    Dense q, k, v, o, f1, f2;
    std::vector<float> ln1_g, ln1_b, ln2_g, ln2_b;
  };
  std::vector<Layer> layers_;

  int hidden_ = 768, heads_ = 12, n_layers_ = 12, inter_ = 3072;
  int conv_dim_[7]{}, conv_kernel_[7]{}, conv_stride_[7]{};
  int conv_pos_k_ = 128, conv_pos_groups_ = 16;
  double ln_eps_ = 1e-5;

#if defined(GSV_AMX_GEMM)
  bool layer_qkv_batch_ready(const Layer& L) const;  // 定义在 .cpp (Layer 完整型后)
  // E12: AMX dense 前向 y[T,out] = x·Wᵀ+b; 激活 panel 缓冲由调用方复用。
  void dense_amx(const Dense& d, const float* x, size_t T, float* y,
                 std::vector<uint8_t>& act_scratch, size_t in_dim,
                 const std::vector<float>& bias) const;
#endif

  struct ConvL {
    const uint16_t* w16 = nullptr;  // [out, in*k] fp16 直读
    std::vector<float> w;           // 无 f16 段回退(不适用; 保留类型以防)
    bool has_gn = false;
    std::vector<float> gn_g, gn_b;
#if defined(GSV_AMX_GEMM)
    kern::AmxPanel conv_panel;      // T13: 装载期预打包(旗后)
    bool conv_panel_ready = false;
#endif
  };

  ConvL convs_[7];
  Dense proj_;
  std::vector<float> proj_ln_g_, proj_ln_b_;
  std::vector<float> pos_w_;  // [H, H/G, K] fp32(weight_norm 融合产物, 源仅存 fp32 段)
  std::vector<uint16_t> pos_w16_;  // 融合产物的一次性 fp16 量化副本(FMLAL 消费, 0.6MB)
  std::vector<float> pos_b_;  // [H] fp32(conv bias)
  std::vector<float> enc_ln_g_, enc_ln_b_;

  // scratch(跨 run 复用)
  std::vector<float> cur_, nxt_, cnn_, cols_, tmp_, x_, pos_out_, qkv_, att_,
      ff_, resid_, smax_;
  std::vector<float> sc_;   // T11: SDPA 全头分数缓冲 [heads·T·T]
  std::vector<float> ovh_;  // T11: 单头 PV 输出暂存 [T·hd]
  // T12: AMX 旗后 SDPA(batched sgemm) 滚动缓冲
  std::vector<float> sdpasc_, sdpa_qg_, sdpa_kg_, sdpa_vtg_;
  std::vector<float> proj_o_, l0_, last_;
  size_t cnn_t_ = 0;
  std::vector<uint16_t> xh_;      // fp16 激活暂存(DenseF16 view FMLAL 前向复用)
  std::vector<uint16_t> cols16_;  // fp16 im2col 暂存(CNN/pos_conv FMLAL 用)
#if defined(GSV_AMX_GEMM)
  std::vector<uint8_t> hub_act_scratch_;  // E12: AMX 激活 panel 缓冲(容量复用)
  std::vector<uint8_t> hub_conv_act_;     // T13: CNN 激活 B 面板(容量复用)
#endif
  // 对拍捕获
  std::vector<float> cap_pos_, cap_encln_, cap_l0attn_, cap_l0ln1_, cap_l0ffn_,
      cap_l0ln2_;

 public:
  double sdpa_ms() const { return sdpa_ms_; }
  bool& sdpa_timing_enabled() { return sdpaTim; }

 private:
  double sdpa_ms_ = 0.0;
  bool sdpaTim = false;
};

}  // namespace gsv::encoder
