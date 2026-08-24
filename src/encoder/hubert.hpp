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

#include <cstddef>
#include <vector>

namespace gsv::rt {
class GsvFile;
}

namespace gsv::encoder {

namespace accel = kern::accel;

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
  void conv_layer(int li, const std::vector<float>& in, int in_c, size_t in_len,
                  std::vector<float>& out, int& out_c, size_t& out_len);

  struct Dense {
    accel::DenseF16 w;
    std::vector<float> b;
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

  struct ConvL {
    std::vector<float> w;  // [out, in*k] im2col 布局
    bool has_gn = false;
    std::vector<float> gn_g, gn_b;
  };
  ConvL convs_[7];
  Dense proj_;
  std::vector<float> proj_ln_g_, proj_ln_b_;
  std::vector<float> pos_w_;  // [H, H/G, K] fp32(weight_norm 融合产物)
  std::vector<float> pos_b_;  // [H] fp32(conv bias)
  std::vector<float> enc_ln_g_, enc_ln_b_;

  // scratch(跨 run 复用)
  std::vector<float> cur_, nxt_, cnn_, cols_, tmp_, x_, pos_out_, qkv_, att_,
      ff_, resid_, smax_;
  std::vector<float> proj_o_, l0_, last_;
  size_t cnn_t_ = 0;
  // 对拍捕获
  std::vector<float> cap_pos_, cap_encln_, cap_l0attn_, cap_l0ln1_, cap_l0ffn_,
      cap_l0ln2_;
};

}  // namespace gsv::encoder
