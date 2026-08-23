// hubert.cpp — HuBERT 推理实现(数值纪律: fp16 权重无损升位, 计算全 fp32; 大矩阵走 sgemm)
#include "encoder/hubert.hpp"

#include "kern/kern.hpp"
#include "runtime/gsv_loader.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace gsv::encoder {

using rt::GsvFile;
using rt::TensorView;

HubertEngine::HubertEngine(const GsvFile& f) {
  // ---- config JSON(版本锁: 形状全部来自配置) ----
  const auto* mc = f.config().find("model_config");
  if (!mc) throw std::runtime_error("hubert.gsv 缺 model_config");
  auto iget = [&](const char* k) -> int {
    const auto* v = mc->find(k);
    if (!v) throw std::runtime_error(std::string("hubert config 缺 ") + k);
    return static_cast<int>(v->as_int());
  };
  hidden_ = iget("hidden_size");
  heads_ = iget("num_attention_heads");
  n_layers_ = iget("num_hidden_layers");
  inter_ = iget("intermediate_size");
  conv_pos_k_ = iget("num_conv_pos_embeddings");
  conv_pos_groups_ = iget("num_conv_pos_embedding_groups");
  ln_eps_ = mc->find("layer_norm_eps") ? mc->find("layer_norm_eps")->as_double() : 1e-5;
  {
    const auto* cd = mc->find("conv_dim");
    const auto* ck = mc->find("conv_kernel");
    const auto* cs = mc->find("conv_stride");
    if (!cd || !ck || !cs || cd->arr.size() != 7)
      throw std::runtime_error("hubert config conv_* 缺失或长度≠7");
    for (size_t i = 0; i < 7; ++i) {
      conv_dim_[i] = static_cast<int>(cd->arr[i].as_int());
      conv_kernel_[i] = static_cast<int>(ck->arr[i].as_int());
      conv_stride_[i] = static_cast<int>(cs->arr[i].as_int());
    }
  }
  const std::string feat_norm = mc->find("feat_extract_norm")
                                    ? mc->find("feat_extract_norm")->as_string()
                                    : std::string("group");
  if (feat_norm != "group") throw std::runtime_error("仅支持 feat_extract_norm=group");

  auto vec_any = [&](const char* name) {
    const TensorView* t = f.tensor(name);
    if (!t) throw std::runtime_error(std::string("hubert 缺张量: ") + name);
    std::vector<float> v(t->numel());
    if (t->has_f16())
      accel::f16_to_f32(t->data_f16_raw(), v.data(), t->numel());
    else
      std::memcpy(v.data(), t->data_f32(), t->numel() * sizeof(float));
    return v;
  };
  auto dense16 = [&](const char* name, size_t rows, size_t cols) {
    const TensorView& t = *f.tensor(name);
    if (t.numel() != rows * cols || !t.has_f16())
      throw std::runtime_error(std::string(name) + ": 形状/f16 段不符");
    Dense d;
    d.w = accel::DenseF16(t.data_f16_raw(), rows, cols);
    return d;
  };

  // ---- CNN 层(权重摊平为 im2col 布局 [out, in*k]; layer0 附 GroupNorm affine) ----
  int in_c = 1;
  for (int li = 0; li < 7; ++li) {
    const std::string p = "feature_extractor.conv_layers." + std::to_string(li) + ".";
    const TensorView& wv = *f.tensor((p + "conv.weight").c_str());
    const int out_c = conv_dim_[li], k = conv_kernel_[li];
    if (static_cast<int>(wv.dims[0]) != out_c || static_cast<int>(wv.dims[1]) != in_c ||
        static_cast<int>(wv.dims[2]) != k)
      throw std::runtime_error("hubert conv 形状与 config 不符");
    ConvL& L = convs_[li];
    L.w.resize(static_cast<size_t>(out_c) * in_c * k);
    if (wv.has_f16())
      accel::f16_to_f32(wv.data_f16_raw(), L.w.data(), L.w.size());
    else
      std::memcpy(L.w.data(), wv.data_f32(), L.w.size() * sizeof(float));
    if (li == 0) {
      L.has_gn = true;
      L.gn_g = vec_any((p + "layer_norm.weight").c_str());
      L.gn_b = vec_any((p + "layer_norm.bias").c_str());
    }
    in_c = out_c;
  }

  proj_ln_g_ = vec_any("feature_projection.layer_norm.weight");
  proj_ln_b_ = vec_any("feature_projection.layer_norm.bias");
  proj_ = dense16("feature_projection.projection.weight", hidden_, conv_dim_[6]);
  proj_.b = vec_any("feature_projection.projection.bias");

  {  // pos_conv 权重是 weight_norm 融合产物 → 只存 fp32 段(convert.py 纪律)
    const TensorView& t = *f.tensor("encoder.pos_conv_embed.conv.weight");
    const size_t expect = static_cast<size_t>(hidden_) * (hidden_ / conv_pos_groups_) *
                          conv_pos_k_;
    if (!t.has_f16() && t.src_dtype == rt::DType::F32 && t.numel() == expect)
      pos_w_.assign(t.data_f32(), t.data_f32() + t.numel());
    else
      throw std::runtime_error("pos_conv 权重应为融合 fp32 [H,H/G,K]");
  }
  pos_b_ = vec_any("encoder.pos_conv_embed.conv.bias");
  enc_ln_g_ = vec_any("encoder.layer_norm.weight");
  enc_ln_b_ = vec_any("encoder.layer_norm.bias");

  layers_.resize(n_layers_);
  for (int l = 0; l < n_layers_; ++l) {
    Layer& L = layers_[l];
    const std::string p = "encoder.layers." + std::to_string(l) + ".";
    L.q = dense16((p + "attention.q_proj.weight").c_str(), hidden_, hidden_);
    L.k = dense16((p + "attention.k_proj.weight").c_str(), hidden_, hidden_);
    L.v = dense16((p + "attention.v_proj.weight").c_str(), hidden_, hidden_);
    L.o = dense16((p + "attention.out_proj.weight").c_str(), hidden_, hidden_);
    L.f1 = dense16((p + "feed_forward.intermediate_dense.weight").c_str(), inter_, hidden_);
    L.f2 = dense16((p + "feed_forward.output_dense.weight").c_str(), hidden_, inter_);
    L.q.b = vec_any((p + "attention.q_proj.bias").c_str());
    L.k.b = vec_any((p + "attention.k_proj.bias").c_str());
    L.v.b = vec_any((p + "attention.v_proj.bias").c_str());
    L.o.b = vec_any((p + "attention.out_proj.bias").c_str());
    L.f1.b = vec_any((p + "feed_forward.intermediate_dense.bias").c_str());
    L.f2.b = vec_any((p + "feed_forward.output_dense.bias").c_str());
    L.ln1_g = vec_any((p + "layer_norm.weight").c_str());
    L.ln1_b = vec_any((p + "layer_norm.bias").c_str());
    L.ln2_g = vec_any((p + "final_layer_norm.weight").c_str());
    L.ln2_b = vec_any((p + "final_layer_norm.bias").c_str());
  }
}

void HubertEngine::gelu(float* x, size_t n) {
  for (size_t i = 0; i < n; ++i)
    x[i] = 0.5f * x[i] * (1.0f + std::erf(x[i] * static_cast<float>(M_SQRT1_2)));
}

// CNN 单层: valid 卷积。输出直接写通道主 [out_c, T](out[C',t] = Σ W'[C',·]·cols[t,·],
// 即 sgemm('N','T')), 后续 GroupNorm/下一层卷积都按通道主消费 —— 全程零转置。
void HubertEngine::conv_layer(int li, const std::vector<float>& in, int in_c, size_t in_len,
                              std::vector<float>& out, int& out_c, size_t& out_len) {
  const int k = conv_kernel_[li], s = conv_stride_[li];
  const size_t T = (in_len - static_cast<size_t>(k)) / static_cast<size_t>(s) + 1;
  out_c = conv_dim_[li];
  cols_.resize(T * static_cast<size_t>(in_c) * k);
  for (size_t t = 0; t < T; ++t)
    for (int c = 0; c < in_c; ++c)
      for (int kk = 0; kk < k; ++kk)
        cols_[t * in_c * k + c * k + kk] =
            in[static_cast<size_t>(c) * in_len + t * s + kk];
  out.resize(static_cast<size_t>(out_c) * T);
  accel::sgemm('N', 'T', out_c, static_cast<int>(T), in_c * k, 1.0f,
               convs_[li].w.data(), in_c * k, cols_.data(), static_cast<int>(in_c * k),
               0.0f, out.data(), static_cast<int>(T));
  out_len = T;
}

size_t HubertEngine::run(const float* waveform, size_t n) {
  // ---- CNN 栈 ----
  cur_.assign(waveform, waveform + n);  // [1, N] 通道×时间布局
  int c = 1;
  size_t len = n;
  for (int li = 0; li < 7; ++li) {
    size_t nl = 0;
    conv_layer(li, cur_, c, len, nxt_, c, nl);
    if (convs_[li].has_gn) {  // GroupNorm(groups=C) → 每通道独立统计(有偏方差)
      const double inv = 1.0 / static_cast<double>(nl);
      for (int ch = 0; ch < c; ++ch) {
        double mu = 0.0, var = 0.0;
        float* p = nxt_.data() + static_cast<size_t>(ch) * nl;
        for (size_t t = 0; t < nl; ++t) mu += p[t];
        mu *= inv;
        for (size_t t = 0; t < nl; ++t) var += (p[t] - mu) * (p[t] - mu);
        var *= inv;
        const double rstd = 1.0 / std::sqrt(var + ln_eps_);
        const double g = convs_[li].gn_g[static_cast<size_t>(ch)];
        const double b = convs_[li].gn_b[static_cast<size_t>(ch)];
        for (size_t t = 0; t < nl; ++t) p[t] = static_cast<float>((p[t] - mu) * rstd * g + b);
      }
    }
    gelu(nxt_.data(), nxt_.size());
    len = nl;
    cur_.swap(nxt_);
  }
  cnn_ = cur_;       // [512, T'] 通道主
  cnn_t_ = len;

  // ---- feature_projection: 转置到帧主 [T,512] → LN(512) → Linear(+bias) ----
  const size_t T = len;
  tmp_.resize(T * conv_dim_[6]);
  for (size_t t = 0; t < T; ++t)
    for (int ch = 0; ch < conv_dim_[6]; ++ch)
      tmp_[t * conv_dim_[6] + ch] = cnn_[static_cast<size_t>(ch) * T + t];
  for (size_t t = 0; t < T; ++t)
    gsv::kern::layernorm(tmp_.data() + t * conv_dim_[6], proj_ln_g_.data(),
                         proj_ln_b_.data(), tmp_.data() + t * conv_dim_[6], conv_dim_[6],
                         ln_eps_);
  x_.resize(T * hidden_);
  proj_.w.forward(tmp_.data(), T, x_.data());
  for (size_t i = 0; i < T * hidden_; ++i) x_[i] += proj_.b[i % hidden_];
  proj_o_ = x_;

  // ---- pos_conv: 分组卷积(pad=K/2 双侧, 输出 T+1 帧) → SamePad 去尾 → GELU → x += · ----
  {
    const int G = conv_pos_groups_, cg = hidden_ / G, K = conv_pos_k_;
    const size_t P = static_cast<size_t>(K) / 2;
    const size_t Tp = T + 1;  // (T+2P-K+1)
    cols_.resize(Tp * static_cast<size_t>(G) * cg * K);
    for (int g = 0; g < G; ++g)
      for (size_t t = 0; t < Tp; ++t)
        for (int j = 0; j < cg; ++j)
          for (int kk = 0; kk < K; ++kk) {
            const size_t src = t + kk;
            const float v =
                (src >= P && src < P + T)
                    ? x_[(src - P) * hidden_ + static_cast<size_t>(g) * cg + j]
                    : 0.f;
            cols_[(static_cast<size_t>(t) * G * cg + static_cast<size_t>(g) * cg + j) * K +
                  kk] = v;
          }
    pos_out_.resize(Tp * hidden_);
    for (int g = 0; g < G; ++g) {
      const float* wg = pos_w_.data() + static_cast<size_t>(g) * cg * cg * K;
      accel::sgemm('N', 'T', static_cast<int>(Tp), cg, cg * K, 1.0f,
                   cols_.data() + static_cast<size_t>(g) * cg * K,
                   static_cast<int>(G) * cg * K, wg, static_cast<int>(cg * K), 0.0f,
                   pos_out_.data() + static_cast<size_t>(g) * cg, hidden_);
    }
    // conv bias(逐输出通道)
    for (size_t t = 0; t < Tp; ++t)
      for (int i = 0; i < hidden_; ++i) pos_out_[t * hidden_ + i] += pos_b_[i];
    gelu(pos_out_.data(), pos_out_.size());          // activation 在 SamePad 之后
    for (size_t t = 0; t < T; ++t)                   // SamePadLayer: 去掉最后一帧
      for (int i = 0; i < hidden_; ++i) x_[t * hidden_ + i] += pos_out_[t * hidden_ + i];
  }
  cap_pos_.assign(pos_out_.begin(), pos_out_.begin() + static_cast<long>(T * hidden_));

  // ---- encoder.layer_norm(dropout=identity) ----
  for (size_t t = 0; t < T; ++t)
    gsv::kern::layernorm(x_.data() + t * hidden_, enc_ln_g_.data(), enc_ln_b_.data(),
                         x_.data() + t * hidden_, hidden_, ln_eps_);
  cap_encln_ = x_;

  // ---- 12 × post-LN 层 ----
  const size_t hd = hidden_ / heads_;
  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
  smax_.resize(T);
  for (int l = 0; l < n_layers_; ++l) {
    const Layer& L = layers_[static_cast<size_t>(l)];
    // fused QKV: qkv[T,3H]
    qkv_.resize(T * 3 * hidden_);
    float* qp = qkv_.data();
    float* kp = qp + T * hidden_;
    float* vp = kp + T * hidden_;
    L.q.w.forward(x_.data(), T, qp);
    L.k.w.forward(x_.data(), T, kp);
    L.v.w.forward(x_.data(), T, vp);
    for (size_t t = 0; t < T; ++t) {
      for (int i = 0; i < hidden_; ++i) {
        qp[t * hidden_ + i] += L.q.b[static_cast<size_t>(i)];
        kp[t * hidden_ + i] += L.k.b[static_cast<size_t>(i)];
        vp[t * hidden_ + i] += L.v.b[static_cast<size_t>(i)];
      }
    }
    // SDPA(无掩码): 每 head 独立
    att_.assign(T * hidden_, 0.f);
    for (int h = 0; h < heads_; ++h) {
      for (size_t q = 0; q < T; ++q) {
        const float* qv = qp + q * hidden_ + h * hd;
        for (size_t kk = 0; kk < T; ++kk) {
          const float* kv = kp + kk * hidden_ + h * hd;
          float dot = 0.f;
          for (size_t e = 0; e < hd; ++e) dot += qv[e] * kv[e];
          smax_[kk] = dot * scale;
        }
        gsv::kern::softmax(smax_.data(), smax_.data(), T);
        float* ov = att_.data() + q * hidden_ + h * hd;
        for (size_t kk = 0; kk < T; ++kk) {
          const float p = smax_[kk];
          const float* vv = vp + kk * hidden_ + h * hd;
          for (size_t e = 0; e < hd; ++e) ov[e] += p * vv[e];
        }
      }
    }
    // out proj + 残差 + post-LN
    resid_ = x_;                                   // attn_residual
    L.o.w.forward(att_.data(), T, x_.data());
    for (size_t i = 0; i < T * hidden_; ++i) x_[i] += L.o.b[i % hidden_];
    if (l == 0) cap_l0attn_.assign(x_.begin(), x_.begin() + static_cast<long>(T * hidden_));
    for (size_t i = 0; i < T * hidden_; ++i) x_[i] += resid_[i];
    for (size_t t = 0; t < T; ++t)
      gsv::kern::layernorm(x_.data() + t * hidden_, L.ln1_g.data(), L.ln1_b.data(),
                           x_.data() + t * hidden_, hidden_, ln_eps_);
    if (l == 0) cap_l0ln1_ = x_;
    // FFN(GELU) + 残差 + final LN
    ff_.resize(T * inter_);
    L.f1.w.forward(x_.data(), T, ff_.data());
    for (size_t i = 0; i < T * inter_; ++i) ff_[i] += L.f1.b[i % inter_];
    gelu(ff_.data(), ff_.size());
    resid_ = x_;
    L.f2.w.forward(ff_.data(), T, x_.data());
    for (size_t i = 0; i < T * hidden_; ++i) x_[i] += L.f2.b[i % hidden_];
    if (l == 0) cap_l0ffn_.assign(x_.begin(), x_.begin() + static_cast<long>(T * hidden_));
    for (size_t i = 0; i < T * hidden_; ++i) x_[i] += resid_[i];
    for (size_t t = 0; t < T; ++t)
      gsv::kern::layernorm(x_.data() + t * hidden_, L.ln2_g.data(), L.ln2_b.data(),
                           x_.data() + t * hidden_, hidden_, ln_eps_);
    if (l == 0) {
      cap_l0ln2_ = x_;
      l0_ = x_;
    }
  }

  last_ = x_;
  return T;
}

}  // namespace gsv::encoder
