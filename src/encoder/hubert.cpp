// hubert.cpp — HuBERT 推理实现(数值纪律: fp16 权重无损升位, 计算全 fp32; 大矩阵走 sgemm)
#include "encoder/hubert.hpp"

#include "kern/kern.hpp"
#include "runtime/gsv_loader.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace gsv::encoder {

using rt::GsvFile;
using rt::TensorView;

// E12: 进程级 AMX 开关与分流模式(与 E8 bert 同口径)
bool& amx_hubert_enabled() {
  static bool b = false;
  return b;
}
AmxEncMode& amx_hubert_mode() {
  static AmxEncMode m = [] {
    const char* e = std::getenv("GSV_AMX_HUBERT_MODE");
    if (!e) return AmxEncMode::kAll;
    if (e[0] == 'f') return AmxEncMode::kFfnOnly;
    if (e[0] == 'n') return AmxEncMode::kNone;
    return AmxEncMode::kAll;
  }();
  return m;
}

// E12 分流门槛(与 E8 bert 一致): T≥48(tile 半填充以上)且 K≥256
constexpr size_t kAmxEncMinRows = 48;
constexpr size_t kAmxEncMinK = 256;

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
    d.w = accel::DenseF16::view_f16(t.data_f16_raw(), rows, cols);  // 零拷贝 FMLAL
#if defined(GSV_AMX_GEMM)
    // E12: 装载期预打包权重 panel(生命周期与 GsvFile 映射同域); 小 K 不打。
    if (amx_hubert_enabled() && amx_hubert_mode() == AmxEncMode::kAll &&
        kern::amx_gemm_available() && cols >= kAmxEncMinK &&
        rows >= kAmxEncMinK) {
      d.w_panel.rows = rows;
      d.w_panel.K = cols;
      kern::amx_pack_into(t.data_f16_raw(), rows, cols, d.w_panel.buf);
      d.w_panel_ready = true;
    }
#endif
    return d;
  };

  // ---- CNN 层(权重 fp16 直读 [out, in*k]; layer0 附 GroupNorm affine) ----
  int in_c = 1;
  for (int li = 0; li < 7; ++li) {
    const std::string p = "feature_extractor.conv_layers." + std::to_string(li) + ".";
    const TensorView& wv = *f.tensor((p + "conv.weight").c_str());
    const int out_c = conv_dim_[li], k = conv_kernel_[li];
    if (static_cast<int>(wv.dims[0]) != out_c || static_cast<int>(wv.dims[1]) != in_c ||
        static_cast<int>(wv.dims[2]) != k)
      throw std::runtime_error("hubert conv 形状与 config 不符");
    ConvL& L = convs_[li];
    if (!wv.has_f16())
      throw std::runtime_error("hubert conv 权重缺 f16 段: " + p);
    L.w16 = wv.data_f16_raw();  // fp16 直读(FMLAL)
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

#if defined(GSV_AMX_GEMM)
namespace {

// E12: 与 E8 bert::cast_pack_B_f32_to_panel 同构的 fp32 激活 → panel 直写
// (含尾 tile 补零; 热路径 reserve+resize 免重复 memset)。
void cast_pack_B_f32(const float* x_fp32, size_t T, size_t K,
                     std::vector<uint8_t>& out_buf) {
  const size_t nt = (T + 31) / 32;
  const size_t need = nt * K * 64 + 64;
  if (out_buf.capacity() < need) out_buf.reserve(need);
  out_buf.resize(need);
  const uintptr_t p = (uintptr_t)out_buf.data();
  uint8_t* dst = out_buf.data() + ((64 - (p & 63)) & 63);
  if (const size_t tr_last = T - (nt - 1) * 32; tr_last < 32) {
    uint8_t* d_tail = dst + (nt - 1) * K * 64;
    const size_t zoff = tr_last * 2;
    for (size_t k = 0; k < K; ++k)
      std::memset(d_tail + k * 64 + zoff, 0, 64 - zoff);
  }
  for (size_t t = 0; t < nt; ++t) {
    const size_t r0 = t * 32;
    const size_t tr = std::min<size_t>(32, T - r0);
    uint8_t* d = dst + t * K * 64;
    if (tr == 32) {
      for (size_t r = 0; r < 32; ++r) {
        const float* src = x_fp32 + (r0 + r) * K;
        uint16_t* col = reinterpret_cast<uint16_t*>(d + r * 2);
        size_t k = 0;
        for (; k + 8 <= K; k += 8) {
          // 8k 跨步写: dst[k*64+r*2] 连续相邻不重叠 — 标量足够(打包占 GEMM <5%)
          for (int j = 0; j < 8; ++j)
            col[(k + j) * 32] = kern::f32_to_f16_scalar(src[k + j]);
        }
        for (; k < K; ++k) col[k * 32] = kern::f32_to_f16_scalar(src[k]);
      }
    } else {
      for (size_t r = 0; r < tr; ++r) {
        const float* src = x_fp32 + (r0 + r) * K;
        uint16_t* col = reinterpret_cast<uint16_t*>(d + r * 2);
        for (size_t k = 0; k < K; ++k) col[k * 32] = kern::f32_to_f16_scalar(src[k]);
      }
      // 尾行已在顶部统一补零; 此处无需再清
    }
  }
}

}  // namespace
#endif  // GSV_AMX_GEMM

void HubertEngine::gelu(float* x, size_t n) {
  for (size_t i = 0; i < n; ++i)
    x[i] = 0.5f * x[i] * (1.0f + std::erf(x[i] * static_cast<float>(M_SQRT1_2)));
}

#if defined(GSV_AMX_GEMM)
// E12: AMX dense — y[T,out] = x[T,in]·W[out,in]ᵀ + bias(与 FMLAL 同口径逐元素加)
// 注: pb.buf 用 swap 而非 move — amx_batch_run 返回后面板寿命结束,
// 缓冲容量经 pb 析构流回调用方(act_scratch), 容量跨调用复用免反复 malloc。
void HubertEngine::dense_amx(const Dense& d, const float* x, size_t T, float* y,
                             std::vector<uint8_t>& act_scratch, size_t in_dim,
                             const std::vector<float>& bias) const {
  cast_pack_B_f32(x, T, in_dim, act_scratch);
  kern::AmxPanel pb;
  pb.rows = T;
  pb.K = in_dim;
  // 契约: buf 首地址需 64B 对齐 — cast_pack_B 已保证 data() 对齐基址。
  pb.buf.swap(act_scratch);
  kern::AmxBatchNode nd;
  nd.phase = 0;
  nd.pa = &pb;          // 激活侧 [T,K]
  nd.pb = &d.w_panel;   // 权重侧 [out,K]
  nd.c = y;
  nd.M = T;
  nd.N = d.w_panel.rows;
  kern::amx_batch_run(&nd, 1);
  pb.buf.swap(act_scratch);  // 收回容量
  if (!bias.empty()) {
    const size_t out = d.w_panel.rows;
    for (size_t t = 0; t < T; ++t)
      for (size_t o = 0; o < out; ++o) y[t * out + o] += bias[o];
  }
}

bool HubertEngine::layer_qkv_batch_ready(const Layer& L) const {
  return L.q.w_panel_ready && L.k.w_panel_ready && L.v.w_panel_ready;
}
#endif

// CNN 单层: valid 卷积(fp16 路径)。im2col 量化到 fp16 cols16_, 权重 fp16 直读,
// gemm_f16x_fmlal: out[out_c, T] = W[out_c, in*k]·colsᵀ (FMLAL 融合 fp32 累加,
// 通道主输出零转置, 后续 GroupNorm/下一层卷积直接消费)。
void HubertEngine::conv_layer(int li, const std::vector<float>& in, int in_c, size_t in_len,
                              std::vector<float>& out, int& out_c, size_t& out_len) {
  // E13 探针: GSV_BERT_CONV_TIMING=1 时逐 CNN 层计时与形状收集(只计时不改行为)
  using clk = std::chrono::steady_clock;
  static thread_local clk::time_point tp_conv0;
  const bool convTim = std::getenv("GSV_BERT_CONV_TIMING") != nullptr;
  if (convTim) tp_conv0 = clk::now();
  const int k = conv_kernel_[li], s = conv_stride_[li];
  const size_t T = (in_len - static_cast<size_t>(k)) / static_cast<size_t>(s) + 1;
  out_c = conv_dim_[li];
  const size_t KK = static_cast<size_t>(in_c) * k;
  cols16_.resize(T * KK);
  for (size_t t = 0; t < T; ++t)
    for (int c = 0; c < in_c; ++c)
      for (int kk = 0; kk < k; ++kk) {
        const __fp16 h = static_cast<__fp16>(
            in[static_cast<size_t>(c) * in_len + t * s + kk]);
        std::memcpy(cols16_.data() + t * KK + static_cast<size_t>(c) * k + kk, &h,
                    sizeof h);
      }
  out.resize(static_cast<size_t>(out_c) * T);
  kern::gemm_f16x_fmlal(convs_[li].w16, cols16_.data(), out.data(),
                        static_cast<size_t>(out_c), T, KK);
  out_len = T;
  if (convTim) {
    const double gemm_ms = std::chrono::duration<double, std::milli>(
                               clk::now() - tp_conv0).count();
    ConvTrace t;
    t.in_c = in_c;
    t.in_len = static_cast<int>(in_len);
    t.out_c = out_c;
    t.k = k;
    t.s = s;
    t.out_len = static_cast<int>(T);
    t.ms = gemm_ms;
    conv_traces_.push_back(t);
  }
}

size_t HubertEngine::run(const float* waveform, size_t n) {
  // E13 探针: GSV_BERT_CONV_TIMING=1 时清空上一轮 trace, GSV_HUBERT_SDPA_TIMING=1 计 SDPA
  using clk = std::chrono::steady_clock;
  conv_traces_.clear();
  sdpa_ms_ = 0.0;
  sdpaTim = std::getenv("GSV_HUBERT_SDPA_TIMING") != nullptr;
  const auto tp_run0 = clk::now();
  // ---- CNN 栈 ----
  const auto tp_cnn_start = clk::now();
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
  const auto tp_cnn1 = clk::now();

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
#if defined(GSV_AMX_GEMM)
  if (proj_.w_panel_ready && T >= kAmxEncMinRows) {
    dense_amx(proj_, tmp_.data(), T, x_.data(), hub_act_scratch_, conv_dim_[6],
              proj_.b);
  } else
#endif
  {
    proj_.w.forward(tmp_.data(), T, x_.data(), xh_);
    for (size_t i = 0; i < T * hidden_; ++i) x_[i] += proj_.b[i % hidden_];
  }
  proj_o_ = x_;

  // ---- pos_conv: 分组卷积(pad=K/2 双侧, 输出 T+1 帧) → SamePad 去尾 → GELU → x += · ----
  // fp16 路径: pos_w_ 是 weight_norm 融合产物(仅存 fp32 段) → 量化到 fp16 后 FMLAL。
  {
    const int G = conv_pos_groups_, cg = hidden_ / G, K = conv_pos_k_;
    const size_t P = static_cast<size_t>(K) / 2;
    const size_t Tp = T + 1;  // (T+2P-K+1)
    const size_t KK = static_cast<size_t>(cg) * K;
    // 按组连续布局: 组 g 的 [Tp,KK] 块紧排(gemm_f16x_fmlal 的 B 侧行 stride=KK)
    cols16_.resize(static_cast<size_t>(G) * Tp * KK);
    pos_w16_.resize(static_cast<size_t>(hidden_) * cg * K);  // 融合产物→fp16 一次性量化
    kern::f32_to_f16(pos_w_.data(), pos_w16_.data(), pos_w16_.size());
    for (int g = 0; g < G; ++g)
      for (size_t t = 0; t < Tp; ++t)
        for (int j = 0; j < cg; ++j)
          for (int kk = 0; kk < K; ++kk) {
            const size_t src = t + kk;
            const float v =
                (src >= P && src < P + T)
                    ? x_[(src - P) * hidden_ + static_cast<size_t>(g) * cg + j]
                    : 0.f;
            const __fp16 h = static_cast<__fp16>(v);
            std::memcpy(cols16_.data() +
                            (static_cast<size_t>(g) * Tp + t) * KK +
                             static_cast<size_t>(j) * K + kk,
                        &h, sizeof h);
          }
    pos_out_.resize(Tp * hidden_);
    tmp_.resize(static_cast<size_t>(cg) * Tp);  // [cg, Tp] 通道主 gemm 直出中转
    for (int g = 0; g < G; ++g) {
      const uint16_t* wg = pos_w16_.data() + static_cast<size_t>(g) * cg * cg * K;
      kern::gemm_f16x_fmlal(wg, cols16_.data() + static_cast<size_t>(g) * Tp * KK,
                            tmp_.data(), cg, Tp, KK);
      // 通道主 [cg,Tp] → 行主 [Tp,hidden] 槽位(消费端布局)
      for (size_t t = 0; t < Tp; ++t)
        for (int i = 0; i < cg; ++i)
          pos_out_[t * hidden_ + static_cast<size_t>(g) * cg + i] =
              tmp_[static_cast<size_t>(i) * Tp + t];
    }
    // conv bias(逐输出通道)
    for (size_t t = 0; t < Tp; ++t)
      for (int i = 0; i < hidden_; ++i) pos_out_[t * hidden_ + i] += pos_b_[i];
    gelu(pos_out_.data(), pos_out_.size());          // activation 在 SamePad 之后
    for (size_t t = 0; t < T; ++t)                   // SamePadLayer: 去掉最后一帧
      for (int i = 0; i < hidden_; ++i) x_[t * hidden_ + i] += pos_out_[t * hidden_ + i];
  }
  const auto tp_pos1 = clk::now();
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
#if defined(GSV_AMX_GEMM)
    if (layer_qkv_batch_ready(L) && T >= kAmxEncMinRows) {
      // E12: QKV 同批三联派发(共享激活 panel, 省两次池往返 — E8 同配方)
      cast_pack_B_f32(x_.data(), T, size_t(hidden_), hub_act_scratch_);
      kern::AmxPanel pb;
      pb.rows = T;
      pb.K = size_t(hidden_);
      pb.buf.swap(hub_act_scratch_);
      kern::AmxBatchNode nd[3];
      for (int i = 0; i < 3; ++i) {
        const Dense& d = i == 0 ? L.q : (i == 1 ? L.k : L.v);
        nd[i].phase = 0;
        nd[i].pa = &pb;
        nd[i].pb = &d.w_panel;
        nd[i].c = i == 0 ? qp : (i == 1 ? kp : vp);
        nd[i].M = T;
        nd[i].N = d.w_panel.rows;
      }
      kern::amx_batch_run(nd, 3);
      pb.buf.swap(hub_act_scratch_);
      auto addb = [&](float* p, const std::vector<float>& b) {
        for (size_t t2 = 0; t2 < T; ++t2)
          for (int i2 = 0; i2 < hidden_; ++i2) p[t2 * hidden_ + i2] += b[size_t(i2)];
      };
      addb(qp, L.q.b);
      addb(kp, L.k.b);
      addb(vp, L.v.b);
    } else
#endif
    {
      L.q.w.forward(x_.data(), T, qp, xh_);
      L.k.w.forward(x_.data(), T, kp, xh_);
      L.v.w.forward(x_.data(), T, vp, xh_);
      for (size_t t = 0; t < T; ++t) {
        for (int i = 0; i < hidden_; ++i) {
          qp[t * hidden_ + i] += L.q.b[static_cast<size_t>(i)];
          kp[t * hidden_ + i] += L.k.b[static_cast<size_t>(i)];
          vp[t * hidden_ + i] += L.v.b[static_cast<size_t>(i)];
        }
      }
    }
    // SDPA(无掩码): 每 head 独立
    const auto tp_sdpa0 = sdpaTim ? clk::now() : clk::time_point{};
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
    if (sdpaTim)
      sdpa_ms_ += std::chrono::duration<double, std::milli>(clk::now() - tp_sdpa0).count();
    // out proj + 残差 + post-LN
    resid_ = x_;                                   // attn_residual
#if defined(GSV_AMX_GEMM)
    if (L.o.w_panel_ready && T >= kAmxEncMinRows) {
      dense_amx(L.o, att_.data(), T, x_.data(), hub_act_scratch_, size_t(hidden_), L.o.b);
    } else
#endif
    {
      L.o.w.forward(att_.data(), T, x_.data(), xh_);
      for (size_t i = 0; i < T * hidden_; ++i) x_[i] += L.o.b[i % hidden_];
    }
    if (l == 0) cap_l0attn_.assign(x_.begin(), x_.begin() + static_cast<long>(T * hidden_));
    for (size_t i = 0; i < T * hidden_; ++i) x_[i] += resid_[i];
    for (size_t t = 0; t < T; ++t)
      gsv::kern::layernorm(x_.data() + t * hidden_, L.ln1_g.data(), L.ln1_b.data(),
                           x_.data() + t * hidden_, hidden_, ln_eps_);
    if (l == 0) cap_l0ln1_ = x_;
    // FFN(GELU) + 残差 + final LN
    ff_.resize(T * inter_);
#if defined(GSV_AMX_GEMM)
    if (L.f1.w_panel_ready && T >= kAmxEncMinRows) {
      dense_amx(L.f1, x_.data(), T, ff_.data(), hub_act_scratch_, size_t(hidden_), L.f1.b);
    } else
#endif
    {
      L.f1.w.forward(x_.data(), T, ff_.data(), xh_);
      for (size_t i = 0; i < T * inter_; ++i) ff_[i] += L.f1.b[i % inter_];
    }
    gelu(ff_.data(), ff_.size());
    resid_ = x_;
#if defined(GSV_AMX_GEMM)
    if (L.f2.w_panel_ready && T >= kAmxEncMinRows) {
      dense_amx(L.f2, ff_.data(), T, x_.data(), hub_act_scratch_, size_t(inter_), L.f2.b);
    } else
#endif
    {
      L.f2.w.forward(ff_.data(), T, x_.data(), xh_);
      for (size_t i = 0; i < T * hidden_; ++i) x_[i] += L.f2.b[i % hidden_];
    }
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
  // E13 探针: 输出本轮 CNN 逐层与 SDPA 耗时(仅 stderr, 不改行为)
  if (std::getenv("GSV_BERT_CONV_TIMING") != nullptr)
    std::fprintf(stderr,
                 "[hubert-seg] cnn_stack=%.1fms(pos_conv=%.1fms enc_ln%.1fms) "
                 "sdpa=%.1fms\n",
                 std::chrono::duration<double, std::milli>(tp_cnn1 - tp_cnn_start).count(),
                 std::chrono::duration<double, std::milli>(tp_pos1 - tp_cnn1).count(),
                 std::chrono::duration<double, std::milli>(clk::now() - tp_pos1).count() - sdpa_ms_,
                 sdpa_ms_);
  if (sdpaTim)
    std::fprintf(stderr, "[hubert-sdpa] sdpa_total=%.1fms T=%zu layers=%d\n",
                 sdpa_ms_, T, n_layers_);
  if (!conv_traces_.empty()) {
    double tot = 0.0;
    for (const auto& t : conv_traces_) tot += t.ms;
    std::fprintf(stderr, "[hubert-conv] total_gemm=%.1fms\n", tot);
    for (size_t i = 0; i < conv_traces_.size(); ++i) {
      const auto& t = conv_traces_[i];
      std::fprintf(stderr,
                   "[hubert-conv] L%zu c%d*len%d k%ds%d -> out[c%d,len%d] fmlal=%.1fms\n",
                   i, t.in_c, t.in_len, t.k, t.s, t.out_c, t.out_len, t.ms);
    }
    (void)tp_run0;
  } else {
    (void)tp_run0;
  }
  return T;
}

}  // namespace gsv::encoder
