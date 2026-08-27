// hubert.cpp — HuBERT 推理实现(数值纪律: fp16 权重无损升位, 计算全 fp32; 大矩阵走 sgemm)
#include "encoder/hubert.hpp"

#include "kern/kern.hpp"
#include "runtime/gsv_loader.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include <arm_neon.h>

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
    // T11: 删除 'f'(ffn-only) 分支 —— 构造期仅 kAll 打包 panel,
    // kFfnOnly 实为空开关(与 FMLAL 完全同路), 已实证(v1_codes)并裁定移除。
    if (e[0] == 'n') return AmxEncMode::kNone;
    return AmxEncMode::kAll;
  }();
  return m;
}

// E12 分流门槛(与 E8 bert 一致): T≥48(tile 半填充以上)且 K≥256
constexpr size_t kAmxEncMinRows = 48;
constexpr size_t kAmxEncMinK = 256;

namespace {
// E14 微观测探针(GSV_HUBERT_MICRO_TIMING=1): 段级耗时累计, 纯观测不改行为。
// 计时点均在段边界(非内循环), 单次 <100 个 clock 调用/run, 失真可忽略。
struct MicroAcc {
  double cnn_im2col = 0, cnn_gemm = 0, cnn_gn = 0, cnn_gelu = 0;
  double pos_im2col = 0, pos_gemm = 0, proj_seg = 0;
  double enc_qkv = 0, enc_outp = 0, enc_f1 = 0, enc_f2 = 0, enc_gelu = 0,
         enc_resid = 0, enc_ln = 0;
  double act_pack = 0, amx_math = 0;  // dense_amx 内部: 激活打包 / AMX 数学
};
thread_local MicroAcc g_micro;
bool micro_on() {
  static const bool b = std::getenv("GSV_HUBERT_MICRO_TIMING") != nullptr;
  return b;
}
double ms_since(std::chrono::steady_clock::time_point t0,
                std::chrono::steady_clock::time_point t1) {
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}
}  // namespace

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
#if defined(GSV_AMX_GEMM)
    // T13: CNN 权重装载期预打包(仅 kAll)。打包是装载期一次性成本,
    // 运行时每次卷积都省 FMLAL 全量重算, 无需形状门槛抛股(除退化防护);
    // 数值谱系 = E12/E5-P2(f16 im2col 输入位型不变, 归约序换 Z 长条链),
    // 由 codes 56 段全量门禁裁决。
    if (amx_hubert_enabled() && amx_hubert_mode() == AmxEncMode::kAll &&
        kern::amx_gemm_available() && wv.numel() >= 4096) {
      L.conv_panel.rows = static_cast<size_t>(out_c);
      L.conv_panel.K = wv.numel() / static_cast<size_t>(out_c);
      kern::amx_pack_into(L.w16, L.conv_panel.rows, L.conv_panel.K,
                          L.conv_panel.buf);
      L.conv_panel_ready = true;
    }
#endif
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

// ---- E14-H1: NEON 无-libm GELU(旗后专用; 微基准裁决 .tmp/gelu_neon_probe2) ----
// erf/erfc: Abramowitz & Stegun 7.1.26 有理近似(|ε|≤1.5e-7, fp32 半支外推);
// exp(-y): y≥0 Taylor 到 r^6 + ln2 双段归约(fdlibm 惯例), k 下夹 -126。
// 探针实测(对 std::erf 全域 [-8,8]): max_abs=4.77e-07, max_rel≈2.0e-4;
// 向量核 ~0.02 ns/el vs 生产标量 erf ~2.4 ns/el。旗后 mel 门放行项;
// 默认路径仍走 gelu() 的 std::erf 原函数, 不受影响。
inline float32x4_t hub_exp_neg(float32x4_t y) {
  const float32x4_t negy = vnegq_f32(y);
  const float32x4_t LOG2EF = vdupq_n_f32(1.44269504088896341f);
  const float32x4_t LN2_HI = vdupq_n_f32(0.693359375f);
  const float32x4_t LN2_LO = vdupq_n_f32(-2.12194440e-4f);
  int32x4_t k = vcvtnq_s32_f32(vmulq_f32(negy, LOG2EF));
  k = vmaxq_s32(k, vdupq_n_s32(-126));
  const float32x4_t kf = vcvtq_f32_s32(k);
  float32x4_t r = vfmsq_f32(negy, kf, LN2_HI);
  r = vfmsq_f32(r, kf, LN2_LO);
  float32x4_t p = vdupq_n_f32(1.f / 720.f);
  p = vfmaq_f32(vdupq_n_f32(1.f / 120.f), r, p);
  p = vfmaq_f32(vdupq_n_f32(1.f / 24.f), r, p);
  p = vfmaq_f32(vdupq_n_f32(1.f / 6.f), r, p);
  p = vfmaq_f32(vdupq_n_f32(0.5f), r, p);
  p = vfmaq_f32(vdupq_n_f32(1.f), r, p);
  p = vfmaq_f32(vdupq_n_f32(1.f), r, p);
  const int32x4_t bits =
      vshlq_n_u32(vreinterpretq_u32_s32(vaddq_s32(k, vdupq_n_s32(127))), 23);
  return vmulq_f32(p, vreinterpretq_f32_s32(bits));
}

// 位精确 GELU(基线同公式 std::erf; 仅用于 H1 快路出口/独立遍)
inline void g_hub_gelu(float* x, size_t n) {
  for (size_t i = 0; i < n; ++i)
    x[i] = 0.5f * x[i] * (1.0f + std::erf(x[i] * static_cast<float>(M_SQRT1_2)));
}


// 标量同构(im2col 取数内联用; 与向量版同公式同系数)


// 段级向量仿射(seg[i]*a+b; 连续内存; 用于 GN 仿射折进取数)

// 向量 GELU 单向量版(供行内联用)
inline float32x4_t hub_gelu_v(float32x4_t x) {
  const float32x4_t z = vmulq_n_f32(x, 0.70710678118654752440f);
  const float32x4_t az = vabsq_f32(z);
  const float32x4_t e = hub_exp_neg(vmulq_f32(az, az));
  const float32x4_t t = vdivq_f32(
      vdupq_n_f32(1.f), vfmaq_n_f32(vdupq_n_f32(1.f), az, 0.3275911f));
  float32x4_t q = vdupq_n_f32(1.061405429f);
  q = vfmaq_f32(vdupq_n_f32(-1.453152027f), t, q);
  q = vfmaq_f32(vdupq_n_f32(1.421413741f), t, q);
  q = vfmaq_f32(vdupq_n_f32(-0.284496736f), t, q);
  q = vfmaq_f32(vdupq_n_f32(0.254829592f), t, q);
  q = vmulq_f32(t, q);
  float32x4_t ef = vsubq_f32(vdupq_n_f32(1.f), vmulq_f32(e, q));
  const uint32x4_t sgn = vcltzq_s32(vreinterpretq_s32_f32(z));
  ef = vbslq_f32(sgn, vnegq_f32(ef), ef);
  return vmulq_f32(vmulq_n_f32(x, 0.5f), vaddq_f32(vdupq_n_f32(1.f), ef));
}

// 行主 f16 cols 行构建(向量化)。im2col 列序 cols[c*k+j] 由 vstN 交错直产;
// 所有 k 路 tap 统一过同一上游变换(GN 仿射或 GELU)。
inline void hub_cols_row_k3(uint16_t* __restrict dst, const float* rA,
                            const float* rB, const float* rC, int C,
                            bool fuse_gelu, const float* __restrict ga,
                            const float* __restrict gb) {
  for (int c = 0; c < C; c += 4) {
    float32x4_t va = vld1q_f32(rA + c);
    float32x4_t vb = vld1q_f32(rB + c);
    float32x4_t vc = vld1q_f32(rC + c);
    if (fuse_gelu) {
      if (ga) {  // 先 GN 仿射再 GELU(L1 上游=L0 原始值)
        va = vfmaq_f32(vld1q_f32(gb + c), va, vld1q_f32(ga + c));
        vb = vfmaq_f32(vld1q_f32(gb + c), vb, vld1q_f32(ga + c));
        vc = vfmaq_f32(vld1q_f32(gb + c), vc, vld1q_f32(ga + c));
      }
      va = hub_gelu_v(va);
      vb = hub_gelu_v(vb);
      vc = hub_gelu_v(vc);
    }
    const float16x4_t ha = vcvt_f16_f32(va);
    const float16x4_t hb = vcvt_f16_f32(vb);
    const float16x4_t hc = vcvt_f16_f32(vc);
    const float16x4x3_t t3{{ha, hb, hc}};
    // vst3 交错: dst[3(c+i)+j]={a_i,b_i,c_i} 恰为 im2col 列序 (c*j)
    vst3_f16(reinterpret_cast<__fp16*>(dst + 3 * c), t3);
  }
}

inline void hub_cols_row_k2(uint16_t* __restrict dst, const float* rA,
                            const float* rB, int C, bool fuse_gelu) {
  for (int c = 0; c < C; c += 4) {
    float32x4_t va = vld1q_f32(rA + c);
    float32x4_t vb = vld1q_f32(rB + c);
    if (fuse_gelu) {
      va = hub_gelu_v(va);
      vb = hub_gelu_v(vb);
    }
    const float16x4x2_t t2{{vcvt_f16_f32(va), vcvt_f16_f32(vb)}};
    vst2_f16(reinterpret_cast<__fp16*>(dst + 2 * c), t2);
  }
}

// f32 → f16 位型(RNE; 与 kern::f32_to_f16_scalar 同口径)

}  // namespace

// ---- T12: SDPA AMX 批量化(旗后; E11-5 配方 + 微基准裁决的 batched sgemm 路径) ----
// 数值口径: 不位级(与 V1/E12 先例同谱系, codes 逐位为门禁)。
//   每层每头两次 fp32 sgemm:
//     QKT: C[T,T] = Q_h[T,HD] · K_h[T,HD]ᵀ  (gather 列切片后连续)
//     PV : C[T,HD] = P[T,T] · VT_h[HD,T]ᵀ   (P 直接吃 softmax f32 输出,
//                                              VT 为列转置 gather 布局)
//   scale/softmax 与标量路径同一序: 先逐元素乘 scale 再 kern::softmax。
// ---- E14-H2: SDPA 专用行 softmax(NEON 批量 max-sum; 旗后 mel 门口径) ----
// 与 kern::softmax 的差异: exp 换成向量化 fast-exp(同 H1 探针内核,
// e^{-y} Taylor r^6 + ln2 双段归约, max 归一后 |y| 有界 → 精度充分),
// 统计(double sum 语义不变 → 用 fp32 向量和; mel 门裁决)。
namespace {
inline float32x4_t sdpa_exp_neg(float32x4_t y) {
  // 语义: 返回 e^{-a}, a = -y ≥ 0。采用 2^{-k}·poly(u), u = -r ∈ [-ln2/2,ln2/2]
  // (先做 -a→双段归约, 再对 u 用正向 e^u 泰勒 —— 与 H1 探针验证同构)
  const float32x4_t LOG2EF = vdupq_n_f32(1.44269504088896341f);
  const float32x4_t LN2_HI = vdupq_n_f32(0.693359375f);
  const float32x4_t LN2_LO = vdupq_n_f32(-2.12194440e-4f);
  int32x4_t k = vcvtnq_s32_f32(vmulq_f32(y, LOG2EF));  // 注意此处用 y(=−a)
  int32x4_t kk = vsubq_s32(vdupq_n_s32(0), k);         // 指数 −k
  kk = vmaxq_s32(kk, vdupq_n_s32(-126));
  // r = y − k·ln2 ≈ 高精度余量(此时 |r| ≤ ln2/2), 对 −r 展开 e^{−r}
  const float32x4_t kf = vcvtq_f32_s32(k);
  float32x4_t r = vfmsq_f32(y, kf, LN2_HI);
  r = vfmsq_f32(r, kf, LN2_LO);
  const float32x4_t u = vnegq_f32(r);                  // u = −r
  float32x4_t p = vdupq_n_f32(1.f / 720.f);
  p = vfmaq_f32(vdupq_n_f32(1.f / 120.f), u, p);
  p = vfmaq_f32(vdupq_n_f32(1.f / 24.f), u, p);
  p = vfmaq_f32(vdupq_n_f32(1.f / 6.f), u, p);
  p = vfmaq_f32(vdupq_n_f32(0.5f), u, p);
  p = vfmaq_f32(vdupq_n_f32(1.f), u, p);
  p = vfmaq_f32(vdupq_n_f32(1.f), u, p);
  const int32x4_t bits =
      vshlq_n_u32(vreinterpretq_u32_s32(vaddq_s32(kk, vdupq_n_s32(127))), 23);
  return vmulq_f32(p, vreinterpretq_f32_s32(bits));
}
inline void sdpa_softmax_row(float* y, size_t n) {
  // 批量 max(向量归约)
  size_t i = 0;
  float32x4_t vmax = vdupq_n_f32(-3.40282347e38f);
  for (; i + 4 <= n; i += 4) vmax = vmaxq_f32(vmax, vld1q_f32(y + i));
  float tmp[4];
  vst1q_f32(tmp, vmax);
  float m = std::max<float>(std::max<float>(tmp[0], tmp[1]),
                            std::max<float>(tmp[2], tmp[3]));
  for (; i < n; ++i) m = std::max(m, y[i]);
  i = 0;
  float32x4_t vsum = vdupq_n_f32(0.f);
  for (; i + 4 <= n; i += 4) {
    // d = m − x ≥ 0 → e^{−d} = e^{x−m}(softmax 标准)。
    const float32x4_t e = sdpa_exp_neg(
        vnegq_f32(vsubq_f32(vld1q_f32(y + i), vdupq_n_f32(m))));
    vst1q_f32(y + i, e);
    vsum = vaddq_f32(vsum, e);
  }
  float s = vaddvq_f32(vsum);
  for (; i < n; ++i) {
    const float e = std::exp(y[i] - m);
    y[i] = e;
    s += e;
  }
  const float inv = 1.0f / s;
  i = 0;
  const float32x4_t vin = vdupq_n_f32(inv);
  for (; i + 4 <= n; i += 4) vst1q_f32(y + i, vmulq_f32(vld1q_f32(y + i), vin));
  for (; i < n; ++i) y[i] *= inv;
}
}  // namespace

void HubertEngine::sdpa_amx_sgemm(float* qp, float* kp, float* vp, size_t T,
                                  size_t hd, float scale, float* att_out) {
  const size_t D = hidden_;
  if (sdpasc_.size() < T * hd) sdpasc_.resize(T * hd);
  if (sdpa_qg_.size() < T * hd) sdpa_qg_.resize(T * hd);
  if (sdpa_kg_.size() < T * hd) sdpa_kg_.resize(T * hd);
  if (sdpa_vtg_.size() < hd * T) sdpa_vtg_.resize(hd * T);
  if (sc_.size() < heads_ * T * T) sc_.resize(heads_ * T * T);
  if (ovh_.size() < T * hd) ovh_.resize(T * hd);

  for (int h = 0; h < heads_; ++h) {
    const size_t off = size_t(h) * hd;
    // gather Q/K 列切片 → 连续 [T,hd]
    for (size_t t = 0; t < T; ++t) {
      std::memcpy(sdpa_qg_.data() + t * hd, qp + t * D + off, hd * sizeof(float));
      std::memcpy(sdpa_kg_.data() + t * hd, kp + t * D + off, hd * sizeof(float));
    }
    float* sc_head = sc_.data() + size_t(h) * T * T;
    // QK^T: sgemm('N','T', M=T, N=T, K=hd); A=Q_g [T,hd], B=K_g [T,hd]
    gsv::kern::accel::sgemm('N', 'T', int(T), int(T), int(hd), 1.0f,
                            sdpa_qg_.data(), int(hd), sdpa_kg_.data(), int(hd),
                            0.0f, sc_head, int(T));
    for (size_t i = 0; i < T * T; ++i) sc_head[i] *= scale;
    // E14-H2: NEON 行 softmax(批量 max-sum + fast-exp)。mel 门 PASS 但
    // N=56 有清单外新增 codes 翻转 → 默认关闭(位级红线优先),
    // GSV_H2_ON=1 复验/可选性能档(spa −26ms)。
    static const bool h2_on =
        [] { const char* e = std::getenv("GSV_H2_ON"); return e && e[0] == '1'; }();
    if (h2_on)
      for (size_t q = 0; q < T; ++q) sdpa_softmax_row(sc_head + q * T, T);
    else
      for (size_t q = 0; q < T; ++q)
        gsv::kern::softmax(sc_head + q * T, sc_head + q * T, T);
    // gather V 列转置 → [hd,T]
    for (size_t t = 0; t < T; ++t)
      for (size_t e = 0; e < hd; ++e)
        sdpa_vtg_[e * T + t] = vp[t * D + off + e];
    // P·V: C[T,hd] = P[T,T] · VT[hd,T]ᵀ
    gsv::kern::accel::sgemm('N', 'T', int(T), int(hd), int(T), 1.0f, sc_head,
                            int(T), sdpa_vtg_.data(), int(T), 0.0f,
                            sdpasc_.data(), int(hd));
    for (size_t q = 0; q < T; ++q)
      std::memcpy(att_out + q * D + off, sdpasc_.data() + q * hd,
                  hd * sizeof(float));
  }
}

// ---- E14-H1: CNN 栈直产快路(旗后; --amx-enc 且七层 conv_panel 就绪) ----
// 结构(全栈帧主化):
//   L0: 波形原生小 GEMM(fp32 权重展开乘加) → 独立像表 im0[T1,512](原始值,
//       无 bias 有定义), GN 双遍统计只产仿射系数表(gn_a/gn_b2), 不回写激活;
//   L1..L6: 逐输出帧直产 im2col f16 Y-panel(kern::pack_panel 布局),
//       取数时融合上游变换 —— L1 做 GN 仿射, L2..L6 做上游激活的 GELU(A&S
//       近似), 然后 gemm_f16_amx_pp 以 cols 为 A、装载期权重 panel 为 B 派发,
//       C=[Lout,Co] 帧主直出(C=W·colsᵀ 与 cols·Wᵀ 数值同序)。
//   L6 输出补 GELU(无下游取数消费) → tmp_ [T6,512] 帧主交还调用方。
//   相对原省去: 独立 im2col 遍 / AMX repack 遍 / 每层 GELU 全量扫描 /
//   GN 归一化全量回写 / 末端 512×T 转置。
// 数值口径:
//   - f16 量化点位不变(取数即量化), 归约仍为 T13 AMX Z 长条链(谱系同现行);
//   - GELU 改 A&S 近似(max_abs 4.77e-07)—— 经 f16 量化吸收后由 mel 门裁决;
//   - GN 统计公式逐运算保持(double 累加, inv 乘, eps 同点)。
size_t HubertEngine::cnn_stack_fused(const float* waveform, size_t n) {
  using clk = std::chrono::steady_clock;
  const auto tp_h0 = clk::now();
  for (int li = 0; li < 7; ++li)
    if (!convs_[li].conv_panel_ready) {
      if (micro_on())
        std::fprintf(stderr, "[hubert-h1] 防御拒: conv%d panel 未就绪\n", li);
      return 0;
    }
  if (!kern::amx_gemm_available()) {
    if (micro_on()) std::fprintf(stderr, "[hubert-h1] 防御拒: AMX 不可用\n");
    return 0;
  }
  // 版本锁门: 本函数的字面量(512/2560/k10/k3/k2/s5/s2)仅在标准
  // chinese-hubert-base 结构下有效; config 不符一律回退原路(不硬编码假设)。
  {
    const int exp_dim[7] = {512, 512, 512, 512, 512, 512, 512};
    const int exp_k[7] = {10, 3, 3, 3, 3, 2, 2};
    const int exp_s[7] = {5, 2, 2, 2, 2, 2, 2};
    for (int li = 0; li < 7; ++li)
      if (conv_dim_[li] != exp_dim[li] || conv_kernel_[li] != exp_k[li] ||
          conv_stride_[li] != exp_s[li]) {
        if (micro_on())
          std::fprintf(stderr, "[hubert-h1] 防御拒: conv%d 形状与支持结构不符\n", li);
        return 0;
      }
  }

  const auto tp_pre0 = micro_on() ? clk::now() : clk::time_point{};
  size_t lens[8];
  lens[0] = n;
  for (int li = 0; li < 7; ++li) {
    const int k = conv_kernel_[li], s = conv_stride_[li];
    lens[li + 1] =
        (lens[li] - static_cast<size_t>(k)) / static_cast<size_t>(s) + 1;
  }
  auto act_dim = [&](int li) { return li == 0 ? 1 : conv_dim_[li - 1]; };

  // ---- L0: 小 GEMM → im0[T1,512](原始值, 位精确复刻) ----
  // 位级口径: 与基线 AMX/FMLAL 一致 —— 输入 tap 先量化 f16(im2col 同点),
  // f16×f16 积在 fp32 内精确; k 严格左折叠(Z 长条链同序)。
  const size_t N1 = lens[1];
  std::vector<float> im0(N1 * 512);
  {
    const __fp16* wp = reinterpret_cast<const __fp16*>(convs_[0].w16);
    alignas(64) float wexp[512][10];
    for (int co = 0; co < 512; ++co)
      for (int kk = 0; kk < 10; ++kk)
        wexp[co][kk] = static_cast<float>(wp[static_cast<size_t>(co) * 10 + kk]);
    float* dst = im0.data();
    // 位精确左折叠(t 外, co 内连续访存; 每 tap 一条标量链同基线序)。
    // 先做 f16 量化(im2col 同点), 再逐 co 展开。
    for (size_t t = 0; t < N1; ++t) {
      const float* win = waveform + t * 5;
      float xv[10];
      for (int kk = 0; kk < 10; ++kk)
        xv[kk] = static_cast<float>(static_cast<__fp16>(win[kk]));
      float* dstrow = dst + t * 512;
      for (int co = 0; co < 512; ++co) {
        const float* wr = wexp[co];
        float s = xv[0] * wr[0];
        s = xv[1] * wr[1] + s;
        s = xv[2] * wr[2] + s;
        s = xv[3] * wr[3] + s;
        s = xv[4] * wr[4] + s;
        s = xv[5] * wr[5] + s;
        s = xv[6] * wr[6] + s;
        s = xv[7] * wr[7] + s;
        s = xv[8] * wr[8] + s;
        s = xv[9] * wr[9] + s;
        dstrow[co] = s;
      }
    }
  }
  if (micro_on()) g_micro.cnn_gemm += ms_since(tp_pre0, clk::now());

  // ---- GN 统计(L0 输出) → 仿射系数表(不回写激活; L1 取数消费) ----
  const auto tp_gn0 = micro_on() ? clk::now() : clk::time_point{};
  alignas(64) float gn_a[512], gn_b2[512];
  {
    const double inv = 1.0 / static_cast<double>(N1);
    for (int ch = 0; ch < 512; ++ch) {
      double mu = 0.0, var = 0.0;
      for (size_t t = 0; t < N1; ++t) mu += im0[static_cast<size_t>(t) * 512 + ch];
      mu *= inv;
      for (size_t t = 0; t < N1; ++t) {
        const double d = static_cast<double>(im0[static_cast<size_t>(t) * 512 + ch]) - mu;
        var += d * d;
      }
      var *= inv;
      const double rstd = 1.0 / std::sqrt(var + ln_eps_);
      const double g = convs_[0].gn_g[static_cast<size_t>(ch)];
      const double b = convs_[0].gn_b[static_cast<size_t>(ch)];
      gn_a[ch] = static_cast<float>(g * rstd);
      gn_b2[ch] = static_cast<float>(b - g * rstd * mu);
    }
  }
  if (micro_on()) g_micro.cnn_gn += ms_since(tp_gn0, clk::now());

  // ---- L1..L6: 位保真激活链 + im2col/打包融合(仅纯搬运) ----
  // 数值策略(H1 近似版撤回后的重构): GELU/GN 必须逐元素复刻基线序列
  // (std::erf、double 统计、相同作用点), 不允许重排 —— 只允许纯搬运算子
  // 融合(im2col+panel 打包合并为一次 NEON 遍)。位级目标: 与原路 CNN 栈
  // 逐值相同。
  std::vector<float> bufs[2];
  size_t cbuf = 0;
  bufs[0].swap(im0);
  kern::AmxPanel cp_buf;

  size_t Lprev = lens[1];   // li==1 时上游有效长度(L0 输出长度)
  auto run_stage = [&](int li) {
    const size_t Lout = lens[li + 1];
    const int C = act_dim(li);
    const int K = conv_kernel_[li], S = conv_stride_[li];
    const size_t KK = static_cast<size_t>(C) * static_cast<size_t>(K);
    float* up = bufs[cbuf].data();
    float* outbuf;
    {
      auto& ob = bufs[cbuf ^ 1];
      const size_t need = Lout * static_cast<size_t>(conv_dim_[li]);
      if (ob.size() < need) ob.resize(need);
      outbuf = ob.data();
    }

    // (a) 位精确变换(与基线同公式同序, 仅布局通道主→帧主):
    //   li==1: 上游是 GN 归一化前的 L0 原始输出 → 在此做 GN(取数已按帧主)
    //   其余层: 上游是"上一 AMX 出口", 其 GELU 已在该阶段出口统一做过
    if (li == 1) {
      const double inv = 1.0 / static_cast<double>(Lprev);
      for (int ch = 0; ch < C; ++ch) {
        double mu = 0.0, var = 0.0;
        for (size_t t = 0; t < Lprev; ++t) mu += up[t * C + ch];
        mu *= inv;
        for (size_t t = 0; t < Lprev; ++t) {
          const double d = static_cast<double>(up[t * C + ch]) - mu;
          var += d * d;
        }
        var *= inv;
        const double rstd = 1.0 / std::sqrt(var + ln_eps_);
        const double g = static_cast<double>(convs_[0].gn_g[static_cast<size_t>(ch)]);
        const double b = static_cast<double>(convs_[0].gn_b[static_cast<size_t>(ch)]);
        for (size_t t = 0; t < Lprev; ++t) {
          float* px = up + t * C + ch;
          *px = static_cast<float>((*px - mu) * rstd * g + b);
        }
      }
      // GN 后接 GELU(基线次序: 每层 conv 出来即过激活), 位精确
      g_hub_gelu(up, static_cast<size_t>(C) * Lprev);
    }
    // 每帧(输出时间步)把 im2col 行构建到行主 cols16_(交错列序), 累计满一个
    // tile 带(32 帧)即交给 packer 合并打包 → 流水化, 免全量第二遍:
    const auto tp_i0 = clk::now();
    cols16_.resize(Lout * KK);
    uint16_t* cb16 = cols16_.data();
    for (size_t t = 0; t < Lout; ++t) {
      const size_t src0 = t * static_cast<size_t>(S);
      const float* rA = up + src0 * static_cast<size_t>(C);
      if (K == 2)
        hub_cols_row_k2(cb16 + t * KK, rA, rA + C, C, /*fuse_gelu=*/false);
      else
        hub_cols_row_k3(cb16 + t * KK, rA, rA + C, rA + 2 * C, C,
                        /*fuse_gelu=*/false, nullptr, nullptr);
    }
    const auto tp_i1 = clk::now();
    kern::amx_pack_into(cols16_.data(), Lout, KK, hub_cols_panel_);
    const auto tp_i2 = clk::now();
    cp_buf.rows = Lout;
    cp_buf.K = KK;
    cp_buf.buf.swap(hub_cols_panel_);
    kern::gemm_f16_amx_pp(cp_buf, convs_[li].conv_panel, outbuf, Lout,
                          static_cast<size_t>(conv_dim_[li]));
    const auto tp_i3 = clk::now();
    if (micro_on()) {
      std::fprintf(stderr, "[h1-stage] L%d im2col=%.1f pack=%.1f gemm=%.1f\n",
                   li, ms_since(tp_i0, tp_i1), ms_since(tp_i1, tp_i2),
                   ms_since(tp_i2, tp_i3));
    }
    cp_buf.buf.swap(hub_cols_panel_);
    g_hub_gelu(outbuf, Lout * static_cast<size_t>(conv_dim_[li]));
    cbuf ^= 1;
  };
  for (int li = 1; li < 7; ++li) {
    run_stage(li);
    Lprev = lens[li + 1];
  }

  // ---- 布局交接: 终缓冲已是 [T6,512] 帧主(GELU 已含在 stage 出口) ----
  const size_t T6 = lens[7];
  tmp_.resize(T6 * 512);
  std::memcpy(tmp_.data(), bufs[cbuf].data(), T6 * 512 * sizeof(float));
  cnn_t_ = T6;
  // 说明: 快路不填充通道主 cnn_ 快照(cnn_out() 视图仅原路语义);
  // pipeline 生产链只消费 tmp_(帧主), 测试对拍走原路不受影响。
  if (micro_on())
    std::fprintf(stderr, "[hubert-h1] cnn_stack_fused=%.1fms\n",
                 ms_since(tp_h0, clk::now()));
  return T6;
}
#endif

void HubertEngine::gelu(float* x, size_t n) {
  for (size_t i = 0; i < n; ++i)
    x[i] = 0.5f * x[i] * (1.0f + std::erf(x[i] * static_cast<float>(M_SQRT1_2)));
}

namespace {

// ---- T11 保序 NEON SDPA 内核(构造性位级等价, V2 反汇编实锤配方) ----
//
// 数值论证(见 .tmp/evidence-V.md / .tmp/t11_gate.bin 验收):
//  · 生产目标码实测(dot 主循环): 向量精确积(fmul.4s) → lane 提取 →
//    升序注入单累加器(分离加, 无收缩)。
//    ⇒ 本内核逐字复刻该形态; vmulq 积精确落入寄存器后再按同序插入,
//      算术图与舍入点序列完全不变(元素级对拍 0/159201 失配)。
//  · 双分数(t, t+1)交错只是并行产两条独立链, 各自仍严格 e 升序左折叠,
//    不引入任何新的结合点。
//  contract(off): 防止此处 dot 被收缩成 fmadd(生产 dot 未收缩)。
#pragma clang fp contract(off)
inline void hubert_sdpa_qkt(const float* qp, const float* kp, size_t T,
                            size_t D, size_t off, size_t hd, float scale,
                            float* sc) {
  const float* qs = qp + off;
  const float* ks = kp + off;
  size_t t = 0;
  for (; t + 2 <= T; t += 2) {
    const float* kv0 = ks + t * D;
    const float* kv1 = kv0 + D;
    for (size_t q = 0; q < T; ++q) {
      const float* qv = qs + q * D;
      float d0 = 0.f, d1 = 0.f;
      size_t e = 0;
      for (; e + 8 <= hd; e += 8) {
        const float32x4_t p00 = vmulq_f32(vld1q_f32(qv + e), vld1q_f32(kv0 + e));
        const float32x4_t p01 =
            vmulq_f32(vld1q_f32(qv + e + 4), vld1q_f32(kv0 + e + 4));
        const float32x4_t p10 = vmulq_f32(vld1q_f32(qv + e), vld1q_f32(kv1 + e));
        const float32x4_t p11 =
            vmulq_f32(vld1q_f32(qv + e + 4), vld1q_f32(kv1 + e + 4));
        d0 = d0 + vgetq_lane_f32(p00, 0);
        d0 = d0 + vgetq_lane_f32(p00, 1);
        d0 = d0 + vgetq_lane_f32(p00, 2);
        d0 = d0 + vgetq_lane_f32(p00, 3);
        d0 = d0 + vgetq_lane_f32(p01, 0);
        d0 = d0 + vgetq_lane_f32(p01, 1);
        d0 = d0 + vgetq_lane_f32(p01, 2);
        d0 = d0 + vgetq_lane_f32(p01, 3);
        d1 = d1 + vgetq_lane_f32(p10, 0);
        d1 = d1 + vgetq_lane_f32(p10, 1);
        d1 = d1 + vgetq_lane_f32(p10, 2);
        d1 = d1 + vgetq_lane_f32(p10, 3);
        d1 = d1 + vgetq_lane_f32(p11, 0);
        d1 = d1 + vgetq_lane_f32(p11, 1);
        d1 = d1 + vgetq_lane_f32(p11, 2);
        d1 = d1 + vgetq_lane_f32(p11, 3);
      }
      for (; e < hd; ++e) {
        d0 = d0 + qv[e] * kv0[e];
        d1 = d1 + qv[e] * kv1[e];
      }
      sc[q * T + t] = d0 * scale;
      sc[q * T + t + 1] = d1 * scale;
    }
  }
  for (; t < T; ++t) {
    const float* kv0 = ks + t * D;
    for (size_t q = 0; q < T; ++q) {
      const float* qv = qs + q * D;
      float d0 = 0.f;
      size_t e = 0;
      for (; e + 4 <= hd; e += 4) {
        const float32x4_t pp = vmulq_f32(vld1q_f32(qv + e), vld1q_f32(kv0 + e));
        d0 = d0 + vgetq_lane_f32(pp, 0);
        d0 = d0 + vgetq_lane_f32(pp, 1);
        d0 = d0 + vgetq_lane_f32(pp, 2);
        d0 = d0 + vgetq_lane_f32(pp, 3);
      }
      for (; e < hd; ++e) d0 = d0 + qv[e] * kv0[e];
      sc[q * T + t] = d0 * scale;
    }
  }
}
#pragma clang fp contract(on)  // 恢复默认(此函数外无其他声明)

// PV(out = Σ_t p[t]·V[t,e]): 每输出元素一条独立链(t 升序), e 维 128-bit
// lane 化。用 vfmaq 融合链 —— 生产目标码的 PV 循环正是 fmla.4s 融合形态
// (与 dot 相反!), 尾部 std::fmaf 同语义。定义级对拍 0/25536 失配。
#pragma clang fp contract(on)
inline void hubert_sdpa_pv(const float* probs, const float* vp, size_t T,
                           size_t D, size_t off, size_t hd, float* ov_head) {
  std::fill(ov_head, ov_head + T * hd, 0.f);
  for (size_t q = 0; q < T; ++q) {
    const float* pr = probs + q * T;
    float* ov = ov_head + q * hd;
    size_t e = 0;
    for (; e + 16 <= hd; e += 16) {
      float32x4_t a0 = vdupq_n_f32(0.f), a1 = vdupq_n_f32(0.f);
      float32x4_t a2 = vdupq_n_f32(0.f), a3 = vdupq_n_f32(0.f);
      for (size_t t = 0; t < T; ++t) {
        const float32x4_t p4 = vdupq_n_f32(pr[t]);
        const float* vv = vp + t * D + off + e;
        a0 = vfmaq_f32(a0, p4, vld1q_f32(vv));
        a1 = vfmaq_f32(a1, p4, vld1q_f32(vv + 4));
        a2 = vfmaq_f32(a2, p4, vld1q_f32(vv + 8));
        a3 = vfmaq_f32(a3, p4, vld1q_f32(vv + 12));
      }
      vst1q_f32(ov + e, a0);
      vst1q_f32(ov + e + 4, a1);
      vst1q_f32(ov + e + 8, a2);
      vst1q_f32(ov + e + 12, a3);
    }
    for (; e < hd; ++e) {
      float s = 0.f;
      for (size_t t = 0; t < T; ++t)
        s = std::fmaf(pr[t], vp[t * D + off + e], s);
      ov[e] = s;
    }
  }
}
#pragma clang fp contract(on)

}  // namespace

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

// E12: AMX dense — y[T,out] = x[T,in]·W[out,in]ᵀ + bias(与 FMLAL 同口径逐元素加)
// 注: pb.buf 用 swap 而非 move — amx_batch_run 返回后面板寿命结束,
// 缓冲容量经 pb 析构流回调用方(act_scratch), 容量跨调用复用免反复 malloc。
void HubertEngine::dense_amx(const Dense& d, const float* x, size_t T, float* y,
                             std::vector<uint8_t>& act_scratch, size_t in_dim,
                             const std::vector<float>& bias) const {
  using clk = std::chrono::steady_clock;
  const auto tp_m0 = clk::now();
  cast_pack_B_f32(x, T, in_dim, act_scratch);
  const auto tp_m1 = micro_on() ? clk::now() : tp_m0;
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
  const auto tp_m2 = micro_on() ? clk::now() : tp_m1;
  pb.buf.swap(act_scratch);  // 收回容量
  if (micro_on()) {
    g_micro.act_pack += ms_since(tp_m0, tp_m1);
    g_micro.amx_math += ms_since(tp_m1, tp_m2);
  }
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
  const auto tp_m0 = clk::now();
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
  const auto tp_m1 = clk::now();
#if defined(GSV_AMX_GEMM)
  // T13: 旗后走 AMX pp —— im2col(cols16_) 输入位型不变, 仅 GEMM 归约后端
  // FMLAL→AMX Z 长条链(E5-P2 同构); B 面板由 cols16_ 行主 [T,KK] 直包。
  if (convs_[li].conv_panel_ready) {
    kern::amx_pack_into(cols16_.data(), T, KK, hub_conv_act_);
    kern::AmxPanel pb;
    pb.rows = T;
    pb.K = KK;
    pb.buf.swap(hub_conv_act_);
    kern::gemm_f16_amx_pp(convs_[li].conv_panel, pb, out.data(),
                          static_cast<size_t>(out_c), T);
    pb.buf.swap(hub_conv_act_);
    out_len = T;
    if (micro_on()) {
      const auto tp_m2 = clk::now();
      g_micro.cnn_im2col += ms_since(tp_m0, tp_m1);
      g_micro.cnn_gemm += ms_since(tp_m1, tp_m2);
    }
    return;
  }
#endif
  kern::gemm_f16x_fmlal(convs_[li].w16, cols16_.data(), out.data(),
                        static_cast<size_t>(out_c), T, KK);
  out_len = T;
  if (micro_on()) {
    const auto tp_m2 = clk::now();
    g_micro.cnn_im2col += ms_since(tp_m0, tp_m1);
    g_micro.cnn_gemm += ms_since(tp_m1, tp_m2);
  }
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
  int c = 1;
  size_t len = n;
  bool h1_fast = false;   // E14-H1: 快路已直产 tmp_(帧主)
#if defined(GSV_AMX_GEMM)
  // E14-H1: 位保真快路(全栈帧主化 + L0 直产 + 融合取数)。实现完成且
  // codes/输出逐位等价, 但实测净耗时高于 T13 基线(单线程 erf-GELU +
  // 每层串行 pack/gemm, 无池重叠), 任务卡目标 −50~60ms 未达成 → 默认关闭,
  // 仅留 GSV_H1_ON=1 复验入口(E15+ 多线程/流水化再评估)。
  {
    const char* h1env = std::getenv("GSV_H1_ON");    // 复验档(默认关)
    const bool h1_on =
        amx_hubert_enabled() && amx_hubert_mode() == AmxEncMode::kAll &&
        h1env && h1env[0] == '1';
    if (h1_on) {
      size_t t6 = cnn_stack_fused(waveform, n);
      if (micro_on())
        std::fprintf(stderr, "[hubert-h1] dispatch t6=%zu (旗=%d mode=%d)\n",
                     t6, static_cast<int>(amx_hubert_enabled()),
                     static_cast<int>(amx_hubert_mode()));
      if (t6 > 0) {
        len = t6;
        h1_fast = true;
      }
    }
  }
#endif
  if (!h1_fast) {
    cur_.assign(waveform, waveform + n);  // [1, N] 通道×时间布局
    for (int li = 0; li < 7; ++li) {
    size_t nl = 0;
    conv_layer(li, cur_, c, len, nxt_, c, nl);
    const auto tp_g0 = micro_on() ? clk::now() : clk::time_point{};
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
        for (size_t t = 0; t < nl; ++t)
          p[t] = static_cast<float>((p[t] - mu) * rstd * g + b);
      }
    }
    const auto tp_g1 = micro_on() ? clk::now() : tp_g0;
    if (micro_on()) g_micro.cnn_gn += ms_since(tp_g0, tp_g1);
    gelu(nxt_.data(), nxt_.size());
    if (micro_on()) {
      const auto tp_g2 = clk::now();
      g_micro.cnn_gelu += ms_since(tp_g1, tp_g2);
    }
    len = nl;
    cur_.swap(nxt_);
  }
  }
  cnn_t_ = len;
  const auto tp_cnn1 = clk::now();

  // ---- feature_projection: 转置到帧主 [T,512] → LN(512) → Linear(+bias) ----
  const size_t T = len;
  const auto tp_pr0 = micro_on() ? clk::now() : clk::time_point{};
  if (!h1_fast) {
    cnn_ = cur_;       // [512, T'] 通道主
    const size_t T2 = len;
    tmp_.resize(T2 * conv_dim_[6]);
    for (size_t t = 0; t < T2; ++t)
      for (int ch = 0; ch < conv_dim_[6]; ++ch)
        tmp_[t * conv_dim_[6] + ch] = cnn_[static_cast<size_t>(ch) * T2 + t];
  }
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
  if (micro_on()) g_micro.proj_seg += ms_since(tp_pr0, clk::now());

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
    const auto tp_pi0 = micro_on() ? clk::now() : clk::time_point{};
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
    const auto tp_pi1 = micro_on() ? clk::now() : tp_pi0;
    if (micro_on()) g_micro.pos_im2col += ms_since(tp_pi0, tp_pi1);
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
    if (micro_on()) g_micro.pos_gemm += ms_since(tp_pi1, clk::now());
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
    const auto tp_qk0 = micro_on() ? clk::now() : clk::time_point{};
    float* qp = qkv_.data();
    float* kp = qp + T * hidden_;
    float* vp = kp + T * hidden_;
#if defined(GSV_AMX_GEMM)
    if (layer_qkv_batch_ready(L) && T >= kAmxEncMinRows) {
      // E12: QKV 同批三联派发(共享激活 panel, 省两次池往返 — E8 同配方)
      cast_pack_B_f32(x_.data(), T, size_t(hidden_), hub_act_scratch_);
      const auto tp_qk1 = micro_on() ? clk::now() : tp_qk0;
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
      if (micro_on()) {
        const auto tp_qk2 = clk::now();
        g_micro.act_pack += ms_since(tp_qk0, tp_qk1);
        g_micro.amx_math += ms_since(tp_qk1, tp_qk2);
      }
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
    if (micro_on()) g_micro.enc_qkv += ms_since(tp_qk0, clk::now());
    // SDPA(无掩码): 每 head 独立 —— T11 保序 NEON 后端(位级等价, 详见上方内核注释)
    // T14 裁决放行(差分归因基准=相对部署基线零新增翻转, 实证 0/56):
    //   随 --amx-enc 默认启用; GSV_T12_SDPA_AMX=0 可显式关闭(复现/仲裁档)
    const auto tp_sdpa0 = sdpaTim ? clk::now() : clk::time_point{};
    att_.assign(T * hidden_, 0.f);
#if defined(GSV_AMX_GEMM) && defined(__aarch64__)
    static const bool t12_sdpa_on =
        [] { const char* e = std::getenv("GSV_T12_SDPA_AMX"); return !(e && e[0] == '0'); }();
    if (t12_sdpa_on && amx_hubert_enabled() && amx_hubert_mode() == AmxEncMode::kAll) {
      sdpa_amx_sgemm(qp, kp, vp, T, hd, scale, att_.data());
    } else
#endif
#if defined(__aarch64__)
    {
      // 全头 score[heads,T,T] 复用 sc_ 缓冲; 每头: QK^T → 逐行 softmax → PV
      if (sc_.size() < heads_ * T * T) sc_.resize(heads_ * T * T);
      ovh_.resize(T * hd);
      for (int h = 0; h < heads_; ++h) {
        const size_t off = size_t(h) * hd;
        float* sc_head = sc_.data() + size_t(h) * T * T;
        hubert_sdpa_qkt(qp, kp, T, size_t(hidden_), off, hd, scale, sc_head);
        for (size_t q = 0; q < T; ++q)
          gsv::kern::softmax(sc_head + q * T, sc_head + q * T, T);
        hubert_sdpa_pv(sc_head, vp, T, size_t(hidden_), off, hd, ovh_.data());
        for (size_t q = 0; q < T; ++q)
          std::memcpy(att_.data() + q * hidden_ + off, ovh_.data() + q * hd,
                      hd * sizeof(float));
      }
    }
#else
    // 非 aarch64 兜底: 原 NEON-free 标量三重循环(同源拷贝)
    for (int h = 0; h < heads_; ++h) {
      const size_t off = size_t(h) * hd;
      for (size_t q = 0; q < T; ++q) {
        const float* qv = qp + q * hidden_ + off;
        for (size_t kk = 0; kk < T; ++kk) {
          const float* kv = kp + kk * hidden_ + off;
          float dot = 0.f;
          for (size_t e = 0; e < hd; ++e) dot += qv[e] * kv[e];
          smax_[kk] = dot * scale;
        }
        gsv::kern::softmax(smax_.data(), smax_.data(), T);
        float* ov = att_.data() + q * hidden_ + off;
        for (size_t kk = 0; kk < T; ++kk) {
          const float p = smax_[kk];
          const float* vv = vp + kk * hidden_ + off;
          for (size_t e = 0; e < hd; ++e) ov[e] += p * vv[e];
        }
      }
    }
#endif
    if (sdpaTim)
      sdpa_ms_ += std::chrono::duration<double, std::milli>(clk::now() - tp_sdpa0).count();
    // out proj + 残差 + post-LN
    const auto tp_op0 = micro_on() ? clk::now() : clk::time_point{};
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
    if (micro_on()) g_micro.enc_outp += ms_since(tp_op0, clk::now());
    const auto tp_ar0 = micro_on() ? clk::now() : clk::time_point{};
    for (size_t i = 0; i < T * hidden_; ++i) x_[i] += resid_[i];
    const auto tp_al0 = micro_on() ? clk::now() : tp_ar0;
    for (size_t t = 0; t < T; ++t)
      gsv::kern::layernorm(x_.data() + t * hidden_, L.ln1_g.data(), L.ln1_b.data(),
                           x_.data() + t * hidden_, hidden_, ln_eps_);
    if (micro_on()) {
      g_micro.enc_resid += ms_since(tp_ar0, tp_al0);
      g_micro.enc_ln += ms_since(tp_al0, clk::now());
    }
    if (l == 0) cap_l0ln1_ = x_;
    // FFN(GELU) + 残差 + final LN
    ff_.resize(T * inter_);
    const auto tp_f10 = micro_on() ? clk::now() : clk::time_point{};
#if defined(GSV_AMX_GEMM)
    if (L.f1.w_panel_ready && T >= kAmxEncMinRows) {
      dense_amx(L.f1, x_.data(), T, ff_.data(), hub_act_scratch_, size_t(hidden_), L.f1.b);
    } else
#endif
    {
      L.f1.w.forward(x_.data(), T, ff_.data(), xh_);
      for (size_t i = 0; i < T * inter_; ++i) ff_[i] += L.f1.b[i % inter_];
    }
    if (micro_on()) g_micro.enc_f1 += ms_since(tp_f10, clk::now());
    const auto tp_ge0 = micro_on() ? clk::now() : clk::time_point{};
    gelu(ff_.data(), ff_.size());
    if (micro_on()) g_micro.enc_gelu += ms_since(tp_ge0, clk::now());
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
  if (micro_on()) {
    std::fprintf(stderr,
                 "[hubert-micro] cnn: im2col=%.1f gemm=%.1f gn=%.1f gelu=%.1f | "
                 "pos: im2col=%.1f gemm=%.1f proj=%.1f | enc: qkv=%.1f(outp=%.1f "
                 "f1=%.1f f2=%.1f) gelu=%.1f resid=%.1f ln=%.1f | amx: pack=%.1f "
                 "math=%.1f\n",
                 g_micro.cnn_im2col, g_micro.cnn_gemm, g_micro.cnn_gn,
                 g_micro.cnn_gelu, g_micro.pos_im2col, g_micro.pos_gemm,
                 g_micro.proj_seg, g_micro.enc_qkv, g_micro.enc_outp,
                 g_micro.enc_f1, g_micro.enc_f2, g_micro.enc_gelu,
                 g_micro.enc_resid, g_micro.enc_ln, g_micro.act_pack,
                 g_micro.amx_math);
    // 重置累计器(仅探针模式下清理; 探针不改行为, 正常模式零开销)
    g_micro = MicroAcc{};
  }
  return T;
}

}  // namespace gsv::encoder
