// sv.cpp — ERes2NetV2 推理实现(E2-ENC fp16 化: 权重 fp16 直读 + FMLAL 矩阵乘,
// LayerNorm/BN/残差/统计量保持 fp32; 激活转 fp16 暂存复用 scratch)
#include "encoder/sv.hpp"

#include "kern/accel.hpp"
#include "kern/gemv_fmlal.hpp"
#include "kern/kern.hpp"
#include "runtime/gsv_loader.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace gsv::encoder {

namespace accel = kern::accel;

using rt::GsvFile;
using rt::TensorView;

namespace {

inline float relu20(float v) { return v < 0.f ? 0.f : (v > 20.f ? 20.f : v); }

// dtype 无关读取: 大矩阵为 f16 存储, 小张量(BN/LN/bias/stem conv 等)为 fp32 存储
std::vector<float> vec_any(const GsvFile& f, const std::string& name) {
  const TensorView* t = f.tensor(name);
  if (!t) throw std::runtime_error("sv 缺张量: " + name);
  std::vector<float> v(t->numel());
  if (t->has_f16())
    kern::accel::f16_to_f32(t->data_f16_raw(), v.data(), v.size());
  else
    std::memcpy(v.data(), t->data_f32(), v.size() * sizeof(float));
  return v;
}

SvEngine::Bn load_bn(const GsvFile& f, const std::string& p) {
  SvEngine::Bn bn;
  bn.g = vec_any(f, p + ".weight");
  bn.b = vec_any(f, p + ".bias");
  bn.mean = vec_any(f, p + ".running_mean");
  bn.var = vec_any(f, p + ".running_var");
  return bn;
}

// 卷积权重: fp16 直读(零静态转换膨胀); 调用方在 conv2d_f16 / Aff::apply 内按需转 fp32 scratch
const uint16_t* load_conv_w(const GsvFile& f, const std::string& name) {
  const TensorView* t = f.tensor(name);
  if (!t) throw std::runtime_error("sv 缺张量: " + name);
  if (!t->has_f16())
    throw std::runtime_error("sv conv 权重应为 fp16 直读: " + name);
  return t->data_f16_raw();
}

}  // namespace

// ---- Bn ----
void SvEngine::Bn::apply(float* x, int c, size_t s) const {
  constexpr double eps = 1e-5;  // torch BatchNorm2d 默认(eval: running stats)
  for (int ci = 0; ci < c; ++ci) {
    const double a =
        static_cast<double>(g[ci]) / std::sqrt(static_cast<double>(var[ci]) + eps);
    const double beta = static_cast<double>(b[ci]);
    const double mu = static_cast<double>(mean[ci]);
    float* row = x + static_cast<size_t>(ci) * s;
    for (size_t i = 0; i < s; ++i)
      row[i] = static_cast<float>((static_cast<double>(row[i]) - mu) * a + beta);
  }
}

// ---- conv2d(im2col → sgemm('N','T') 直接产出通道主输出) ----
void SvEngine::conv2d(const float* in, int c_in, int h, int w, const float* wt,
                      int c_out, int kh, int kw, int stride, int pad,
                      std::vector<float>& cols, std::vector<float>& out) {
  const int oh = (h + 2 * pad - kh) / stride + 1;
  const int ow = (w + 2 * pad - kw) / stride + 1;
  const size_t S = static_cast<size_t>(oh) * ow;
  const int K = c_in * kh * kw;
  cols.resize(S * K);
  for (int oy = 0; oy < oh; ++oy)
    for (int ox = 0; ox < ow; ++ox) {
      float* row = cols.data() + (static_cast<size_t>(oy) * ow + ox) * K;
      for (int c = 0; c < c_in; ++c)
        for (int ky = 0; ky < kh; ++ky)
          for (int kx = 0; kx < kw; ++kx) {
            const int iy = oy * stride - pad + ky;
            const int ix = ox * stride - pad + kx;
            row[(static_cast<size_t>(c) * kh + ky) * kw + kx] =
                (iy >= 0 && iy < h && ix >= 0 && ix < w)
                    ? in[(static_cast<size_t>(c) * h + iy) * w + ix]
                    : 0.f;
          }
    }
  out.assign(static_cast<size_t>(c_out) * S, 0.f);
  accel::sgemm('N', 'T', c_out, static_cast<int>(S), K, 1.0f, wt, K, cols.data(), K,
               0.0f, out.data(), static_cast<int>(S));
}

// fp16 直读版 conv2d: im2col 量化到 cols16_(fp16) + gemm_f16x_fmlal
// (out[c_out,S] = W[c_out,K]·cols[S,K]ᵀ, FMLAL 融合 fp32 累加; 通道主输出)
void SvEngine::conv2d_f16(const float* in, int c_in, int h, int w, const uint16_t* w16,
                         int c_out, int kh, int kw, int stride, int pad,
                         std::vector<float>& /*cols_unused*/,
                         std::vector<float>& out) {
  const int oh = (h + 2 * pad - kh) / stride + 1;
  const int ow = (w + 2 * pad - kw) / stride + 1;
  const size_t S = static_cast<size_t>(oh) * ow;
  const size_t K = static_cast<size_t>(c_in) * kh * kw;
  cols16_.resize(S * K);
  for (int oy = 0; oy < oh; ++oy)
    for (int ox = 0; ox < ow; ++ox) {
      uint16_t* row = cols16_.data() + (static_cast<size_t>(oy) * ow + ox) * K;
      for (int c = 0; c < c_in; ++c)
        for (int ky = 0; ky < kh; ++ky)
          for (int kx = 0; kx < kw; ++kx) {
            const int iy = oy * stride - pad + ky;
            const int ix = ox * stride - pad + kx;
            const __fp16 hv =
                (iy >= 0 && iy < h && ix >= 0 && ix < w)
                    ? static_cast<__fp16>(
                          in[(static_cast<size_t>(c) * h + iy) * w + ix])
                    : __fp16(0);
            std::memcpy(row + (static_cast<size_t>(c) * kh + ky) * kw + kx, &hv,
                        sizeof hv);
          }
    }
  out.assign(static_cast<size_t>(c_out) * S, 0.f);
  kern::gemm_f16x_fmlal(w16, cols16_.data(), out.data(), static_cast<size_t>(c_out),
                        S, K);
}

// ---- Aff(fusion.AFF): xo = x*(1+tanh(att)) + ds*(1-tanh(att)) ----
void SvEngine::Aff::apply(const float* x, const float* ds, float* out, int h, int w,
                           std::vector<uint16_t>& xh, std::vector<uint16_t>& xh2) {
  const size_t s = static_cast<size_t>(h) * w;
#ifdef C1_TRACE
  std::fprintf(stderr, "    [aff ch=%d inter=%d s=%zu w1=%p w2=%p bn1=%zu/%zu bn2=%zu/%zu]\n",
               ch, inter, s, (const void*)w1, (const void*)w2, bn1.g.size(), bn1.var.size(),
               bn2.g.size(), bn2.var.size());
#endif
  cat_.resize(2 * static_cast<size_t>(ch) * s);
  std::memcpy(cat_.data(), x, static_cast<size_t>(ch) * s * sizeof(float));
  std::memcpy(cat_.data() + static_cast<size_t>(ch) * s, ds,
              static_cast<size_t>(ch) * s * sizeof(float));
  att_.resize(static_cast<size_t>(inter) * s);
  // gemm1: att[inter,S] = W1[inter,2ch]·cat[2ch,S]。收缩维 2ch 在 cat 的行向 ——
  // 转置量化 cat→[S,2ch]fp16 后 gemm(C[S,inter]=catᵀ·W1ᵀ) 再转回通道主。
  if (xh.size() < 2 * static_cast<size_t>(ch) * s) xh.resize(2 * static_cast<size_t>(ch) * s);
  kern::f32_trans_to_f16(cat_.data(), xh.data(), 2 * ch, s);  // [2ch,S]→[S,2ch]fp16
  att_t_.resize(static_cast<size_t>(inter) * s);  // 暂存 attᵀ [S,inter]
  kern::gemm_f16x_fmlal(xh.data(), w1, att_t_.data(), s,
                        static_cast<size_t>(inter), 2 * static_cast<size_t>(ch));
  for (size_t ss = 0; ss < s; ++ss)
    for (int e = 0; e < inter; ++e)
      att_[static_cast<size_t>(e) * s + ss] =
          att_t_[ss * static_cast<size_t>(inter) + static_cast<size_t>(e)];
#ifdef C1_TRACE
  std::fprintf(stderr, "    [aff gemm1 done]\n");
#endif
  for (int e = 0; e < inter; ++e) {
    float* row = att_.data() + static_cast<size_t>(e) * s;
    const float b = b1[static_cast<size_t>(e)];
    for (size_t i = 0; i < s; ++i) row[i] += b;
  }
  bn1.apply(att_.data(), inter, s);
  gsv::kern::silu(att_.data(), att_.data(), att_.size());
  att_t_.resize(static_cast<size_t>(ch) * s);
  // gemm2 同构: att_t[ch,S] = W2[ch,inter]·att[inter,S] → 转置量化 + gemm + 转回
  if (xh2.size() < static_cast<size_t>(inter) * s) xh2.resize(static_cast<size_t>(inter) * s);
  kern::f32_trans_to_f16(att_.data(), xh2.data(), inter, s);  // [inter,S]→[S,inter]fp16
  cat_.resize(static_cast<size_t>(ch) * s);  // 复用 cat_ 暂存 gemm2 输出ᵀ [S,ch]
  kern::gemm_f16x_fmlal(xh2.data(), w2, cat_.data(), s,
                        static_cast<size_t>(ch), static_cast<size_t>(inter));
  for (size_t ss = 0; ss < s; ++ss)
    for (int e = 0; e < ch; ++e)
      att_t_[static_cast<size_t>(e) * s + ss] =
          cat_[ss * static_cast<size_t>(ch) + static_cast<size_t>(e)];
#ifdef C1_TRACE
  std::fprintf(stderr, "    [aff gemm2 done]\n");
#endif
  for (int e = 0; e < ch; ++e) {
    float* row = att_t_.data() + static_cast<size_t>(e) * s;
    const float b = b2[static_cast<size_t>(e)];
    for (size_t i = 0; i < s; ++i) row[i] += b;
  }
  bn2.apply(att_t_.data(), ch, s);
  for (size_t i = 0; i < static_cast<size_t>(ch) * s; ++i) {
    const float t = std::tanh(att_t_[i]);
    out[i] = x[i] * (1.0f + t) + ds[i] * (1.0f - t);
  }
  last_out.assign(out, out + static_cast<size_t>(ch) * s);
}

// ---- Block ----
void SvEngine::Block::apply(const float* in, int c_in, int h_in, int w_in,
                            SvEngine& eng) {
  const int co1 = width * scale;
#ifdef C1_TRACE
  std::fprintf(stderr, "  [blk aff=%d w=%d sc=%d exp=%d st=%d cin=%d %dx%d]\n",
               aff, width, scale, exp_planes, stride, c_in, h_in, w_in);
#endif
  // conv1(k1, stride, p0) + bn1 + ReLU20
  // 注意: in 可能就是 eng.tmp_(调用方把主缓冲喂进来) —— 必须写入另一缓冲防别名。
  int h = h_in, w = w_in;
  if (stride == 1) {
    // 1x1 s1: 零 im2col。通道主 [c_in,S]·Wᵀ[co1,c_in] —— 转置量化激活后 FMLAL:
    //   nxt[co1,S] = W·inᵀ → 先算 [S,co1]=inᵀ·Wᵀ 再转回通道主。
    const size_t s = static_cast<size_t>(h) * w;
    if (eng.xh_.size() < static_cast<size_t>(c_in) * s)
      eng.xh_.resize(static_cast<size_t>(c_in) * s);
    kern::f32_trans_to_f16(in, eng.xh_.data(), c_in, s);  // [c_in,S]→[S,c_in]fp16
    eng.tmp_.resize(static_cast<size_t>(co1) * s);        // 暂存 [S,co1]
    kern::gemm_f16x_fmlal(eng.xh_.data(), conv1_w, eng.tmp_.data(), s,
                          static_cast<size_t>(co1), static_cast<size_t>(c_in));
    eng.nxt_.resize(static_cast<size_t>(co1) * s);
    for (size_t ss = 0; ss < s; ++ss)
      for (int e = 0; e < co1; ++e)
        eng.nxt_[static_cast<size_t>(e) * s + ss] =
            eng.tmp_[ss * static_cast<size_t>(co1) + static_cast<size_t>(e)];
  } else {
    eng.conv2d_f16(in, c_in, h, w, conv1_w, co1, 1, 1, stride, 0, eng.cols_, eng.nxt_);
    h = (h - 1) / stride + 1;
    w = (w - 1) / stride + 1;
  }
  const size_t s = static_cast<size_t>(h) * w;
  bn1.apply(eng.nxt_.data(), co1, s);
  for (float& v : eng.nxt_) v = relu20(v);
#ifdef C1_TRACE
  std::fprintf(stderr, "  [conv1+bn ok s=%zu]\n", s);
#endif

  // 分块处理链(torch 语义): sp 从原始块 0 出发, 每轮处理后作为下一轮 fuse/相加的左操作数
  eng.cur_.assign(eng.nxt_.begin(), eng.nxt_.end());  // spx([co1,s])
  eng.nxt_.clear();
  eng.tmp2_.resize(static_cast<size_t>(co1) * s);     // 拼接缓冲(槽位 i × width)
  eng.nxt_.reserve(static_cast<size_t>(width) * s);
  auto chunk = [&](int i) -> const float* {
    return eng.cur_.data() + static_cast<size_t>(i) * width * s;
  };
  std::vector<float> sp(static_cast<size_t>(width) * s);
  std::vector<float> fus(static_cast<size_t>(width) * s);
  for (int i = 0; i < scale; ++i) {
    if (i > 0) {
#ifdef C1_TRACE
      if (aff) std::fprintf(stderr, "  [fuse i=%d]\n", i);
#endif
      if (aff) {
        fuses[static_cast<size_t>(i - 1)].apply(sp.data(), chunk(i), fus.data(), h, w,
                                               eng.xh_, eng.xh2_);
        std::memcpy(sp.data(), fus.data(), sp.size() * sizeof(float));
      } else {
        for (size_t j = 0; j < sp.size(); ++j) sp[j] += chunk(i)[j];
      }
    }
    // convs[i](3x3 p1 s1) + bns + ReLU20 → 拼接槽位 i
    eng.conv2d_f16(i == 0 ? chunk(0) : sp.data(), width, h, w, convs_w[static_cast<size_t>(i)],
                   width, 3, 3, 1, 1, eng.cols_, eng.nxt_);
    bns[static_cast<size_t>(i)].apply(eng.nxt_.data(), width, s);
    for (float& v : eng.nxt_) v = relu20(v);
    std::memcpy(eng.tmp2_.data() + static_cast<size_t>(i) * width * s, eng.nxt_.data(),
                eng.nxt_.size() * sizeof(float));
    std::memcpy(sp.data(), eng.nxt_.data(), eng.nxt_.size() * sizeof(float));
  }

  // conv3(k1) + bn3
  eng.conv2d_f16(eng.tmp2_.data(), co1, h, w, conv3_w, exp_planes, 1, 1, 1, 0,
                 eng.cols_, eng.nxt_);
  bn3.apply(eng.nxt_.data(), exp_planes, s);

#ifdef C1_TRACE
  std::fprintf(stderr, "  [conv3 done, shortcut=%d]\n", has_shortcut);
#endif
  // shortcut(输入仍是 [c_in,h_in,w_in], 经 stride 下采样到输出分辨率 —— 必须用原始 h_in/w_in!)
  if (!has_shortcut) {
    for (size_t j = 0; j < eng.nxt_.size(); ++j) eng.nxt_[j] += in[j];
  } else {
    eng.conv2d_f16(in, c_in, h_in, w_in, sc_w, exp_planes, 1, 1, stride, 0, eng.cols_,
                   eng.tmp_);
    sc_bn.apply(eng.tmp_.data(), exp_planes, s);
    for (size_t j = 0; j < eng.nxt_.size(); ++j) eng.nxt_[j] += eng.tmp_[j];
  }
  for (float& v : eng.nxt_) v = relu20(v);
#ifdef C1_TRACE
  std::fprintf(stderr, "  [blk done]\n");
#endif
  last_out.assign(eng.nxt_.begin(), eng.nxt_.end());
}

// ---- 加载 ----
void SvEngine::load_block(int l, int i, bool expect_aff) {
  const std::string p = "layer" + std::to_string(l) + "." + std::to_string(i) + ".";
  Block& blk = stages_[l][static_cast<size_t>(i)];
  blk.aff = expect_aff;
  const TensorView* cw = f_->tensor(p + "convs.0.weight");
  if (!cw) throw std::runtime_error("sv 缺 " + p + "convs.0.weight");
  blk.width = static_cast<int>(cw->dims[0]);
  blk.scale = 4;  // eres2netv2w24s4ep4; 由下方张量计数校验
  blk.exp_planes =
      static_cast<int>(f_->tensor((p + "bn3.weight").c_str())->dims[0]);

  int n_convs = 0;
  while (f_->tensor((p + "convs." + std::to_string(n_convs) + ".weight").c_str()))
    ++n_convs;
  if (n_convs != blk.scale)
    throw std::runtime_error("sv block scale 与 convs 数量不符: " + p);

  blk.conv1_w = load_conv_w(*f_, p + "conv1.weight");
  blk.bn1 = load_bn(*f_, p + "bn1");
  for (int j = 0; j < blk.scale; ++j) {
    blk.convs_w.push_back(
        load_conv_w(*f_, p + "convs." + std::to_string(j) + ".weight"));
    blk.bns.push_back(load_bn(*f_, p + "bns." + std::to_string(j)));
  }
  blk.conv3_w = load_conv_w(*f_, p + "conv3.weight");
  blk.bn3 = load_bn(*f_, p + "bn3");
  if (blk.aff) {
    for (int j = 0; j < blk.scale - 1; ++j) {
      Aff a;
      const std::string fp = p + "fuse_models." + std::to_string(j) + ".local_att.";
      a.inter = static_cast<int>(
          f_->tensor((fp + "0.weight").c_str())->dims[0]);  // [inter, 2*width,1,1]
      a.ch = blk.width;
      a.w1 = load_conv_w(*f_, fp + "0.weight");
      a.w2 = load_conv_w(*f_, fp + "3.weight");
      a.b1 = vec_any(*f_, fp + "0.bias");   // Conv2d 默认 bias=True!
      a.b2 = vec_any(*f_, fp + "3.bias");
      a.bn1 = load_bn(*f_, fp + "1");
      a.bn2 = load_bn(*f_, fp + "4");
      blk.fuses.push_back(std::move(a));
    }
  }
  blk.has_shortcut = f_->tensor((p + "shortcut.0.weight").c_str()) != nullptr;
  if (blk.has_shortcut) {
    blk.sc_w = load_conv_w(*f_, p + "shortcut.0.weight");
    blk.sc_bn = load_bn(*f_, p + "shortcut.1");
  }
}

SvEngine::SvEngine(const GsvFile& f) : f_(&f) {
  conv1_w_ = vec_any(f, "conv1.weight");  // stem conv: fp32 常驻(仅 576 参, 无 f16 段)
  bn1_ = load_bn(f, "bn1");

  struct StageDef {
    int blocks, stride;
    bool aff;
  };
  constexpr StageDef defs[4] = {{3, 1, false}, {4, 2, false}, {6, 2, true}, {3, 2, true}};
  o_layer_.resize(5);
  for (int l = 1; l <= 4; ++l) {
    stages_[l].resize(defs[l - 1].blocks);
    for (int i = 0; i < defs[l - 1].blocks; ++i) {
      Block& blk = stages_[l][static_cast<size_t>(i)];
      load_block(l, i, defs[l - 1].aff);
      blk.stride = (i == 0) ? defs[l - 1].stride : 1;
    }
    c_after_[l] = stages_[l].back().exp_planes;
  }
  l3ds_w_ = load_conv_w(f, "layer3_ds.weight");  // [2048,1024,3,3]
  fuse34_.inter = static_cast<int>(
      f.tensor("fuse34.local_att.0.weight")->dims[0]);  // 512
  fuse34_.ch =
      static_cast<int>(f.tensor("fuse34.local_att.3.weight")->dims[0]);  // 2048
  fuse34_.w1 = load_conv_w(f, "fuse34.local_att.0.weight");
  fuse34_.w2 = load_conv_w(f, "fuse34.local_att.3.weight");
  fuse34_.b1 = vec_any(f, "fuse34.local_att.0.bias");
  fuse34_.b2 = vec_any(f, "fuse34.local_att.3.bias");
  fuse34_.bn1 = load_bn(f, "fuse34.local_att.1");
  fuse34_.bn2 = load_bn(f, "fuse34.local_att.4");
}

size_t SvEngine::forward3(const float* fbk, size_t frames) {
  // fbank [T,80] 行主 → [C=80][H=80 频率维][W=T 时间维]
  const int F = 80;
  const int T = static_cast<int>(frames);
#ifdef C1_TRACE
  std::fprintf(stderr, "[fwd3 T=%d]\n", T);
#endif
  cur_.resize(static_cast<size_t>(F) * T);
  for (int t = 0; t < T; ++t)
    for (int fch = 0; fch < F; ++fch)
      cur_[static_cast<size_t>(fch) * T + t] = fbk[static_cast<size_t>(t) * F + fch];

  // conv1(3x3 p1 s1) + bn1 + ReLU20
  conv2d(cur_.data(), 1, F, T, conv1_w_.data(), 64, 3, 3, 1, 1, cols_, tmp_);
  const int m_ch = static_cast<int>(bn1_.g.size());
  bn1_.apply(tmp_.data(), m_ch, static_cast<size_t>(F) * T);
  // 注意: 主干入口是 F.relu(torch.nn.functional), 不是块内的 Hardtanh(0,20)
  for (float& v : tmp_) v = v < 0.f ? 0.f : v;
  o_conv1_ = tmp_;

  // layer1..4(逐 stage 跟踪分辨率)
  int h = F, w = T, c = m_ch;
#ifdef C1_TRACE
  std::fprintf(stderr, "[conv1 done h=%d w=%d]\n", h, w);
#endif
  int h3 = h, w3 = w;  // layer3 输出分辨率(layer3_ds 用)
  for (int l = 1; l <= 4; ++l) {
    for (Block& blk : stages_[l]) {
#ifdef C1_TRACE
      std::fprintf(stderr, "[L%d blk in=%dx%dx%d]\n", l, c, h, w);
#endif
      blk.apply(tmp_.data(), c, h, w, *this);
      tmp_.swap(nxt_);
      if (blk.stride != 1) {
        h = (h - 1) / blk.stride + 1;
        w = (w - 1) / blk.stride + 1;
      }
      c = blk.exp_planes;
    }
    o_layer_[static_cast<size_t>(l)] = tmp_;
    if (l == 3) {
      h3 = h;
      w3 = w;
    }
  }

  // layer3_ds(1024→2048, k3, s2, p1) 作用在 layer3 输出上(fp16 直读)
  conv2d_f16(o_layer_[3].data(), c_after_[3], h3, w3, l3ds_w_, c_after_[4], 3, 3, 2,
         1, cols_, ds_);

  // fuse34(AFF) on (layer4 输出, ds)
  o_fuse_.resize(ds_.size());
  fuse34_.apply(tmp_.data(), ds_.data(), o_fuse_.data(), h, w, xh_, xh2_);

  // flatten(C×F) → mean over T
  const int C = c_after_[4];  // 2048
  emb_.assign(static_cast<size_t>(C) * h, 0.f);
  for (int ci = 0; ci < C; ++ci)
    for (int fy = 0; fy < h; ++fy) {
      const float* row = o_fuse_.data() + (static_cast<size_t>(ci) * h + fy) * w;
      double acc = 0.0;
      for (int t = 0; t < w; ++t) acc += row[t];
      emb_[static_cast<size_t>(ci) * h + fy] = static_cast<float>(acc / w);
    }
  return emb_.size();
}

}  // namespace gsv::encoder
