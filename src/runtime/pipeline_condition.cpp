// pipeline_condition.cpp — C2: decode-condition 数值实现 (口径见 .hpp)
#include "runtime/pipeline_condition.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numbers>

#include "kern/accel.hpp"
#include "sovits/sovits_types.hpp"  // load_tensor_f32 (f16 自动升位)

namespace gsv::rt::pipeline {

using kern::accel::sgemm;

// ---------- 权重加载 ----------
void ConditionBuilder::Lin::run(const float* x, float* y) const {
  // y[out] = W[out,in] · x[in] + b
  //   ≡ 行主 y[1,out] = x[1,in]·Wᵀ (与 kern accel 的 'N','T' 约定一致)
  sgemm('N', 'T', 1, int(out), int(in), 1.0f, x, int(in), W.data(), int(in),
        0.0f, y, int(out));
  for (size_t o = 0; o < out; ++o) y[o] += b[o];
}

static void expectTensor(const rt::GsvFile& f, std::string_view name,
                         std::vector<float>& dst,
                         std::initializer_list<size_t> dims) {
  sovits::load_tensor_f32(f, name, dst, dims);
}

void ConditionBuilder::load(const rt::GsvFile& f) {
  expectTensor(f, "ref_enc.spectral.0.fc.weight", sp0_.W, {128, 704});
  expectTensor(f, "ref_enc.spectral.0.fc.bias", sp0_.b, {128});
  sp0_.in = 704;
  sp0_.out = 128;
  expectTensor(f, "ref_enc.spectral.3.fc.weight", sp3_.W, {128, 128});
  expectTensor(f, "ref_enc.spectral.3.fc.bias", sp3_.b, {128});
  sp3_.in = sp3_.out = 128;

  for (int l = 0; l < 2; ++l) {
    const std::string p = "ref_enc.temporal." + std::to_string(l) + ".conv1.conv.";
    expectTensor(f, p + "weight", tc_[l], {256, 128, 5});
    expectTensor(f, p + "bias", tb_[l], {256});
  }

  // MHA: nn.Linear bias=True; 若 gsv 无 bias 张量则按零处理
  auto loadLin = [&](std::string_view base, Lin* l, size_t d) {
    expectTensor(f, std::string(base) + ".weight", l->W, {d, d});
    l->in = l->out = d;
    const auto* bt = f.tensor(std::string(base) + ".bias");
    if (bt) {
      sovits::load_tensor_f32(f, std::string(base) + ".bias", l->b, {d});
    } else {
      l->b.assign(d, 0.f);
    }
  };
  loadLin("ref_enc.slf_attn.w_qs", &attWq_, 128);
  loadLin("ref_enc.slf_attn.w_ks", &attWk_, 128);
  loadLin("ref_enc.slf_attn.w_vs", &attWv_, 128);
  loadLin("ref_enc.slf_attn.fc", &attFc_, 128);

  expectTensor(f, "ref_enc.fc.fc.weight", fc_.W, {1024, 128});
  expectTensor(f, "ref_enc.fc.fc.bias", fc_.b, {1024});
  fc_.in = 128;
  fc_.out = 1024;

  expectTensor(f, "sv_emb.weight", svProj_.W, {1024, 20480});
  if (f.tensor("sv_emb.bias")) {
    expectTensor(f, "sv_emb.bias", svProj_.b, {1024});
  } else {
    svProj_.b.assign(1024, 0.f);
  }
  svProj_.in = 20480;
  svProj_.out = 1024;

  expectTensor(f, "ge_to512.weight", geTo512_.W, {512, 1024});
  expectTensor(f, "ge_to512.bias", geTo512_.b, {512});
  geTo512_.in = 1024;
  geTo512_.out = 512;

  expectTensor(f, "prelu.weight", preluSlope_, {1024});
}

// ---------- 线性谱 (torch.stft 口径) ----------
namespace {

// 迭代 radix-2 复数 FFT (n=2048, 每参考音频一次构建旋转因子)
struct Rfft2048 {
  std::vector<float> cosTab, sinTab;  // [n/2]
  std::vector<uint32_t> rev;          // 位反转
  static constexpr int N = 2048;
  Rfft2048() {
    cosTab.resize(N / 2);
    sinTab.resize(N / 2);
    for (int i = 0; i < N / 2; ++i) {
      const double a = -2.0 * std::numbers::pi_v<double> * i / N;
      cosTab[i] = float(std::cos(a));
      sinTab[i] = float(std::sin(a));
    }
    rev.resize(N);
    int bits = 0;
    for (int v = N; v > 1; v >>= 1) ++bits;
    for (uint32_t i = 0; i < uint32_t(N); ++i) {
      uint32_t r = 0;
      for (int b = 0; b < bits; ++b)
        if (i & (1u << b)) r |= 1u << (bits - 1 - b);
      rev[i] = r;
    }
  }
  void run(const float* re, float* outRe, float* outIm) const {
    alignas(64) float xr[N], xi[N];
    for (int i = 0; i < N; ++i) {
      xr[i] = re[rev[i]];
      xi[i] = 0.f;
    }
    for (int len = 2; len <= N; len <<= 1) {
      const int half = len >> 1, step = N / len;
      for (int i = 0; i < N; i += len) {
        for (int j = 0; j < half; ++j) {
          const float wr = cosTab[size_t(j) * step],
                      wi = sinTab[size_t(j) * step];
          const float ur = xr[i + j], ui = xi[i + j];
          const float vr = xr[i + j + half] * wr - xi[i + j + half] * wi;
          const float vi = xr[i + j + half] * wi + xi[i + j + half] * wr;
          xr[i + j] = ur + vr;
          xi[i + j] = ui + vi;
          xr[i + j + half] = ur - vr;
          xi[i + j + half] = ui - vi;
        }
      }
    }
    std::memcpy(outRe, xr, sizeof(xr));
    std::memcpy(outIm, xi, sizeof(xi));
  }
};

inline double hannPeriodic(int n, int N) {
  return 0.5 - 0.5 * std::cos(2.0 * std::numbers::pi_v<double> * n / N);
}

}  // namespace

void ConditionBuilder::spectrogram(const float* audio, size_t n,
                                   std::vector<float>& spec, size_t& frames) {
  static const Rfft2048 fft;
  // hann_window(2048, periodic=True)
  static std::vector<float> win;
  if (win.empty()) {
    win.resize(kWin);
    for (int i = 0; i < kWin; ++i) win[i] = float(hannPeriodic(i, kWin));
  }
  // 反射填充 (n_fft-hop)/2 = 704
  constexpr int kPad = (kFft - kHop) / 2;
  const int64_t N = int64_t(n);
  auto refl = [&](int64_t i) -> float {
    if (i < 0) i = -i;
    if (i >= N) i = 2 * (N - 1) - i;  // reflect 不含边界
    return audio[size_t(i)];
  };
  // 帧数按"先反射填充 704×2 再取窗"计: T = 1 + (n_pad - win) / hop
  const int64_t NPadded = N + 2 * kPad;
  frames = NPadded >= kFft ? size_t((NPadded - kFft) / kHop) + 1 : 1;
  spec.assign(kSpecBins * frames, 0.f);
  std::vector<float> buf(kFft), re(kFft), im(kFft);
  for (size_t t = 0; t < frames; ++t) {
    const int64_t off = int64_t(t) * kHop;
    for (int i = 0; i < kFft; ++i) buf[i] = refl(off + i - kPad) * win[i];
    fft.run(buf.data(), re.data(), im.data());
    float* col = spec.data() + t;  // [bin][T] 行主: bin 行内按帧步进
    for (int b = 0; b <= kFft / 2; ++b) {
      const double p = double(re[b]) * re[b] + double(im[b]) * im[b];
      col[size_t(b) * frames] = float(std::sqrt(p + 1e-8));
    }
  }
}

// ---------- ref_enc 主干 ----------
inline float ConditionBuilder::mish(float x) {
  // x·tanh(softplus(x)); softplus=log1p(exp(x)) 稳定形式
  const double xd = x;
  const double sp = xd > 20.0 ? xd : std::log1p(std::exp(xd));
  return float(xd * std::tanh(sp));
}

void ConditionBuilder::conv1dGlu(const float* in, size_t T, int layer,
                                 float* out) const {
  // in/out 均为 [C=128][T]; conv 权重 [256,128,5], pad=2
  // E13-MIX 前基线原样实现(默认路径位级红线; T8 从 818146b^ 原样恢复)
  const std::vector<float>& Wt = tc_[size_t(layer)];
  const std::vector<float>& B = tb_[size_t(layer)];
  std::vector<float> conv(size_t(256) * T, 0.f);
  for (size_t c = 0; c < 256; ++c) {
    float* dst = conv.data() + c * T;
    const float* wbase = Wt.data() + size_t(c) * 128 * 5;
    for (size_t t = 0; t < T; ++t) {
      double acc = double(B[c]);
      for (int k = 0; k < 5; ++k) {
        const int64_t ti = int64_t(t) + k - 2;
        if (ti < 0 || ti >= int64_t(T)) continue;  // zero padding
        double inner = 0.0;
        for (size_t ci = 0; ci < 128; ++ci)
          inner += double(wbase[size_t(ci) * 5 + size_t(k)]) *
                   in[ci * T + size_t(ti)];
        acc += inner;
      }
      dst[t] = float(acc);
    }
  }
  // GLU: 前 128 半 × σ(后 128 半) + 残差
  for (size_t c = 0; c < 128; ++c) {
    const float* x1 = conv.data() + c * T;
    const float* x2 = conv.data() + (128 + c) * T;
    for (size_t t = 0; t < T; ++t) {
      out[c * T + t] =
          x1[t] / (1.f + std::exp(-x2[t])) + in[c * T + t];
    }
  }
}

void ConditionBuilder::conv1dGluBatched(const float* in, size_t T, int layer,
                                        float* out) const {
  const std::vector<float>& Wt = tc_[size_t(layer)];
  const std::vector<float>& B = tb_[size_t(layer)];
  std::vector<float> x2(size_t(T) * 640, 0.f);
  for (size_t ci = 0; ci < 128; ++ci)
    for (int k = 0; k < 5; ++k) {
      float* col = x2.data() + size_t(ci) * 5 + size_t(k);
      for (size_t t = 0; t < T; ++t) {
        const int64_t ti = int64_t(t) + k - 2;
        if (ti < 0 || ti >= int64_t(T)) continue;  // 零填充(同基线)
        col[size_t(t) * 640] = in[ci * T + size_t(ti)];
      }
    }
  std::vector<float> y(size_t(T) * 256);
  sgemm('N', 'T', int(T), 256, 640, 1.0f, x2.data(), 640, Wt.data(), 640,
        0.0f, y.data(), 256);
  // bias + GLU + 残差(基线同公式)
  for (size_t t = 0; t < T; ++t) {
    const float* yr = y.data() + t * 256;
    for (size_t c = 0; c < 128; ++c) {
      const float g = (yr[size_t(c)] + B[size_t(c)]) /
                      (1.f + std::exp(-(yr[128 + size_t(c)] + B[128 + size_t(c)])));
      out[size_t(c) * T + t] = g + in[size_t(c) * T + t];
    }
  }
}

void ConditionBuilder::refEnc(const float* condIn, size_t T,
                              std::vector<float>& pooled) const {
  // T8: amxEnc_ 派发 — 默认 false 走基线标量实现(位级红线),
  //     --amx-enc 时走 T6 批量化实现(mel 门口径)。
  if (amxEnc_) refEncBatched(condIn, T, pooled);
  else         refEncScalar(condIn, T, pooled);
}

void ConditionBuilder::refEncScalar(const float* condIn, size_t T,
                              std::vector<float>& pooled) const {
  // E13 探针: ref_enc 内部分段(GSV_COND_TIMING 门控, 只计时不改行为)
  using clk = std::chrono::steady_clock;
  const bool condTim = std::getenv("GSV_COND_TIMING") != nullptr;
  auto cms = [](clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  // condIn: [T][704] → spectral → [T][128]
  std::vector<float> h(T * 128), tmp(T * 128);
  const auto tr0 = clk::now();
  for (size_t t = 0; t < T; ++t) sp0_.run(condIn + t * 704, h.data() + t * 128);
  for (size_t t = 0; t < T; ++t) {
    float* r = h.data() + t * 128;
    for (size_t c = 0; c < 128; ++c) r[c] = mish(r[c]);
  }
  for (size_t t = 0; t < T; ++t) sp3_.run(h.data() + t * 128, tmp.data() + t * 128);
  for (size_t t = 0; t < T; ++t) {
    float* r = tmp.data() + t * 128;
    for (size_t c = 0; c < 128; ++c) r[c] = mish(r[c]);
  }
  const auto tr1 = clk::now();

  // temporal: 转成 [C,T] 做 conv, 再转回 [T,C]
  std::vector<float> ct(128 * T), co(128 * T), tt(T * 128);
  for (size_t t = 0; t < T; ++t)
    for (size_t c = 0; c < 128; ++c) ct[c * T + t] = tmp[t * 128 + c];
  conv1dGlu(ct.data(), T, 0, co.data());
  conv1dGlu(co.data(), T, 1, ct.data());
  const auto tr2 = clk::now();
  for (size_t t = 0; t < T; ++t)
    for (size_t c = 0; c < 128; ++c) tt[t * 128 + c] = ct[c * T + t];

  // MHA (mask=None): q,k,v = Linear(x); 每 head 切片 64 维
  std::vector<float> q(T * 128), k(T * 128), v(T * 128), attnO(T * 128);
  for (size_t t = 0; t < T; ++t) {
    attWq_.run(tt.data() + t * 128, q.data() + t * 128);
    attWk_.run(tt.data() + t * 128, k.data() + t * 128);
    attWv_.run(tt.data() + t * 128, v.data() + t * 128);
  }
  const auto tr3 = clk::now();  // MHA 线性(q/k/v)结束
  const double temperature = std::sqrt(128.0);
  for (int head = 0; head < 2; ++head) {
    const size_t off = size_t(head) * 64;
    for (size_t ti = 0; ti < T; ++ti) {  // 对每个 query 位置
      std::vector<double> score(T);
      double mx = -1e30;
      for (size_t tj = 0; tj < T; ++tj) {
        double s = 0.0;
        for (size_t d = 0; d < 64; ++d)
          s += double(q[ti * 128 + off + d]) * k[tj * 128 + off + d];
        score[tj] = s / temperature;
        if (score[tj] > mx) mx = score[tj];
      }
      double sum = 0.0;
      for (size_t tj = 0; tj < T; ++tj) {
        score[tj] = std::exp(score[tj] - mx);
        sum += score[tj];
      }
      for (size_t d = 0; d < 64; ++d) {
        double acc = 0.0;
        for (size_t tj = 0; tj < T; ++tj)
          acc += (score[tj] / sum) * v[tj * 128 + off + d];
        attnO[ti * 128 + off + d] = float(acc);
      }
    }
  }
  const auto tr4 = clk::now();  // attention 核心结束
  // output = fc(attnO) + residual
  std::vector<float> fo(T * 128);
  for (size_t t = 0; t < T; ++t) {
    attFc_.run(attnO.data() + t * 128, fo.data() + t * 128);
    for (size_t c = 0; c < 128; ++c)
      fo[t * 128 + c] += tt[t * 128 + c];
  }
  const auto tr5 = clk::now();  // attFc 结束
  // fc → [T,1024], 时间平均池化
  std::vector<double> pool(1024, 0.0);
  std::vector<float> row(1024);
  for (size_t t = 0; t < T; ++t) {
    fc_.run(fo.data() + t * 128, row.data());
    for (size_t c = 0; c < 1024; ++c) pool[c] += row[c];
  }
  pooled.assign(1024, 0.f);
  for (size_t c = 0; c < 1024; ++c) pooled[c] = float(pool[c] / double(T));
  if (condTim)
    std::fprintf(stderr,
                 "[refenc-timing] spectral=%.1fms temporal_conv=%.1fms "
                 "qkv_lin=%.1fms attn_core=%.1fms attfc=%.1fms fc_pool=%.1fms T=%zu\n",
                 cms(tr0, tr1), cms(tr1, tr2), cms(tr2, tr3), cms(tr3, tr4),
                 cms(tr4, tr5), cms(tr5, clk::now()), T);
}

void ConditionBuilder::refEncBatched(const float* condIn, size_t T,
                              std::vector<float>& pooled) const {
  // E13 探针: ref_enc 内部分段(GSV_COND_TIMING 门控, 只计时不改行为)
  using clk = std::chrono::steady_clock;
  const bool condTim = std::getenv("GSV_COND_TIMING") != nullptr;
  auto cms = [](clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  // condIn: [T][704] → spectral → [T][128]
  // T6: sp0_/sp3_ 逐行标量 sgemm(每行单次 M=1 调用) → 整批矩阵积(M=T 一次)
  std::vector<float> h(T * 128), tmp(T * 128);
  const auto tr0 = clk::now();
  {  // h[T,128] = condIn[T,704] · W₀ᵀ + b₀ ; mish 原位
    sgemm('N', 'T', int(T), 128, 704, 1.0f, condIn, 704, sp0_.W.data(), 704,
          0.0f, h.data(), 128);
    for (size_t t = 0; t < T; ++t) {
      float* r = h.data() + t * 128;
      for (int c = 0; c < 128; ++c) r[c] += sp0_.b[size_t(c)];
      for (size_t c = 0; c < 128; ++c) r[c] = mish(r[c]);
    }
  }
  {  // tmp[T,128] = h[T,128] · W₃ᵀ + b₃ ; mish
    sgemm('N', 'T', int(T), 128, 128, 1.0f, h.data(), 128, sp3_.W.data(), 128,
          0.0f, tmp.data(), 128);
    for (size_t t = 0; t < T; ++t) {
      float* r = tmp.data() + t * 128;
      for (int c = 0; c < 128; ++c) r[c] += sp3_.b[size_t(c)];
      for (size_t c = 0; c < 128; ++c) r[c] = mish(r[c]);
    }
  }
  const auto tr1 = clk::now();

  // temporal: 转成 [C,T] 做 conv, 再转回 [T,C]
  std::vector<float> ct(128 * T), co(128 * T), tt(T * 128);
  for (size_t t = 0; t < T; ++t)
    for (size_t c = 0; c < 128; ++c) ct[c * T + t] = tmp[t * 128 + c];
  conv1dGluBatched(ct.data(), T, 0, co.data());
  conv1dGluBatched(co.data(), T, 1, ct.data());
  const auto tr2 = clk::now();
  for (size_t t = 0; t < T; ++t)
    for (size_t c = 0; c < 128; ++c) tt[t * 128 + c] = ct[c * T + t];

  // MHA (mask=None): q,k,v = Linear(x); 每 head 切片 64 维
  // T6: q/k/v 按批 sgemm(每层一次 M=T 矩阵积)
  std::vector<float> q(T * 128), k(T * 128), v(T * 128), attnO(T * 128);
  sgemm('N', 'T', int(T), 128, 128, 1.0f, tt.data(), 128, attWq_.W.data(),
        128, 0.0f, q.data(), 128);
  sgemm('N', 'T', int(T), 128, 128, 1.0f, tt.data(), 128, attWk_.W.data(),
        128, 0.0f, k.data(), 128);
  sgemm('N', 'T', int(T), 128, 128, 1.0f, tt.data(), 128, attWv_.W.data(),
        128, 0.0f, v.data(), 128);
  // bias 批量加(q/k/v 各 [T,128])
  for (std::vector<float>* qb : {&q, &k, &v}) {
    const Lin& L = (qb == &q) ? attWq_ : (qb == &k) ? attWk_ : attWv_;
    for (size_t t = 0; t < T; ++t) {
      float* r = qb->data() + t * 128;
      for (int c = 0; c < 128; ++c) r[c] += L.b[size_t(c)];
    }
  }
  const auto tr3 = clk::now();  // MHA 线性(q/k/v)结束
  const double temperature = std::sqrt(128.0);
  // T6: attention 核心批量化 — 每头一次 QKᵀ sgemm([T,64]·[T,64]ᵀ→[T,T]) +
  //     行内 softmax(max 减除后 exp, double 累加和与基线同式) + PV sgemm;
  //     替换基线 2×T×(T×64 + T×64) 标量三重循环(15.6ms@T369)
  {
    std::vector<float> sc(size_t(T) * T);
    for (int head = 0; head < 2; ++head) {
      const size_t off = size_t(head) * 64;
      sgemm('N', 'T', int(T), int(T), 64, 1.0f, q.data() + off, 128,
            k.data() + off, 128, 0.0f, sc.data(), int(T));
      // 温度缩放(逐元素, 与基线同为求和后除常量)
      for (size_t idx = 0; idx < sc.size(); ++idx)
        sc[idx] = float(double(sc[idx]) / temperature);
      for (size_t ti = 0; ti < T; ++ti) {
        float* row = sc.data() + ti * T;
        double mx = -1e30;
        for (size_t tj = 0; tj < T; ++tj)
          if (double(row[tj]) > mx) mx = double(row[tj]);
        double sum = 0.0;
        for (size_t tj = 0; tj < T; ++tj) {
          const double e = std::exp(double(row[tj]) - mx);
          row[tj] = float(e);
          sum += e;
        }
        const double inv = 1.0 / sum;
        for (size_t tj = 0; tj < T; ++tj)
          row[tj] = float(double(row[tj]) * inv);
      }
      // attnO 头块 = P·V_h
      sgemm('N', 'N', int(T), 64, int(T), 1.0f, sc.data(), int(T),
            v.data() + off, 128, 0.0f, attnO.data() + off, 128);
    }
  }
  const auto tr4 = clk::now();  // attention 核心结束
  // output = fc(attnO) + residual   (T6: 批量 sgemm)
  std::vector<float> fo(T * 128);
  {
    sgemm('N', 'T', int(T), 128, 128, 1.0f, attnO.data(), 128,
          attFc_.W.data(), 128, 0.0f, fo.data(), 128);
    for (size_t t = 0; t < T; ++t) {
      float* r = fo.data() + t * 128;
      for (int c = 0; c < 128; ++c)
        r[c] += attFc_.b[size_t(c)] + tt[t * 128 + size_t(c)];
    }
  }
  const auto tr5 = clk::now();  // attFc 结束
  // fc → [T,1024], 时间平均池化 (T6: 批量 sgemm + 双精度列累加与基线同口径)
  {
    std::vector<float> y(size_t(T) * 1024);
    sgemm('N', 'T', int(T), 1024, 128, 1.0f, fo.data(), 128, fc_.W.data(),
          128, 0.0f, y.data(), 1024);
    std::vector<double> pool(1024, 0.0);
    for (size_t t = 0; t < T; ++t) {
      float* r = y.data() + t * 1024;
      for (int c = 0; c < 1024; ++c) r[c] += fc_.b[size_t(c)];
      for (int c = 0; c < 1024; ++c) pool[size_t(c)] += double(r[c]);
    }
    pooled.assign(1024, 0.f);
    for (size_t c = 0; c < 1024; ++c) pooled[c] = float(pool[c] / double(T));
  }
  if (condTim) {
    const auto tr6 = clk::now();
    std::fprintf(stderr,
                 "[refenc-timing] spectral=%.1fms temporal_conv=%.1fms "
                 "qkv_lin=%.1fms attn_core=%.1fms attfc=%.1fms fc_pool=%.1fms "
                 "T=%zu\n",
                 cms(tr0, tr1), cms(tr1, tr2), cms(tr2, tr3), cms(tr3, tr4),
                 cms(tr4, tr5), cms(tr5, tr6), T);
  }
}
void ConditionBuilder::compute(const float* audio32k, size_t n,
                               const float* svEmb20480,
                               DecodeCondition* out) const {
  // E13 探针: GSV_COND_TIMING=1 时 stderr 输出 cond 链各段耗时(只加计时, 不改行为)
  using clk = std::chrono::steady_clock;
  const bool condTim = std::getenv("GSV_COND_TIMING") != nullptr;
  const auto tp_all = clk::now();
  auto cms = [](clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  std::vector<float> spec;
  size_t frames = 0;
  const auto tp0 = clk::now();
  spectrogram(audio32k, n, spec, frames);
  const auto tp1 = clk::now();
  // 条件输入 = spec[:704] 每帧列 → 转置为 [T][704]
  std::vector<float> condIn(frames * 704);
  for (size_t t = 0; t < frames; ++t)
    for (size_t b = 0; b < 704; ++b)
      condIn[t * 704 + b] = spec[b * frames + t];
  const auto tp2 = clk::now();

  out->ge.assign(1024, 0.f);
  std::vector<float> gePre(1024);
  refEnc(condIn.data(), frames, gePre);
  const auto tp3 = clk::now();
  // + sv_proj(sv_emb)
  std::vector<float> svProj(1024);
  svProj_.run(svEmb20480, svProj.data());
  for (size_t c = 0; c < 1024; ++c) {
    double g = double(gePre[c]) + double(svProj[c]);
    out->ge[c] = g < 0 ? float(g * preluSlope_[c]) : float(g);  // PReLU
  }
  const auto tp4 = clk::now();
  out->ge_text.resize(512);
  geTo512_.run(out->ge.data(), out->ge_text.data());  // ge^T→Linear→(ge_text)^T
  const auto tp5 = clk::now();
  if (condTim)
    std::fprintf(stderr,
                 "[cond-timing] spectrogram=%.1fms transpose=%.1fms ref_enc=%.1fms "
                 "svproj_prelu=%.1fms ge_to512=%.1fms TOTAL=%.1fms frames=%zu\n",
                 cms(tp0, tp1), cms(tp1, tp2), cms(tp2, tp3), cms(tp3, tp4),
                 cms(tp4, tp5), cms(tp_all, tp5), frames);
}

}  // namespace gsv::rt::pipeline
