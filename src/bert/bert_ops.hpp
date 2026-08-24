// bert_model.hpp — B8: 通用 Transformer 编码栈 (chinese_bert.py / g2pw torch_api.py 共用)
// 结构: emb(tok+pos+type)→LN → N×{MHA→dense→+res→LN, GELU FFN→dense→+res→LN} (post-LN)
// 数值口径: fp32 全程; LN eps 参数化 (roberta 1e-12 / g2pw 1e-5);
//           extended mask = (1-mask)*neg (roberta: finfo.min, g2pw: -10000);
//           GELU 用 erf 精确式; softmax 行内减 max。
#pragma once
#include <cmath>
#include <string>
#include <vector>

#include "bert/bert_io.hpp"
#include "kern/accel.hpp"

namespace gsv::bert {

constexpr float kFinfoMin = -3.40282347e+38f;

inline float gelu_erf(float x) {
  return 0.5f * x * (1.f + erff(x * 0.70710678118654752440f));
}

// 行内数值稳定 softmax
inline void softmax_rows(float* s, size_t rows, size_t cols) {
  for (size_t i = 0; i < rows; ++i) {
    float* r = s + i * cols;
    float mx = r[0];
    for (size_t j = 1; j < cols; ++j) mx = mx > r[j] ? mx : r[j];
    float sum = 0.f;
    for (size_t j = 0; j < cols; ++j) {
      r[j] = std::exp(r[j] - mx);
      sum += r[j];
    }
    const float inv = 1.f / sum;
    for (size_t j = 0; j < cols; ++j) r[j] *= inv;
  }
}

struct Linear {  // y[T,out] = x[T,in]·Wᵀ + b; fp16 直读权重 + FMLAL 前向
  kern::accel::DenseF16 w;      // view(f16 指针)或 fp32 常驻(无 f16 段时)
  std::vector<float> b;
  size_t in = 0, out = 0;
  void load(const rt::GsvFile& f, const std::string& pfx, size_t o, size_t i,
            bool has_bias = true) {
    out = o;
    in = i;
    const auto* t = f.tensor(pfx + ".weight");
    if (!t) {
      std::fprintf(stderr, "bert: missing tensor %s.weight\n", pfx.c_str());
      std::abort();
    }
    if (t->has_f16()) {
      w = kern::accel::DenseF16::view_f16(t->data_f16_raw(), o, i);  // 零拷贝 FMLAL
    } else {
      std::vector<float> wf;
      load_tensor_f32(f, pfx + ".weight", wf, {o, i});
      w = kern::accel::DenseF16(wf.data(), o, i);
    }
    if (has_bias) load_tensor_f32(f, pfx + ".bias", b, {o});
  }
  // xh: 复用 fp16 激活暂存(引擎持有, 跨层复用)
  void forward(const Matrix& x, Matrix& y, std::vector<uint16_t>& xh) const {
    y.reset(x.rows, out);
    w.forward(x.d.data(), x.rows, y.d.data(), xh);
    if (!b.empty())
      for (size_t t = 0; t < x.rows; ++t)
        for (size_t o = 0; o < out; ++o) y.d[t * out + o] += b[o];
  }
};

struct LayerNorm1D {
  std::vector<float> g, bta;
  float eps = 1e-12f;
  void load(const rt::GsvFile& f, const std::string& pfx, size_t c, float e) {
    eps = e;
    load_tensor_f32(f, pfx + ".weight", g, {c});
    load_tensor_f32(f, pfx + ".bias", bta, {c});
  }
  void forward(Matrix& x) const {  // 就地
    const size_t c = x.cols;
    for (size_t t = 0; t < x.rows; ++t) {
      float* r = x.d.data() + t * c;
      float mean = 0.f;
      for (size_t i = 0; i < c; ++i) mean += r[i];
      mean /= float(c);
      float var = 0.f;
      for (size_t i = 0; i < c; ++i) {
        const float dv = r[i] - mean;
        var += dv * dv;
      }
      var /= float(c);
      const float inv = 1.f / std::sqrt(var + eps);
      for (size_t i = 0; i < c; ++i) r[i] = (r[i] - mean) * inv * g[i] + bta[i];
    }
  }
};

struct BertLayer {
  Linear q, k, v;          // attention.self
  Linear attn_out;         // attention.output.dense
  LayerNorm1D attn_ln;     // attention.output.LayerNorm
  Linear inter;            // intermediate.dense
  LayerNorm1D out_ln;      // output.LayerNorm
  Linear ffn_out;          // output.dense
  size_t heads = 16;

  void load(const rt::GsvFile& f, const std::string& pfx, size_t hidden,
            size_t n_heads, size_t inter_dim, float eps) {
    heads = n_heads;
    q.load(f, pfx + ".attention.self.query", hidden, hidden);
    k.load(f, pfx + ".attention.self.key", hidden, hidden);
    v.load(f, pfx + ".attention.self.value", hidden, hidden);
    attn_out.load(f, pfx + ".attention.output.dense", hidden, hidden);
    attn_ln.load(f, pfx + ".attention.output.LayerNorm", hidden, eps);
    inter.load(f, pfx + ".intermediate.dense", inter_dim, hidden);
    ffn_out.load(f, pfx + ".output.dense", hidden, inter_dim);
    out_ln.load(f, pfx + ".output.LayerNorm", hidden, eps);
  }

  void forward(const Matrix& x, const std::vector<float>& ext_mask,
               Matrix& y, Matrix& scr, Matrix& ctxh,
               std::vector<uint16_t>& xh) const {
    const size_t T = x.rows, C = x.cols, H = heads, dh = C / H;
    // QKV 投影(FMLAL fp16×fp16→fp32)
    Matrix qm, km, vm;
    q.forward(x, qm, xh);
    k.forward(x, km, xh);
    v.forward(x, vm, xh);
    // 逐头: S = Q_h·K_hᵀ/√dh (+mask 列) → softmax → ctx[:,head] = S·V_h
    scr.reset(T, T);
    ctxh.reset(T, C);
    for (size_t h = 0; h < H; ++h) {
      const size_t off = h * dh;
      float* S = scr.d.data();
      // A=Q_h [T,dh] (ptr qm+off, ld=C), B=K_hᵀ [dh,T] (ptr km+off, ld=C)
      kern::accel::sgemm('N', 'T', int(T), int(T), int(dh),
                         1.f / std::sqrt(float(dh)), qm.d.data() + off, int(C),
                         km.d.data() + off, int(C), 0.f, S, int(T));
      if (!ext_mask.empty())
        for (size_t i = 0; i < T; ++i)
          for (size_t j = 0; j < T; ++j) S[i * T + j] += ext_mask[j];
      softmax_rows(S, T, T);
      // ctx 子矩阵 [T,dh]: ldc=C 直接写入 ctxh+off (不连续子矩阵)
      kern::accel::sgemm('N', 'N', int(T), int(dh), int(T), 1.f, S, int(T),
                         vm.d.data() + off, int(C), 0.f,
                         ctxh.d.data() + off, int(C));
    }
    y.reset(T, C);
    attn_out.forward(ctxh, y, xh);   // dense(FMLAL)
    for (size_t i = 0; i < y.d.size(); ++i) y.d[i] += x.d[i];  // res(fp32)
    attn_ln.forward(y);   // LayerNorm fp32
    // FFN(FMLAL)
    Matrix mid;
    inter.forward(y, mid, xh);
    for (float& z : mid.d) z = gelu_erf(z);  // GELU fp32
    Matrix f2;
    ffn_out.forward(mid, f2, xh);
    for (size_t i = 0; i < f2.d.size(); ++i) f2.d[i] += y.d[i];  // res(fp32)
    out_ln.forward(f2);
    y.d.swap(f2.d);
  }

  void forward(const Matrix& x, const std::vector<float>& ext_mask,
               Matrix& y, Matrix& scr, Matrix& ctxh) const {
    std::vector<uint16_t> xh;
    forward(x, ext_mask, y, scr, ctxh, xh);
  }
};

}  // namespace gsv::bert
