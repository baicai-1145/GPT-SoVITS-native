// bert_model.hpp — B8: 通用 Transformer 编码栈 (chinese_bert.py / g2pw torch_api.py 共用)
// 结构: emb(tok+pos+type)→LN → N×{MHA→dense→+res→LN, GELU FFN→dense→+res→LN} (post-LN)
// 数值口径: fp32 全程; LN eps 参数化 (roberta 1e-12 / g2pw 1e-5);
//           extended mask = (1-mask)*neg (roberta: finfo.min, g2pw: -10000);
//           GELU 用 erf 精确式; softmax 行内减 max。
#pragma once
#include <arm_neon.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "bert/bert_io.hpp"
#include "kern/accel.hpp"
#include "kern/gemv_fmlal.hpp"  // E8: f32_to_f16_scalar (scalar fp32→fp16)
#if defined(GSV_AMX_GEMM)
#include "kern/gemm_f16_amx.hpp"
#endif

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

// ---- E8: AMX 后端开关 (--amx 决定) + 形状分流门槛 ----
// 进程级。决定 Linear::load 时是否预打包权重 panel; 运行期不与 AMX
// 池共存/竞争(只读标志位)。默认值由 CMake/CLI 在 Engine load 前设置。
inline bool& amx_bert_enabled() {
  static bool b = false;
  return b;
}

// AMX 分流门槛: T 太小或 K 太小打不抵打包开销 (与 amx_bench 实测对齐);
// 大形状走 AmxPanel 预打包 GEMM, 小/奇形回退 FMLAL。
// 任务卡口径"小 M(<64) 回退"指 tile (32×32) 半填充以下才明显亏损;
// 但生产文本常见 Ln=40~60 (LLM 转 fp16 后 pack 后仍 60%+ tile 利用),
// 纳入 AMX 收益区。kAmxMinK=256 避开深层 dispatcher 成本。
inline constexpr size_t kAmxMinRows = 48;
inline constexpr size_t kAmxMinK = 256;

// E8: AMX 分流模式 — GSV_AMX_BERT_MODE 环境变量控制:
//   all  = QKV+attn_out+FFN 全切 AMX (激进)
//   ffn  = 仅 W1/W2(inter/ffn_out) 切 AMX (保守回退; 任务卡指定的 fallback)
//   none = 全 FMLAL (= --amx 不开)
enum class AmxBertMode { kAll, kFfnOnly, kNone };
inline AmxBertMode& amx_bert_mode() {
  static AmxBertMode m = [] {
    const char* e = std::getenv("GSV_AMX_BERT_MODE");
    if (!e) return AmxBertMode::kAll;
    if (e[0] == 'f') return AmxBertMode::kFfnOnly;
    if (e[0] == 'n') return AmxBertMode::kNone;
    return AmxBertMode::kAll;
  }();
  return m;
}

// 调度限制: 同进程 SoVITS 已独占 amx_batch_run / amx_chain_run 调度路径;
// BERT 仍走 amx_batch_run(单节点) 与之同池, 互不混用。

// ---- E8: 融合 cast+pack —— fp32 [T,K] 行主 → AMX B-panel 直读布局 ----
// 同 im2col_to_panel_f16 但无 im2col(纯转置); 4×4 fp16 NEON 路径与
// pack_panel 完全一致(M=rows=激活行数, K=输入通道)。
//  源:  x_fp32 [T, K] 行主
//  输出: out_buf [ntile * K * 64] 字节布局, ntile = (T+31)/32,
//        tail tile 行补零。调用方负责 out_buf 容量 ≥ ntile*K*64+64。
// 注: 不带 ones 列(Linear 无 bias 折叠; bias 在 GEMM 之后逐元素加)。
inline void cast_pack_B_f32_to_panel(const float* x_fp32, size_t T, size_t K,
                                     std::vector<uint8_t>& out_buf) {
  const size_t nt = (T + 31) / 32;
  // E8: 热路径 — 预留容量免反复 realloc; 仅需写入区+尾部 tile 补零区,
  //     用 resize 保容 (不重填)，未覆盖字节由后续逐 k 写入或补零循环处理。
  const size_t need = nt * K * 64 + 64;
  if (out_buf.capacity() < need) out_buf.reserve(need);
  out_buf.resize(need);  // 容量足够时仅调整 size, 不再 memset; 尾部补零见下
  const uintptr_t p = (uintptr_t)out_buf.data();
  const size_t base = (64 - (p & 63)) & 63;
  uint8_t* dst = out_buf.data() + base;
  // E8: 尾部 tile 补零区预填 (tr..31 行 × 全 K); 其余区域随后全部覆写。
  if (const size_t tr_last = T - (nt - 1) * 32; tr_last < 32) {
    uint8_t* d_tail = dst + (nt - 1) * K * 64;
    const size_t zoff = tr_last * 2;           // 每 k 块内从该字节起补零
    for (size_t k = 0; k < K; ++k)
      std::memset(d_tail + k * 64 + zoff, 0, 64 - zoff);
  }
  // NEON 4×4 fp16 转置 (与 pack_panel 同构)
  auto trans4x4 = [](uint16x4_t x0, uint16x4_t x1, uint16x4_t x2,
                      uint16x4_t x3, uint16x4_t& o0, uint16x4_t& o1,
                      uint16x4_t& o2, uint16x4_t& o3) {
    const float16x4x2_t a01 = vtrn_f16(vreinterpret_f16_u16(x0),
                                       vreinterpret_f16_u16(x1));
    const float16x4x2_t a23 = vtrn_f16(vreinterpret_f16_u16(x2),
                                       vreinterpret_f16_u16(x3));
    const float32x2x2_t b0 = vtrn_f32(
        vreinterpret_f32_f16(a01.val[0]), vreinterpret_f32_f16(a23.val[0]));
    const float32x2x2_t b1 = vtrn_f32(
        vreinterpret_f32_f16(a01.val[1]), vreinterpret_f32_f16(a23.val[1]));
    o0 = vreinterpret_u16_f16(b0.val[0]);
    o1 = vreinterpret_u16_f16(b1.val[0]);
    o2 = vreinterpret_u16_f16(b0.val[1]);
    o3 = vreinterpret_u16_f16(b1.val[1]);
  };
  for (size_t t = 0; t < nt; ++t) {
    const size_t r0 = t * 32;
    const size_t tr = std::min<size_t>(32, T - r0);
    uint8_t* d = dst + t * K * 64;
    if (tr == 32) {  // 满 tile
      for (size_t r = 0; r < 32; r += 4) {
        const float* s0 = x_fp32 + (r0 + r) * K;
        const float* s1 = s0 + K, *s2 = s0 + 2 * K, *s3 = s0 + 3 * K;
        size_t k = 0;
        for (; k + 4 <= K; k += 4) {
          const float16x4_t v0 = vcvt_f16_f32(vld1q_f32(s0 + k));
          const float16x4_t v1 = vcvt_f16_f32(vld1q_f32(s1 + k));
          const float16x4_t v2 = vcvt_f16_f32(vld1q_f32(s2 + k));
          const float16x4_t v3 = vcvt_f16_f32(vld1q_f32(s3 + k));
          uint16x4_t o0, o1, o2, o3;
          trans4x4(v0, v1, v2, v3, o0, o1, o2, o3);
          uint8_t* c = d + k * 64 + r * 2;
          vst1_u16((uint16_t*)(c), o0);
          vst1_u16((uint16_t*)(c + 64), o1);
          vst1_u16((uint16_t*)(c + 128), o2);
          vst1_u16((uint16_t*)(c + 192), o3);
        }
        for (; k < K; ++k) {  // K 尾标量
          uint8_t* c = d + k * 64 + r * 2;
          *(uint16_t*)(c)     = kern::f32_to_f16_scalar(s0[k]);
          *(uint16_t*)(c + 2) = kern::f32_to_f16_scalar(s1[k]);
          *(uint16_t*)(c + 4) = kern::f32_to_f16_scalar(s2[k]);
          *(uint16_t*)(c + 6) = kern::f32_to_f16_scalar(s3[k]);
        }
      }
    } else {  // 尾 tile: 标量
      for (size_t r = 0; r < 32; r += 4) {
        const size_t nr = r < tr ? std::min<size_t>(4, tr - r) : 0;
        const float* s0 = nr > 0 ? x_fp32 + (r0 + r) * K : nullptr;
        const float* s1 = nr > 1 ? s0 + K : nullptr;
        const float* s2 = nr > 2 ? s0 + 2 * K : nullptr;
        const float* s3 = nr > 3 ? s0 + 3 * K : nullptr;
        for (size_t k = 0; k < K; ++k) {
          uint8_t* col = d + k * 64 + r * 2;
          if (s0) *(uint16_t*)col     = kern::f32_to_f16_scalar(s0[k]); else *(uint16_t*)col = 0;
          if (s1) *(uint16_t*)(col + 2) = kern::f32_to_f16_scalar(s1[k]); else *(uint16_t*)(col + 2) = 0;
          if (s2) *(uint16_t*)(col + 4) = kern::f32_to_f16_scalar(s2[k]); else *(uint16_t*)(col + 4) = 0;
          if (s3) *(uint16_t*)(col + 6) = kern::f32_to_f16_scalar(s3[k]); else *(uint16_t*)(col + 6) = 0;
        }
      }
    }
  }
}

struct Linear {  // y[T,out] = x[T,in]·Wᵀ + b; fp16 直读权重 + FMLAL 前向
  kern::accel::DenseF16 w;      // view(f16 指针)或 fp32 常驻(无 f16 段时)
  std::vector<float> b;
  size_t in = 0, out = 0;
#if defined(GSV_AMX_GEMM)
  // 预打包权重 panel (W side: rows=out, K=in); 由 load() 装载期一次
  // amx_pack_into() 填充; 生命周期 = BertModel 栈(GsvFile 持有者同域)。
  kern::AmxPanel w_panel;
  bool w_panel_ready = false;    // amx 使能 && f16 源 && K>=kAmxMinK && 角色允许
#endif
  void load(const rt::GsvFile& f, const std::string& pfx, size_t o, size_t i,
            bool has_bias = true, bool ffn_role = false) {
    out = o;
    in = i;
    const auto* t = f.tensor(pfx + ".weight");
    if (!t) {
      std::fprintf(stderr, "bert: missing tensor %s.weight\n", pfx.c_str());
      std::abort();
    }
    const uint16_t* f16_raw = nullptr;
    std::vector<float> wf_lifted;  // 仅当需升位时占用
    if (t->has_f16()) {
      w = kern::accel::DenseF16::view_f16(t->data_f16_raw(), o, i);  // 零拷贝 FMLAL
      f16_raw = t->data_f16_raw();
    } else {
      load_tensor_f32(f, pfx + ".weight", wf_lifted, {o, i});
      w = kern::accel::DenseF16(wf_lifted.data(), o, i);
    }
    if (has_bias) load_tensor_f32(f, pfx + ".bias", b, {o});

#if defined(GSV_AMX_GEMM)
    // E8: 装载期预打包权重 panel; K 太小 (例如 K<256) 不打包, 运行期全 FMLAL。
    // 分流模式: all = 全部 dense; ffn = 仅 W1/W2(inter/ffn_out)。
    bool mode_allows = false;
    switch (amx_bert_mode()) {
      case AmxBertMode::kAll:     mode_allows = true; break;
      case AmxBertMode::kFfnOnly: mode_allows = ffn_role; break;
      case AmxBertMode::kNone:    mode_allows = false; break;
    }
    if (f16_raw && amx_bert_enabled() && mode_allows &&
        kern::amx_gemm_available() &&
        in >= kAmxMinK && out >= kAmxMinK) {
      w_panel.rows = out;
      w_panel.K = in;
      kern::amx_pack_into(f16_raw, out, in, w_panel.buf);
      w_panel_ready = true;
    } else {
      w_panel_ready = false;
    }
#endif
    (void)f16_raw;   // 抑制 GSV_AMX_GEMM 关闭时的未用警告
    (void)ffn_role;  // 同上 (仅 AMX 模式分流用)
  }
  // xh: 复用 fp16 激活暂存(引擎持有, 跨层复用)
  void forward(const Matrix& x, Matrix& y, std::vector<uint16_t>& xh) const {
    const size_t T = x.rows;
    y.reset(T, out);
#if defined(GSV_AMX_GEMM)
    if (w_panel_ready && T >= kAmxMinRows) {
      // 融合 cast+pack 激活 panel + 单节点 amx_batch_run
      // 契约: C[M,N] = pa·pbᵀ, y[T,out] = x_act[T,in]·W[out,in]ᵀ
      //   ⇒ M=T(pa=激活), N=out(pb=权重panel 预打包)
      thread_local std::vector<uint8_t> act_buf;
      cast_pack_B_f32_to_panel(x.d.data(), T, in, act_buf);
      kern::AmxPanel pb;
      pb.rows = T;
      pb.K = in;
      pb.buf = std::move(act_buf);
      kern::AmxBatchNode nd;
      nd.phase = 0;
      nd.pa = &pb;          // 激活侧 [T, K]
      nd.pb = &w_panel;     // 权重侧 [out, K]
      nd.c = y.d.data();
      nd.M = T;
      nd.N = out;
      kern::amx_batch_run(&nd, 1);
      // bias 逐元素加(与 FMLAL 路径同口径)
      if (!b.empty())
        for (size_t t = 0; t < T; ++t)
          for (size_t o = 0; o < out; ++o) y.d[t * out + o] += b[o];
      return;
    }
#endif
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

#if defined(GSV_AMX_GEMM)
  // E8: Q/K/V 共享同一输入 panel — 三节点同批派发 (省 2 次池往返).
  // 仅当三者均已就绪时启用; 否则各自回退 forward 单发。
  bool qkv_batch_ready() const {
    return q.w_panel_ready && k.w_panel_ready && v.w_panel_ready;
  }
  void qkv_forward_batch(const Matrix& x, Matrix& qm, Matrix& km, Matrix& vm) const {
    const size_t T = x.rows;
    qm.reset(T, q.out);
    km.reset(T, k.out);
    vm.reset(T, v.out);
    thread_local std::vector<uint8_t> act_buf;
    cast_pack_B_f32_to_panel(x.d.data(), T, in_of_q(), act_buf);
    kern::AmxPanel pb;
    pb.rows = T;
    pb.K = in_of_q();
    pb.buf = std::move(act_buf);
    kern::AmxBatchNode nd[3];
    nd[0].phase = 0; nd[0].pa = &pb; nd[0].pb = &q.w_panel; nd[0].c = qm.d.data(); nd[0].M = T; nd[0].N = q.out;
    nd[1].phase = 0; nd[1].pa = &pb; nd[1].pb = &k.w_panel; nd[1].c = km.d.data(); nd[1].M = T; nd[1].N = k.out;
    nd[2].phase = 0; nd[2].pa = &pb; nd[2].pb = &v.w_panel; nd[2].c = vm.d.data(); nd[2].M = T; nd[2].N = v.out;
    kern::amx_batch_run(nd, 3);
    add_bias(qm, q);
    add_bias(km, k);
    add_bias(vm, v);
  }
  static void add_bias(Matrix& y, const Linear& lin) {
    if (lin.b.empty()) return;
    for (size_t t = 0; t < y.rows; ++t)
      for (size_t o = 0; o < lin.out; ++o) y.d[t * lin.out + o] += lin.b[o];
  }
  size_t in_of_q() const { return q.in; }
#endif

  void load(const rt::GsvFile& f, const std::string& pfx, size_t hidden,
            size_t n_heads, size_t inter_dim, float eps) {
    heads = n_heads;
    // E8: FFN 角色标记 (inter/ffn_out) — ffn-only 回退模式仅这两层切 AMX
    q.load(f, pfx + ".attention.self.query", hidden, hidden);
    k.load(f, pfx + ".attention.self.key", hidden, hidden);
    v.load(f, pfx + ".attention.self.value", hidden, hidden);
    attn_out.load(f, pfx + ".attention.output.dense", hidden, hidden);
    attn_ln.load(f, pfx + ".attention.output.LayerNorm", hidden, eps);
    inter.load(f, pfx + ".intermediate.dense", inter_dim, hidden,
               /*has_bias=*/true, /*ffn_role=*/true);
    ffn_out.load(f, pfx + ".output.dense", hidden, inter_dim,
                 /*has_bias=*/true, /*ffn_role=*/true);
    out_ln.load(f, pfx + ".output.LayerNorm", hidden, eps);
  }

  void forward(const Matrix& x, const std::vector<float>& ext_mask,
               Matrix& y, Matrix& scr, Matrix& ctxh,
               std::vector<uint16_t>& xh) const {
    (void)xh;  // E8: xh 仅 FMLAL 路径用, AMX 同批分支不引用 (抑制警告)
    const size_t T = x.rows, C = x.cols, H = heads, dh = C / H;
    // QKV 投影(FMLAL fp16×fp16→fp32 或 E8 AMX; AMX 同批三联派发)
    Matrix qm, km, vm;
#if defined(GSV_AMX_GEMM)
    if (qkv_batch_ready() && T >= kAmxMinRows) {
      qkv_forward_batch(x, qm, km, vm);
    } else {
      q.forward(x, qm, xh);
      k.forward(x, km, xh);
      v.forward(x, vm, xh);
    }
#else
    (void)T;
    q.forward(x, qm, xh);
    k.forward(x, km, xh);
    v.forward(x, vm, xh);
#endif
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
    attn_out.forward(ctxh, y, xh);   // dense(FMLAL/AMX)
    for (size_t i = 0; i < y.d.size(); ++i) y.d[i] += x.d[i];  // res(fp32)
    attn_ln.forward(y);   // LayerNorm fp32
    // FFN(FMLAL/AMX)
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
