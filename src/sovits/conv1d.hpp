// conv1d.hpp — SoVITS 卷积原语 (E5-P2: w_f16 单副本 + AMX/sgemm 计算分流)
//   Conv1d:  same-padding (含 dilation), im2col → AMX pp 或 sgemm 升位
//   ConvT1d: ConvTranspose1d(k, stride=u, pad=(k-u)/2), 相位分解 GEMM + scatter
//
// 权重纪律: 仅存 w_f16 位型单副本 (E2-SOV 采纳成果); 无 fp32 常驻副本。
//
// 计算分流 (运行时 --amx, 编译期 -DGSV_AMX_GEMM):
//   大块 (out_c ≥ kAmxMinOutC=64) 且 AMX 可用:
//     权重 AmxPanel 装载时预打包一次; 激活每调用打包 (col_T f16 已就绪);
//     kern::gemm_f16_amx_pp — 内部落 AMX 专用线程池, 与 cblas 线程互斥隔离。
//   其余 (GEMV 形 out_c<64 / --amx 关 / 无 AMX 硬件):
//     sgemm 升位路径 — w_f16 即时升 f32 (thread_local), im2col fp32,
//     数值与现网 fp32 链路同构 (差异仅权重一次 fp16 舍入, golden 门禁口径覆盖)。
//
// 重要: 同一进程内 amx_batch_run 与 gemm_f16_amx_pp 不可混用 (专用池状态会
// 被两种派发路径搅乱, 实测表现成错乱/崩溃)。本模块统一只经 amx_batch_run
// 单节点封装 (amx_gemm_1) 派发, 杜绝混用。
#pragma once

#include "kern/accel.hpp"
#include "sovits/sovits_types.hpp"
#if defined(GSV_AMX_GEMM)
#include "kern/gemm_f16_amx.hpp"

// 统一经 amx_batch_run 单节点派发, 避免与 gemm_f16_amx_pp 混用 (见文件头注释)。
// 不可用时 amx_batch_run 内部自动回退 gemm_f16_amx_pp (同样不混用)。
inline void amx_gemm_1(const gsv::kern::AmxPanel& pa, const gsv::kern::AmxPanel& pb,
                      float* c, size_t M, size_t N) {
  gsv::kern::AmxBatchNode nd;
  nd.phase = 0;
  nd.pa = &pa;
  nd.pb = &pb;
  nd.c = c;
  nd.M = M;
  nd.N = N;
  gsv::kern::amx_batch_run(&nd, 1);
}
#endif

#include <cmath>
#include <vector>

namespace gsv::sovits {

// AMX 分流的 M 下界: conv_post (Co=1 GEMV 形) 打包开销不抵收益 → sgemm;
// 实测 M≥24 全形状反超 (tile 32 行, M=24 填充率已 75%)。
constexpr size_t kAmxMinOutC = 24;

// 进程级 AMX 使能 (--amx 开关; 默认关)。须在 engine load 前设置:
// 决定装载时是否预打包权重 panel。
inline bool& amx_enabled() {
  static bool b = false;
  return b;
}

class Conv1d {
 public:
  size_t out_c = 0, in_c = 0, k = 0, dilation = 1;
  std::vector<uint16_t> w_f16;  // [out,in,k] 位型存储 (唯一权重副本)
  std::vector<float> b;         // [out]

#if defined(GSV_AMX_GEMM)
  kern::AmxPanel panel_;  // [out, K=in*k] 预打包 (仅 amx_enabled 且大块时)
#endif
  bool has_panel_ = false;
#if defined(GSV_AMX_GEMM)
  // E10: resblock 批量图执行需要直访 panel 与就绪态 (仅读)
  bool amx_ready() const { return has_panel_; }
  const kern::AmxPanel& amx_panel() const { return panel_; }
#else
  bool amx_ready() const { return false; }
#endif

  void load(const rt::GsvFile& f, std::string_view prefix, size_t o, size_t i,
            size_t kk, size_t dil = 1) {
    out_c = o;
    in_c = i;
    k = kk;
    dilation = dil;
    std::string p(prefix);
    if (!w_f16.empty()) return;  // 已加载
    load_tensor_f16(f, p + ".weight", w_f16, {o, i, kk});
    const auto* tb = f.tensor(p + ".bias");
    if (tb) {
      load_tensor_f32(f, p + ".bias", b);
    } else {
      b.assign(o, 0.f);  // conv_post 无 bias (is_bias=False)
    }
#if defined(GSV_AMX_GEMM)
    // 预打包: k>1 大块 + k==1 点积形 (E6: 直写 panel 后 k==1 也反超 sgemm)
    if (amx_enabled() && out_c >= kAmxMinOutC) {
      // bias 折叠: 权重增广末列 = b[o] (f16), 配合激活 ones 列
      const size_t Kw2 = in_c * k;
      std::vector<uint16_t> wa(out_c * (Kw2 + 1));
      for (size_t oo = 0; oo < out_c; ++oo) {
        for (size_t pp = 0; pp < Kw2; ++pp)
          wa[oo * (Kw2 + 1) + pp] = w_f16[oo * Kw2 + pp];
        wa[oo * (Kw2 + 1) + Kw2] =
            f16_round(oo < b.size() ? b[oo] : 0.f);
      }
      panel_ = kern::amx_pack(wa.data(), out_c, Kw2 + 1);
      has_panel_ = true;
    }
#endif
  }

  // y[C_out, T] = conv(x[C_in, T]) ; same padding, 输出长度 == T
  // 允许 &x == &y (内部临时缓冲保护)
  void forward(const Tensor2D& x_in, Tensor2D& y) const {
    if (&x_in == &y) {
      Tensor2D tmp;
      forward(x_in, tmp);
      y.d.swap(tmp.d);
      y.C = tmp.C;
      y.T = tmp.T;
      return;
    }
    const Tensor2D& x = x_in;
    const size_t T = x.T;
    const size_t K = in_c * k;
#if defined(GSV_AMX_GEMM)
    // ---- AMX 路径: k>1 大块与 k==1 点积形; im2col 直接写 panel 布局 ----
    if (has_panel_ && k > 1 && dilation == 1 &&
        T >= 64 && kern::amx_gemm_available()) {
      // GEMM 全量覆写输出, 免 memset
      y.C = out_c;
      y.T = T;
      y.d.resize(out_c * T);
      thread_local std::vector<uint8_t> act_panel;
      const bool prof = sov_timing_enabled();
      const double tp0 = prof ? now_ms_e6() : 0.0;
      im2col_to_panel_f16(x.d.data(), in_c, T, k, dilation, act_panel,
                          /*append_ones_col=*/true);
      if (prof) g_conv_split().prep += now_ms_e6() - tp0;
      kern::AmxPanel pb;
      pb.rows = T;
      pb.K = K + 1;
      pb.buf = std::move(act_panel);
      const double tg0 = prof ? now_ms_e6() : 0.0;
      amx_gemm_1(panel_, pb, y.d.data(), out_c, T);
      if (prof) {
        g_conv_split().gemm += now_ms_e6() - tg0;
        ++g_conv_split().n;
      }
      return;  // bias 已折叠进 GEMM
    }
    // k==1 点积形: 同一直写布局 (k=1 退化为 cast_transpose→panel)
    if (has_panel_ && k == 1 && dilation == 1 &&
        T >= 64 && kern::amx_gemm_available()) {
      const bool prof = sov_timing_enabled();
      y.C = out_c;
      y.T = T;
      y.d.resize(out_c * T);
      thread_local std::vector<uint8_t> act_panel;
      const double tp1 = prof ? now_ms_e6() : 0.0;
      im2col_to_panel_f16(x.d.data(), in_c, T, 1, 1, act_panel,
                          /*append_ones_col=*/true);
      if (prof) g_conv_split().prep += now_ms_e6() - tp1;
      kern::AmxPanel pb;
      pb.rows = T;
      pb.K = K + 1;
      pb.buf = std::move(act_panel);
      const double tg1 = prof ? now_ms_e6() : 0.0;
      amx_gemm_1(panel_, pb, y.d.data(), out_c, T);
      if (prof) {
        g_conv_split().gemm += now_ms_e6() - tg1;
        ++g_conv_split().n;
      }
      return;  // bias 已折叠进 GEMM
    }
#endif
    // ---- sgemm 升位路径 (回退安全: 无 AMX/--amx 关/GEMV 形) ----
    // w_f16 即时升位 (thread_local 复用容量; 权重仍单副本)
    thread_local std::vector<float> wf;
    wf.resize(w_f16.size());
    kern::accel::f16_to_f32(w_f16.data(), wf.data(), w_f16.size());
    y.C = out_c;  // GEMM beta=0 全量覆写, 免 memset
    y.T = T;
    y.d.resize(out_c * T);
    if (k == 1 && dilation == 1) {
      // 逐点: y = W[out, in] · x[in, T]
      kern::accel::sgemm('N', 'N', static_cast<int>(out_c),
                         static_cast<int>(T), static_cast<int>(in_c), 1.f,
                         wf.data(), static_cast<int>(in_c), x.d.data(),
                         static_cast<int>(T), 0.f, y.d.data(),
                         static_cast<int>(T));
    } else {
      // im2col: col[in*k, T] (fp32, 与现网 main 同构)
      thread_local std::vector<float> col;
      col.resize(K * T);
      const int pad_l = static_cast<int>((k - 1) * dilation) / 2;
      for (size_t i = 0; i < in_c; ++i) {
        const float* xr = x.row(i);
        for (size_t t = 0; t < T; ++t) {
          for (size_t kk = 0; kk < k; ++kk) {
            long long src = static_cast<long long>(t) +
                            static_cast<long long>(kk * dilation) - pad_l;
            col[(i * k + kk) * T + t] =
                (src >= 0 && src < static_cast<long long>(T)) ? xr[src] : 0.f;
          }
        }
      }
      // y[out,T] = W[out, in*k] · col[in*k, T]
      kern::accel::sgemm('N', 'N', static_cast<int>(out_c), static_cast<int>(T),
                         static_cast<int>(K), 1.f, wf.data(),
                         static_cast<int>(K), col.data(), static_cast<int>(T),
                         0.f, y.d.data(), static_cast<int>(T));
    }
    add_bias(y);
  }

  void add_bias(Tensor2D& y) const {
    for (size_t c = 0; c < out_c; ++c) {
      float* yr = y.row(c);
      const float bc = b[c];
      for (size_t t = 0; t < y.T; ++t) yr[t] += bc;
    }
  }
};

// ConvTranspose1d: out_len = T*u ; weight [in, out, k] (torch 布局!), bias[out]
// 相位分解: y[:, j*u+d-pad] += Σ_i W[i,o,d]·x[:,j], d∈[0,k)
// 每相位一次 GEMM(tmp[o,T] = Wd[o,in]·x[in,T]) 后按相位 scatter。
class ConvT1d {
 public:
  size_t out_c = 0, in_c = 0, k = 0, stride_u = 1;
  std::vector<uint16_t> w_f16;  // [k][out,in] 相位切片位型 (装载时预抽取)
  std::vector<float> b;         // [out]

#if defined(GSV_AMX_GEMM)
  std::vector<kern::AmxPanel> panels_;  // 每 d 一个 [out,in] panel
#endif
  bool has_panels_ = false;

  void load(const rt::GsvFile& f, std::string_view prefix, size_t i, size_t o,
            size_t kk, size_t u) {
    out_c = o;
    in_c = i;
    k = kk;
    stride_u = u;
    std::string p(prefix);
    std::vector<uint16_t> raw;  // torch [in,out,k]
    load_tensor_f16(f, p + ".weight", raw, {i, o, kk});
    w_f16.resize(k * o * i);
    for (size_t d = 0; d < k; ++d)
      for (size_t oo = 0; oo < o; ++oo)
        for (size_t ii = 0; ii < i; ++ii)
          w_f16[(d * o + oo) * i + ii] = raw[(ii * o + oo) * k + d];
    const auto* tb = f.tensor(p + ".bias");
    if (tb)
      load_tensor_f32(f, p + ".bias", b);
    else
      b.assign(o, 0.f);
#if defined(GSV_AMX_GEMM)
    if (amx_enabled() && out_c >= kAmxMinOutC) {
      panels_.reserve(k);
      for (size_t d = 0; d < k; ++d)
        panels_.push_back(
            kern::amx_pack(w_f16.data() + d * o * i, o, i));
      has_panels_ = true;
    }
#endif
  }

  // 允许 &x == &y ; scratch 由调用方复用 (Generator 提供)
  void forward(const Tensor2D& x_in, Tensor2D& y, Tensor2D& scratch) const {
    if (&x_in == &y) {
      Tensor2D tmp;
      forward(x_in, tmp, scratch);
      y.d.swap(tmp.d);
      y.C = tmp.C;
      y.T = tmp.T;
      return;
    }
    const Tensor2D& x = x_in;
    const size_t T = x.T;
    const long long pad = static_cast<long long>((k - stride_u) / 2);
    const size_t Tout = T * stride_u;
    y.reset(out_c, Tout);  // scatter 用 +=, 需要零起点
#if defined(GSV_AMX_GEMM)
    if (has_panels_ && kern::amx_gemm_available()) {
      // 激活 panel 每调用打包一次, 所有相位复用 (buffer 容量复用免反复分配);
      // k=1 时 im2col_to_panel 退化为 cast_transpose 直写 panel 布局
      thread_local std::vector<uint8_t> pb_buf;
      im2col_to_panel_f16(x.d.data(), in_c, T, 1, 1, pb_buf);
      thread_local kern::AmxPanel pb;
      pb.rows = T;
      pb.K = in_c;
      pb.buf = std::move(pb_buf);
      // E9 批量派发: 相位间无依赖, 分块一次提交 (节点输出不得重叠)。
      // 小形状 (M<64 或 MNK<2^18) 不进 batch, 逐发 (同样走 amx_batch_run 封装,
      // 全程不混用 gemm_f16_amx_pp, 见文件头注释)。
      const bool small_shape =
          out_c < 64 || out_c * in_c * T < (size_t{1} << 18);
      if (!small_shape && k > 1) {
        // 分块约束: 单块 scratch ≤ 64MB (节点输出不重叠, 每块一次派发)
        const size_t ph = std::max<size_t>(
            1, std::min<size_t>(k, (64u << 20) / (out_c * T * 4 + 1)));
        static thread_local Tensor2D chunk;
        static thread_local std::vector<kern::AmxBatchNode> nodes;
        for (size_t d0 = 0; d0 < k; d0 += ph) {
          const size_t nd = std::min(ph, k - d0);
          chunk.C = nd * out_c;
          chunk.T = T;
          chunk.d.resize(nd * out_c * T);  // 全覆写免 memset
          nodes.clear();
          for (size_t i = 0; i < nd; ++i) {
            kern::AmxBatchNode nd_node;
            nd_node.phase = 0;  // 相位间无依赖, 同批并行
            nd_node.pa = &panels_[d0 + i];
            nd_node.pb = &pb;
            nd_node.c = chunk.d.data() + i * out_c * T;
            nd_node.M = out_c;
            nd_node.N = T;
            nodes.push_back(nd_node);
          }
          kern::amx_batch_run(nodes.data(), nodes.size());
          for (size_t i = 0; i < nd; ++i)
            scatter_raw(chunk.d.data() + i * out_c * T, out_c, T, y, Tout,
                        pad, d0 + i);
        }
        pb_buf = std::move(pb.buf);
      } else {
        scratch.C = out_c;  // GEMM 全量覆写, 免 memset
        scratch.T = T;
        scratch.d.resize(out_c * T);
        for (size_t d = 0; d < k; ++d) {
          amx_gemm_1(panels_[d], pb, scratch.d.data(), out_c, T);
          scatter_phase(scratch, y, T, Tout, pad, d);
        }
        pb_buf = std::move(pb.buf);  // 归还容量供下次复用
      }
    } else
#endif
    {
      // sgemm 升位路径: 每相位即时升位 wd (thread_local)
      thread_local std::vector<float> wd;
      wd.resize(out_c * in_c);
      scratch.reset(out_c, T);
      for (size_t d = 0; d < k; ++d) {
        kern::accel::f16_to_f32(w_f16.data() + d * out_c * in_c, wd.data(),
                                out_c * in_c);
        kern::accel::sgemm('N', 'N', static_cast<int>(out_c),
                           static_cast<int>(T), static_cast<int>(in_c), 1.f,
                           wd.data(), static_cast<int>(in_c), x.d.data(),
                           static_cast<int>(T), 0.f, scratch.d.data(),
                           static_cast<int>(T));
        scatter_phase(scratch, y, T, Tout, pad, d);
      }
    }
    for (size_t c = 0; c < out_c; ++c) {
      float* yr = y.row(c);
      const float bc = b[c];
      for (size_t t = 0; t < Tout; ++t) yr[t] += bc;
    }
  }

 private:
  void scatter_phase(const Tensor2D& scratch, Tensor2D& y, size_t T,
                     size_t Tout, long long pad, size_t d) const {
    scatter_raw(scratch.d.data(), out_c, T, y, Tout, pad, d);
  }
  void scatter_raw(const float* scratch, size_t oc, size_t T, Tensor2D& y,
                   size_t Tout, long long pad, size_t d) const {
    for (size_t o = 0; o < oc; ++o) {
      const float* sr = scratch + o * T;
      float* yr = y.row(o);
      for (size_t j = 0; j < T; ++j) {
        long long opos = static_cast<long long>(j) * stride_u +
                         static_cast<long long>(d) - pad;
        if (opos < 0 || opos >= static_cast<long long>(Tout)) continue;
        yr[static_cast<size_t>(opos)] += sr[j];
      }
    }
  }
};

}  // namespace gsv::sovits
