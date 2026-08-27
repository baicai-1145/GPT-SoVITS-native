// sv.cpp — ERes2NetV2 推理实现(E2-ENC fp16 化: 权重 fp16 直读 + FMLAL 矩阵乘,
// LayerNorm/BN/残差/统计量保持 fp32; 激活转 fp16 暂存复用 scratch)
#include "encoder/sv.hpp"

#include "kern/accel.hpp"
#include "kern/gemv_fmlal.hpp"
#include "kern/kern.hpp"
#include "runtime/gsv_loader.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <map>
#include <stdexcept>
#include <string>
#include <arm_neon.h>

namespace gsv::encoder {

namespace accel = kern::accel;

using rt::GsvFile;
using rt::TensorView;

// E12: 进程级 SV AMX 使能(与 HuBERT 共用 --amx-enc 开关)
bool& amx_sv_enabled() {
  static bool b = false;
  return b;
}

// E12 隔离排查: 按类别禁用 SV AMX 路径(GSV_SV_AMX_SKIP=conv1,convs,conv3,sc)
static unsigned amx_sv_skip_mask() {
  static unsigned m = [] {
    const char* e = std::getenv("GSV_SV_AMX_SKIP");
    if (!e) return 0u;
    std::string s(e);
    unsigned v = 0;
    if (s.find("conv1") != std::string::npos) v |= 1u;
    if (s.find("convs") != std::string::npos) v |= 2u;
    if (s.find("conv3") != std::string::npos) v |= 4u;
    if (s.find("sc") != std::string::npos) v |= 8u;
    if (s.find("aff") != std::string::npos) v |= 16u;
    if (s.find("l3ds") != std::string::npos) v |= 32u;
    return v;
  }();
  return m;
}

// E12 分流门槛(E12 原始): S≥48 且 K≥256
constexpr size_t kAmxSvMinRows = 48;
constexpr size_t kAmxSvMinK = 256;

// E13-SV/T3 装载期权重预打包门槛(.tmp/t3_probe 同形实测):
//   净赚者需要面板就绪; 面板内存/一次性成本极小(<百KB 级), 从宽。
//   E14-SV/C1: 补充 w24 块 3x3 族(rows=24,K=216,L1.convs)——原门槛拒之门外,
//   微基准(.tmp/c1_probe, 配对 min-of-11 @load≈3) 实测 AMX 全链(pack+gemm)
//   2.97ms vs FMLAL 全链(im2col+gemm) 9.67ms = 3.26x 净赚, 三形状稳定。
inline bool amx_sv_worth_packing(size_t rows, size_t K) {
  if (rows >= kAmxSvMinRows && K >= 64)
    return true;
  return rows >= 24 && K >= 192;
}

// E13-SV/T3 派发侧判据: 是否走 AMX(pp)。保证不 regress E12 原有任何
// 已命中位点(原线优先返回 true), 在此基础上按微基准实测净赚域扩展:
//   M=m_rows≥64 且 S≥2048 且 K≥64 —— 排除了两类实测输家:
//   a) 薄 M(=inter 24, 如 L3.aff1 0.6x 打包成本>计算);
//   b) 特小 S/K 组合(池往返+激活面板成本占比过高)。
inline bool amx_sv_go(size_t m_rows, size_t s, size_t K) {
  if (m_rows >= kAmxSvMinRows && s >= kAmxSvMinRows && K >= kAmxSvMinK)
    return true;
  return m_rows >= 64 && s >= 2048 && K >= 64;
}

namespace {
// E13-SV 探针: GSV_SV_AMX_TRACE=1 时对 FMLAL 回退分支打印调用身份(零行为变更)
inline bool amx_trace() {
  static bool t = [] {
    const char* e = std::getenv("GSV_SV_AMX_TRACE");
    return e && *e;
  }();
  return t;
}
}  // namespace

namespace {

inline float relu20(float v) { return v < 0.f ? 0.f : (v > 20.f ? 20.f : v); }

// E13-SV/T4: relu20 的 NEON 版(vmax/vmin 精确, 与标量语义一致)
inline void relu20_neon(float* p, size_t n) {
  const float32x4_t v0 = vdupq_n_f32(0.f);
  const float32x4_t v20 = vdupq_n_f32(20.f);
  const size_t rc = n & ~size_t(3);
  size_t i = 0;
  for (; i + 4 <= rc; i += 4) {
    vst1q_f32(p + i, vmaxq_f32(vminq_f32(vld1q_f32(p + i), v20), v0));
  }
  for (; i < n; ++i) p[i] = relu20(p[i]);
}

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

// 卷积/AFF 权重: 优先 f16 段零拷贝直读; 无 f16 段的小权重(fp32 段存储, 如
// local_att.3) 一次性量化到 own 常驻副本(fp16, 内存减半) —— 前向仍全 FMLAL。
// E12: 当预打包开启时(amx_sv_enabled + 形状分流门槛), 把 panel 填进 *panel_out。
const uint16_t* load_conv_w(const GsvFile& f, const std::string& name,
                           std::vector<uint16_t>* own = nullptr
#if defined(GSV_AMX_GEMM)
                           , kern::AmxPanel* panel_out = nullptr,
                           size_t rows_expect = 0, size_t k_expect = 0
#endif
                           ) {
  const TensorView* t = f.tensor(name);
  if (!t) throw std::runtime_error("sv 缺张量: " + name);
  const uint16_t* ret = t->has_f16() ? t->data_f16_raw() : nullptr;
  if (!ret) {
    if (!own) throw std::runtime_error("sv conv 权重无 f16 段且未提供量化副本槽: " + name);
    own->resize(t->numel());
    std::vector<float> wf(t->numel());
    std::memcpy(wf.data(), t->data_f32(), wf.size() * sizeof(float));
    kern::f32_to_f16(wf.data(), own->data(), wf.size());
    ret = own->data();
  }
#if defined(GSV_AMX_GEMM)
  if (panel_out && amx_sv_enabled() && kern::amx_gemm_available() &&
      rows_expect >= kAmxSvMinRows && k_expect >= kAmxSvMinK) {
    panel_out->rows = rows_expect;
    panel_out->K = k_expect;
    kern::amx_pack_into(ret, rows_expect, k_expect, panel_out->buf);
  }
#else
  (void)rows_expect; (void)k_expect;
#endif
  return ret;
}

}  // namespace

// ---- Bn ----
namespace {
// E13 探针: GSV_SV_INNER_TIMING=1 时聚合 conv2d / conv2d_f16 内 sgemm/fmlal 累计耗时
struct SvInnerTimers {
  double conv2d_ms = 0.0;
  double conv2d_f16_ms = 0.0;
  double aff1_ms = 0.0;
  double aff2_ms = 0.0;
  double blkconv1_ms = 0.0;
  int aff_n = 0;
  long blkconv1_s = 0;
  // 按形状聚合的 FMLAL 回退耗时(key= "cin,h,w k,kw s -> cout")
  std::map<std::string, std::pair<int, double>> f16_by_shape;
  double stage_ms[5] = {0.0, 0.0, 0.0, 0.0, 0.0};  // stage1..4 分段
  // E13-SV/T4: 非 GEMM 元操作分解(BN/ReLU/加法/拷贝/pool/im2col/panel打包/tanh门控)
  double meta_bn_ms = 0.0, meta_relu_ms = 0.0, meta_add_ms = 0.0;
  double meta_copy_ms = 0.0, meta_pool_ms = 0.0, meta_im2col_ms = 0.0;
  double meta_pack_ms = 0.0, meta_aff_ms = 0.0;
  // E14-SV/S1/C0: 站点级细分(GSV_SV_INNER_TIMING=1 门控, 零行为变更)
  std::map<std::string, std::pair<int, double>> site;
};
thread_local SvInnerTimers g_sv_inner;
bool sv_inner_tim() { return std::getenv("GSV_SV_INNER_TIMING") != nullptr; }

// 站点计时器(RAII): 计入 g_sv_inner.site[site]={次数,ms}
struct SvSite {
  const char* site;
  std::chrono::steady_clock::time_point t0;
  bool on;
  explicit SvSite(const char* s)
      : site(s), on(sv_inner_tim()) {
    if (on) t0 = std::chrono::steady_clock::now();
  }
  ~SvSite() {
    if (!on) return;
    auto& e = g_sv_inner.site[site];
    e.first++;
    e.second += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0).count();
  }
};// 计时起点: 关闭时返回零值 epoch(与现探针风格一致)
inline std::chrono::steady_clock::time_point sv_tp() {
  return sv_inner_tim() ? std::chrono::steady_clock::now()
                        : std::chrono::steady_clock::time_point{};
}
inline void sv_acc(std::chrono::steady_clock::time_point t0, double& dst) {
  if (sv_inner_tim())
    dst += std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0).count();
}
}  // namespace

void SvEngine::Bn::apply(float* x, int c, size_t s) const {
  const auto tp = sv_tp();
  if (!amx_sv_enabled()) {
    // 默认路径(旗关): 保持 C1 原双精度链位级不变(HANDOFF §6)
    constexpr double eps0 = 1e-5;
    for (int ci = 0; ci < c; ++ci) {
      const double a0 = static_cast<double>(g[ci]) /
                        std::sqrt(static_cast<double>(var[ci]) + eps0);
      const double beta0 = static_cast<double>(b[ci]);
      const double mu0 = static_cast<double>(mean[ci]);
      float* row0 = x + static_cast<size_t>(ci) * s;
      for (size_t i = 0; i < s; ++i)
        row0[i] = static_cast<float>((static_cast<double>(row0[i]) - mu0) * a0 + beta0);
    }
    sv_acc(tp, g_sv_inner.meta_bn_ms);
    return;
  }
  // E13-SV/T4c(--amx-enc 档): 每 channel 预算一次 scale/beta(fp32), 行内
  // NEON FMA。double 链属实现精度选择(非 torch 语义要求); 门禁 corr/mel。
  constexpr float eps = 1e-5f;
  const int rc = int(s & ~size_t(3));
  for (int ci = 0; ci < c; ++ci) {
    const float a = g[ci] / std::sqrt(var[ci] + eps);
    const float beta = b[ci];
    const float mu = mean[ci];
    float* row = x + static_cast<size_t>(ci) * s;
    const float32x4_t va = vdupq_n_f32(a);
    const float32x4_t vmu = vdupq_n_f32(mu);
    const float32x4_t vbeta = vdupq_n_f32(beta);
    int i = 0;
    for (; i + 4 <= rc; i += 4) {
      const float32x4_t vc = vsubq_f32(vld1q_f32(row + i), vmu);
      vst1q_f32(row + i, vfmaq_f32(vbeta, vc, va));  // beta + (x-mu)*a
    }
    // 尾部逐点(与主链同公式)
    for (; i < int(s); ++i)
      row[i] = (row[i] - mu) * a + beta;
  }
  sv_acc(tp, g_sv_inner.meta_bn_ms);
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
  const auto tp_im = sv_tp();
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
  sv_acc(tp_im, g_sv_inner.meta_im2col_ms);
  out.assign(static_cast<size_t>(c_out) * S, 0.f);
  const auto tp_sg0 = sv_inner_tim() ? std::chrono::steady_clock::now()
                                     : std::chrono::steady_clock::time_point{};
  accel::sgemm('N', 'T', c_out, static_cast<int>(S), K, 1.0f, wt, K, cols.data(), K,
               0.0f, out.data(), static_cast<int>(S));
  if (sv_inner_tim())
    g_sv_inner.conv2d_ms += std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - tp_sg0).count();
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
  const auto tp_im2 = sv_tp();
  // E13-SV/T4d: k3s3p1 情形整张量一次向量 FCVT(位级同标量), im2col 时
  // 只做 f16 散布写 —— 转换次数从 S*K=9*S*W² 降到 C*H*W(约 /9·C/W 倍),
  // w24 convs(FMLAL 保留族)的回退成本大头即在此。
  if (kh == 3 && kw == 3 && stride == 1 && pad == 1 && h > 2 && w > 2) {
    const size_t total = size_t(c_in) * h * w;
    if (cvt16_.size() < total) cvt16_.resize(total);
    {
      size_t i = 0;
      const float* sp = in;
      uint16_t* dp = cvt16_.data();
      for (; i + 8 <= total; i += 8, sp += 8, dp += 8) {
        const float32x4_t v0 = vld1q_f32(sp);
        const float32x4_t v1 = vld1q_f32(sp + 4);
        vst1q_f16(reinterpret_cast<__fp16*>(dp),
                  vcombine_f16(vcvt_f16_f32(v0), vcvt_f16_f32(v1)));
      }
      for (; i < total; ++i, ++sp, ++dp)
        *dp = kern::f32_to_f16_scalar(*sp);
    }
    size_t s_idx2 = 0;
    for (int oy = 0; oy < oh; ++oy)
      for (int ox = 0; ox < ow; ++ox, ++s_idx2) {
        uint16_t* row = cols16_.data() + s_idx2 * K;
        for (int c = 0; c < c_in; ++c) {
          const uint16_t* plane = cvt16_.data() + (size_t)c * h * w;
          for (int ky = 0; ky < 3; ++ky) {
            const int iy = oy + ky - 1;
            uint16_t* dk = row + (size_t)c * 9 + size_t(ky) * 3;
            if (iy < 0 || iy >= h) {
              dk[0] = dk[1] = dk[2] = 0;  // 复用缓冲: 越界行必须显式补零
              continue;
            }
            const uint16_t* rowp = plane + (size_t)iy * w;
            for (int kx = 0; kx < 3; ++kx) {
              const int ix = ox + kx - 1;
              dk[kx] = (ix >= 0 && ix < w) ? rowp[ix] : 0;
            }
          }
        }
      }
    sv_acc(tp_im2, g_sv_inner.meta_im2col_ms);
  } else {
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
    sv_acc(tp_im2, g_sv_inner.meta_im2col_ms);
  }
  out.assign(static_cast<size_t>(c_out) * S, 0.f);
  const auto tp_fl0 = sv_inner_tim() ? std::chrono::steady_clock::now()
                                     : std::chrono::steady_clock::time_point{};
  kern::gemm_f16x_fmlal(w16, cols16_.data(), out.data(), static_cast<size_t>(c_out),
                        S, K);
  if (sv_inner_tim()) {
    g_sv_inner.conv2d_f16_ms += std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() - tp_fl0).count();
    char key[80];
    std::snprintf(key, sizeof key, "%d,%d,%d k%ds%d->%d", c_in, h, w, kh, kw,
                  c_out);
    auto& e = g_sv_inner.f16_by_shape[key];
    e.first++;
    e.second += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - tp_fl0).count();
  }
}

#if defined(GSV_AMX_GEMM)
// E12: 直接把输入激活卷积 im2col 结果写成 AMX 激活 panel 布局。
// 逻辑同 conv2d_f16 的 im2col, 但写到 panel 布局而非中间行主缓冲 —— 免一次
// S*K 中间写和重复分配(热路径 reserve+resize)。输出: nt 个 tile, 每 tile
// 每 k 一行 64B(k = c_in*kh*kw 展平; 与 kern::pack_panel 同构)。
namespace {
// E13-SV/T4: k1/s1 快速路径 —— 源在通道内连续(s 主序), panel 对固定 c 是
// 步长 32 半字的等值散布。四宽 FCVT(RNE, 与标量 static_cast<__fp16> 位级一致)
// + 8B 向量存。
void pack_k1s1_rows_neon(const float* in, int c_in, size_t S,
                         size_t rend, uint16_t* tb) {
  const int rc = int(rend & ~size_t(3));
  for (int c = 0; c < c_in; ++c) {
    const float* chp = in + (size_t)c * S;
    uint16_t* crow = tb + (size_t)c * 32;
    int r = 0;
    for (; r + 4 <= rc; r += 4) {
      const float32x4_t v = vld1q_f32(chp + r);
      vst1_u16(crow + r, vreinterpret_u16_f16(vcvt_f16_f32(v)));
    }
    for (; r < int(rend); ++r)
      crow[r] = kern::f32_to_f16_scalar(chp[r]);
  }
}

// E13-SV/T4: 3x3 s1 p1 快速路径。张量已整批向量 FCVT 到 f16(位级与标量
// 一致)。注意 j0 起的 32 个展平样本可能跨源行边界(w=737 时 tile 跨 1~2
// 行)，按「同行连续段」分段处理，段内做三条水平窗拷贝。
void pack_3x3s1_scatter_neon(const std::vector<uint16_t>& tmp16, int c_in,
                             int h, int w, size_t j0, size_t rend,
                             uint16_t* tb) {
  // 注意: 激活面板缓冲跨调用复用容量, 无效 k 位必须显式写零
  // (与标量路径的全覆盖语义一致), 不得残留上一次的字节。
  size_t r = 0;
  while (r < rend) {
    const size_t flat = j0 + r;
    const int yy = int(flat / w);
    const int x0 = int(flat % w);
    const size_t seg =
        std::min(rend - r, size_t(w) - size_t(x0));  // 本段同行长度
    for (int c = 0; c < c_in; ++c) {
      const uint16_t* plane = tmp16.data() + (size_t)c * h * w;
      uint16_t* crow = tb + (size_t)(c * 9) * 32;
      for (int ky = 0; ky < 3; ++ky) {
        const int iy = yy + ky - 1;
        const uint16_t* row =
            (iy >= 0 && iy < h) ? plane + (size_t)iy * w : nullptr;
        for (int kx = 0; kx < 3; ++kx) {
          const int d = kx - 1;
          uint16_t* dk = crow + (size_t(ky * 3 + kx) * 32);
          // 与标量参考同义: 每个 panel 位必写一次, 源越界写 0。
          // 有效 q 区间: 0 ≤ x0+q+d < w 且行在界内
          const int qlo = std::max(0, -(x0 + d));       // sx≥0
          const int qhi = std::min(int(seg), w - x0 - d);  // sx<w exclusive
          for (int q = 0; q < qlo; ++q) dk[r + q] = 0;
          const int qe2 = std::max(qhi, qlo);
          if (row) {
            for (int q = qlo; q < qe2; ++q) dk[r + q] = row[x0 + d + q];
            for (int q = qe2; q < int(seg); ++q) dk[r + q] = 0;
          } else {
            for (int q = qlo; q < int(seg); ++q) dk[r + q] = 0;
          }
        }
      }
    }
    r += seg;
  }
}
}  // namespace

void pack_conv_cols_to_panel(const float* in, int c_in, int h, int w,
                             int kh, int kw, int stride, int pad, size_t S, size_t K,
                             std::vector<uint8_t>& buf) {
  const size_t nt = (S + 31) / 32;
  const size_t need = nt * K * 64 + 64;
  if (buf.capacity() < need) buf.reserve(need);
  buf.resize(need);
  const uintptr_t p = (uintptr_t)buf.data();
  uint8_t* dst = buf.data() + ((64 - (p & 63)) & 63);
  if (const size_t tr_last = S - (nt - 1) * 32; tr_last < 32) {
    uint8_t* d_tail = dst + (nt - 1) * K * 64;
    const size_t zoff = tr_last * 2;
    for (size_t k = 0; k < K; ++k)
      std::memset(d_tail + k * 64 + zoff, 0, 64 - zoff);
  }
#if defined(GSV_AMX_GEMM)
  // E13-SV 临时调试开关: 置位时禁用两条快速打包路径(对拍用)
  const bool force_scalar = [] { const char* e = std::getenv("GSV_SV_PACK_SCALAR"); return e && *e; }();
  // T4: 1x1 s1 卷积的 im2col=恒等映射(h/w 平铺与展平空间同序) —
  // 全部 AMX 激活体积的大头(L1-L4 全部 conv3/sc/AFF/中后期 conv1)走此路。
  if (!force_scalar && kh == 1 && kw == 1 && stride == 1 && pad == 0) {
    for (size_t t = 0; t < nt; ++t) {
      const size_t j0 = t * 32;
      const size_t rend = std::min(size_t(32), S - j0);
      pack_k1s1_rows_neon(in + j0, c_in, S, rend,
                          reinterpret_cast<uint16_t*>(dst + t * K * 64));
    }
    return;
  }
  // T4b: 3x3 s1 p1(convs 组形状)。h/w 平铺展平序下窗口仅含垂直两个
  // 整行越界情形, 其余访问是三条水平连续段 —— 行级 cvt 一次复用三窗。
  if (!force_scalar && kh == 3 && kw == 3 && stride == 1 && pad == 1 && h > 2 && w > 2) {
    // 整张量一次向量 FCVT(约 im2col 转换量的 1/9)，逐 tile 只做散布拷贝
    const size_t total = size_t(c_in) * h * w;
    thread_local std::vector<uint16_t> t_cvt3;
    if (t_cvt3.size() < total) t_cvt3.resize(total);
    {
      size_t i = 0;
      const float* sp = in;
      uint16_t* dp = t_cvt3.data();
      for (; i + 4 <= total; i += 4, sp += 4, dp += 4) {
        const float32x4_t v = vld1q_f32(sp);
        vst1_u16(dp, vreinterpret_u16_f16(vcvt_f16_f32(v)));
      }
      for (; i < total; ++i, ++sp, ++dp)
        *dp = kern::f32_to_f16_scalar(*sp);
    }
    for (size_t t = 0; t < nt; ++t) {
      const size_t j0 = t * 32;
      const size_t rend = std::min(size_t(32), S - j0);
      pack_3x3s1_scatter_neon(t_cvt3, c_in, h, w, j0, rend,
                              reinterpret_cast<uint16_t*>(dst + t * K * 64));
    }
    return;
  }
#endif
  // 输出位置: 对 panel 行 s(样本)与列 k(收缩维): dst[k*64+s_in_tile*2]
  // 逐输入通道扫不太自然(panel 布局按 s 外循环); 仍按 S 主序扫有效位置
  const int oh = (h + 2 * pad - kh) / stride + 1;
  const int ow = (w + 2 * pad - kw) / stride + 1;
  size_t s_idx = 0;
  for (int oy = 0; oy < oh; ++oy)
    for (int ox = 0; ox < ow; ++ox, ++s_idx) {
      const size_t tile = s_idx / 32;
      const size_t row_in_tile = s_idx % 32;
      uint16_t* base = reinterpret_cast<uint16_t*>(dst + tile * K * 64);
      for (int c = 0; c < c_in; ++c)
        for (int ky = 0; ky < kh; ++ky)
          for (int kx = 0; kx < kw; ++kx) {
            const int iy = oy * stride - pad + ky;
            const int ix = ox * stride - pad + kx;
            const float v = (iy >= 0 && iy < h && ix >= 0 && ix < w)
                                ? in[(static_cast<size_t>(c) * h + iy) * w + ix]
                                : 0.f;
            const size_t k = (static_cast<size_t>(c) * kh + ky) * kw + kx;
            base[k * 32 + row_in_tile] = kern::f32_to_f16_scalar(v);
          }
    }
}

// E12: AMX 卷积 — 1x1 直连(im2col=identity)或已预 pack 面板。
// 与 conv2d_f16 同语义(out[c_out,S] 通道主); FMAM 路径回退即原实现。
// E14-SV/C0: 可选站点标签(仅探针归因用, 不影响数值)。
void SvEngine::conv2d_amx(const kern::AmxPanel& w_panel, const float* in,
                          int c_in, int h, int w, int kh, int kw, int stride,
                          int pad, int c_out, std::vector<float>& out,
                          const char* site /*=nullptr*/) {
  const int oh = (h + 2 * pad - kh) / stride + 1;
  const int ow = (w + 2 * pad - kw) / stride + 1;
  const size_t S = static_cast<size_t>(oh) * ow;
  const size_t K = w_panel.K;
  out.resize(static_cast<size_t>(c_out) * S);
  {
    static thread_local int n_trace = 0;
    if (const char* t = std::getenv("GSV_SV_AMX_TRACE"); t && *t && n_trace < 64)
      std::fprintf(stderr, "[sv-amx] %dx%dx%d k=%dx%d s%d -> out[%d,S=%zu]\n",
                   c_in, h, w, kh, kw, stride, c_out, S);
    // GSV_SV_AMX_DUMP=<n>: 第 n 次 AMX conv 调用时把输入/输出 dump 到 /tmp
    if (const char* d = std::getenv("GSV_SV_AMX_DUMP")) {
      const int want = std::atoi(d);
      static thread_local int n_dump = 0;
      if (want > 0 && ++n_dump == want) {
        FILE* fp = std::fopen("/tmp/sv_amx_in.bin", "wb");
        if (fp) { std::fwrite(in, 4, size_t(c_in) * h * w, fp); std::fclose(fp); }
        fp = std::fopen("/tmp/sv_amx_out.bin", "wb");
        if (fp) { std::fwrite(out.data(), 4, out.size(), fp); std::fclose(fp); }
        fp = std::fopen("/tmp/sv_amx_meta.txt", "w");
        if (fp) { std::fprintf(fp, "%d %d %d %d %d %d %d %zu\n", c_in, h, w, kh, kw, stride, c_out, S); std::fclose(fp); }
      }
    }
  }
  {
    const auto tp_pk = sv_tp();
    pack_conv_cols_to_panel(in, c_in, h, w, kh, kw, stride, pad, S, K,
                            sv_act_scratch_);
    if (sv_inner_tim()) {
      const double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - tp_pk).count();
      g_sv_inner.meta_pack_ms += ms;
      auto& e = g_sv_inner.site[site ? site : "amx_pack"];
      e.first++;
      e.second += ms;
    }
  }
  kern::AmxPanel pb;
  pb.rows = S;
  pb.K = K;
  pb.buf.swap(sv_act_scratch_);
  // E12: 契约 C[M,N]=pa·pbᵀ — pa=权重[c_out,K], pb=激活[S,K], 输出通道主。
  // 注: amx_batch_run 与 gemm_f16_amx_pp 数值序一致, 单发直调以隔离验收。
  kern::gemm_f16_amx_pp(w_panel, pb, out.data(), static_cast<size_t>(c_out), S);
  pb.buf.swap(sv_act_scratch_);  // 收回容量复用
}
#endif

// ---- Aff(fusion.AFF): xo = x*(1+tanh(att)) + ds*(1-tanh(att)) ----
void SvEngine::Aff::apply(const float* x, const float* ds, float* out, int h, int w,
                           std::vector<uint16_t>& xh, std::vector<uint16_t>& xh2
#if defined(GSV_AMX_GEMM)
                           , std::vector<uint8_t>& amx_scratch
#endif
                           ) {
  const size_t s = static_cast<size_t>(h) * w;
  cat_.resize(2 * static_cast<size_t>(ch) * s);
  {
    SvSite st("aff_catbuild");
    std::memcpy(cat_.data(), x, static_cast<size_t>(ch) * s * sizeof(float));
    std::memcpy(cat_.data() + static_cast<size_t>(ch) * s, ds,
                static_cast<size_t>(ch) * s * sizeof(float));
  }
  att_.resize(static_cast<size_t>(inter) * s);
#if defined(GSV_AMX_GEMM)
  if (w1_panel_ready && !(amx_sv_skip_mask() & 16u) && s >= kAmxSvMinRows &&
      size_t(2 * ch) >= kAmxSvMinK) {
    // E12: gemm1 走 AMX — cat 已在上面填好(通道主 [2ch,s]),
    // 转置直写激活 panel 后单节点派发。
    {
      const auto tp_pk = sv_tp();
      pack_conv_cols_to_panel(cat_.data(), /*c_in=*/2 * ch,
                              /*h=*/1, /*w=*/static_cast<int>(s),
                              1, 1, 1, 0, s, size_t(2 * ch), amx_scratch);
      if (sv_inner_tim()) {
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - tp_pk).count();
        g_sv_inner.meta_pack_ms += ms;
        auto& e = g_sv_inner.site["aff_w1_pack"];
        e.first++;
        e.second += ms;
      }
    }
    kern::AmxPanel pb;
    pb.rows = s;
    pb.K = size_t(2 * ch);
    pb.buf.swap(amx_scratch);
    // 契约: C[M,N]=pa·pbᵀ — pa=W1[inter,2ch], pb=catᵀ[s,2ch], C=[inter,s]
    kern::AmxBatchNode nd;
    nd.phase = 0; nd.pa = &w1_panel; nd.pb = &pb;
    nd.c = att_.data(); nd.M = size_t(inter); nd.N = s;
    kern::amx_batch_run(&nd, 1);
    pb.buf.swap(amx_scratch);  // 收回容量
  } else
#endif
  {
    // FMAL 回退: 转置量化 cat→[S,2ch]fp16 后 gemm(A=W1, B=catᵀ)
    if (amx_trace())
      std::fprintf(stderr, "[sv-fb] aff1 ch=%d inter=%d S=%zu\n", ch, inter, s);
    if (xh.size() < 2 * static_cast<size_t>(ch) * s) xh.resize(2 * static_cast<size_t>(ch) * s);
    kern::f32_trans_to_f16(cat_.data(), xh.data(), 2 * ch, s);
    const auto tp_aff1 = sv_inner_tim() ? std::chrono::steady_clock::now()
                                        : std::chrono::steady_clock::time_point{};
    kern::gemm_f16x_fmlal(w1, xh.data(), att_.data(),
                          static_cast<size_t>(inter), s,
                          2 * static_cast<size_t>(ch));
    if (sv_inner_tim()) {
      g_sv_inner.aff1_ms += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - tp_aff1).count();
      ++g_sv_inner.aff_n;
    }
  }
#ifdef C1_TRACE
  std::fprintf(stderr, "    [aff gemm1 done]\n");
#endif
  {
    // E13-SV/T4: 偏置逐行 NEON 加法(精确同序, 逐元素加法无重排)
    const int rc = int(s & ~size_t(3));
    for (int e = 0; e < inter; ++e) {
      float* row = att_.data() + static_cast<size_t>(e) * s;
      const float32x4_t vb = vdupq_n_f32(b1[static_cast<size_t>(e)]);
      int i = 0;
      for (; i + 4 <= rc; i += 4)
        vst1q_f32(row + i, vaddq_f32(vld1q_f32(row + i), vb));
      for (; i < int(s); ++i) row[i] += b1[static_cast<size_t>(e)];
    }
  }
  bn1.apply(att_.data(), inter, s);
  {
    SvSite st("aff_silu");
    gsv::kern::silu(att_.data(), att_.data(), att_.size());
  }
  att_t_.resize(static_cast<size_t>(ch) * s);
  // gemm2 同构: att_t[ch,S] = W2[ch,inter]·att[inter,S] → 转置量化 + gemm 直出通道主
#if defined(GSV_AMX_GEMM)
  if (w2_panel_ready && !(amx_sv_skip_mask() & 16u) && s >= kAmxSvMinRows &&
      size_t(inter) >= kAmxSvMinK) {
    // E12: gemm2 走 AMX — att 通道主 [inter,s] 直转 panel
    {
      const auto tp_pk = sv_tp();
      pack_conv_cols_to_panel(att_.data(), /*c_in=*/inter,
                              /*h=*/1, /*w=*/static_cast<int>(s),
                              1, 1, 1, 0, s, size_t(inter), amx_scratch);
      if (sv_inner_tim()) {
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - tp_pk).count();
        g_sv_inner.meta_pack_ms += ms;
        auto& e = g_sv_inner.site["aff_w2_pack"];
        e.first++;
        e.second += ms;
      }
    }
    kern::AmxPanel pb;
    pb.rows = s;
    pb.K = size_t(inter);
    pb.buf.swap(amx_scratch);
    // 契约: C[M,N]=pa·pbᵀ — pa=W2[ch,inter], pb=attᵀ[s,inter], C=[ch,s]
    kern::AmxBatchNode nd;
    nd.phase = 0; nd.pa = &w2_panel; nd.pb = &pb;
    nd.c = att_t_.data(); nd.M = size_t(ch); nd.N = s;
    kern::amx_batch_run(&nd, 1);
    pb.buf.swap(amx_scratch);  // 收回容量
  } else
#endif
  {
    if (amx_trace())
      std::fprintf(stderr, "[sv-fb] aff2 ch=%d inter=%d S=%zu\n", ch, inter, s);
    if (xh2.size() < static_cast<size_t>(inter) * s) xh2.resize(static_cast<size_t>(inter) * s);
    kern::f32_trans_to_f16(att_.data(), xh2.data(), inter, s);  // [inter,S]→[S,inter]fp16
    const auto tp_aff2 = sv_inner_tim() ? std::chrono::steady_clock::now()
                                        : std::chrono::steady_clock::time_point{};
    kern::gemm_f16x_fmlal(w2, xh2.data(), att_t_.data(),
                          static_cast<size_t>(ch), s,
                          static_cast<size_t>(inter));
    if (sv_inner_tim()) {
      g_sv_inner.aff2_ms += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - tp_aff2).count();
    }
  }
#ifdef C1_TRACE
  std::fprintf(stderr, "    [aff gemm2 done]\n");
#endif
  {
    const int rc = int(s & ~size_t(3));
    for (int e = 0; e < ch; ++e) {
      float* row = att_t_.data() + static_cast<size_t>(e) * s;
      const float32x4_t vb = vdupq_n_f32(b2[static_cast<size_t>(e)]);
      int i = 0;
      for (; i + 4 <= rc; i += 4)
        vst1q_f32(row + i, vaddq_f32(vld1q_f32(row + i), vb));
      for (; i < int(s); ++i) row[i] += b2[static_cast<size_t>(e)];
    }
  }
  bn2.apply(att_t_.data(), ch, s);
  {
    const auto tp_af = sv_tp();
    for (size_t i = 0; i < static_cast<size_t>(ch) * s; ++i) {
      const float t = std::tanh(att_t_[i]);
      out[i] = x[i] * (1.0f + t) + ds[i] * (1.0f - t);
    }
    if (sv_inner_tim()) {
      const double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - tp_af).count();
      g_sv_inner.meta_aff_ms += ms;
      auto& e = g_sv_inner.site["aff_gate"];
      e.first++;
      e.second += ms;
    }
  }
  {
    SvSite st("aff_lastout");
    last_out.assign(out, out + static_cast<size_t>(ch) * s);
  }
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
  // E13-SV/T4: conv1 输出直写 eng.cur_, 消除原先 nxt_→cur_ 的整块拷贝
  std::vector<float>& c1out = eng.cur_;
#if defined(GSV_AMX_GEMM)
  if (conv1_panel_ready && !(amx_sv_skip_mask() & 1u) &&
      amx_sv_go(size_t(co1), size_t(h_in) * w_in, size_t(c_in))) {
    // E13-SV/T3: k1 卷积恒等 dense GEMM, k==1 时 im2col 无重叠复用,
    // k1s2 与 k1s1 数学一致(每输出位置独立取单列, 边界零填等等价);
    // 解除 E12 的 stride==1 分流限制, 入口块 conv1(s2) 走 AMX 直派。
    // 派发门槛从 S/K≥48/256 收紧为 amx_sv_go(基于 t3_probe 实测净赚域)。
    eng.conv2d_amx(conv1_panel, in, c_in, h_in, w_in, 1, 1, stride, 0, co1,
                   c1out, "blk_conv1");
    if (stride != 1) {  // 下游 BN/分块逻辑按新分辨率(与 else 分支同式)
      h = (h_in - 1) / stride + 1;
      w = (w_in - 1) / stride + 1;
    }
  } else
#endif
  if (stride == 1) {
    // 1x1 s1: 零 im2col。in 通道主 [c_in,S] → 转置量化 xh_[S,c_in];
    // gemm(A=W[co1,c_in], B=xh) 直接得通道主输出 —— 不经 tmp_
    // (in 可能别名 eng.tmp_, 不得写回)。
    const size_t s = static_cast<size_t>(h) * w;
    if (amx_trace())
      std::fprintf(stderr, "[sv-fb] blk_conv1 cin=%d->%d S=%zu\n", c_in, co1, s);
    if (eng.xh_.size() < static_cast<size_t>(c_in) * s)
      eng.xh_.resize(static_cast<size_t>(c_in) * s);
    kern::f32_trans_to_f16(in, eng.xh_.data(), c_in, s);
    c1out.resize(static_cast<size_t>(co1) * s);
    const auto tp_c1 = sv_inner_tim() ? std::chrono::steady_clock::now()
                                      : std::chrono::steady_clock::time_point{};
    kern::gemm_f16x_fmlal(conv1_w, eng.xh_.data(), c1out.data(),
                          static_cast<size_t>(co1), s,
                          static_cast<size_t>(c_in));
    if (sv_inner_tim()) {
      g_sv_inner.blkconv1_ms += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - tp_c1).count();
      g_sv_inner.blkconv1_s += long(s);
    }
  } else {
    eng.conv2d_f16(in, c_in, h, w, conv1_w, co1, 1, 1, stride, 0, eng.cols_, c1out);
    h = (h - 1) / stride + 1;
    w = (w - 1) / stride + 1;
  }
  const size_t s = static_cast<size_t>(h) * w;
  {
    const auto tp_bn1 = sv_tp();
    bn1.apply(c1out.data(), co1, s);
    relu20_neon(c1out.data(), c1out.size());
    sv_acc(tp_bn1, g_sv_inner.meta_relu_ms);  // bn1+relu20 合并窗口(见证据说明)
  }
#ifdef C1_TRACE
  std::fprintf(stderr, "  [conv1+bn ok s=%zu]\n", s);
#endif

  // 分块处理链(torch 语义): sp 从原始块 0 出发, 每轮处理后作为下一轮 fuse/相加的左操作数
  // E13-SV/T4: conv1 已直写 eng.cur_(c1out), 此处无需再整块拷贝。
  // 注意: c1out 语义 = [co1, s] 即 concat 的槽位 0..3 连续排布(chunk(i) 基址)。
  eng.cur_.resize(static_cast<size_t>(co1) * s);      // conv1 输出恰好是 co1×s
  eng.tmp2_.resize(static_cast<size_t>(co1) * s);     // 拼接缓冲(槽位 i × width)
  auto chunk = [&](int i) -> const float* {
    return eng.cur_.data() + static_cast<size_t>(i) * width * s;
  };
  std::vector<float> sp, fus;
  {
    SvSite st("blk_spfus_ctor");
    // 等价于原 vector(n) 构造: 值初始化(零填充整块) — 探针归因此段耗时
    sp.resize(static_cast<size_t>(width) * s);
    fus.resize(static_cast<size_t>(width) * s);
  }
  for (int i = 0; i < scale; ++i) {
    if (i > 0) {
#ifdef C1_TRACE
      if (aff) std::fprintf(stderr, "  [fuse i=%d]\n", i);
#endif
      if (aff) {
        fuses[static_cast<size_t>(i - 1)].apply(sp.data(), chunk(i), fus.data(), h, w,
                                               eng.xh_, eng.xh2_
#if defined(GSV_AMX_GEMM)
                                               , eng.sv_act_scratch_
#endif
        );
        std::memcpy(sp.data(), fus.data(), sp.size() * sizeof(float));
      } else {
        const auto tp_ad = sv_tp();
        // E13-SV/T4: 向量四宽加法(load 完成后才 store, 别名安全; 精确同序)
        const float* rp = chunk(i);
        size_t j = 0;
        const size_t rc = sp.size() & ~size_t(3);
        for (; j + 4 <= rc; j += 4)
          vst1q_f32(sp.data() + j,
                    vaddq_f32(vld1q_f32(sp.data() + j), vld1q_f32(rp + j)));
        for (; j < sp.size(); ++j) sp[j] += rp[j];
        sv_acc(tp_ad, g_sv_inner.meta_add_ms);
      }
    }
    // convs[i](3x3 p1 s1) + bns + ReLU20 → 拼接槽位 i
#if defined(GSV_AMX_GEMM)
    // E14-SV/C1 门槛(微基准 .tmp/c1_probe 配对 min-of-11 @load≈3):
    //   - S≥512: 全部 convs 命中点实际最小 S=L4.b2 696(L4 族 6.81x 净赚);
    //     原 S≥2048 会把 L4 踢回 FMLAL(12 次 43.5ms 回归) —— 已撤销。
    //   - K=width*9≥192: 新纳入 w24 块(K=216, L1 族 2.9-3.4x 净赚);
    //     保留原排除段(更小 width 无实例且探针外无证据)。
    if (i < int(convs_panels_ready.size()) && convs_panels_ready[size_t(i)] &&
        !(amx_sv_skip_mask() & 2u) &&
        static_cast<size_t>(h) * static_cast<size_t>(w) >= 512 &&
        static_cast<size_t>(width) * 9 >= 192) {
      eng.nxt_.resize(static_cast<size_t>(width) * s);
      eng.conv2d_amx(convs_panels[size_t(i)], i == 0 ? chunk(0) : sp.data(),
                     width, h, w, 3, 3, 1, 1, width, eng.nxt_, "blk_convs");
    } else
#endif
    {
      if (amx_trace())
        std::fprintf(stderr, "[sv-fb] convs i=%d w=%d S=%zu\n", i, width,
                     static_cast<size_t>(h) * w);
      eng.conv2d_f16(i == 0 ? chunk(0) : sp.data(), width, h, w,
                     convs_w[static_cast<size_t>(i)],
                     width, 3, 3, 1, 1, eng.cols_, eng.nxt_);
    }
    bns[static_cast<size_t>(i)].apply(eng.nxt_.data(), width, s);
    {
      const auto tp_rl = sv_tp();
      relu20_neon(eng.nxt_.data(), eng.nxt_.size());
      std::memcpy(eng.tmp2_.data() + static_cast<size_t>(i) * width * s,
                  eng.nxt_.data(), eng.nxt_.size() * sizeof(float));
      std::memcpy(sp.data(), eng.nxt_.data(), eng.nxt_.size() * sizeof(float));
      sv_acc(tp_rl, g_sv_inner.meta_copy_ms);  // relu+双拷贝合并窗口
    }
  }

  // conv3(k1) + bn3
#if defined(GSV_AMX_GEMM)
  if (conv3_panel_ready && !(amx_sv_skip_mask() & 4u) &&
      amx_sv_go(size_t(exp_planes), s, size_t(co1))) {
    eng.nxt_.resize(static_cast<size_t>(exp_planes) * s);
    eng.conv2d_amx(conv3_panel, eng.tmp2_.data(), co1, h, w, 1, 1, 1, 0,
                   exp_planes, eng.nxt_, "blk_conv3");
  } else
#endif
  {
    if (amx_trace())
      std::fprintf(stderr, "[sv-fb] conv3 cin=%d->%d S=%zu\n", co1, exp_planes,
                   static_cast<size_t>(h) * w);
    eng.conv2d_f16(eng.tmp2_.data(), co1, h, w, conv3_w, exp_planes, 1, 1, 1, 0,
                   eng.cols_, eng.nxt_);
  }
  bn3.apply(eng.nxt_.data(), exp_planes, s);

#ifdef C1_TRACE
  std::fprintf(stderr, "  [conv3 done, shortcut=%d]\n", has_shortcut);
#endif
  // shortcut(输入仍是 [c_in,h_in,w_in], 经 stride 下采样到输出分辨率 —— 必须用原始 h_in/w_in!)
  if (!has_shortcut) {
    const auto tp_ad2 = sv_tp();
{
      const float* bp = in;
      const size_t n2 = eng.nxt_.size();
      size_t j = 0;
      const size_t rc = n2 & ~size_t(3);
      for (; j + 4 <= rc; j += 4)
        vst1q_f32(eng.nxt_.data() + j,
                  vaddq_f32(vld1q_f32(eng.nxt_.data() + j), vld1q_f32(bp + j)));
      for (; j < n2; ++j) eng.nxt_[j] += bp[j];
    }
    sv_acc(tp_ad2, g_sv_inner.meta_add_ms);
  } else {
#if defined(GSV_AMX_GEMM)
    if (sc_panel_ready && !(amx_sv_skip_mask() & 8u) &&
        amx_sv_go(size_t(exp_planes), size_t(h_in) * w_in, size_t(c_in))) {
      eng.tmp_.resize(static_cast<size_t>(exp_planes) * s);
      eng.conv2d_amx(sc_panel, in, c_in, h_in, w_in, 1, 1, stride, 0,
                     exp_planes, eng.tmp_, "blk_sc");
    } else
#endif
    {
      if (amx_trace())
        std::fprintf(stderr, "[sv-fb] sc cin=%d->%d stride=%d S=%zu\n", c_in,
                     exp_planes, stride, static_cast<size_t>(h_in) * w_in);
      eng.conv2d_f16(in, c_in, h_in, w_in, sc_w, exp_planes, 1, 1, stride, 0, eng.cols_,
                     eng.tmp_);
    }
    sc_bn.apply(eng.tmp_.data(), exp_planes, s);
    {
      const auto tp_ad3 = sv_tp();
{
      const float* bp = eng.tmp_.data();
      const size_t n2 = eng.nxt_.size();
      size_t j = 0;
      const size_t rc = n2 & ~size_t(3);
      for (; j + 4 <= rc; j += 4)
        vst1q_f32(eng.nxt_.data() + j,
                  vaddq_f32(vld1q_f32(eng.nxt_.data() + j), vld1q_f32(bp + j)));
      for (; j < n2; ++j) eng.nxt_[j] += bp[j];
    }
      sv_acc(tp_ad3, g_sv_inner.meta_add_ms);
    }
  }
  relu20_neon(eng.nxt_.data(), eng.nxt_.size());
#ifdef C1_TRACE
  std::fprintf(stderr, "  [blk done]\n");
#endif
  {
    const auto tp_asg = sv_tp();
    last_out.assign(eng.nxt_.begin(), eng.nxt_.end());
    if (sv_inner_tim()) {
      g_sv_inner.meta_copy_ms += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - tp_asg).count();
      auto& e = g_sv_inner.site["blk_finalassign"];
      e.first++;
      e.second += std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - tp_asg).count();
    }
  }
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

  const int co1_local = blk.width * blk.scale;
  const int cin_conv1 = static_cast<int>(
      f_->tensor((p + "conv1.weight").c_str())->dims[1]);
  blk.conv1_w = load_conv_w(*f_, p + "conv1.weight", nullptr
#if defined(GSV_AMX_GEMM)
      , &blk.conv1_panel, size_t(co1_local), size_t(cin_conv1)
#endif
  );
  blk.bn1 = load_bn(*f_, p + "bn1");
  for (int j = 0; j < blk.scale; ++j) {
    blk.convs_w.push_back(
        load_conv_w(*f_, p + "convs." + std::to_string(j) + ".weight"));
    blk.bns.push_back(load_bn(*f_, p + "bns." + std::to_string(j)));
  }
  blk.conv3_w = load_conv_w(*f_, p + "conv3.weight", nullptr
#if defined(GSV_AMX_GEMM)
      , &blk.conv3_panel, size_t(blk.exp_planes), size_t(co1_local)
#endif
  );
  blk.bn3 = load_bn(*f_, p + "bn3");
  if (blk.aff) {
    for (int j = 0; j < blk.scale - 1; ++j) {
      Aff a;
      const std::string fp = p + "fuse_models." + std::to_string(j) + ".local_att.";
      a.inter = static_cast<int>(
          f_->tensor((fp + "0.weight").c_str())->dims[0]);  // [inter, 2*width,1,1]
      a.ch = blk.width;
      a.w1 = load_conv_w(*f_, fp + "0.weight");
      a.w2 = load_conv_w(*f_, fp + "3.weight", &a.w2_own);  // 可能无 f16 段→量化副本
      a.b1 = vec_any(*f_, fp + "0.bias");   // Conv2d 默认 bias=True!
      a.b2 = vec_any(*f_, fp + "3.bias");
      a.bn1 = load_bn(*f_, fp + "1");
      a.bn2 = load_bn(*f_, fp + "4");
      blk.fuses.push_back(std::move(a));
    }
  }
  blk.has_shortcut = f_->tensor((p + "shortcut.0.weight").c_str()) != nullptr;
  if (blk.has_shortcut) {
    const int cin_sc = static_cast<int>(
        f_->tensor((p + "shortcut.0.weight").c_str())->dims[1]);
    blk.sc_w = load_conv_w(*f_, p + "shortcut.0.weight", nullptr
#if defined(GSV_AMX_GEMM)
        , &blk.sc_panel, size_t(blk.exp_planes), size_t(cin_sc)
#endif
    );
    blk.sc_bn = load_bn(*f_, p + "shortcut.1");
  }

#if defined(GSV_AMX_GEMM)
  // E12: 装载期预打包 AMX 权重 panel。形状分流见 amx_sv_worth_packing(T3 收紧)。
  if (amx_sv_enabled() && kern::amx_gemm_available()) {
    auto try_pack = [&](kern::AmxPanel& panel, const uint16_t* w, size_t rows,
                        size_t K, bool& ready) {
      if (amx_sv_worth_packing(rows, K) && w) {
        panel.rows = rows;
        panel.K = K;
        kern::amx_pack_into(w, rows, K, panel.buf);
        ready = true;
      }
    };
    (void)co1_local;
    try_pack(blk.conv1_panel, blk.conv1_w, size_t(co1_local), size_t(cin_conv1),
             blk.conv1_panel_ready);
    try_pack(blk.conv3_panel, blk.conv3_w, size_t(blk.exp_planes),
             size_t(co1_local), blk.conv3_panel_ready);
    for (int j = 0; j < blk.scale; ++j) {
      kern::AmxPanel p3;
      bool ready = false;
      // 3x3 convs 形状 [width, width*9]
      try_pack(p3, blk.convs_w[size_t(j)], size_t(blk.width),
               size_t(blk.width) * 9, ready);
      blk.convs_panels.push_back(std::move(p3));
      blk.convs_panels_ready.push_back(ready);
    }
    if (blk.has_shortcut) {
      const int cin_sc = static_cast<int>(f_->tensor((p + "shortcut.0.weight").c_str())->dims[1]);
      try_pack(blk.sc_panel, blk.sc_w, size_t(blk.exp_planes), size_t(cin_sc),
               blk.sc_panel_ready);
    }
    // AFF fuse 层
    for (auto& aff : blk.fuses) {
      if (!aff.w1) continue;
      try_pack(aff.w1_panel, aff.w1, size_t(aff.inter), size_t(2 * aff.ch),
               aff.w1_panel_ready);
      try_pack(aff.w2_panel, aff.w2, size_t(aff.ch), size_t(aff.inter),
               aff.w2_panel_ready);
    }
  }
#endif
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
  fuse34_.w2 = load_conv_w(f, "fuse34.local_att.3.weight", &fuse34_.w2_own);
  fuse34_.b1 = vec_any(f, "fuse34.local_att.0.bias");
  fuse34_.b2 = vec_any(f, "fuse34.local_att.3.bias");
  fuse34_.bn1 = load_bn(f, "fuse34.local_att.1");
  fuse34_.bn2 = load_bn(f, "fuse34.local_att.4");
#if defined(GSV_AMX_GEMM)
  // E12: l3ds(3x3, [2048,1024*9])与 fuse34 w1/w2 装载期预打包
  if (amx_sv_enabled() && kern::amx_gemm_available()) {
    auto try_pack = [&](kern::AmxPanel& panel, const uint16_t* w, size_t rows,
                        size_t K, bool& ready) {
      if (amx_sv_worth_packing(rows, K) && w) {
        panel.rows = rows;
        panel.K = K;
        kern::amx_pack_into(w, rows, K, panel.buf);
        ready = true;
      }
    };
    try_pack(l3ds_panel_, l3ds_w_, size_t(c_after_[4]),
             size_t(c_after_[3]) * 9, l3ds_panel_ready_);
    try_pack(fuse34_.w1_panel, fuse34_.w1, size_t(fuse34_.inter),
             size_t(2 * fuse34_.ch), fuse34_.w1_panel_ready);
    try_pack(fuse34_.w2_panel, fuse34_.w2, size_t(fuse34_.ch),
             size_t(fuse34_.inter), fuse34_.w2_panel_ready);
  }
#endif
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
  const auto tp_fwd0 = sv_inner_tim() ? std::chrono::steady_clock::now()
                                      : std::chrono::steady_clock::time_point{};
  conv2d(cur_.data(), 1, F, T, conv1_w_.data(), 64, 3, 3, 1, 1, cols_, tmp_);
  const int m_ch = static_cast<int>(bn1_.g.size());
  bn1_.apply(tmp_.data(), m_ch, static_cast<size_t>(F) * T);
  // 注意: 主干入口是 F.relu(torch.nn.functional), 不是块内的 Hardtanh(0,20)
  for (float& v : tmp_) v = v < 0.f ? 0.f : v;
  o_conv1_ = tmp_;
  const auto tp_stem = sv_inner_tim() ? std::chrono::steady_clock::now()
                                      : std::chrono::steady_clock::time_point{};
  std::chrono::steady_clock::time_point tp_stage_last = tp_stem;

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
    if (sv_inner_tim())
      g_sv_inner.stage_ms[static_cast<size_t>(l)] +=
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() -
              (l == 1 ? tp_stem : tp_stage_last)).count();
    tp_stage_last = sv_inner_tim() ? std::chrono::steady_clock::now()
                                   : std::chrono::steady_clock::time_point{};
  }

  // layer3_ds(1024→2048, k3, s2, p1) 作用在 layer3 输出上(fp16 直读)
#if defined(GSV_AMX_GEMM)
  if (l3ds_panel_ready_ && !(amx_sv_skip_mask() & 32u) &&
      static_cast<size_t>(h3) * static_cast<size_t>(w3) >= kAmxSvMinRows &&
      static_cast<size_t>(c_after_[3]) * 9 >= kAmxSvMinK) {
    conv2d_amx(l3ds_panel_, o_layer_[3].data(), c_after_[3], h3, w3, 3, 3, 2,
               1, c_after_[4], ds_, "l3ds");
  } else
#endif
  conv2d_f16(o_layer_[3].data(), c_after_[3], h3, w3, l3ds_w_, c_after_[4], 3, 3, 2,
         1, cols_, ds_);

  // fuse34(AFF) on (layer4 输出, ds)
  o_fuse_.resize(ds_.size());
  fuse34_.apply(tmp_.data(), ds_.data(), o_fuse_.data(), h, w, xh_, xh2_
#if defined(GSV_AMX_GEMM)
                , sv_act_scratch_
#endif
  );

  // flatten(C×F) → mean over T
  const int C = c_after_[4];  // 2048
  {
    const auto tp_pool = sv_tp();
    emb_.assign(static_cast<size_t>(C) * h, 0.f);
    for (int ci = 0; ci < C; ++ci)
      for (int fy = 0; fy < h; ++fy) {
        const float* row = o_fuse_.data() + (static_cast<size_t>(ci) * h + fy) * w;
        double acc = 0.0;
        for (int t = 0; t < w; ++t) acc += row[t];
        emb_[static_cast<size_t>(ci) * h + fy] = static_cast<float>(acc / w);
      }
    sv_acc(tp_pool, g_sv_inner.meta_pool_ms);
  }
  // E13 探针: 输出本轮 conv2d/conv2d_f16 内核累计耗时(仅 stderr, 不改行为)
  if (sv_inner_tim()) {
    std::fprintf(stderr,
                 "[sv-inner] conv2d_sgemm_total=%.1fms conv2d_f16_fmlal_total=%.1fms "
                 "aff_w1=%.1fms(%d次) aff_w2=%.1fms blk_conv1=%.1fms(S=%ld) T=%d\n",
                 g_sv_inner.conv2d_ms, g_sv_inner.conv2d_f16_ms,
                 g_sv_inner.aff1_ms, g_sv_inner.aff_n, g_sv_inner.aff2_ms,
                 g_sv_inner.blkconv1_ms, g_sv_inner.blkconv1_s, T);
    std::fprintf(stderr,
                 "[sv-meta] bn=%.1f relu=%.1f add=%.1f copy=%.1f pool=%.1f "
                 "im2col=%.1f pack=%.1f affgate=%.1f\n",
                 g_sv_inner.meta_bn_ms, g_sv_inner.meta_relu_ms,
                 g_sv_inner.meta_add_ms, g_sv_inner.meta_copy_ms,
                 g_sv_inner.meta_pool_ms, g_sv_inner.meta_im2col_ms,
                 g_sv_inner.meta_pack_ms, g_sv_inner.meta_aff_ms);
    std::vector<std::pair<std::string, std::pair<int, double>>> ranked(
        g_sv_inner.f16_by_shape.begin(), g_sv_inner.f16_by_shape.end());
    std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
      return a.second.second > b.second.second;
    });
    double cum = 0.0;
    for (int i = 0; i < int(ranked.size()) && i < 14; ++i) {
      cum += ranked[size_t(i)].second.second;
      std::fprintf(stderr, "[sv-f16rank] #%d x%d %8.1fms(cum%7.1f) %s\n", i,
                   ranked[size_t(i)].second.first, ranked[size_t(i)].second.second,
                   cum, ranked[size_t(i)].first.c_str());
    }
    std::vector<std::pair<std::string, std::pair<int, double>>> srank(
        g_sv_inner.site.begin(), g_sv_inner.site.end());
    std::sort(srank.begin(), srank.end(), [](const auto& a, const auto& b) {
      return a.second.second > b.second.second;
    });
    double scum = 0.0;
    for (int i = 0; i < int(srank.size()); ++i) {
      scum += srank[size_t(i)].second.second;
      std::fprintf(stderr, "[sv-siterank] #%d x%d %8.1fms(cum%7.1f) %s\n", i,
                   srank[size_t(i)].second.first, srank[size_t(i)].second.second,
                   scum, srank[size_t(i)].first.c_str());
    }
    const auto clk_end = std::chrono::steady_clock::now();
    std::fprintf(stderr,
                 "[sv-seg] stem=%.1f stage1=%.1f stage2=%.1f stage3=%.1f "
                 "stage4=%.1f l3ds_fuse34_emb=%.1fms\n",
                 std::chrono::duration<double, std::milli>(tp_stem - tp_fwd0).count(),
                 g_sv_inner.stage_ms[1], g_sv_inner.stage_ms[2],
                 g_sv_inner.stage_ms[3], g_sv_inner.stage_ms[4],
                 std::chrono::duration<double, std::milli>(clk_end - tp_stage_last).count());
  }
  return emb_.size();
}

}  // namespace gsv::encoder
