// sovits_engine.hpp — SoVITS 全链路引擎 (B3+B4)
// 链路 (SynthesizerTrn.decode 路径):
//   codes → quantizer.decode → nearest×2 上采样
//   → enc_p(ssl_proj/Enc3/text_emb/Enc6/MRTE(ge_text)/Enc3/proj) → m_p,logs_p
//   → z_p = m_p + noise·exp(logs_p)·noise_scale   [noise 由外部输入, 对齐 golden]
//   → flow.reverse(z_p, ge[1024]) → z
//   → dec(z, ge) → wav float[-1,1] → ×32768 int16 刻度
#pragma once

#include <cmath>
#include <string>
#include <vector>

#include "runtime/gsv_loader.hpp"
#include "sovits/dec.hpp"
#include "sovits/enc_p.hpp"
#include "sovits/flow.hpp"
#include "sovits/quantizer.hpp"
#include "sovits/wav_writer.hpp"

namespace gsv::sovits {

class SovitsEngine {
 public:
  // use_amx: E5-P2 — 装载时预打包 AMX 权重 panel (--amx 开关; 默认关)。
  // 运行时无 AMX 硬件时自动回退 sgemm (见 conv1d.hpp 分流逻辑)。
  void load(const std::string& gsv_path, bool use_amx = false) {
    sovits::amx_enabled() = use_amx;
    rt::GsvFile f(gsv_path);
    quant_.load(f, /*codebook*/ 1024, /*dim*/ 768);
    enc_p_.load(f);
    flow_.load(f);
    dec_.load(f);
  }

  // 完整 decode; noise 可为空指针 (此时内部不注入噪声 — z_p=m_p)
  struct Inputs {
    const int64_t* codes = nullptr;
    size_t n_codes = 0;
    const int64_t* phones = nullptr;
    size_t n_phones = 0;
    const float* ge = nullptr;       // [1024] (时间维 1)
    const float* ge_text = nullptr;  // [512]
    const float* noise = nullptr;    // [192 * Tq] 行主 [C,Tq]; 可空
    size_t Tq_expected = 0;          // 校验用 (= n_codes*2)
  };

  void run(const Inputs& in, Tensor2D& wav_out, const Dumper& dm) {
    const size_t Tc = in.n_codes;
    const size_t Tq = Tc * 2;

    // 1. RVQ decode → [768, Tc]
    Tensor2D quantized;
    quant_.decode(in.codes, Tc, quantized);
    dm.dump("h_quantizer_dec", quantized);

    // 2. nearest ×2 上采样 → [768, Tq]
    Tensor2D quant_up(768, Tq);
    for (size_t c = 0; c < 768; ++c)
      for (size_t t = 0; t < Tc; ++t) {
        const float v = quantized.d[c * Tc + t];
        quant_up.d[c * Tq + 2 * t] = v;
        quant_up.d[c * Tq + 2 * t + 1] = v;
      }
    dm.dump("h_encp_input", quant_up);

    // 3. enc_p
    Tensor2D ge_text(512, 1);
    for (size_t i = 0; i < 512; ++i) ge_text.d[i] = in.ge_text[i];
    Tensor2D m_p, logs_p;
    enc_p_.forward(quant_up, in.phones, in.n_phones, ge_text, m_p, logs_p, dm);
    dm.dump("dbg_m_p", m_p);
    dm.dump("dbg_logs_p", logs_p);

    // 4. z_p = m_p + noise · exp(logs_p) · 0.5
    Tensor2D z_p(192, Tq);
    for (size_t i = 0; i < 192 * Tq; ++i)
      z_p.d[i] = m_p.d[i] + (in.noise ? in.noise[i] : 0.f) *
                                std::exp(logs_p.d[i]) * kNoiseScale;
    dm.dump("h_flow_in", z_p);

    // 5. flow reverse (条件 ge 1024)
    Tensor2D ge(1024, 1);
    for (size_t i = 0; i < 1024; ++i) ge.d[i] = in.ge[i];
    Tensor2D mask(1, Tq);
    for (size_t t = 0; t < Tq; ++t) mask.d[t] = 1.f;
    flow_.forward_reverse(z_p, mask, ge, dm);
    dm.dump("h_flow", z_p);

    // 6. dec
    dec_.forward(z_p, ge, wav_out, dm);
    dm.dump("h_dec", wav_out);
  }

  static constexpr float kNoiseScale = 0.5f;

 private:
  Quantizer quant_;
  TextEncoder enc_p_;
  Flow flow_;
  Generator dec_;
};

}  // namespace gsv::sovits
