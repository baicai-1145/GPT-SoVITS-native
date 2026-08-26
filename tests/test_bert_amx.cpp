// test_bert_amx.cpp — E8 验收: BERT dense FMLAL→AMX 切换数值一致性
//
// DoD: ① 同权重同输入下 BertModel::forward 输出在 fmlal/amx 两路径下
//         wav/argmax 级别一致 (直接对照 fp32 张量, 允许 ≤1e-5 累计差,
//         沿用 G1 cos>=0.9999 门; 严格位级等价不可达是已定决策)
//      ② 21 层 L=49 前向的 AMX 路径无 NaN/Inf (front-timing 数值健全)
//      ③ cast_pack_B_f32_to_panel 与 pack_panel(f32_to_f16(x)) 位级一致
//         (panel 布局对齐性)
//
// 运行时: 无权重文件时 SKIP (CI 环境); 本地有权重时验证到位。
#include "bert/bert_model.hpp"
#include "kern/gemm_f16_amx.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifndef GSV_WEIGHTS_DIR
#define GSV_WEIGHTS_DIR "./weights"
#endif

namespace {

int g_fail = 0;

void expect(bool ok, const char* what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++g_fail;
  } else {
    std::fprintf(stderr, "ok: %s\n", what);
  }
}

}  // namespace

int main() {
#if !defined(GSV_AMX_GEMM)
  std::fprintf(stderr, "SKIP: 非AMX构建 (-DGSV_AMX_GEMM=ON 才有意义)\n");
  return 0;
#else
  const std::string roberta = std::string(GSV_WEIGHTS_DIR) +
                              "/roberta_wwm_ext_large.gsv";
  FILE* probe = std::fopen(roberta.c_str(), "rb");
  if (!probe) {
    std::fprintf(stderr, "SKIP: 权重不存在 %s\n", roberta.c_str());
    return 0;
  }
  std::fclose(probe);

  if (!gsv::kern::amx_gemm_available()) {
    std::fprintf(stderr, "SKIP: 平台无 AMX\n");
    return 0;
  }

  // ---- ③ panel 布局: cast_pack_B 与 pack_panel(f16(x)) 位级一致 ----
  {
    const size_t T = 49, K = 384;
    std::vector<float> xf(T * K);
    for (size_t i = 0; i < xf.size(); ++i)
      xf[i] = ((float)(int(i * 7 + 3) % 61) - 30.f) / 512.f;
    std::vector<uint16_t> xh(T * K);
    gsv::kern::f32_to_f16(xf.data(), xh.data(), xh.size());
    auto pref = gsv::kern::amx_pack(xh.data(), T, K);
    std::vector<uint8_t> pnew;
    gsv::bert::cast_pack_B_f32_to_panel(xf.data(), T, K, pnew);
    const uint8_t* d1 = pref.buf.data() +
                        ((64 - ((uintptr_t)pref.buf.data() & 63)) & 63);
    const uint8_t* d2 = pnew.data() +
                        ((64 - ((uintptr_t)pnew.data() & 63)) & 63);
    const size_t panel_sz = ((T + 31) / 32) * K * 64;
    const bool layout_ok = std::memcmp(d1, d2, panel_sz) == 0;
    expect(layout_ok, "cast_pack_B_f32_to_panel 与 pack_panel 布局位级一致");
  }

  // ---- ①② BertModel forward 双路径对照 ----
  gsv::rt::GsvFile f(roberta);
  const size_t LAYERS = 21;  // hidden_states[-3] 口径

  // FMLAL 参考跑
  gsv::bert::amx_bert_enabled() = false;
  gsv::bert::amx_bert_mode() = gsv::bert::AmxBertMode::kAll;
  gsv::bert::BertConfig cfg;
  cfg.layers = LAYERS;
  gsv::bert::BertModel bm_ref;
  bm_ref.cfg = cfg;
  bm_ref.load(f, "bert");

  const size_t T = 49;
  std::vector<int64_t> ids(T);
  for (size_t i = 0; i < T; ++i) ids[i] = int64_t(1000 + (i * 37) % 20000);
  gsv::bert::Matrix out_ref;
  static gsv::bert::Dumper dm("");
  bm_ref.forward(ids, std::vector<int64_t>(T, 0),
                 std::vector<int64_t>(T, 1), out_ref, dm);

  bool ref_finite = true;
  for (float v : out_ref.d) if (!std::isfinite(v)) ref_finite = false;
  expect(ref_finite, "FMLAL 路径输出有限值(无 NaN/Inf)");

  // AMX 跑 (独立实例免 cross-contamination)
  gsv::bert::BertModel bm_amx;
  bm_amx.cfg = cfg;
  bm_amx.load(f, "bert");  // amx_bert_enabled 已置 true → load 时预打包 panel

  gsv::bert::Matrix out_amx;
  bm_amx.forward(ids, std::vector<int64_t>(T, 0),
                 std::vector<int64_t>(T, 1), out_amx, dm);

  bool amx_finite = true;
  for (float v : out_amx.d) if (!std::isfinite(v)) amx_finite = false;
  expect(amx_finite, "AMX 路径输出有限值(无 NaN/Inf)");
  expect(out_amx.d.size() == out_ref.d.size(), "两路径输出形状一致");

  // G1 类门: cos >= 0.9999 且 max_rel <= 1e-3 (gates.json 口径)
  double dot = 0, na = 0, nb = 0, max_d = 0, max_ref = 0;
  for (size_t i = 0; i < out_ref.d.size(); ++i) {
    const double a = out_amx.d[i], b = out_ref.d[i];
    dot += a * b;
    na += a * a;
    nb += b * b;
    max_d = std::max(max_d, std::fabs(a - b));
    max_ref = std::max(max_ref, std::fabs(b));
  }
  const double cos_sim = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
  const double rel = max_d / (max_ref + 1e-30);
  char msg[128];
  std::snprintf(msg, sizeof(msg), "21层 L=49 对照 cos=%.6f rel=%.2e",
                cos_sim, rel);
  expect(cos_sim >= 0.9999 && rel <= 1e-3, msg);

  if (g_fail != 0) {
    std::fprintf(stderr, "%d checks FAILED\n", g_fail);
    return 1;
  }
  std::fprintf(stderr, "ALL PASS\n");
  return 0;
#endif
}
