// test_ar_unit.cpp — AR 引擎单测(B12; 不含 golden 依赖的部分)
//
// 覆盖:
//   1) 正弦位置编码行 vs long double 参考式
//   2) repetition penalty 序列语义(重复 token 累次施压 / 正负号方向)
//   3) EOS 掩蔽(idx<11 剔除 EOS 列)
//   4) prefill(纯因果) 与逐 token decode 的 KV cache 一致性 —— 用真实权重跑
//      文本长度=0 的纯因果 prefill, 末行输出 vs 逐步 decode 输出对拍
//   5) generate 冒烟: 真 pair 最小输入跑通并 EOS 终止
#include "ar/t2s_engine.hpp"

#include "test_util.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#ifndef GSV_WEIGHTS_DIR
#define GSV_WEIGHTS_DIR "weights"
#endif
#ifndef GSV_AR_RAW_DIR
#define GSV_AR_RAW_DIR ""
#endif

using gsv::ar::GenResult;
using gsv::ar::T2SDims;
using gsv::ar::T2SEngine;

static const char* kWeights = GSV_WEIGHTS_DIR "/ar_s1v3.gsv";

GSV_TEST(pe_row_matches_reference) {
  gsv::rt::GsvFile f(kWeights);
  T2SEngine eng(f);
  const size_t D = eng.dims().d_model;
  std::vector<float> pe(D);
  // 参考式用 double 计算; 容差 1e-4 —— 本引擎与 torch 同为 fp32 逐算子计算,
  // 大位置角(如 pos=1234·div)的 fp32 舍入在 ~3e-5 量级, 不应用 long double 过严约束。
  const double log_inc = -std::log(10000.0) / static_cast<double>(D);
  for (size_t pos : {size_t{0}, size_t{1}, size_t{7}, size_t{166}, size_t{1234}}) {
    eng.pe_row(pe.data(), pos);
    for (size_t d = 0; d < D; d += 2) {
      const double div = std::exp(static_cast<double>(d) * log_inc);
      const double ang = static_cast<double>(pos) * div;
      CHECK_NEAR(pe[d], std::sin(ang), 1e-4);
      CHECK_NEAR(pe[d + 1], std::cos(ang), 1e-4);
    }
  }
}

GSV_TEST(repetition_penalty_semantics) {
  gsv::rt::GsvFile f(kWeights);
  T2SEngine eng(f);
  // logits: idx0=+2.0, idx1=-3.0, idx2=0.5, 其余 0
  std::vector<float> lg(eng.dims().vocab, 0.f);
  lg[0] = 2.0f;
  lg[1] = -3.0f;
  lg[2] = 0.5f;
  // 历史 {0,0}: 正向压低 2/1.35/1.35≈1.098 > 0.5 → argmax 仍为 0
  std::vector<int32_t> hist = {0, 0};
  int raw = -1;
  CHECK(eng.greedy_sample(lg.data(), hist, true, &raw) == 0);
  CHECK(raw == 0);
  // 历史 {2}×6: 0.5/1.35^6≈0.111 → argmax 仍为 0
  hist.assign(6, 2);
  CHECK(eng.greedy_sample(lg.data(), hist, true, &raw) == 0);
  // 负向放大: idx1 已出现 3 次(-3*1.35³≈-7.4), idx2 未出现(-2.9) → 取 2
  {
    std::vector<float> lg2(eng.dims().vocab, -100.f);
    lg2[1] = -3.0f;
    lg2[2] = -2.9f;
    hist.assign(3, 1);
    CHECK(eng.greedy_sample(lg2.data(), hist, true, &raw) == 2);
  }
}

GSV_TEST(eos_mask_first_steps) {
  gsv::rt::GsvFile f(kWeights);
  T2SEngine eng(f);
  const int eos = static_cast<int>(eng.dims().eos);
  std::vector<float> lg(eng.dims().vocab, -1.f);
  lg[eos] = 10.f;  // 未惩罚 argmax 必为 EOS
  lg[5] = 3.f;
  int raw = -1;
  std::vector<int32_t> hist;
  // eos_allowed=false(idx<11): raw 记 EOS(golden tokens 口径), 实际采样取 5
  const int s = eng.greedy_sample(lg.data(), hist, false, &raw);
  CHECK(raw == eos);
  CHECK(s == 5);
  // eos_allowed=true: 直接采到 EOS
  CHECK(eng.greedy_sample(lg.data(), hist, true, &raw) == eos);
  CHECK(raw == eos);
}

// prefill(text_len=0 ⇒ 纯因果) 与逐 token decode 对拍:
// 同一输入先整段过 block_prefill, 再逐 token 过 block_decode, 末位置隐藏态应一致。
// 输入取正弦 PE 行本身 —— 非平凡、可复现、不依赖未公开的内部查表。
GSV_TEST(prefill_causal_matches_stepwise_decode) {
  gsv::rt::GsvFile f(kWeights);
  T2SEngine eng(f);
  const T2SDims& dm = eng.dims();
  const size_t D = dm.d_model, NL = dm.n_layers;
  const size_t N = 24;

  const auto cap = N + 8;
  std::vector<std::vector<float>> ka(NL), va(NL), kb(NL), vb(NL);
  for (size_t l = 0; l < NL; ++l) {
    ka[l].assign(cap * D, 0.f);
    va[l].assign(cap * D, 0.f);
    kb[l].assign(cap * D, 0.f);
    vb[l].assign(cap * D, 0.f);
  }
  std::vector<float> xa(N * D), pe(D);
  for (size_t t = 0; t < N; ++t) {
    eng.pe_row(pe.data(), t);
    for (size_t d = 0; d < D; ++d) xa[t * D + d] = pe[d];
  }
  std::vector<float> refA = xa;
  for (size_t l = 0; l < NL; ++l)
    eng.block_prefill(l, refA.data(), N, /*pos=*/0, /*text_len=*/0,
                      ka[l].data(), va[l].data());

  // 路径 B: 逐 token decode
  std::vector<float> last(D);
  for (size_t t = 0; t < N; ++t) {
    std::vector<float> xt(D);
    eng.pe_row(pe.data(), t);
    for (size_t d = 0; d < D; ++d) xt[d] = pe[d];
    for (size_t l = 0; l < NL; ++l)
      eng.block_decode(l, xt.data(), t, t + 1, kb[l].data(), vb[l].data());
    if (t + 1 == N) last = xt;
  }
  double max_abs = 0;
  for (size_t d = 0; d < D; ++d)
    max_abs = std::max(max_abs,
                       std::fabs(static_cast<double>(last[d]) - refA[(N - 1) * D + d]));
  CHECK_NEAR(max_abs, 0.0, 5e-4);  // GEMM/GEMV 归约序差, 远小于 G1 门限

  // KV cache 内容一致性抽查
  double kv_diff = 0;
  for (size_t l = 0; l < NL; l += 7)
    for (size_t i = 0; i < N * D; i += 97)
      kv_diff = std::max(kv_diff,
                         std::fabs(static_cast<double>(ka[l][i]) - kb[l][i]));
  CHECK_NEAR(kv_diff, 0.0, 5e-4);
}

GSV_TEST(generate_smoke_eos_stop) {
  // 用真实 pair 输入跑冒烟: EOS 终止且步数合理(数据缺失则跳过)
  static const char* kRawDir = GSV_AR_RAW_DIR;
  if (!kRawDir || !*kRawDir) return;
  const std::string pdir = std::string(kRawDir) + "/vo_BZLQ001_4_hutao_02__s0";
  FILE* mf = std::fopen((pdir + "/meta.txt").c_str(), "r");
  if (!mf) return;  // 数据未导出时跳过(单测不硬依赖本地数据)
  long T = 0, P = 0, gs = 0;
  if (std::fscanf(mf, "%ld %ld %ld", &T, &P, &gs) != 3) {
    std::fclose(mf);
    return;
  }
  std::fclose(mf);

  auto rd_i64 = [&](const char* nm, size_t n) {
    std::vector<int64_t> v(n);
    FILE* fp = std::fopen((pdir + "/" + nm).c_str(), "rb");
    if (fp) {
      if (std::fread(v.data(), 8, n, fp) != n) v.clear();
      std::fclose(fp);
    }
    return v;
  };
  auto rd_f32 = [&](const char* nm, size_t n) {
    std::vector<float> v(n);
    FILE* fp = std::fopen((pdir + "/" + nm).c_str(), "rb");
    if (fp) {
      if (std::fread(v.data(), 4, n, fp) != n) v.clear();
      std::fclose(fp);
    }
    return v;
  };
  const auto phones = rd_i64("phones.i64", static_cast<size_t>(T));
  const auto prompt = rd_i64("prompt.i64", static_cast<size_t>(P));
  const auto bert = rd_f32("bert.f32", static_cast<size_t>(T) * 1024);

  gsv::rt::GsvFile f(kWeights);
  T2SEngine eng(f);
  GenResult r = eng.generate(phones.data(), phones.size(), prompt.data(),
                             prompt.size(), bert.data());
  CHECK(r.hit_eos);
  CHECK(r.steps >= 11);  // 前 11 步禁止 EOS ⇒ 至少 11 步
  CHECK(r.sampled.size() + 1 >= r.steps);
  CHECK(r.logits_last.size() == eng.dims().vocab);
}

GSV_TEST_MAIN()
