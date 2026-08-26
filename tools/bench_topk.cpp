// bench_topk.cpp — E4: topk_sample 速度/数值基线
// 单头 1025 词表, k=15 采样, 验证与 greedy argmax 一致性 + 速度
#include "ar/t2s_engine.hpp"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

int main(int argc, char** argv) {
  const char* weights = (argc > 1) ? argv[1] : nullptr;
  if (!weights) { std::fprintf(stderr, "usage: %s weights.gsv\n", argv[0]); return 1; }
  gsv::rt::GsvFile wf(weights);
  gsv::ar::T2SEngine eng(wf);
  const int V = static_cast<int>(eng.dims().vocab);
  std::printf("V=%d  kRepPenalty=%.3f  kEosMaskSteps=%zu\n", V,
              gsv::ar::T2SEngine::kRepPenalty, gsv::ar::T2SEngine::kEosMaskSteps);

  // 构造一组随机 logits, history 50 token
  std::vector<float> logits(V);
  std::mt19937_64 r(42);
  std::normal_distribution<float> d(0, 1);
  for (auto& v : logits) v = d(r) * 0.5f;
  std::vector<int32_t> history;
  for (int i = 0; i < 50; ++i) history.push_back(r() % V);

  // ---- 1) 一致性: 5 次 sample 独立 RNG 跑 100 步, 看每步的 raw argmax vs 选中 ----
  // initialize scratch (normally done in generate())
  eng.init_topk_scratch();
  gsv::ar::SamplingParams sp;
  sp.mode = gsv::ar::SamplingParams::Mode::TopK;
  sp.top_k = 15; sp.top_p = 1.0f; sp.temperature = 1.0f; sp.rep_penalty = 1.35f; sp.seed = 42;
  std::mt19937_64 rng(sp.seed);
  int sel_top1 = 0, raw_top1 = 0, agree = 0, disagree = 0;
  for (int step = 0; step < 100; ++step) {
    int raw = -1;
    int sel = eng.topk_sample(logits.data(), history, true, &raw, sp, rng);
    if (step == 0) { sel_top1 = sel; raw_top1 = raw; }
    if (sel == raw) ++agree; else ++disagree;
    // 把选中 push 进 history, 重新生成 logits 模拟下一步
    history.push_back(sel);
    for (auto& v : logits) v = d(rng) * 0.5f;
  }
  std::printf("100 步 topk 采样: agree_with_argmax=%d disagree=%d\n", agree, disagree);
  std::printf("首步: raw_argmax=%d  selected=%d  (无 argmax 一致时随机性正常)\n", raw_top1, sel_top1);

  // ---- 2) 速度: 1万次 topk_sample 调用 ----
  // 重建 state
  for (auto& v : logits) v = d(rng) * 0.5f;
  history.resize(50);
  rng.seed(sp.seed);
  const int N = 10000;
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < N; ++i) {
    int raw = -1;
    eng.topk_sample(logits.data(), history, true, &raw, sp, rng);
    // 不 push history, 固定 state
  }
  auto t1 = std::chrono::steady_clock::now();
  double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N;
  std::printf("topk_sample: %.2f us/call  (1万次平均)\n", us);

  // ---- 3) 确定性: 同 seed 两次跑, 选中序列应一致 ----
  rng.seed(sp.seed);
  std::vector<int> seq1, seq2;
  for (int i = 0; i < 50; ++i) {
    int raw = -1;
    seq1.push_back(eng.topk_sample(logits.data(), history, true, &raw, sp, rng));
  }
  rng.seed(sp.seed);
  for (int i = 0; i < 50; ++i) {
    int raw = -1;
    seq2.push_back(eng.topk_sample(logits.data(), history, true, &raw, sp, rng));
  }
  int same = 0;
  for (size_t i = 0; i < seq1.size(); ++i) if (seq1[i] == seq2[i]) ++same;
  std::printf("确定性: 同 seed 50 步采样 = %d/50 一致\n", same);

  // ---- 4) temperature=0 退贪心 ----
  gsv::ar::SamplingParams sp_g;
  sp_g.mode = gsv::ar::SamplingParams::Mode::TopK;
  sp_g.top_k = 15; sp_g.temperature = 0.0f; sp_g.seed = 42;
  rng.seed(42);
  int sel_g = eng.topk_sample(logits.data(), history, true, nullptr, sp_g, rng);
  std::printf("temperature=0 首步选中=%d (应为 raw_argmax)\n", sel_g);

  return 0;
}
