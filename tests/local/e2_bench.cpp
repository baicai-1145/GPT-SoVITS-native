// e2_bench.cpp — E2-ENC 验收: 编码器耗时/内存对比 harness。
// 模式: <fp32|fp16> —— 分支决定路径; 输出加载后 RSS/引擎前向耗时(多次取中位)。
#include <mach/mach.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "bert/bert_model.hpp"
#include "encoder/hubert.hpp"
#include "encoder/sv.hpp"
#include "encoder/fbank.hpp"
#include "runtime/gsv_loader.hpp"
#include "runtime/wav.hpp"

using namespace gsv;

static size_t rss_kb() {
  task_vm_info v;
  mach_msg_type_number_t c = TASK_VM_INFO_COUNT;
  if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&v, &c) != KERN_SUCCESS)
    return 0;
  return v.phys_footprint / 1024;
}

static double ms_of(std::vector<double>& v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: %s <weightsDir> <refWav16k> <mode(fp32|fp16)>\n",
                 argv[0]);
    return 2;
  }
  const std::string wdir = argv[1], wav = argv[2], mode = argv[3];

  // ---- RoBERTa ----
  double bert_ms;
  size_t rss_after_bert;
  {
    rt::GsvFile f(wdir + "/roberta_wwm_ext_large.gsv");
    const size_t t0 = rss_kb();
    bert::BertModel bm;
    bm.cfg = bert::BertConfig{};
    bm.load(f, "bert");
    rss_after_bert = rss_kb();
    const size_t T = 32, C = 1024;
    bert::Matrix x;
    x.reset(T, C);
    for (size_t i = 0; i < x.d.size(); ++i)
      x.d[i] = float((int(i) % 13) - 6) * 0.05f;
    std::vector<float> ext(T, 0.f);
    bert::Matrix y, scr, ctxh;
    std::vector<uint16_t> xh;
    std::vector<double> ts;
    for (int r = 0; r < 7; ++r) {
      auto a = std::chrono::high_resolution_clock::now();
      for (size_t L = 0; L < bm.cfg.layers; ++L) {
        bm.stack[L].forward(x, ext, y, scr, ctxh, xh);
        x.d.swap(y.d);
      }
      ts.push_back(std::chrono::duration<double, std::milli>(
                       std::chrono::high_resolution_clock::now() - a)
                       .count());
    }
    bert_ms = ms_of(ts);
    std::printf("roberta: rss_after_load=%zuKB(+%zuKB) fwd24L[T=32]=%.1fms (%s)\n",
                rss_after_bert, rss_after_bert - t0, bert_ms, mode.c_str());
  }

  // ---- HuBERT ----
  double hub_ms;
  size_t rss_after_hub;
  {
    rt::GsvFile f(wdir + "/hubert_base.gsv");
    encoder::HubertEngine hub(f);
    rss_after_hub = rss_kb();
    rt::wav::WavFile w = rt::wav::load_wav(wav);
    std::vector<float> wav16k = w.samples;
    wav16k.insert(wav16k.end(), 9600, 0.f);
    std::vector<double> ts;
    for (int r = 0; r < 5; ++r) {
      auto a = std::chrono::high_resolution_clock::now();
      hub.run(wav16k.data(), wav16k.size());
      ts.push_back(std::chrono::duration<double, std::milli>(
                       std::chrono::high_resolution_clock::now() - a)
                       .count());
    }
    hub_ms = ms_of(ts);
    std::printf("hubert: rss_after_load=%zuKB fwd=%.1fms (%s)\n", rss_after_hub,
                hub_ms, mode.c_str());
  }

  // ---- SV ----
  double sv_ms;
  {
    rt::GsvFile f(wdir + "/eres2netv2_sv.gsv");
    encoder::SvEngine sv(f);
    rt::wav::WavFile w = rt::wav::load_wav(wav);
    size_t frames = 0;
    const std::vector<float> fb =
        encoder::kaldi_fbank_80(w.samples.data(), w.samples.size(), &frames);
    std::vector<double> ts;
    for (int r = 0; r < 5; ++r) {
      auto a = std::chrono::high_resolution_clock::now();
      sv.forward3(fb.data(), frames);
      ts.push_back(std::chrono::duration<double, std::milli>(
                       std::chrono::high_resolution_clock::now() - a)
                       .count());
    }
    sv_ms = ms_of(ts);
    std::printf("sv: fwd=%.1fms (%s)\n", sv_ms, mode.c_str());
  }
  return 0;
}
