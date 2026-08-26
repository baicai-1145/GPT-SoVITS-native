// AR decode 热态基准: 进程内连跑 N 次 generate, 探针报每次 decode
#include <cstdio>
#include <vector>
#include <random>
#include <string>
#include "ar/t2s_engine.hpp"
#include "runtime/gsv_loader.hpp"
int main(int argc, char** argv) {
  const char* w = argc>1?argv[1]:"/Users/baicai1145/gsv-weights";
  char path[512]; snprintf(path,sizeof path,"%s/ar_s1v3.gsv",w);
  auto f = std::make_unique<gsv::rt::GsvFile>(path);
  gsv::ar::T2SEngine eng(*f);
  // --fp16-all 开关
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--fp16-all") {
      gsv::ar::T2SEngine::Fp16Options o;
      o.kv = true; o.gemv = true;
      eng.set_fp16(o);
    }
  }
  // 简单 phones: 23 个音素 (与测试句同), bert 全零, prompt 199 个随机语义 id
  std::vector<int64_t> phones(23); for(int i=0;i<23;i++) phones[i]=5+i;
  std::vector<float> bert(23*768, 0.f);
  std::mt19937 rng(42);
  std::vector<int64_t> prompt(199);
  for(auto&p:prompt) p = rng()%1000;
  for(int r=0;r<5;r++){
    auto res = eng.generate(phones.data(), phones.size(), prompt.data(), prompt.size(), bert.data());
    fprintf(stderr,"run%d: steps=%zu prefill=%.2f decode=%.2f (%.3f ms/tok)\n",
            r, res.steps, eng.last_prefill_ms(), eng.last_decode_ms(),
            eng.last_decode_ms()/res.steps);
  }
  return 0;
}
