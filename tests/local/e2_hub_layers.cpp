// e2_hub_layers.cpp — HuBERT 分层 dump(cnn/proj/pos/enc_ln/l0), 定位 fp16 偏差层。
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "encoder/hubert.hpp"
#include "runtime/gsv_loader.hpp"
#include "runtime/wav.hpp"

namespace fs = std::filesystem;
using namespace gsv;

static void dump_f32(const std::string& p, const float* d, size_t n) {
  std::ofstream o(p, std::ios::binary);
  o.write(reinterpret_cast<const char*>(d), std::streamsize(n * 4));
}

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: %s <weightsDir> <refWav16k> <outDir>\n", argv[0]);
    return 2;
  }
  rt::GsvFile f(argv[1] + std::string("/hubert_base.gsv"));
  encoder::HubertEngine hub(f);
  rt::wav::WavFile w = rt::wav::load_wav(argv[2]);
  std::vector<float> wav16k = w.samples;
  wav16k.insert(wav16k.end(), 9600, 0.f);
  const size_t T = hub.run(wav16k.data(), wav16k.size());
  fs::create_directories(argv[3]);
  std::string d = argv[3];
  dump_f32(d + "/cnn.f32", hub.cnn_out().data(), hub.cnn_out().size());
  dump_f32(d + "/proj.f32", hub.proj_out().data(), hub.proj_out().size());
  dump_f32(d + "/pos.f32", hub.pos_out_view().data(),
           T * 768);  // 只取前 T 帧(SamePad 后)
  dump_f32(d + "/encln.f32", hub.encln_out_view().data(), T * 768);
  dump_f32(d + "/l0.f32", hub.l0_out().data(), T * 768);
  dump_f32(d + "/last.f32", hub.out().data(), T * 768);
  std::printf("T=%zu all dumped\n", T);
  return 0;
}
