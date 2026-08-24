// e2_sv_layers.cpp — SV 分层 dump(conv1/layer1..4/fuse34/emb), 定位 fp16 偏差层。
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "encoder/fbank.hpp"
#include "encoder/sv.hpp"
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
  rt::GsvFile f(std::string(argv[1]) + "/eres2netv2_sv.gsv");
  encoder::SvEngine sv(f);
  rt::wav::WavFile w = rt::wav::load_wav(argv[2]);
  size_t frames = 0;
  const std::vector<float> fb =
      encoder::kaldi_fbank_80(w.samples.data(), w.samples.size(), &frames);
  sv.forward3(fb.data(), frames);
  fs::create_directories(argv[3]);
  std::string d = argv[3];
  dump_f32(d + "/conv1.f32", sv.conv1_out().data(), sv.conv1_out().size());
  for (int l = 1; l <= 4; ++l) {
    const auto& v = sv.layer_out(l);
    dump_f32(d + "/layer" + std::to_string(l) + ".f32", v.data(), v.size());
  }
  dump_f32(d + "/fuse34.f32", sv.fuse34_out().data(), sv.fuse34_out().size());
  dump_f32(d + "/emb.f32", sv.emb_out().data(), sv.emb_out().size());
  std::printf("frames=%zu all dumped\n", frames);
  return 0;
}
