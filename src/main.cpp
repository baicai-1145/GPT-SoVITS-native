// main.cpp — C2: gsv-native CLI 全链路入口。
//
// 用法:
//   gsv-native --text "<中文文本>" --ref-wav ref.wav [选项]
//
// 选项:
//   --weights DIR    模型目录 (ar_s1v3.gsv / sovits_v2ProPlus.gsv /
//                    hubert_base.gsv / eres2netv2_sv.gsv /
//                    roberta_wwm_ext_large.gsv)      [默认 ./weights]
//   --data DIR       文本前端数据 (jieba_trie.bin/pinyin.bin/roberta_vocab.txt)
//                                                    [默认 src/runtime/data]
//   --out FILE       输出 wav 路径 (int16 PCM @32kHz) [默认 out.wav]
//   --cut N          分句方法 0..5 (对应 cut0..cut5, 0=不切句) [默认 1]
//   --seed N         SoVITS 噪声种子 [默认 42]
//   --threads N      线程数提示 (0=自动) [默认 0]
//   --no-cache       禁用参考特征磁盘缓存
//   -h, --help       显示本帮助
//
// 参考音频要求: 单声道/立体声均可, 时长 3~10 秒 (16 kHz 下 48000..160000 样本)。
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "runtime/pipeline.hpp"
#include "sovits/wav_writer.hpp"

namespace {

double nowMs() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

void printHelp(const char* argv0) {
  std::printf(
      "gsv-native — GPT-SoVITS v2ProPlus 纯 CPU 推理 (v2ProPlus 家族)\n"
      "\n"
      "用法: %s --text \"<文本>\" --ref-wav <ref.wav> [选项]\n"
      "\n"
      "必选:\n"
      "  --text STR        要合成的 UTF-8 中文文本\n"
      "  --ref-wav FILE    参考音频 (3~10 秒)\n"
      "\n"
      "选项:\n"
      "  --weights DIR     模型 .gsv 目录            [默认 ./weights]\n"
      "  --data DIR        文本前端数据目录          [默认 "
      "src/runtime/data]\n"
      "  --out FILE        输出 wav (int16@32k)      [默认 out.wav]\n"
      "  --cut N           分句法 0..5 (cut0..cut5)  [默认 1]\n"
      "  --seed N          噪声种子                  [默认 42]\n"
      "  --threads N       线程数 (0=自动)           [默认 0]\n"
      "  --prompt-text S   参考文本 (空串禁用)   [默认 原来你也玩原神。]\n"
      "  --no-cache        禁用参考特征缓存\n"
      "  -h/--help         本帮助\n",
      argv0);
}

}  // namespace

int main(int argc, char** argv) {
  std::string text, refWav, weights = "weights", data = "src/runtime/data",
                              out = "out.wav";
  gsv::rt::pipeline::PipelineOptions opt;
  bool haveText = false, haveRef = false;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "错误: %s 缺少参数值\n", name);
        exit(2);
      }
      return argv[++i];
    };
    if (a == "--help" || a == "-h") {
      printHelp(argv[0]);
      return 0;
    } else if (a == "--text") {
      text = next("--text");
      haveText = true;
    } else if (a == "--ref-wav") {
      refWav = next("--ref-wav");
      haveRef = true;
    } else if (a == "--weights") {
      weights = next("--weights");
    } else if (a == "--data") {
      data = next("--data");
    } else if (a == "--out") {
      out = next("--out");
    } else if (a == "--cut") {
      opt.cut_method = std::atoi(next("--cut").c_str());
      if (opt.cut_method < 0 || opt.cut_method > 5) {
        std::fprintf(stderr, "错误: --cut 取值 0..5\n");
        return 2;
      }
    } else if (a == "--seed") {
      opt.seed = std::strtoull(next("--seed").c_str(), nullptr, 10);
    } else if (a == "--threads") {
      opt.threads = std::atoi(next("--threads").c_str());
    } else if (a == "--prompt-text") {
      opt.prompt_text = next("--prompt-text");
    } else if (a == "--no-cache") {
      opt.use_ref_cache = false;
    } else {
      std::fprintf(stderr, "错误: 未知参数 '%s' (--help 查看用法)\n", a.c_str());
      return 2;
    }
  }

  if (!haveText || !haveRef) {
    std::fprintf(stderr,
                 "错误: 必须提供 --text 与 --ref-wav (--help 查看用法)\n");
    return 2;
  }
  for (const auto& [dir, what] :
       {std::pair{weights, "模型目录"}, {data, "数据目录"}}) {
    if (!std::filesystem::exists(dir)) {
      std::fprintf(stderr, "错误: %s不存在: %s\n", what, dir.c_str());
      return 2;
    }
  }
  if (!std::filesystem::exists(refWav)) {
    std::fprintf(stderr, "错误: 参考音频不存在: %s\n", refWav.c_str());
    return 2;
  }

  const double tStart = nowMs();
  gsv::rt::pipeline::Pipeline pipe;
  std::string err;
  if (!pipe.load(weights, data, opt, &err)) {
    std::fprintf(stderr, "错误: 模型加载失败: %s\n", err.c_str());
    return 1;
  }
  const double tLoaded = nowMs();

  gsv::rt::pipeline::SynthResult res;
  if (!pipe.synthesize(text, refWav, &res, &err)) {
    std::fprintf(stderr, "错误: 合成失败: %s\n", err.c_str());
    return 1;
  }
  const double tDone = nowMs();
  // 首包延迟(近似口径) = 首段 AR+Vocoder 耗时; 参考条件构建未单独计时,
  // 仅记录不作为性能主张 (M5 bench harness 再细化)。
  const double firstPacketMs =
      res.segments.empty()
          ? 0.0
          : res.segments.front().ar_ms + res.segments.front().voc_ms;

  if (res.audio.empty()) {
    std::fprintf(stderr, "错误: 未产出任何音频\n");
    return 1;
  }
  try {
    gsv::sovits::write_wav_int16(out, res.audio.data(),
                                       res.audio.size(), res.sr);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "错误: 写出失败: %s\n", e.what());
    return 1;
  }

  const double audioSec = double(res.audio.size()) / double(res.sr);
  const double inferSec = (tDone - tLoaded) / 1000.0;
  std::printf("输出: %s (%.2f 秒 @%u Hz, %d 段)\n", out.c_str(), audioSec,
              res.sr, int(res.segments.size()));
  std::printf("参考: prompt_semantic=%zu%s, ge/ge_text 已构建%s\n",
              res.prompt_semantic.size(),
              res.ref_from_cache ? "(缓存)" : "",
              res.ref_from_cache ? "(缓存)" : "");
  for (size_t i = 0; i < res.segments.size(); ++i) {
    const auto& s = res.segments[i];
    std::printf("  段%02zu phones=%zu tokens=%zu ar=%.0fms voc=%.0fms eos=%d\n",
                i, s.phones.size(), s.tokens.size(), s.ar_ms, s.voc_ms,
                int(s.hit_eos));
  }
  std::printf(
      "计时: load=%.0fms infer=%.0fms RTF=%.3f 首包≈%.0fms\n",
      tLoaded - tStart, tDone - tLoaded, audioSec > 0 ? inferSec / audioSec : 0,
      firstPacketMs);
  return 0;
}
