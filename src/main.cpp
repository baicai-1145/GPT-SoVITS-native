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
      "  --prompt-text S   参考文本: 必须为参考音频的逐字转写,\n"
      "                    错配将导致严重音质/复读退化 [默认空=no_prompt_text]\n"
      "  --no-cache        禁用参考特征缓存\n"
      "  --dump-sovits-in F 全链跑后快照 SoVITS 输入到 F (供重放)\n"
      "  --sovits-in F     重放模式: 只跑 SoVITS (跳过文本前端/AR, 无需 --text/--ref-wav)\n"
      "  --overlap         流水重叠模式: AR(N+1) ‖ SoVITS(N) (数值同串行)\n"
      "  --timing-csv F    per-segment 三阶段耗时 CSV 输出路径\n"
      "  --amx             SoVITS conv 启用 AMX fp16 GEMM 后端(实验; 默认关)\n"
      "  --sample          AR 采样对齐 python(top_k=15/pen=1.35, 根治长文本复读; 默认贪心=位级口径)\n"
      "  --sample-top-k N  自定义 top_k (隐含 --sample)\n"
      "  --sample-seed S   采样种子(默认随机)\n"
      "  -h/--help         本帮助\n",
      argv0);
}

}  // namespace

int main(int argc, char** argv) {
  std::string text, refWav, weights = "weights", data = "src/runtime/data",
                              out = "out.wav";
  std::string sovitsInPath;  // --sovits-in: 重放模式 (只跑 SoVITS)
  gsv::rt::pipeline::PipelineOptions opt;
  int opt_threads = 0;
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
      opt_threads = std::atoi(next("--threads").c_str());
      opt.threads = opt_threads;
    } else if (a == "--prompt-text") {
      opt.prompt_text = next("--prompt-text");
    } else if (a == "--overlap") {
      opt.overlap = true;
    } else if (a == "--amx") {
      opt.sovits_amx = true;
    } else if (a == "--fp16") {
      opt.ar_fp16_kv = true;
    } else if (a == "--fp16-all") {
      opt.ar_fp16_kv = true;
      opt.ar_fp16_gemv = true;
    } else if (a == "--sample") {
      // E4: python 口径采样(top_k=15/pen=1.35) 根治长段贪心复读
      opt.ar_sample_on = true;
      opt.ar_sampling.mode = gsv::ar::SamplingParams::Mode::TopK;
    } else if (a == "--sample-top-k") {
      if (++i >= argc) throw std::runtime_error("--sample-top-k 需参数");
      opt.ar_sample_on = true;
      opt.ar_sampling.mode = gsv::ar::SamplingParams::Mode::TopK;
      opt.ar_sampling.top_k = std::strtoul(argv[i], nullptr, 10);
    } else if (a == "--sample-seed") {
      if (++i >= argc) throw std::runtime_error("--sample-seed 需参数");
      opt.ar_sampling.seed = std::strtoull(argv[i], nullptr, 10);
    } else if (a == "--timing-csv") {
      opt.timing_csv = next("--timing-csv");
    } else if (a == "--dump-sovits-in") {
      opt.sovits_in_dump = next("--dump-sovits-in");
    } else if (a == "--sovits-in") {
      sovitsInPath = next("--sovits-in");
    } else if (a == "--no-cache") {
      opt.use_ref_cache = false;
    } else {
      std::fprintf(stderr, "错误: 未知参数 '%s' (--help 查看用法)\n", a.c_str());
      return 2;
    }
  }

  if (!sovitsInPath.empty()) {
    // 重放模式只需快照文件, 不要求 --text/--ref-wav
  } else if (!haveText || !haveRef) {
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
  if (sovitsInPath.empty() && !std::filesystem::exists(refWav)) {
    std::fprintf(stderr, "错误: 参考音频不存在: %s\n", refWav.c_str());
    return 2;
  }

  // §4 线程旋钮: --threads N 经 VECLIB_MAXIMUM_THREADS 限制 Accelerate 内部
  // GEMM 线程宽度 (prefill/VITS/BERT); AR decode 为单线程 NEON GEMV 本就保守。
  // 必须在首次 Accelerate 调用(权重加载)前设置。0/缺省 = 不限制(全 P 核)。
  if (opt_threads > 0) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%d", opt_threads);
    ::setenv("VECLIB_MAXIMUM_THREADS", buf, /*overwrite=*/1);
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
  if (!sovitsInPath.empty()) {
    // 重放模式: 跳过 textfront/AR, 只跑 SoVITS (需 --ref-wav 不需要)
    if (!pipe.synthesizeFromSovitsIn(sovitsInPath, &res, &err)) {
      std::fprintf(stderr, "错误: SoVITS重放失败: %s\n", err.c_str());
      return 1;
    }
  } else if (!pipe.synthesize(text, refWav, &res, &err)) {
    std::fprintf(stderr, "错误: 合成失败: %s\n", err.c_str());
    return 1;
  }
  const double tDone = nowMs();
  // 首包延迟 = 合成开始→首段音频就绪 (pipeline 精确测量, 两模式同语义);
  // 不含模型加载时间, 仅记录不作为性能主张。
  const double firstPacketMs = res.first_packet_ms;

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
