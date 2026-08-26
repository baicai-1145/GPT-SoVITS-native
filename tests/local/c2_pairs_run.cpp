// c2_pairs_run.cpp — C2 验收 (b): CLI 全链路 phones/tokens 与 pairs golden 对照。
// 复用 Pipeline(与 gsv_native 完全同路径), 把每段 phones/tokens/prompt 长度
// 写成 JSON 供 python 侧对照 tests/golden/pairs/*.pt。
//
// 用法: ./c2_pairs_run <weightsDir> <dataDir> <refWav> <out.json>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

#include "runtime/pipeline.hpp"

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr, "usage: %s <weightsDir> <dataDir> <refWav> <out.json>\n",
                 argv[0]);
    return 2;
  }
  // 与 tools/golden_export.py 的 SENTENCES 对齐 (验收取前 3 句 zh)
  static const char* kSents[] = {
      "你好，世界。",
      "今天天气真不错，我们去公园散步吧。",
      "重庆的火锅店终于开张了。",
  };

  gsv::rt::pipeline::Pipeline pipe;
  gsv::rt::pipeline::PipelineOptions opt;
  opt.cut_method = 0;  // pairs 导出用 cut0 单段完整推理
#if defined(GSV_AMX_GEMM)
  // E8: 允许用 GSV_AMX_BERT=1 环境变量验收 AMX 路径的 golden 对齐
  if (std::getenv("GSV_AMX_BERT")) {
    opt.bert_amx = true;
    std::fprintf(stderr, "[c2_pairs_run] GSV_AMX_BERT=1 → bert_amx enabled\n");
  }
#endif
  std::string err;
  if (!pipe.load(argv[1], argv[2], opt, &err)) {
    std::fprintf(stderr, "load fail: %s\n", err.c_str());
    return 1;
  }

  std::ofstream out(argv[4]);
  out << "{\"ref\":\"" << argv[3] << "\",\"runs\":[\n";
  for (int i = 0; i < 3; ++i) {
    gsv::rt::pipeline::SynthResult res;
    if (!pipe.synthesize(kSents[i], argv[3], &res, &err)) {
      std::fprintf(stderr, "synth[%d] fail: %s\n", i, err.c_str());
      return 1;
    }
    const auto& seg = res.segments.at(0);
    char buf[128];
    std::snprintf(buf, sizeof(buf), "%zu", res.prompt_semantic.size());
    out << "{\"sentence_idx\":" << i
        << ",\"prompt_len\":" << buf
        << ",\"prompt_codes\":[";
    for (size_t k = 0; k < res.prompt_semantic.size(); ++k)
      out << res.prompt_semantic[k]
          << (k + 1 < res.prompt_semantic.size() ? "," : "");
    // 注: SegmentResult.phones 为段自身(不含提示前缀); pairs 的 phones_ids
    // 含提示文本前缀, 对照在 tools/check_c2_pairs.py 里拼接。
    out << "],\"phones\":[";
    for (size_t k = 0; k < seg.phones.size(); ++k)
      out << seg.phones[k] << (k + 1 < seg.phones.size() ? "," : "");
    out << "],\"raw_tokens\":[";
    for (size_t k = 0; k < seg.raw_argmax.size(); ++k)
      out << seg.raw_argmax[k] << (k + 1 < seg.raw_argmax.size() ? "," : "");
    out << "],\"tokens\":[";
    for (size_t k = 0; k < seg.tokens.size(); ++k)
      out << seg.tokens[k] << (k + 1 < seg.tokens.size() ? "," : "");
    out << "],\"hit_eos\":" << (seg.hit_eos ? "true" : "false") << "}";
    out << (i + 1 < 3 ? ",\n" : "\n");
  }
  out << "]}\n";
  std::printf("wrote %s\n", argv[4]);
  return 0;
}
