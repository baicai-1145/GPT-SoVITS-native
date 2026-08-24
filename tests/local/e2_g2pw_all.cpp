// e2_g2pw_all.cpp — E2-ENC 验收: pairs 全部唯一句子的 phones(G2PW 多音字路径)输出。
// 只跑 TextFrontend(+G2PW 注入), 不加载 AR/SoVITS/HuBERT —— 快速全量。
// golden 的 phones_ids 含 prompt 前缀(15 phones), 由 python 侧拼接对照。
//
// 用法: ./e2_g2pw_all <weightsDir> <dataDir> <sentences.txt(utf8 每行一句)> <out.tsv>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "textfront/textfront.h"

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr, "usage: %s <weightsDir> <dataDir> <sentences.txt> <out.tsv>\n",
                 argv[0]);
    return 2;
  }
  const std::string wdir = argv[1], dataDir = argv[2], sentPath = argv[3],
                    outPath = argv[4];

  gsv::textfront::TextFrontend::G2pwOptions g2pwOpt;
  g2pwOpt.gsvPath = wdir + "/g2pw_bert.gsv";
  g2pwOpt.assetsBin = dataDir + "/g2pw_assets.bin";
  g2pwOpt.vocabPath = dataDir + "/bert_vocab.txt";

  gsv::textfront::TextFrontend tf;
  std::string err;
  if (!tf.load(dataDir + "/jieba_trie.bin", dataDir + "/pinyin.bin", &err,
               dataDir + "/cmudict.bin", &g2pwOpt)) {
    std::fprintf(stderr, "tf load fail: %s\n", err.c_str());
    return 1;
  }

  std::ifstream sf(sentPath);
  if (!sf) { std::fprintf(stderr, "缺句子文件 %s\n", sentPath.c_str()); return 2; }
  std::ofstream out(outPath);
  std::string line;
  while (std::getline(sf, line)) {
    if (line.empty()) continue;
    gsv::textfront::TextFrontend::Result one;
    if (!tf.process(line, &one, /*cutMethod=*/0)) {
      std::fprintf(stderr, "process fail: %s\n", one.error.c_str());
      return 1;
    }
    out << line << "\t";
    for (size_t k = 0; k < one.phones.size(); ++k)
      out << one.phones[k] << (k + 1 < one.phones.size() ? "," : "");
    out << "\n";
  }
  std::printf("wrote %s\n", outPath.c_str());
  return 0;
}
