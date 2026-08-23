// g2pw_e2e.cpp — B6 验收 harness
// 用法:
//   g2pw_e2e tok   <vocab.txt> <g2pw_assets.bin> <tokens.json> <out.txt>
//   g2pw_e2e conv  <gsv.gsv> <g2pw_assets.bin> <vocab.txt> <pinyin.json> <out.txt>
#include "runtime/mini_json.hpp"
#include "textfront/g2pw.hpp"
#include "textfront/wordpiece.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

using namespace gsv::textfront;

static std::string slurp(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

// JSON 解析复用 runtime/mini_json.hpp (只读依赖)
struct JDoc {
  std::vector<std::pair<std::string, std::vector<std::string>>> entries;
};

static bool parseJsonMap(const std::string& src, JDoc* doc) {
  auto v = gsv::rt::json::parse(src.data(), src.size());
  if (!v.is(gsv::rt::json::JType::Object)) return false;
  for (auto& [k, val] : v.obj) {
    const auto* o = val.is(gsv::rt::json::JType::Object) ? &val : nullptr;
    if (!o) continue;
    std::string text;
    std::vector<std::string> arr;
    if (const auto* t = o->find("text"); t && t->is(gsv::rt::json::JType::String))
      text = t->s;
    if (const auto* a = o->find("tokens"); a && a->is(gsv::rt::json::JType::Array))
      for (auto& e : a->arr) arr.push_back(e.s);
    if (const auto* a = o->find("pinyins"); a && a->is(gsv::rt::json::JType::Array))
      for (auto& e : a->arr) arr.push_back(e.s);
    if (const auto* a = o->find("ids"); a) {}  // ids 不用于 diff
    doc->entries.emplace_back(std::move(text), std::move(arr));
  }
  return true;
}

int main(int argc, char** argv) {
  if (argc < 6) {
    std::fprintf(stderr,
                 "usage: %s tok <vocab> <assets.bin> <tokens.json> <out>\n"
                 "       %s conv <gsv> <assets.bin> <vocab> <pinyin.json> <out>\n",
                 argv[0], argv[0]);
    return 1;
  }
  const std::string mode = argv[1];
  if (mode == "tok") {
    WordPieceVocab vocab;
    if (!vocab.load(argv[2])) {
      std::fprintf(stderr, "vocab load failed\n");
      return 1;
    }
    NormTable nt;
    {
      FILE* f = fopen(argv[3], "rb");
      if (!f) return 1;
      char magic[8];
      fread(magic, 1, 8, f);
      fseek(f, -4, SEEK_END);  // 不行——需要顺序读。改用 G2PWAssets::load 加载 norm
      fclose(f);
    }
    // norm 表在 assets.bin 里: 借助 G2PWAssets 解析(会顺带载入其余段, 内存可接受)
    G2PWAssets a;
    if (!a.load(argv[3], argv[2])) {
      std::fprintf(stderr, "assets load failed\n");
      return 1;
    }
    JDoc doc;
    if (!parseJsonMap(slurp(argv[4]), &doc)) {
      std::fprintf(stderr, "json parse failed\n");
      return 1;
    }
    FILE* out = fopen(argv[5], "w");
    size_t nfail = 0, ntok = 0;
    for (auto& [text, golden_toks] : doc.entries) {
      std::vector<std::string> toks;
      std::vector<int> ids;
      std::vector<size_t> starts;
      tokenizeAndMapFull(text, a.norm, vocab, &toks, &ids, &starts);
      ntok += toks.size();
      bool ok = toks.size() == golden_toks.size();
      for (size_t k = 0; ok && k < toks.size(); ++k)
        if (toks[k] != golden_toks[k]) ok = false;
      if (!ok) {
        ++nfail;
        fprintf(out, "FAIL %s\ncpp : ", text.c_str());
        for (auto& t : toks) fprintf(out, "%s ", t.c_str());
        fprintf(out, "\ngold: ");
        for (auto& t : golden_toks) fprintf(out, "%s ", t.c_str());
        fprintf(out, "\n");
      }
    }
    fprintf(out, "\n==== tok: %zu/%zu sentences pass (%zu tokens)\n",
            doc.entries.size() - nfail, doc.entries.size(), ntok);
    fclose(out);
    printf("tok done: %zu/%zu pass\n", doc.entries.size() - nfail,
           doc.entries.size());
    return nfail ? 1 : 0;
  }
  // conv mode
  G2PWConverter conv;
  std::string err;
  if (!conv.load(argv[2], argv[3], argv[4], &err)) {
    std::fprintf(stderr, "converter load failed: %s\n", err.c_str());
    return 1;
  }
  JDoc doc;
  if (!parseJsonMap(slurp(argv[5]), &doc)) {
    std::fprintf(stderr, "json parse failed\n");
    return 1;
  }
  FILE* out = fopen(argv[6], "w");
  size_t nfail = 0, nchar = 0;
  for (auto& [text, golden_py] : doc.entries) {
    auto py = conv.convert(text);
    nchar += py.size();
    bool ok = py.size() == golden_py.size();
    for (size_t k = 0; ok && k < py.size(); ++k)
      if (py[k] != golden_py[k]) ok = false;
    if (!ok) {
      ++nfail;
      fprintf(out, "FAIL %s\n     cpp:", text.c_str());
      for (auto& t : py) fprintf(out, " %s", t.c_str());
      fprintf(out, "\n   gold:");
      for (auto& t : golden_py) fprintf(out, " %s", t.c_str());
      fprintf(out, "\n");
    }
  }
  fprintf(out, "\n==== conv: %zu/%zu sentences pass (%zu chars)\n",
          doc.entries.size() - nfail, doc.entries.size(), nchar);
  fclose(out);
  printf("conv done: %zu/%zu pass (%zu chars)\n", doc.entries.size() - nfail,
         doc.entries.size(), nchar);
  return nfail ? 1 : 0;
}
