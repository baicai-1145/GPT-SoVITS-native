// jieba_diff_test.cpp — B5 acceptance: native posseg.lcut port must produce
// byte-identical (word, flag) sequences vs jieba_fast for every fixture case.
//
// Usage: jieba_diff_test <jieba_trie.bin> <fixtures.txt>
// Exit 0 = all cases pass; 1 = mismatches; 2 = usage/IO error.
#include "jieba.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using gsv::textfront::JiebaSegmenter;
using gsv::textfront::PosToken;

static std::string unescape(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            char c = s[++i];
            switch (c) {
                case 'n': out.push_back('\n'); break;
                case 't': out.push_back('\t'); break;
                case 'r': out.push_back('\r'); break;
                case '\\': out.push_back('\\'); break;
                default: out.push_back('\\'); out.push_back(c); break;
            }
        } else {
            out.push_back(s[i]);
        }
    }
    return out;
}

struct Case {
    std::string id;
    std::string text;
    std::vector<PosToken> expected;
};

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <jieba_trie.bin> <fixtures.txt>\n", argv[0]);
        return 2;
    }
    JiebaSegmenter seg;
    std::string err;
    if (!seg.load(argv[1], &err)) {
        std::fprintf(stderr, "load failed: %s\n", err.c_str());
        return 2;
    }

    std::ifstream in(argv[2]);
    if (!in) {
        std::fprintf(stderr, "cannot open fixtures: %s\n", argv[2]);
        return 2;
    }
    std::string line;
    if (!std::getline(in, line) || line != "GSVFIX01") {
        std::fprintf(stderr, "bad fixture header\n");
        return 2;
    }

    std::vector<Case> cases;
    Case cur;
    bool curOpen = false;
    while (std::getline(in, line)) {
        if (line.empty()) {
            if (curOpen) { cases.push_back(std::move(cur)); cur = {}; curOpen = false; }
            continue;
        }
        if (line.rfind("S\t", 0) == 0) {
            curOpen = true;
            size_t t1 = line.find('\t', 2);
            cur.id = line.substr(2, t1 - 2);
            cur.text = unescape(line.substr(t1 + 1));
        } else if (line.rfind("W\t", 0) == 0) {
            size_t t1 = line.find('\t', 2);
            PosToken tok;
            tok.word = unescape(line.substr(2, t1 - 2));
            tok.flag = line.substr(t1 + 1);
            cur.expected.push_back(std::move(tok));
        }
    }
    if (curOpen) cases.push_back(std::move(cur));

    size_t nTok = 0;
    size_t failures = 0;
    for (const auto& c : cases) {
        std::vector<PosToken> got;
        seg.lcut(c.text, &got);
        nTok += c.expected.size();
        bool ok = got.size() == c.expected.size();
        if (ok) {
            for (size_t i = 0; i < got.size(); ++i) {
                if (got[i].word != c.expected[i].word ||
                    got[i].flag != c.expected[i].flag) {
                    ok = false;
                    if (failures < 10) {
                        std::printf("MISMATCH [%s] token %zu/%zu:\n  exp: '%s'/'%s'\n"
                                    "  got: '%s'/'%s'\n",
                                    c.id.c_str(), i + 1, got.size(),
                                    c.expected[i].word.c_str(),
                                    c.expected[i].flag.c_str(),
                                    got[i].word.c_str(), got[i].flag.c_str());
                    }
                    break;
                }
            }
        } else if (failures < 10) {
            std::printf("MISMATCH [%s]: expected %zu tokens, got %zu\n  text: %s\n",
                        c.id.c_str(), c.expected.size(), got.size(),
                        c.text.c_str());
        }
        if (!ok) ++failures;
    }

    std::printf("cases=%zu tokens=%zu failures=%zu\n", cases.size(), nTok,
                failures);
    return failures == 0 ? 0 : 1;
}
