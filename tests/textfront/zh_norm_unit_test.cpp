// zh_norm_unit_test.cpp — unit tests for the B7 textfront layers:
//   symbols2 roundtrip, num.py verbalizers, NSW rule cases through the full
//   TextNormalizer chain, punctuation layer, and an end-to-end g2p smoke.
// Build:
//   clang++ -std=c++20 -O2 -Wall -Wextra -Werror -Isrc/textfront \
//     tests/textfront/zh_norm_unit_test.cpp \
//     src/textfront/{chinese_g2p,zh_verbalize,zh_rules,zh_norm,jieba,pinyin,tone_sandhi}.cpp
// Run (g2p smoke needs the data bins):
//   ./zh_norm_unit_test src/textfront/data/jieba_trie.bin \
//                       src/textfront/data/pinyin.bin
#include <cstdio>
#include <string>
#include <vector>

#include "chinese_g2p.h"
#include "zh_num.h"
#include "symbols2.hpp"

using namespace gsv::textfront;

static int gFailed = 0;

static void expect(const std::string& what, const std::u32string& got,
                   const char* wantUtf8) {
    // decode want
    std::u32string want;
    for (size_t i = 0; i < std::string(wantUtf8).size();) {
        unsigned char b = static_cast<unsigned char>(wantUtf8[i]);
        if (b < 0x80) { want.push_back(b); ++i; continue; }
        uint32_t cp = 0; size_t extra = 0;
        if ((b & 0xE0) == 0xC0) { cp = b & 0x1F; extra = 1; }
        else if ((b & 0xF0) == 0xE0) { cp = b & 0x0F; extra = 2; }
        else { cp = b & 0x07; extra = 3; }
        for (size_t k = 1; k <= extra; ++k)
            cp = (cp << 6) | (static_cast<unsigned char>(wantUtf8[i + k]) & 0x3F);
        i += extra + 1;
        want.push_back(static_cast<char32_t>(cp));
    }
    if (got != want) {
        auto hex = [](const std::u32string& t) {
            std::string o;
            char b[16];
            for (char32_t c : t) {
                std::snprintf(b, sizeof b, "%X,", static_cast<unsigned>(c));
                o += b;
            }
            return o;
        };
        std::printf("FAIL %s\n  got  %s\n  want %s\n  gots %s wants %s\n",
                    what.c_str(),
                    std::string(got.begin(), got.end()).c_str(), wantUtf8,
                    hex(got).c_str(), hex(want).c_str());
        ++gFailed;
    }
}

static void expectInt(const std::string& what, long got, long want) {
    if (got != want) {
        std::printf("FAIL %s: got %ld want %ld\n", what.c_str(), got, want);
        ++gFailed;
    }
}

static U32 u8(const char* s) {
    U32 o;
    std::string t(s);
    for (size_t i = 0; i < t.size();) {
        unsigned char b = static_cast<unsigned char>(t[i]);
        if (b < 0x80) { o.push_back(b); ++i; continue; }
        uint32_t cp = 0; size_t extra = 0;
        if ((b & 0xE0) == 0xC0) { cp = b & 0x1F; extra = 1; }
        else if ((b & 0xF0) == 0xE0) { cp = b & 0x0F; extra = 2; }
        else { cp = b & 0x07; extra = 3; }
        for (size_t k = 1; k <= extra; ++k)
            cp = (cp << 6) | (static_cast<unsigned char>(t[i + k]) & 0x3F);
        i += extra + 1;
        o.push_back(static_cast<char32_t>(cp));
    }
    return o;
}

int main(int argc, char** argv) {
    // ------------------------------------------------------------------
    // symbols2 roundtrip
    // ------------------------------------------------------------------
    for (int id = 0; id < kSymbols2Count; ++id)
        expectInt("symbols2 roundtrip id " + std::to_string(id),
                  symbols2Id(kSymbols2[id]), id);

    // ------------------------------------------------------------------
    // num.py verbalizers
    // ------------------------------------------------------------------
    expect("cardinal 10", verbalizeCardinal(u8("10")), "十");
    expect("cardinal 105", verbalizeCardinal(u8("105")), "一百零五");
    expect("cardinal 2000", verbalizeCardinal(u8("2000")), "二千");
    expect("cardinal 100000000", verbalizeCardinal(u8("100000000")), "一亿");
    expect("num2str .5", num2str(u8(".5")), "零点五");
    expect("num2str 3.14", num2str(u8("3.14")), "三点一四");
    expect("num2str 20", num2str(u8("20")), "二十");
    expect("digit alt_one", verbalizeDigit(u8("13812345678"), true),
           "幺三八幺二三四五六七八");
    expect("digit plain", verbalizeDigit(u8("2023"), false), "二零二三");

    // ------------------------------------------------------------------
    // TextNormalizer chain (sentence-level rules -> _post_replace)
    // ------------------------------------------------------------------
    TextNormalizer tn;
    auto norm = [&](const char* s) {
        return tn.normalizeSentence(u8(s));
    };
    expect("date", norm("2023年11月5日"), "二零二三年十一月五日");
    expect("time half", norm("09:30"), "九点半");
    expect("time range", norm("08:30~17:00营业"), "八点半至十七点营业");
    expect("time sec", norm("14:05:20"), "十四点零五分二十秒");
    expect("temp neg", norm("-3°C"), "零下三度");
    expect("temp sheshidu bugcompat", norm("20摄氏度"), "二十度");
    expect("percent", norm("50%"), "百分之五十");
    expect("fraction", norm("1/2"), "二分之一");
    expect("mobile", norm("13812345678"), "幺三八幺二三四五六七八");
    expect("telephone", norm("010-12345678"),
           "零幺零减幺二三四五六七八");
    expect("national", norm("400-123-4567"), "四零零减幺二三减四五六七");
    expect("version", norm("1.2.3"), "一点二点三");
    expect("asmd", norm("1+1=2"), "一加一等于二");
    expect("power", norm("2的平方写作2²"), "二的平方写作二的二次方");
    expect("default num", norm("编号00078"), "编号零零零七八");
    expect("quantifier liang", norm("2个苹果"), "两个苹果");
    expect("t2s", norm("機器學習"), "机器学习");
    // DEFAULT_NUM turns the 3+ digit run into 幺-style readings
    expect("f2h", norm("ＡＢＣ１２３"),
           "ABC\xe5\xb9\xba\xe4\xba\x8c\xe4\xb8\x89");

    // ------------------------------------------------------------------
    // punctuation layer
    // ------------------------------------------------------------------
    // python replace_punctuation maps 。->. and ，->, without merging;
    // consecutive-punct collapsing happens later (chinese2.text_normalize)
    expect("rep map + keep", replacePunctuation(u8("你好。，世界！！")),
           "\xe4\xbd\xa0\xe5\xa5\xbd.,\xe4\xb8\x96\xe7\x95\x8c!!");
    expect("collapse",
           collapsePunct(replacePunctuation(u8("啊。。！？？"))),
           "\xe5\x95\x8a.");
    expect("en um en", replacePunctuation(u8("嗯呣")),
           "\xe6\x81\xa9\xe6\xaf\x8d");

    // ------------------------------------------------------------------
    // end-to-end g2p smoke (needs data bins)
    // ------------------------------------------------------------------
    if (argc >= 3) {
        ChineseG2p g2p;
        std::string err;
        if (!g2p.load(argv[1], argv[2], &err)) {
            std::printf("FAIL g2p load: %s\n", err.c_str());
            ++gFailed;
        } else {
            G2pResult r;
            if (!g2p.run("你好，世界。", &r)) {
                std::printf("FAIL g2p run: %s\n", r.error.c_str());
                ++gFailed;
            } else {
                // reference: n i2 h ao3 , sh ir4 j ie4 .
                const int want[] = {227, 167, 158, 119, 1,
                                    251, 214, 221, 194, 3};
                const int wantW2ph[] = {2, 2, 1, 2, 2, 1};
                expectInt("g2p phone count",
                          static_cast<long>(r.phones.size()), 10);
                for (int k = 0; k < 10; ++k)
                    expectInt("g2p phone[" + std::to_string(k) + "]",
                              r.phones[k], want[k]);
                expectInt("g2p word2ph count",
                          static_cast<long>(r.word2ph.size()), 6);
                for (int k = 0; k < 6; ++k)
                    expectInt("g2p word2ph[" + std::to_string(k) + "]",
                              r.word2ph[k], wantW2ph[k]);
            }
            // erhua: 女儿 must NOT merge (not_erhua), 小院儿 must merge
            if (!g2p.run("小院儿真安静", &r)) {
                std::printf("FAIL g2p erhua run: %s\n", r.error.c_str());
                ++gFailed;
            }
        }
    }

    if (gFailed == 0) {
        std::printf("ALL ZH_NORM UNIT TESTS PASSED\n");
        return 0;
    }
    std::printf("%d FAILURES\n", gFailed);
    return 1;
}
