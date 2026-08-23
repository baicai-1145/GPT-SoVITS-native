// textfront_unit_test.cpp — B9 unit tests: TextFrontend segmentation
// boundaries, end-to-end process invariants (sum(word2ph)==len(phones)),
// resolver injection, and mixed digits/punct/English long-sentence cases.
//
// Build:
//   clang++ -std=c++20 -O2 -Wall -Wextra -Werror -Isrc/textfront \
//     tests/textfront/textfront_unit_test.cpp \
//     src/textfront/{textfront,chinese_g2p,zh_verbalize,zh_rules,zh_norm,jieba,pinyin,tone_sandhi}.cpp
// Run:
//   ./textfront_unit_test src/textfront/data/jieba_trie.bin \
//                         src/textfront/data/pinyin.bin
#include <cstdio>
#include <string>
#include <vector>

#include "chinese_g2p.h"
#include "pinyin.h"
#include "textfront.h"

using namespace gsv::textfront;

static int gFailed = 0;

static void expectStr(const std::string& what, const std::string& got,
                      const std::string& want) {
    if (got != want) {
        std::printf("FAIL %s\n  got  %s\n  want %s\n", what.c_str(),
                    got.c_str(), want.c_str());
        ++gFailed;
    }
}

static void expectLong(const std::string& what, long long got,
                       long long want) {
    if (got != want) {
        std::printf("FAIL %s: got %lld want %lld\n", what.c_str(), got, want);
        ++gFailed;
    }
}

static void expectSize(const std::string& what, size_t got, size_t want) {
    expectLong(what, static_cast<long long>(got),
               static_cast<long long>(want));
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf(
            "usage: %s <jieba_trie.bin> <pinyin.bin>\n"
            "(split-only assertions run without data files)\n",
            argv[0]);
    }

    // ------------------------------------------------------------------
    // segmentation boundaries (no data needed)
    // ------------------------------------------------------------------

    auto decode = [](const char* s) {
        std::u32string o;
        std::string t(s);
        for (size_t i = 0; i < t.size();) {
            unsigned char b = static_cast<unsigned char>(t[i]);
            if (b < 0x80) { o.push_back(b); ++i; continue; }
            uint32_t cp = 0; size_t extra = 0;
            if ((b & 0xE0) == 0xC0) { cp = b & 0x1F; extra = 1; }
            else if ((b & 0xF0) == 0xE0) { cp = b & 0x0F; extra = 2; }
            else { cp = b & 0x07; extra = 3; }
            for (size_t k = 1; k <= extra; ++k)
                cp =
                    (cp << 6) | (static_cast<unsigned char>(t[i + k]) & 0x3F);
            i += extra + 1;
            o.push_back(static_cast<char32_t>(cp));
        }
        return o;
    };
    auto enc = [](const std::u32string& t) {
        std::string o;
        for (char32_t c : t) {
            if (c < 0x80) {
                o += static_cast<char>(c);
            } else {
                unsigned char b[4];
                int n;
                if (c < 0x800) { n = 2; b[0] = 0xC0 | (c >> 6); }
                else if (c < 0x10000) { n = 3; b[0] = 0xE0 | (c >> 12); }
                else { n = 4; b[0] = 0xF0 | (c >> 18); }
                for (int k = 1; k < n; ++k)
                    b[k] = 0x80 | ((c >> (6 * (n - k - 1))) & 0x3F);
                o.append(reinterpret_cast<char*>(b), n);
            }
        }
        return o;
    };

    // short unpunctuated input: prepend 。, cut1 whole-input path, then the
    // post-step appends 。 because the tail is not a splitor
    {
        auto v = TextFrontend::splitSentences(decode("你好"), 1);
        expectSize("short input -> 1 sentence", v.size(), 1);
        expectStr("short input text", enc(v[0]), "\xe3\x80\x82\xe4\xbd\xa0\xe5\xa5\xbd\xe3\x80\x82");
    }
    // cut1: range(0,n,4) with LAST entry replaced by None => the tail group
    // swallows everything remaining (5 pieces -> ONE group here)
    {
        auto v = TextFrontend::splitSentences(
            decode("你好，世界。测试！完成。再来，结束。"), 1);
        expectSize("cut1 tail-swallow -> 1 segment", v.size(), 1);
    }
    {
        auto v =
            TextFrontend::splitSentences(decode("一，二。三！四？五。"), 1);
        expectSize("cut1 five pieces -> 1 group", v.size(), 1);
        // get_first("一")==len1 <4 => 。 prepended before cut1
        expectStr("cut1 group0", enc(v[0]),
                  "\xe3\x80\x82\xe4\xb8\x80\xef\xbc\x8c\xe4\xba\x8c"
                  "\xe3\x80\x82\xe4\xb8\x89\xef\xbc\x81\xe5\x9b\x9b"
                  "\xef\xbc\x9f\xe4\xba\x94\xe3\x80\x82");
    }
    // pure-punctuation input is dropped by cut filters
    {
        auto v = TextFrontend::splitSentences(decode("。。。！！！"), 1);
        expectSize("pure punct dropped", v.size(), 0);
    }
    // cut5 keeps decimal points intact
    {
        auto v = TextFrontend::splitSentences(decode("圆周率是3.14哦。"), 5);
        expectSize("cut5 decimal kept whole", v.size(), 1);
        expectStr("cut5 text", enc(v[0]), "圆周率是3.14哦。");
    }
    // cut4 does not split 3.14 but splits prose dots
    {
        auto v = TextFrontend::splitSentences(decode("A B.First.Second 3.14!"),
                                              4);
        bool sawDecimal = false;
        for (auto& s : v)
            if (enc(s).find("3.14") != std::string::npos) sawDecimal = true;
        expectLong("cut4 keeps 3.14", sawDecimal ? 1 : 0, 1);
    }
    // >510-char single piece goes through split_big_text
    {
        // no punctuation anywhere: cut1 returns the whole input, the post
        // step appends 。, then >510 triggers split_big_text which yields
        // ["", <text>, "。"] (the empty leading piece mirrors python)
        std::string longText;
        for (int k = 0; k < 60; ++k) longText += "这一句很长没有标点符号";
        auto v = TextFrontend::splitSentences(decode(longText.c_str()), 1);
        expectSize("long no-punct -> ['', text, '。']", v.size(), 3);
        expectStr("big-text leading empty piece", enc(v[0]), "");
        expectStr("big-text middle piece", enc(v[1]), longText);
        expectStr("big-text trailing splitor", enc(v[2]), "\xe3\x80\x82");
    }

    // ------------------------------------------------------------------
    // end-to-end process (needs data bins)
    // ------------------------------------------------------------------
    if (argc >= 3) {
        TextFrontend tf;
        std::string err;
        if (!tf.load(argv[1], argv[2], &err,
                     "src/textfront/data/cmudict.bin")) {
            std::printf("FAIL tf load: %s\n", err.c_str());
            return 1;
        }
        struct Case {
            const char* text;
            int method;
            const char* what;
        };
        Case cases[] = {
            {"你好，世界。", 1, "simple"},
            {"2024年3月15日，气温-3°C~35度，湿度50%。", 1, "digits+units"},
            {"电话13800138000或010-12345678，版本1.2.3已发布！", 5,
             "phones+version"},
            {"混合English与中文的long text，包含GPT-SoVITS专有名词。"
             "第二句话有数字12345和百分比99.9%。",
             2, "mixed en/zh long"},
            {"小院儿的媳妇儿在胡同儿口说话儿。", 1, "erhua"},
            {"第一句。", 0, "cut0"},
            {"我用AI写代码，效率真高。", 1, "mixed zh/en"},
            {"The quick brown fox jumps over the lazy dog.", 0,
             "pure en"},
            {"GPT-SoVITS 是一个开源的语音合成项目。", 1, "en prefix"},
            {"第1名是Tom，第2名是Jerry。", 4, "multi-seg en"},
        };
        for (auto& tc : cases) {
            TextFrontend::Result r;
            if (!tf.process(tc.text, &r, tc.method)) {
                std::printf("FAIL process[%s]: %s\n", tc.what,
                            r.error.c_str());
                ++gFailed;
                continue;
            }
            long long wsum = 0;
            for (int w : r.word2ph) wsum += w;
            expectLong(std::string("sum(word2ph)==len(phones) [") +
                           tc.what + "]",
                       wsum, static_cast<long long>(r.phones.size()));
            expectSize(std::string("sentences nonempty [") + tc.what + "]",
                       r.sentences.empty() ? 0 : 1, 1);
            // sentences are UTF-8; check the last CODEPOINT
            auto endsWithSplitor = [](const std::string& s) {
                if (s.empty()) return false;
                // ASCII splitors end with a plain byte
                unsigned char lb = static_cast<unsigned char>(s.back());
                if (lb < 0x80)
                    return std::string(",.?!~-").find(
                               static_cast<char>(lb)) != std::string::npos;
                // multi-byte: walk back over UTF-8 continuation bytes to the
                // lead byte, then decode
                size_t i = s.size() - 1;
                while (i > 0 &&
                       (static_cast<unsigned char>(s[i]) & 0xC0) == 0x80)
                    --i;
                unsigned char b = static_cast<unsigned char>(s[i]);
                uint32_t cp = 0; size_t extra = 0;
                if ((b & 0xE0) == 0xC0) { cp = b & 0x1F; extra = 1; }
                else if ((b & 0xF0) == 0xE0) { cp = b & 0x0F; extra = 2; }
                else { cp = b & 0x07; extra = 3; }
                for (size_t k = 1; k <= extra; ++k)
                    cp = (cp << 6) |
                         (static_cast<unsigned char>(s[i + k]) & 0x3F);
                const char* S = "\xef\xbc\x8c\xe3\x80\x82\xef\xbc\x9f"
                                "\xef\xbc\x81,.?!~:\xef\xbc\x9a\xe2\x80\x94\xe2\x80\xa6";
                std::u32string sv;
                for (size_t k = 0; k < std::string(S).size();) {
                    unsigned char bb = static_cast<unsigned char>(S[k]);
                    if (bb < 0x80) { sv.push_back(bb); ++k; continue; }
                    uint32_t c2 = 0; size_t ex = 0;
                    if ((bb & 0xE0) == 0xC0) { c2 = bb & 0x1F; ex = 1; }
                    else if ((bb & 0xF0) == 0xE0) { c2 = bb & 0x0F; ex = 2; }
                    else { c2 = bb & 0x07; ex = 3; }
                    for (size_t q = 1; q <= ex; ++q)
                        c2 = (c2 << 6) |
                             (static_cast<unsigned char>(S[k + q]) & 0x3F);
                    k += ex + 1;
                    sv.push_back(static_cast<char32_t>(c2));
                }
                return sv.find(static_cast<char32_t>(cp)) !=
                       std::u32string::npos;
            };
            for (auto& s : r.sentences) {
                expectStr(std::string("sentence ends with splitor [") +
                              tc.what + "]",
                          endsWithSplitor(s) ? "ok" : "bad", "ok");
            }
        }

        // known golden: 你好，世界。
        {
            TextFrontend::Result r;
            if (!tf.process("你好，世界。", &r, 1)) {
                std::printf("FAIL known golden run: %s\n", r.error.c_str());
                ++gFailed;
            } else {
                // get_first("你好")==len 2 <4 => the frontend prepends 。
                // exactly like TextPreprocessor.pre_seg_text
                const int want[] = {3,   227, 167, 158, 119, 1,
                                    251, 214, 221, 194, 3};
                expectSize("known phones len", r.phones.size(), 11);
                for (size_t k = 0; k < 11 && k < r.phones.size(); ++k)
                    expectLong("known phone[" + std::to_string(k) + "]",
                               r.phones[k], want[k]);
                expectSize("known word2ph len", r.word2ph.size(), 7);
                expectSize("known sentences", r.sentences.size(), 1);
                expectStr("known segment", r.sentences[0],
                          "\xe3\x80\x82\xe4\xbd\xa0\xe5\xa5\xbd"
                          "\xef\xbc\x8c\xe4\xb8\x96\xe7\x95\x8c\xe3\x80\x82");
            }
        }

        // resolver injection: a custom PinyinResolver must be reachable via
        // setResolver without changing call sites. We verify the seam by
        // injecting the built-in resolver object itself (behaviour
        // identical) — B6 will pass its own implementation.
        {
            PypinyinResolver builtin;
            if (!builtin.load(argv[2], &err)) {
                std::printf("FAIL pinyin reload: %s\n", err.c_str());
                ++gFailed;
            } else {
                tf.setResolver(&builtin);
                TextFrontend::Result r;
                if (!tf.process("你好，世界。", &r, 1)) {
                    std::printf("FAIL injected run: %s\n", r.error.c_str());
                    ++gFailed;
                } else {
                    expectSize("injected phones len", r.phones.size(), 11);
                }
                tf.setResolver(nullptr);  // restore default
                TextFrontend::Result r2;
                if (!tf.process("你好，世界。", &r2, 1)) {
                    std::printf("FAIL default-restore run\n");
                    ++gFailed;
                } else {
                    expectSize("restored phones len", r2.phones.size(), 11);
                }
            }
        }
    }

    if (gFailed == 0) {
        std::printf("ALL TEXTFRONT UNIT TESTS PASSED\n");
        return 0;
    }
    std::printf("%d FAILURES\n", gFailed);
    return 1;
}
