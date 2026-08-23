// zh_norm.cpp — TextNormalizer chain + chinese2 punctuation layer.
// Chain order mirrors text_normlization.py normalize_sentence exactly.
#include "zh_norm.h"

#include <algorithm>

#include "lexicon.hpp"
#include "zh_num.h"
#include "zh_tables.hpp"

namespace gsv::textfront {
namespace {

U32 dec(const std::string& s) {
    U32 o;
    for (size_t i = 0; i < s.size();) {
        unsigned char b = static_cast<unsigned char>(s[i]);
        if (b < 0x80) { o.push_back(b); ++i; continue; }
        uint32_t cp = 0; size_t extra = 0;
        if ((b & 0xE0) == 0xC0) { cp = b & 0x1F; extra = 1; }
        else if ((b & 0xF0) == 0xE0) { cp = b & 0x0F; extra = 2; }
        else { cp = b & 0x07; extra = 3; }
        for (size_t k = 1; k <= extra; ++k)
            cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3F);
        i += extra + 1;
        o.push_back(static_cast<char32_t>(cp));
    }
    return o;
}

// str.strip(): unicode whitespace both ends (practical White_Space set)
bool isPySpace(char32_t c) {
    return c == U' ' || c == U'\t' || c == U'\n' || c == U'\r' ||
           c == U'\v' || c == U'\f' || c == 0x0085 || c == 0x00A0 ||
           c == 0x1680 || (c >= 0x2000 && c <= 0x200A) || c == 0x2028 ||
           c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}

U32 stripPy(const U32& s) {
    size_t b = 0, e = s.size();
    while (b < e && isPySpace(s[b])) ++b;
    while (e > b && isPySpace(s[e - 1])) --e;
    return s.substr(b, e - b);
}

// F2H translate chain (constants.py)
char32_t f2h(char32_t c) {
    if (c >= 0xFF21 && c <= 0xFF3A) return c - 0xFF21 + U'A';
    if (c >= 0xFF41 && c <= 0xFF5A) return c - 0xFF41 + U'a';
    if (c >= 0xFF10 && c <= 0xFF19) return c - 0xFF10 + U'0';
    if (c == 0x3000) return U' ';
    return c;
}

bool inSet(const U32& set, char32_t c) {
    return set.find(c) != U32::npos;
}

}  // namespace

U32 t2sText(const U32& in) {
    U32 out;
    out.reserve(in.size());
    for (char32_t c : in) {
        uint32_t key = static_cast<uint32_t>(c);
        size_t lo = 0, hi = kT2sPairs_len;
        uint32_t repl = key;
        while (lo < hi) {
            size_t mid = lo + (hi - lo) / 2;
            if (kT2sPairs[mid].first < key) lo = mid + 1;
            else if (kT2sPairs[mid].first > key) hi = mid;
            else { repl = kT2sPairs[mid].second; break; }
        }
        out.push_back(static_cast<char32_t>(repl));
    }
    return out;
}

std::vector<U32> TextNormalizer::split(const U32& text) const {
    U32 t;
    t.reserve(text.size());
    for (char32_t c : text)
        if (c != U' ') t.push_back(c);
    // filter specials [——《》【】<>{}()（）#&@“”^_|\\]
    static const U32 specials = dec("——《》【】<>{}()（）#&@“”^_|\\");
    U32 filtered;
    for (char32_t c : t)
        if (!inSet(specials, c)) filtered.push_back(c);
    // SENTENCE_SPLITOR: ([：、，；。？！,;?!][”’]?) -> \1\n
    static const U32 splitor = dec("：、，；。？！,;?!");
    static const U32 quotes = dec("”’");
    U32 out;
    for (size_t i = 0; i < filtered.size(); ++i) {
        char32_t c = filtered[i];
        out.push_back(c);
        if (inSet(splitor, c)) {
            if (i + 1 < filtered.size() && inSet(quotes, filtered[i + 1])) {
                out.push_back(filtered[++i]);
            }
            out.push_back(U'\n');
        }
    }
    U32 strippedAll = stripPy(out);
    std::vector<U32> sentences;
    size_t st = 0;
    for (size_t k = 0; k <= strippedAll.size(); ++k) {
        if (k == strippedAll.size() || strippedAll[k] == U'\n') {
            sentences.push_back(stripPy(strippedAll.substr(st, k - st)));
            st = k + 1;
        }
    }
    // re.split never yields zero parts; mirror [''] for empty input
    if (sentences.empty()) sentences.push_back(U32());
    return sentences;
}

namespace {

// greek letter replacements of _post_replace, in source order
const std::pair<const char*, const char*> kGreek[] = {
    {"α", "阿尔法"}, {"β", "贝塔"},   {"γ", "伽玛"},   {"Γ", "伽玛"},
    {"δ", "德尔塔"}, {"Δ", "德尔塔"}, {"ε", "艾普西龙"}, {"ζ", "捷塔"},
    {"η", "依塔"},   {"θ", "西塔"},   {"Θ", "西塔"},   {"ι", "艾欧塔"},
    {"κ", "喀帕"},   {"λ", "拉姆达"}, {"Λ", "拉姆达"}, {"μ", "缪"},
    {"ν", "拗"},     {"ξ", "克西"},   {"Ξ", "克西"},   {"ο", "欧米克伦"},
    {"π", "派"},     {"Π", "派"},     {"ρ", "肉"},     {"ς", "西格玛"},
    {"Σ", "西格玛"}, {"σ", "西格玛"}, {"τ", "套"},     {"υ", "宇普西龙"},
    {"φ", "服艾"},   {"Φ", "服艾"},   {"χ", "器"},     {"ψ", "普赛"},
    {"Ψ", "普赛"},   {"ω", "欧米伽"}, {"Ω", "欧米伽"},
};

void replaceAll(U32& t, const U32& key, const U32& val) {
    size_t p;
    while ((p = t.find(key)) != U32::npos) t.replace(p, key.size(), val);
}

U32 fromCpLocal(char32_t c) { return U32(1, c); }

}  // namespace

U32 TextNormalizer::postReplace(U32 sentence) const {
    replaceAll(sentence, dec("/"), dec("每"));
    static const char* kDigits10 = "一二三四五六七八九十";
    for (uint32_t k = 0; k < 10; ++k)
        replaceAll(sentence, fromCpLocal(0x2460 + k),
                   dec(std::string(kDigits10 + k * 3, 3)));
    for (auto& gr : kGreek) replaceAll(sentence, dec(gr.first), dec(gr.second));
    replaceAll(sentence, dec("+"), dec("加"));
    replaceAll(sentence, dec("-"), dec("减"));
    replaceAll(sentence, dec("×"), dec("乘"));
    replaceAll(sentence, dec("÷"), dec("除"));
    replaceAll(sentence, dec("="), dec("等"));
    static const U32 delset = dec("-——《》【】<=>{}()（）#&@“”^_|\\");
    U32 out;
    for (char32_t c : sentence)
        if (!inSet(delset, c)) out.push_back(c);
    return out;
}

U32 TextNormalizer::normalizeSentence(const U32& sentence) const {
    U32 s = t2sText(sentence);
    for (auto& c : s) c = f2h(c);

    s = subDate(s);
    s = subDate2(s);
    s = subTimes(s);  // RANGE pass then TIME pass
    s = subToRange(s);
    s = subTemperature(s);
    s = replaceMeasure(s);
    for (int guard = 0; guard < 10000 && asmdSearch(s); ++guard)
        s = subAsmd(s);
    s = subPower(s);
    s = subFrac(s);
    s = subPercentage(s);
    s = subMobilePhone(s);
    s = subTelephone(s);
    s = subNationalUniformNumber(s);
    s = subRange(s);
    s = subInteger(s);
    s = subVersionNum(s);
    s = subDecimalNum(s);
    s = subPositiveQuantifiers(s);
    s = subDefaultNum(s);
    s = subNumber(s);
    return postReplace(std::move(s));
}

std::vector<U32> TextNormalizer::normalize(const U32& text) const {
    auto sentences = split(text);
    for (auto& sen : sentences) sen = normalizeSentence(sen);
    return sentences;
}

// ---------------------------------------------------------------------------
// chinese2 punctuation layer
// ---------------------------------------------------------------------------

U32 replacePunctuation(const U32& text) {
    U32 t = text;
    replaceAll(t, dec("嗯"), dec("恩"));
    replaceAll(t, dec("呣"), dec("母"));
    // rep_map in insertion order; keys are mutually non-overlapping except
    // none, so plain sequential scan in dict order suffices
    for (size_t k = 0; k < kRepMap_len; ++k)
        replaceAll(t, dec(kRepMap[k].first), dec(kRepMap[k].second));
    // keep [\u4e00-\u9fa5 ! ? … , . -]
    U32 out;
    for (char32_t c : t) {
        bool keep = (c >= 0x4E00 && c <= 0x9FA5) || c == U'!' || c == U'?' ||
                    c == U'…' || c == U',' || c == U'.' || c == U'-';
        if (keep) out.push_back(c);
    }
    return out;
}

U32 collapsePunct(const U32& text) {
    auto isP = [](char32_t c) {
        return c == U'!' || c == U'?' || c == U'…' || c == U',' ||
               c == U'.' || c == U'-';
    };
    U32 out;
    size_t i = 0;
    while (i < text.size()) {
        if (isP(text[i])) {
            char32_t first = text[i];
            out.push_back(first);
            while (i < text.size() && isP(text[i])) ++i;
        } else {
            out.push_back(text[i++]);
        }
    }
    return out;
}

}  // namespace gsv::textfront
