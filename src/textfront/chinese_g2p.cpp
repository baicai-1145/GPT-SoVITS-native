// chinese_g2p.cpp — see chinese_g2p.h. Line-level port of chinese2.py
// replace_punctuation / g2p / _g2p (is_g2pw=False) / _merge_erhua /
// _map_initial_final_to_phones.
#include "chinese_g2p.h"

#include <cstdio>

#include "symbols2.hpp"
#include "tone_sandhi.h"
#include "lexicon.hpp"
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

std::string enc(const U32& t) {
    std::string o;
    for (char32_t c : t) {
        if (c < 0x80) o += static_cast<char>(c);
        else {
            unsigned char b[4]; int n;
            if (c < 0x800) { n = 2; b[0] = 0xC0 | (c >> 6); }
            else if (c < 0x10000) { n = 3; b[0] = 0xE0 | (c >> 12); }
            else { n = 4; b[0] = 0xF0 | (c >> 18); }
            for (int k = 1; k < n; ++k)
                b[k] = 0x80 | ((c >> (6 * (n - k - 1))) & 0x3F);
            o.append(reinterpret_cast<char*>(b), n);
        }
    }
    return o;
}

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

bool isPunct(char32_t c) {
    return c == U'!' || c == U'?' || c == U'…' || c == U',' || c == U'.' ||
           c == U'-';
}

// re.split("(?<=[punct])\s*", text), dropping parts whose strip() is empty
std::vector<U32> splitAfterPunct(const U32& t) {
    std::vector<U32> out;
    size_t start = 0;
    for (size_t i = 0; i < t.size(); ++i) {
        if (isPunct(t[i])) {
            out.push_back(t.substr(start, i + 1 - start));
            size_t j = i + 1;
            while (j < t.size() && isPySpace(t[j])) ++j;
            start = j;
            i = j - 1;
        }
    }
    out.push_back(t.substr(start));
    std::vector<U32> kept;
    for (auto& p : out)
        if (!stripPy(p).empty()) kept.push_back(std::move(p));
    return kept;
}

// re.sub("[a-zA-Z]+", "", seg)
U32 stripAsciiLetters(const U32& seg) {
    U32 out;
    for (char32_t c : seg)
        if (!((c >= U'a' && c <= U'z') || (c >= U'A' && c <= U'Z')))
            out.push_back(c);
    return out;
}

bool u32InList(const char* const* list, size_t len, const U32& w) {
    std::string e = enc(w);
    for (size_t k = 0; k < len; ++k)
        if (e == list[k]) return true;
    return false;
}

// ---------------------------------------------------------------------------
// chinese2._merge_erhua
// ---------------------------------------------------------------------------

void mergeErhua(std::vector<std::string>* initials,
                std::vector<std::string>* finals, const U32& word,
                std::string_view pos) {
    const size_t n = finals->size();
    // fix er1
    if (n > 0 && (*finals)[n - 1] == "er1" && word[n - 1] == 0x513F /*儿*/)
        (*finals)[n - 1] = "er2";

    bool inNot = u32InList(kNotErhua, kNotErhua_len, word);
    bool inMust = u32InList(kMustErhua, kMustErhua_len, word);
    if (!inMust && (inNot || pos == "a" || pos == "j" || pos == "nr"))
        return;

    // "……" 等情况直接返回
    if (n != word.size()) return;

    // 与前一个字发同音
    std::vector<std::string> newI, newF;
    for (size_t i = 0; i < n; ++i) {
        std::string phn = (*finals)[i];
        if (i == n - 1 && word[i] == 0x513F &&
            (phn == "er2" || phn == "er5") && !newF.empty()) {
            // word[-2:] not in not_erhua
            U32 tail2 = word.size() >= 2 ? word.substr(word.size() - 2, 2)
                                         : word;
            if (!u32InList(kNotErhua, kNotErhua_len, tail2))
                phn = std::string("er") + newF.back().back();
        }
        newI.push_back((*initials)[i]);
        newF.push_back(std::move(phn));
    }
    *initials = std::move(newI);
    *finals = std::move(newF);
}

// ---------------------------------------------------------------------------
// chinese2._map_initial_final_to_phones
// ---------------------------------------------------------------------------

const OpenCpopEntry* opencpopFind(const std::string& pinyin) {
    // NOTE: opencpop-strict.txt is NOT strictly sorted (jv/juan/jue...),
    // so this must stay a linear scan in file order
    for (size_t k = 0; k < kOpenCpopMap_len; ++k)
        if (pinyin == kOpenCpopMap[k].pinyin) return &kOpenCpopMap[k];
    return nullptr;
}

bool mapInitialFinalToPhones(const std::string& c, const std::string& v,
                             std::vector<std::string>* out,
                             std::string* err) {
    if (c == v) {
        // assert c in punctuation
        bool ok = v.size() == 1 || (v == "…");
        char32_t cp = 0;
        if (!v.empty()) cp = dec(v)[0];
        if (!(v.size() <= 4 && isPunct(cp)) || !ok) {
            *err = "assert punct: '" + c + "'";
            return false;  // python AssertionError
        }
        out->push_back(v);
        return true;
    }
    if (v.empty()) {
        // python: ''[-1] raises IndexError
        *err = "empty finals";
        return false;
    }
    std::string vwt = v.substr(0, v.size() - 1);
    char tone = v.back();
    if (tone != '1' && tone != '2' && tone != '3' && tone != '4' &&
        tone != '5') {
        *err = "assert tone";
        return false;
    }
    std::string pinyin = c + vwt;
    if (!c.empty()) {
        if (vwt == "uei") pinyin = c + "ui";
        else if (vwt == "iou") pinyin = c + "iu";
        else if (vwt == "uen") pinyin = c + "un";
    } else {
        if (vwt == "ing") pinyin = "ying";
        else if (vwt == "i") pinyin = "yi";
        else if (vwt == "in") pinyin = "yin";
        else if (vwt == "u") pinyin = "wu";
        else if (!vwt.empty()) {
            char32_t f0 = vwt.front();
            if (f0 == U'v') pinyin = "yu" + vwt.substr(1);
            else if (f0 == U'e') pinyin = "e" + vwt.substr(1);
            else if (f0 == U'i') pinyin = "y" + vwt.substr(1);
            else if (f0 == U'u') pinyin = "w" + vwt.substr(1);
        } else {
            // pinyin == "" ; python: ""[0] raises IndexError
            *err = "empty pinyin";
            return false;
        }
    }
    const OpenCpopEntry* entry = opencpopFind(pinyin);
    if (!entry) {
        *err = "opencpop miss: " + pinyin;
        return false;  // python assert
    }
    // value is "<consonant> <vowel>"
    std::string joined;
    for (size_t k = 0; k < entry->n; ++k) {
        if (k) joined += " ";
        joined += entry->phones[k];
    }
    size_t sp = joined.find(' ');
    std::string newC = joined.substr(0, sp);
    std::string newV = joined.substr(sp + 1);
    newV += tone;
    out->push_back(newC);
    out->push_back(newV);
    return true;
}

}  // namespace

bool ChineseG2p::load(const std::string& triePath,
                      const std::string& pinyinPath, std::string* err) {
    if (!jb_.load(triePath, err)) return false;
    if (!resolver_.load(pinyinPath, err)) return false;
    loaded_ = true;
    return true;
}

U32 ChineseG2p::textNormalize(const U32& text) const {
    auto sentences = tn_.normalize(text);
    U32 dest;
    for (auto& sen : sentences) dest += replacePunctuation(sen);
    return collapsePunct(dest);
}

bool ChineseG2p::run(const std::string& utf8Text, G2pResult* out) const {
    out->phones.clear();
    out->word2ph.clear();
    out->ok = false;
    out->error.clear();
    if (!loaded_) {
        out->error = "not loaded";
        return false;
    }
    U32 text = dec(utf8Text);
    U32 dest = textNormalize(text);

    auto segmentsRaw = splitAfterPunct(dest);
    std::vector<U32> segments;
    for (auto& seg : segmentsRaw) segments.push_back(stripAsciiLetters(seg));

    ToneSandhi sandhi(&jb_);

    for (auto& seg : segments) {
        if (seg.empty()) continue;
        std::string segUtf8 = enc(seg);
        std::vector<PosToken> toks;
        jb_.lcut(segUtf8, &toks);
        std::vector<SegToken> cut;
        for (auto& t : toks) cut.push_back({dec(t.word), t.flag});
        sandhi.preMergeForModify(&cut, resolver_);

        for (auto& tok : cut) {
            if (tok.flag == "eng") continue;
            std::vector<std::string> initials, finals;
            getInitialsFinals(resolver_, tok.word.data(), tok.word.size(),
                              &initials, &finals);
            bool ok = true;
            sandhi.modifiedTone(tok.word, tok.flag, resolver_, &finals, &ok);
            if (!ok) {
                out->error = "modified_tone raise on '" + enc(tok.word) + "'";
                return false;  // python would raise
            }
            mergeErhua(&initials, &finals, tok.word, tok.flag);

            if (initials.size() != finals.size()) {
                out->error = "init/final mismatch";
                return false;
            }
            for (size_t k = 0; k < initials.size(); ++k) {
                std::vector<std::string> ph;
                if (!mapInitialFinalToPhones(initials[k], finals[k], &ph,
                                             &out->error)) {
                    return false;
                }
                for (auto& p : ph) {
                    int id = symbols2Id(p);
                    if (id < 0) {
                        // clean_text maps unknown phones to UNK
                        id = symbols2Id("UNK");
                    }
                    out->phones.push_back(id);
                }
                out->word2ph.push_back(static_cast<int>(ph.size()));
            }
        }
    }
    out->ok = true;
    return true;
}

}  // namespace gsv::textfront
