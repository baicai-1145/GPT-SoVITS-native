// english.cpp — port of text/english.py en_G2p (dict/letter/OOV rules).
// Neural word-segment fallback and nltk pos_tag are intentionally out of
// scope (see english.h). The zh-all path only reaches this for Latin
// segments; pairs s4/s9 and the mixed-sentence corpus never trigger the
// unsupported branches. Consumes src/textfront/data/cmudict.bin produced by
// tools/export_cmudict.py.
#include "english.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "en_norm.h"

namespace gsv::textfront {

namespace {

// binary layout (little-endian), see tools/export_cmudict.py:
//   "CMUBIN1" u32 nPhones (u8 len, bytes)...
//   u32 nWords (u8 wlen, bytes, u8 plen, plen x u8)
//   u32 nHomographs (u8 wlen, bytes, u8 p1, p1 x u8, u8 p2, p2 x u8, u8 pos)
struct Dict {
    std::vector<std::string> phones;             // id -> arpa string
    std::vector<std::string> words;              // sorted ascending
    std::vector<std::vector<uint8_t>> phonesIdx; // per word, phone ids
    std::vector<std::string> hwords;             // sorted ascending
    std::vector<std::vector<uint8_t>> hP1, hP2;
    std::vector<uint8_t> hPos;
    bool loaded = false;

    int phoneId(const std::string& p) const {
        for (size_t i = 0; i < phones.size(); ++i)
            if (phones[i] == p) return static_cast<int>(i);
        return -1;
    }
    const std::vector<uint8_t>* lookup(const std::string& w) const {
        auto it = std::lower_bound(words.begin(), words.end(), w);
        if (it != words.end() && *it == w)
            return &phonesIdx[static_cast<size_t>(it - words.begin())];
        return nullptr;
    }
    // python homograph path uses pron2 when pos==""
    const std::vector<uint8_t>* homo(const std::string& w) const {
        auto it = std::lower_bound(hwords.begin(), hwords.end(), w);
        if (it != hwords.end() && *it == w) {
            size_t i = static_cast<size_t>(it - hwords.begin());
            return &hP2[i];
        }
        return nullptr;
    }
};

Dict& sharedDict() {
    static Dict d;
    return d;
}

bool parseDict(const std::string& buf, Dict* d, std::string* err) {
    if (buf.size() < 7 || buf.compare(0, 7, "CMUBIN1") != 0) {
        *err = "bad magic";
        return false;
    }
    size_t i = 7;  // magic is the 7-byte "CMUBIN1"
    auto rdU32 = [&](uint32_t& v) -> bool {
        if (i + 4 > buf.size()) return false;
        v = static_cast<uint8_t>(buf[i]) | (static_cast<uint8_t>(buf[i + 1]) << 8) |
            (static_cast<uint8_t>(buf[i + 2]) << 16) |
            (static_cast<uint8_t>(buf[i + 3]) << 24);
        i += 4;
        return true;
    };
    auto rdU8 = [&](uint8_t& v) -> bool {
        if (i + 1 > buf.size()) return false;
        v = static_cast<uint8_t>(buf[i]);
        i += 1;
        return true;
    };
    uint32_t nPhones = 0;
    if (!rdU32(nPhones)) return false;
    d->phones.resize(nPhones);
    for (uint32_t k = 0; k < nPhones; ++k) {
        uint8_t len = 0;
        if (!rdU8(len)) return false;
        if (i + len > buf.size()) return false;
        d->phones[k].assign(buf, i, len);
        i += len;
    }
    uint32_t nWords = 0;
    if (!rdU32(nWords)) return false;
    d->words.resize(nWords);
    d->phonesIdx.resize(nWords);
    std::string prev;
    for (uint32_t k = 0; k < nWords; ++k) {
        uint8_t wl = 0;
        if (!rdU8(wl)) return false;
        if (i + wl > buf.size()) return false;
        std::string w(buf, i, wl);
        i += wl;
        uint8_t pl = 0;
        if (!rdU8(pl)) return false;
        if (i + pl > buf.size()) return false;
        std::vector<uint8_t> ids(
            reinterpret_cast<const uint8_t*>(buf.data()) + i,
            reinterpret_cast<const uint8_t*>(buf.data()) + i + pl);
        i += pl;
        if (!prev.empty() && !(prev < w)) {
            *err = "cmudict not sorted at " + w;
            return false;
        }
        prev = w;
        d->words[k] = w;
        d->phonesIdx[k] = std::move(ids);
    }
    uint32_t nH = 0;
    if (!rdU32(nH)) return false;
    d->hwords.resize(nH);
    d->hP1.resize(nH);
    d->hP2.resize(nH);
    d->hPos.resize(nH);
    std::string prevh;
    for (uint32_t k = 0; k < nH; ++k) {
        uint8_t wl = 0;
        if (!rdU8(wl)) return false;
        std::string w(buf, i, wl);
        i += wl;
        uint8_t p1 = 0;
        if (!rdU8(p1)) return false;
        std::vector<uint8_t> h1(reinterpret_cast<const uint8_t*>(buf.data()) + i,
                                reinterpret_cast<const uint8_t*>(buf.data()) + i + p1);
        i += p1;
        uint8_t p2 = 0;
        if (!rdU8(p2)) return false;
        std::vector<uint8_t> h2(reinterpret_cast<const uint8_t*>(buf.data()) + i,
                                reinterpret_cast<const uint8_t*>(buf.data()) + i + p2);
        i += p2;
        uint8_t pos = 0;
        if (!rdU8(pos)) return false;
        if (!prevh.empty() && !(prevh < w)) {
            *err = "homographs not sorted at " + w;
            return false;
        }
        prevh = w;
        d->hwords[k] = w;
        d->hP1[k] = std::move(h1);
        d->hP2[k] = std::move(h2);
        d->hPos[k] = pos;
    }
    return true;
}

bool hasLetter(const std::string& w) {
    for (char c : w)
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) return true;
    return false;
}

// simple_word_tokenize: [A-Za-z]+(?:'[A-Za-z]+)? | [.,?!\-]
std::vector<std::string> tokenize(const std::string& text) {
    std::vector<std::string> out;
    size_t i = 0;
    auto isAlpha = [](char c) {
        return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
    };
    while (i < text.size()) {
        char c = text[i];
        if (isAlpha(c)) {
            size_t start = i;
            while (i < text.size() && isAlpha(text[i])) ++i;
            if (i < text.size() && text[i] == '\'' && i + 1 < text.size() &&
                isAlpha(text[i + 1])) {
                ++i;
                while (i < text.size() && isAlpha(text[i])) ++i;
            }
            out.push_back(text.substr(start, i - start));
        } else if (c == '.' || c == ',' || c == '?' || c == '!' || c == '-') {
            out.push_back(std::string(1, c));
            ++i;
        } else {
            ++i;  // whitespace / other skipped by tokenizer
        }
    }
    return out;
}

void pushPron(const Dict& d, const std::vector<uint8_t>* p,
              std::vector<std::string>& out) {
    if (!p) return;
    for (uint8_t id : *p) {
        if (id >= d.phones.size()) continue;
        std::string ph = d.phones[id];
        if (ph == "<pad>" || ph == "</s>" || ph == "<s>" || ph == " " ||
            ph == "UW" || ph == "<unk>") {
            if (ph == "<unk>") out.push_back("UNK");
            continue;
        }
        if (ph == "'") ph = "-";
        out.push_back(ph);
    }
}

}  // namespace

// english.py text_normalize: rep_map then expend.normalize then collapse.
// BUG-COMPAT: rep_map's first two keys are REGEX CHARACTER CLASSES used as
// literal dict keys ("[;:\uff0c\uff1b]" and "["\u2019]"); after re.escape they
// only match those exact bracket strings in input, i.e. NEVER in normal
// text. Only the three plain single-char entries (\u3002 \uff01 \uff1f)
// ever fire. Everything else (ASCII ;:, fullwidth variants, curly quotes)
// passes through unmapped and is later DROPPED by expend.normalize's
// charset filter [^ A-Za-z'.,?!-]. Reproduced verbatim: map only the three,
// let the charset pass delete the rest.
static std::string normalizeText(const std::string& utf8) {
    static const std::vector<std::pair<std::string, std::string>> reps = {
        {"\xe3\x80\x82", "."},  // 。
        {"\xef\xbc\x81", "!"},  // ！
        {"\xef\xbc\x9f", "?"},  // ？
    };
    std::string t;
    size_t i = 0;
    while (i < utf8.size()) {
        bool hit = false;
        for (const auto& kv : reps) {
            if (utf8.compare(i, kv.first.size(), kv.first) == 0) {
                t += kv.second;
                i += kv.first.size();
                hit = true;
                break;
            }
        }
        if (!hit) t += utf8[i++];
    }
    t = en::normalize(t);
    // replace_consecutive_punctuation: ([punct\s])([punct])+ -> first
    auto isPunct = [](char c) {
        return c == '!' || c == '?' || c == ',' || c == '.' || c == '-';
    };
    auto isWs = [](char c) {
        return c == ' ' || c == '\t' || c == '\n' || c == '\r';
    };
    std::string o;
    size_t j = 0;
    while (j < t.size()) {
        o += t[j];
        if (isPunct(t[j]) || isWs(t[j])) {
            size_t k2 = j + 1;
            while (k2 < t.size() && isPunct(t[k2])) ++k2;
            j = k2;
        } else {
            ++j;
        }
    }
    return o;
}

bool EnglishG2p::load(const std::string& cmudictPath, std::string* err) {
    FILE* f = fopen(cmudictPath.c_str(), "rb");
    if (!f) {
        *err = "cannot open cmudict.bin";
        return false;
    }
    std::string buf;
    char tmp[1 << 16];
    size_t n;
    while ((n = fread(tmp, 1, sizeof(tmp), f)) > 0) buf.append(tmp, n);
    fclose(f);
    Dict& d = sharedDict();
    if (!parseDict(buf, &d, err)) return false;
    d.loaded = true;
    loaded_ = true;
    return true;
}

std::vector<std::string> EnglishG2p::g2p(const std::string& text,
                                         std::vector<int>* perTokenLens) const {
    std::vector<std::string> out;
    if (perTokenLens) perTokenLens->clear();
    Dict& d = sharedDict();
    if (!d.loaded) return out;

    std::string norm = normalizeText(text);
    std::vector<std::string> toks = tokenize(norm);
    for (auto& o_word : toks) {
        std::string w = o_word;
        for (char& c : w) c = static_cast<char>(tolower(c));
        std::vector<std::string> pron;
        if (!hasLetter(w)) {
            pron.push_back(o_word);  // punctuation token verbatim
        } else if (o_word.size() == 1) {
            if (o_word == "A") {
                pron.push_back("EY1");
            } else {
                const std::vector<uint8_t>* p = d.lookup(w);
                pushPron(d, p, pron);
            }
        } else {
            const std::vector<uint8_t>* h = d.homo(w);
            if (h) {
                pushPron(d, h, pron);
            } else {
                const std::vector<uint8_t>* p = d.lookup(w);
                if (p) {
                    pushPron(d, p, pron);
                } else if (o_word.size() <= 3) {
                    // spell out each letter
                    for (char c : w) {
                        std::string lc(1, c);
                        const std::vector<uint8_t>* lp = d.lookup(lc);
                        pushPron(d, lp, pron);
                    }
                } else {
                    // OOV >3 letters: wordsegment/neural fallback not ported.
                    // Spell out letters so we never crash; excluded from the
                    // acceptance corpus.
                    for (char c : w) {
                        std::string lc(1, c);
                        const std::vector<uint8_t>* lp = d.lookup(lc);
                        pushPron(d, lp, pron);
                    }
                }
            }
        }
        // normalize_pronunciation + replace_phs; the phone_units flatten
        // path inserts NO separator between tokens (gap units have no
        // phones), so nothing is added here either.
        size_t before = out.size();
        for (auto& ph : pron) {
            if (ph == "<pad>" || ph == "</s>" || ph == "<s>" || ph == "UW")
                continue;
            out.push_back(ph == "<unk>" ? "UNK" : ph);
        }
        if (perTokenLens)
            perTokenLens->push_back(static_cast<int>(out.size() - before));
    }
    return out;
}

}  // namespace gsv::textfront
