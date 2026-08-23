// pinyin.cpp — see pinyin.h for the contract and provenance of each piece.
// Faithful port of pypinyin 0.55: seg/simpleseg + seg/mmseg + converter +
// contrib/tone_convert + contrib/neutral_tone + standard.convert_finals.
#include "pinyin.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace gsv::textfront {
namespace {

// ---------------------------------------------------------------------------
// UTF-8 helpers (strict decode; inputs are valid UTF-8)
// ---------------------------------------------------------------------------

void decodeUtf8(std::string_view s, std::u32string* out) {
    out->clear();
    size_t i = 0;
    const size_t n = s.size();
    while (i < n) {
        unsigned char b0 = static_cast<unsigned char>(s[i]);
        if (b0 < 0x80) {
            out->push_back(static_cast<char32_t>(b0));
            i += 1;
            continue;
        }
        uint32_t cp = 0;
        size_t extra = 0;
        if ((b0 & 0xE0) == 0xC0) { cp = b0 & 0x1Fu; extra = 1; }
        else if ((b0 & 0xF0) == 0xE0) { cp = b0 & 0x0Fu; extra = 2; }
        else if ((b0 & 0xF8) == 0xF0) { cp = b0 & 0x07u; extra = 3; }
        if (extra > 0 && i + extra < n) {
            bool ok = true;
            for (size_t k = 1; k <= extra; ++k) {
                unsigned char bk = static_cast<unsigned char>(s[i + k]);
                if ((bk & 0xC0) != 0x80) { ok = false; break; }
                cp = (cp << 6) | (bk & 0x3Fu);
            }
            if (ok) {
                out->push_back(static_cast<char32_t>(cp));
                i += extra + 1;
                continue;
            }
        }
        out->push_back(U'\uFFFD');
        i += 1;
    }
}

void encodeUtf8(const std::u32string& cps, std::string* out) {
    out->clear();
    for (char32_t c : cps) {
        if (c < 0x80) {
            *out += static_cast<char>(c);
        } else if (c < 0x800) {
            *out += static_cast<char>(0xC0 | (c >> 6));
            *out += static_cast<char>(0x80 | (c & 0x3F));
        } else if (c < 0x10000) {
            *out += static_cast<char>(0xE0 | (c >> 12));
            *out += static_cast<char>(0x80 | ((c >> 6) & 0x3F));
            *out += static_cast<char>(0x80 | (c & 0x3F));
        } else {
            *out += static_cast<char>(0xF0 | (c >> 18));
            *out += static_cast<char>(0x80 | ((c >> 12) & 0x3F));
            *out += static_cast<char>(0x80 | ((c >> 6) & 0x3F));
            *out += static_cast<char>(0x80 | (c & 0x3F));
        }
    }
}

// ---------------------------------------------------------------------------
// pypinyin constant tables (style/_constants.py, phonetic_symbol.py,
// standard.py) — algorithmic constants, hardcoded verbatim.
// ---------------------------------------------------------------------------

// _INITIALS (order matters: two-letter initials precede their first letter)
const char* const kInitials[] = {"b", "p", "m", "f", "d", "t", "n", "l",
                                 "g", "k", "h", "j", "q", "x", "zh", "ch",
                                 "sh", "r", "z", "c", "s"};
constexpr size_t kInitialsLen = sizeof(kInitials) / sizeof(kInitials[0]);

// _FINALS (ü encoded as U+00FC = \xC3\xBC, ê as U+00EA)
const char* const kFinals[] = {
    "i", "u", "\xC3\xBC", "a", "ia", "ua", "o", "uo", "e", "ie",
    "\xC3\xBC" "e", "ai", "uai", "ei", "uei", "ao", "iao", "ou", "iou",
    "an", "ian", "uan", "\xC3\xBC" "an", "en", "in", "uen", "\xC3\xBC" "n",
    "ang", "iang", "uang", "eng", "ing", "ueng", "ong", "iong",
    "er", "\xC3\xAA"};
constexpr size_t kFinalsLen = sizeof(kFinals) / sizeof(kFinals[0]);

bool inFinals(const std::u32string& s) {
    for (size_t i = 0; i < kFinalsLen; ++i) {
        std::u32string lit;
        decodeUtf8(kFinals[i], &lit);
        if (s == lit) return true;
    }
    return false;
}

// RE_HANS character class (constants.py, SUPPORT_UCS4 branch)
struct Range { uint32_t lo, hi; };
constexpr Range kHansRanges[] = {
    {0x3007, 0x3007}, {0xE815, 0xE864}, {0xFA18, 0xFA18},
    {0x3400, 0x4DBF}, {0x4E00, 0x9FFF}, {0xF900, 0xFAFF},
    {0x20000, 0x2A6DF}, {0x2A703, 0x2B73F}, {0x2B740, 0x2B81D},
    {0x2B825, 0x2BF6E}, {0x2C029, 0x2CE93}, {0x2D016, 0x2D016},
    {0x2D11B, 0x2EBD9}, {0x2F80A, 0x2FA1F}, {0x30000, 0x3134A},
    {0x300F7, 0x31288}, {0x30EDD, 0x30EDE}, {0x31350, 0x32389},
};

bool isHansCp(uint32_t c) {
    for (const auto& r : kHansRanges)
        if (c >= r.lo && c <= r.hi) return true;
    return false;
}

// phonetic_symbol.py single-char keys: marked codepoint -> letter+digit cps
struct SymCp {
    uint32_t marked;
    const char* numbered;  // ASCII
};
const SymCp kSymSingle[] = {
    {0x0101, "a1"}, {0x00E1, "a2"}, {0x01CE, "a3"}, {0x00E0, "a4"},
    {0x0113, "e1"}, {0x00E9, "e2"}, {0x011B, "e3"}, {0x00E8, "e4"},
    {0x014D, "o1"}, {0x00F3, "o2"}, {0x01D2, "o3"}, {0x00F2, "o4"},
    {0x012B, "i1"}, {0x00ED, "i2"}, {0x01D0, "i3"}, {0x00EC, "i4"},
    {0x016B, "u1"}, {0x00FA, "u2"}, {0x01D4, "u3"}, {0x00F9, "u4"},
    {0x00FC, "v"},
    {0x01D6, "v1"}, {0x01D8, "v2"}, {0x01DA, "v3"}, {0x01DC, "v4"},
    {0x0144, "n2"}, {0x0148, "n3"}, {0x01F9, "n4"},
    {0x1E3F, "m2"},                       // ḿ (single char!)
    {0x1EBF, "\xC3\xAA" "2"},             // ế -> ê2
    {0x1EC1, "\xC3\xAA" "4"},             // ề -> ê4
};
// multi-char keys applied afterwards, source order ("m̄","m̀","ê̄","ê̌")
}  // namespace

// multi-char phonetic symbol keys live here to keep the table near use-site
namespace {
struct SymEntryMulti_t {
    const char* marked;
    const char* numbered;
};
const SymEntryMulti_t kSymMulti[] = {
    {"m\xCC\x84", "m1"},                    // m + U+0304
    {"m\xCC\x80", "m4"},                    // m + U+0300
    {"\xC3\xAA\xCC\x84", "\xC3\xAA" "1"},   // ê + U+0304 -> ê1
    {"\xC3\xAA\xCC\x8C", "\xC3\xAA" "3"},   // ê + U+030C -> ê3
};
}  // namespace

namespace {

void replaceSymbolToNumber(const std::u32string& in, std::u32string* out) {
    // pass 1: single-char keys via direct mapping
    out->clear();
    for (char32_t c : in) {
        bool hit = false;
        for (const auto& e : kSymSingle) {
            if (c == e.marked) {
                for (const char* q = e.numbered; *q; ++q) out->push_back(*q);
                hit = true;
                break;
            }
        }
        if (!hit) out->push_back(c);
    }
    // pass 2: multi-char keys via substring replacement, dict order
    for (const auto& e : kSymMulti) {
        std::u32string pat, rep;
        decodeUtf8(e.marked, &pat);
        decodeUtf8(e.numbered, &rep);
        if (pat.empty()) continue;
        for (size_t i = 0; i + pat.size() <= out->size();) {
            if (std::equal(pat.begin(), pat.end(), out->begin() + i)) {
                out->replace(out->begin() + static_cast<long>(i), out->begin() + static_cast<long>(i + pat.size()), rep.begin(), rep.end());
                i += rep.size();
            } else {
                ++i;
            }
        }
    }
}

// standard.py convert_zero_consonant / convert_uv / convert_iou /
// convert_uei / convert_uen operating on tone-mark-free strings
bool isAsciiLower(uint32_t c) { return c >= 'a' && c <= 'z'; }

bool endsWithTailRun(const std::u32string& s, const char* tail,
                     size_t* headLen) {
    std::u32string t;
    decodeUtf8(tail, &t);
    if (s.size() < t.size() + 1) return false;
    size_t base = s.size() - t.size();
    for (size_t i = 0; i < t.size(); ++i)
        if (s[base + i] != t[i]) return false;
    for (size_t i = 0; i < base; ++i)
        if (!isAsciiLower(s[i])) return false;
    *headLen = base;
    return true;
}

void convertFinals(const std::u32string& in, std::u32string* out) {
    std::u32string p = in;
    const std::u32string raw = in;
    // zero consonant: y/w prefixes
    if (!raw.empty() && raw[0] == U'y') {
        std::u32string noY = p.substr(1);
        char32_t first = noY.empty() ? 0 : noY[0];
        if (first == U'u' || first == 0x016B || first == 0xFA ||
            first == 0x1D4 || first == 0xF9) {
            p.clear();
            p.push_back(0xFC);  // ü
            p.insert(p.end(), noY.begin() + 1, noY.end());
        } else if (first == U'i' || first == 0x12B || first == 0xED ||
                   first == 0x1D3 || first == 0xEC) {
            p = noY;
        } else {
            p.clear();
            p.push_back(U'i');
            p.insert(p.end(), noY.begin(), noY.end());
        }
    } else if (!raw.empty() && raw[0] == U'w') {
        std::u32string noW = p.substr(1);
        char32_t first = noW.empty() ? 0 : noW[0];
        if (first == U'u' || first == 0x16B || first == 0xFA ||
            first == 0x1D4 || first == 0xF9) {
            p = noW;
        } else {
            p.clear();
            p.push_back(U'u');
            p.insert(p.end(), noW.begin(), noW.end());
        }
    }
    // zero-consonant step reverts to the raw input when its result is not a
    // valid final (standard.py: `if pinyin not in _FINALS: return raw_pinyin`);
    // the uv/iou/uei/uen restorations below then ALWAYS run.
    if (!inFinals(p)) p = raw;
    // convert_uv: ^(j|q|x)u(.*)$ -> j/q/x + ü + rest
    if (p.size() >= 2 && (p[0] == U'j' || p[0] == U'q' || p[0] == U'x') &&
        p[1] == U'u') {
        p[1] = 0xFC;
    }
    // convert_iou: ^([a-z]+)iu$ -> head + "iou"
    {
        size_t head = 0;
        if (endsWithTailRun(p, "iu", &head)) {
            std::u32string t(p.begin(), p.begin() + head);
            t += U'i';
            t += U'o';
            t += U'u';
            p = t;
        }
    }
    // convert_uei: ([a-z]+)ui$ -> head + "uei"
    {
        size_t head = 0;
        if (endsWithTailRun(p, "ui", &head)) {
            std::u32string t(p.begin(), p.begin() + head);
            t += U'u';
            t += U'e';
            t += U'i';
            p = t;
        }
    }
    // convert_uen: ([a-z]+)un$ -> head + "uen"
    {
        size_t head = 0;
        if (endsWithTailRun(p, "un", &head)) {
            std::u32string t(p.begin(), p.begin() + head);
            t += U'u';
            t += U'e';
            t += U'n';
            p = t;
        }
    }
    *out = p;
}

// style/_utils.py get_initials(pinyin, strict)
void getInitialsCps(const std::u32string& p, bool strict, std::u32string* out) {
    out->clear();
    const size_t n = strict ? kInitialsLen : kInitialsLen + 2;
    for (size_t i = 0; i < n; ++i) {
        const char* ini = i < kInitialsLen ? kInitials[i]
                                           : (i == kInitialsLen ? "y" : "w");
        std::u32string t;
        decodeUtf8(ini, &t);
        if (p.size() >= t.size() && std::equal(t.begin(), t.end(), p.begin())) {
            *out = t;
            return;
        }
    }
}
// style/_utils.py get_finals(pinyin_no_marks, strict=True)
void getFinalsCps(const std::u32string& p, std::u32string* out) {
    std::u32string conv;
    convertFinals(p, &conv);
    std::u32string ini;
    getInitialsCps(conv, true, &ini);
    std::u32string finals(conv.begin() + ini.size(), conv.end());
    if (!inFinals(finals)) {
        std::u32string ini2;
        getInitialsCps(conv, false, &ini2);
        std::u32string finals2(conv.begin() + ini2.size(), conv.end());
        if (inFinals(finals2)) {
            *out = finals2;
        } else {
            out->clear();
        }
        return;
    }
    *out = finals;
}

}  // namespace

// ---------------------------------------------------------------------------
// public style conversions
// ---------------------------------------------------------------------------

std::string toInitials(std::string_view pinyin) {
    std::u32string p;
    decodeUtf8(pinyin, &p);
    std::u32string ini;
    getInitialsCps(p, true, &ini);
    std::string s;
    encodeUtf8(ini, &s);
    return s;
}

std::string toFinalsTone3(std::string_view pinyin) {
    // to_finals_tone3(..., neutral_tone_with_five=False) then the
    // NeutralToneWith5Mixin post step appends '5' when there is no digit;
    // combined: finals + (existing tone digit or '5').
    std::u32string p;
    decodeUtf8(pinyin, &p);
    // p = p.replace('5', '')
    std::u32string noFive;
    for (char32_t c : p)
        if (c != U'5') noFive.push_back(c);

    std::u32string noMarks;
    replaceSymbolToNumber(noFive, &noMarks);
    std::u32string stripped;
    for (char32_t c : noMarks)
        if (!(c >= U'0' && c <= U'9')) stripped.push_back(c);
    for (char32_t& c : stripped)
        if (c == U'v') c = 0xFC;  // .replace('v', 'ü')

    std::u32string finals;
    getFinalsCps(stripped, &finals);
    if (finals.empty()) {
        return std::string();
    }
    // numbers = digits in replace_symbol_to_number(no-five pinyin)
    int digit = -1;
    {
        std::u32string numbered;
        replaceSymbolToNumber(noFive, &numbered);
        for (char32_t c : numbered) {
            if (c >= U'0' && c <= U'9') {
                digit = c - U'0';
                break;
            }
        }
    }
    char tail = digit >= 0 ? static_cast<char>('0' + digit) : '5';
    // _fix_v_u(finals, finals, v_to_u=False): replace ü with v
    for (char32_t& c : finals)
        if (c == 0xFC) c = U'v';
    std::string s;
    encodeUtf8(finals, &s);
    s.push_back(tail);
    return s;
}

// ---------------------------------------------------------------------------
// PypinyinResolver
// ---------------------------------------------------------------------------

struct PypinyinResolver::Impl {
    std::vector<uint8_t> blob;

    std::vector<uint32_t> cps;
    struct PhEntry {
        uint32_t off, len, sylOff;
        uint16_t sylCnt;
    };
    std::vector<PhEntry> phrases;
    std::vector<uint8_t> sylPool;
    struct ChEntry {
        uint32_t cp, sylOff;
        uint16_t sylCnt;
    };
    std::vector<ChEntry> chars;
    std::vector<std::pair<uint32_t, uint32_t>> t2s;
    std::vector<std::pair<uint32_t, uint32_t>> numRanges;

    int cmpKey(const char32_t* q, uint32_t qlen, uint32_t off, uint32_t len) const {
        const uint32_t* k = cps.data() + off;
        uint32_t m = qlen < len ? qlen : len;
        for (uint32_t i = 0; i < m; ++i) {
            if (q[i] != k[i]) return q[i] < k[i] ? -1 : 1;
        }
        if (qlen != len) return qlen < len ? -1 : 1;
        return 0;
    }

    int findPhrase(const char32_t* q, uint32_t qlen) const {
        size_t lo = 0, hi = phrases.size();
        while (lo < hi) {
            size_t mid = lo + (hi - lo) / 2;
            int c = cmpKey(q, qlen, phrases[mid].off, phrases[mid].len);
            if (c == 0) return static_cast<int>(mid);
            if (c < 0) hi = mid; else lo = mid + 1;
        }
        return -1;
    }

    bool prefixExists(const char32_t* q, uint32_t qlen) const {
        size_t lo = 0, hi = phrases.size();
        while (lo < hi) {
            size_t mid = lo + (hi - lo) / 2;
            const PhEntry& e = phrases[mid];
            int c = cmpKey(q, qlen, e.off, e.len);
            if (c == 0) return true;
            if (c < 0) hi = mid; else lo = mid + 1;
        }
        if (lo >= phrases.size()) return false;
        const PhEntry& e = phrases[lo];
        if (e.len < qlen) return false;
        return cmpKey(q, qlen, e.off, qlen) == 0;
    }

    int findChar(uint32_t cp) const {
        size_t lo = 0, hi = chars.size();
        while (lo < hi) {
            size_t mid = lo + (hi - lo) / 2;
            if (chars[mid].cp < cp) lo = mid + 1;
            else if (chars[mid].cp > cp) hi = mid;
            else return static_cast<int>(mid);
        }
        return -1;
    }

    void firstSyllables(uint32_t sylOff, uint16_t cnt,
                        std::vector<PinyinSyl>* out) const {
        size_t pos = sylOff;
        for (uint16_t r = 0; r < cnt; ++r) {
            uint8_t nsyl = sylPool[pos++];
            for (uint8_t k = 0; k < nsyl; ++k) {
                uint8_t nb = sylPool[pos++];
                if (k == 0) {
                    PinyinSyl syl;
                    syl.text.assign(
                        reinterpret_cast<const char*>(&sylPool[pos]), nb);
                    syl.raw = false;
                    out->push_back(std::move(syl));
                }
                pos += nb;
            }
        }
    }

// mmseg.Seg.cut over one maximal Han run (no_non_phrases=True) — Impl method
    void mmsegCutHan(const char32_t* han, size_t begin, size_t end,
                     std::vector<std::pair<size_t, size_t>>* words) const {
        size_t remain = begin;
        while (remain < end) {
            size_t lastValidLen = 0, lastValidIdx = 0;
            bool broke = false;
            for (size_t index = remain; index < end; ++index) {
                size_t wlen = index - remain + 1;
                if (prefixExists(han + remain, static_cast<uint32_t>(wlen))) {
                    if (findPhrase(han + remain,
                                   static_cast<uint32_t>(wlen)) >= 0) {
                        lastValidLen = wlen;
                        lastValidIdx = index + 1;
                    }
                } else {
                    broke = true;
                    if (lastValidLen) {
                        words->emplace_back(remain, lastValidIdx);
                        remain = lastValidIdx;
                    } else {
                        // strict mode: no valid word -> emit first character
                        words->emplace_back(remain, remain + 1);
                        remain += 1;
                    }
                    break;
                }
            }
            if (broke) continue;
            if (lastValidLen) {
                words->emplace_back(remain, lastValidIdx);
                remain = lastValidIdx;
            } else if (remain < end &&
                       findPhrase(han + remain,
                                  static_cast<uint32_t>(end - remain)) >= 0) {
                words->emplace_back(remain, end);
                break;
            } else {
                for (size_t k = remain; k < end; ++k)
                    words->emplace_back(k, k + 1);
                break;
            }
        }
    }

    void pushSingleChar(uint32_t cp, std::vector<PinyinSyl>* out) const {
        int idx = findChar(cp);
        if (idx >= 0) {
            firstSyllables(chars[idx].sylOff, chars[idx].sylCnt, out);
        } else {
            // Han character missing from PINYIN_DICT: the char itself becomes
            // the style-conversion input (to_initials/to_finals yield ""),
            // unlike non-Han runs which pass through verbatim (errors=default).
            std::u32string one(1, static_cast<char32_t>(cp));
            PinyinSyl syl;
            encodeUtf8(one, &syl.text);
            syl.raw = false;
            out->push_back(std::move(syl));
        }
    }
};

PypinyinResolver::~PypinyinResolver() {
    delete impl_;
    impl_ = nullptr;
}

bool PypinyinResolver::load(const std::string& path, std::string* err) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        if (err) *err = "cannot open " + path;
        return false;
    }
    std::fseek(f, 0, SEEK_END);
    long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (sz <= 0) {
        std::fclose(f);
        if (err) *err = "empty file " + path;
        return false;
    }
    std::vector<uint8_t> blob(static_cast<size_t>(sz));
    size_t rd = std::fread(blob.data(), 1, blob.size(), f);
    std::fclose(f);
    if (rd != blob.size()) {
        if (err) *err = "short read on " + path;
        return false;
    }
    return loadMemory(blob.data(), blob.size(), err);
}

bool PypinyinResolver::loadMemory(const void* data, size_t size, std::string* err) {
    delete impl_;
    impl_ = nullptr;
    auto fail = [err](const std::string& msg) {
        if (err) *err = msg;
        return false;
    };
    auto d = static_cast<const uint8_t*>(data);

    if (size < 24 || std::memcmp(d, "GSPPYX01", 8) != 0)
        return fail("pinyin.bin: bad magic");
    uint32_t version, nSections;
    std::memcpy(&version, d + 8, 4);
    std::memcpy(&nSections, d + 12, 4);
    if (version != 1) return fail("pinyin.bin: unsupported version");

    auto section = [&](const char* name, const uint8_t** out, size_t* nbytes) {
        char padded[8];
        std::memset(padded, 0, 8);
        size_t nl = std::strlen(name);
        if (nl > 8) return false;
        std::memcpy(padded, name, nl);
        for (uint32_t i = 0; i < nSections; ++i) {
            size_t off = 16 + 32ull * i;
            if (off + 32 > size) return false;
            if (std::memcmp(d + off, padded, 8) == 0) {
                uint64_t so, sn;
                std::memcpy(&so, d + off + 8, 8);
                std::memcpy(&sn, d + off + 16, 8);
                if (so + sn > size) return false;
                *out = d + so;
                *nbytes = static_cast<size_t>(sn);
                return true;
            }
        }
        return false;
    };

    auto rdU16 = [](const uint8_t* q) {
        uint16_t v; std::memcpy(&v, q, 2); return v;
    };
    auto rdU32 = [](const uint8_t* q) {
        uint32_t v; std::memcpy(&v, q, 4); return v;
    };

    auto impl = std::make_unique<Impl>();
    impl->blob.assign(d, d + size);

    const uint8_t* p = nullptr;
    size_t nb = 0;
    if (!section("cps", &p, &nb)) return fail("missing cps");
    {
        size_t count = nb / 4;
        impl->cps.resize(count);
        for (size_t i = 0; i < count; ++i) impl->cps[i] = rdU32(p + 4 * i);
    }
    if (!section("phrases", &p, &nb)) return fail("missing phrases");
    {
        uint32_t n = rdU32(p);
        impl->phrases.resize(n);
        const uint8_t* q = p + 4;
        for (uint32_t i = 0; i < n; ++i) {
            impl->phrases[i].off = rdU32(q); q += 4;
            impl->phrases[i].len = rdU16(q); q += 2;
            impl->phrases[i].sylOff = rdU32(q); q += 4;
            impl->phrases[i].sylCnt = rdU16(q); q += 2;
            q += 2;
        }
    }
    if (!section("sylpool", &p, &nb)) return fail("missing sylpool");
    impl->sylPool.assign(p, p + nb);
    if (!section("chars", &p, &nb)) return fail("missing chars");
    {
        uint32_t n = rdU32(p);
        impl->chars.resize(n);
        const uint8_t* q = p + 4;
        for (uint32_t i = 0; i < n; ++i) {
            impl->chars[i].cp = rdU32(q); q += 4;
            impl->chars[i].sylOff = rdU32(q); q += 4;
            impl->chars[i].sylCnt = rdU16(q); q += 2;
            q += 2;
        }
    }
    if (!section("t2s", &p, &nb)) return fail("missing t2s");
    {
        uint32_t n = rdU32(p);
        impl->t2s.resize(n);
        const uint8_t* q = p + 4;
        for (uint32_t i = 0; i < n; ++i) {
            impl->t2s[i].first = rdU32(q); q += 4;
            impl->t2s[i].second = rdU32(q); q += 4;
        }
    }
    if (!section("numrange", &p, &nb)) return fail("missing numrange");
    {
        uint32_t n = rdU32(p);
        impl->numRanges.resize(n);
        const uint8_t* q = p + 4;
        for (uint32_t i = 0; i < n; ++i) {
            impl->numRanges[i].first = rdU32(q); q += 4;
            impl->numRanges[i].second = rdU32(q); q += 4;
        }
    }

    impl_ = impl.release();
    if (err) err->clear();
    return true;
}

uint32_t PypinyinResolver::t2s(uint32_t cp) const {
    if (!impl_) return cp;
    auto& v = impl_->t2s;
    size_t lo = 0, hi = v.size();
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (v[mid].first < cp) lo = mid + 1;
        else if (v[mid].first > cp) hi = mid;
        else return v[mid].second;
    }
    return cp;
}

bool PypinyinResolver::isNumeric(uint32_t cp) const {
    if (!impl_) return cp >= '0' && cp <= '9';
    for (const auto& r : impl_->numRanges)
        if (cp >= r.first && cp <= r.second) return true;
    return false;
}

void PypinyinResolver::resolve(const char32_t* word, size_t len,
                               std::vector<PinyinSyl>* syllables) const {
    if (!impl_) throw std::runtime_error("PypinyinResolver used before load()");
    syllables->clear();

    size_t i = 0;
    while (i < len) {
        bool han = isHansCp(word[i]);
        size_t j = i + 1;
        while (j < len && isHansCp(word[j]) == han) ++j;
        if (!han) {
            PinyinSyl syl;
            std::u32string run(word + i, j - i);
            encodeUtf8(run, &syl.text);
            syl.raw = true;
            syllables->push_back(std::move(syl));
            i = j;
            continue;
        }
        std::vector<std::pair<size_t, size_t>> words;
        impl_->mmsegCutHan(word, i, j, &words);
        for (auto& w : words) {
            size_t wlen = w.second - w.first;
            int idx = impl_->findPhrase(word + w.first,
                                        static_cast<uint32_t>(wlen));
            if (idx >= 0) {
                impl_->firstSyllables(impl_->phrases[idx].sylOff,
                                      impl_->phrases[idx].sylCnt, syllables);
            } else {
                for (size_t k = w.first; k < w.second; ++k)
                    impl_->pushSingleChar(static_cast<uint32_t>(word[k]),
                                          syllables);
            }
        }
        i = j;
    }
}

void getInitialsFinals(const PinyinResolver& resolver, const char32_t* word,
                       size_t len, std::vector<std::string>* initials,
                       std::vector<std::string>* finals) {
    std::vector<PinyinSyl> syls;
    resolver.resolve(word, len, &syls);
    initials->clear();
    finals->clear();
    for (const auto& s : syls) {
        if (s.raw) {
            initials->push_back(s.text);
            finals->push_back(s.text);
        } else {
            initials->push_back(toInitials(s.text));
            finals->push_back(toFinalsTone3(s.text));
        }
    }
}

}  // namespace gsv::textfront
namespace gsv::textfront {

void PypinyinResolver::segWords(const char32_t* word, size_t len,
                                std::vector<std::u32string>* out) const {
    if (!impl_) throw std::runtime_error("PypinyinResolver used before load()");
    out->clear();
    size_t i = 0;
    while (i < len) {
        bool han = isHansCp(word[i]);
        size_t j = i + 1;
        while (j < len && isHansCp(word[j]) == han) ++j;
        if (!han) {
            out->emplace_back(word + i, j - i);
        } else {
            std::vector<std::pair<size_t, size_t>> words;
            impl_->mmsegCutHan(word, i, j, &words);
            for (auto& w : words)
                out->emplace_back(word + w.first, w.second - w.first);
        }
        i = j;
    }
}

}  // namespace gsv::textfront
