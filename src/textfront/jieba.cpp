// jieba.cpp — native port of jieba_fast.posseg.lcut() (HMM=True default).
//
// Mirrors, function by function, the jieba_fast sources:
//   Tokenizer.get_DAG / Tokenizer.calc      -> buildDag / computeRoute
//   POSTokenizer.__cut_DAG                  -> cutDag
//   POSTokenizer.__cut_internal             -> lcut (block splitting)
//   POSTokenizer.__cut_detail               -> cutDetail
//   POSTokenizer.__cut + posseg.viterbi     -> hmmCut / viterbi
//
// Binary container layout ("GSVJTB01") is documented in
// tools/export_jieba_trie.py; every numeric value in the file is the exact
// f64/i64 from the python objects (struct.pack round-trip).

#include "jieba.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace gsv::textfront {
namespace {

constexpr uint16_t kTagNone = 0xFFFF;
constexpr double kMinFloat = -3.14e100;  // jieba finalseg/posseg MIN_FLOAT
constexpr double kMinInf = -std::numeric_limits<double>::infinity();
constexpr uint32_t kInvalidState = 0xFFFF;

// ---------------------------------------------------------------------------
// UTF-8
// ---------------------------------------------------------------------------

void decodeUtf8(std::string_view s, std::u32string* out) {
    out->clear();
    out->reserve(s.size());
    size_t i = 0, n = s.size();
    while (i < n) {
        unsigned char b0 = static_cast<unsigned char>(s[i]);
        if (b0 < 0x80) {
            out->push_back(static_cast<char32_t>(b0));
            i += 1;
            continue;
        }
        uint32_t cp = 0;
        size_t extra = 0;
        if ((b0 & 0xE0) == 0xC0) {
            cp = b0 & 0x1Fu;
            extra = 1;
        } else if ((b0 & 0xF0) == 0xE0) {
            cp = b0 & 0x0Fu;
            extra = 2;
        } else if ((b0 & 0xF8) == 0xF0) {
            cp = b0 & 0x07u;
            extra = 3;
        }
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
        // Invalid byte: replacement char, advance one byte. Golden inputs are
        // valid UTF-8 so this path is never exercised by acceptance tests.
        out->push_back(U'\uFFFD');
        i += 1;
    }
}

void encodeUtf8(const char32_t* s, size_t len, std::string* out) {
    for (size_t i = 0; i < len; ++i) {
        char32_t c = s[i];
        if (c < 0x80) {
            out->push_back(static_cast<char>(c));
        } else if (c < 0x800) {
            out->push_back(static_cast<char>(0xC0 | (c >> 6)));
            out->push_back(static_cast<char>(0x80 | (c & 0x3F)));
        } else if (c < 0x10000) {
            out->push_back(static_cast<char>(0xE0 | (c >> 12)));
            out->push_back(static_cast<char>(0x80 | ((c >> 6) & 0x3F)));
            out->push_back(static_cast<char>(0x80 | (c & 0x3F)));
        } else {
            out->push_back(static_cast<char>(0xF0 | (c >> 18)));
            out->push_back(static_cast<char>(0x80 | ((c >> 12) & 0x3F)));
            out->push_back(static_cast<char>(0x80 | ((c >> 6) & 0x3F)));
            out->push_back(static_cast<char>(0x80 | (c & 0x3F)));
        }
    }
}

// ---------------------------------------------------------------------------
// Python regex character classes used by jieba_fast.posseg (str patterns)
// ---------------------------------------------------------------------------

inline bool isAsciiDigit(uint32_t c) { return c >= '0' && c <= '9'; }
inline bool isAsciiAlpha(uint32_t c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}
inline bool isAsciiAlnum(uint32_t c) { return isAsciiDigit(c) || isAsciiAlpha(c); }

// re_han_internal = [\u4E00-\u9FD5a-zA-Z0-9+#&\._]
inline bool isHanInternal(uint32_t c) {
    if (c >= 0x4E00 && c <= 0x9FD5) return true;
    if (isAsciiAlnum(c)) return true;
    return c == '+' || c == '#' || c == '&' || c == '.' || c == '_';
}

// re_han_detail = [\u4E00-\u9FD5]
inline bool isHanDetail(uint32_t c) { return c >= 0x4E00 && c <= 0x9FD5; }

// re_num = [\.0-9]
inline bool isNumChar(uint32_t c) { return c == '.' || isAsciiDigit(c); }

// python \s for str patterns (verified empirically against CPython re)
inline bool isPySpace(uint32_t c) {
    switch (c) {
        case '\t': case '\n': case 0x0B: case '\f': case '\r':
        case 0x1C: case 0x1D: case 0x1E: case 0x1F:
        case 0x85: case 0xA0: case 0x1680:
        case 0x2028: case 0x2029: case 0x202F: case 0x205F: case 0x3000:
            return true;
        default:
            return c >= 0x2000 && c <= 0x200A;
    }
}

// ---------------------------------------------------------------------------
// Little-endian readers with bounds checks
// ---------------------------------------------------------------------------

class ByteReader {
public:
    ByteReader(const uint8_t* data, size_t size) : d_(data), n_(size) {}

    void seek(size_t off) { check(off, 0); pos_ = off; }
    size_t pos() const { return pos_; }

    uint8_t u8() { check(pos_, 1); return d_[pos_++]; }

    uint16_t u16() { check(pos_, 2); uint16_t v; std::memcpy(&v, d_ + pos_, 2); pos_ += 2; return v; }
    uint32_t u32() { check(pos_, 4); uint32_t v; std::memcpy(&v, d_ + pos_, 4); pos_ += 4; return v; }
    uint64_t u64() { check(pos_, 8); uint64_t v; std::memcpy(&v, d_ + pos_, 8); pos_ += 8; return v; }
    int64_t i64() { return static_cast<int64_t>(u64()); }
    double f64() { check(pos_, 8); double v; std::memcpy(&v, d_ + pos_, 8); pos_ += 8; return v; }

private:
    void check(size_t off, size_t bytes) const {
        if (off + bytes > n_) throw std::runtime_error("jieba_trie: truncated section");
    }
    const uint8_t* d_;
    size_t n_;
    size_t pos_ = 0;
};

}  // namespace

// ---------------------------------------------------------------------------
// Impl
// ---------------------------------------------------------------------------

struct JiebaSegmenter::Impl {
    std::vector<uint8_t> blob;  // owns file bytes

    // dictionary trie (sorted compact prefix tree)
    std::vector<char32_t> cps;      // concatenated keys
    struct Entry {
        uint32_t off;
        uint32_t len;
        uint16_t tag;    // kTagNone for prefix-only nodes
        int64_t freq;    // 0 => prefix-only (python FREQ falsy)
    };
    std::vector<Entry> entries;     // sorted lexicographically by key
    int64_t total = 0;
    std::vector<std::string> tags;

    // POS HMM
    uint32_t S = 0;
    std::vector<uint8_t> st_bmes;   // 'B'|'M'|'E'|'S'
    std::vector<uint16_t> st_tag;
    std::vector<double> start_p;
    std::vector<uint32_t> trans_off;  // S+1
    std::vector<uint16_t> trans_dst;
    std::vector<double> trans_p;
    std::vector<uint32_t> emit_off;   // S+1
    std::vector<uint32_t> emit_cp;
    std::vector<double> emit_p;
    std::vector<uint32_t> cs_cp;      // sorted
    std::vector<uint32_t> cs_first;
    std::vector<uint32_t> cs_cnt;
    std::vector<uint16_t> cs_list;

    // finalseg (plain Tokenizer.cut / cut_for_search), states "BEMS"
    bool hasFinalseg = false;
    double fstart[4] = {0, 0, 0, 0};
    uint32_t ftransOff[5] = {0, 0, 0, 0, 0};
    std::vector<uint16_t> ftransDst;
    std::vector<double> ftransP;
    uint32_t femitOff[5] = {0, 0, 0, 0, 0};
    std::vector<uint32_t> femitCp;
    std::vector<double> femitP;

    // ---- dictionary lookups -------------------------------------------

    int cmpKey(const char32_t* q, uint32_t qlen, const Entry& e) const {
        const char32_t* k = cps.data() + e.off;
        uint32_t m = qlen < e.len ? qlen : e.len;
        for (uint32_t i = 0; i < m; ++i) {
            if (q[i] != k[i]) return q[i] < k[i] ? -1 : 1;
        }
        if (qlen != e.len) return qlen < e.len ? -1 : 1;
        return 0;
    }

    // index of exact match, or -1
    int findExact(const char32_t* q, uint32_t qlen) const {
        if (entries.empty()) return -1;
        size_t lo = 0, hi = entries.size();
        while (lo < hi) {
            size_t mid = lo + (hi - lo) / 2;
            int c = cmpKey(q, qlen, entries[mid]);
            if (c == 0) {
                // binary search may land mid-run only for duplicate keys,
                // which cannot happen (keys unique)
                return static_cast<int>(mid);
            }
            if (c < 0) hi = mid; else lo = mid + 1;
        }
        return -1;
    }

    // FREQ.get(key) > 0 ?
    bool freqPositive(const char32_t* q, uint32_t qlen) const {
        int idx = findExact(q, qlen);
        return idx >= 0 && entries[idx].freq > 0;
    }

    // FREQ membership (words and zero-freq prefixes alike)
    bool member(const char32_t* q, uint32_t qlen) const {
        return findExact(q, qlen) >= 0;
    }

    const char* tagOf(const char32_t* q, uint32_t qlen) const {
        int idx = findExact(q, qlen);
        if (idx >= 0 && entries[idx].tag != kTagNone) {
            return tags[entries[idx].tag].c_str();
        }
        static const char kX[] = "x";
        return kX;  // word_tag_tab.get(word, 'x')
    }

    // ---- HMM lookups ----------------------------------------------------

    const uint16_t* charStates(uint32_t cp, uint32_t* cnt) const {
        size_t lo = 0, hi = cs_cp.size();
        while (lo < hi) {
            size_t mid = lo + (hi - lo) / 2;
            if (cs_cp[mid] < cp) lo = mid + 1; else hi = mid;
        }
        if (lo < cs_cp.size() && cs_cp[lo] == cp) {
            *cnt = cs_cnt[lo];
            return cs_list.data() + cs_first[lo];
        }
        *cnt = 0;
        return nullptr;
    }

    double emitProb(uint32_t y, uint32_t cp) const {
        uint32_t b = emit_off[y], e = emit_off[y + 1];
        // binary search emit_cp[b,e) (sorted by exporter)
        while (b < e) {
            uint32_t mid = b + (e - b) / 2;
            if (emit_cp[mid] < cp) b = mid + 1;
            else if (emit_cp[mid] > cp) e = mid;
            else return emit_p[mid];
        }
        return kMinFloat;  // missing emission
    }

    double transProb(uint32_t y0, uint32_t y) const {
        uint32_t b = trans_off[y0], e = trans_off[y0 + 1];
        for (uint32_t i = b; i < e; ++i) {
            if (trans_dst[i] == y) return trans_p[i];
        }
        return kMinInf;  // trans_p[y0].get(y, MIN_INF)
    }

    // ---- finalseg lookups (missing -> MIN_FLOAT everywhere) -------------

    double ftransProb(uint32_t y0, uint32_t y) const {
        for (uint32_t i = ftransOff[y0]; i < ftransOff[y0 + 1]; ++i) {
            if (ftransDst[i] == y) return ftransP[i];
        }
        return kMinFloat;
    }

    double femitProb(uint32_t y, uint32_t cp) const {
        uint32_t b = femitOff[y], e = femitOff[y + 1];
        while (b < e) {
            uint32_t mid = b + (e - b) / 2;
            if (femitCp[mid] < cp) b = mid + 1;
            else if (femitCp[mid] > cp) e = mid;
            else return femitP[mid];
        }
        return kMinFloat;
    }

    // ---- segmentation core ------------------------------------------------

    // Tokenizer.get_DAG
    void buildDag(const char32_t* sen, uint32_t n,
                  std::vector<std::vector<uint32_t>>* dag) const {
        dag->assign(n, {});
        for (uint32_t k = 0; k < n; ++k) {
            auto& lst = (*dag)[k];
            uint32_t i = k;
            while (i < n && member(sen + k, i - k + 1)) {
                if (freqPositive(sen + k, i - k + 1)) lst.push_back(i);
                ++i;
            }
            if (lst.empty()) lst.push_back(k);
        }
    }

    struct RouteCell {
        double prob;
        uint32_t end;
    };

    // Tokenizer.calc — max over tuples (score, x); ties take larger x.
    void computeRoute(const char32_t* sen, uint32_t n,
                      const std::vector<std::vector<uint32_t>>& dag,
                      std::vector<RouteCell>* route) const {
        const double logtotal = std::log(static_cast<double>(total));
        route->assign(n + 1, {0.0, 0});
        for (uint32_t idx = n; idx-- > 0;) {
            double best = 0;
            uint32_t bestX = 0;
            bool first = true;
            for (uint32_t x : dag[idx]) {
                int e = findExact(sen + idx, x - idx + 1);
                int64_t f = (e >= 0 && entries[e].freq > 0) ? entries[e].freq : 1;
                // FREQ.get(w) or 1  ->  log(f)
                double lf = (e >= 0 && entries[e].freq > 0)
                                ? std::log(static_cast<double>(f))
                                : 0.0;  // log(1)
                double score = (lf - logtotal) + (*route)[x + 1].prob;
                if (first || score >= best) {  // >= : ascending x, ties -> larger x
                    best = score;
                    bestX = x;
                    first = false;
                }
            }
            (*route)[idx] = {best, bestX};
        }
    }

    void emitSpan(const char32_t* sen, uint32_t begin, uint32_t end,
                  std::vector<PosToken>* out) const {
        PosToken tok;
        encodeUtf8(sen + begin, end - begin, &tok.word);
        tok.flag = tagOf(sen + begin, end - begin);
        out->push_back(std::move(tok));
    }

    void emitLiteral(std::string_view word, const char* flag,
                     std::vector<PosToken>* out) const {
        PosToken tok;
        tok.word.assign(word);
        tok.flag = flag;
        out->push_back(std::move(tok));
    }

    static void encodeInto(const char32_t* sen, uint32_t begin, uint32_t end,
                           std::string* dst) {
        dst->clear();
        encodeUtf8(sen + begin, end - begin, dst);
    }

    // POSTokenizer.__cut_detail
    void cutDetail(const char32_t* sen, uint32_t begin, uint32_t end,
                   std::vector<PosToken>* out) const {
        uint32_t i = begin;
        while (i < end) {
            if (isHanDetail(sen[i])) {
                uint32_t j = i;
                while (j < end && isHanDetail(sen[j])) ++j;
                hmmCut(sen, i, j, out);
                i = j;
            } else {
                uint32_t j = i;
                while (j < end && !isHanDetail(sen[j])) ++j;
                // re_skip_detail = ([\.0-9]+|[a-zA-Z0-9]+) splitting;
                // matched runs get literal 'm'/'eng', separators get 'x'
                uint32_t k = i;
                while (k < j) {
                    if (isNumChar(sen[k])) {
                        uint32_t m = k;
                        while (m < j && isNumChar(sen[m])) ++m;
                        emitLiteral(std::string_view(), "m", out);
                        encodeInto(sen, k, m, &out->back().word);
                        k = m;
                    } else if (isAsciiAlnum(sen[k])) {
                        uint32_t m = k;
                        while (m < j && isAsciiAlnum(sen[m])) ++m;
                        emitLiteral(std::string_view(), "eng", out);
                        encodeInto(sen, k, m, &out->back().word);
                        k = m;
                    } else {
                        uint32_t m = k;
                        while (m < j && !isNumChar(sen[m]) && !isAsciiAlnum(sen[m])) ++m;
                        emitLiteral(std::string_view(), "x", out);
                        encodeInto(sen, k, m, &out->back().word);
                        k = m;
                    }
                }
                i = j;
            }
        }
    }

    // Flush a run of buffered single characters (POSTokenizer.__cut_DAG buf).
    void flushBuf(const char32_t* sen, uint32_t begin, uint32_t end,
                  std::vector<PosToken>* out) const {
        if (end - begin == 1) {
            emitSpan(sen, begin, end, out);
        } else if (!freqPositive(sen + begin, end - begin)) {
            // not FREQ.get(buf) -> HMM detail cut
            cutDetail(sen, begin, end, out);
        } else {
            for (uint32_t k = begin; k < end; ++k) emitSpan(sen, k, k + 1, out);
        }
    }

    // POSTokenizer.__cut_DAG (HMM=True)
    void cutDag(const char32_t* sen, uint32_t begin, uint32_t end,
                std::vector<PosToken>* out) const {
        const uint32_t n = end - begin;
        const char32_t* blk = sen + begin;
        std::vector<std::vector<uint32_t>> dag;
        buildDag(blk, n, &dag);
        std::vector<RouteCell> route;
        computeRoute(blk, n, dag, &route);

        uint32_t x = 0;
        bool bufActive = false;
        uint32_t bufBegin = 0, bufEnd = 0;
        while (x < n) {
            uint32_t y = route[x].end + 1;
            if (y - x == 1) {
                if (!bufActive) { bufBegin = x; bufActive = true; }
                bufEnd = y;
            } else {
                if (bufActive) {
                    flushBuf(blk, bufBegin, bufEnd, out);
                    bufActive = false;
                }
                emitSpan(blk, x, y, out);
            }
            x = y;
        }
        if (bufActive) flushBuf(blk, bufBegin, bufEnd, out);
    }

    // POSTokenizer.__cut_internal non-Han-block branch
    void cutSkipBlock(const char32_t* sen, uint32_t begin, uint32_t end,
                      std::vector<PosToken>* out) const {
        uint32_t i = begin;
        while (i < end) {
            // re_skip_internal = (\r\n|\s)
            if (sen[i] == U'\r' && i + 1 < end && sen[i + 1] == U'\n') {
                emitLiteral("\r\n", "x", out);
                i += 2;
                continue;
            }
            if (isPySpace(sen[i])) {
                std::string w;
                encodeUtf8(sen + i, 1, &w);
                emitLiteral(w, "x", out);
                i += 1;
                continue;
            }
            uint32_t j = i;
            while (j < end) {
                if (sen[j] == U'\r' && j + 1 < end && sen[j + 1] == U'\n') break;
                if (isPySpace(sen[j])) break;
                ++j;
            }
            // per-character tagging; NOTE bug-compat: re_eng.match(x) tests
            // the PIECE's first character, not the current character xx.
            bool pieceEng = isAsciiAlnum(sen[i]);
            for (uint32_t k = i; k < j; ++k) {
                if (isNumChar(sen[k])) {
                    std::string w;
                    encodeUtf8(sen + k, 1, &w);
                    emitLiteral(w, "m", out);
                } else if (pieceEng) {
                    std::string w;
                    encodeUtf8(sen + k, 1, &w);
                    emitLiteral(w, "eng", out);
                } else {
                    std::string w;
                    encodeUtf8(sen + k, 1, &w);
                    emitLiteral(w, "x", out);
                }
            }
            i = j;
        }
    }

    // POSTokenizer.__cut — BMES grouping after viterbi
    void hmmCut(const char32_t* sen, uint32_t begin, uint32_t end,
                std::vector<PosToken>* out) const {
        std::vector<uint32_t> states;
        viterbi(sen + begin, end - begin, &states);
        uint32_t beginPos = 0, nexti = 0;
        for (uint32_t i = 0; i < states.size(); ++i) {
            char bmes = static_cast<char>(st_bmes[states[i]]);
            if (bmes == 'B') {
                beginPos = i;
            } else if (bmes == 'E') {
                PosToken tok;
                encodeUtf8(sen + begin + beginPos, i - beginPos + 1, &tok.word);
                tok.flag = tags[st_tag[states[i]]];
                out->push_back(std::move(tok));
                nexti = i + 1;
            } else if (bmes == 'S') {
                PosToken tok;
                encodeUtf8(sen + begin + i, 1, &tok.word);
                tok.flag = tags[st_tag[states[i]]];
                out->push_back(std::move(tok));
                nexti = i + 1;
            }
        }
        if (nexti < states.size()) {
            PosToken tok;
            encodeUtf8(sen + begin + nexti, states.size() - nexti, &tok.word);
            tok.flag = tags[st_tag[states[nexti]]];
            out->push_back(std::move(tok));
        }
    }

    // posseg.viterbi — states are (BMES, tag) pairs; exported ids preserve
    // lexicographic (bmes, tag) order so integer compare == tuple compare.
    void viterbi(const char32_t* obs, uint32_t T,
                 std::vector<uint32_t>* routeOut) const {
        routeOut->clear();
        if (T == 0) return;

        std::vector<double> prev(S, 0.0), cur(S, 0.0);
        std::vector<std::vector<uint32_t>> mem(
            T, std::vector<uint32_t>(S, kInvalidState));
        std::vector<std::vector<uint8_t>> present(
            T, std::vector<uint8_t>(S, 0));

        // init: for y in states.get(obs[0], all_states)
        {
            uint32_t cnt = 0;
            const uint16_t* sts = charStates(obs[0], &cnt);
            for (uint32_t k = 0; k < cnt; ++k) {
                uint32_t y = sts[k];
                prev[y] = start_p[y] + emitProb(y, obs[0]);
                present[0][y] = 1;
                mem[0][y] = kInvalidState;
            }
            if (cnt == 0) {
                for (uint32_t y = 0; y < S; ++y) {
                    prev[y] = start_p[y] + emitProb(y, obs[0]);
                    present[0][y] = 1;
                }
            }
        }

        std::vector<uint8_t> expectNext(S, 0);
        std::vector<uint32_t> prevStates, obsStates;

        for (uint32_t t = 1; t < T; ++t) {
            prevStates.clear();
            std::fill(expectNext.begin(), expectNext.end(), 0);
            for (uint32_t y0 = 0; y0 < S; ++y0) {
                if (!present[t - 1][y0]) continue;
                if (trans_off[y0 + 1] > trans_off[y0]) {  // len(trans_p[y0]) > 0
                    prevStates.push_back(y0);
                    for (uint32_t i = trans_off[y0]; i < trans_off[y0 + 1]; ++i)
                        expectNext[trans_dst[i]] = 1;
                }
            }
            if (prevStates.empty()) {
                // max() over an empty sequence raises in python; unreachable
                // for the shipped model tables.
                throw std::runtime_error("jieba viterbi: no reachable prev state");
            }

            // obs_states = set(states.get(obs[t], all)) & expect_next
            obsStates.clear();
            {
                uint32_t cnt = 0;
                const uint16_t* sts = charStates(obs[t], &cnt);
                if (sts) {
                    for (uint32_t k = 0; k < cnt; ++k)
                        if (expectNext[sts[k]]) obsStates.push_back(sts[k]);
                } else {
                    for (uint32_t y = 0; y < S; ++y)
                        if (expectNext[y]) obsStates.push_back(y);
                }
                if (obsStates.empty()) {
                    if (sts) {
                        // fall back to full expect_next set
                        for (uint32_t y = 0; y < S; ++y)
                            if (expectNext[y]) obsStates.push_back(y);
                    }
                    // (when sts was null the filtered loop already equals
                    // expect_next; both-empty -> all_states fallback below)
                    if (obsStates.empty()) {
                        for (uint32_t y = 0; y < S; ++y) obsStates.push_back(y);
                    }
                }
            }

            for (uint32_t y : obsStates) {
                double best = kMinInf;
                uint32_t bestY0 = kInvalidState;
                bool first = true;
                for (uint32_t y0 : prevStates) {
                    double em = emitProb(y, obs[t]);
                    double cand =
                        (prev[y0] + transProb(y0, y)) + em;
                    if (first || cand > best ||
                        (cand == best && y0 > bestY0)) {
                        best = cand;
                        bestY0 = y0;
                        first = false;
                    }
                }
                cur[y] = best;
                present[t][y] = 1;
                mem[t][y] = bestY0;
            }
            std::copy(cur.begin(), cur.end(), prev.begin());
        }

        // last = max((V[-1][y], y) for y in mem_path[-1])
        uint32_t lastState = kInvalidState;
        double lastProb = kMinInf;
        for (uint32_t y = 0; y < S; ++y) {
            if (!present[T - 1][y]) continue;
            if (lastState == kInvalidState || prev[y] >= lastProb) {
                lastProb = prev[y];
                lastState = y;
            }
        }
        if (lastState == kInvalidState)
            throw std::runtime_error("jieba viterbi: empty final path");

        int64_t i = static_cast<int64_t>(T) - 1;
        uint32_t st = lastState;
        std::vector<uint32_t>& r = *routeOut;
        r.assign(T, 0);
        while (i >= 0) {
            r[static_cast<size_t>(i)] = st;
            st = mem[static_cast<size_t>(i)][st];
            --i;
        }
    }

    // ---- plain Tokenizer.cut (HMM=True) + finalseg ------------------------
    // Used by ToneSandhi._split_word via cutForSearch. Differs from the
    // posseg path: re_han_default includes '%', buf fallback goes through
    // tag-less finalseg.cut instead of __cut_detail, and non-Han blocks
    // yield raw pieces without m/eng/x tagging.

    // finalseg viterbi: states B,E,M,S (ids 0..3); missing transition AND
    // emission both fall back to MIN_FLOAT; PrevStatus fixed table.
    void viterbiFinalseg(const char32_t* obs, uint32_t T,
                         uint8_t* routeOut) const {
        static constexpr uint8_t kPrev[4][3] = {
            {1, 3, 0xFF},  // B <- E,S
            {0, 2, 0xFF},  // E <- B,M
            {2, 0, 0xFF},  // M <- M,B
            {3, 1, 0xFF},  // S <- S,E
        };
        double V[2][4];
        uint8_t mem[512][4];
        if (T > 512) throw std::runtime_error("finalseg: block too long");
        for (uint32_t y = 0; y < 4; ++y) {
            V[0][y] = fstart[y] + femitProb(y, obs[0]);
            mem[0][y] = 0xFF;
        }
        for (uint32_t t = 1; t < T; ++t) {
            const auto& prevRow = V[(t - 1) & 1];
            auto& curRow = V[t & 1];
            for (uint8_t y = 0; y < 4; ++y) {
                double em = femitProb(y, obs[t]);
                double best = kMinInf;
                uint8_t bestY0 = 0xFF;
                bool first = true;
                for (int k = 0; k < 3 && kPrev[y][k] != 0xFF; ++k) {
                    uint8_t y0 = kPrev[y][k];
                    double cand = (prevRow[y0] + ftransProb(y0, y)) + em;
                    if (first || cand > best ||
                        (cand == best && y0 > bestY0)) {
                        best = cand;
                        bestY0 = y0;
                        first = false;
                    }
                }
                curRow[y] = best;
                mem[t][y] = bestY0;
            }
        }
        // end: max over 'E'(1), 'S'(3); tuple tie -> larger id (= larger char)
        uint8_t last = (V[(T - 1) & 1][3] >= V[(T - 1) & 1][1]) ? 3 : 1;
        int64_t i = static_cast<int64_t>(T) - 1;
        uint8_t st = last;
        while (i >= 0) {
            routeOut[i] = st;
            st = mem[i][st];
            --i;
        }
    }

    // finalseg.__cut: BMES grouping into words (Force_Split_Words is empty:
    // only add_word(freq=0) populates it, and no user dict exists here)
    void finalsegHanCut(const char32_t* sen, uint32_t begin, uint32_t end,
                        std::vector<PosToken>* out) const {
        std::vector<uint8_t> route(end - begin);
        viterbiFinalseg(sen + begin, end - begin, route.data());
        uint32_t beginPos = 0, nexti = 0;
        const char* bmes = "BEMS";
        for (uint32_t i = 0; i < route.size(); ++i) {
            char st = bmes[route[i]];
            PosToken tok;
            tok.flag = "x";
            if (st == 'B') {
                beginPos = i;
                continue;
            } else if (st == 'E') {
                encodeUtf8(sen + begin + beginPos, i - beginPos + 1, &tok.word);
                out->push_back(std::move(tok));
                nexti = i + 1;
            } else if (st == 'S') {
                encodeUtf8(sen + begin + i, 1, &tok.word);
                out->push_back(std::move(tok));
                nexti = i + 1;
            }
        }
        if (nexti < route.size()) {
            PosToken tok;
            tok.flag = "x";
            encodeUtf8(sen + begin + nexti, route.size() - nexti, &tok.word);
            out->push_back(std::move(tok));
        }
    }

    // finalseg.cut: Han runs -> BMES viterbi; other runs split by
    // ([a-zA-Z0-9]+(?:\.\d+)?%?) and ALL pieces yielded verbatim.
    void finalsegCut(const char32_t* sen, uint32_t begin, uint32_t end,
                     std::vector<PosToken>* out) const {
        uint32_t i = begin;
        while (i < end) {
            if (isHanDetail(sen[i])) {
                uint32_t j = i;
                while (j < end && isHanDetail(sen[j])) ++j;
                finalsegHanCut(sen, i, j, out);
                i = j;
            } else {
                uint32_t j = i;
                while (j < end && !isHanDetail(sen[j])) ++j;
                uint32_t k = i;
                while (k < j) {
                    if (isAsciiAlnum(sen[k])) {
                        uint32_t m = k;
                        while (m < j && isAsciiAlnum(sen[m])) ++m;
                        if (m < j && sen[m] == U'.' && m + 1 < j &&
                            isAsciiDigit(sen[m + 1])) {
                            ++m;  // consume '.'
                            while (m < j && isAsciiDigit(sen[m])) ++m;
                        }
                        if (m < j && sen[m] == U'%') ++m;
                        emitSpan(sen, k, m, out);
                        k = m;
                    } else {
                        uint32_t m = k;
                        while (m < j && !isAsciiAlnum(sen[m])) ++m;
                        emitSpan(sen, k, m, out);
                        k = m;
                    }
                }
                i = j;
            }
        }
    }

    // Tokenizer.__cut_DAG (no tags): buf flush uses finalseg.cut for OOV
    void cutDagTokenizer(const char32_t* sen, uint32_t begin, uint32_t end,
                         std::vector<PosToken>* out) const {
        const uint32_t n = end - begin;
        const char32_t* blk = sen + begin;
        std::vector<std::vector<uint32_t>> dag;
        buildDag(blk, n, &dag);
        std::vector<RouteCell> route;
        computeRoute(blk, n, dag, &route);

        uint32_t x = 0;
        bool bufActive = false;
        uint32_t bufBegin = 0, bufEnd = 0;
        auto flush = [&]() {
            if (bufEnd - bufBegin == 1) {
                emitSpan(blk, bufBegin, bufEnd, out);
            } else if (!freqPositive(blk + bufBegin, bufEnd - bufBegin)) {
                finalsegCut(blk, bufBegin, bufEnd, out);
            } else {
                for (uint32_t k = bufBegin; k < bufEnd; ++k)
                    emitSpan(blk, k, k + 1, out);
            }
        };
        while (x < n) {
            uint32_t y = route[x].end + 1;
            if (y - x == 1) {
                if (!bufActive) { bufBegin = x; bufActive = true; }
                bufEnd = y;
            } else {
                if (bufActive) { flush(); bufActive = false; }
                emitSpan(blk, x, y, out);
            }
            x = y;
        }
        if (bufActive) flush();
    }

    // Tokenizer.cut(HMM=True): re_han_default INCLUDES '%'
    static bool isHanDefault(uint32_t c) {
        return isHanInternal(c) || c == '%';
    }

    void cutTokenizer(const char32_t* sen, uint32_t n,
                      std::vector<PosToken>* out) const {
        uint32_t i = 0;
        while (i < n) {
            if (isHanDefault(sen[i])) {
                uint32_t j = i;
                while (j < n && isHanDefault(sen[j])) ++j;
                cutDagTokenizer(sen, i, j, out);
                i = j;
            } else {
                uint32_t j = i;
                while (j < n && !isHanDefault(sen[j])) ++j;
                // re_skip_default = (\r\n|\s): whitespace tokens raw, other
                // characters individually
                uint32_t k = i;
                while (k < j) {
                    if (sen[k] == U'\r' && k + 1 < j && sen[k + 1] == U'\n') {
                        emitLiteral("\r\n", "x", out);
                        k += 2;
                    } else if (isPySpace(sen[k])) {
                        std::string w;
                        encodeUtf8(sen + k, 1, &w);
                        emitLiteral(w, "x", out);
                        k += 1;
                    } else {
                        uint32_t m = k;
                        while (m < j) {
                            if (sen[m] == U'\r' && m + 1 < j && sen[m + 1] == U'\n') break;
                            if (isPySpace(sen[m])) break;
                            ++m;
                        }
                        for (uint32_t c2 = k; c2 < m; ++c2) {
                            std::string w;
                            encodeUtf8(sen + c2, 1, &w);
                            emitLiteral(w, "x", out);
                        }
                        k = m;
                    }
                }
                i = j;
            }
        }
    }
};

JiebaSegmenter::~JiebaSegmenter() {
    delete impl_;
    impl_ = nullptr;
}

bool JiebaSegmenter::load(const std::string& path, std::string* err) {
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

bool JiebaSegmenter::loadMemory(const void* data, size_t size, std::string* err) {
    delete impl_;
    impl_ = nullptr;

    auto fail = [err](const std::string& msg) {
        if (err) *err = msg;
        return false;
    };
    auto dataU8 = static_cast<const uint8_t*>(data);

    if (size < 24 || std::memcmp(dataU8, "GSVJTB01", 8) != 0)
        return fail("jieba_trie: bad magic");
    uint32_t version, nSections;
    std::memcpy(&version, dataU8 + 8, 4);
    std::memcpy(&nSections, dataU8 + 12, 4);
    if (version != 1) return fail("jieba_trie: unsupported version");

    auto section = [&](const char* name, const uint8_t** out, size_t* nbytes)
        -> bool {
        char padded[8];
        static_assert(sizeof(padded) == 8);
        std::memset(padded, 0, sizeof(padded));
        const size_t nlen = std::strlen(name);
        if (nlen > 8) return false;
        std::memcpy(padded, name, nlen);
        const size_t dirBase = 16;
        for (uint32_t i = 0; i < nSections; ++i) {
            size_t off = dirBase + 32ull * i;
            if (off + 32 > size) return false;
            if (std::memcmp(dataU8 + off, padded, 8) == 0) {
                uint64_t so, sn;
                std::memcpy(&so, dataU8 + off + 8, 8);
                std::memcpy(&sn, dataU8 + off + 16, 8);
                if (so + sn > size) return false;
                *out = dataU8 + so;
                *nbytes = static_cast<size_t>(sn);
                return true;
            }
        }
        return false;
    };

    auto impl = std::make_unique<Impl>();
    impl->blob.assign(dataU8, dataU8 + size);  // own copy keeps API simple

    const uint8_t* p = nullptr;
    size_t nb = 0;

    if (!section("dictmeta", &p, &nb)) return fail("missing dictmeta");
    {
        ByteReader r(p, nb);
        impl->total = r.i64();
    }
    if (!section("tags", &p, &nb)) return fail("missing tags");
    {
        ByteReader r(p, nb);
        uint32_t nTags = r.u32();
        impl->tags.resize(nTags);
        for (uint32_t i = 0; i < nTags; ++i) {
            uint8_t len = r.u8();
            impl->tags[i].resize(len);
            for (uint8_t k = 0; k < len; ++k)
                impl->tags[i][k] = static_cast<char>(r.u8());
        }
    }
    if (!section("cps", &p, &nb)) return fail("missing cps");
    {
        ByteReader r(p, nb);
        size_t count = nb / 4;
        impl->cps.resize(count);
        for (size_t i = 0; i < count; ++i) impl->cps[i] = r.u32();
    }
    if (!section("entries", &p, &nb)) return fail("missing entries");
    {
        ByteReader r(p, nb);
        uint32_t count = r.u32();
        impl->entries.resize(count);
        for (uint32_t i = 0; i < count; ++i) {
            impl->entries[i].off = r.u32();
            impl->entries[i].len = r.u16();
            impl->entries[i].tag = r.u16();
            impl->entries[i].freq = r.i64();
        }
    }
    if (!section("states", &p, &nb)) return fail("missing states");
    {
        ByteReader r(p, nb);
        uint32_t S = r.u32();
        impl->S = S;
        impl->st_bmes.resize(S);
        impl->st_tag.resize(S);
        for (uint32_t i = 0; i < S; ++i) {
            impl->st_bmes[i] = r.u8();
            impl->st_tag[i] = r.u16();
            r.u8();  // pad
        }
    }
    if (!section("hmmstart", &p, &nb)) return fail("missing hmmstart");
    {
        ByteReader r(p, nb);
        impl->start_p.resize(impl->S);
        for (uint32_t i = 0; i < impl->S; ++i) impl->start_p[i] = r.f64();
    }
    if (!section("hmmtrans", &p, &nb)) return fail("missing hmmtrans");
    {
        ByteReader r(p, nb);
        impl->trans_off.assign(impl->S + 1, 0);
        impl->trans_dst.clear();
        impl->trans_p.clear();
        for (uint32_t s = 0; s < impl->S; ++s) {
            uint32_t nnz = r.u32();
            impl->trans_off[s] = static_cast<uint32_t>(impl->trans_dst.size());
            for (uint32_t i = 0; i < nnz; ++i) {
                impl->trans_dst.push_back(r.u16());
                r.u8();  // pad
                r.u8();  // pad
                impl->trans_p.push_back(r.f64());
            }
        }
        impl->trans_off[impl->S] = static_cast<uint32_t>(impl->trans_dst.size());
    }
    if (!section("hmmitmap", &p, &nb)) return fail("missing hmmitmap");
    {
        ByteReader r(p, nb);
        impl->emit_off.assign(impl->S + 1, 0);
        impl->emit_cp.clear();
        impl->emit_p.clear();
        for (uint32_t s = 0; s < impl->S; ++s) {
            impl->emit_off[s] = static_cast<uint32_t>(impl->emit_cp.size());
            uint32_t n = r.u32();
            for (uint32_t i = 0; i < n; ++i) {
                impl->emit_cp.push_back(r.u32());
                impl->emit_p.push_back(r.f64());
            }
        }
        impl->emit_off[impl->S] = static_cast<uint32_t>(impl->emit_cp.size());
    }
    if (!section("chars", &p, &nb)) return fail("missing chars");
    {
        ByteReader r(p, nb);
        uint32_t nChars = r.u32();
        impl->cs_cp.resize(nChars);
        impl->cs_first.resize(nChars);
        impl->cs_cnt.resize(nChars);
        for (uint32_t i = 0; i < nChars; ++i) {
            impl->cs_cp[i] = r.u32();
            impl->cs_first[i] = r.u32();
            impl->cs_cnt[i] = r.u32();
        }
        uint32_t listLen = r.u32();
        impl->cs_list.resize(listLen);
        for (uint32_t i = 0; i < listLen; ++i) impl->cs_list[i] = r.u16();
    }

    // finalseg sections are optional (older binaries); cutForSearch needs them
    if (section("fstart", &p, &nb)) {
        ByteReader r(p, nb);
        for (int i = 0; i < 4; ++i) impl->fstart[i] = r.f64();
        if (!section("ftrans", &p, &nb)) return fail("ftrans missing");
        {
            ByteReader rt(p, nb);
            for (int s = 0; s < 4; ++s) {
                impl->ftransOff[s] = static_cast<uint32_t>(impl->ftransDst.size());
                uint32_t nnz = rt.u32();
                for (uint32_t i = 0; i < nnz; ++i) {
                    impl->ftransDst.push_back(rt.u16());
                    rt.u8(); rt.u8();  // pad
                    impl->ftransP.push_back(rt.f64());
                }
            }
            impl->ftransOff[4] = static_cast<uint32_t>(impl->ftransDst.size());
        }
        if (!section("femit", &p, &nb)) return fail("femit missing");
        {
            ByteReader re_(p, nb);
            for (int s = 0; s < 4; ++s) {
                impl->femitOff[s] = static_cast<uint32_t>(impl->femitCp.size());
                uint32_t n = re_.u32();
                for (uint32_t i = 0; i < n; ++i) {
                    impl->femitCp.push_back(re_.u32());
                    impl->femitP.push_back(re_.f64());
                }
            }
            impl->femitOff[4] = static_cast<uint32_t>(impl->femitCp.size());
        }
        impl->hasFinalseg = true;
    }

    impl_ = impl.release();
    if (err) err->clear();
    return true;
}

void JiebaSegmenter::lcut(std::string_view utf8_sentence,
                          std::vector<PosToken>* out) const {
    if (!impl_) throw std::runtime_error("JiebaSegmenter::lcut before load()");
    std::u32string sen;
    decodeUtf8(utf8_sentence, &sen);
    const char32_t* s = sen.data();
    const uint32_t n = static_cast<uint32_t>(sen.size());

    // blocks = re_han_internal.split(sentence): alternate non-match /
    // match runs. Matching blocks consist solely of HanInternal characters.
    uint32_t i = 0;
    while (i < n) {
        if (isHanInternal(s[i])) {
            uint32_t j = i;
            while (j < n && isHanInternal(s[j])) ++j;
            impl_->cutDag(s, i, j, out);   // re_han_internal.match(blk) branch
            i = j;
        } else {
            uint32_t j = i;
            while (j < n && !isHanInternal(s[j])) ++j;
            impl_->cutSkipBlock(s, i, j, out);
            i = j;
        }
    }
}

void JiebaSegmenter::cutForSearch(std::string_view utf8_sentence,
                                  std::vector<std::string>* out) const {
    if (!impl_) throw std::runtime_error("JiebaSegmenter::cutForSearch before load()");
    if (!impl_->hasFinalseg)
        throw std::runtime_error(
            "jieba_trie.bin lacks finalseg tables; regenerate with "
            "tools/export_jieba_trie.py");
    std::u32string sen;
    decodeUtf8(utf8_sentence, &sen);
    std::vector<PosToken> words;  // plain Tokenizer.cut words (tags unused)
    impl_->cutTokenizer(sen.data(), static_cast<uint32_t>(sen.size()), &words);

    for (const PosToken& w : words) {
        std::u32string cps;
        decodeUtf8(w.word, &cps);
        const char32_t* p = cps.data();
        const uint32_t n = static_cast<uint32_t>(cps.size());
        // gram2 hits for len>2, gram3 hits for len>3, then the word itself
        if (n > 2) {
            for (uint32_t i = 0; i + 1 < n; ++i) {
                if (impl_->freqPositive(p + i, 2)) {
                    std::string g;
                    encodeUtf8(p + i, 2, &g);
                    out->push_back(std::move(g));
                }
            }
        }
        if (n > 3) {
            for (uint32_t i = 0; i + 2 < n; ++i) {
                if (impl_->freqPositive(p + i, 3)) {
                    std::string g;
                    encodeUtf8(p + i, 3, &g);
                    out->push_back(std::move(g));
                }
            }
        }
        out->push_back(w.word);
    }
}

}  // namespace gsv::textfront
