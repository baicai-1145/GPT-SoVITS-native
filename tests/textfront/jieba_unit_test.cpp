// jieba_unit_test.cpp — B5 unit tests on constructed cases.
//
// Builds a miniature "GSVJTB01" blob by hand (same layout as
// tools/export_jieba_trie.py) so trie round-trips, DAG max-probability path
// selection, buf mechanics and the POS-HMM viterbi are validated against
// hand-computed expectations, independent of the shipped dictionary.
//
// Exit 0 = all pass.
#include "jieba.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

using gsv::textfront::JiebaSegmenter;
using gsv::textfront::PosToken;

static int g_failures = 0;

#define CHECK(cond, msg)                                                  \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, msg);     \
            ++g_failures;                                                 \
        }                                                                 \
    } while (0)

static void checkTokens(const std::vector<PosToken>& got,
                        const std::vector<std::pair<std::string, std::string>>& exp,
                        const char* what) {
    bool ok = got.size() == exp.size();
    if (ok) {
        for (size_t i = 0; i < got.size() && ok; ++i)
            ok = got[i].word == exp[i].first && got[i].flag == exp[i].second;
    }
    if (!ok) {
        std::printf("FAIL %s: token mismatch (got %zu, want %zu)\n", what,
                    got.size(), exp.size());
        for (size_t i = 0; i < got.size(); ++i)
            std::printf("  got[%zu] '%s'/'%s'\n", i, got[i].word.c_str(),
                        got[i].flag.c_str());
        for (size_t i = 0; i < exp.size(); ++i)
            std::printf(" want[%zu] '%s'/'%s'\n", i, exp[i].first.c_str(),
                        exp[i].second.c_str());
        ++g_failures;
    } else {
        std::printf("ok   %s\n", what);
    }
}

// ---------------------------------------------------------------------------
// mini-blob builder (mirrors the exporter's section layout, little-endian;
// host arm64 is LE so raw casts suffice)
// ---------------------------------------------------------------------------
static void putU16(std::string& s, uint16_t v) { s.append(reinterpret_cast<const char*>(&v), 2); }
static void putU32(std::string& s, uint32_t v) { s.append(reinterpret_cast<const char*>(&v), 4); }
static void putU64(std::string& s, uint64_t v) { s.append(reinterpret_cast<const char*>(&v), 8); }
static void putI64(std::string& s, int64_t v) { s.append(reinterpret_cast<const char*>(&v), 8); }
static void putF64(std::string& s, double v) { s.append(reinterpret_cast<const char*>(&v), 8); }

struct MiniEntry {
    std::u32string key;
    int64_t freq;
    int tag;  // index into tags, -1 = none
};

struct ToyHmm {
    // states sorted lexicographically by (bmes, tag)
    std::vector<std::pair<char, std::string>> states;
    std::vector<double> start;
    std::vector<std::vector<std::pair<int, double>>> trans;  // [from] -> (to,p)
    std::vector<std::vector<std::pair<uint32_t, double>>> emit;  // [state] -> (cp,p)
    std::vector<std::pair<uint32_t, std::vector<int>>> charStates;
};

static std::string buildBlob(int64_t total, const std::vector<std::string>& tags,
                             const std::vector<MiniEntry>& entries,
                             const ToyHmm& hmm) {
    std::vector<MiniEntry> es = entries;
    std::sort(es.begin(), es.end(),
              [](const MiniEntry& a, const MiniEntry& b) { return a.key < b.key; });

    std::u32string pool;
    std::string entriesSec;
    putU32(entriesSec, static_cast<uint32_t>(es.size()));
    for (const auto& e : es) {
        putU32(entriesSec, static_cast<uint32_t>(pool.size()));
        putU16(entriesSec, static_cast<uint16_t>(e.key.size()));
        putU16(entriesSec, e.tag < 0 ? 0xFFFFu : static_cast<uint16_t>(e.tag));
        putI64(entriesSec, e.freq);
        pool += e.key;
    }
    std::string cpsSec;
    for (char32_t c : pool) putU32(cpsSec, static_cast<uint32_t>(c));

    std::string metaSec;
    putI64(metaSec, total);

    std::string tagsSec;
    putU32(tagsSec, static_cast<uint32_t>(tags.size()));
    for (const auto& t : tags) {
        tagsSec.push_back(static_cast<char>(t.size()));
        tagsSec += t;
    }

    std::string statesSec;
    putU32(statesSec, static_cast<uint32_t>(hmm.states.size()));
    for (const auto& st : hmm.states) {
        statesSec.push_back(st.first);
        int tid = -1;
        for (size_t i = 0; i < tags.size(); ++i)
            if (tags[i] == st.second) tid = static_cast<int>(i);
        putU16(statesSec, tid < 0 ? 0xFFFFu : static_cast<uint16_t>(tid));
        statesSec.push_back('\0');
    }

    std::string startSec;
    for (double p : hmm.start) putF64(startSec, p);

    std::string transSec;
    for (const auto& row : hmm.trans) {
        putU32(transSec, static_cast<uint32_t>(row.size()));
        for (const auto& [dst, p] : row) {
            putU16(transSec, static_cast<uint16_t>(dst));
            putU16(transSec, 0);
            putF64(transSec, p);
        }
    }

    std::string emitSec;
    for (const auto& row : hmm.emit) {
        putU32(emitSec, static_cast<uint32_t>(row.size()));
        for (const auto& [cp, p] : row) {
            putU32(emitSec, cp);
            putF64(emitSec, p);
        }
    }

    std::string charsSec;
    putU32(charsSec, static_cast<uint32_t>(hmm.charStates.size()));
    uint32_t acc = 0;
    for (const auto& [cp, lst] : hmm.charStates) {
        putU32(charsSec, cp);
        putU32(charsSec, acc);
        putU32(charsSec, static_cast<uint32_t>(lst.size()));
        acc += static_cast<uint32_t>(lst.size());
    }
    putU32(charsSec, acc);
    for (const auto& [cp, lst] : hmm.charStates)
        for (int v : lst) putU16(charsSec, static_cast<uint16_t>(v));

    struct Sec {
        const char* name;
        const std::string* data;
    };
    Sec secs[] = {
        {"dictmeta", &metaSec},     {"tags", &tagsSec},
        {"cps", &cpsSec},           {"entries", &entriesSec},
        {"states", &statesSec},     {"hmmstart", &startSec},
        {"hmmtrans", &transSec},    {"hmmitmap", &emitSec},
        {"chars", &charsSec},
    };
    std::string out = "GSVJTB01";
    putU32(out, 1);
    putU32(out, 9);
    size_t off = 16 + 32 * 9;
    for (auto& s : secs) {
        char name[8];
        std::memset(name, 0, 8);
        std::memcpy(name, s.name, std::strlen(s.name));
        out.append(name, 8);
        putU64(out, off);
        putU64(out, s.data->size());
        putU64(out, 0);  // reserved
        off += s.data->size();
    }
    for (auto& s : secs) out += *s.data;
    return out;
}

// ---------------------------------------------------------------------------
// toy model: dict AB=100/n, A=10/x, B=5/x ; HMM over chars 甲(丙)乙(丁)
// states sorted lex: ('B','n'),('E','n'),('M','n'),('S','x')
// ---------------------------------------------------------------------------
static std::string makeToyBlob(int64_t* totalOut) {
    const int64_t total = 115;
    *totalOut = total;

    ToyHmm h;
    h.states = {{'B', "n"}, {'E', "n"}, {'M', "n"}, {'S', "x"}};
    h.start = {-1.0, -5.0, -7.0, -0.2};
    h.trans.resize(4);
    h.trans[0] = {{1, -0.1}, {2, -2.5}};          // B -> E | M
    h.trans[1] = {{0, -0.3}, {3, -1.0}};          // E -> B | S
    h.trans[2] = {{1, -0.2}, {2, -3.0}};          // M -> E | M
    h.trans[3] = {{0, -0.5}, {3, -0.6}};          // S -> B | S

    const uint32_t JIA = U'甲', YI = U'乙', BING = U'丙';
    static_assert(U'丙' < U'甲');  // 0x4E19 < 0x7532; note 乙 U+4E59 is between
    h.emit.resize(4);
    h.emit[0] = {{BING, -0.2}, {JIA, -0.05}};     // B emits 甲/丙 (sorted!)
    h.emit[1] = {{YI, -0.06}};                    // E emits 乙
    h.emit[2] = {};                               // M emits nothing
    h.emit[3] = {{YI, -1.1}, {BING, -1.2}, {JIA, -1.0}};  // S weakly, sorted

    h.charStates = {  // cs_cp sorted ascending
        {YI, {1, 3}},    // 乙 may continue as E or S
        {BING, {0, 3}},  // 丙
        {JIA, {0, 3}},   // 甲 may start as B or S
    };

    auto u32s = [](const char* utf8) {
        // tiny helper: only BMP chars used here
        std::u32string r;
        for (size_t i = 0; utf8[i];) {
            unsigned char b = static_cast<unsigned char>(utf8[i]);
            if (b < 0x80) { r.push_back(b); i += 1; }
            else if ((b & 0xE0) == 0xE0) {
                r.push_back(static_cast<char32_t>(((b & 0x0Fu) << 12) |
                            ((static_cast<unsigned char>(utf8[i + 1]) & 0x3Fu) << 6) |
                            (static_cast<unsigned char>(utf8[i + 2]) & 0x3Fu)));
                i += 3;
            } else {  // 2-byte (unused here but kept for safety)
                r.push_back(static_cast<char32_t>(((b & 0x1Fu) << 6) |
                            (static_cast<unsigned char>(utf8[i + 1]) & 0x3Fu)));
                i += 2;
            }
        }
        return r;
    };

    return buildBlob(
        total,
        {"n", "x"},
        {
            {u32s("AB"), 100, 0},
            {u32s("A"), 10, 1},
            {u32s("B"), 5, 1},
        },
        h);
}

int main() {
    int64_t total = 0;
    std::string blob = makeToyBlob(&total);
    CHECK(blob.size() > 100, "blob built");

    JiebaSegmenter seg;
    std::string err;
    CHECK(seg.loadMemory(blob.data(), blob.size(), &err), err.c_str());

    // --- trie round-trip through public behavior --------------------------
    // DAG max-prob: log(100)-log(115) > log(10)+log(5)-2*log(115) => "AB"
    std::vector<PosToken> toks;
    seg.lcut("AB", &toks);  // note: lcut appends; tests clear before each call
    checkTokens(toks, {{"AB", "n"}}, "trie/DAG picks max-prob word AB");

    // single real word A keeps its own token (len==1 buf branch, dict tag x)
    toks.clear();
    seg.lcut("A", &toks);
    checkTokens(toks, {{"A", "x"}}, "single-char word uses dict tag");

    // OOV alnum run goes through __cut_detail -> 'eng' literal tag
    toks.clear();
    seg.lcut("ACB", &toks);
    checkTokens(toks, {{"ACB", "eng"}}, "OOV alnum buf -> cut_detail eng");

    // "ABX": DAG route picks multi-char AB then single X; X tail flushed
    // via len==1 branch with default tag 'x'
    toks.clear();
    seg.lcut("ABX", &toks);
    checkTokens(toks, {{"AB", "n"}, {"X", "x"}}, "dict word then OOV tail");

    // mixed Han + ascii block boundary: HMM path on toy emissions
    // 甲乙: init B(甲)= -1-0.05=-1.05, S(甲)=-0.2-1.0=-1.2
    // t=1: E from B: -1.05-0.1-0.06=-1.21 ; S from B: -1.05-1.0-1.1=-3.15
    //      E from S: -1.2+(-inf)=-inf ; S from S: -1.2-0.6-1.1=-2.9
    // final: E(-1.21) vs S(-2.9) -> E wins -> one word 甲乙 tagged n
    toks.clear();
    seg.lcut("甲乙", &toks);
    checkTokens(toks, {{"甲乙", "n"}}, "toy viterbi groups 甲乙 as B-E");

    // 丙乙 same structure -> grouped too
    toks.clear();
    seg.lcut("丙乙", &toks);
    checkTokens(toks, {{"丙乙", "n"}}, "toy viterbi groups 丙乙");

    // missing emission everywhere forces MIN_FLOAT fallback but still yields
    // a valid path (M has no emissions; charStates gate keeps it reachable)
    toks.clear();
    seg.lcut("甲甲乙", &toks);
    CHECK(!toks.empty(), "viterbi with sparse emissions still segments");
    std::string joined;
    for (auto& t : toks) joined += t.word;
    CHECK(joined == "甲甲乙", "no characters lost in segmentation");

    if (g_failures == 0) std::printf("ALL UNIT TESTS PASSED\n");
    return g_failures == 0 ? 0 : 1;
}
