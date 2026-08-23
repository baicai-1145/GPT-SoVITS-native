// textfront.cpp — see textfront.h. Line-level port of
// TTS_infer_pack/text_segmentation_method.py (split/split_big_text/cut0..5)
// and TextPreprocessor.pre_seg_text + replace_consecutive_punctuation.
#include "textfront.h"

namespace gsv::textfront {
namespace {

U32 dec(const std::string& s) {
    U32 o;
    for (size_t i = 0; i < s.size();) {
        unsigned char b = static_cast<unsigned char>(s[i]);
        if (b < 0x80) { o.push_back(b); i++; continue; }
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

// strip with a specific char set (python str.strip("chars"))
U32 stripChars(const U32& s, const U32& chars) {
    size_t b = 0, e = s.size();
    auto in = [&](char32_t c) { return chars.find(c) != U32::npos; };
    while (b < e && in(s[b])) ++b;
    while (e > b && in(s[e - 1])) --e;
    return s.substr(b, e - b);
}

bool inSet(const char* lit, char32_t c) {
    static thread_local U32 cached;
    static thread_local const char* cachedLit = nullptr;
    if (cachedLit != lit) {
        cached = dec(lit);
        cachedLit = lit;
    }
    return cached.find(c) != U32::npos;
}

// TTS_infer_pack `splits` — the segmentation punctuation class
constexpr const char* kSplits = "，。？！,.?!~:：—…";
// text_segmentation_method.punctuation — subset filter includes space
constexpr const char* kPunctWithSpace = "！?…,,,.- ";
// ---------------------------------------------------------------------------
// cut helpers
// ---------------------------------------------------------------------------

// split(): replace …… -> 。 and —— -> ，; append 。 when tail not a splitor;
// then scan and emit [tail..head] pieces ending at each splitor inclusive.
std::vector<U32> pySplit(const U32& text) {
    U32 t;
    for (size_t i = 0; i < text.size(); ++i) {
        if (text[i] == U'…' && i + 1 < text.size() && text[i + 1] == U'…') {
            t += dec("。");
            ++i;
        } else if (text[i] == U'—' && i + 1 < text.size() &&
                   text[i + 1] == U'—') {
            t += dec("，");
            ++i;
        } else {
            t.push_back(text[i]);
        }
    }
    // python would IndexError on empty; guard to keep native total
    if (t.empty()) return {};
    if (!inSet(kSplits, t.back())) t.push_back(U'。');
    std::vector<U32> out;
    size_t tail = 0;
    for (size_t head = 0; head < t.size(); ++head) {
        if (inSet(kSplits, t[head])) {
            out.push_back(t.substr(tail, head + 1 - tail));
            tail = head + 1;
        }
    }
    return out;
}

// set(item).issubset(punctSet): every char of item is in the set
bool allCharsIn(const U32& item, const char* set) {
    if (item.empty()) return true;  // set([]).issubset(x) is True
    for (char32_t c : item)
        if (!inSet(set, c)) return false;
    return true;
}

std::vector<U32> joinNl(const std::vector<U32>& pieces) {
    // python joins with "\n"; we keep the piece vector (equivalent because
    // the caller immediately re-splits on '\n')
    std::vector<U32> out;
    for (auto& p : pieces)
        if (!p.empty()) out.push_back(p);
    return out;
}

// cut0: no split; pure-punct input becomes the sentinel "/n"
std::vector<U32> cut0(const U32& inp) {
    if (!allCharsIn(inp, kPunctWithSpace)) return {inp};
    return {dec("/n")};
}

// cut1: group every four pieces (runtime default)
//
// python quirk: split_idx = list(range(0, len(inps), 4)) with the LAST
// entry replaced by None, so the final group swallows ALL remaining
// pieces instead of stopping after four.
std::vector<U32> cut1(const U32& inp) {
    U32 trimmed = stripChars(inp, dec("\n"));
    auto inps = pySplit(trimmed);
    if (inps.empty()) return {};  // python crashes on empty; unreachable here
    std::vector<size_t> idx;
    for (size_t k = 0; k < inps.size(); k += 4) idx.push_back(k);
    std::vector<U32> opts;
    if (idx.size() > 1) {
        // boundaries are idx[0..m-2]; the last listed boundary is dropped
        // (python's [-1] = None), so the tail group runs to the end
        for (size_t k = 0; k + 1 < idx.size(); ++k) {
            U32 acc;
            size_t end = (k + 2 == idx.size()) ? inps.size() : idx[k + 1];
            for (size_t j = idx[k]; j < end; ++j) acc += inps[j];
            opts.push_back(std::move(acc));
        }
    } else {
        opts.push_back(trimmed);
    }
    std::vector<U32> kept;
    for (auto& o : opts)
        if (!allCharsIn(o, kPunctWithSpace)) kept.push_back(std::move(o));
    return joinNl(kept);
}

// cut2: pack until cumulative length > 50; short tail merges back
std::vector<U32> cut2(const U32& inp) {
    U32 trimmed = stripChars(inp, dec("\n"));
    auto inps = pySplit(trimmed);
    if (inps.size() < 2) return {inp};
    std::vector<U32> opts;
    long summ = 0;
    U32 tmp;
    for (auto& piece : inps) {
        summ += static_cast<long>(piece.size());
        tmp += piece;
        if (summ > 50) {
            summ = 0;
            opts.push_back(std::move(tmp));
            tmp.clear();
        }
    }
    if (!tmp.empty()) opts.push_back(std::move(tmp));
    if (opts.size() > 1 &&
        opts.back().size() < 50) {  // 尾段太短并入前一段
        opts[opts.size() - 2] += opts.back();
        opts.pop_back();
    }
    std::vector<U32> kept;
    for (auto& o : opts)
        if (!allCharsIn(o, kPunctWithSpace)) kept.push_back(std::move(o));
    return joinNl(kept);
}

// cut3: split on 中文句号 only
std::vector<U32> cut3(const U32& inp) {
    U32 trimmed = stripChars(stripChars(inp, dec("\n")), dec("。"));
    std::vector<U32> opts;
    size_t st = 0;
    for (size_t k = 0; k <= trimmed.size(); ++k) {
        if (k == trimmed.size() || trimmed[k] == U'。') {
            opts.push_back(trimmed.substr(st, k - st));
            st = k + 1;
        }
    }
    std::vector<U32> kept;
    for (auto& o : opts)
        if (!allCharsIn(o, kPunctWithSpace)) kept.push_back(std::move(o));
    return joinNl(kept);
}

// cut4: split on '.' not between digits
std::vector<U32> cut4(const U32& inp) {
    U32 trimmed = stripChars(stripChars(inp, dec("\n")), dec("."));
    std::vector<U32> raw;
    size_t st = 0;
    for (size_t k = 0; k <= trimmed.size(); ++k) {
        bool sep = false;
        if (k < trimmed.size() && trimmed[k] == U'.') {
            bool prevDig =
                k > 0 && trimmed[k - 1] >= U'0' && trimmed[k - 1] <= U'9';
            bool nextDig = k + 1 < trimmed.size() &&
                           trimmed[k + 1] >= U'0' && trimmed[k + 1] <= U'9';
            sep = !prevDig && !nextDig;
        }
        if (k == trimmed.size() || sep) {
            raw.push_back(trimmed.substr(st, k - st));
            st = k + 1;
        }
    }
    std::vector<U32> kept;
    for (auto& o : raw)
        if (!allCharsIn(o, kPunctWithSpace)) kept.push_back(std::move(o));
    return joinNl(kept);
}

// cut5: split at every pund (the digit-dot-digit branch appends too —
// dead condition mirrored as-is from the source)
std::vector<U32> cut5(const U32& inp) {
    constexpr const char* kPunds = ",.;!?、，。？！；：…";
    U32 trimmed = stripChars(inp, dec("\n"));
    auto isDigit = [](char32_t c) { return c >= U'0' && c <= U'9'; };
    std::vector<U32> mergeitems;
    U32 items;
    for (size_t i = 0; i < trimmed.size(); ++i) {
        char32_t c = trimmed[i];
        bool keepGoing =
            // dead-looking branch from the source: a dot between digits is
            // appended WITHOUT flushing the buffer
            c == U'.' && i > 0 && i < trimmed.size() - 1 &&
            isDigit(trimmed[i - 1]) && isDigit(trimmed[i + 1]);
        items.push_back(c);
        if (inSet(kPunds, c) && !keepGoing) {
            mergeitems.push_back(std::move(items));
            items.clear();
        }
    }
    if (!items.empty()) mergeitems.push_back(std::move(items));
    std::vector<U32> opt;
    for (auto& m : mergeitems)
        if (!allCharsIn(m, kPunds)) opt.push_back(std::move(m));
    return opt;
}

// split_big_text(text, max_len=510): re.split keeping separators, greedy
// pack into pieces <= max_len (python compares len(current+seg) > max_len)
std::vector<U32> splitBigText(const U32& text) {
    constexpr size_t kMaxLen = 510;
    std::vector<U32> segs;
    {
        // re.split with a capture group keeps each separator as its OWN
        // segment: "a。b" -> ["a","。","b"], "。a" -> ["","。","a"]
        U32 cur;
        for (char32_t c : text) {
            if (inSet(kSplits, c)) {
                segs.push_back(std::move(cur));
                cur.clear();
                segs.push_back(U32(1, c));
            } else {
                cur.push_back(c);
            }
        }
        segs.push_back(std::move(cur));  // trailing segment ("" after sep)
    }
    std::vector<U32> result;
    U32 current;
    for (auto& seg : segs) {
        if (current.size() + seg.size() > kMaxLen) {
            result.push_back(std::move(current));
            current = seg;
        } else {
            current += seg;
        }
    }
    if (!current.empty()) result.push_back(std::move(current));
    return result;
}

// re.sub(r"\W+", "", text) == "" — python3 \w covers unicode letters and
// numbers. Approximation exact for BMP CJK/fullwidth inputs: word iff
// ASCII alnum / '_' or a non-ASCII codepoint outside the common symbol
// blocks (general punctuation, CJK punctuation, fullwidth punctuation).
bool isPureSymbol(const U32& t) {
    auto isWordChar = [](char32_t c) {
        if (c < 0x80)
            return (c >= U'0' && c <= U'9') || (c >= U'a' && c <= U'z') ||
                   (c >= U'A' && c <= U'Z') || c == U'_';
        if ((c >= 0x2000 && c <= 0x206F) ||  // general punctuation …—“”
            (c >= 0x3000 && c <= 0x303F) ||  // CJK punct 、。《》【】
            (c >= 0xFF01 && c <= 0xFF0F) ||  // ！＂＃＄％＆＇（）＊＋，－．／
            (c >= 0xFF1A && c <= 0xFF20) ||  // ：；＜＝＞？＠
            (c >= 0xFF3B && c <= 0xFF40) ||  // ［＼］＾＿｀
            (c >= 0xFF5B && c <= 0xFF65) ||  // ｛｜｝～｡｢｣､･
            (c >= 0x2E00 && c <= 0x2E7F) ||  // supplemental punctuation
            (c >= 0x1F000))                  // emoji / symbols
            return false;
        return true;
    };
    for (char32_t c : t)
        if (isWordChar(c)) return false;
    return true;
}

// merge_short_text_in_array(texts, threshold=5)
std::vector<U32> mergeShort(const std::vector<U32>& texts) {
    if (texts.size() < 2) return texts;
    std::vector<U32> result;
    U32 acc;
    for (auto& ele : texts) {
        acc += ele;
        if (acc.size() >= 5) {
            result.push_back(std::move(acc));
            acc.clear();
        }
    }
    if (!acc.empty()) {
        if (result.empty())
            result.push_back(std::move(acc));
        else
            result.back() += acc;
    }
    return result;
}

}  // namespace

std::vector<U32> TextFrontend::splitSentences(const U32& text,
                                              int cutMethod) {
    // pre_seg_text step 1-2
    U32 t = stripChars(text, dec("\n"));
    if (t.empty()) return {};
    // get_first(text): first re.split(SPLITS) chunk, stripped
    U32 first;
    for (char32_t c : t) {
        if (inSet(kSplits, c)) break;
        first.push_back(c);
    }
    first = stripPy(first);
    if (!inSet(kSplits, t.front()) && first.size() < 4)
        t.insert(0, dec("。"));

    // step 3: cut method
    std::vector<U32> pieces;
    switch (cutMethod) {
        case 0: pieces = cut0(t); break;
        case 1: pieces = cut1(t); break;
        case 2: pieces = cut2(t); break;
        case 3: pieces = cut3(t); break;
        case 4: pieces = cut4(t); break;
        case 5: pieces = cut5(t); break;
        default: return {};
    }

    // step 4-5: the pieces ARE the \n-split result already (joinNl dropped
    // empty strings like python's filter_text dropping ""/" ")
    auto texts = mergeShort(pieces);

    // step 7
    std::vector<U32> out;
    for (auto item : texts) {
        if (stripPy(item).empty()) continue;
        if (isPureSymbol(item)) continue;
        if (!inSet(kSplits, item.back())) item += dec("。");
        if (item.size() > 510) {
            // python texts.extend(split_big_text(item)) — keeps every piece,
            // including a possible leading empty one
            for (auto& big : splitBigText(item))
                out.push_back(std::move(big));
        } else {
            out.push_back(std::move(item));
        }
    }
    return out;
}

bool TextFrontend::load(const std::string& triePath,
                        const std::string& pinyinPath, std::string* err) {
    return g2p_.load(triePath, pinyinPath, err);
}

bool TextFrontend::process(const std::string& utf8Text, Result* out,
                           int cutMethod) const {
    out->sentences.clear();
    out->phones.clear();
    out->word2ph.clear();
    out->ok = false;
    out->error.clear();

    // preprocess() step 1: collapse consecutive punctuation globally
    U32 collapsed = collapsePunct(dec(utf8Text));

    auto segments = splitSentences(collapsed, cutMethod);
    for (auto& seg : segments) {
        G2pResult r;
        if (!g2p_.run(enc(seg), &r)) {
            out->error = "segment '" + enc(seg) + "': " + r.error;
            return false;
        }
        out->sentences.push_back(enc(seg));
        for (int id : r.phones) out->phones.push_back(id);
        for (int w : r.word2ph) out->word2ph.push_back(w);
    }
    out->ok = true;
    return true;
}

}  // namespace gsv::textfront
