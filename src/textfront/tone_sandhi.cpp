// tone_sandhi.cpp — see tone_sandhi.h. Line-level port of
// GPT-SoVITS-CPUFast/GPT_SoVITS/text/tone_sandhi.py.
#include "tone_sandhi.h"

#include <algorithm>
#include <cstdio>

#include "jieba.h"
#include "lexicon.hpp"

namespace gsv::textfront {
namespace {

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------

std::u32string toCps(const std::string& s) {
    std::u32string o;
    for (size_t i = 0; i < s.size();) {
        unsigned char b0 = static_cast<unsigned char>(s[i]);
        uint32_t cp = 0;
        size_t extra = 0;
        if (b0 < 0x80) { cp = b0; extra = 0; }
        else if ((b0 & 0xE0) == 0xC0) { cp = b0 & 0x1F; extra = 1; }
        else if ((b0 & 0xF0) == 0xE0) { cp = b0 & 0x0F; extra = 2; }
        else { cp = b0 & 0x07; extra = 3; }
        for (size_t k = 1; k <= extra; ++k)
            cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3F);
        i += extra + 1;
        o.push_back(static_cast<char32_t>(cp));
    }
    return o;
}

bool inSet(const char* const* arr, size_t n, std::u32string_view w) {
    for (size_t i = 0; i < n; ++i) {
        if (w == toCps(arr[i])) return true;
    }
    return false;
}

bool inMustNeural(std::u32string_view w) {
    return inSet(kMustNeuralToneWords, kMustNeuralToneWords_len, w);
}
bool inMustNotNeural(std::u32string_view w) {
    return inSet(kMustNotNeuralToneWords, kMustNotNeuralToneWords_len, w);
}

// python x[:-1] + digit over a str: drop the last CHARACTER. Returns false
// when the element is empty ("python raises").
bool popLastAppend(std::string* s, char digit) {
    if (s->empty()) return false;
    size_t i = s->size() - 1;
    while (i > 0 && (static_cast<unsigned char>((*s)[i]) & 0xC0) == 0x80) --i;
    if ((static_cast<unsigned char>((*s)[i]) & 0xC0) == 0x80) return false;
    s->resize(i);
    s->push_back(digit);
    return true;
}

// python elem[-1] == c; false when empty (callers that mirror unguarded
// python indexing translate false into the "python raises" outcome).
bool lastCharIs(const std::string& s, char c) {
    if (s.empty()) return false;
    size_t i = s.size() - 1;
    while (i > 0 && (static_cast<unsigned char>(s[i]) & 0xC0) == 0x80) --i;
    if ((static_cast<unsigned char>(s[i]) & 0xC0) != 0x80) return s[i] == c;
    return false;
}

// python word[-2:] — up to two trailing codepoints
std::u32string tailTwo(std::u32string_view w) {
    const size_t n = w.size();
    return std::u32string(w.substr(n >= 2 ? n - 2 : 0));
}

bool inCharClass(std::u32string_view set, char32_t c) {
    return set.find(c) != std::u32string_view::npos;
}

// lazy_pinyin(word, FINALS_TONE3, neutral_tone_with_five=True)
void finalsTone3List(const PypinyinResolver& resolver, const char32_t* word,
                     size_t len, std::vector<std::string>* out) {
    std::vector<PinyinSyl> syls;
    resolver.resolve(word, len, &syls);
    out->clear();
    for (const auto& s : syls) {
        out->push_back(s.raw ? s.text : toFinalsTone3(s.text));
    }
}

// _all_tone_three: every element ends with tone digit 3. An empty element
// means python would raise (''[-1]) -> *raised=true.
bool allToneThree(const std::vector<std::string>& f, bool* raised) {
    for (const auto& x : f) {
        if (!lastCharIs(x, '3')) {
            if (x.empty()) *raised = true;
            return false;
        }
    }
    return true;
}

bool isReduplication(std::u32string_view w) {
    return w.size() == 2 && w[0] == w[1];
}

// ---------------------------------------------------------------------------
// merge helpers (mirror ToneSandhi._merge_*)
// ---------------------------------------------------------------------------

void mergeBu(std::vector<SegToken>* seg) {
    std::vector<SegToken> out;
    std::u32string last;
    for (const auto& t : *seg) {
        std::u32string w = t.word;
        if (last == U"不") w.insert(0, U"不");
        if (w != U"不") out.push_back({w, t.flag});
        last = w;
    }
    if (last == U"不") out.push_back({U"不", "d"});
    *seg = std::move(out);
}

// returns false when python would raise (new_seg[-1] on an empty list)
bool mergeYi(std::vector<SegToken>* seg) {
    std::vector<SegToken> ns;
    size_t i = 0;
    while (i < seg->size()) {
        const SegToken cur = (*seg)[i];
        bool merged = false;
        if (i >= 1 && cur.word == U"一" && i + 1 < seg->size()) {
            // last = new_seg[-1] if new_seg else seg[i-1]
            bool haveLast = !ns.empty();
            const SegToken last = haveLast ? ns.back() : (*seg)[i - 1];
            const SegToken next = (*seg)[i + 1];
            if (last.word == next.word && last.flag == "v" &&
                next.flag == "v") {
                if (!haveLast) return false;  // new_seg[-1] assignment
                SegToken m;
                m.word = last.word + U"一" + next.word;
                m.flag = last.flag;
                ns.back() = std::move(m);
                i += 2;
                merged = true;
            }
        }
        if (!merged) {
            ns.push_back(cur);
            i += 1;
        }
    }
    std::vector<SegToken> out;
    for (const auto& t : ns) {
        if (!out.empty() && out.back().word == U"一") {
            out.back().word += t.word;
        } else {
            out.push_back(t);
        }
    }
    *seg = std::move(out);
    return true;
}

void mergeReduplication(std::vector<SegToken>* seg) {
    std::vector<SegToken> out;
    for (const auto& t : *seg) {
        if (!out.empty() && out.back().word == t.word) {
            out.back().word += t.word;
        } else {
            out.push_back(t);
        }
    }
    *seg = std::move(out);
}

// shared body of _merge_continuous_three_tones (mode2=false) and
// _merge_continuous_three_tones_2 (mode2=true); false = python raised.
bool mergeContinuousThreeTones(std::vector<SegToken>* seg,
                               const PypinyinResolver& resolver, bool mode2) {
    const size_t n = seg->size();
    std::vector<std::vector<std::string>> fl(n);
    for (size_t i = 0; i < n; ++i) {
        finalsTone3List(resolver, (*seg)[i].word.data(),
                        (*seg)[i].word.size(), &fl[i]);
    }
    std::vector<char> mergeLast(n, 0);
    std::vector<SegToken> out;
    for (size_t i = 0; i < n; ++i) {
        bool cond = false;
        if (i >= 1 && !mergeLast[i - 1]) {
            bool raised = false;
            bool c;
            if (mode2) {
                // fl[i-1][-1][-1]=='3' and fl[i][0][-1]=='3'
                c = !fl[i - 1].empty() && !fl[i].empty() &&
                    lastCharIs(fl[i - 1].back(), '3') &&
                    lastCharIs(fl[i].front(), '3');
                if ((fl[i - 1].empty() || fl[i].empty())) raised = true;
            } else {
                c = allToneThree(fl[i - 1], &raised) &&
                    allToneThree(fl[i], &raised);
            }
            if (raised) {
                fprintf(stderr, "_merge_continuous_three_tones failed\n");
                return false;
            }
            cond = c;
        }
        if (cond) {
            if (!isReduplication((*seg)[i - 1].word) &&
                (*seg)[i - 1].word.size() + (*seg)[i].word.size() <= 3) {
                if (out.empty()) {
                    fprintf(stderr, "_merge_continuous_three_tones failed\n");
                    return false;
                }
                out.back().word += (*seg)[i].word;
                mergeLast[i] = 1;
            } else {
                out.push_back((*seg)[i]);
            }
        } else {
            out.push_back((*seg)[i]);
        }
    }
    *seg = std::move(out);
    return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// ToneSandhi public API
// ---------------------------------------------------------------------------

void ToneSandhi::preMergeForModify(std::vector<SegToken>* seg,
                                   const PypinyinResolver& resolver) const {
    mergeBu(seg);
    if (!mergeYi(seg)) fprintf(stderr, "_merge_yi failed\n");
    mergeReduplication(seg);
    if (!mergeContinuousThreeTones(seg, resolver, /*mode2=*/false))
        fprintf(stderr, "_merge_continuous_three_tones failed\n");
    if (!mergeContinuousThreeTones(seg, resolver, /*mode2=*/true))
        fprintf(stderr, "_merge_continuous_three_tones_2 failed\n");
    // _merge_er
    {
        std::vector<SegToken> out;
        for (size_t i = 0; i < seg->size(); ++i) {
            const SegToken& t = (*seg)[i];
            if (i >= 1 && t.word == U"儿" && (*seg)[i - 1].word != U"#") {
                out.back().word += t.word;
            } else {
                out.push_back(t);
            }
        }
        *seg = std::move(out);
    }
}

namespace {

// jieba.cut_for_search(word) -> stable-sorted by length -> two-part split
bool splitWord(const JiebaSegmenter* jb, const std::u32string& word,
               std::u32string* first, std::u32string* second) {
    std::string utf8;
    for (char32_t c : word) {
        if (c < 0x80) utf8 += static_cast<char>(c);
        else if (c < 0x800) {
            utf8 += static_cast<char>(0xC0 | (c >> 6));
            utf8 += static_cast<char>(0x80 | (c & 0x3F));
        } else if (c < 0x10000) {
            utf8 += static_cast<char>(0xE0 | (c >> 12));
            utf8 += static_cast<char>(0x80 | ((c >> 6) & 0x3F));
            utf8 += static_cast<char>(0x80 | (c & 0x3F));
        } else {
            utf8 += static_cast<char>(0xF0 | (c >> 18));
            utf8 += static_cast<char>(0x80 | ((c >> 12) & 0x3F));
            utf8 += static_cast<char>(0x80 | ((c >> 6) & 0x3F));
            utf8 += static_cast<char>(0x80 | (c & 0x3F));
        }
    }
    std::vector<std::string> parts;
    jb->cutForSearch(utf8, &parts);
    if (parts.empty()) return false;
    std::vector<std::pair<size_t, std::u32string>> dec;
    dec.reserve(parts.size());
    for (auto& t : parts) dec.emplace_back(0, toCps(t));
    for (auto& d : dec) d.first = d.second.size();
    std::stable_sort(dec.begin(), dec.end(),
                     [](const auto& a, const auto& b) { return a.first < b.first; });
    const std::u32string& fs = dec[0].second;
    size_t pos = std::u32string::npos;
    if (fs.size() <= word.size() && !fs.empty()) {
        for (size_t i = 0; i + fs.size() <= word.size(); ++i) {
            if (std::u32string_view(word).substr(i, fs.size()) == fs) {
                pos = i;
                break;
            }
        }
    } else if (!fs.empty()) {
        pos = static_cast<size_t>(-2);  // force the "else" branch like python
    }
    if (fs.empty()) return false;  // str.find('')==0 path; cannot happen from cut_for_search
    if (pos == 0) {
        // [first_subword, second_subword]
        *first = fs;
        *second = word.substr(fs.size());
    } else {
        // [second_subword, first_subword] with second = word[:-len(first)]
        *first = word.substr(0,
                             word.size() >= fs.size() ? word.size() - fs.size() : 0);
        *second = fs;
    }
    return true;
}

}  // namespace

void ToneSandhi::modifiedTone(const std::u32string& word, std::string_view pos,
                              const PypinyinResolver& resolver,
                              std::vector<std::string>* finals,
                              bool* ok) const {
    *ok = true;
    const size_t n = word.size();

    // ---- _bu_sandhi ----
    if (n == 3 && word[1] == U'不') {
        if (!popLastAppend(&(*finals)[1], '5')) { *ok = false; return; }
    } else {
        for (size_t i = 0; i < n && *ok; ++i) {
            if (word[i] == U'不' && i + 1 < n &&
                lastCharIs((*finals)[i + 1], '4')) {
                if (!popLastAppend(&(*finals)[i], '2')) { *ok = false; return; }
            }
        }
        if (!*ok) return;
    }

    // ---- _yi_sandhi ----
    {
        bool hasYi = false, allNumeric = true;
        for (char32_t c : word) {
            if (c == U'一') hasYi = true;
            else if (!(c >= U'0' && c <= U'9') &&
                     !resolver.isNumeric(static_cast<uint32_t>(c)))
                allNumeric = false;
        }
        static const std::u32string kPuncU32 = toCps(kSandhiPunc);
        if (hasYi && allNumeric) {
            // number sequence containing 一: unchanged
        } else if (n == 3 && word[1] == U'一' && word[0] == word[2]) {
            if (!popLastAppend(&(*finals)[1], '5')) { *ok = false; return; }
        } else if (n >= 2 && word[0] == U'第' && word[1] == U'一') {
            if (!popLastAppend(&(*finals)[1], '1')) { *ok = false; return; }
        } else {
            for (size_t i = 0; i < n; ++i) {
                if (word[i] != U'一' || i + 1 >= n) continue;
                if (lastCharIs((*finals)[i + 1], '4')) {
                    if (!popLastAppend(&(*finals)[i], '2')) { *ok = false; return; }
                } else if (kPuncU32.find(word[i + 1]) == std::u32string::npos) {
                    if (!popLastAppend(&(*finals)[i], '4')) { *ok = false; return; }
                }
            }
        }
    }

    // ---- _neural_sandhi ----
    {
        for (size_t j = 0; j < n && *ok; ++j) {
            if (j >= 1 && word[j] == word[j - 1] && !pos.empty() &&
                (pos[0] == 'n' || pos[0] == 'v' || pos[0] == 'a') &&
                !inMustNotNeural(word)) {
                if (!popLastAppend(&(*finals)[j], '5')) { *ok = false; return; }
            }
        }
        if (!*ok) return;

        long geIdx = -1;
        for (size_t i = 0; i < n; ++i)
            if (word[i] == U'个') { geIdx = static_cast<long>(i); break; }
        static const std::u32string_view kTailNeural =
            U"吧呢哈啊呐噻嘛吖嗨呐哦哒额滴哩哟喽啰耶喔诶";
        static const std::u32string_view kDe = U"的地得";
        static const std::u32string_view kLeZheGuo = U"了着过";
        static const std::u32string_view kMenZi = U"们子";
        static const std::u32string_view kShangXiaLi = U"上下里";
        static const std::u32string_view kLaiQu = U"来去";
        static const std::u32string_view kLaiQuPrev = U"上下进出回过起开";
        static const std::u32string_view kGePrev = U"几有两半多各整每做是";
        const std::string posStr(pos);

        auto neutralizeBack = [&]() {
            if (finals->empty() || !popLastAppend(&finals->back(), '5')) {
                *ok = false;
            }
        };
        if (n >= 1 && inCharClass(kTailNeural, word[n - 1])) {
            neutralizeBack();
        } else if (n >= 1 && inCharClass(kDe, word[n - 1])) {
            neutralizeBack();
        } else if (n == 1 && inCharClass(kLeZheGuo, word[0]) &&
                   (posStr == "ul" || posStr == "uz" || posStr == "ug")) {
            neutralizeBack();
        } else if (n > 1 && inCharClass(kMenZi, word[n - 1]) &&
                   (posStr == "r" || posStr == "n") &&
                   !inMustNotNeural(word)) {
            neutralizeBack();
        } else if (n > 1 && inCharClass(kShangXiaLi, word[n - 1]) &&
                   (posStr == "s" || posStr == "l" || posStr == "f")) {
            neutralizeBack();
        } else if (n > 1 && inCharClass(kLaiQu, word[n - 1]) &&
                   inCharClass(kLaiQuPrev, word[n - 2])) {
            neutralizeBack();
        } else if ((geIdx >= 1 &&
                    (resolver.isNumeric(static_cast<uint32_t>(word[geIdx - 1])) ||
                     inCharClass(kGePrev, word[geIdx - 1]))) ||
                   word == U"个") {
            const size_t idx = geIdx >= 0 ? static_cast<size_t>(geIdx) : 0;
            if (idx >= finals->size() ||
                !popLastAppend(&(*finals)[idx], '5'))
                *ok = false;
        } else if (!finals->empty()) {
            if (inMustNeural(word) || inMustNeural(tailTwo(word)))
                neutralizeBack();
        } else if (inMustNeural(word) || inMustNeural(tailTwo(word))) {
            *ok = false;  // finals[-1] on empty
        }
        if (!*ok) return;

        // conventional neural via _split_word
        std::u32string first, second;
        if (!splitWord(jb_, word, &first, &second)) { *ok = false; return; }
        const size_t cutRaw = first.size();
        // python finals[:cut] clamps an oversized cut safely
        const size_t cut = cutRaw < finals->size() ? cutRaw : finals->size();
        std::vector<std::string> partA(finals->begin(),
                                       finals->begin() + static_cast<long>(cut));
        std::vector<std::string> partB(finals->begin() + static_cast<long>(cut),
                                       finals->end());
        const std::u32string* subs[2] = {&first, &second};
        std::vector<std::string>* parts[2] = {&partA, &partB};
        for (int i = 0; i < 2; ++i) {
            const std::u32string& w = *subs[i];
            if (w.empty()) continue;  // "" not in / ""[-2:]="" not in set
            if (inMustNeural(w) || inMustNeural(tailTwo(w))) {
                if (parts[i]->empty() ||
                    !popLastAppend(&parts[i]->back(), '5')) {
                    *ok = false;  // finals_list[i][-1] IndexError
                    return;
                }
            }
        }
        finals->assign(partA.begin(), partA.end());
        finals->insert(finals->end(), partB.begin(), partB.end());
    }

    // ---- _three_sandhi ----
    {
        bool raised = false;
        const bool all3 = allToneThree(*finals, &raised);
        if (raised) { *ok = false; return; }
        if (n == 2 && all3) {
            if (!popLastAppend(&(*finals)[0], '2')) { *ok = false; return; }
        } else if (n == 3) {
            std::u32string first, second;
            if (!splitWord(jb_, word, &first, &second)) { *ok = false; return; }
            if (all3) {
                if (first.size() == 2) {
                    if (!popLastAppend(&(*finals)[0], '2')) { *ok = false; return; }
                    if (!popLastAppend(&(*finals)[1], '2')) { *ok = false; return; }
                } else if (first.size() == 1) {
                    if (!popLastAppend(&(*finals)[1], '2')) { *ok = false; return; }
                }
            } else {
                const size_t cutRaw = first.size();
                const size_t cut =
                    cutRaw < finals->size() ? cutRaw : finals->size();
                std::vector<std::string> partA(finals->begin(),
                                               finals->begin() + static_cast<long>(cut));
                std::vector<std::string> partB(finals->begin() + static_cast<long>(cut),
                                               finals->end());
                if (partA.size() + partB.size() == finals->size()) {
                    for (int i = 0; i < 2; ++i) {
                        std::vector<std::string>& sub =
                            i == 0 ? partA : partB;
                        bool r2 = false;
                        const bool s3 = allToneThree(sub, &r2);
                        if (r2) { *ok = false; return; }
                        if (s3 && sub.size() == 2) {
                            if (!popLastAppend(&sub[0], '2')) { *ok = false; return; }
                        } else if (i == 1 && !s3 && !sub.empty() &&
                                   !partA.empty() &&
                                   lastCharIs(sub[0], '3') &&
                                   lastCharIs(partA.back(), '3')) {
                            if (!popLastAppend(&partA.back(), '2')) {
                                *ok = false;
                                return;
                            }
                        }
                    }
                }
                finals->assign(partA.begin(), partA.end());
                finals->insert(finals->end(), partB.begin(), partB.end());
            }
        } else if (n == 4) {
            // python finals[:2]/finals[2:] — a short list yields an empty
            // sublist whose all() is vacuously true, then sub[0] raises.
            const size_t half = finals->size() < 2 ? finals->size() : 2;
            std::vector<std::string> partA(finals->begin(), finals->begin() + static_cast<long>(half));
            std::vector<std::string> partB(finals->begin() + static_cast<long>(half), finals->end());
            std::vector<std::string> flat;
            flat.reserve(finals->size());
            for (int i = 0; i < 2; ++i) {
                std::vector<std::string>& sub = i == 0 ? partA : partB;
                bool r2 = false;  // r2 set only when an element is empty
                const bool s3 = allToneThree(sub, &r2);
                if (r2) { *ok = false; return; }
                if (s3) {
                    if (sub.empty() || !popLastAppend(&sub[0], '2')) {
                        *ok = false;  // sub[0] on [] -> IndexError
                        return;
                    }
                }
                flat.insert(flat.end(), sub.begin(), sub.end());
            }
            *finals = std::move(flat);
        }
    }
}

}  // namespace gsv::textfront
