// langsegmenter.cpp — 见 langsegmenter.hpp。split_lang(splitter.py/detector.py/
// utils.py/config.py) + langsegmenter.py + fast_langdetect 预处理的逐行移植。
//
// 已核实的上游关键怪癖(全部保留, fixture 位级对照通过为准):
//   * _get_languages 是 no-op(smart_merge 里算完 cur_lang 从不回写!)—
//     子串语种只在 _init_substr_lang 时确定一次
//   * getTexts 设 merge_across_digit=False → 跨数字合并被跳过;
//     数字片由 langsegmenter.py 的 9 分支上下文归属兜底
//   * 检测前预处理: '\n'→' ', >80 码点截断, 全大写或大写占比>0.8 且长度>5
//     转小写(fast-langdetect normalize_input/max_input_length 默认值)
//   * 单字"折"被 fasttext 判 ja 属模型固有行为(fixture 原样保留)
#include "langsegmenter.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>

#include "runtime/mini_json.hpp"

namespace gsv::textfront {
namespace {

using gsv::rt::json::JValue;

// ------------------------------ 基础工具 ------------------------------

inline bool cpIsDigitPy(char32_t c) { return c >= U'0' && c <= U'9'; }

inline bool isSpacePy(char32_t c) {  // python str.isspace() BMP 口径
    switch (c) {
        case U' ': case U'\t': case U'\n': case U'\r': case U'\v':
        case U'\f': case 0x1C: case 0x1D: case 0x1E: case 0x1F:
        case 0x85: case 0xA0: case 0x1680:
            return true;
        default:
            return (c >= 0x2000 && c <= 0x200A) || c == 0x2028 ||
                   c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
    }
}

inline bool containsHangulCp(char32_t c) { return c >= 0xAC00 && c <= 0xD7AF; }
inline bool containsZhJaCp(char32_t c) {
    return (c >= 0x4E00 && c <= 0x9FFF) || (c >= 0x3040 && c <= 0x30FF) ||
           c == 0x3005;  // utils.py zh_ja_pattern 含 々(0x3005)
}
inline bool containsJaKana(const std::u32string& t) {
    for (char32_t c : t)
        if ((c >= 0x3040 && c <= 0x30FF) || c == 0x3005) return true;
    return false;
}

std::string u32to8(const std::u32string& t) {
    std::string o;
    for (char32_t c : t) {
        if (c < 0x80) { o.push_back(char(c)); continue; }
        unsigned char b[4]; int n;
        if (c < 0x800) { n = 2; b[0] = 0xC0 | (c >> 6); }
        else if (c < 0x10000) { n = 3; b[0] = 0xE0 | (c >> 12); }
        else { n = 4; b[0] = 0xF0 | (c >> 18); }
        for (int k = 1; k < n; ++k)
            b[k] = 0x80 | ((c >> (6 * (n - k - 1))) & 0x3F);
        o.append(reinterpret_cast<char*>(b), size_t(n));
    }
    return o;
}

std::u32string u8to32(const std::string& s) {
    std::u32string o;
    for (size_t i = 0; i < s.size();) {
        unsigned char b = uint8_t(s[i]);
        if (b < 0x80) { o.push_back(char32_t(b)); i++; continue; }
        uint32_t cp = 0; int extra = 0;
        if ((b & 0xE0) == 0xC0) { cp = b & 0x1Fu; extra = 1; }
        else if ((b & 0xF0) == 0xE0) { cp = b & 0x0Fu; extra = 2; }
        else if ((b & 0xF8) == 0xF0) { cp = b & 0x07u; extra = 3; }
        else { o.push_back(U'\uFFFD'); i++; continue; }
        for (int k = 1; k <= extra && i + size_t(k) < s.size(); ++k)
            cp = (cp << 6) | (uint8_t(s[i + size_t(k)]) & 0x3Fu);
        i += size_t(extra) + 1;
        o.push_back(char32_t(cp));
    }
    return o;
}

// PUNCTUATION 字符集(split_lang/split/utils.py 原文常量)
const char* kPunctUtf8 =
    "〜~,.;:!?，。！？；：、·([{<（【《〈「『“‘)]}>）】》〉」』”’\"-_———#$%&……￥"
    "'*+<=>?@[\\]^_`{|}~";

const std::u32string& punctSet() {
    static const std::u32string set = [] {
        std::u32string s = u8to32(kPunctUtf8);
        std::sort(s.begin(), s.end());
        s.erase(std::unique(s.begin(), s.end()), s.end());
        return s;
    }();
    return set;
}

bool inPunctSet(char32_t c) {
    const std::u32string& s = punctSet();
    return std::binary_search(s.begin(), s.end(), c);
}

// ------------------------------ Budoux ------------------------------
class BudouxParser {
public:
    bool loadJson(const std::string& path, std::string* err);
    bool ready() const { return loaded_; }

    std::vector<std::string> parse(const std::string& utf8) const;

private:
    struct Cat {
        std::unordered_map<std::u32string, int> uni;   // 单字键(UW*)
        std::unordered_map<std::u32string, int> multi; // 多字键(BW*/TW*)
    };
    Cat uw_[7];  // [1..6]
    Cat bw_[4];  // [1..3]
    Cat tw_[5];  // [1..4]
    double baseScore_ = 0;
    bool loaded_ = false;
};

bool BudouxParser::loadJson(const std::string& path, std::string* err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        if (err) *err = "无法打开 budoux 模型: " + path;
        return false;
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    std::string bytes = ss.str();
    JValue root;
    try {
        root = gsv::rt::json::parse(bytes.data(), bytes.size());
    } catch (const std::exception& e) {
        if (err) *err = "budoux JSON 解析失败 " + path + ": " + e.what();
        return false;
    }
    long total = 0;
    auto fill = [&](Cat& cat, const JValue* v) {
        if (!v || !v->is(gsv::rt::json::JType::Object)) return;
        for (const auto& kv : v->obj) {
            int score = kv.second.is(gsv::rt::json::JType::Int)
                            ? int(kv.second.i)
                            : int(kv.second.d);
            std::u32string key = u8to32(kv.first);
            if (key.size() == 1) cat.uni[key] = score;
            else cat.multi[key] = score;
            total += score;
        }
    };
    static const char* uwNames[] = {"UW1", "UW2", "UW3", "UW4", "UW5", "UW6"};
    for (int n = 1; n <= 6; ++n)
        fill(uw_[size_t(n)], root.find(uwNames[n - 1]));
    static const char* bwNames[] = {"BW1", "BW2", "BW3"};
    for (int n = 1; n <= 3; ++n)
        fill(bw_[size_t(n)], root.find(bwNames[n - 1]));
    static const char* twNames[] = {"TW1", "TW2", "TW3", "TW4"};
    for (int n = 1; n <= 4; ++n)
        fill(tw_[size_t(n)], root.find(twNames[n - 1]));
    baseScore_ = -double(total) * 0.5;
    loaded_ = true;
    return true;
}

std::vector<std::string> BudouxParser::parse(const std::string& utf8) const {
    std::u32string sent = u8to32(utf8);
    if (sent.empty()) return {};
    std::vector<std::string> chunks;
    std::u32string cur(1, sent[0]);
    auto lu1 = [&](const Cat& c, const std::u32string& k) -> int {
        auto it = c.uni.find(k);
        return it == c.uni.end() ? 0 : it->second;
    };
    auto luN = [&](const Cat& c, const std::u32string& k) -> int {
        auto it = c.multi.find(k);
        return it == c.multi.end() ? 0 : it->second;
    };
    auto sub = [&](size_t a, size_t b) { return sent.substr(a, b - a); };
    for (size_t i = 1; i < sent.size(); ++i) {
        double score = baseScore_;
        if (i > 2) score += lu1(uw_[1], sub(i - 3, i - 2));
        if (i > 1) score += lu1(uw_[2], sub(i - 2, i - 1));
        score += lu1(uw_[3], sub(i - 1, i));
        score += lu1(uw_[4], sub(i, i + 1));
        if (i + 1 < sent.size()) score += lu1(uw_[5], sub(i + 1, i + 2));
        if (i + 2 < sent.size()) score += lu1(uw_[6], sub(i + 2, i + 3));

        if (i > 1) score += luN(bw_[1], sub(i - 2, i));
        score += luN(bw_[2], sub(i - 1, i + 1));
        if (i + 1 < sent.size()) score += luN(bw_[3], sub(i, i + 2));

        if (i > 2) score += luN(tw_[1], sub(i - 3, i));
        if (i > 1) score += luN(tw_[2], sub(i - 2, i + 1));
        if (i + 1 < sent.size()) score += luN(tw_[3], sub(i - 1, i + 2));
        if (i + 2 < sent.size()) score += luN(tw_[4], sub(i, i + 3));

        if (score > 0) {
            chunks.push_back(u32to8(cur));
            cur.clear();
        }
        cur.push_back(sent[i]);
    }
    chunks.push_back(u32to8(cur));
    return chunks;
}

// 双解析器单例(ja 先跑、zh-hans 后跑 —— splitter.py 模块级初始化)
struct BudouxHolder {
    BudouxParser jp, zh;
};
BudouxHolder& budouxHolder() {
    static BudouxHolder h;
    return h;
}

// ------------------------ fast-langdetect 预处理 ------------------------
std::string preprocessDetect(const std::string& utf8In) {
    std::u32string t = u8to32(utf8In);
    for (auto& c : t)
        if (c == U'\n') c = U' ';
    if (t.size() > 80) t.resize(80);  // max_input_length=80
    // fast_langdetect._normalize_text(normalize_input=True):
    //   text.isupper() → 全小写; 或 大写字母数 > 0.8*字母总数 且 len>5 → 小写
    auto isUpperA = [](char32_t c) { return c >= U'A' && c <= U'Z'; };
    auto isLowerA = [](char32_t c) { return c >= U'a' && c <= U'z'; };
    size_t upper = 0, alpha = 0;
    bool anyCased = false, allCasedUpper = true;
    for (char32_t c : t) {
        if (isUpperA(c)) {
            ++upper;
            ++alpha;
            anyCased = true;
        } else if (isLowerA(c)) {
            ++alpha;
            anyCased = true;
            allCasedUpper = false;
        }
    }
    bool doLower = (anyCased && allCasedUpper) ||
                   (upper * 10 > alpha * 8 && t.size() > 5);
    if (!doLower) return u32to8(t);
    for (auto& c : t)
        if (isUpperA(c)) c = c + 32;
    return u32to8(t);
}

// python fasttext detect 的对齐语义: detector.detect(text,k[,threshold])
// 已由 FastTextLid 实现; 这里只做预处理包装。
std::string detectOnce(const FastTextLid& lid, const std::u32string& text,
                       std::string* lowerOut) {
    std::vector<FastTextLid::Pred> preds;
    lid.predict(preprocessDetect(u32to8(text)), 1, 0.0f, &preds);
    if (preds.empty()) return {};
    std::string l = preds[0].lang;
    for (auto& c : l) c = char(tolower(uint8_t(c)));
    if (lowerOut) *lowerOut = l;
    return l;
}

std::vector<std::string> possibleDetectionList(const FastTextLid& lid,
                                               const std::u32string& text) {
    // detector.possible_detection_list: 去 '\n' + strip 再 k=5/threshold=0.01
    std::u32string t;
    for (char32_t c : text)
        if (c != U'\n') t.push_back(c);
    size_t b = 0, e = t.size();
    while (b < e && isSpacePy(t[b])) ++b;
    while (e > b && isSpacePy(t[e - 1])) --e;
    t = t.substr(b, e - b);
    std::vector<FastTextLid::Pred> preds;
    lid.predict(preprocessDetect(u32to8(t)), 5, 0.01f, &preds);
    std::vector<std::string> out;
    for (auto& p : preds) {
        std::string l = p.lang;
        for (auto& c : l) c = char(tolower(uint8_t(c)));
        out.push_back(std::move(l));
    }
    return out;
}

std::string detectCombined(const FastTextLid& lid, const std::u32string& text,
                           int langSectionType /*0 others, 2 zh_ja*/) {
    if (langSectionType == 2 && containsJaKana(text)) return "ja";
    return detectOnce(lid, text, nullptr);
}

}  // namespace

// ===========================================================================
// split_lang LangSplitter 全链 (splitter.py)
// ===========================================================================
namespace {

enum class SecType { ZhJa = 0, Ko, Digit, Newline, Punct, Others };

struct SubStr {
    std::string lang;
    std::u32string text;
    int index = 0;
    size_t length = 0;
};

struct Section {
    SecType type = SecType::Others;
    std::u32string text;
    std::vector<SubStr> subs;
};

class Splitter {
public:
    explicit Splitter(const FastTextLid* lid) : lid_(lid) {}

    // merge_across_digit=False / 其余开关为默认 True 的 split_by_lang
    std::vector<SubStr> splitByLang(const std::u32string& text) const {
        std::vector<Section> sections = preSplit(text);
        sections = doSplit(sections);
        sections = mergeAcrossPunct(std::move(sections));
        sections = mergeAcrossNewline(std::move(sections));
        for (auto& sec : sections)
            if (sec.type == SecType::ZhJa)
                sec.subs = specialMergeZhJa(sec.subs);
        std::vector<SubStr> out;
        for (auto& sec : sections)
            out.insert(out.end(), sec.subs.begin(), sec.subs.end());
        return out;
    }

private:
    const FastTextLid* lid_;

    // ZH_JA_LANG_MAP / LangSegmenter.DEFAULT_LANG_MAP 见 initSubstrLang

    std::vector<Section> preSplit(const std::u32string& rawText) const {
        std::u32string text = strip(rawText);
        std::vector<Section> sections;
        SecType cur = SecType::Others;
        std::u32string acc;
        auto flush = [&]() {
            if (!acc.empty()) {
                sections.push_back(Section{cur, acc, {}});
                acc.clear();
            }
        };
        for (size_t index = 0; index < text.size(); ++index) {
            char32_t ch = text[index];
            if (containsZhJaCp(ch)) {
                if (cur != SecType::ZhJa) { flush(); cur = SecType::ZhJa; }
            } else if (containsHangulCp(ch)) {
                if (cur != SecType::Ko) { flush(); cur = SecType::Ko; }
            } else if (cpIsDigitPy(ch)) {
                if (cur != SecType::Digit) { flush(); cur = SecType::Digit; }
            } else if (inPunctSet(ch)) {
                // For English, "'" is a part of word(前字符非空白时并入当前词)
                if (!(ch == U'\'' && index > 0 && !isSpacePy(text[index - 1]))) {
                    flush();
                    cur = SecType::Punct;
                }
            } else if (isSpacePy(ch)) {
                if (ch == U'\n' || ch == U'\r') {
                    flush();
                    cur = SecType::Newline;
                }
            } else {
                if (cur != SecType::Others) { flush(); cur = SecType::Others; }
            }
            acc.push_back(ch);
        }
        flush();
        return sections;
    }

    std::vector<Section> doSplit(std::vector<Section> secs) const {
        int sectionIndex = 0;
        for (auto& section : secs) {
            size_t sectionLen = section.text.size();  // u32: 码点数
            switch (section.type) {
                // 注: python 各分支都以当前 section_index 为 substr 初值;
                // 下面统一先置 0, 循环尾的 += sectionIndex 一次性归位
                case SecType::Punct:
                    section.subs.push_back(
                        {"punctuation", section.text, 0, sectionLen});
                    break;
                case SecType::Digit:
                    section.subs.push_back(
                        {"digit", section.text, 0, sectionLen});
                    break;
                case SecType::Newline:
                    section.subs.push_back(
                        {"newline", section.text, 0, sectionLen});
                    break;
                case SecType::ZhJa: {
                    auto texts = parseZhJa(section.text);
                    section.subs = initSubstrLang(texts, SecType::ZhJa);
                    break;
                }
                case SecType::Ko:
                    section.subs.push_back({"ko", section.text, 0, sectionLen});
                    break;
                case SecType::Others: {
                    auto texts = parseWithoutZhJa(section.text);
                    section.subs = initSubstrLang(texts, SecType::Others);
                    break;
                }
            }
            for (auto& s : section.subs) s.index += sectionIndex;
            sectionIndex += int(sectionLen);  // sectionLen 已是码点数
        }
        return smartMergeAll(secs);
    }

    std::vector<SubStr> initSubstrLang(const std::vector<std::u32string>& texts,
                                       SecType st) const {
        // _init_substr_lang: ZH_JA 段用 ZH_JA_LANG_MAP;
        // 其他段用 self.lang_map = LangSegmenter.DEFAULT_LANG_MAP(gsv 版:
        //   zh/yue/wuu/zh-cn→zh, ko→ko, ja→ja, en→en; zh-tw→x 不在此路)
        std::vector<SubStr> out;
        int idx = 0;
        bool zhja = st == SecType::ZhJa;
        for (auto& t : texts) {
            std::string det = detectCombined(*lid_, t, zhja ? 2 : 0);
            std::string lang = gsMap(det, zhja);
            out.push_back(SubStr{lang, t, idx, t.size()});
            idx += int(t.size());
        }
        return out;
    }

    static const std::unordered_map<std::string, std::string>& gsDefaultMap() {
        static const std::unordered_map<std::string, std::string> m{
            {"zh", "zh"},   {"yue", "zh"}, {"wuu", "zh"},
            {"zh-cn", "zh"}, {"zh-tw", "x"}, {"ko", "ko"},
            {"ja", "ja"},   {"en", "en"}};
        return m;
    }
    static const std::unordered_map<std::string, std::string>& zhJaMap() {
        static const std::unordered_map<std::string, std::string> m{
            {"zh", "zh"},    {"yue", "zh"},  {"wuu", "zh"},
            {"zh-cn", "zh"}, {"zh-tw", "x"}, {"ja", "ja"}};
        return m;
    }
    static std::string gsMap(const std::string& det, bool zhja) {
        const auto& m = zhja ? zhJaMap() : gsDefaultMap();
        auto it = m.find(det);
        return it != m.end() ? it->second : std::string("x");
    }

    std::vector<Section> smartMergeAll(std::vector<Section> secs) const {
        for (auto& section : secs) {
            if (section.type == SecType::Punct ||
                section.type == SecType::Newline)
                continue;
            section.subs = smartMerge(section.subs, section.type);
        }
        return secs;
    }

    // ---- _parse_zh_ja / _parse_without_zh_ja ----
    std::vector<std::u32string> parseWithoutZhJa(
        const std::u32string& text) const {
        std::vector<std::u32string> words;
        bool existSpace = false;
        std::u32string chars;
        for (char32_t ch : text) {
            if (!isSpacePy(ch)) {
                if (existSpace) {
                    words.push_back(chars);
                    chars.clear();
                    existSpace = false;
                }
                chars.push_back(ch);
            } else {
                existSpace = true;
                chars.push_back(ch);
            }
        }
        if (!chars.empty()) words.push_back(chars);
        return words;
    }

    std::vector<std::u32string> parseZhJa(const std::u32string& text) const {
        const BudouxHolder& h = budouxHolder();
        std::vector<std::string> zhJp;
        for (auto& s : h.jp.parse(u32to8(text)))
            for (auto& z : h.zh.parse(s)) zhJp.push_back(std::move(z));
        std::vector<std::u32string> ns;
        for (auto& raw : zhJp) {
            std::u32string cur = u8to32(raw);
            if (ns.empty()) { ns.push_back(cur); continue; }
            bool lKana = containsJaKana(ns.back());
            bool cKana = containsJaKana(cur);
            bool same = lKana == cKana;
            if (lKana && same) ns.back() += cur;
            else if (!lKana && same && cur.size() == 1) ns.back() += cur;
            else ns.push_back(cur);
        }
        if (ns.size() >= 2 && ns[0].size() == 1) {
            ns[1] = ns[0] + ns[1];
            ns.erase(ns.begin());
        }
        return ns;
    }

    // ---- smart merge 系列 ----
    static std::vector<SubStr> mergeSubstrings(std::vector<SubStr> subs) {
        std::vector<SubStr> out;
        std::string lastLang;
        for (auto& b : subs) {
            if (out.empty() || b.lang != lastLang) out.push_back(b);
            else {
                out.back().text += b.text;
                out.back().length += b.length;
            }
            lastLang = b.lang;
        }
        return out;
    }

    bool isMergeMiddleToTwoSide(const SubStr& left, const SubStr& mid,
                                const SubStr& right) const {
        bool noKana = !containsJaKana(mid.text);
        auto pl = possibleDetectionList(*lid_, mid.text);
        bool leftPossible =
            std::find(pl.begin(), pl.end(), left.lang) != pl.end();
        bool midX = mid.lang == "x";
        bool shortMidLongSides =
            mid.length <= 4 && left.length + right.length >= 6 && noKana;
        return (leftPossible && noKana) || midX || shortMidLongSides;
    }

    bool isCurShortAndNearLong(const SubStr& cur, const SubStr& near) const {
        bool curNoKana = !containsJaKana(cur.text);
        bool nearNoKana = !containsJaKana(near.text);
        bool bothNoKana = curNoKana && nearNoKana;
        bool nearLongZh =
            near.length > cur.length && near.length >= 6 && near.lang == "zh";
        bool curShortNearLongZh = nearLongZh && bothNoKana;
        auto pl = possibleDetectionList(*lid_, cur.text);
        bool nearPossible =
            std::find(pl.begin(), pl.end(), near.lang) != pl.end();
        return curShortNearLongZh || (nearPossible && bothNoKana);
    }

    static std::vector<SubStr>& mergeMiddleToTwoSide(
        std::vector<SubStr>& subs, const Splitter* self) {
        if (subs.size() <= 2) return subs;
        for (size_t i = 0; i + 2 < subs.size(); ++i) {
            const SubStr& l = subs[i];
            SubStr& mid = subs[i + 1];
            const SubStr& r = subs[i + 2];
            if (l.lang == r.lang && l.lang != "x" && l.lang != "newline")
                if (self->isMergeMiddleToTwoSide(l, mid, r)) mid.lang = l.lang;
        }
        return subs;
    }

    static std::string nearestLang(const std::vector<SubStr>& subs,
                                   size_t index, bool searchLeft) {
        if (searchLeft) {
            for (size_t i = 1; i < subs.size(); ++i) {
                if (index >= i) {
                    const SubStr& s = subs[index - i];
                    if (s.lang != "x" && s.lang != "digit") return s.lang;
                }
            }
        } else {
            for (size_t i = 1; i < subs.size(); ++i) {
                size_t ri = index + i;
                if (ri < subs.size()) {
                    const SubStr& s = subs[ri];
                    if (s.lang != "x" && s.lang != "digit") return s.lang;
                }
            }
        }
        return subs[index].lang;
    }

    static bool mergeDirectionLeft(const std::vector<SubStr>& subs,
                                   size_t index) {
        if (index == 0) return false;
        if (index == subs.size() - 1) return true;
        return subs[index - 1].text.size() >= subs[index + 1].text.size();
    }

    std::vector<SubStr> fillUnknown(std::vector<SubStr> subs) const {
        for (size_t i = 0; i < subs.size(); ++i) {
            if (subs[i].lang == "x") {
                bool searchLeft = i == 0 ? false
                                         : (i == subs.size() - 1
                                                ? true
                                                : mergeDirectionLeft(subs, i));
                subs[i].lang = nearestLang(subs, i, searchLeft);
            }
        }
        return subs;
    }

    std::vector<SubStr> mergeSideToNear(std::vector<SubStr> subs) const {
        if (subs.empty()) return subs;
        {
            bool needR = false;
            if (subs.size() >= 2) {
                auto pl = possibleDetectionList(*lid_, subs[0].text);
                needR = subs[0].lang == "x" ||
                        isCurShortAndNearLong(subs[0], subs[1]) ||
                        (std::find(pl.begin(), pl.end(), subs[1].lang) !=
                             pl.end() &&
                         subs[1].length <= 5);
            } else {
                needR = subs[0].lang == "x";
            }
            if (needR) subs[0].lang = nearestLang(subs, 0, false);
        }
        {
            size_t n = subs.size();
            bool needL = false;
            if (n >= 2) {
                auto pl = possibleDetectionList(*lid_, subs[n - 1].text);
                needL = subs[n - 1].lang == "x" ||
                        isCurShortAndNearLong(subs[n - 1], subs[n - 2]) ||
                        (std::find(pl.begin(), pl.end(), subs[n - 2].lang) !=
                             pl.end() &&
                         subs[n - 2].length <= 5);
            } else {
                needL = subs[n - 1].lang == "x";
            }
            if (needL) subs[n - 1].lang = nearestLang(subs, n - 1, true);
        }
        return subs;
    }

    std::vector<SubStr> smartMerge(std::vector<SubStr> subsIn,
                                   SecType st) const {
        // _smart_merge 精确步骤(注意 _get_languages 为无副作用遍历, 已省略)
        std::vector<SubStr> subs = mergeSubstrings(std::move(subsIn));
        subs = mergeMiddleToTwoSideMut(std::move(subs));
        subs = mergeSubstrings(std::move(subs));
        subs = fillUnknown(std::move(subs));
        subs = mergeSideToNear(std::move(subs));
        subs = mergeSubstrings(std::move(subs));
        // (_get_languages 第二次 — 无回写, no-op)
        subs = mergeMiddleToTwoSideMut(std::move(subs));
        subs = mergeSubstrings(std::move(subs));
        (void)st;
        return subs;
    }

    std::vector<SubStr> mergeMiddleToTwoSideMut(std::vector<SubStr> subs) const {
        return std::move(mergeMiddleToTwoSide(subs, this));
    }

    std::vector<SubStr> specialMergeZhJa(std::vector<SubStr> subs) const {
        if (subs.size() == 1) return subs;
        std::unordered_map<std::string, long long> lenBy{
            {"zh", 0}, {"ja", 0}, {"x", 0}, {"digit", 0},
            {"punctuation", 0}, {"newline", 0}};
        std::vector<SubStr> ns;
        size_t index = 0;
        while (index < subs.size()) {
            SubStr cur = subs[index];
            lenBy[cur.lang] += static_cast<long long>(cur.length);
            if (index == 0) {
                const SubStr& right = subs[index + 1];
                bool rZJ = right.lang == "zh" || right.lang == "ja";
                bool cZJX = cur.lang == "zh" || cur.lang == "ja" ||
                            cur.lang == "x";
                if (rZJ && cZJX && cur.length * 10 < right.length) {
                    ns.push_back(SubStr{right.lang, cur.text + right.text,
                                        cur.index,
                                        cur.length + right.length});
                    ++index;
                } else {
                    ns.push_back(cur);
                }
            } else if (index == subs.size() - 1) {
                SubStr& left = ns.back();
                bool lZJ = left.lang == "zh" || left.lang == "ja";
                bool cZJX = cur.lang == "zh" || cur.lang == "ja" ||
                            cur.lang == "x";
                if (lZJ && cZJX && cur.length * 10 < left.length) {
                    left.text += cur.text;
                    left.length += cur.length;
                } else {
                    ns.push_back(cur);
                }
            } else {
                const SubStr& right = subs[index + 1];
                const SubStr& nl = ns.back();
                if ((nl.lang == "zh" || nl.lang == "ja") &&
                    nl.lang == right.lang && cur.lang != "en" &&
                    cur.length * 10 < nl.length + right.length) {
                    ns.back().text += cur.text + right.text;
                    ns.back().length += cur.length + right.length;
                    ++index;
                } else {
                    ns.push_back(cur);
                }
            }
            ++index;
        }
        if (lenBy["x"] > 0) {
            std::string maxLang;
            long long best = -1;
            for (auto& kv : lenBy)
                if (kv.second > best) {
                    best = kv.second;
                    maxLang = kv.first;
                }
            for (auto& s : ns)
                if (s.lang == "x") s.lang = maxLang;
        }
        if (lenBy["ja"] >= lenBy["zh"] * 10) {
            for (auto& s : ns)
                if (s.lang == "zh") s.lang = "ja";
        }
        return mergeSubstrings(std::move(ns));
    }

    // ---- across punctuation/newline(基于 section 层) ----
    static void reindexSubs(Section& s, bool skipFirstExisting) {
        for (size_t k = 1; k < s.subs.size(); ++k)
            s.subs[k].index = s.subs[k - 1].index + s.subs[k - 1].length;
        (void)skipFirstExisting;
    }

    std::vector<Section> mergeAcrossPunctBase(std::vector<Section> sections) const {
        // _merge_substrings_across_punctuation(not_merge_punctuation="")
        std::vector<Section> news;
        if (sections.empty()) return news;
        news.push_back(sections[0]);
        for (size_t index = 1; index < sections.size(); ++index) {
            Section& prev = news.back();
            Section& cur = sections[index];
            bool oneIs = prev.type == SecType::Punct ||
                         cur.type == SecType::Punct;
            if (!oneIs) {
                news.push_back(cur);
                continue;
            }
            if (prev.type == SecType::Punct &&
                prev.subs.front().text != U"/n") {  // not_merge_punct="" 恒真
                prev.text += cur.text;
                prev.type = cur.type;
                prev.subs.back().text += cur.subs.front().text;
                prev.subs.back().length += cur.subs.front().length;
                prev.subs.back().lang = cur.subs.front().lang;
                prev.subs.insert(prev.subs.end(), cur.subs.begin() + 1,
                                 cur.subs.end());
                reindexSubs(prev, false);
            } else if (cur.type == SecType::Punct &&
                       cur.subs.front().text != U"/n") {
                prev.text += cur.text;
                prev.subs.back().text += cur.subs.front().text;
                prev.subs.back().length += cur.subs.front().length;
                prev.subs.insert(prev.subs.end(), cur.subs.begin() + 1,
                                 cur.subs.end());
                reindexSubs(prev, false);
            } else {
                news.push_back(cur);
            }
        }
        return news;
    }

    std::vector<Section> mergeAcrossPunct(std::vector<Section> sections) const {
        std::vector<Section> news = mergeAcrossPunctBase(std::move(sections));
        std::vector<Section> merged;
        for (auto& s : news) {
            if (!merged.empty() && merged.back().type == s.type) {
                merged.back().text += s.text;
                merged.back().subs.insert(merged.back().subs.end(),
                                          s.subs.begin(), s.subs.end());
            } else {
                merged.push_back(s);
            }
        }
        for (size_t si = 0; si < merged.size(); ++si) {
            Section& sec = merged[si];
            if (si == 0) {
                for (size_t k = 1; k < sec.subs.size(); ++k)
                    sec.subs[k].index =
                        sec.subs[k - 1].index + sec.subs[k - 1].length;
            } else {
                for (size_t k = 0; k < sec.subs.size(); ++k) {
                    if (k == 0) {
                        const Section& p = merged[si - 1];
                        sec.subs[k].index =
                            p.subs.back().index + p.subs.back().length;
                    } else {
                        sec.subs[k].index =
                            sec.subs[k - 1].index + sec.subs[k - 1].length;
                    }
                }
            }
        }
        for (auto& sec : merged) sec.subs = mergeFlatPunct(std::move(sec.subs));
        return merged;
    }

    std::vector<Section> mergeAcrossNewline(std::vector<Section> sections) const {
        if (sections.empty()) return sections;
        std::vector<Section> news;
        news.push_back(sections[0]);
        for (size_t index = 1; index < sections.size(); ++index) {
            Section& prev = news.back();
            Section& cur = sections[index];
            if (prev.type == SecType::Newline) {
                prev.type = cur.type;
                prev.text += cur.text;
                prev.subs.insert(prev.subs.end(), cur.subs.begin(),
                                 cur.subs.end());
                for (size_t k = 1; k < prev.subs.size(); ++k)
                    prev.subs[k].index =
                        prev.subs[k - 1].index + prev.subs[k - 1].length;
            } else if (cur.type == SecType::Newline) {
                prev.text += cur.text;
                prev.subs.insert(prev.subs.end(), cur.subs.begin(),
                                 cur.subs.end());
                if (prev.subs.size() >= 2)
                    prev.subs.back().index =
                        prev.subs[prev.subs.size() - 2].index +
                        prev.subs[prev.subs.size() - 2].length;
            } else {
                news.push_back(cur);
            }
        }
        std::vector<Section> merged;
        for (auto& s : news) {
            if (!merged.empty() && merged.back().type == s.type) {
                merged.back().text += s.text;
                merged.back().subs.insert(merged.back().subs.end(),
                                          s.subs.begin(), s.subs.end());
            } else {
                merged.push_back(s);
            }
        }
        for (size_t si = 0; si < merged.size(); ++si) {
            Section& sec = merged[si];
            if (si == 0) {
                for (size_t k = 1; k < sec.subs.size(); ++k)
                    sec.subs[k].index =
                        sec.subs[k - 1].index + sec.subs[k - 1].length;
            } else {
                for (size_t k = 0; k < sec.subs.size(); ++k) {
                    if (k == 0) {
                        const Section& p = merged[si - 1];
                        sec.subs[k].index =
                            p.subs.back().index + p.subs.back().length;
                    } else {
                        sec.subs[k].index =
                            sec.subs[k - 1].index + sec.subs[k - 1].length;
                    }
                }
            }
        }
        for (auto& sec : merged) sec.subs = mergeFlatNewline(std::move(sec.subs));
        return merged;
    }

    // _merge_substrings_across_punctuation(substrings 级, not_merge_punct="")
    //   substring.lang=="punctuation" → 并入前段;
    //   否则同语种/首段 → 文本并入且 new.lang 保持自己(除非自己是 punctuation)
    static std::vector<SubStr> mergeFlatPunct(std::vector<SubStr> subs) {
        std::vector<SubStr> out;
        std::string lastLang = "";
        for (auto& s : subs) {
            if (!out.empty()) {
                if (s.lang == "punctuation") {
                    out.back().text += s.text;
                    out.back().length += s.length;
                } else if (s.lang == lastLang || lastLang.empty()) {
                    out.back().text += s.text;
                    out.back().length += s.length;
                    if (out.back().lang == "punctuation")
                        out.back().lang = s.lang;
                } else {
                    out.push_back(s);
                }
            } else {
                out.push_back(s);
            }
            lastLang = s.lang;
        }
        return out;
    }

    // _merge_substrings_across_newline(substrings 级)
    static std::vector<SubStr> mergeFlatNewline(std::vector<SubStr> subs) {
        std::vector<SubStr> out;
        std::string lastLang = "";
        for (auto& s : subs) {
            if (!out.empty()) {
                if (s.lang == "newline") {
                    out.back().text += s.text;
                    out.back().length += s.length;
                } else if (s.lang == lastLang || lastLang.empty()) {
                    out.back().text += s.text;
                    out.back().length += s.length;
                    if (out.back().lang == "newline") out.back().lang = s.lang;
                } else {
                    out.push_back(s);
                }
            } else {
                out.push_back(s);
            }
            lastLang = s.lang;
        }
        return out;
    }

    static std::u32string strip(const std::u32string& s) {
        size_t b = 0, e = s.size();
        while (b < e && isSpacePy(s[b])) ++b;
        while (e > b && isSpacePy(s[e - 1])) --e;
        return s.substr(b, e - b);
    }
};

}  // namespace

// ===========================================================================
// langsegmenter.py 顶层规则
// ===========================================================================
namespace {

std::u32string fullCjk(const std::u32string& text) {
    static const uint32_t ranges[][2] = {
        {0x4E00, 0x9FFF},   {0x3400, 0x4DB5},   {0x20000, 0x2A6DD},
        {0x2A700, 0x2B73F}, {0x2B740, 0x2B81F}, {0x2B820, 0x2CEAF},
        {0x2CEB0, 0x2EBEF}, {0x30000, 0x3134A}, {0x31350, 0x323AF},
        {0x2EBF0, 0x2EE5D}};
    std::u32string out;
    for (char32_t ch : text) {
        bool inCjk = false;
        for (auto& r : ranges)
            if (uint32_t(ch) >= r[0] && uint32_t(ch) <= r[1]) inCjk = true;
        bool hitRe = (ch >= U'0' && ch <= U'9') ||
                     (ch >= 0x3001 && ch <= 0x301C) ||  // 、-〜(正则 range)
                     ch == U'。' || ch == U'！' || ch == U'？' || ch == U'.' ||
                     ch == U'!' || ch == U'?' || ch == 0x2026 || ch == U' ' ||
                     ch == U'/';
        if (inCjk || hitRe) out.push_back(ch);
    }
    return out;
}

bool fullEn(const std::u32string& text) {
    bool hasAlpha = false;
    for (char32_t c : text) {
        bool ok = (c >= U'A' && c <= U'Z') || (c >= U'a' && c <= U'z') ||
                  (c >= U'0' && c <= U'9') || isSpacePy(c) ||
                  (c >= 0x20 && c <= 0x7E) || (c >= 0x2000 && c <= 0x206F) ||
                  (c >= 0x3000 && c <= 0x303F) ||
                  (c >= 0xFF00 && c <= 0xFFEF);
        if (!ok) return false;
        if ((c >= U'A' && c <= U'Z') || (c >= U'a' && c <= U'z')) hasAlpha = true;
    }
    return hasAlpha;
}

// split_jako(tag_lang, item): kana/hangul 正则切分
std::vector<SubStr> splitJako(const std::string& tagLang,
                              const std::u32string& itemText,
                              const std::string& itemLang) {
    std::vector<SubStr> list;
    auto isTargetChar = [&](char32_t c) -> bool {
        if (tagLang == "ja")
            return (c >= 0x3041 && c <= 0x3096) || c == 0x3099 ||
                   c == 0x309A || (c >= 0x30A1 && c <= 0x30FA) || c == 0x30FC;
        return (c >= 0x1100 && c <= 0x11FF) || (c >= 0x3130 && c <= 0x318F) ||
               (c >= 0xAC00 && c <= 0xD7AF);
    };
    auto isSepChar = [&](char32_t c) -> bool {
        return (c >= U'0' && c <= U'9') || (c >= 0x3001 && c <= 0x301C) ||
               c == U'。' || c == U'！' || c == U'？' || c == U'.' ||
               c == U'!' || c == U'?' || c == 0x2026 || c == U' ';
    };
    // re.finditer(pattern): 目标串(可夹分隔符再接目标串)
    size_t tag = 0;
    size_t i = 0;
    std::u32string out = itemText;
    while (i < out.size()) {
        if (!isTargetChar(out[i])) { ++i; continue; }
        size_t start = i;
        size_t j = i;
        while (j < out.size() && isTargetChar(out[j])) ++j;
        // ([sep]+ target*)*: 尝试扩展
        bool extended = true;
        while (extended) {
            extended = false;
            size_t k = j;
            while (k < out.size() && isSepChar(out[k])) ++k;
            size_t afterSeps = k;
            if (afterSeps > j && afterSeps < out.size() &&
                isTargetChar(out[afterSeps])) {
                size_t m = afterSeps;
                while (m < out.size() && (isTargetChar(out[m]) ||
                                          (m > afterSeps && isSepChar(out[m]))))
                    ++m;
                j = m;
                extended = true;
            }
        }
        size_t matchEnd = j;
        if (start > tag)
            list.push_back({itemLang, out.substr(tag, start - tag), 0,
                            start - tag});
        list.push_back({tagLang, out.substr(start, matchEnd - start), 0,
                        matchEnd - start});
        tag = matchEnd;
        i = matchEnd;
    }
    if (tag < out.size())
        list.push_back({itemLang, out.substr(tag), 0, out.size() - tag});
    return list;
}

size_t cpCount(const std::string& utf8) {
    size_t n = 0;
    for (size_t i = 0; i < utf8.size();) {
        unsigned char b = uint8_t(utf8[i]);
        i += (b < 0x80) ? 1 : (b & 0xE0) == 0xC0 ? 2 : (b & 0xF0) == 0xE0 ? 3 : 4;
        ++n;
    }
    return n;
}

bool endsWithAny(const std::string& utf8, const char32_t* set, size_t n) {
    if (utf8.empty()) return false;
    std::u32string t = u8to32(utf8);
    for (size_t i = 0; i < n; ++i)
        if (t.back() == set[i]) return true;
    return false;
}
bool startsWithAny(const std::string& utf8, const char32_t* set, size_t n) {
    if (utf8.empty()) return false;
    std::u32string t = u8to32(utf8);
    for (size_t i = 0; i < n; ++i)
        if (t.front() == set[i]) return true;
    return false;
}
constexpr char32_t kCtxPunct[] = {U',', U'.', U'!', U'?', 0xFF0C, 0x3002,
                                  0xFF01, 0xFF1F};
constexpr size_t kCtxPunctN = 8;
constexpr char32_t kPeriod[] = {0x3002, U'.'};  // 。和 .
constexpr size_t kPeriodN = 2;

// merge_lang(list, item)
void mergeInto(std::vector<LangPieceCpp>* list, const std::string& lang,
               const std::u32string& text) {
    if (!list->empty() && list->back().lang == lang) {
        list->back().text += u32to8(text);
    } else {
        list->push_back(LangPieceCpp{lang, u32to8(text)});
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// 公共入口实现
// ---------------------------------------------------------------------------

bool LangSegmenterCpp::load(const std::string& lidBinPath,
                            const std::string& budouxDir, std::string* err) {
    if (!lid_.load(lidBinPath, err)) return false;
    static bool budouxReady = false;
    if (!budouxReady) {
        BudouxHolder& h = budouxHolder();
        std::string e1, e2;
        bool a = h.jp.loadJson(budouxDir + "/ja.json", &e1);
        bool b = h.zh.loadJson(budouxDir + "/zh-hans.json", &e2);
        if (!(a && b)) {
            if (err) *err = "budoux 加载失败: " + (a ? e2 : e1);
            return false;
        }
        budouxReady = true;
    }
    loaded_ = true;
    return true;
}

std::vector<LangPieceCpp> LangSegmenterCpp::getTexts(
    const std::string& utf8Text, const std::string& defaultLang) const {
    std::vector<LangPieceCpp> langList;
    Splitter splitter(&lid_);
    std::u32string text32 = u8to32(utf8Text);
    std::vector<SubStr> substrs = splitter.splitByLang(text32);

    bool haveNum = false;


    for (const auto& item : substrs) {
        std::string dlang = item.lang;
        const std::u32string dtext = item.text;

        if (dlang == "digit") {
            if (!defaultLang.empty()) {
                dlang = defaultLang;
            } else {
                haveNum = true;
            }
            mergeInto(&langList, dlang, dtext);
            continue;
        }

        if (fullEn(dtext)) {
            mergeInto(&langList, "en", dtext);
            continue;
        }

        if (!defaultLang.empty()) {
            mergeInto(&langList, defaultLang, dtext);
            continue;
        }

        std::vector<SubStr> jaList;
        if (dlang != "ja") jaList = splitJako("ja", dtext, dlang);
        if (jaList.empty())
            jaList.push_back(SubStr{dlang, dtext, item.index, item.length});

        std::vector<SubStr> tempList;
        for (auto& koItem : jaList) {
            std::vector<SubStr> koList;
            if (koItem.lang != "ko")
                koList = splitJako("ko", koItem.text, koItem.lang);
            if (!koList.empty())
                tempList.insert(tempList.end(), koList.begin(), koList.end());
            else
                tempList.push_back(koItem);
        }

        if (tempList.size() == 1) {
            if (dlang == "x") {
                std::u32string cjk = fullCjk(dtext);
                if (!cjk.empty()) mergeInto(&langList, "zh", cjk);
                else mergeInto(&langList, dlang, dtext);
            } else {
                mergeInto(&langList, dlang, dtext);
            }
            continue;
        }

        for (auto& ti : tempList) {
            if (ti.lang == "x") {
                std::u32string cjk = fullCjk(ti.text);
                if (!cjk.empty()) mergeInto(&langList, "zh", cjk);
                else mergeInto(&langList, ti.lang, ti.text);
            } else {
                mergeInto(&langList, ti.lang, ti.text);
            }
        }
    }

    // 有数字: 上下文归属(langsegmenter.py 精确分支)
    if (haveNum) {
        std::vector<LangPieceCpp>& temp_list = langList;
        std::vector<LangPieceCpp> newList;
        for (size_t i = 0; i < temp_list.size(); ++i) {
            LangPieceCpp item = temp_list[i];
            if (item.lang == "digit") {
                if (!defaultLang.empty()) {
                    item.lang = defaultLang;
                } else if (!newList.empty() && i == temp_list.size() - 1) {
                    item.lang = newList.back().lang;
                } else if (newList.empty() && i < temp_list.size() - 1) {
                    item.lang = temp_list[1].lang;
                } else if (!newList.empty() && i < temp_list.size() - 1) {
                    if (newList.back().lang == temp_list[i + 1].lang) {
                        item.lang = newList.back().lang;
                    } else if (endsWithAny(newList.back().text, kCtxPunct,
                                           kCtxPunctN)) {
                        // python: lang_list[-1]['text'][-1] in [,.!?,。！？]
                        item.lang = temp_list[i + 1].lang;
                    } else if (startsWithAny(temp_list[i + 1].text, kCtxPunct,
                                             kCtxPunctN)) {
                        item.lang = newList.back().lang;
                    } else if (endsWithAny(item.text, kPeriod, kPeriodN)) {
                        // python: temp_item['text'][-1] in [。.]
                        item.lang = newList.back().lang;
                    } else if (cpCount(newList.back().text) >=
                               cpCount(temp_list[i + 1].text)) {
                        item.lang = newList.back().lang;
                    } else {
                        item.lang = temp_list[i + 1].lang;
                    }
                } else {
                    item.lang = "zh";
                }
            }
            mergeInto(&newList, item.lang, u8to32(item.text));
        }
        langList.swap(newList);
    }

    // 筛 x
    {
        std::vector<LangPieceCpp>& temp_list = langList;
        std::vector<LangPieceCpp> newList;
        for (size_t i = 0; i < temp_list.size(); ++i) {
            LangPieceCpp item = temp_list[i];
            if (item.lang == "x") {
                if (!newList.empty()) item.lang = newList.back().lang;
                else if (temp_list.size() > 1) item.lang = temp_list[1].lang;
                else item.lang = "zh";
            }
            mergeInto(&newList, item.lang, u8to32(item.text));
        }
        langList.swap(newList);
    }
    return langList;
}

}  // namespace gsv::textfront
