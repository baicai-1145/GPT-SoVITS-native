// en_norm.cpp — port of text/en_normalization/expend.py::normalize plus the
// inflect-engine subset it depends on (number_to_words, ordinal, grouped
// year reading). Python semantics reproduced with hand-written leftmost-first
// scanners (same discipline as zh_rules.cpp).
//
// Bug-compat notes (do NOT "fix"):
// - _expand_measurement strips "." from the captured number before parsing;
//   plural decision uses int(num-without-dot)==1 and absence of a decimal
//   part ("1.2s" -> "1.2 seconds", "20h" -> "20 hours").
// - RE_INTEGER r"(?:^|\s+)(-)(\d+)" swallows ONE leading whitespace run:
//   "temp -5 deg" -> "tempnegative 5 deg".
// - _expand_decimal_number keeps the integer part RAW and space-joins the
//   fraction digits ("13.234" -> "13 point 2 3 4").
// - _expend_fraction with denominator 1 returns only the numerator words.
// - measurement suffix alternation is leftmost-first in python ("km" wins
//   before "km/h"); reproduced by trying candidates in source order.
// - _expand_ordinal feeds "105th" to inflect.number_to_words which returns the
//   ordinal phrase; equivalent to our ordinalOf(int).
#include "en_norm.h"

#include <cstdint>
#include <string>
#include <vector>

namespace gsv::textfront::en {

namespace {

bool isDigit(char c) { return c >= '0' && c <= '9'; }
bool isAlphaEn(char c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

// ------------------------------------------------------------------
// UTF-8 helpers
// ------------------------------------------------------------------
void appendCp(std::string& out, uint32_t cp) {
    if (cp < 0x80) {
        out += static_cast<char>(cp);
        return;
    }
    unsigned char b[4];
    int n;
    if (cp < 0x800) { n = 2; b[0] = 0xC0 | (cp >> 6); }
    else if (cp < 0x10000) { n = 3; b[0] = 0xE0 | (cp >> 12); }
    else { n = 4; b[0] = 0xF0 | (cp >> 18); }
    for (int k = 1; k < n; ++k)
        b[k] = 0x80 | ((cp >> (6 * (n - k - 1))) & 0x3F);
    out.append(reinterpret_cast<char*>(b), static_cast<size_t>(n));
}

size_t decodeAt(const std::string& s, size_t i, uint32_t& cp) {
    unsigned char b = static_cast<unsigned char>(s[i]);
    if (b < 0x80) { cp = b; return 1; }
    uint32_t v = 0; size_t extra = 0;
    if ((b & 0xE0) == 0xC0) { v = b & 0x1F; extra = 1; }
    else if ((b & 0xF0) == 0xE0) { v = b & 0x0F; extra = 2; }
    else { v = b & 0x07; extra = 3; }
    if (i + extra >= s.size()) { cp = b; return 1; }
    for (size_t k = 1; k <= extra; ++k)
        v = (v << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3F);
    cp = v;
    return extra + 1;
}

// ------------------------------------------------------------------
// inflect subset
// ------------------------------------------------------------------
const char* kOnes[] = {"zero", "one", "two", "three", "four", "five", "six",
                       "seven", "eight", "nine", "ten", "eleven", "twelve",
                       "thirteen", "fourteen", "fifteen", "sixteen",
                       "seventeen", "eighteen", "nineteen"};
const char* kTensWord[] = {"", "", "twenty", "thirty", "forty", "fifty",
                           "sixty", "seventy", "eighty", "ninety"};
const char* kScale[] = {"", " thousand", " million", " billion", " trillion",
                        " quadrillion"};

std::string twoDigits(int v) {
    if (v < 20) return kOnes[v];
    std::string o = kTensWord[v / 10];
    if (v % 10) o += std::string("-") + kOnes[v % 10];
    return o;
}

// inflect number_to_words(n, andword). andword="" -> no "and"; groups
// separated by ", ". andword="and" -> "and" between hundred and remainder in
// each >100 group.
std::string num2words(long long n, const char* andword) {
    if (n == 0) return "zero";
    bool neg = n < 0;
    if (neg) n = -n;
    int groups[6] = {0, 0, 0, 0, 0, 0};
    int gi = 0;
    while (n > 0 && gi < 6) {
        groups[gi++] = static_cast<int>(n % 1000);
        n /= 1000;
    }
    auto groupStr = [&](int v) -> std::string {
        int h = v / 100, rest = v % 100;
        std::string o;
        if (h) o += std::string(kOnes[h]) + " hundred";
        if (rest) {
            if (h) {
                if (andword[0])
                    o += std::string(" ") + andword + " ";
                else
                    o += " ";
            }
            o += twoDigits(rest);
        }
        return o;
    };
    std::string out;
    bool first = false;
    for (int g = gi - 1; g >= 0; --g) {
        if (groups[g] == 0) continue;
        std::string part = groupStr(groups[g]);
        if (part.empty()) continue;
        if (first) {
            // inflect: comma after a scale group only when this group has a
            // hundreds component ("one thousand, two hundred thirty-four"
            // but "one thousand fifty-four" / "one hundred thousand one")
            if (groups[g] >= 100) out += ", ";
            else out += " ";
        }
        out += part;
        if (g > 0) out += kScale[g];
        first = true;
    }
    if (neg) out = "minus " + out;
    return out;
}

std::string num2wordsPlain(long long n) { return num2words(n, ""); }

std::string ordinalOf(long long n) {
    std::string s = num2words(n, "and");
    size_t pos = 0;
    for (size_t k = 0; k < s.size(); ++k)
        if (s[k] == ' ' || s[k] == '-') pos = k + 1;
    std::string head = s.substr(0, pos);
    std::string tail = s.substr(pos);
    static const std::pair<const char*, const char*> kSpecial[] = {
        {"one", "first"}, {"two", "second"}, {"three", "third"},
        {"four", "fourth"}, {"five", "fifth"}, {"six", "sixth"},
        {"seven", "seventh"}, {"eight", "eighth"}, {"nine", "ninth"},
        {"ten", "tenth"}, {"eleven", "eleventh"}, {"twelve", "twelfth"},
        {"thirteen", "thirteenth"}, {"fourteen", "fourteenth"},
        {"fifteen", "fifteenth"}, {"sixteen", "sixteenth"},
        {"seventeen", "seventeenth"}, {"eighteen", "eighteenth"},
        {"nineteen", "nineteenth"}, {"twenty", "twentieth"},
        {"thirty", "thirtieth"}, {"forty", "fortieth"},
        {"fifty", "fiftieth"}, {"sixty", "sixtieth"},
        {"seventy", "seventieth"}, {"eighty", "eightieth"},
        {"ninety", "ninetieth"},
    };
    for (auto& kv : kSpecial)
        if (tail == kv.first) return head + kv.second;
    if (!tail.empty() && tail.back() == 'y') {
        tail.resize(tail.size() - 1);
        tail += "ieth";
        return head + tail;
    }
    return head + tail + "th";
}

std::string yearWords(int n) {
    auto chunk = [](int x) -> std::string {
        if (x == 0) return "oh oh";
        if (x < 10) return std::string("oh ") + kOnes[x];
        return twoDigits(x);
    };
    return chunk(n / 100) + " " + chunk(n % 100);
}

std::string expandPlainNumber(long long n) {
    if (n > 1000 && n < 3000) {
        if (n == 2000) return "two thousand";
        if (n > 2000 && n < 2010)
            return "two thousand " + num2words(n % 100, "");
        if (n % 100 == 0) return num2words(n / 100, "") + " hundred";
        return yearWords(static_cast<int>(n));
    }
    return num2words(n, "");
}

}  // namespace

// ------------------------------------------------------------------
// staged passes (expend.normalize order)
// ------------------------------------------------------------------
namespace {

template <typename Match>
std::string subAll(const std::string& text, Match&& match) {
    std::string out;
    size_t i = 0;
    while (i < text.size()) {
        size_t end = i;
        std::string rep;
        if (match(text, i, end, rep)) {
            out += rep;
            i = end;
        } else {
            out += text[i++];
        }
    }
    return out;
}

bool atAsciiWordChar(const std::string& t, size_t i) {
    if (i >= t.size()) return false;
    char c = t[i];
    return isAlphaEn(c) || isDigit(c) ||
           static_cast<unsigned char>(c) >= 0x80;
}

// python \b (unicode): boundary before i means start-of-string or the
// preceding char is NOT a word char ([A-Za-z0-9_] or unicode alnum).
bool wordBoundaryBefore(const std::string& t, size_t i) {
    if (i == 0) return true;
    char c = t[i - 1];
    bool wc = isAlphaEn(c) || isDigit(c) || static_cast<unsigned char>(c) >= 0x80;
    return !wc;
}
bool wordBoundaryAfter(const std::string& t, size_t i) {  // i = end index
    return !atAsciiWordChar(t, i);
}

long long parseNumAt(const std::string& t, size_t i, size_t* end) {
    long long v = 0;
    while (i < t.size() && isDigit(t[i])) {
        v = v * 10 + (t[i] - '0');
        ++i;
    }
    *end = i;
    return v;
}

// P1: r"\b([0-9]+)\. " -> ordinalOf(n) + ", "
std::string passOrdinalDot(const std::string& t) {
    return subAll(t, [&](const std::string& s, size_t i, size_t& end,
                         std::string& rep) -> bool {
        if (!wordBoundaryBefore(s, i)) return false;
        if (!isDigit(s[i])) return false;
        size_t numEnd;
        long long v = parseNumAt(s, i, &numEnd);
        if (numEnd == i || numEnd + 1 >= s.size() || s[numEnd] != '.' ||
            s[numEnd + 1] != ' ')
            return false;
        rep = ordinalOf(v) + ", ";
        end = numEnd + 2;
        return true;
    });
}

bool isSuperscript(uint32_t cp) {
    switch (cp) {
        case 0x2070: case 0xB9: case 0xB2: case 0xB3: case 0x2074:
        case 0x2075: case 0x2076: case 0x2077: case 0x2078: case 0x2079:
        case 0x02B2: case 0x02B8: case 0x207F:
            return true;
        default:
            return false;
    }
}

int opLenAt(const std::string& t, size_t i, std::string* opText) {
    char c = t[i];
    if (c == '+' || c == '-' || c == '=') {
        if (opText) *opText = std::string(1, c);
        return 1;
    }
    if (static_cast<unsigned char>(c) == 0xC3 && i + 1 < t.size()) {
        unsigned char c2 = static_cast<unsigned char>(t[i + 1]);
        if (c2 == 0x97) { if (opText) *opText = "\xc3\x97"; return 2; }
        if (c2 == 0xB7) { if (opText) *opText = "\xc3\xb7"; return 2; }
    }
    return 0;
}
std::string asmdMap(const std::string& op) {
    if (op == "+") return " plus ";
    if (op == "-") return " minus ";
    if (op == "\xc3\x97") return " times ";
    if (op == "\xc3\xb7") return " divided by ";
    return " Equals ";
}

// operand: (-?)((\d+)(\.\d+)?|(\.\d+)|[A-Za-z])(superscripts*)
bool scanOperand(const std::string& t, size_t i, size_t& end) {
    size_t p = i;
    if (p < t.size() && t[p] == '-' && p + 1 < t.size() &&
        (isDigit(t[p + 1]) || t[p + 1] == '.')) {
        ++p;
    }
    if (p < t.size() && isDigit(t[p])) {
        while (p < t.size() && isDigit(t[p])) ++p;
        if (p + 1 < t.size() && t[p] == '.' && isDigit(t[p + 1])) {
            ++p;
            while (p < t.size() && isDigit(t[p])) ++p;
        }
    } else if (p < t.size() && t[p] == '.' && p + 1 < t.size() &&
               isDigit(t[p + 1])) {
        ++p;
        while (p < t.size() && isDigit(t[p])) ++p;
    } else if (p < t.size() && isAlphaEn(t[p])) {
        ++p;
    } else {
        return false;
    }
    while (p < t.size()) {
        uint32_t cp;
        size_t n = decodeAt(t, p, cp);
        if (isSuperscript(cp)) p += n;
        else break;
    }
    if (p == i) return false;
    end = p;
    return true;
}

// P2: while RE_ASMD matches, replace left op right
std::string passAsmd(const std::string& t) {
    std::string cur = t;
    for (int guard = 0; guard < 10000; ++guard) {
        std::string out;
        bool replaced = false;
        size_t i = 0;
        while (i < cur.size()) {
            size_t lEnd;
            if (scanOperand(cur, i, lEnd)) {
                size_t p = lEnd;
                while (p < cur.size() &&
                       (cur[p] == ' ' || cur[p] == '\t')) ++p;
                std::string op;
                int ol = (p < cur.size()) ? opLenAt(cur, p, &op) : 0;
                if (ol && p > lEnd) {
                    size_t afterOp = p + ol;
                    size_t ws2 = afterOp;
                    while (ws2 < cur.size() &&
                           (cur[ws2] == ' ' || cur[ws2] == '\t')) ++ws2;
                    if (ws2 > afterOp) {
                        size_t rEnd;
                        if (scanOperand(cur, ws2, rEnd)) {
                            out += cur.substr(i, lEnd - i);
                            out += asmdMap(op);
                            out += cur.substr(ws2, rEnd - ws2);
                            i = rEnd;
                            replaced = true;
                            continue;
                        }
                    }
                }
            }
            out += cur[i++];
        }
        cur = out;
        if (!replaced) break;
    }
    return cur;
}

// P3: r"(?:^|\s+)(-)(\d+)" -> "negative "+num (swallows leading ws)
std::string passNegative(const std::string& t) {
    std::string out;
    size_t i = 0;
    while (i < t.size()) {
        bool wsStart = (i == 0);
        size_t j = i;
        while (j < t.size() && (t[j] == ' ' || t[j] == '\t')) ++j;
        if (!wsStart && j == i) { out += t[i++]; continue; }
        if (j < t.size() && t[j] == '-' && j + 1 < t.size() &&
            isDigit(t[j + 1])) {
            out += "negative ";
            i = j + 1;
            while (i < t.size() && isDigit(t[i])) out += t[i++];
        } else {
            out += t[i++];
        }
    }
    return out;
}

// P4: r"([0-9][0-9,]+[0-9])" -> strip commas
std::string passCommas(const std::string& t) {
    return subAll(t, [&](const std::string& s, size_t i, size_t& end,
                         std::string& rep) -> bool {
        if (!isDigit(s[i])) return false;
        size_t p = i + 1;
        size_t commas = 0;
        while (p < s.size() && (isDigit(s[p]) || s[p] == ',')) {
            if (s[p] == ',') ++commas;
            ++p;
        }
        if (commas == 0 || p == i + 1 || !isDigit(s[p - 1])) return false;
        rep = s.substr(i, p - i);
        size_t q = 0;
        while ((q = rep.find(',')) != std::string::npos) rep.erase(q, 1);
        end = p;
        return true;
    });
}

// P5: r"\b([01]?[0-9]|2[0-3]):([0-5][0-9])\b"
std::string passTime(const std::string& t) {
    return subAll(t, [&](const std::string& s, size_t i, size_t& end,
                         std::string& rep) -> bool {
        if (!wordBoundaryBefore(s, i)) return false;
        if (!isDigit(s[i])) return false;
        int h = 0;
        size_t p = i;
        if (s[i] == '2') {
            // 2[0-3] first; on failure backtrack to the [01]?[0-9]
            // single-digit alternative ("2:13" -> hour=2)
            if (i + 1 < s.size() && isDigit(s[i + 1]) &&
                s[i + 1] <= '3') {
                h = 20 + (s[i + 1] - '0');
                p = i + 2;
            } else {
                h = 2;
                p = i + 1;
            }
        } else if (s[i] == '0' || s[i] == '1') {
            if (i + 1 < s.size() && isDigit(s[i + 1])) {
                h = (s[i] - '0') * 10 + (s[i + 1] - '0');
                p = i + 2;
            } else { h = s[i] - '0'; p = i + 1; }
        } else { h = s[i] - '0'; p = i + 1; }
        if (p >= s.size() || s[p] != ':') return false;
        ++p;
        if (p + 1 >= s.size() || !isDigit(s[p]) || !isDigit(s[p + 1]) ||
            (s[p] - '0') * 10 + (s[p + 1] - '0') > 59)
            return false;
        int m = (s[p] - '0') * 10 + (s[p + 1] - '0');
        p += 2;
        if (!wordBoundaryAfter(s, p)) return false;
        const char* period = h < 12 ? "a.m." : "p.m.";
        int hh = h > 12 ? h - 12 : h;
        std::string out = num2wordsPlain(hh);
        if (m == 0) out += std::string(" o'clock ") + period;
        else out += " " + num2wordsPlain(m) + " " + period;
        rep = out;
        end = p;
        return true;
    });
}

// P6: measurement. suffixes tried leftmost-first.
const char* kMeasOrder[] = {"m", "km", "km/h", "ft", "L", "tbsp", "tsp", "h",
                            "min", "s", "\xc2\xb0" "C", "\xc2\xb0" "F"};
// index-aligned with kMeasOrder (includes the km/h entry)
const char* kMeasWord[][2] = {
    {"meter", "meters"}, {"kilometer", "kilometers"},
    {"kilometer per hour", "kilometers per hour"},
    {"feet", "feet"},
    {"liter", "liters"}, {"tablespoon", "tablespoons"}, {"teaspoon", "teaspoons"},
    {"hour", "hours"}, {"minute", "minutes"}, {"second", "seconds"},
    {"degree celsius", "degrees celsius"},
    {"degree fahrenheit", "degrees fahrenheit"},
};
std::string passMeasure(const std::string& t) {
    return subAll(t, [&](const std::string& s, size_t i, size_t& end,
                         std::string& rep) -> bool {
        if (!wordBoundaryBefore(s, i)) return false;
        if (!isDigit(s[i])) return false;
        size_t p = i;
        while (p < s.size() && isDigit(s[p])) ++p;
        bool hadDot = false;
        if (p + 1 < s.size() && s[p] == '.' && isDigit(s[p + 1])) {
            hadDot = true;
            ++p;
            while (p < s.size() && isDigit(s[p])) ++p;
        }
        // python alternation is leftmost-first BUT the trailing \b forces
        // backtracking: a candidate whose boundary check fails falls through
        // to the next alternative ("82min" -> m fails \b -> ... -> min).
        int pickSuffix = -1;
        size_t suffixLen = 0;
        for (int k = 0; k < 12; ++k) {
            size_t cl = std::char_traits<char>::length(kMeasOrder[k]);
            if (s.compare(p, cl, kMeasOrder[k]) == 0 &&
                wordBoundaryAfter(s, p + cl)) {
                pickSuffix = k;
                suffixLen = cl;
                break;
            }
        }
        if (pickSuffix < 0) return false;
        size_t after = p + suffixLen;
        // build number string (without suffix) + " " + plural word
        std::string num = s.substr(i, p - i);
        long long iv = 0;
        for (char c : num) if (isDigit(c)) iv = iv * 10 + (c - '0');
        int idx = (hadDot || iv != 1) ? 1 : 0;
        rep = num + " " + kMeasWord[pickSuffix][idx];
        end = after;
        return true;
    });
}

// currency helper: parts -> words
std::string expandCurrency(const std::string& num, const char* unit,
                            const char* unitPl, const char* sub,
                            const char* subPl) {
    size_t dot = num.find('.');
    long long dollars = 0, cents = 0;
    if (dot == std::string::npos) {
        for (char c : num) if (isDigit(c)) dollars = dollars * 10 + (c - '0');
    } else {
        std::string d = num.substr(0, dot), c = num.substr(dot + 1);
        for (char ch : d) if (isDigit(ch)) dollars = dollars * 10 + (ch - '0');
        std::string centsStr = c;
        while (centsStr.size() < 2) centsStr += '0';
        for (char ch : centsStr) if (isDigit(ch)) cents = cents * 10 + (ch - '0');
    }
    if (dollars == 0 && cents == 0) return std::string("zero ") + unitPl;
    auto word = [](long long x, const char* sg, const char* pl) {
        return x == 1 ? std::string(sg) : std::string(pl);
    };
    if (dollars && cents)
        return num2wordsPlain(dollars) + " " + word(dollars, unit, unitPl) +
               " and " + num2wordsPlain(cents) + " " + word(cents, sub, subPl);
    if (dollars)
        return num2wordsPlain(dollars) + " " + word(dollars, unit, unitPl);
    return num2wordsPlain(cents) + " " + word(cents, sub, subPl);
}

bool scanCurrencyNum(const std::string& t, size_t& p) {
    size_t start = p;
    while (p < t.size() && (isDigit(t[p]) || t[p] == '.' || t[p] == ',')) ++p;
    if (p == start || !isDigit(t[p - 1])) return false;
    return true;
}

// P7/P8 pound, P9/P10 dollar. Symbol-anchored.
std::string passCurrency(const std::string& t, bool dollar) {
    std::string out;
    size_t i = 0;
    auto symLen = [&](size_t p) -> int {
        if (dollar) return t[p] == '$' ? 1 : 0;
        return (static_cast<unsigned char>(t[p]) == 0xC2 && p + 1 < t.size() &&
                static_cast<unsigned char>(t[p + 1]) == 0xA3)
                   ? 2
                   : 0;
    };
    while (i < t.size()) {
        int sl = symLen(i);
        if (sl) {
            size_t p = i + sl;
            if (scanCurrencyNum(t, p)) {
                std::string num = t.substr(i + sl, p - (i + sl));
                out += dollar ? expandCurrency(num, "dollar", "dollars", "cent",
                                                "cents")
                              : expandCurrency(num, "pound", "pounds", "penny",
                                                "pence");
                i = p;
                continue;
            }
        }
        // trailing symbol: scan digits first
        size_t save = i;
        size_t p = i;
        if (scanCurrencyNum(t, p)) {
            int ssl = symLen(p);
            if (ssl) {
                std::string num = t.substr(save, p - save);
                out += dollar ? expandCurrency(num, "dollar", "dollars", "cent",
                                                "cents")
                              : expandCurrency(num, "pound", "pounds", "penny",
                                                "pence");
                i = p + ssl;
                continue;
            }
        }
        out += t[i++];
    }
    return out;
}

// P11: r"([0-9]+\.\s*[0-9]+)" -> intStr + " point " + space digits
std::string passDecimal(const std::string& t) {
    return subAll(t, [&](const std::string& s, size_t i, size_t& end,
                         std::string& rep) -> bool {
        if (!isDigit(s[i])) return false;
        size_t p = i;
        while (p < s.size() && isDigit(s[p])) ++p;
        if (p >= s.size() || s[p] != '.') return false;
        ++p;
        while (p < s.size() && s[p] == ' ') ++p;
        if (p >= s.size() || !isDigit(s[p])) return false;
        std::string intStr = s.substr(i, p - i - 1);  // from i to just before '.'
        std::string frac;
        while (p < s.size() && isDigit(s[p])) { frac += s[p]; ++p; }
        std::string joined;
        for (size_t k = 0; k < frac.size(); ++k) {
            if (k) joined += " ";
            joined += frac[k];
        }
        rep = intStr + " point " + joined;
        end = p;
        return true;
    });
}

// P12: r"([0-9]+/[0-9]+)"
std::string passFraction(const std::string& t) {
    return subAll(t, [&](const std::string& s, size_t i, size_t& end,
                         std::string& rep) -> bool {
        if (!isDigit(s[i])) return false;
        size_t p = i;
        while (p < s.size() && isDigit(s[p])) ++p;
        if (p >= s.size() || s[p] != '/') return false;
        ++p;
        if (p >= s.size() || !isDigit(s[p])) return false;
        size_t q = p;
        while (p < s.size() && isDigit(s[p])) ++p;
        long long num = 0, den = 0;
        for (size_t k = i; k < (q - 1); ++k) num = num * 10 + (s[k] - '0');
        for (size_t k = q; k < p; ++k) den = den * 10 + (s[k] - '0');
        std::string numWords = num2wordsPlain(num);
        std::string denom;
        if (den == 2) denom = num > 1 ? "halves" : "half";
        else if (den == 1) denom = "";
        else {
            denom = ordinalOf(den);
            if (num > 1) denom += "s";
        }
        rep = numWords + (denom.empty() ? "" : " " + denom);
        end = p;
        return true;
    });
}

// P13: r"[0-9]+(st|nd|rd|th)" no boundary -> ordinalOf(int)
std::string passOrdinalSuffix(const std::string& t) {
    return subAll(t, [&](const std::string& s, size_t i, size_t& end,
                         std::string& rep) -> bool {
        if (!isDigit(s[i])) return false;
        size_t p = i;
        while (p < s.size() && isDigit(s[p])) ++p;
        if (p + 2 <= s.size() &&
            ((s[p] == 's' && s[p + 1] == 't') ||
             (s[p] == 'n' && s[p + 1] == 'd') ||
             (s[p] == 'r' && s[p + 1] == 'd') ||
             (s[p] == 't' && s[p + 1] == 'h'))) {
            long long v = 0;
            for (size_t k = i; k < p; ++k) v = v * 10 + (s[k] - '0');
            rep = ordinalOf(v);
            end = p + 2;
            return true;
        }
        return false;
    });
}

// P14: r"[0-9]+" -> expandPlainNumber
std::string passNumber(const std::string& t) {
    return subAll(t, [&](const std::string& s, size_t i, size_t& end,
                         std::string& rep) -> bool {
        if (!isDigit(s[i])) return false;
        size_t p = i;
        long long v = 0;
        while (p < s.size() && isDigit(s[p])) { v = v * 10 + (s[p] - '0'); ++p; }
        rep = expandPlainNumber(v);
        end = p;
        return true;
    });
}

// P15: NFD strip accents (combining marks U+0300..U+036F dropped). Compact
// decomposition table; unknown precomposed chars pass through (the charset
// filter removes them later).
struct Decomp { uint32_t base; uint32_t comb; };
uint32_t decompose(uint32_t cp, uint32_t* comb) {
    switch (cp) {
        case 0xC0: case 0xE0: *comb = 0x0300; return cp == 0xC0 ? 'A' : 'a';
        case 0xC1: case 0xE1: *comb = 0x0301; return cp == 0xC1 ? 'A' : 'a';
        case 0xC2: case 0xE2: *comb = 0x0302; return cp == 0xC2 ? 'A' : 'a';
        case 0xC3: case 0xE3: *comb = 0x0303; return cp == 0xC3 ? 'A' : 'a';
        case 0xC4: case 0xE4: *comb = 0x0308; return cp == 0xC4 ? 'A' : 'a';
        case 0xC5: case 0xE5: *comb = 0x030A; return cp == 0xC5 ? 'A' : 'a';
        case 0xC7: case 0xE7: *comb = 0x0327; return cp == 0xC7 ? 'C' : 'c';
        case 0xC8: case 0xE8: *comb = 0x0300; return cp == 0xC8 ? 'E' : 'e';
        case 0xC9: case 0xE9: *comb = 0x0301; return cp == 0xC9 ? 'E' : 'e';
        case 0xCA: case 0xEA: *comb = 0x0302; return cp == 0xCA ? 'E' : 'e';
        case 0xCB: case 0xEB: *comb = 0x0308; return cp == 0xCB ? 'E' : 'e';
        case 0xCC: case 0xEC: *comb = 0x0300; return cp == 0xCC ? 'I' : 'i';
        case 0xCD: case 0xED: *comb = 0x0301; return cp == 0xCD ? 'I' : 'i';
        case 0xCE: case 0xEE: *comb = 0x0302; return cp == 0xCE ? 'I' : 'i';
        case 0xCF: case 0xEF: *comb = 0x0308; return cp == 0xCF ? 'I' : 'i';
        case 0xD1: case 0xF1: *comb = 0x0303; return cp == 0xD1 ? 'N' : 'n';
        case 0xD2: case 0xF2: *comb = 0x0300; return cp == 0xD2 ? 'O' : 'o';
        case 0xD3: case 0xF3: *comb = 0x0301; return cp == 0xD3 ? 'O' : 'o';
        case 0xD4: case 0xF4: *comb = 0x0302; return cp == 0xD4 ? 'O' : 'o';
        case 0xD5: case 0xF5: *comb = 0x0303; return cp == 0xD5 ? 'O' : 'o';
        case 0xD6: case 0xF6: *comb = 0x0308; return cp == 0xD6 ? 'O' : 'o';
        case 0xD9: case 0xF9: *comb = 0x0300; return cp == 0xD9 ? 'U' : 'u';
        case 0xDA: case 0xFA: *comb = 0x0301; return cp == 0xDA ? 'U' : 'u';
        case 0xDB: case 0xFB: *comb = 0x0302; return cp == 0xDB ? 'U' : 'u';
        case 0xDC: case 0xFC: *comb = 0x0308; return cp == 0xDC ? 'U' : 'u';
        case 0xDD: case 0xFD: *comb = 0x0301; return cp == 0xDD ? 'Y' : 'y';
        case 0xDF: return 0;  // ß: no decomposition; pass through
        case 0xD0: case 0xF0: return 0;
        case 0xDE: case 0xFE: return 0;
        case 0x152: case 0x153: return 0;
        default: return 0;  // unknown -> caller keeps original
    }
}
std::string passStripAccents(const std::string& t) {
    std::string out;
    size_t i = 0;
    while (i < t.size()) {
        uint32_t cp;
        size_t n = decodeAt(t, i, cp);
        uint32_t comb = 0;
        uint32_t base = decompose(cp, &comb);
        if (base) {
            appendCp(out, base);
            // comb is a combining mark; drop it (Mn) rather than appending
        } else {
            out.append(t, i, n);
        }
        i += n;
    }
    return out;
}

// P16: % -> " percent"
std::string passPercent(const std::string& t) {
    std::string out;
    for (char c : t) {
        if (c == '%') out += " percent";
        else out += c;
    }
    return out;
}

// P17: charset filter keep space, A-Za-z, ' . , ? ! -
std::string passCharset(const std::string& t) {
    std::string out;
    size_t i = 0;
    while (i < t.size()) {
        uint32_t cp;
        size_t n = decodeAt(t, i, cp);
        bool keep = (cp == ' ' || (cp >= 'A' && cp <= 'Z') ||
                     (cp >= 'a' && cp <= 'z') || cp == '\'' || cp == '.' ||
                     cp == ',' || cp == '?' || cp == '!' || cp == '-');
        if (keep) out.append(t, i, n);
        i += n;
    }
    return out;
}

// P18: i.e./e.g. (case-insensitive, literal dots) -> that is / for example
std::string passIeEg(const std::string& t) {
    std::string out;
    size_t i = 0;
    auto low = [](char c) { return static_cast<char>(tolower(c)); };
    while (i < t.size()) {
        bool hit = false;
        if (i + 3 < t.size() && low(t[i]) == 'i' && t[i + 1] == '.' &&
            low(t[i + 2]) == 'e' && t[i + 3] == '.') {
            out += "that is";
            i += 4;
            hit = true;
        } else if (i + 3 < t.size() && low(t[i]) == 'e' && t[i + 1] == '.' &&
                   low(t[i + 2]) == 'g' && t[i + 3] == '.') {
            out += "for example";
            i += 4;
            hit = true;
        }
        if (!hit) out += t[i++];
    }
    return out;
}

// P19: r"(?<!^)(?<![\s])([A-Z])" -> insert space before uppercase not at
// start and not preceded by whitespace
std::string passUppercaseSplit(const std::string& t) {
    std::string out;
    for (size_t i = 0; i < t.size(); ++i) {
        char c = t[i];
        if (c >= 'A' && c <= 'Z' && i > 0) {
            char prev = t[i - 1];
            if (prev != ' ' && prev != '\t' && prev != '\n' &&
                prev != '\r' && prev != '\v' && prev != '\f')
                out += ' ';
        }
        out += c;
    }
    return out;
}

}  // namespace

std::string normalize(const std::string& utf8) {
    std::string t = utf8;
    t = passOrdinalDot(t);
    t = passAsmd(t);
    t = passNegative(t);
    t = passCommas(t);
    t = passTime(t);
    t = passMeasure(t);
    t = passCurrency(t, /*dollar=*/false);  // pounds first
    t = passCurrency(t, /*dollar=*/true);
    t = passDecimal(t);
    t = passFraction(t);
    t = passOrdinalSuffix(t);
    t = passNumber(t);
    t = passStripAccents(t);
    t = passPercent(t);
    t = passCharset(t);
    t = passIeEg(t);
    t = passUppercaseSplit(t);
    return t;
}

}  // namespace gsv::textfront::en
