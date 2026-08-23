// zh_rules.cpp — NSW replacement scanners (chronology/num/phonecode/
// quantifier). Part 1: dates & times. Part 2 appended below.
#include "zh_num.h"

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

bool isDig(char32_t c) { return c >= U'0' && c <= U'9'; }

size_t runDigits(const U32& s, size_t i) {
    size_t j = i;
    while (j < s.size() && isDig(s[j])) ++j;
    return j;
}

template <typename Match>
U32 subAll(const U32& s, Match&& match) {
    U32 out;
    size_t i = 0;
    while (i < s.size()) {
        auto r = match(i);
        if (r.second > i) {
            out += r.first;
            i = r.second;
        } else {
            out.push_back(s[i]);
            ++i;
        }
    }
    return out;
}

U32 stripZeros(const U32& x) {
    size_t p = x.find_first_not_of(U'0');
    return p == U32::npos ? U32() : x.substr(p);
}

// ---------------------------------------------------------------------------
// RE_DATE: (\d{4}|\d{2})年((0?[1-9]|1[0-2])月)?(((0?[1-9])|((1|2)[0-9])|30|31)([日号]))?
// ---------------------------------------------------------------------------

}  // namespace

U32 subDate(const U32& s) {
    return subAll(s, [&](size_t i) -> std::pair<U32, size_t> {
        size_t yearLen = 0;
        size_t dEnd = runDigits(s, i);
        if (dEnd - i >= 4 && i + 4 < s.size() && s[i + 4] == U'年') yearLen = 4;
        else if (dEnd - i >= 2 && i + 2 < s.size() && s[i + 2] == U'年')
            yearLen = 2;
        if (!yearLen) return {U32(), i};
        size_t pos = i + yearLen + 1;
        size_t monthLen = 0, dayLen = 0;
        char32_t dayUnit = 0;
        // month: (0?[1-9]|1[0-2])月
        if (pos < s.size()) {
            char32_t c = s[pos];
            auto twoThenYue = [&](char32_t lo, char32_t hi) {
                return pos + 2 < s.size() && s[pos + 1] >= lo &&
                       s[pos + 1] <= hi && s[pos + 2] == U'月';
            };
            if (c == U'0') {
                if (twoThenYue(U'1', U'9')) monthLen = 2;
            } else if (c == U'1') {
                // branch 0?[1-9]: single '1' then 月
                if (pos + 1 < s.size() && s[pos + 1] == U'月') monthLen = 1;
                else if (twoThenYue(U'0', U'2')) monthLen = 2;  // 1[0-2]
            } else if (c >= U'2' && c <= U'9') {
                if (pos + 1 < s.size() && s[pos + 1] == U'月') monthLen = 1;
            }
        }
        if (monthLen) pos += monthLen + 1;
        // day: ((0?[1-9])|((1|2)[0-9])|30|31)([日号])
        auto unitAfter = [&](size_t lastIdx) {
            if (lastIdx + 1 < s.size() &&
                (s[lastIdx + 1] == U'日' || s[lastIdx + 1] == U'号')) {
                dayUnit = s[lastIdx + 1];
                dayLen = lastIdx - pos + 1;
                return true;
            }
            return false;
        };
        if (pos < s.size()) {
            char32_t c = s[pos];
            // branch 1: 0?[1-9]
            if (c == U'0') {
                if (pos + 1 < s.size() && s[pos + 1] >= U'1' &&
                    s[pos + 1] <= U'9')
                    unitAfter(pos + 1);
            } else if (c >= U'1' && c <= U'9') {
                unitAfter(pos);
            }
            if (!dayLen) {
                // branch 2: (1|2)[0-9]
                if ((c == U'1' || c == U'2') && pos + 1 < s.size() &&
                    isDig(s[pos + 1]))
                    unitAfter(pos + 1);
            }
            if (!dayLen) {
                // branch 3: 30 | 31
                if (c == U'3' && pos + 1 < s.size() &&
                    (s[pos + 1] == U'0' || s[pos + 1] == U'1'))
                    unitAfter(pos + 1);
            }
        }
        size_t end = pos + (dayLen ? dayLen + 1 : 0);
        U32 result = verbalizeDigit(s.substr(i, yearLen), false) + dec("年");
        if (monthLen)
            result +=
                verbalizeCardinal(s.substr(pos - monthLen - 1, monthLen)) +
                dec("月");
        if (dayLen) {
            result += verbalizeCardinal(s.substr(pos, dayLen)) +
                      dec(dayUnit == U'日' ? "日" : "号");
        }
        return {result, end};
    });
}

// RE_DATE2: (\d{4})([- /.])(0[1-9]|1[012])\2(0[1-9]|[12][0-9]|3[01])
U32 subDate2(const U32& s) {
    return subAll(s, [&](size_t i) -> std::pair<U32, size_t> {
        if (runDigits(s, i) - i != 4) return {U32(), i};
        size_t yEnd = i + 4;
        if (yEnd >= s.size()) return {U32(), i};
        char32_t sep = s[yEnd];
        if (!(sep == U'-' || sep == U'/' || sep == U' ' || sep == U'.'))
            return {U32(), i};
        size_t j = yEnd + 1;
        auto monthOk = [&]() {
            return j + 1 < s.size() &&
                   ((s[j] == U'0' && s[j + 1] >= U'1' && s[j + 1] <= U'9') ||
                    (s[j] == U'1' && s[j + 1] >= U'0' && s[j + 1] <= U'2'));
        };
        if (!monthOk() || j + 2 >= s.size() || s[j + 2] != sep)
            return {U32(), i};
        size_t k = j + 3;
        bool dayOk = false;
        if (k + 1 < s.size()) {
            dayOk = (s[k] == U'0' && s[k + 1] >= U'1' && s[k + 1] <= U'9') ||
                    ((s[k] == U'1' || s[k] == U'2') && isDig(s[k + 1])) ||
                    (s[k] == U'3' &&
                     (s[k + 1] == U'0' || s[k + 1] == U'1'));
        }
        if (!dayOk) return {U32(), i};
        U32 result = verbalizeDigit(s.substr(i, 4), false) + dec("年") +
                     verbalizeCardinal(s.substr(j, 2)) + dec("月") +
                     verbalizeCardinal(s.substr(k, 2)) + dec("日");
        return {result, k + 2};
    });
}

// ---------------------------------------------------------------------------
// RE_TIME_RANGE / RE_TIME — shared replace_time
//   hour: ([0-1]?[0-9]|2[0-3]); min [0-5][0-9]; opt (:([0-5][0-9]))
// ---------------------------------------------------------------------------

namespace {

// one H:MM(:SS)? block at i; returns end or i on failure
typedef struct TimeBlk {
    U32 hour, minute, second;
} TimeBlk;

size_t matchTimeBlock(const U32& s, size_t i, TimeBlk* blk) {
    size_t hLen = 0;
    if (i < s.size()) {
        char32_t c = s[i];
        if (c == U'0' || c == U'1') {
            if (i + 2 < s.size() && isDig(s[i + 1]) && s[i + 2] == U':')
                hLen = 2;
            else if (i + 1 < s.size() && s[i + 1] == U':')
                hLen = 1;
        } else if (isDig(c)) {
            if (i + 1 < s.size() && s[i + 1] == U':') hLen = 1;
        }
        if (!hLen && c == U'2' && i + 2 < s.size() && s[i + 1] >= U'0' &&
            s[i + 1] <= U'3' && s[i + 2] == U':')
            hLen = 2;
    }
    if (!hLen) return i;
    blk->hour = s.substr(i, hLen);
    size_t mPos = i + hLen + 1;
    if (mPos + 1 >= s.size() || !(s[mPos] >= U'0' && s[mPos] <= U'5') ||
        !isDig(s[mPos + 1]))
        return i;
    blk->minute = s.substr(mPos, 2);
    size_t end = mPos + 2;
    blk->second.clear();
    if (end + 2 < s.size() && s[end] == U':' && s[end + 1] >= U'0' &&
        s[end + 1] <= U'5' && isDig(s[end + 2])) {
        blk->second = s.substr(end + 1, 2);
        end += 3;
    }
    return end;
}

void appendTimePart(U32* result, const U32& minute, bool hasSecond,
                    const U32& second, bool useFirstMinuteForHalf,
                    const U32& firstMinute) {
    U32 minStrip = stripZeros(minute);
    if (!minStrip.empty()) {
        // python replace_time range half checks int(minute) of the FIRST
        // minute for both halves (bug-compat)
        const U32& halfCmp =
            useFirstMinuteForHalf ? firstMinute : minute;
        if (halfCmp == U"30")
            *result += dec("半");
        else
            *result += timeNum2strPub(minute) + dec("分");
    }
    if (hasSecond) {
        U32 secStrip = stripZeros(second);
        if (!secStrip.empty())
            *result += timeNum2strPub(second) + dec("秒");
    }
}

}  // namespace

U32 subTimes(const U32& s) {
    // pass 1: RE_TIME_RANGE
    U32 pass1 = subAll(s, [&](size_t i) -> std::pair<U32, size_t> {
        TimeBlk b1, b2;
        size_t e1 = matchTimeBlock(s, i, &b1);
        if (e1 == i) return {U32(), i};
        if (e1 >= s.size() || (s[e1] != U'~' && s[e1] != U'-'))
            return {U32(), i};
        size_t start2 = e1 + 1;
        size_t e2 = matchTimeBlock(s, start2, &b2);
        if (e2 == start2) return {U32(), i};
        U32 result = num2str(b1.hour) + dec("点");
        appendTimePart(&result, b1.minute, !b1.second.empty(), b1.second,
                       false, b1.minute);
        result += dec("至") + num2str(b2.hour) + dec("点");
        appendTimePart(&result, b2.minute, !b2.second.empty(), b2.second,
                       true, b1.minute);
        return {result, e2};
    });
    // pass 2: RE_TIME
    U32 pass2 = subAll(pass1, [&](size_t i) -> std::pair<U32, size_t> {
        TimeBlk b;
        size_t e = matchTimeBlock(pass1, i, &b);
        if (e == i) return {U32(), i};
        U32 result = num2str(b.hour) + dec("点");
        appendTimePart(&result, b.minute, !b.second.empty(), b.second, false,
                       b.minute);
        return {result, e};
    });
    return pass2;
}

// ---------------------------------------------------------------------------
// part 2: numeric / unit / phone rules (order-sensitive, leftmost-first)
// ---------------------------------------------------------------------------

namespace {

bool isSupChar(char32_t c) {
    return c == 0x2070 || c == 0x00B9 || c == 0x00B2 || c == 0x00B3 ||
           (c >= 0x2074 && c <= 0x2079) || c == 0x02E3 || c == 0x02B8 ||
           c == 0x207F;
}
size_t runSup(const U32& s, size_t i) {
    size_t j = i;
    while (j < s.size() && isSupChar(s[j])) ++j;
    return j;
}

char32_t supDigit(char32_t c) {
    switch (c) {
        case 0x2070: return U'0';
        case 0x00B9: return U'1';
        case 0x00B2: return U'2';
        case 0x00B3: return U'3';
        case 0x2074: return U'4';
        case 0x2075: return U'5';
        case 0x2076: return U'6';
        case 0x2077: return U'7';
        case 0x2078: return U'8';
        case 0x2079: return U'9';
        case 0x02E3: return U'x';
        case 0x02B8: return U'y';
        case 0x207F: return U'n';
    }
    return 0;
}

U32 fromCp(char32_t c) { return U32(1, c); }

// ordered unit list matcher: first alternative that matches wins
struct UnitList {
    std::vector<U32> alts;
};
const U32* matchUnit(const UnitList& ul, const U32& s, size_t i, size_t* len) {
    for (const auto& u : ul.alts) {
        if (i + u.size() <= s.size() &&
            std::equal(u.begin(), u.end(), s.begin() + static_cast<long>(i))) {
            *len = u.size();
            return &u;
        }
    }
    return nullptr;
}

UnitList makeList(std::initializer_list<const char*> lits) {
    UnitList ul;
    for (auto* l : lits) ul.alts.push_back(dec(l));
    return ul;
}

// RE_TO_RANGE unit alternation in exact source order (m before mm!)
const UnitList& toRangeUnits() {
    static const UnitList ul = makeList({
        "%", "°C", "℃", "度", "摄氏度", "cm2", "cm²", "cm3",
        "cm³", "cm", "db", "ds", "kg", "km", "m2", "m²", "m³",
        "m3", "ml", "m", "mm", "s",
    });
    return ul;
}

// RE_ASMD operand: (-?)(\d+(\.\d+)?[sup]*) | (\.\d+[sup]*) | ([A-Za-z][sup]*)
struct AsmdOpnd {
    bool ok = false;
    size_t end = 0;
    U32 text;
};
AsmdOpnd matchOperand(const U32& s, size_t i) {
    AsmdOpnd o;
    // branch 1
    size_t j = i;
    if (j < s.size() && s[j] == U'-') ++j;
    if (j < s.size() && isDig(s[j])) {
        size_t e = runDigits(s, j);
        if (e + 1 < s.size() && s[e] == U'.' && isDig(s[e + 1]))
            e = runDigits(s, e + 1);
        e = runSup(s, e);
        o.ok = true;
        o.end = e;
        o.text = s.substr(i, e - i);
        return o;
    }
    // branch 2
    if (s[i] == U'.' && i + 1 < s.size() && isDig(s[i + 1])) {
        size_t e = runSup(s, runDigits(s, i + 1));
        o.ok = true;
        o.end = e;
        o.text = s.substr(i, e - i);
        return o;
    }
    // branch 3
    char32_t c = s[i];
    if ((c >= U'a' && c <= U'z') || (c >= U'A' && c <= U'Z')) {
        size_t e = runSup(s, i + 1);
        o.ok = true;
        o.end = e;
        o.text = s.substr(i, e - i);
    }
    return o;
}

bool isAsmdOp(char32_t c) {
    return c == U'+' || c == U'-' || c == 0x00D7 || c == 0x00F7 || c == U'=';
}

U32 asmdMap(char32_t op) {
    switch (op) {
        case U'+': return dec("加");
        case U'-': return dec("减");
        case 0x00D7: return dec("乘");
        case 0x00F7: return dec("除");
        default: return dec("等于");
    }
}

// RE_NUMBER replace semantics shared by NUMBER / DECIMAL / RANGE operands
U32 replaceNumberHit(bool dotForm, bool neg, const U32& digitsOrPure) {
    // bug-compat: python replace_number reads group(5), the FULL "(\.\d+)"
    // match including the dot, so pure decimals go through
    // num2str(".5") -> 零点五
    if (dotForm) return num2str(U32(1, U'.') + digitsOrPure);
    U32 r = neg ? dec("负") : U32();
    r += num2str(digitsOrPure);
    return r;
}

// generic RE_NUMBER.sub over t
U32 numberSub(const U32& t) {
    return subAll(t, [&](size_t i) -> std::pair<U32, size_t> {
        size_t j = i;
        bool neg = false;
        if (j < t.size() && t[j] == U'-') { neg = true; ++j; }
        if (j < t.size() && isDig(t[j])) {
            size_t d = runDigits(t, j);
            size_t e = d;
            if (e + 1 < t.size() && t[e] == U'.' && isDig(t[e + 1]))
                e = runDigits(t, e + 1);
            return {replaceNumberHit(false, neg, t.substr(j, e - j)), e};
        }
        // sign backtrack happens implicitly: retry without '-'
        if (neg) {
            if (isDig(t[i])) { /* handled above only when '-' present */
            }
        }
        // branch 2: \.(\d+)
        if (t[i] == U'.' && i + 1 < t.size() && isDig(t[i + 1])) {
            size_t e = runDigits(t, i + 1);
            return {replaceNumberHit(true, false, t.substr(i + 1, e - i - 1)),
                    e};
        }
        return {U32(), i};
    });
}

}  // namespace

U32 subToRange(const U32& s) {
    // verbatim replacement: group(0).replace("~", "至")
    return subAll(s, [&](size_t i) -> std::pair<U32, size_t> {
        auto operand = [&](size_t p, size_t* end) -> bool {
            size_t q = p;
            if (q < s.size() && s[q] == U'-') ++q;
            if (q < s.size() && isDig(s[q])) {
                size_t e = runDigits(s, q);
                if (e + 1 < s.size() && s[e] == U'.' && isDig(s[e + 1]))
                    e = runDigits(s, e + 1);
                *end = e;
                return true;
            }
            if (p < s.size() && s[p] == U'.' && p + 1 < s.size() &&
                isDig(s[p + 1])) {
                *end = runDigits(s, p + 1);
                return true;
            }
            return false;
        };
        size_t e1;
        if (!operand(i, &e1)) return {U32(), i};
        size_t ulen = 0;
        if (!matchUnit(toRangeUnits(), s, e1, &ulen)) return {U32(), i};
        size_t mid = e1 + ulen;
        if (mid >= s.size() || s[mid] != U'~') return {U32(), i};
        size_t e2;
        if (!operand(mid + 1, &e2)) return {U32(), i};
        size_t ulen2 = 0;
        if (!matchUnit(toRangeUnits(), s, e2, &ulen2)) return {U32(), i};
        size_t total = e2 + ulen2 - i;
        U32 result = s.substr(i, total);
        size_t p;
        while ((p = result.find(U'~')) != U32::npos)
            result.replace(p, 1, dec("至"));
        return {result, e2 + ulen2};
    });
}

U32 subTemperature(const U32& s) {
    // (-?)(\d+(\.\d+)?)(°C|℃|度|摄氏度)
    return subAll(s, [&](size_t i) -> std::pair<U32, size_t> {
        size_t j = i;
        bool neg = false;
        if (s[j] == U'-') { neg = true; ++j; }
        if (!isDig(s[j])) return {U32(), i};
        size_t d = runDigits(s, j);
        size_t e = d;
        if (e + 1 < s.size() && s[e] == U'.' && isDig(s[e + 1]))
            e = runDigits(s, e + 1);
        // unit alternation in order
        const char* units[] = {"°C", "℃", "度", "摄氏度"};
        size_t uLen = 0;
        const char* hit = nullptr;
        for (auto* u : units) {
            U32 t = dec(u);
            if (e + t.size() <= s.size() &&
                std::equal(t.begin(), t.end(), s.begin() + static_cast<long>(e))) {
                uLen = t.size();
                hit = u;
                break;
            }
        }
        if (!hit) return {U32(), i};
        // bug-compat: python reads match.group(3) as the unit, but group 3
        // is the inner (\.\d+)? decimal group -- it can never be "摄氏度",
        // so the replacement always emits 度.
        (void)hit;
        U32 result = neg ? dec("零下") : U32();
        result += num2str(s.substr(j, e - j));
        result += dec("度");
        return {result, e + uLen};
    });
}

U32 replaceMeasure(const U32& s) {
    U32 t = s;
    for (size_t k = 0; k < kMeasureDict_len; ++k) {
        U32 key = dec(kMeasureDict[k].first);
        U32 val = dec(kMeasureDict[k].second);
        size_t p;
        while ((p = t.find(key)) != U32::npos)
            t.replace(p, key.size(), val);
    }
    return t;
}

bool asmdSearch(const U32& s) {
    for (size_t i = 0; i < s.size(); ++i) {
        auto l = matchOperand(s, i);
        if (!l.ok || l.end >= s.size() || !isAsmdOp(s[l.end])) continue;
        auto r = matchOperand(s, l.end + 1);
        if (r.ok) return true;
    }
    return false;
}

U32 subAsmd(const U32& s) {
    return subAll(s, [&](size_t i) -> std::pair<U32, size_t> {
        auto l = matchOperand(s, i);
        if (!l.ok || l.end >= s.size() || !isAsmdOp(s[l.end]))
            return {U32(), i};
        auto r = matchOperand(s, l.end + 1);
        if (!r.ok) return {U32(), i};
        return {l.text + asmdMap(s[l.end]) + r.text, r.end};
    });
}

U32 subPower(const U32& s) {
    return subAll(s, [&](size_t i) -> std::pair<U32, size_t> {
        if (!isSupChar(s[i])) return {U32(), i};
        size_t e = runSup(s, i);
        U32 num;
        for (size_t k = i; k < e; ++k) num.push_back(supDigit(s[k]));
        return {dec("的") + num + dec("次方"), e};
    });
}

U32 subFrac(const U32& s) {
    // (-?)(\d+)/(\d+)
    return subAll(s, [&](size_t i) -> std::pair<U32, size_t> {
        size_t j = i;
        bool neg = false;
        if (s[j] == U'-') { neg = true; ++j; }
        size_t n = runDigits(s, j);
        if (n == j || n >= s.size() || s[n] != U'/') return {U32(), i};
        size_t d = runDigits(s, n + 1);
        if (d == n + 1) return {U32(), i};
        U32 result = neg ? dec("负") : U32();
        result += num2str(s.substr(n + 1, d - n - 1)) + dec("分之") +
                  num2str(s.substr(j, n - j));
        return {result, d};
    });
}

U32 subPercentage(const U32& s) {
    // (-?)(\d+(\.\d+)?)%
    return subAll(s, [&](size_t i) -> std::pair<U32, size_t> {
        size_t j = i;
        bool neg = false;
        if (s[j] == U'-') { neg = true; ++j; }
        if (!isDig(s[j])) return {U32(), i};
        size_t d = runDigits(s, j);
        size_t e = d;
        if (e + 1 < s.size() && s[e] == U'.' && isDig(s[e + 1]))
            e = runDigits(s, e + 1);
        if (e >= s.size() || s[e] != U'%') return {U32(), i};
        return {(neg ? dec("负") : U32()) + dec("百分之") +
                    num2str(s.substr(j, e - j)),
                e + 1};
    });
}

namespace {

// verbalize phone text like python phone2str(mobile)
U32 phone2str(const U32& text, bool mobile) {
    std::vector<U32> parts;
    if (mobile) {
        // strip('+').split(): remove leading/trailing '+', split on spaces
        size_t b = 0, e = text.size();
        while (b < e && text[b] == U'+') ++b;
        while (e > b && text[e - 1] == U'+') --e;
        size_t k = b;
        while (k < e) {
            while (k < e && text[k] == U' ') ++k;
            size_t st = k;
            while (k < e && text[k] != U' ') ++k;
            if (k > st) parts.push_back(text.substr(st, k - st));
        }
    } else {
        size_t st = 0;
        for (size_t k = 0; k <= text.size(); ++k) {
            if (k == text.size() || text[k] == U'-') {
                parts.push_back(text.substr(st, k - st));
                st = k + 1;
            }
        }
    }
    U32 result;
    for (size_t k = 0; k < parts.size(); ++k) {
        if (k) result += dec("，");
        result += verbalizeDigit(parts[k], true);
    }
    return result;
}

bool prevNotDigit(const U32& s, size_t i) {
    return i == 0 || !isDig(s[i - 1]);
}

}  // namespace

U32 subMobilePhone(const U32& s) {
    // (?<!\d)((\+?86 ?)?1([38]\d|5[0-35-9]|7[678]|9[89])\d{8})(?!\d)
    return subAll(s, [&](size_t i) -> std::pair<U32, size_t> {
        if (!prevNotDigit(s, i)) return {U32(), i};
        // try with prefix (\+?86 ?)? then without
        for (int prefMode = 0; prefMode < 2; ++prefMode) {
            size_t body = i;
            bool usedPrefix = false;
            if (prefMode == 0) {
                size_t q = i;
                if (q < s.size() && s[q] == U'+') ++q;
                if (q + 1 < s.size() && s[q] == U'8' && s[q + 1] == U'6') {
                    q += 2;
                    if (q < s.size() && s[q] == U' ') ++q;
                    body = q;
                    usedPrefix = true;
                }
                if (!usedPrefix && s[i] == U'+')
                    continue;  // '+' must be consumed by prefix
            }
            if (body >= s.size() || s[body] != U'1') continue;
            size_t rest = body + 1;
            bool twoOk = false;
            if (rest + 1 < s.size()) {
                char32_t a = s[rest], b = s[rest + 1];
                twoOk = ((a == U'3' || a == U'8') && isDig(b)) ||
                        (a == U'5' && b >= U'0' && b <= U'3') ||
                        (a == U'5' && b >= U'5' && b <= U'9') ||
                        (a == U'7' && (b == U'6' || b == U'7' || b == U'8')) ||
                        (a == U'9' && (b == U'8' || b == U'9'));
            }
            if (!twoOk) continue;
            if (rest + 10 > s.size()) continue;
            bool tailDigits = true;
            for (size_t k = rest + 2; k + 1 < rest + 10; ++k)
                if (!isDig(s[k])) { tailDigits = false; break; }
            if (!tailDigits) continue;
            size_t tailEnd = rest + 2 + 8;
            if (tailEnd < s.size() && isDig(s[tailEnd])) continue;  // (?!\d)
            U32 text = s.substr(i, tailEnd - i);
            return {phone2str(text, true), tailEnd};
        }
        return {U32(), i};
    });
}

U32 subTelephone(const U32& s) {
    // (?<!\d)((0(10|2[1-3]|[3-9]\d{2})-?)?[1-9]\d{6,7})(?!\d)
    return subAll(s, [&](size_t i) -> std::pair<U32, size_t> {
        if (!prevNotDigit(s, i)) return {U32(), i};
        for (int areaMode = 0; areaMode < 2; ++areaMode) {
            size_t local = i;
            if (areaMode == 0) {
                size_t q = i;
                if (q >= s.size() || s[q] != U'0') continue;
                ++q;
                bool codeOk = false;
                size_t afterCode = 0;
                // 10 | 2[1-3] | [3-9]\d{2}, first-match order
                if (q + 1 < s.size() && s[q] == U'1' && s[q + 1] == U'0') {
                    codeOk = true;
                    afterCode = q + 2;
                } else if (q + 1 < s.size() && s[q] == U'2' &&
                           s[q + 1] >= U'1' && s[q + 1] <= U'3') {
                    codeOk = true;
                    afterCode = q + 2;
                } else if (q + 2 < s.size() && s[q] >= U'3' &&
                           s[q] <= U'9' && isDig(s[q + 1]) &&
                           isDig(s[q + 2])) {
                    codeOk = true;
                    afterCode = q + 3;
                }
                if (!codeOk) continue;
                if (afterCode < s.size() && s[afterCode] == U'-')
                    ++afterCode;
                local = afterCode;
            }
            if (local >= s.size() || !(s[local] >= U'1' && s[local] <= U'9'))
                continue;
            size_t dEnd = runDigits(s, local);
            size_t count = dEnd - local;  // [1-9] plus \d{6,7}
            // greedy 7 then 6, lookahead each
            for (size_t take : {size_t(8), size_t(7)}) {
                if (count < take) continue;
                size_t end = local + take;
                if (end < s.size() && isDig(s[end])) continue;  // (?!\d)
                U32 text = s.substr(i, end - i);
                return {phone2str(text, false), end};
            }
        }
        return {U32(), i};
    });
}

U32 subNationalUniformNumber(const U32& s) {
    // (400)(-)?\d{3}(-)?\d{4}
    return subAll(s, [&](size_t i) -> std::pair<U32, size_t> {
        size_t j = i;
        if (j + 3 > s.size() || s[j] != U'4' || s[j + 1] != U'0' ||
            s[j + 2] != U'0')
            return {U32(), i};
        j += 3;
        if (j < s.size() && s[j] == U'-') ++j;
        if (j + 3 > s.size()) return {U32(), i};
        if (!(isDig(s[j]) && isDig(s[j + 1]) && isDig(s[j + 2])))
            return {U32(), i};
        j += 3;
        if (j < s.size() && s[j] == U'-') ++j;
        if (j + 4 > s.size()) return {U32(), i};
        for (int k = 0; k < 4; ++k)
            if (!isDig(s[j + static_cast<size_t>(k)])) return {U32(), i};
        j += 4;
        return {phone2str(s.substr(i, j - i), false), j};
    });
}

U32 subRange(const U32& s) {
    // (?<![\d\+\-\×÷=])((-?)((\d+)(\.\d+)?))[-~]((-?)((\d+)(\.\d+)?))(?![\d\+\-\×÷=])
    auto badBefore = [&](size_t i) {
        if (i == 0) return false;
        char32_t c = s[i - 1];
        return isDig(c) || c == U'+' || c == U'-' || c == 0x00D7 ||
               c == 0x00F7 || c == U'=';
    };
    auto badAfter = [&](size_t i) {
        if (i >= s.size()) return false;
        char32_t c = s[i];
        return isDig(c) || c == U'+' || c == U'-' || c == 0x00D7 ||
               c == 0x00F7 || c == U'=';
    };
    auto operand = [&](size_t p, size_t* end) -> bool {
        size_t q = p;
        if (q < s.size() && s[q] == U'-') ++q;
        if (q >= s.size() || !isDig(s[q])) return false;
        size_t e = runDigits(s, q);
        if (e + 1 < s.size() && s[e] == U'.' && isDig(s[e + 1]))
            e = runDigits(s, e + 1);
        *end = e;
        return true;
    };
    return subAll(s, [&](size_t i) -> std::pair<U32, size_t> {
        if (badBefore(i)) return {U32(), i};
        size_t e1;
        if (!operand(i, &e1)) return {U32(), i};
        if (e1 >= s.size() || (s[e1] != U'-' && s[e1] != U'~'))
            return {U32(), i};
        size_t start2 = e1 + 1;
        size_t e2;
        if (!operand(start2, &e2)) return {U32(), i};
        if (badAfter(e2)) return {U32(), i};
        U32 first = numberSub(s.substr(i, e1 - i));
        U32 second = numberSub(s.substr(start2, e2 - start2));
        return {first + dec("到") + second, e2};
    });
}

U32 subInteger(const U32& s) {
    // (-)(\d+)
    return subAll(s, [&](size_t i) -> std::pair<U32, size_t> {
        if (s[i] != U'-') return {U32(), i};
        size_t d = runDigits(s, i + 1);
        if (d == i + 1) return {U32(), i};
        return {dec("负") + num2str(s.substr(i + 1, d - i - 1)), d};
    });
}

U32 subVersionNum(const U32& s) {
    // ((\d+)(\.\d+)(\.\d+)?(\.\d+)+): digits followed by >=2 dot-groups
    return subAll(s, [&](size_t i) -> std::pair<U32, size_t> {
        size_t d = runDigits(s, i);
        if (d == i) return {U32(), i};
        size_t e = d;
        int groups = 0;
        while (e + 1 < s.size() && s[e] == U'.' && isDig(s[e + 1])) {
            e = runDigits(s, e + 1);
            ++groups;
        }
        if (groups < 2) return {U32(), i};
        U32 result;
        for (size_t k = i; k < e; ++k) {
            if (s[k] == U'.')
                result += dec("点");
            else
                result += num2str(s.substr(k, 1));
        }
        return {result, e};
    });
}

U32 subDecimalNum(const U32& s) {
    // (-?)((\d+)(\.\d+))|(\.(\d+))
    return subAll(s, [&](size_t i) -> std::pair<U32, size_t> {
        size_t j = i;
        bool neg = false;
        if (s[j] == U'-') { neg = true; ++j; }
        if (isDig(s[j])) {
            size_t d = runDigits(s, j);
            if (d + 1 < s.size() && s[d] == U'.' && isDig(s[d + 1])) {
                size_t e = runDigits(s, d + 1);
                return {replaceNumberHit(false, neg, s.substr(j, e - j)), e};
            }
        }
        if (s[i] == U'.' && i + 1 < s.size() && isDig(s[i + 1])) {
            size_t e = runDigits(s, i + 1);
            return {replaceNumberHit(true, false, s.substr(i + 1, e - i - 1)),
                    e};
        }
        return {U32(), i};
    });
}

namespace {
const UnitList& quantifierUnits() {
    static const UnitList ul = [] {
        // kQuantifierAlts is an array of literal alternatives in regex
        // alternation order (first match at a position wins)
        UnitList u;
        for (size_t k = 0; k < kQuantifierAlts_len; ++k)
            u.alts.push_back(dec(kQuantifierAlts[k]));
        return u;
    }();
    return ul;
}
}  // namespace

U32 subPositiveQuantifiers(const U32& s) {
    // (\d+)([多余几\+])? COM_QUANTIFIERS
    return subAll(s, [&](size_t i) -> std::pair<U32, size_t> {
        size_t d = runDigits(s, i);
        if (d == i) return {U32(), i};
        size_t pos = d;
        U32 mid;
        if (pos < s.size()) {
            char32_t c = s[pos];
            if (c == U'多' || c == U'余' || c == U'几' || c == U'+') {
                mid = fromCp(c);
                ++pos;
            }
        }
        size_t ulen = 0;
        const U32* unit = matchUnit(quantifierUnits(), s, pos, &ulen);
        if (!unit) return {U32(), i};
        U32 number = num2str(s.substr(i, d - i));
        if (number == dec("二")) number = dec("两");
        U32 m2 = mid;
        if (m2 == fromCp(U'+')) m2 = dec("多");
        return {number + m2 + *unit, pos + ulen};
    });
}

U32 subDefaultNum(const U32& s) {
    // \d{3}\d*
    return subAll(s, [&](size_t i) -> std::pair<U32, size_t> {
        size_t d = runDigits(s, i);
        if (d - i < 3) return {U32(), i};
        return {verbalizeDigit(s.substr(i, d - i), true), d};
    });
}

U32 subNumber(const U32& s) {
    // (-?)((\d+)(\.\d+)?)|(\.(\d+))
    return subAll(s, [&](size_t i) -> std::pair<U32, size_t> {
        size_t j = i;
        bool neg = false;
        if (s[j] == U'-') { neg = true; ++j; }
        if (j < s.size() && isDig(s[j])) {
            size_t d = runDigits(s, j);
            size_t e = d;
            if (e + 1 < s.size() && s[e] == U'.' && isDig(s[e + 1]))
                e = runDigits(s, e + 1);
            return {replaceNumberHit(false, neg, s.substr(j, e - j)), e};
        }
        if (s[i] == U'.' && i + 1 < s.size() && isDig(s[i + 1])) {
            size_t e = runDigits(s, i + 1);
            return {replaceNumberHit(true, false, s.substr(i + 1, e - i - 1)),
                    e};
        }
        return {U32(), i};
    });
}

}  // namespace gsv::textfront
