// zh_verbalize.cpp — num.py verbalizers (getValue/verbalize_cardinal/
// verbalize_digit/num2str) ported line-by-line.
#include "zh_num.h"

#include <utility>
#include <vector>

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

const char* digitName(uint32_t d) {
    static const char* k[] = {"零", "一", "二", "三", "四", "五", "六",
                              "七", "八", "九"};
    return k[d];
}

}  // namespace

std::vector<U32> getValueImpl(const U32& v, bool useZero) {
    size_t nz = v.find_first_not_of(U'0');
    U32 stripped = nz == U32::npos ? U32() : v.substr(nz);
    if (stripped.empty()) return {};
    if (stripped.size() == 1) {
        if (useZero && stripped.size() < v.size())
            return {dec("零"), dec(digitName(stripped[0] - U'0'))};
        return {dec(digitName(stripped[0] - U'0'))};
    }
    size_t unit = 0;
    for (size_t p : {size_t(8), size_t(4), size_t(3), size_t(2), size_t(1)}) {
        if (p < stripped.size()) { unit = p; break; }
    }
    const char* us =
        unit == 1 ? "十" : unit == 2 ? "百" : unit == 3 ? "千"
                        : unit == 4 ? "万"
                                    : "亿";
    // recursion uses the un-stripped argument slices, like python
    auto firstPart = getValueImpl(v.substr(0, v.size() - unit), true);
    auto secondPart = getValueImpl(v.substr(v.size() - unit), true);
    firstPart.push_back(dec(us));
    for (auto& x : secondPart) firstPart.push_back(std::move(x));
    return firstPart;
}

namespace {
U32 joinParts(const std::vector<U32>& parts) {
    U32 r;
    for (auto& p : parts) r += p;
    return r;
}
}  // namespace

U32 verbalizeCardinal(const U32& value) {
    if (value.empty()) return {};
    size_t nz = value.find_first_not_of(U'0');
    U32 vs = nz == U32::npos ? U32() : value.substr(nz);
    if (vs.empty()) return dec("零");
    auto sym = getValueImpl(vs, true);
    // 一十* -> 十*
    if (sym.size() >= 2 && sym[0] == dec("一") && sym[1] == dec("十"))
        sym.erase(sym.begin());
    return joinParts(sym);
}

U32 verbalizeDigit(const U32& v, bool altOne) {
    U32 r;
    for (char32_t c : v)
        if (isDig(c)) r += dec(digitName(c - U'0'));
    if (altOne) {
        U32 yi = dec("一"), yao = dec("幺");
        size_t p;
        while ((p = r.find(yi)) != U32::npos) r.replace(p, yi.size(), yao);
    }
    return r;
}

U32 num2str(const U32& v) {
    size_t dot = v.find(U'.');
    U32 integer = dot == U32::npos ? v : v.substr(0, dot);
    U32 decimal = dot == U32::npos ? U32() : v.substr(dot + 1);
    U32 result = verbalizeCardinal(integer);
    bool endsZero = !decimal.empty() && decimal.back() == U'0';
    size_t lnz = decimal.find_last_not_of(U'0');
    U32 decTrim = lnz == U32::npos ? U32() : decimal.substr(0, lnz + 1);
    if (endsZero) decTrim.push_back(U'0');
    if (!decTrim.empty()) {
        if (result.empty()) result = dec("零");
        result += dec("点");
        result += verbalizeDigit(decTrim, false);
    }
    return result;
}

U32 timeNum2strPub(const U32& numStr) {
    size_t nz = numStr.find_first_not_of(U'0');
    U32 stripped = nz == U32::npos ? U32() : numStr.substr(nz);
    U32 result = num2str(stripped);
    if (!numStr.empty() && numStr[0] == U'0') result.insert(0, dec("零"));
    return result;
}

}  // namespace gsv::textfront
