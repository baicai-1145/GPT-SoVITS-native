// zh_num.h — number verbalization (num.py) and the NSW replacement rules of
// the zh_normalization chain, as hand-written scanners over codepoints.
// Each `subXxx` mirrors re.X.sub(replace_x, s): left-to-right non-overlapping.
#pragma once

#include <string>
#include <vector>

namespace gsv::textfront {

using U32 = std::u32string;

// num.py verbalizers
U32 verbalizeCardinal(const U32& value);
U32 verbalizeDigit(const U32& v, bool altOne);
U32 num2str(const U32& v);
U32 timeNum2strPub(const U32& numStr);  // _time_num2str
std::vector<U32> getValueImpl(const U32& v, bool useZero);

// chronology
U32 subDate(const U32& s);       // RE_DATE
U32 subDate2(const U32& s);      // RE_DATE2
U32 subTimes(const U32& s);      // RE_TIME_RANGE first, then RE_TIME

// num / quantifier / phonecode
U32 subToRange(const U32& s);
U32 subTemperature(const U32& s);
U32 replaceMeasure(const U32& s);
U32 subAsmd(const U32& s);       // single pass; caller loops while search hits
bool asmdSearch(const U32& s);
U32 subPower(const U32& s);
U32 subFrac(const U32& s);
U32 subPercentage(const U32& s);
U32 subMobilePhone(const U32& s);
U32 subTelephone(const U32& s);
U32 subNationalUniformNumber(const U32& s);
U32 subRange(const U32& s);
U32 subInteger(const U32& s);
U32 subVersionNum(const U32& s);
U32 subDecimalNum(const U32& s);
U32 subPositiveQuantifiers(const U32& s);
U32 subDefaultNum(const U32& s);
U32 subNumber(const U32& s);

}  // namespace gsv::textfront
