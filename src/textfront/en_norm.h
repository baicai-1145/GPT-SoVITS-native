// en_norm.h — English text normalization, a faithful port of CPUFast
// text/en_normalization/expend.py::normalize (including the subset of the
// `inflect` engine it uses: number_to_words, ordinal, grouped year reading).
#pragma once
#include <string>

namespace gsv::textfront::en {

// expend.normalize(text): ordinals ("1. "), ASMD, negative numbers, comma
// groups, times, measurements, currency, decimals, fractions, "1st"-style
// ordinals, plain numbers, accent stripping, percent, charset filter,
// i.e./e.g., uppercase splitting.
std::string normalize(const std::string& utf8);

}  // namespace gsv::textfront::en
