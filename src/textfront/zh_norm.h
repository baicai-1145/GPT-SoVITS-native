// zh_norm.h — native port of CPUFast text/zh_normalization (TextNormalizer)
// and the chinese2.py punctuation layer. Golden contract:
//   TextNormalizer().normalize(text) == split()/normalizeSentence() chain
//   replace_punctuation / replace_consecutive_punctuation == chinese2.py
// All regexes from num.py/chronology.py/phonecode.py/quantifier.py are
// reimplemented as hand-written scanners (no third-party regex engine).
#pragma once

#include <string>
#include <vector>

namespace gsv::textfront {

using U32 = std::u32string;

// char_convert.tranditional_to_simplified (per-codepoint table).
U32 t2sText(const U32& in);

class TextNormalizer {
public:
    // _split(text, lang="zh"): drop spaces/specials, splitor -> \n, split.
    std::vector<U32> split(const U32& text) const;
    // normalize_sentence
    U32 normalizeSentence(const U32& sentence) const;
    // normalize
    std::vector<U32> normalize(const U32& text) const;

private:
    U32 postReplace(U32 s) const;
};

// chinese2.replace_punctuation (嗯->恩 呣->母, rep_map, keep Han+punct)
U32 replacePunctuation(const U32& text);

// chinese2.replace_consecutive_punctuation
U32 collapsePunct(const U32& text);

}  // namespace gsv::textfront
