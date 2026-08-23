// pinyin.h — native port of the pypinyin 0.55 resolution chain used by
// CPUFast `text/chinese2.py::_get_initials_finals` (the no-G2PW path).
//
// Golden contract (verified against env python):
//   initials/finals == zip(
//       lazy_pinyin(word, neutral_tone_with_five=True, style=Style.INITIALS),
//       lazy_pinyin(word, neutral_tone_with_five=True, style=Style.FINALS_TONE3))
//
// Algorithm mirrored from pypinyin sources:
//   core.Pinyin.seg          -> simple_seg (RE_HANS runs) + mmseg forward
//                               maximum matching over PHRASES_DICT keys
//                               (no_non_phrases=True)
//   DefaultConverter         -> phrase entry wins over per-char dict;
//                               heteronym=False takes the FIRST reading
//                               (phrase runs / char candidates)
//   style/_tone_convert.py   -> to_initials(strict=True),
//                               to_finals_tone3 + NeutralToneWith5Mixin
//                               (append '5' when no tone digit)
//   standard.py              -> convert_finals (zero-consonant y/w,
//                               j/q/x+u->v, iou/uei/uen restoration)
//
// Data: src/textfront/data/pinyin.bin produced by tools/export_pypinyin.py
// (format "GSPPYX01", see that script's docstring).
//
// Pluggability: `PinyinResolver` is the seam where B6's G2PW converter will
// be injected (it produces per-character TONE-marked syllables directly);
// everything downstream (initials/finals/sandhi/phones) consumes the same
// interface, so polyphonic logic never hardcodes at call sites.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace gsv::textfront {

// One resolved segmentation unit: a tone-marked pinyin syllable from the
// dictionary, or the original substring passed through verbatim
// (pypinyin errors='default' — raw units skip style conversion entirely).
struct PinyinSyl {
    std::string text;
    bool raw;
};

// Resolves a word (codepoints) to tone-MARKED syllables, one entry per
// segmentation unit (Han chars individually, non-Han runs verbatim).
class PinyinResolver {
public:
    virtual ~PinyinResolver() = default;
    virtual void resolve(const char32_t* word, size_t len,
                         std::vector<PinyinSyl>* syllables) const = 0;

    // chr(cp).isnumeric() equivalent (python str.isnumeric), consumed by
    // ToneSandhi's yi/bu rules. Default covers ASCII digits; the pypinyin
    // implementation overrides with the exported BMP table. After
    // normalization all numerals are Han characters, so subclasses that
    // only handle Han readings can keep the default.
    virtual bool isNumeric(uint32_t cp) const {
        return cp >= U'0' && cp <= U'9';
    }
};

class PypinyinResolver : public PinyinResolver {
public:
    PypinyinResolver() = default;
    ~PypinyinResolver();
    PypinyinResolver(const PypinyinResolver&) = delete;
    PypinyinResolver& operator=(const PypinyinResolver&) = delete;

    bool load(const std::string& path, std::string* err);
    bool loadMemory(const void* data, size_t size, std::string* err);

    void resolve(const char32_t* word, size_t len,
                 std::vector<PinyinSyl>* syllables) const override;

    // Exposes the lazy_pinyin segmentation for tests: Han runs split by
    // mmseg forward-maximum matching, other characters grouped into runs.
    void segWords(const char32_t* word, size_t len,
                  std::vector<std::u32string>* out) const;

    // Traditional -> simplified single codepoint (identity when absent).
    uint32_t t2s(uint32_t cp) const;

    // chr(cp).isnumeric() equivalent over the exported BMP table.
    bool isNumeric(uint32_t cp) const override;

private:
    struct Impl;
    Impl* impl_ = nullptr;
};

// Port of chinese2._get_initials_finals(word): zip of the INITIALS /
// FINALS_TONE3 lazy_pinyin outputs; raw passthrough units appear unchanged
// in both lists.
void getInitialsFinals(const PinyinResolver& resolver, const char32_t* word,
                       size_t len, std::vector<std::string>* initials,
                       std::vector<std::string>* finals);

// Port of pypinyin.contrib.tone_convert.to_initials(pinyin, strict=True).
std::string toInitials(std::string_view pinyin);

// Port of to_finals_tone3(pinyin, strict=True, neutral_tone_with_five=true)
// including the NeutralToneWith5Mixin '5' suffix behaviour.
std::string toFinalsTone3(std::string_view pinyin);

}  // namespace gsv::textfront
