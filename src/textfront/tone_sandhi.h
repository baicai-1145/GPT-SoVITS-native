// tone_sandhi.h — native port of CPUFast text/tone_sandhi.py (ToneSandhi).
//
// Golden contract: pre_merge_for_modify(psg.lcut(seg)) and
// modified_tone(word, pos, finals) reproduce the python class exactly,
// including:
//   - _split_word via jieba.cut_for_search + stable sort by length
//   - the try/except-swallowed merges (_merge_yi and both
//     _merge_continuous_three_tones passes): a python IndexError inside
//     (empty finals element) aborts THAT merge only, keeping the input seg
//   - modified_tone has NO such guard: an empty finals element where python
//     would index x[-1] means "python raises" — reported via `ok=false` so
//     the caller can mark the segment as a known gap.
#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "pinyin.h"

namespace gsv::textfront {

class JiebaSegmenter;

// Segmentation token with codepoint word storage for per-char rules.
struct SegToken {
    std::u32string word;
    std::string flag;
};

class ToneSandhi {
public:
    explicit ToneSandhi(const JiebaSegmenter* jb) : jb_(jb) {}

    // tone_modifier.pre_merge_for_modify(seg). Python swallows failures of
    // _merge_yi / _merge_continuous_three_tones{,_2}; this returns the same
    // surviving sequence (failures printed to stderr like the python prints).
    void preMergeForModify(std::vector<SegToken>* seg,
                           const PypinyinResolver& resolver) const;

    // tone_modifier.modified_tone(word, pos, finals). Mutates `finals`
    // in place. `ok=false` mirrors "python would raise" (empty finals
    // element indexed); on ok=false contents are unspecified.
    void modifiedTone(const std::u32string& word, std::string_view pos,
                      const PypinyinResolver& resolver,
                      std::vector<std::string>* finals, bool* ok) const;

private:
    const JiebaSegmenter* jb_;
};

}  // namespace gsv::textfront
