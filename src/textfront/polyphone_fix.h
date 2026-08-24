// polyphone_fix.h — correct_pronunciation table port (B6/M3 final gap).
//
// CPUFast order of operations per segmented word, AFTER the sentence-level
// G2PW pass and BEFORE initials/finals conversion:
//   1. phrase_override_dict[word]              -> whole-word replacement
//   2. pp_dict[word] when len(word)>1          -> whole-word replacement
//   3. else per-character fallback             -> pp_dict[ch][0] replaces
//      only characters present in the table; other slots keep the G2PW
//      reading.
// All readings are pypinyin TONE3 ("yi2"). Data lives in
// data/polyphone_overrides.bin produced by tools/export_polyphone_overrides.py
// (runtime-loaded, not committed — same policy as cmudict.bin).
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace gsv::textfront {

class PolyphoneFixTable {
public:
    bool load(const std::string& binPath, std::string* err);

    // Port of correct_pronunciation(word, word_pinyins): mutates `rds`
    // (one TONE3 string per codepoint of `word`). Missing table -> no-op,
    // matching "no overrides installed".
    void apply(const std::u32string& word, std::vector<std::string>* rds) const;

    bool loaded() const { return loaded_; }

private:
    using RdList = std::vector<std::string>;
    // phrase-level: checked first as a whole word
    std::unordered_map<std::u32string, RdList> phrase_;
    // single-character domain for the per-cp fallback
    std::unordered_map<uint32_t, std::string> single_;
    bool loaded_ = false;
};

}  // namespace gsv::textfront
