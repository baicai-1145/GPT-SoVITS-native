// jieba.h — native port of jieba_fast.posseg.lcut() (the exact tokenizer call
// used by GPT-SoVITS CPUFast `text/chinese.py::_g2p`, default HMM=True).
//
// Golden contract: for any UTF-8 input `s`, lcut(s) must produce the same
// (word, tag) sequence as
//     [ (w.word, w.flag) for w in jieba_fast.posseg.lcut(s) ]
//
// Data source: src/textfront/data/jieba_trie.bin produced offline by
// tools/export_jieba_trie.py (see that script's docstring for the binary
// layout "GSVJTB01"). The file bundles:
//   * dict.txt frequency table as a sorted compact prefix tree
//     (every real word + every proper prefix, mirroring jieba's FREQ dict,
//     where prefixes carry freq==0),
//   * the POS tag pool,
//   * the POS-HMM (char_state_tab / prob_start / prob_trans / prob_emit)
//     as f64 log-probability tables identical to the pickled originals.
//
// Faithfulness notes (all verified against jieba_fast 0.39/0.53 sources):
//   * DAG + max-probability path replicate Tokenizer.calc/get_DAG including
//     the tie-break "on equal score take the largest end index".
//   * Single-char runs without dictionary hits go through the posseg
//     `__cut_detail` path (Han runs -> per-(state,pos) viterbi; other runs ->
//     m/eng/x tagging), NOT finalseg.
//   * Missing transition probability is -inf, missing emission is -3.14e100;
//     tie-breaking compares (prob, state-tuple) lexicographically, which the
//     exported state ids preserve.
//   * Python `\s`, `[\u4E00-\u9FD5...]` character classes and the
//     `[a-zA-Z0-9]`-prefix quirks of the tagging loops are replicated
//     bug-for-bug.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace gsv::textfront {

struct PosToken {
    std::string word;  // UTF-8
    std::string flag;  // POS tag, ASCII
};

class JiebaSegmenter {
public:
    JiebaSegmenter() = default;
    ~JiebaSegmenter();
    JiebaSegmenter(const JiebaSegmenter&) = delete;
    JiebaSegmenter& operator=(const JiebaSegmenter&) = delete;

    // Loads a "GSVJTB01" binary produced by tools/export_jieba_trie.py.
    // Returns false and fills *err on malformed input.
    bool load(const std::string& path, std::string* err);

    // Same as load() but from an in-memory buffer (used by unit tests that
    // build constructed-case dictionaries). Copies the data.
    bool loadMemory(const void* data, size_t size, std::string* err);

    // Mirrors jieba_fast.posseg.lcut(sentence) with default HMM=True.
    // Appends results to *out. Requires a successful load().
    void lcut(std::string_view utf8_sentence, std::vector<PosToken>* out) const;

private:
    struct Impl;
    Impl* impl_ = nullptr;
};

}  // namespace gsv::textfront
