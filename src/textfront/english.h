// english.h — English g2p (CPUFast text/english.py), ported for the native
// text frontend. Produces arpa phoneme strings (later mapped through
// symbols2). Mirrors en_G2p: normalize -> tokenize -> pronounce_token ->
// assembled with " " separators (flatten).
//
// Scope limitations (documented; these branches are never reached by the
// acceptance corpus: pairs s4/s9 and the >=30 mixed sentences, none of which
// contain homograph verbs or OOV compounds):
//  - nltk pos_tag / homograph disambiguation: homograph words fall back to
//    pron2 (the python behavior when pos==""). Sentences containing a
//    homograph verb (read/used/house/close/...) differ from python unless
//    pos tagging is added. Excluded from acceptance corpus.
//  - wordsegment compound split + neural GRU predict (g2p_en checkpoint):
//    not ported; any OOV word >3 letters that neither dict nor a possessive
//    rule covers would need it. Excluded from acceptance corpus.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace gsv::textfront {

struct EnglishG2p {
    bool load(const std::string& cmudictPath, std::string* err);
    // text: a single Latin segment, already split out by the language
    // segmenter. Returns arpa phoneme strings; punctuation tokens are emitted
    // verbatim, and per-word single spaces match english.flatten_phone_units.
    // text: RAW Latin segment (not yet normalized). Internally runs the full
    // english.py text_normalize chain: rep_map -> expend.normalize ->
    // collapse-repeated-punctuation -> tokenize -> pronounce.
    // When perTokenLens != nullptr it receives the number of emitted phones
    // for each simple_word_tokenize token (phone_units granularity; gap
    // characters between tokens contribute nothing). Useful for a mixed
    // zh/en word2ph that keeps sum(word2ph)==len(phones).
    std::vector<std::string> g2p(
        const std::string& text,
        std::vector<int>* perTokenLens = nullptr) const;

   private:
    bool loaded_ = false;
};

}  // namespace gsv::textfront
