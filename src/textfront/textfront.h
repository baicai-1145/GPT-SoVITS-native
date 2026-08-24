// textfront.h — TextFrontend: the M3 runtime orchestration layer.
//
// process(text) reproduces CPUFast TTS_infer_pack.TextPreprocessor.preprocess
// for lang="zh" followed by the chinese2 (is_g2pw=False) phone chain per
// segment:
//
//   1. replace_consecutive_punctuation   (set { ! ? … , . - }, same as
//                                        text.symbols.punctuation)
//   2. pre_seg_text segmentation 口径 (see splitSentences below)
//   3. per segment: TextNormalizer -> replace_punctuation ->
//      split-after-punct -> psg.lcut -> pre_merge_for_modify ->
//      getInitialsFinals(resolver) -> modified_tone -> _merge_erhua ->
//      _map_initial_final_to_phones -> symbols2 ids
//
// The resolver is pluggable (PinyinResolver interface): the default is the
// built-in pypinyin table; B6's G2PW converter injects via setResolver()
// without touching call sites.
#pragma once

#include <string>
#include <vector>

#include "chinese_g2p.h"

namespace gsv::textfront {

using U32 = std::u32string;

class TextFrontend {
public:
    // B6: G2PW polyphone engine paths. When provided to load(), the
    // pipeline builds a G2PWConverter(+resolver) and installs it as the
    // default polyphone path (sentence-level decisions). All three files
    // are runtime-loaded, never baked in; load() fails with a friendly
    // message when one is missing.
    struct G2pwOptions {
        std::string gsvPath;    // e.g. weights/g2pw_bert.gsv (~911MB)
        std::string assetsBin;  // e.g. data/g2pw_assets.bin (~8.6MB)
        std::string vocabPath;  // e.g. data/bert_vocab.txt
        std::string overridesBin;  // data/polyphone_overrides.bin (~2.6MB);
                                   // empty disables correct_pronunciation
    };
    struct Result {
        std::vector<std::string> sentences;  // synthesis segments, UTF-8;
                                             // each ends with a splits char
        std::vector<int> phones;             // symbols2 ids, all segments
        std::vector<int> word2ph;            // concatenated across segments
        bool ok = true;
        std::string error;
    };

    // data paths: jieba trie bin + pinyin bin; cmudictPath is optional and
    // enables the English segment path (mixed zh/en inputs). When absent,
    // Latin segments fail with an error instead of silently misreading.
    bool load(const std::string& triePath, const std::string& pinyinPath,
              std::string* err, const std::string& cmudictPath = {},
              const G2pwOptions* g2pw = nullptr);
    ~TextFrontend();

    // Injection seam for B6's G2PW converter. nullptr restores the default
    // pypinyin behaviour. The object must outlive this TextFrontend.
    void setResolver(const PinyinResolver* r) { g2p_.setResolver(r); }


    // Full pipeline. cutMethod: TTS_infer_pack text_segmentation_method id,
    // 0..5 ("cut0".."cut5"); runtime default is "cut1".
    bool process(const std::string& utf8Text, Result* out,
                 int cutMethod = 1) const;

    // TTS_infer_pack segmentation only (pre_seg_text), exposed for golden
    // fixtures and tests. Returns final synthesis segments as codepoints.
    //
    // 口径 (mirrors TextPreprocessor.pre_seg_text, lang="zh"):
    //   input has already been through replace_consecutive_punctuation
    //   1. strip('\n'); empty -> []
    //   2. if front char not in SPLITS and len(get_first(text)) < 4:
    //      prepend 。   (short non-punctuated input still forms a sentence)
    //   3. apply cutN(text) -> '\n'-joined pieces (see cut* below)
    //   4. collapse "\n\n" -> "\n"; split on '\n'
    //   5. filter_text: drop "" and " " entries
    //   6. merge_short_text_in_array(items, 5): concat until len>=5
    //   7. per item: skip len(strip())==0; skip pure-symbol
    //      (re.sub(r"\W+","",item)==""); append 。 when tail not in SPLITS;
    //      len>510 -> split_big_text (re-split on SPLITS keeping seps,
    //      greedy pack <=510)
    static std::vector<U32> splitSentences(const U32& text, int cutMethod);

private:
    ChineseG2p g2p_;
    void* en_ = nullptr;
    void* g2pwConv_ = nullptr;      // G2PWConverter (owned when g2pw given)
    void* g2pwResolver_ = nullptr;  // G2PWResolver (owned)
    void* polyFix_ = nullptr;       // PolyphoneFixTable (owned when given)  // std::unique_ptr<EnglishG2p>, type-erased to keep
                          // english.h out of this header
};

}  // namespace gsv::textfront
