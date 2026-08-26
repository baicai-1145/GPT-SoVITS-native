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

#include <memory>
#include <string>
#include <vector>

#include "chinese_g2p.h"

namespace gsv::textfront {

using U32 = std::u32string;

class LangSegmenterCpp;
class FastTextLid;

// FE-AUTO-1: 前端语种模式(CPUFast TTS_infer_pack lang 参数对应物)
enum class TextLangMode {
    Auto = 0,   // getTexts(text) 空参口径: 切分后 zh/en 走各自 G2P,
                // ja/ko 片 stderr 告警跳过, punctuation 兕底空格串
    AllZh = 1,  // getTexts(text, "zh") 口径(B10 现行为, golden 位级红线)
};

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
    // FE-AUTO-1: budouxDir 为 auto 模式所需的 budoux 模型目录(ja.json +
    // zh-hans.json); langModelPath 为 fasttext lid.176.bin。两者都提供时
    // 才启用 auto 模式(all_zh 不依赖它们)。
    bool load(const std::string& triePath, const std::string& pinyinPath,
              std::string* err, const std::string& cmudictPath = {},
              const G2pwOptions* g2pw = nullptr,
              const std::string& langModelPath = {},
              const std::string& budouxDir = {});
    ~TextFrontend();

    // Injection seam for B6's G2PW converter. nullptr restores the default
    // pypinyin behaviour. The object must outlive this TextFrontend.
    void setResolver(const PinyinResolver* r) { g2p_.setResolver(r); }


    // Full pipeline. cutMethod: TTS_infer_pack text_segmentation_method id,
    // 0..5 ("cut0".."cut5"); runtime default is "cut1".
    // mode: Auto(默认, FE-AUTO-1) / AllZh(B10 口径, 位级不变)。
    bool process(const std::string& utf8Text, Result* out,
                 int cutMethod = 1,
                 TextLangMode mode = TextLangMode::Auto) const;

    // 语种切分测试口(fixture 对照用): 返回 [lang, text] 片。
    // 所有者语义: getLangPieces 仅在 load(langModel,budoux) 成功后可用;
    // 模式透传 getTexts(default_lang)。不可用时返回空且写 error。
    std::vector<std::pair<std::string, std::string>> getLangPieces(
        const std::string& utf8Text, TextLangMode mode,
        std::string* error) const;

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
    void* polyFix_ = nullptr;       // PolyphoneFixTable (owned when given)
    void* langSeg_ = nullptr;  // LangSegmenterCpp (owned; FE-AUTO-1, 可空)
                               // unique_ptr 会把不完整类型泄露给 pipeline TU
                          // std::unique_ptr<EnglishG2p>, type-erased to keep
                          // english.h out of this header
};

}  // namespace gsv::textfront
