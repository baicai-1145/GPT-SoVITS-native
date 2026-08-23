// chinese_g2p.h — full native chain of CPUFast text/chinese2.py
// (is_g2pw=False path): text_normalize -> punctuation layer -> g2p ->
// symbols2 ids. Python-raising edge cases (modified_tone index errors,
// punct asserts, opencpop misses) are reported via ok=false.
#pragma once

#include <string>
#include <vector>

#include "jieba.h"
#include "pinyin.h"
#include "zh_norm.h"

namespace gsv::textfront {

struct G2pResult {
    std::vector<int> phones;   // symbols2 ids
    std::vector<int> word2ph;  // one entry per char, phones contributed
    bool ok = true;
    std::string error;  // non-empty when ok=false ("python would raise")
};

class ChineseG2p {
public:
    // Loads jieba trie + pinyin tables. Data paths required.
    bool load(const std::string& triePath, const std::string& pinyinPath,
              std::string* err);

    // Injection seam for B6's G2PW converter: overrides the default
    // pypinyin readings for resolve()/getInitialsFinals()/isNumeric().
    // Pass nullptr to restore the built-in PypinyinResolver. The pointed
    // object must outlive this instance.
    void setResolver(const PinyinResolver* r) { external_ = r; }
    const PinyinResolver& resolver() const {
        return external_ ? *external_ : resolver_;
    }

    // Full pipeline for one input text (UTF-8).
    bool run(const std::string& utf8Text, G2pResult* out) const;

    // Exposed for tests: chinese2.text_normalize + replace_consecutive
    U32 textNormalize(const U32& text) const;

private:
    JiebaSegmenter jb_;
    PypinyinResolver resolver_;
    const PinyinResolver* external_ = nullptr;  // B6 injection seam
    TextNormalizer tn_;
    bool loaded_ = false;
};

}  // namespace gsv::textfront
