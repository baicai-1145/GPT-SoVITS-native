// langsegmenter.hpp — CPUFast LangSegmenter.getTexts(text[, default_lang])
// 的原生育移植(位级口径, fixture: tests/textfront/fixtures/langsegment_auto.json)。
//
// 三层结构(与 python 完全对应):
//   1. BudouxParser   — ja.json/zh-hans.json 决策树分词(split_lang 内部使用)
//   2. LangSplitter   — pre_split/_split/_smart_merge_*/merge_across_* 全链
//   3. LangSegmenter::getTexts — digit 兜底/x 回收(full_cjk)/split_jako/
//      merge_lang/punctuation 伪语种
//
// 语种检测 = fasttext lid.176.bin (fasttext_lid.hpp), 口径:
//   fast_lang_detect(text): detect(k=1) 取首标签小写;
//   possible_detection_list(text): detect(k=5, threshold=0.01) 的全部标签。
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "fasttext_lid.hpp"

namespace gsv::textfront {

class FastTextLid;

struct LangPieceCpp {
    std::string lang;  // zh/ja/ko/en/de/fr/x/digit/punctuation/newline
    std::string text;  // UTF-8
};

class LangSegmenterCpp {
public:
    // modelPath: lid.176.bin。budoux 模型 json 目录(budoux 自带数据,
    // 运行时部署放 data/budoux/)。加载失败写 err 返回 false。
    bool load(const std::string& lidBinPath, const std::string& budouxDir,
              std::string* err);

    // python: getTexts(text, default_lang="")。defaultLang 非空走 all_zh 式
    // 短路全并链(digit 并入 default, 短英文 full_en 路径等);空串=auto。
    std::vector<LangPieceCpp> getTexts(const std::string& utf8Text,
                                       const std::string& defaultLang = {}) const;

    const FastTextLid* detector() const { return &lid_; }

private:
    FastTextLid lid_;
    bool loaded_ = false;
};

}  // namespace gsv::textfront
