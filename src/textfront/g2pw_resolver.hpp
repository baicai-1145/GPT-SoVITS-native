// g2pw_resolver.hpp — B6: G2PW 多音字决策的 PolyphoneResolver 形适配器
//
// 对齐 exec-txt `src/textfront/pinyin.h` 的注入缝:
//   struct PinyinSyl { std::string text; bool raw; };
//   class PinyinResolver { virtual void resolve(const char32_t*, size_t,
//                             std::vector<PinyinSyl>*) const; };
// 语义(同 pypinyin): 汉字逐字给带调拼音(raw=false), 非 Han 连续段原样透传
// (raw=true); 数字读法由上游 ToneSandhi 处理。
//
// 与 CPUFast 编排的差异说明(合并时需决策者裁决):
//   CPUFast 的真实链路是句级 —— chinese2.py 以词组拼音构造 preset_partial_results
//   注入 G2PWConverter.convert() 的 preset 参数, G2PW 只回填多音字槽位。
//   PinyinResolver 缝是词级接口; 词内简转繁变长(如 内存→記憶體)时输出按繁体域
//   长度对齐, 调用方按位置取用。句级入口请直接用 G2PWConverter::convert。
#pragma once
#include <string>
#include <unordered_map>
#include <vector>

#include "textfront/g2pw.hpp"
#include "textfront/pinyin.h"

namespace gsv::textfront {

// PinyinSyl now lives in pinyin.h (single definition; the resolver seam
// owns it). The local duplicate is gone.

namespace detail {
inline char32_t kMarkTable[4][6] = {
    {U'ā', U'ō', U'ē', U'ī', U'ū', U'ǖ'},
    {U'á', U'ó', U'é', U'í', U'ú', U'ǘ'},
    {U'ǎ', U'ǒ', U'ě', U'ǐ', U'ǔ', U'ǚ'},
    {U'à', U'ò', U'è', U'ì', U'ù', U'ǜ'},
};
}  // namespace detail

// TONE3 (数字调尾) → pypinyin Style.TONE 带调字母; 调5=轻声去调尾
inline std::string tone3ToMarked(const std::string& s) {
  if (s.size() < 2) return s;
  const char t = s.back();
  if (t < '1' || t > '5') return s;
  std::string body = s.substr(0, s.size() - 1);
  if (t == '5') return body;
  // utf8 展开
  std::vector<uint32_t> cps;
  for (size_t i = 0; i < body.size();) {
    uint32_t cp;
    utf8_decode(body, i, cp);
    cps.push_back(cp);
  }
  // 标调位置优先级: a > o/e > 并列 iu/ui 取后一个 > 其余末元音
  auto idx = [&](uint32_t c) {
    for (size_t k = 0; k < cps.size(); ++k)
      if (cps[k] == c) return int(k);
    return -1;
  };
  int p = idx(U'a');
  if (p < 0) {
    int o = idx(U'o'), e = idx(U'e');
    if (o >= 0 && e >= 0)
      p = o < e ? o : e;
    else if (o >= 0)
      p = o;
    else if (e >= 0)
      p = e;
    else {
      int iv = -1, uv = -1, vv = -1;
      for (int k = int(cps.size()) - 1; k >= 0; --k) {
        if (iv < 0 && cps[k] == U'i') iv = k;
        if (uv < 0 && cps[k] == U'u') uv = k;
        if (vv < 0 && cps[k] == U'v') vv = k;
      }
      if (iv >= 0 && uv >= 0)
        p = iv > uv ? iv : uv;  // iu 标 u, ui 标 i
      else if (iv >= 0)
        p = iv;
      else if (uv >= 0)
        p = uv;
      else if (vv >= 0)
        p = vv;
      else
        return body;
    }
  }
  static const uint32_t base[6] = {U'a', U'o', U'e', U'i', U'u', U'v'};
  int col = -1;
  for (int c = 0; c < 6; ++c)
    if (cps[p] == base[c]) col = c;
  if (col < 0) return body;
  cps[p] = detail::kMarkTable[t - '1'][col];
  std::string out;
  for (uint32_t cp : cps) utf8_append(cp, out);
  return out;
}

// PolyphoneResolver 适配器 — 正式继承 exec-txt 的 PinyinResolver 抽象基类。
// resolve() 保持词级语义(回退路径); resolveSentence() 是 B6 注入的主路径:
// 整句送 convert(), BERT 拿到完整上下文, 多音字决策与 python 句级链路一致
// (实测: 单字"长"=chang2 而 句级"成长"[1]=zhang3, 词级必错)。
class G2PWResolver : public PinyinResolver {
 public:
  // numSrc: the built-in pypinyin resolver providing the isNumeric BMP
  // table. Without it, ToneSandhi's yi/bu/ge quantifier rules silently
  // misfire on Han digits ('一个' would keep ge4 instead of neutralizing
  // to ge5 — found the hard way against pairs s9).
  explicit G2PWResolver(const G2PWConverter* conv,
                        const PinyinResolver* numSrc = nullptr)
      : conv_(conv), numSrc_(numSrc) {}

  bool isNumeric(uint32_t cp) const override {
    return numSrc_ ? numSrc_->isNumeric(cp)
                   : PinyinResolver::isNumeric(cp);
  }

  void resolve(const char32_t* word, size_t len,
               std::vector<PinyinSyl>* syllables) const override {
    *syllables = sylsFromReadings(convertCps(word, len));
  }

  bool resolveSentence(const char32_t* text, size_t len,
                       std::vector<PinyinSyl>* syllables) const override {
    *syllables = sylsFromReadings(convertCps(text, len));
    return true;
  }

  // Raw TONE3 domain for chinese2-style correct_pronunciation: python
  // applies the polyphone dictionaries to g2pw's tone3 outputs BEFORE the
  // marked-pinyin conversion, so the pipeline needs this layer exposed.
  bool resolveSentenceTone3(const char32_t* text, size_t len,
                            std::vector<std::string>* out) const override {
    *out = convertCps(text, len);
    return true;
  }

  const G2PWConverter* conv_;
  const PinyinResolver* numSrc_ = nullptr;

 private:
  // convert() 输出长度 == 输入码点数 (s2t 逐字保长)
  std::vector<std::string> convertCps(const char32_t* word, size_t len) const {
    std::string utf8;
    utf8.reserve(len * 3);
    for (size_t i = 0; i < len; ++i) utf8_append(uint32_t(word[i]), utf8);
    return conv_->convert(utf8);
  }

  static std::vector<PinyinSyl> sylsFromReadings(
      const std::vector<std::string>& py) {
    std::vector<PinyinSyl> out;
    out.reserve(py.size());
    for (const auto& e : py) {
      const bool has_tone = !e.empty() && e.back() >= '1' && e.back() <= '5';
      if (has_tone)
        out.push_back({tone3ToMarked(e), false});
      else
        out.push_back({e, true});
    }
    return out;
  }
};

}  // namespace gsv::textfront
