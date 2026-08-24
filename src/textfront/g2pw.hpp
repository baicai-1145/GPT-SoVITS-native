// g2pw.hpp — B6: G2PW 推理串联 (tokenizer + BERT-base + heads → 每字带调拼音)
// 口径: CPUFast text/g2pw/{base_api,torch_api,dataset}.py + opencc_s2tw.py
//
// enable_non_tradional_chinese=True ⇒ 整句先 simplified_to_traditional_tw(简转繁,
// 词组贪心+单字+TWVariants, 可能变长), poly/mono 判定/窗口/tokenizer 全在繁体域;
// pypinyin 默认读音对 t2s(繁→简, 逐字保长) 域整句调用 — simple_seg → mmseg
// (no_non_phrases, 词组优先) → 单字表; 多音字经 BERT-base(12L×768d eps=1e-5
// mask=-10000) + pos_classifier + 三阶 descriptor mask 权重 → masked argmax;
// 输出 bopomofo+tone 经 bopomofo_convert_dict 转 TONE3 pinyin。
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "bert/bert_io.hpp"
#include "bert/bert_model.hpp"
#include "textfront/wordpiece.hpp"

namespace gsv::textfront {

// ---- utf8 工具 (命名空间级, 两类共用) ----
inline size_t gpUtf8Len(std::string_view s) {
  size_t n = 0;
  for (size_t i = 0; i < s.size();) {
    uint32_t cp;
    utf8_decode(s, i, cp);
    ++n;
  }
  return n;
}
inline std::string gpCpToStr(uint32_t cp) {
  std::string s;
  utf8_append(cp, s);
  return s;
}
template <typename Vec>
inline std::string gpCpsToStr(const Vec& cps, size_t from, size_t to) {
  std::string s;
  for (size_t k = from; k < to; ++k) utf8_append(cps[k], s);
  return s;
}
// pypinyin 汉字判定域 (实测: 无音时补 "5" 的范围)
inline bool gpIsHan(uint32_t cp) {
  return (cp >= 0x3400 && cp <= 0x4DBF) || (cp >= 0x4E00 && cp <= 0x9FFF) ||
         (cp >= 0xF900 && cp <= 0xFAFF);
}

struct G2PWAssets {
  std::vector<std::string> labels;  // 1305 bopomofo+tone
  WordPieceVocab vocab;
  NormTable norm;
  std::unordered_set<uint32_t> poly_chars;
  std::unordered_map<uint32_t, std::string> mono;
  std::unordered_map<uint32_t, size_t> char2id;
  std::vector<uint8_t> masks;  // [n_char × n_labels]
  size_t n_labels = 0, n_chars = 0;
  std::unordered_map<std::string, std::string> bopomofo_conv;
  std::unordered_map<uint32_t, std::string> pinyin_full;  // PINYIN_DICT 全量单字首音 TONE3
  // pypinyin phrases_dict (TONE3)
  std::unordered_map<std::string, std::vector<std::string>> phrases;
  std::unordered_set<std::string> phrase_prefix;
  // t2s / OpenCC s2tw
  std::unordered_map<uint32_t, std::string> t2s;
  std::unordered_map<std::string, std::string> s2t_phrase_map;
  std::unordered_set<std::string> s2t_phrase_prefix;
  size_t s2t_max_phrase_cp = 0;
  std::unordered_map<uint32_t, std::string> s2t_chars, tw_variants;

  bool load(const std::string& bin_path, const std::string& vocab_path);
};

inline bool G2PWAssets::load(const std::string& bin_path,
                             const std::string& vocab_path) {
  FILE* f = fopen(bin_path.c_str(), "rb");
  if (!f) return false;
  auto rd = [&](void* p, size_t n) { return fread(p, 1, n, f) == n; };
  char magic[8];
  if (!rd(magic, 8) || memcmp(magic, "GSVG2PW2", 8) != 0) {
    fclose(f);
    return false;
  }
  auto rstrs = [&](std::vector<std::string>* out) {
    uint32_t n;
    if (!rd(&n, 4)) return false;
    out->clear();
    out->reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
      uint16_t len;
      if (!rd(&len, 2)) return false;
      std::string s(len, '\0');
      if (!rd(s.data(), len)) return false;
      out->push_back(std::move(s));
    }
    return true;
  };

  if (!rstrs(&labels)) goto fail;
  n_labels = labels.size();
  {  // poly chars
    uint32_t n;
    if (!rd(&n, 4)) goto fail;
    poly_chars.reserve(n * 2);
    for (uint32_t i = 0; i < n; ++i) {
      uint16_t len;
      if (!rd(&len, 2)) goto fail;
      std::string s(len, '\0');
      if (!rd(s.data(), len)) goto fail;
      size_t ii = 0;
      uint32_t cp = 0;
      utf8_decode(s, ii, cp);
      poly_chars.insert(cp);
    }
  }
  {  // mono
    uint32_t n;
    if (!rd(&n, 4)) goto fail;
    mono.reserve(n * 2);
    for (uint32_t i = 0; i < n; ++i) {
      uint32_t cp;
      uint16_t vl;
      if (!rd(&cp, 4) || !rd(&vl, 2)) goto fail;
      std::string v(vl, '\0');
      if (!rd(v.data(), vl)) goto fail;
      mono.emplace(cp, std::move(v));
    }
  }
  {  // char2id + masks
    uint32_t n;
    if (!rd(&n, 4)) goto fail;
    n_chars = n;
    char2id.reserve(n * 2);
    for (uint32_t i = 0; i < n; ++i) {
      uint32_t cp, id;
      if (!rd(&cp, 4) || !rd(&id, 4)) goto fail;
      char2id.emplace(cp, id);
    }
    masks.resize(n * n_labels);
    if (!rd(masks.data(), masks.size())) goto fail;
  }
  {  // bopomofo_conv
    uint32_t n;
    if (!rd(&n, 4)) goto fail;
    bopomofo_conv.reserve(n * 2);
    for (uint32_t i = 0; i < n; ++i) {
      uint16_t kl, vl;
      if (!rd(&kl, 2)) goto fail;
      std::string k(kl, '\0');
      if (!rd(k.data(), kl)) goto fail;
      if (!rd(&vl, 2)) goto fail;
      std::string v(vl, '\0');
      if (!rd(v.data(), vl)) goto fail;
      bopomofo_conv.emplace(std::move(k), std::move(v));
    }
  }
  {  // pinyin_full (cp → TONE3)
    uint32_t n;
    if (!rd(&n, 4)) goto fail;
    pinyin_full.reserve(n * 2);
    for (uint32_t i = 0; i < n; ++i) {
      uint32_t cp;
      uint16_t vl;
      if (!rd(&cp, 4) || !rd(&vl, 2)) goto fail;
      std::string v(vl, '\0');
      if (!rd(v.data(), vl)) goto fail;
      pinyin_full.emplace(cp, std::move(v));
    }
  }
  {  // t2s
    uint32_t n;
    if (!rd(&n, 4)) goto fail;
    t2s.reserve(n * 2);
    for (uint32_t i = 0; i < n; ++i) {
      uint16_t kl, vl;
      if (!rd(&kl, 2)) goto fail;
      std::string k(kl, '\0');
      if (!rd(k.data(), kl)) goto fail;
      if (!rd(&vl, 2)) goto fail;
      std::string v(vl, '\0');
      if (!rd(v.data(), vl)) goto fail;
      size_t ii = 0;
      uint32_t cp = 0;
      utf8_decode(k, ii, cp);
      t2s.emplace(cp, std::move(v));
    }
  }
  {  // phrases (TONE3)
    uint32_t n;
    if (!rd(&n, 4)) goto fail;
    phrases.reserve(n * 2);
    for (uint32_t i = 0; i < n; ++i) {
      uint16_t wl, ns;
      if (!rd(&wl, 2)) goto fail;
      std::string w(wl, '\0');
      if (!rd(w.data(), wl)) goto fail;
      if (!rd(&ns, 2)) goto fail;
      std::vector<std::string> syls;
      syls.reserve(ns);
      for (uint16_t j = 0; j < ns; ++j) {
        uint16_t sl;
        if (!rd(&sl, 2)) goto fail;
        std::string sy(sl, '\0');
        if (!rd(sy.data(), sl)) goto fail;
        syls.push_back(std::move(sy));
      }
      for (size_t b = 0; b < w.size();) {
        uint32_t c;
        utf8_decode(w, b, c);
        if (b < w.size()) phrase_prefix.insert(w.substr(0, b));
      }
      phrases.emplace(std::move(w), std::move(syls));
    }
  }
  {  // s2t phrases → map + 真前缀集
    uint32_t n;
    if (!rd(&n, 4)) goto fail;
    s2t_phrase_map.reserve(n * 2);
    for (uint32_t i = 0; i < n; ++i) {
      uint16_t kl, vl;
      if (!rd(&kl, 2)) goto fail;
      std::string k(kl, '\0');
      if (!rd(k.data(), kl)) goto fail;
      if (!rd(&vl, 2)) goto fail;
      std::string v(vl, '\0');
      if (!rd(v.data(), vl)) goto fail;
      const size_t ncp = gpUtf8Len(k);
      if (ncp > s2t_max_phrase_cp) s2t_max_phrase_cp = ncp;
      // python _build_prefixes: range(1, len(key)) — 即除整词外的所有真前缀
      {
        size_t bi = 0;
        size_t cp_i = 0;
        while (bi < k.size()) {
          if (cp_i >= 1) s2t_phrase_prefix.insert(k.substr(0, bi));
          uint32_t c;
          utf8_decode(k, bi, c);
          ++cp_i;
        }
      }
      s2t_phrase_map.emplace(std::move(k), std::move(v));
    }
  }
  {  // s2t chars + tw variants
    uint32_t n;
    if (!rd(&n, 4)) goto fail;
    for (uint32_t i = 0; i < n; ++i) {
      uint32_t cp;
      uint16_t vl;
      if (!rd(&cp, 4) || !rd(&vl, 2)) goto fail;
      std::string v(vl, '\0');
      if (!rd(v.data(), vl)) goto fail;
      s2t_chars.emplace(cp, std::move(v));
    }
    if (!rd(&n, 4)) goto fail;
    for (uint32_t i = 0; i < n; ++i) {
      uint32_t cp;
      uint16_t vl;
      if (!rd(&cp, 4) || !rd(&vl, 2)) goto fail;
      std::string v(vl, '\0');
      if (!rd(v.data(), vl)) goto fail;
      tw_variants.emplace(cp, std::move(v));
    }
  }
  {  // normalize 映射
    uint32_t n;
    if (!rd(&n, 4)) goto fail;
    for (uint32_t i = 0; i < n; ++i) {
      uint32_t cp;
      uint16_t vl;
      if (!rd(&cp, 4) || !rd(&vl, 2)) goto fail;
      std::string v(vl, '\0');
      if (!rd(v.data(), vl)) goto fail;
      norm.put(cp, std::move(v));
    }
  }
  fclose(f);
  return vocab.load(vocab_path);
fail:
  fclose(f);
  return false;
}

// utils.py::tokenize_and_map 完整复刻: tokens + ids + 每 token 字节起点
inline void tokenizeAndMapFull(std::string_view text, const NormTable& nt,
                               const WordPieceVocab& vocab,
                               std::vector<std::string>* tokens,
                               std::vector<int>* ids,
                               std::vector<size_t>* tok_byte_start) {
  std::vector<std::pair<size_t, size_t>> words;
  wordizeAndMap(text, &words);
  tokens->clear();
  ids->clear();
  tok_byte_start->clear();
  for (auto& ws : words) {
    const size_t word_start = ws.first;
    std::string_view w = text.substr(ws.first, ws.second - ws.first);
    const std::string normed = normalizeWord(w, nt);
    std::vector<std::string> parts;
    size_t i = 0;
    while (i < normed.size()) {
      while (i < normed.size() && (normed[i] == ' ' || normed[i] == '\t' ||
                                   normed[i] == '\n' || normed[i] == '\r'))
        ++i;
      size_t st = i;
      while (i < normed.size() &&
             !(normed[i] == ' ' || normed[i] == '\t' || normed[i] == '\n' ||
               normed[i] == '\r'))
        ++i;
      if (i > st) parts.push_back(normed.substr(st, i - st));
    }
    std::vector<std::string> wp;
    std::vector<std::string> word_tokens;
    for (auto& p : parts) {
      wp.clear();
      wordPieceSegment(p, vocab, &wp);
      for (auto& t : wp) word_tokens.push_back(t);
    }
    if (word_tokens.empty() ||
        (word_tokens.size() == 1 && word_tokens[0] == "[UNK]")) {
      tokens->push_back("[UNK]");
      ids->push_back(vocab.unk_id);
      tok_byte_start->push_back(word_start);
      continue;
    }
    size_t cur = word_start;
    for (auto& t : word_tokens) {
      std::string_view core = t;
      if (core.rfind("##", 0) == 0) core = core.substr(2);
      cur += core.size();  // 归一化后文本内字节长 (中文单字等长场景与码点一致)
      tokens->push_back(t);
      ids->push_back(vocab.tokenToId(t));
      tok_byte_start->push_back(cur - core.size());
    }
  }
}

class G2PWConverter {
 public:
  // fp16 直读视图要求 GsvFile 生命周期覆盖 bert_ —— 由本对象持有(堆上懒开)
  std::unique_ptr<rt::GsvFile> f_;

  bool load(const std::string& gsv_path, const std::string& assets_bin,
            const std::string& vocab_txt, std::string* err) {
    if (!assets_.load(assets_bin, vocab_txt)) {
      *err = "g2pw assets load failed";
      return false;
    }
    f_ = std::make_unique<rt::GsvFile>(gsv_path);
    bert_.cfg.hidden = 768;
    bert_.cfg.heads = 12;
    bert_.cfg.layers = 12;
    bert_.cfg.inter = 3072;
    bert_.cfg.ln_eps = 1e-5f;
    bert_.cfg.mask_neg = -10000.f;
    bert_.load(*f_);
    bert::load_tensor_f32(*f_, "classifier.weight", cls_w_, {1305, 768});
    bert::load_tensor_f32(*f_, "classifier.bias", cls_b_, {1305});
    bert::load_tensor_f32(*f_, "pos_classifier.weight", pos_w_, {11, 768});
    bert::load_tensor_f32(*f_, "pos_classifier.bias", pos_b_, {11});
    bert::load_tensor_f32(*f_, "descriptor_bias.weight", desc_bias_, {1, 1305});
    bert::load_tensor_f32(*f_, "char_descriptor.weight", char_desc_, {3582, 1305});
    bert::load_tensor_f32(*f_, "second_order_descriptor.weight", so_desc_,
                          {39402, 1305});
    return true;
  }

  // 每字符带调拼音 (长度 = s2t 后句长, 与 python __call__ 一致)
  std::vector<std::string> convert(
      const std::string& text_utf8,
      const std::vector<const std::string*>* preset_partial = nullptr) const {
    const std::vector<uint32_t> trad = toTraditional(text_utf8);
    const size_t L = trad.size();
    std::vector<std::string> partial(L, "?");
    // simp 域 (pypinyin defaults; t2s 逐字保长)
    std::vector<uint32_t> simp(L);
    for (size_t i = 0; i < L; ++i) {
      auto it = assets_.t2s.find(trad[i]);
      simp[i] = it == assets_.t2s.end() ? trad[i] : firstCpOf(it->second);
    }
    std::vector<size_t> poly_idx;
    std::vector<bool> need_default(L, false);
    for (size_t i = 0; i < L; ++i) {
      const uint32_t ch = trad[i];
      if (preset_partial && (*preset_partial)[i]) {
        partial[i] = *(*preset_partial)[i];
      } else if (assets_.poly_chars.count(ch)) {
        poly_idx.push_back(i);
      } else {
        auto m = assets_.mono.find(ch);
        if (m != assets_.mono.end())
          partial[i] = styleConvert(m->second);
        else
          need_default[i] = true;
      }
    }
    if (std::any_of(need_default.begin(), need_default.end(),
                    [](bool b) { return b; }))
      fillDefaultPinyins(simp, need_default, partial);
    if (!poly_idx.empty()) {
      constexpr size_t kCtx = 16;
      const size_t left =
          poly_idx.front() > kCtx ? poly_idx.front() - kCtx : 0;
      const size_t right = std::min(L, poly_idx.back() + kCtx + 1);
      std::string window;
      for (size_t k = left; k < right; ++k) utf8_append(trad[k], window);
      for (size_t qi : poly_idx) {
        const int pred = predictChar(window, qi - left);
        if (pred >= 0)
          partial[qi] = styleConvert(assets_.labels[size_t(pred)]);
      }
    }
    return partial;
  }

 private:
  // ---- opencc_s2tw: 词组贪心(probe 升序遇非前缀停) → 单字 → TWVariants ----
  std::vector<uint32_t> toTraditional(const std::string& text) const {
    std::vector<uint32_t> cps;
    {
      size_t i = 0;
      const size_t L = gpUtf8Len(text);
      cps.reserve(L);
      for (size_t k = 0; k < L; ++k) {
        uint32_t cp;
        utf8_decode(text, i, cp);
        cps.push_back(cp);
      }
    }
    auto slice = [&](size_t from, size_t to) {
      return gpCpsToStr(cps, from, to);
    };
    std::vector<uint32_t> stage1;
    size_t pos = 0;
    while (pos < cps.size()) {
      const size_t limit = std::min(cps.size(), pos + assets_.s2t_max_phrase_cp);
      const std::string* matched = nullptr;
      size_t matched_len = 0;
      for (size_t probe = pos + 1; probe <= limit; ++probe) {
        std::string chunk = slice(pos, probe);
        auto it = assets_.s2t_phrase_map.find(chunk);
        if (it != assets_.s2t_phrase_map.end()) {
          matched = &it->second;
          matched_len = probe - pos;
        }
        if (!assets_.s2t_phrase_prefix.count(chunk)) break;
      }
      if (matched) {
        size_t bi = 0;
        while (bi < matched->size()) {
          uint32_t cp;
          utf8_decode(*matched, bi, cp);
          stage1.push_back(cp);
        }
        pos += matched_len;
        continue;
      }
      auto it = assets_.s2t_chars.find(cps[pos]);
      if (it != assets_.s2t_chars.end()) {
        size_t bi = 0;
        while (bi < it->second.size()) {
          uint32_t cp;
          utf8_decode(it->second, bi, cp);
          stage1.push_back(cp);
        }
      } else {
        stage1.push_back(cps[pos]);
      }
      ++pos;
    }
    for (auto& cp : stage1) {
      auto it = assets_.tw_variants.find(cp);
      if (it != assets_.tw_variants.end()) {
        size_t bi = 0;
        uint32_t v = cp;
        utf8_decode(it->second, bi, v);
        cp = v;
      }
    }
    return stage1;
  }

  // pypinyin 整句默认读音复刻 (simple_seg + mmseg + non-Han run 合并错位):
  // 元素序列 = Han run 经 mmseg 展开逐字音 / non-Han maximal run 折叠为单元素(原串);
  // _prepare_data 按 i 逐位取 result[i][0] — 合并处之后整体左移错位, 精确照抄。
  void fillDefaultPinyins(const std::vector<uint32_t>& simp,
                          const std::vector<bool>& need,
                          std::vector<std::string>& partial) const {
    const size_t L = simp.size();
    std::vector<std::string> elems;
    {
      size_t i = 0;
      while (i < L) {
        if (gpIsHan(simp[i])) {
          size_t j = i;
          while (j < L && gpIsHan(simp[j])) ++j;
          // mmseg 正向最大匹配 (no_non_phrases=true)
          size_t remain = i;
          while (remain < j) {
            std::string last_valid;
            size_t last_valid_end = remain;
            bool broke = false;
            for (size_t idx = remain; idx < j; ++idx) {
              std::string word = gpCpsToStr(simp, remain, idx + 1);
              // python 顺序: 先查词组命中, 再查前缀断裂 (整词不在真前缀集里)
              if (assets_.phrases.count(word)) {
                last_valid = word;
                last_valid_end = idx + 1;
              }
              if (!assets_.phrase_prefix.count(word)) {
                broke = true;
                break;
              }
            }
            if (!broke && last_valid.empty() &&
                assets_.phrases.count(gpCpsToStr(simp, remain, j))) {
              last_valid = gpCpsToStr(simp, remain, j);  // python for-else
              last_valid_end = j;
            }
            size_t word_end;
            if (!last_valid.empty()) {
              for (auto& sy : assets_.phrases.at(last_valid))
                elems.push_back(sy);
              word_end = last_valid_end;
            } else {
              elems.push_back(defaultPinyinOf(simp[remain]));
              word_end = remain + 1;
            }
            remain = word_end;
          }
          i = j;
        } else {
          // non-Han maximal run → 单元素原串
          size_t j = i;
          while (j < L && !gpIsHan(simp[j])) ++j;
          elems.push_back(gpCpsToStr(simp, i, j));
          i = j;
        }
      }
    }
    // 逐位回填 need 位
    for (size_t i = 0; i < L; ++i) {
      if (!need[i]) continue;
      if (i < elems.size()) partial[i] = elems[i];
      else break;  // python 会 IndexError — 语料已剔除该类句
    }
  }

  int predictChar(const std::string& window, size_t query_id) const {
    std::vector<std::string> tokens;
    std::vector<int> ids;
    std::vector<size_t> starts;
    tokenizeAndMapFull(window, assets_.norm, assets_.vocab, &tokens, &ids,
                       &starts);
    const size_t WBYTES = window.size();
    if (query_id >= gpUtf8Len(window) || ids.empty()) return -1;
    // query 码点 → 字节偏移 (starts 记录归一化前字节域)
    size_t qbyte = 0;
    {
      size_t i = 0, k = 0;
      while (i < window.size() && k < query_id) {
        uint32_t cp;
        utf8_decode(window, i, cp);
        ++k;
      }
      qbyte = std::min(i, window.size());
    }
    long long tok_pos = -1;
    for (size_t t = 0; t < starts.size(); ++t) {
      const size_t nxt =
          t + 1 < starts.size() ? starts[t + 1] : WBYTES;
      if (qbyte >= starts[t] && qbyte < nxt) {
        tok_pos = static_cast<long long>(t);
        break;
      }
    }
    if (tok_pos < 0) return -1;

    const size_t T = tokens.size() + 2;
    std::vector<int64_t> input_ids(T), tt(T, 0), am(T, 1);
    input_ids[0] = 101;
    for (size_t i = 0; i < ids.size(); ++i) input_ids[i + 1] = int64_t(ids[i]);
    input_ids[T - 1] = 102;

    bert::Matrix h;
    bert_.forward(input_ids, tt, am, h, dmNull());
    float hv[768];
    for (size_t c = 0; c < 768; ++c)
      hv[c] = h.d[(size_t(tok_pos) + 1) * 768 + c];

    // pos_pred = argmax(pos_classifier(h))
    int best_pos = 0;
    {
      float bv = -1e30f;
      for (int o = 0; o < 11; ++o) {
        float acc = pos_b_[o];
        const float* wrow = pos_w_.data() + o * 768;
        for (size_t c = 0; c < 768; ++c) acc += wrow[c] * hv[c];
        if (acc > bv) {
          bv = acc;
          best_pos = o;
        }
      }
    }
    // G2PWModel.forward 口径:
    //   probs = softmax(logits) ⊙ w, w = σ(bias_o+cd[char,o]+so[char·11+pos,o])·phoneme_mask
    //   decision = argmax(exp(logits - max(logits)) · w)
    const uint32_t qcp = codepointAt(window, query_id);
    auto cid = assets_.char2id.find(qcp);
    if (cid == assets_.char2id.end()) return -1;
    const uint8_t* mask_row =
        assets_.masks.data() + cid->second * assets_.n_labels;
    const float* cd_row = char_desc_.data() + cid->second * 1305;
    const float* so_row =
        so_desc_.data() + (cid->second * 11 + size_t(best_pos)) * 1305;
    std::vector<float> logits(assets_.n_labels);
    float lmax = -1e30f;
    for (size_t o = 0; o < assets_.n_labels; ++o) {
      float acc = cls_b_[o];
      const float* wrow = cls_w_.data() + o * 768;
      for (size_t c = 0; c < 768; ++c) acc += wrow[c] * hv[c];
      logits[o] = acc;
      if (acc > lmax) lmax = acc;
    }
    float best_val = -1.0f;  // score = exp(...)·w ∈ [0,1]
    int best_cls = -1;
    for (size_t o = 0; o < assets_.n_labels; ++o) {
      const float w = 1.f / (1.f + std::exp(
                                     -(desc_bias_[o] + cd_row[o] + so_row[o]))) *
                      float(mask_row[o]);
      if (w <= 0.f) continue;
      const float score = std::exp(logits[o] - lmax) * w;
      if (score > best_val) {
        best_val = score;
        best_cls = int(o);
      }
    }
    return best_cls;
  }

  std::string styleConvert(const std::string& bopomofo) const {
    if (bopomofo.empty()) return bopomofo;
    const char tone = bopomofo.back();
    if (tone < '1' || tone > '5') return bopomofo;
    const std::string comp = bopomofo.substr(0, bopomofo.size() - 1);
    auto it = assets_.bopomofo_conv.find(comp);
    if (it == assets_.bopomofo_conv.end()) return bopomofo;
    return it->second + tone;
  }

  // pypinyin single_pinyin 口径: 有音给音; 无音汉字原字+"5"; 非 Han 原样
  std::string defaultPinyinOf(uint32_t ch) const {
    auto it = assets_.pinyin_full.find(ch);
    if (it != assets_.pinyin_full.end()) return it->second;
    if (gpIsHan(ch)) return gpCpToStr(ch) + "5";
    return gpCpToStr(ch);
  }

  static uint32_t codepointAt(std::string_view s, size_t idx) {
    size_t i = 0, k = 0;
    while (i < s.size()) {
      uint32_t cp;
      utf8_decode(s, i, cp);
      if (k == idx) return cp;
      ++k;
    }
    return 0xFFFFFFFFu;
  }
  static uint32_t firstCpOf(const std::string& sv) {
    size_t i = 0;
    uint32_t cp = 0;
    utf8_decode(sv, i, cp);
    return cp;
  }
  static const bert::Dumper& dmNull() {
    static bert::Dumper d("");
    return d;
  }

  G2PWAssets assets_;
  bert::BertModel bert_;
  std::vector<float> cls_w_, cls_b_, pos_w_, pos_b_;
  std::vector<float> desc_bias_, char_desc_, so_desc_;
};

}  // namespace gsv::textfront
