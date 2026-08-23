// wordpiece.hpp — B6: HF tokenizers BertNormalizer/BertPreTokenizer/WordPiece 复刻
// 口径: tokenize_and_map(utils.py) 对每个 word 单独 encode(word, add_special_tokens=False)
//   流水线: BertNormalizer(clean_text+handle_chinese_chars+lowercase,strip_accents=None)
//         → BertPreTokenizer(空白切) → WordPiece(## 贪心, max_input_chars_per_word=100)
// normalize 表由 tools/export_g2pw_assets.py 用 HF 库逐码点导出(仅差异项), 保证口径一致。
#pragma once
#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace gsv::textfront {

inline bool utf8_decode(std::string_view s, size_t& i, uint32_t& cp) {
  const unsigned char c0 = static_cast<unsigned char>(s[i]);
  if (c0 < 0x80) {
    cp = c0;
    i += 1;
  } else if ((c0 >> 5) == 0x6 && i + 1 < s.size()) {
    cp = (uint32_t(c0 & 0x1F) << 6) | (s[i + 1] & 0x3F);
    i += 2;
  } else if ((c0 >> 4) == 0xE && i + 2 < s.size()) {
    cp = (uint32_t(c0 & 0x0F) << 12) | (uint32_t(s[i + 1] & 0x3F) << 6) |
         (s[i + 2] & 0x3F);
    i += 3;
  } else if ((c0 >> 3) == 0x1E && i + 3 < s.size()) {
    cp = (uint32_t(c0 & 0x07) << 18) | (uint32_t(s[i + 1] & 0x3F) << 12) |
         (uint32_t(s[i + 2] & 0x3F) << 6) | (s[i + 3] & 0x3F);
    i += 4;
  } else {
    cp = 0xFFFD;
    i += 1;
  }
  return true;
}

inline void utf8_append(uint32_t cp, std::string& out) {
  if (cp < 0x80) {
    out.push_back(char(cp));
  } else if (cp < 0x800) {
    out.push_back(char(0xC0 | (cp >> 6)));
    out.push_back(char(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    out.push_back(char(0xE0 | (cp >> 12)));
    out.push_back(char(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(char(0x80 | (cp & 0x3F)));
  } else {
    out.push_back(char(0xF0 | (cp >> 18)));
    out.push_back(char(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back(char(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(char(0x80 | (cp & 0x3F)));
  }
}

// utils.py::wordize_and_map — 空格跳过 / [a-zA-Z0-9]+ 连续段 / 其余单字符
inline void wordizeAndMap(std::string_view text,
                          std::vector<std::pair<size_t, size_t>>* words_utf8) {
  size_t i = 0;
  while (i < text.size()) {
    const unsigned char c = static_cast<unsigned char>(text[i]);
    if (c == ' ') {
      ++i;
      continue;
    }
    const bool alnum = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                       (c >= '0' && c <= '9');
    if (alnum) {
      const size_t start = i;
      while (i < text.size()) {
        const unsigned char d = static_cast<unsigned char>(text[i]);
        if ((d >= 'a' && d <= 'z') || (d >= 'A' && d <= 'Z') ||
            (d >= '0' && d <= '9'))
          ++i;
        else
          break;
      }
      words_utf8->emplace_back(start, i);
    } else {
      uint32_t cp;
      size_t j = i;
      utf8_decode(text, j, cp);
      words_utf8->emplace_back(i, j);
      i = j;
    }
  }
}

struct WordPieceVocab {
  std::vector<std::string> id2tok;
  std::unordered_map<std::string, int> tok2id;
  int unk_id = -1;

  bool load(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    std::string line;
    int c;
    while ((c = fgetc(f)) != EOF) {
      if (c == '\n') {
        tok2id[line] = int(id2tok.size());
        id2tok.push_back(line);
        line.clear();
      } else if (c != '\r') {
        line.push_back(char(c));
      }
    }
    fclose(f);
    auto it = tok2id.find("[UNK]");
    unk_id = it == tok2id.end() ? -1 : it->second;
    return !id2tok.empty();
  }

  int tokenToId(const std::string& t) const {
    auto it = tok2id.find(t);
    return it == tok2id.end() ? unk_id : it->second;
  }
};

// 单码点归一化表: cp→normalized utf8 (仅差异项; 未命中恒等)
class NormTable {
 public:
  void put(uint32_t cp, std::string s) { m_.emplace(cp, std::move(s)); }
  const std::string* lookup(uint32_t cp) const {
    auto it = m_.find(cp);
    return it == m_.end() ? nullptr : &it->second;
  }
  bool loadBinary(FILE* f, uint32_t count);

 private:
  std::unordered_map<uint32_t, std::string> m_;
};

// BertNormalizer 单 word 归一化: 逐码点查表拼接
inline std::string normalizeWord(std::string_view word, const NormTable& nt) {
  // 快路径: 全 ASCII 小写字母数字 → 恒等
  bool fast = true;
  for (size_t i = 0; i < word.size();) {
    uint32_t cp;
    utf8_decode(word, i, cp);
    if (!nt.lookup(cp)) continue;
    fast = false;
    break;
  }
  if (fast) return std::string(word);
  std::string out;
  for (size_t i = 0; i < word.size();) {
    uint32_t cp;
    utf8_decode(word, i, cp);
    if (const std::string* s = nt.lookup(cp))
      out += *s;
    else
      utf8_append(cp, out);
  }
  return out;
}

// WordPiece greedy 匹配 (输入为一段无空白文本)
inline void wordPieceSegment(std::string_view w, const WordPieceVocab& vocab,
                             std::vector<std::string>* out) {
  constexpr size_t kMaxChars = 100;
  // 字符计数按 unicode 码点
  size_t nchars = 0;
  for (size_t i = 0; i < w.size();) {
    uint32_t cp;
    utf8_decode(w, i, cp);
    ++nchars;
  }
  if (nchars > kMaxChars) {
    out->push_back("[UNK]");
    return;
  }
  const std::string kCont = "##";
  size_t start = 0;
  std::vector<std::string> pieces;
  while (start < w.size()) {
    size_t end = w.size();
    std::string cur;
    int match_id = -1;
    while (start < end) {
      cur.assign(w.substr(start, end - start));
      if (start > 0) cur.insert(0, kCont);
      if (vocab.tok2id.count(cur)) {
        match_id = vocab.tok2id.at(cur);
        break;
      }
      // 回退一个字符
      size_t e = end;
      uint32_t cp;
      --e;
      while (e > start && (static_cast<unsigned char>(w[e]) & 0xC0) == 0x80) --e;
      (void)cp;
      end = e;
    }
    if (match_id < 0) {
      out->push_back("[UNK]");
      return;
    }
    pieces.push_back(cur);
    // 前进到匹配末尾
    size_t adv = cur.size();
    if (start > 0) adv -= kCont.size();
    start += adv;
  }
  for (auto& p : pieces) out->push_back(std::move(p));
}

// utils.py::tokenize_and_map 的 token 序列部分 (word 级 [UNK] 折叠规则同源)
inline void tokenizeAndMapTokens(std::string_view text,
                                 const NormTable& nt,
                                 const WordPieceVocab& vocab,
                                 std::vector<std::string>* tokens,
                                 std::vector<int>* ids) {
  std::vector<std::pair<size_t, size_t>> words;
  wordizeAndMap(text, &words);
  tokens->clear();
  ids->clear();
  for (auto& ws : words) {
    std::string_view w = text.substr(ws.first, ws.second - ws.first);
    const std::string normed = normalizeWord(w, nt);
    // BertPreTokenizer: 空白切分
    std::vector<std::string_view> parts;
    size_t i = 0;
    while (i < normed.size()) {
      while (i < normed.size() &&
             (normed[i] == ' ' || normed[i] == '\t' || normed[i] == '\n' ||
              normed[i] == '\r'))
        ++i;
      size_t s = i;
      while (i < normed.size() &&
             !(normed[i] == ' ' || normed[i] == '\t' || normed[i] == '\n' ||
               normed[i] == '\r'))
        ++i;
      if (i > s) parts.push_back(std::string_view(normed).substr(s, i - s));
    }
    std::vector<std::string> wp;
    for (auto& p : parts) wordPieceSegment(p, vocab, &wp);
    if (wp.empty() || (wp.size() == 1 && wp[0] == "[UNK]")) {
      tokens->push_back("[UNK]");
      ids->push_back(vocab.unk_id);
    } else {
      for (auto& t : wp) {
        tokens->push_back(t);
        ids->push_back(vocab.tokenToId(t));
      }
    }
  }
}

}  // namespace gsv::textfront
