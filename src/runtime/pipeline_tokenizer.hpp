// pipeline_tokenizer.hpp — C2: roberta-wwm-ext-large 的 BERT tokenizer 复刻。
//
// 口径 = HuggingFace tokenizers:
//   BertNormalizer{clean_text=T, handle_chinese_chars=T, strip_accents=null,
//                  lowercase=T} → BertPreTokenizer → WordPiece(##, unk=[UNK])
//   → TemplateProcessing "[CLS] A [SEP]"。
//
// 与 torch 侧 load_tokenizer(...)(fast tokenizer)对齐性由 tests 用 B8 fixtures
// 的 input_ids 对照验证。已知简化(记录于 CALIBRATION 由决策者归档):
//   - strip_accents 仅覆盖 Latin-1/Latin-Ext-A 常用预组合字符(本项目语料
//     中文为主, 未命中时退化为原字符);
//   - lowercase 采用 ASCII+Latin 范围小写(CJK 不受影响);
//   - 标点类 = ASCII 标点 + 常用全角/CJK 标点表(tokenizers 用完整 Unicode P 类)。
#pragma once

#include <array>
#include <cstdio>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace gsv::rt::pipeline {

class BertTokenizer {
 public:
  static constexpr int kUnkId = 100, kClsId = 101, kSepId = 102;

  bool load(const std::string& vocabPath, std::string* err) {
    FILE* fp = fopen(vocabPath.c_str(), "rb");
    if (!fp) {
      if (err) *err = "cannot open vocab: " + vocabPath;
      return false;
    }
    std::string line;
    int c;
    id_.clear();
    while ((c = fgetc(fp)) != EOF) {
      if (c == '\n') {
        id_[line] = static_cast<int>(id_.size());
        line.clear();
      } else {
        line.push_back(static_cast<char>(c));
      }
    }
    if (!line.empty()) id_[line] = static_cast<int>(id_.size());
    fclose(fp);
    return true;
  }

  // 返回 [CLS] ids... [SEP](不含 PAD); tt/mask 全 0/1 同长
  void encode(const std::string& utf8, std::vector<int64_t>* ids,
              std::vector<int64_t>* tt, std::vector<int64_t>* mask) const {
    const std::vector<std::string> words = pretokenize(utf8);
    ids->clear();
    tt->clear();
    mask->clear();
    ids->push_back(kClsId);
    for (const auto& w : words) wordpiece(w, ids);
    ids->push_back(kSepId);
    if (tt) tt->assign(ids->size(), 0);
    if (mask) mask->assign(ids->size(), 1);
  }

 private:
  std::unordered_map<std::string, int> id_;

  // ---- BertNormalizer + BertPreTokenizer ----
  // 返回切好的"词"(含标点单字); WordPiece 以词为单位做 greedy 匹配
  static bool isCjk(char32_t c) {
    return (c >= 0x4E00 && c <= 0x9FFF) || (c >= 0x3400 && c <= 0x4DBF) ||
           (c >= 0xF900 && c <= 0xFAFF) || (c >= 0x20000 && c <= 0x2FA1F);
  }
  static bool isAsciiPunct(char32_t c) {
    return (c >= 33 && c <= 47) || (c >= 58 && c <= 64) ||
           (c >= 91 && c <= 96) || (c >= 123 && c <= 126);
  }
  static bool isPunct(char32_t c) {
    if (isAsciiPunct(c)) return true;
    // 常用全角/CJK 标点 (Unicode P 类子集, 覆盖项目语料)
    static const std::array<char32_t, 32> extra = {
        0xFF01u, 0xFF03u, 0xFF05u, 0xFF06u, 0xFF07u, 0xFF08u, 0xFF09u,
        0xFF0Au, 0xFF0Cu, 0x3002u, 0xFF1Au, 0xFF1Bu, 0xFF1Fu, 0x3001u,
        0x00B7u, 0x2026u, 0x2014u, 0xFF5Eu, 0x300Au, 0x300Bu, 0x3008u,
        0x3009u, 0x201Cu, 0x201Du, 0x2018u, 0x2019u, 0x3010u, 0x3011u,
        0x3014u, 0x3015u, 0x300Cu, 0x300Du};
    for (char32_t p : extra)
      if (p == c) return true;
    return false;
  }

  static void decodeCp(const std::u32string& u, std::string* out) {
    for (char32_t c : u) {
      if (c < 0x80) {
        out->push_back(static_cast<char>(c));
      } else if (c < 0x800) {
        out->push_back(static_cast<char>(0xC0 | (c >> 6)));
        out->push_back(static_cast<char>(0x80 | (c & 0x3F)));
      } else if (c < 0x10000) {
        out->push_back(static_cast<char>(0xE0 | (c >> 12)));
        out->push_back(static_cast<char>(0x80 | ((c >> 6) & 0x3F)));
        out->push_back(static_cast<char>(0x80 | (c & 0x3F)));
      } else {
        out->push_back(static_cast<char>(0xF0 | (c >> 18)));
        out->push_back(static_cast<char>(0x80 | ((c >> 12) & 0x3F)));
        out->push_back(static_cast<char>(0x80 | ((c >> 6) & 0x3F)));
        out->push_back(static_cast<char>(0x80 | (c & 0x3F)));
      }
    }
  }

  std::vector<std::string> pretokenize(const std::string& utf8) const {
    // UTF-8 → codepoints
    std::u32string u;
    for (size_t i = 0; i < utf8.size();) {
      unsigned char b = static_cast<unsigned char>(utf8[i]);
      char32_t cp = b;
      size_t n = 1;
      if (b >= 0xF0) {
        cp = b & 0x07;
        n = 4;
      } else if (b >= 0xE0) {
        cp = b & 0x0F;
        n = 3;
      } else if (b >= 0xC0) {
        cp = b & 0x1F;
        n = 2;
      }
      for (size_t k = 1; k < n && i + k < utf8.size(); ++k)
        cp = (cp << 6) | (static_cast<unsigned char>(utf8[i + k]) & 0x3F);
      i += n;
      // clean_text: 控制字符删除, 空白归一为空格
      if (cp == 0x7F || (cp < 0x20 && cp != '\t' && cp != '\n' && cp != '\r'))
        continue;
      if (cp == '\t' || cp == '\n' || cp == '\r' || cp == ' ') {
        u.push_back(U' ');
        continue;
      }
      // handle_chinese_chars: CJK 前后加空格
      if (isCjk(cp)) u.push_back(U' ');
      // strip_accents(Latin 常用预组合) + lowercase
      u.push_back(stripAccent(lower(cp)));
      if (isCjk(cp)) u.push_back(U' ');
    }

    // BertPreTokenizer: 按空格切词, 词内标点再逐字切分
    std::vector<std::string> words;
    std::u32string cur;
    auto flush = [&]() {
      if (!cur.empty()) {
        words.emplace_back();
        decodeCp(cur, &words.back());
        cur.clear();
      }
    };
    for (char32_t c : u) {
      if (c == ' ') {
        flush();
      } else if (isPunct(c)) {
        flush();
        words.emplace_back();
        decodeCp(std::u32string(1, c), &words.back());
      } else {
        cur.push_back(c);
      }
    }
    flush();
    return words;
  }

  static char32_t lower(char32_t c) {
    if (c >= U'A' && c <= U'Z') return c + 32;
    if (c >= 0xC0 && c <= 0xDE && c != 0xD7) return c + 32;  // Latin-1 大写
    return c;
  }
  // 预组合带音符字母 → 基字母 (仅 Latin-1 Supplement + Latin Ext-A 高频区)
  static char32_t stripAccent(char32_t c) {
    static const std::u32string accented =
        U"ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ";
    static const std::u32string bases =
        U"AAAAAAACEEEEIIIIDNOOOOOOUUUUYPBsaaaaaaaceeeeiiiionoooooouuuuypy";
    const size_t n = accented.size() < bases.size() ? accented.size() : bases.size();
    for (size_t i = 0; i < n; ++i)
      if (accented[i] == c) return bases[i];
    return c;
  }

  // WordPiece: greedy longest-match-first; 首段无 ##, 后续段带 ##;
  // 任一位置无匹配或超长 → 整词 [UNK]
  void wordpiece(const std::string& word, std::vector<int64_t>* ids) const {
    const size_t MAX_CHARS = 100;
    std::u32string u;
    for (size_t i = 0; i < word.size();) {
      unsigned char b = static_cast<unsigned char>(word[i]);
      char32_t cp = b;
      size_t n = 1;
      if (b >= 0xF0) {
        cp = b & 0x07;
        n = 4;
      } else if (b >= 0xE0) {
        cp = b & 0x0F;
        n = 3;
      } else if (b >= 0xC0) {
        cp = b & 0x1F;
        n = 2;
      }
      for (size_t k = 1; k < n && i + k < word.size(); ++k)
        cp = (cp << 6) | (static_cast<unsigned char>(word[i + k]) & 0x3F);
      i += n;
      u.push_back(cp);
    }
    if (u.empty() || u.size() > MAX_CHARS) {
      ids->push_back(kUnkId);
      return;
    }
    std::vector<int64_t> pieces;
    size_t pos = 0;
    bool first = true;
    while (pos < u.size()) {
      size_t bestEnd = 0;  // 找最长匹配 (按字符数)
      std::string piece;
      for (size_t end = u.size(); end > pos; --end) {
        piece.clear();
        if (!first) piece += "##";
        decodeCp(u.substr(pos, end - pos), &piece);
        if (id_.count(piece)) {
          bestEnd = end;
          break;
        }
      }
      if (bestEnd == 0) {
        // HF WordPiece: 任一段失败 → 整词 [UNK]
        ids->push_back(kUnkId);
        return;
      }
      pieces.push_back(id_.at(piece));
      pos = bestEnd;
      first = false;
    }
    ids->insert(ids->end(), pieces.begin(), pieces.end());
  }
};

}  // namespace gsv::rt::pipeline
