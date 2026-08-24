// polyphone_fix.cpp — see polyphone_fix.h.
//
// Bin layout ("GSVPOLY1", little-endian, produced by
// tools/export_polyphone_overrides.py):
//   magic[8] | u32 nPhrase | nPhrase * entry | u32 nSingle | nSingle * entry
//   entry := u8 cpN | cpN * u32 codepoint | u8 rdN | rdN * (u8 len, bytes)
#include "polyphone_fix.h"

#include <cstdint>
#include <cstring>
#include <fstream>

namespace gsv::textfront {

namespace {

bool readU32(const std::string& b, size_t& off, uint32_t* v) {
    if (off + 4 > b.size()) return false;
    std::memcpy(v, b.data() + off, 4);
    off += 4;
    return true;
}

}  // namespace

bool PolyphoneFixTable::load(const std::string& binPath, std::string* err) {
    std::ifstream f(binPath, std::ios::binary);
    if (!f) {
        if (err) *err = "cannot open " + binPath;
        return false;
    }
    std::string b((std::istreambuf_iterator<char>(f)),
                  std::istreambuf_iterator<char>());
    const std::string magic = "GSVPOLY1";
    if (b.size() < 8 || b.compare(0, 8, magic) != 0) {
        if (err) *err = "bad magic in " + binPath;
        return false;
    }
    phrase_.clear();
    single_.clear();
    size_t off = 8;
    for (int section = 0; section < 2; ++section) {
        uint32_t n = 0;
        if (!readU32(b, off, &n)) {
            if (err) *err = "truncated header in " + binPath;
            return false;
        }
        for (uint32_t e = 0; e < n; ++e) {
            if (off >= b.size()) {
                if (err) *err = "truncated entry in " + binPath;
                return false;
            }
            const uint8_t cpN = static_cast<uint8_t>(b[off++]);
            std::u32string word;
            for (uint8_t c = 0; c < cpN; ++c) {
                uint32_t cp = 0;
                if (!readU32(b, off, &cp)) {
                    if (err) *err = "truncated word in " + binPath;
                    return false;
                }
                word.push_back(static_cast<char32_t>(cp));
            }
            if (off >= b.size()) {
                if (err) *err = "truncated readings in " + binPath;
                return false;
            }
            const uint8_t rdN = static_cast<uint8_t>(b[off++]);
            RdList rds;
            rds.reserve(rdN);
            for (uint8_t r = 0; r < rdN; ++r) {
                if (off >= b.size()) {
                    if (err) *err = "truncated reading in " + binPath;
                    return false;
                }
                const uint8_t len = static_cast<uint8_t>(b[off++]);
                if (off + len > b.size()) {
                    if (err) *err = "truncated reading body in " + binPath;
                    return false;
                }
                rds.emplace_back(b.data() + off, len);
                off += len;
            }
            if (section == 0)
                phrase_[word] = std::move(rds);
            else if (word.size() == 1)
                single_.emplace(static_cast<uint32_t>(word[0]),
                                std::move(rds[0]));
        }
    }
    loaded_ = true;
    return true;
}

void PolyphoneFixTable::apply(const std::u32string& word,
                              std::vector<std::string>* rds) const {
    if (!loaded_ || rds == nullptr) return;
    // 1/2: whole-word replacement (phrase overrides first, then pp_dict
    // multi-char entries — the exporter merged both into `phrase_`)
    auto it = phrase_.find(word);
    if (it != phrase_.end()) {
        if (it->second.size() == rds->size())
            *rds = it->second;  // python indexes pinyins by char position
        return;
    }
    // 3: per-character fallback — replace only table hits
    for (size_t i = 0; i < word.size() && i < rds->size(); ++i) {
        auto s = single_.find(word[i]);
        if (s != single_.end()) (*rds)[i] = s->second;
    }
}

}  // namespace gsv::textfront
