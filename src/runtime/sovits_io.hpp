// sovits_io.hpp — SoVITS 专攻模式: 文本前端+AR 输出的磁盘快照 (E6/E8 迭代加速)
//
//   dump: 全链跑一次后, 把每段 SegSovIn + DecodeCondition(ge/ge_text) 序列化落盘。
//   load: 重放时跳过 textfront/AR/BERT, 直接喂 stageVoc。
//
// 格式 (小端): "GSVSIN1\0" | u32 nSeg | f32 ge[1024] | f32 ge_text[512]
//   每段: u32 codesN i64[] | u32 argmaxN i32[] | u32 phonesN i64[]
//         | u32 noiseN f32[] | u32 textN bytes | u8 hitEos | u8 empty
//
// 用途: SoVITS 优化迭代不再重付 textfront+AR 成本;
//       noise 快照保证重放与原链位级同输入 (RNG 序无关化)。
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "pipeline.hpp"

namespace gsv::rt::pipeline {

inline constexpr char kSovitsInMagic[8] = {'G', 'S', 'V', 'S', 'I', 'N', '1', '\0'};

inline bool writeU32(FILE* f, uint32_t v) { return fwrite(&v, 4, 1, f) == 1; }
inline bool readU32(FILE* f, uint32_t* v) { return fread(v, 4, 1, f) == 1; }

template <typename T>
inline bool writeVec(FILE* f, const std::vector<T>& v) {
  static_assert(std::is_trivially_copyable_v<T>);
  if (!writeU32(f, static_cast<uint32_t>(v.size()))) return false;
  return v.empty() || fwrite(v.data(), sizeof(T), v.size(), f) == v.size();
}
template <typename T>
inline bool readVec(FILE* f, std::vector<T>* v) {
  static_assert(std::is_trivially_copyable_v<T>);
  uint32_t n = 0;
  if (!readU32(f, &n)) return false;
  v->resize(n);
  return n == 0 || fread(v->data(), sizeof(T), n, f) == n;
}

inline bool dumpSovitsIn(const std::string& path, const DecodeCondition& cond,
                         const std::vector<Pipeline::SegSovIn>& segs,
                         std::string* err) {
  FILE* f = fopen(path.c_str(), "wb");
  if (!f) {
    if (err) *err = "dumpSovitsIn: 无法写入 " + path;
    return false;
  }
  bool ok = fwrite(kSovitsInMagic, 8, 1, f) == 1 &&
            writeU32(f, static_cast<uint32_t>(segs.size())) &&
            writeVec(f, cond.ge) && writeVec(f, cond.ge_text);
  for (const auto& s : segs) {
    ok = ok && writeVec(f, s.codes) && writeVec(f, s.rawArgmax) &&
         writeVec(f, s.phonesSeg) && writeVec(f, s.noise);
    uint32_t tn = static_cast<uint32_t>(s.normText.size());
    ok = ok && writeU32(f, tn) &&
         (tn == 0 || fwrite(s.normText.data(), 1, tn, f) == tn) &&
         fwrite(&s.hitEos, 1, 1, f) == 1 && fwrite(&s.empty, 1, 1, f) == 1;
    if (!ok) break;
  }
  fclose(f);
  if (!ok && err) *err = "dumpSovitsIn: 写入失败 " + path;
  return ok;
}

inline bool loadSovitsIn(const std::string& path, DecodeCondition* cond,
                         std::vector<Pipeline::SegSovIn>* segs, std::string* err) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) {
    if (err) *err = "loadSovitsIn: 无法打开 " + path;
    return false;
  }
  char magic[8];
  uint32_t n = 0;
  bool ok = fread(magic, 8, 1, f) == 1 && memcmp(magic, kSovitsInMagic, 8) == 0 &&
            readU32(f, &n) && readVec(f, &cond->ge) && readVec(f, &cond->ge_text);
  if (ok && (cond->ge.size() != 1024 || cond->ge_text.size() != 512)) {
    if (err) *err = "loadSovitsIn: ge/ge_text 维度异常";
    ok = false;
  }
  segs->clear();
  segs->resize(n);
  for (auto& s : *segs) {
    ok = ok && readVec(f, &s.codes) && readVec(f, &s.rawArgmax) &&
         readVec(f, &s.phonesSeg) && readVec(f, &s.noise);
    uint32_t tn = 0;
    ok = ok && readU32(f, &tn);
    s.normText.resize(tn);
    ok = ok && (tn == 0 || fread(s.normText.data(), 1, tn, f) == tn) &&
         fread(&s.hitEos, 1, 1, f) == 1 && fread(&s.empty, 1, 1, f) == 1;
    if (!ok) break;
  }
  fclose(f);
  if (!ok) {
    if (err) *err = "loadSovitsIn: 解析失败或截断 " + path;
    return false;
  }
  return true;
}

}  // namespace gsv::rt::pipeline
