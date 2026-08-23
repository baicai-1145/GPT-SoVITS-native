// ref_cache.hpp — 参考特征缓存(C1): wav → (hubert_hidden, sv_emb) 的磁盘缓存
//
// 缓存键 = sha256("v1|" + abs_path + "|" + mtime(sec:ns) + "|" + tag),
// 文件 ~/.cache/gsv-native/gsv-ref-<key>.bin。
// 格式: "GSVREFB1"(8B) | u32 ver(1) | u64 hubert_n | u64 sv_n |
//       f32 hubert[hubert_n]([1,T,768] 行主) | f32 sv[sv_n]([1,20480])
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace gsv::encoder {

struct RefEntry {
  std::vector<float> hubert;  // [T*768]
  std::vector<float> sv;      // [20480]
};

class RefCache {
 public:
  explicit RefCache(std::string tag);

  // 命中返回 true 并填充 entry(文件缺失/损坏/mtime 不符一律按未命中处理)
  bool load(const std::string& wav_path, RefEntry& entry) const;

  // 写入缓存(mkdir -p); 失败仅静默返回 false —— 缓存永不阻塞主流程
  bool store(const std::string& wav_path, const RefEntry& entry) const;

  static std::string cache_key(const std::string& wav_path, const std::string& tag);

 private:
  std::string tag_;
};

}  // namespace gsv::encoder
