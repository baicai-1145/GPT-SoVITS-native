// ref_cache.cpp — sha256 + 磁盘缓存实现(零第三方依赖)
#include "encoder/ref_cache.hpp"

#include <array>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <stdexcept>

namespace gsv::encoder {

namespace {

constexpr char kMagic[8] = {'G', 'S', 'V', 'R', 'E', 'F', 'B', '1'};
constexpr uint32_t kVersion = 1;
const std::filesystem::path kCacheDir = [] {
  const char* home = std::getenv("HOME");
  return std::filesystem::path(home ? home : "/tmp") / ".cache" / "gsv-native";
}();

// ---- SHA-256(独立实现, FIPS 180-4) ----
struct Sha256 {
  uint32_t h[8];
  uint64_t len = 0;
  std::array<uint8_t, 64> buf{};
  size_t buf_len = 0;

  static constexpr uint32_t K[64] = {
      0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
      0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
      0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
      0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
      0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
      0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
      0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
      0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
      0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
      0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
      0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

  static uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

  explicit Sha256() {
    static const uint32_t iv[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372,
                                   0xa54ff53a, 0x510e527f, 0x9b05688c,
                                   0x1f83d9ab, 0x5be0cd19};
    std::memcpy(h, iv, sizeof(h));
  }

  void block(const uint8_t* p) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i)
      w[i] = (uint32_t(p[4 * i]) << 24) | (uint32_t(p[4 * i + 1]) << 16) |
             (uint32_t(p[4 * i + 2]) << 8) | uint32_t(p[4 * i + 3]);
    for (int i = 16; i < 64; ++i) {
      const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
    uint32_t e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 64; ++i) {
      const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      const uint32_t ch = (e & f) ^ (~e & g);
      const uint32_t t1 = hh + S1 + ch + K[i] + w[i];
      const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      const uint32_t t2 = S0 + maj;
      hh = g; g = f; f = e; e = d + t1;
      d = c; c = b; b = a; a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d;
    h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
  }

  void update(const void* data, size_t n) {
    const auto* p = static_cast<const uint8_t*>(data);
    len += n;
    while (n > 0) {
      if (buf_len == 0 && n >= 64) {
        block(p);
        p += 64;
        n -= 64;
        continue;
      }
      const size_t take = std::min(n, size_t(64) - buf_len);
      std::memcpy(buf.data() + buf_len, p, take);
      buf_len += take;
      p += take;
      n -= take;
      if (buf_len == 64) {
        block(buf.data());
        buf_len = 0;
      }
    }
  }

  std::string final() {
    const uint64_t bitlen = len * 8;
    const uint8_t pad = 0x80;
    update(&pad, 1);
    const uint8_t zero = 0;
    while (buf_len != 56) update(&zero, 1);
    const uint8_t lb[8] = {uint8_t(bitlen >> 56), uint8_t(bitlen >> 48),
                           uint8_t(bitlen >> 40), uint8_t(bitlen >> 32),
                           uint8_t(bitlen >> 24), uint8_t(bitlen >> 16),
                           uint8_t(bitlen >> 8), uint8_t(bitlen)};
    // 直接写长度块(update 会再计一次 len, 但已不再影响输出)
    std::memcpy(buf.data() + 56, lb, 8);
    block(buf.data());
    buf_len = 0;
    std::string out;
    out.reserve(64);
    for (uint32_t v : h)
      for (int sh = 28; sh >= 0; sh -= 4) {
        const uint8_t nib = uint8_t(v >> sh) & 0xF;
        out += char(nib < 10 ? '0' + nib : 'a' + nib - 10);
      }
    return out;
  }
};

std::string sha256_hex(const std::string& s) {
  Sha256 h;
  h.update(s.data(), s.size());
  return h.final();
}

// mtime(sec:nsec) — 文件不存在/取失败返回 "0:0"
std::string mtime_key(const std::string& path) {
  std::error_code ec;
  const auto t = std::filesystem::last_write_time(path, ec);
  if (ec) return "0:0";
#if defined(__APPLE__)
  // file_clock → 系统 ns 的换算没有公开 API; 取 duration 自 epoch 的纳秒数
  // (macOS file_time epoch 与系统时钟不同源, 但同一机器上单调稳定即可作为指纹)
  const auto ns = static_cast<unsigned long long>(t.time_since_epoch().count());
  return std::to_string(ns);
#else
  const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                      t.time_since_epoch())
                      .count();
  return std::to_string(static_cast<unsigned long long>(ns));
#endif
}

void put_u32(std::ofstream& o, uint32_t v) {
  const uint8_t b[4] = {uint8_t(v), uint8_t(v >> 8), uint8_t(v >> 16), uint8_t(v >> 24)};
  o.write(reinterpret_cast<const char*>(b), 4);
}
void put_u64(std::ofstream& o, uint64_t v) {
  const uint8_t b[8] = {uint8_t(v),   uint8_t(v >> 8),  uint8_t(v >> 16),
                        uint8_t(v >> 24), uint8_t(v >> 32), uint8_t(v >> 40),
                        uint8_t(v >> 48), uint8_t(v >> 56)};
  o.write(reinterpret_cast<const char*>(b), 8);
}
bool get_exact(std::ifstream& in, void* dst, size_t n) {
  in.read(static_cast<char*>(dst), std::streamsize(n));
  return in.gcount() == std::streamsize(n);
}
bool get_u32(std::ifstream& in, uint32_t& v) {
  uint8_t b[4];
  if (!get_exact(in, b, 4)) return false;
  v = uint32_t(b[0]) | (uint32_t(b[1]) << 8) | (uint32_t(b[2]) << 16) |
      (uint32_t(b[3]) << 24);
  return true;
}
bool get_u64(std::ifstream& in, uint64_t& v) {
  uint8_t b[8];
  if (!get_exact(in, b, 8)) return false;
  v = 0;
  for (int i = 7; i >= 0; --i) v = (v << 8) | b[i];
  return true;
}

}  // namespace

RefCache::RefCache(std::string tag) : tag_(std::move(tag)) {}

std::string RefCache::cache_key(const std::string& wav_path, const std::string& tag) {
  return sha256_hex("v1|" + wav_path + "|" + mtime_key(wav_path) + "|" + tag);
}

bool RefCache::load(const std::string& wav_path, RefEntry& entry) const {
  const std::filesystem::path fp =
      kCacheDir / ("gsv-ref-" + cache_key(wav_path, tag_) + ".bin");
  std::ifstream in(fp, std::ios::binary);
  if (!in) return false;
  char magic[8];
  if (!get_exact(in, magic, 8)) return false;
  if (std::memcmp(magic, kMagic, 8) != 0) return false;
  uint32_t ver = 0;
  if (!get_u32(in, ver) || ver != kVersion) return false;
  uint64_t hn = 0, sn = 0;
  if (!get_u64(in, hn) || !get_u64(in, sn)) return false;
  entry.hubert.resize(size_t(hn));
  entry.sv.resize(size_t(sn));
  if (!get_exact(in, entry.hubert.data(), size_t(hn) * 4)) return false;
  if (!get_exact(in, entry.sv.data(), size_t(sn) * 4)) return false;
  return true;
}

bool RefCache::store(const std::string& wav_path, const RefEntry& entry) const {
  std::error_code ec;
  std::filesystem::create_directories(kCacheDir, ec);
  const std::filesystem::path fp =
      kCacheDir / ("gsv-ref-" + cache_key(wav_path, tag_) + ".bin");
  std::ofstream o(fp, std::ios::binary | std::ios::trunc);
  if (!o) return false;
  o.write(kMagic, 8);
  put_u32(o, kVersion);
  put_u64(o, entry.hubert.size());
  put_u64(o, entry.sv.size());
  o.write(reinterpret_cast<const char*>(entry.hubert.data()),
          std::streamsize(entry.hubert.size() * 4));
  o.write(reinterpret_cast<const char*>(entry.sv.data()),
          std::streamsize(entry.sv.size() * 4));
  return bool(o);
}

}  // namespace gsv::encoder
