// wav.cpp — WAV 只读解析实现
//
// 解析纪律: 严格按 RIFF chunk 走(长度奇数字节补齐), 只认 WAVE;
// fmt 兼容 PCM/float/EXTENSIBLE(校验 SubFormat GUID 前 2 字节);
// 数据一律转 mono float32: u8/s16/s24 除以满幅, s32/float 直接归一。
#include "runtime/wav.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace gsv::rt::wav {

namespace {

[[noreturn]] void fail(const std::string& path, const char* why) {
  throw std::runtime_error("wav[" + path + "]: " + why);
}

uint16_t rd_u16(const uint8_t* p) {
  return static_cast<uint16_t>(p[0] | (p[1] << 8));
}
uint32_t rd_u32(const uint8_t* p) {
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

std::vector<uint8_t> read_all(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) fail(path, "打不开文件");
  std::fseek(f, 0, SEEK_END);
  const long n = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  if (n < 12) fail(path, "文件过短");
  std::vector<uint8_t> buf(static_cast<size_t>(n));
  if (std::fread(buf.data(), 1, buf.size(), f) != buf.size()) fail(path, "读失败");
  std::fclose(f);
  return buf;
}

}  // namespace

WavFile load_wav(const std::string& path) {
  const std::vector<uint8_t> buf = read_all(path);
  size_t off = 0;
  if (std::memcmp(buf.data(), "RIFF", 4) != 0) fail(path, "缺 RIFF 魔数");
  if (std::memcmp(buf.data() + 8, "WAVE", 4) != 0) fail(path, "非 WAVE 类型");
  off = 12;

  // ---- 扫 chunk: 先找 fmt ----
  uint16_t audio_format = 0, num_channels = 0, bits = 0;
  uint32_t sample_rate = 0;
  bool have_fmt = false;
  size_t data_off = 0, data_len = 0;

  while (off + 8 <= buf.size()) {
    const uint8_t* ch = buf.data() + off;
    const uint32_t sz = rd_u32(ch + 4);
    const size_t body = off + 8;
    if (body + sz > buf.size()) fail(path, "chunk 越界");
    if (std::memcmp(ch, "fmt ", 4) == 0 && !have_fmt) {
      if (sz < 16) fail(path, "fmt 过短");
      audio_format = rd_u16(ch + 8);
      num_channels = rd_u16(ch + 10);
      sample_rate = rd_u32(ch + 12);
      bits = rd_u16(ch + 22);
      if (audio_format == 0xFFFE) {  // EXTENSIBLE: 校验 SubFormat 前两字节
        if (sz < 40) fail(path, "EXTENSIBLE fmt 过短");
        audio_format = rd_u16(ch + 8 + 24);
      }
      if (audio_format != 1 && audio_format != 3)
        fail(path, "不支持的编码(仅 PCM/float)");
      if (num_channels == 0 || num_channels > 64) fail(path, "声道数非法");
      have_fmt = true;
    } else if (std::memcmp(ch, "data", 4) == 0 && have_fmt) {
      data_off = body;
      data_len = sz;
      break;  // fmt 在前已满足
    }
    off = body + sz + (sz & 1);  // RIFF 字段奇数长度需对齐填充
  }
  if (!have_fmt || data_len == 0) fail(path, "缺 fmt/data chunk");

  const size_t bytes_per_sample = static_cast<size_t>(bits) / 8;
  if (bits % 8 != 0 || bytes_per_sample == 0 || bytes_per_sample > 4)
    fail(path, "位宽不支持");
  if (audio_format == 3 && bits != 32) fail(path, "float 仅支持 32bit");
  if (audio_format == 1 && bits != 8 && bits != 16 && bits != 24 && bits != 32)
    fail(path, "PCM 位宽不支持");

  const size_t frames = data_len / (bytes_per_sample * num_channels);
  if (frames == 0) fail(path, "空音频");

  WavFile w;
  w.sample_rate = sample_rate;
  w.num_channels = num_channels;
  w.samples.resize(frames);

  const uint8_t* p = buf.data() + data_off;
  const float norm16 = 1.0f / 32768.0f;
  for (size_t i = 0; i < frames; ++i) {
    float acc = 0.f;
    for (uint16_t c = 0; c < num_channels; ++c) {
      const uint8_t* s = p + (i * num_channels + c) * bytes_per_sample;
      float v = 0.f;
      switch (bytes_per_sample) {
        case 1: {  // u8 无符号
          v = (static_cast<int>(s[0]) - 128) / 128.0f;
          break;
        }
        case 2: {  // s16
          const int16_t x = static_cast<int16_t>(rd_u16(s));
          v = static_cast<float>(x) * norm16;
          break;
        }
        case 3: {  // s24 小端符号扩展
          int32_t x = static_cast<int32_t>(s[0]) | (static_cast<int32_t>(s[1]) << 8) |
                      (static_cast<int32_t>(s[2]) << 16);
          if (x & 0x800000) x |= ~0xFFFFFF;  // 符号扩展
          v = static_cast<float>(x) / 8388608.0f;
          break;
        }
        case 4: {
          if (audio_format == 3) {  // float32
            std::memcpy(&v, s, 4);
          } else {  // s32
            const int32_t x = static_cast<int32_t>(rd_u32(s));
            v = static_cast<float>(x) / 2147483648.0f;
          }
          break;
        }
        default: fail(path, "位宽不支持");
      }
      acc += v;
    }
    w.samples[i] = acc / static_cast<float>(num_channels);
  }
  return w;
}

}  // namespace gsv::rt::wav
