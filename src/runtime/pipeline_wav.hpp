// pipeline_wav.hpp — C2: 标准 WAVE (PCM16) 写出。
//
// 注: src/sovits/wav_writer.hpp 的字段宽度有缺陷(audio_fmt/ch/align/bits 按
// 4 字节写出, 非 PCM 标准的 2/2/2/2), 外部解析器无法读取 —— 已上报决策者,
// 待其修正后本文件可移除。在修正前 CLI 使用本实现保证产物可读。
#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace gsv::rt::pipeline {

inline void write_wav_int16(const std::string& path, const float* samples,
                            size_t n, uint32_t sr) {
  FILE* fp = fopen(path.c_str(), "wb");
  if (!fp) throw std::runtime_error("cannot open " + path);
  const uint32_t data_bytes = uint32_t(n * 2);
  const uint16_t pcm_tag = 1, ch = 1, align = 2, bits = 16;
  const uint32_t byte_rate = sr * 2;
  std::vector<int16_t> pcm(n);
  for (size_t i = 0; i < n; ++i) {
    const float v = samples[i] * 32768.f;
    long lv = static_cast<long>(v);  // 截断语义同 numpy astype(int16)
    if (lv > 32767) lv = 32767;
    if (lv < -32768) lv = -32768;
    pcm[i] = static_cast<int16_t>(lv);
  }
  auto w4 = [&](uint32_t v) { fwrite(&v, 4, 1, fp); };
  auto w2 = [&](uint16_t v) { fwrite(&v, 2, 1, fp); };
  fwrite("RIFF", 1, 4, fp);
  w4(36 + data_bytes);
  fwrite("WAVE", 1, 4, fp);
  fwrite("fmt ", 1, 4, fp);
  w4(16);
  w2(pcm_tag);
  w2(ch);
  w4(sr);
  w4(byte_rate);
  w2(align);
  w2(bits);
  fwrite("data", 1, 4, fp);
  w4(data_bytes);
  fwrite(pcm.data(), 2, n, fp);
  fclose(fp);
}

}  // namespace gsv::rt::pipeline
