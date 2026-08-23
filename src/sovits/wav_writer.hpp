// wav_writer.hpp — 16bit PCM WAV 写出 (int16 刻度 = float×32768, 截断语义同 numpy astype)
#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace gsv::sovits {

inline void write_wav_int16(const std::string& path, const float* samples,
                            size_t n, uint32_t sr) {
  FILE* fp = fopen(path.c_str(), "wb");
  if (!fp) throw std::runtime_error("cannot open " + path);
  const uint32_t data_bytes = static_cast<uint32_t>(n * 2);
  const char riff[4] = {'R', 'I', 'F', 'F'};
  const char wave[4] = {'W', 'A', 'V', 'E'};
  const char fmt[4] = {'f', 'm', 't', ' '};
  const char data[4] = {'d', 'a', 't', 'a'};
  // fmt 块字段宽度按 WAV 规范: sz/采样率/字节率为 u32, 格式/声道/对齐/位深为 u16
  const uint32_t fmt_sz = 16, byte_rate = sr * 2;
  const uint16_t audio_fmt = 1, ch = 1, align = 2, bits = 16;
  fwrite(riff, 1, 4, fp);
  const uint32_t riff_sz = 36 + data_bytes;
  fwrite(&riff_sz, 4, 1, fp);
  fwrite(wave, 1, 4, fp);
  fwrite(fmt, 1, 4, fp);
  fwrite(&fmt_sz, 4, 1, fp);
  fwrite(&audio_fmt, 2, 1, fp);
  fwrite(&ch, 2, 1, fp);
  fwrite(&sr, 4, 1, fp);
  fwrite(&byte_rate, 4, 1, fp);
  fwrite(&align, 2, 1, fp);
  fwrite(&bits, 2, 1, fp);
  fwrite(data, 1, 4, fp);
  fwrite(&data_bytes, 4, 1, fp);
  std::vector<int16_t> pcm(n);
  for (size_t i = 0; i < n; ++i) {
    // numpy astype(int16): 向零截断 + wrap; float→int 直接截断
    const float v = samples[i] * 32768.f;
    long lv = static_cast<long>(v);  // truncation toward zero
    if (lv > 32767) lv = 32767;      // 超界饱和 (tanh 输出理论上不会触发)
    if (lv < -32768) lv = -32768;
    pcm[i] = static_cast<int16_t>(lv);
  }
  fwrite(pcm.data(), 2, n, fp);
  fclose(fp);
}

}  // namespace gsv::sovits
