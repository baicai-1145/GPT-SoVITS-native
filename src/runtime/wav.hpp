// wav.hpp — 最小 WAV(RIFF/WAVE) 只读解析器(运行时公共件, C1 授权新增)
//
// 范围: 解码 PCM(u8/s16/s24/s32) 与 IEEE float32 到 mono float32(-1..1 归一),
// 多声道做等权平均下混。不提供重采样/写盘 —— 16k 转换属上游管线职责。
//
// 用法:
//   gsv::rt::wav::WavFile w = gsv::rt::wav::load_wav("ref.wav");
//   w.samples  // mono float32
//   w.sample_rate, w.num_channels  // 原始文件属性(mono 已下混, channels 为原声道数)
//
// 错误: 一切非法输入抛 std::runtime_error(带路径与原因)。
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace gsv::rt::wav {

struct WavFile {
  uint32_t sample_rate = 0;
  uint16_t num_channels = 0;   // 文件原声道数(samples 已下混为 mono)
  std::vector<float> samples;  // mono, 与文件同长(sample 数)
};

// 解析并解码整个文件。支持: RIFF/WAVE, fmt=PCM(1)/float(3)/EXTENSIBLE(0xFFFE)。
WavFile load_wav(const std::string& path);

}  // namespace gsv::rt::wav
