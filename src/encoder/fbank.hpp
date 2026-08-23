// fbank.hpp — Kaldi 风格 fbank 前端(SV eres2net 输入), 对照 torchaudio.compliance.kaldi.fbank
//
// 固定参数(CPUFast sv.py 调用点): num_mel_bins=80, sample_frequency=16000, dither=0,
// 其余取 torchaudio 默认: povey 窗 / 25ms 窗长(400 点) / 10ms 移位(160 点) /
// snip_edges=true / remove_dc_offset / preemphasis=0.97 / round_to_power_of_two(512) /
// use_power=true / use_log_fbank=true / low_freq=20 / high_freq=0(→Nyquist)。
//
// 数值纪律: FFT/窗/统计全 double 计算后落 fp32 —— 与 torch float32 路径的差异
// 远小于 SV 主干自身的锚点噪声带(见 tests/golden/CALIBRATION.md B12/C1 节)。
#pragma once

#include <cstddef>
#include <vector>

namespace gsv::encoder {

// 输出 [frames, 80] 行主(mel 对数能量)。waveform 为 mono float32。
std::vector<float> kaldi_fbank_80(const float* waveform, size_t n, size_t* frames_out);

}  // namespace gsv::encoder
