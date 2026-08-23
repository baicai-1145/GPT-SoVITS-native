// fbank.cpp — Kaldi fbank 实现(逐算子对照 eres2net/kaldi.py = torchaudio.compliance.kaldi)
//
// 流程(_get_window → spectrogram → mel):
//   1. 分帧: snip_edges, m = 1+(N-400)/160, 帧 [m,400]
//   2. remove_dc_offset: 每帧减均值
//   3. preemphasis 0.97: x'[j] = x[j] - a·x[max(0,j-1)](replicate-pad 语义,
//      即 j==0 时 x[0]·(1-a))
//   4. povey 窗: hann(periodic=false)^0.85
//   5. 右侧补零到 512 → rfft → |X|² (use_power)
//   6. kaldi mel 滤波器组 80×257(low=20Hz, high=Nyquist), 矩阵积
//   7. log(max(·, FLT_EPSILON))
#include "encoder/fbank.hpp"

#include <cmath>
#include <cstring>

namespace gsv::encoder {

namespace {

constexpr int kWsz = 400;    // 帧长采样数(25ms@16k)
constexpr int kWsh = 160;    // 帧移采样数(10ms@16k)
constexpr int kFftN = 512;   // round_to_power_of_two
constexpr int kMelBins = 80;
constexpr float kEps = 1.1920928955078125e-07f;  // FLT_EPSILON(kaldi _get_epsilon)

// ---- 手写迭代式 radix-2 FFT(double 内部精度), 只做实输入 rfft 的前 N/2+1 个点 ----
void fft_radix2(double* re, double* im, int n) {
  // 位反转置换
  for (int i = 1, j = 0; i < n; ++i) {
    int bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      std::swap(re[i], re[j]);
      std::swap(im[i], im[j]);
    }
  }
  for (int len = 2; len <= n; len <<= 1) {
    const double ang = -2.0 * M_PI / static_cast<double>(len);
    const double wr = std::cos(ang), wi = std::sin(ang);
    for (int i = 0; i < n; i += len) {
      double cr = 1.0, ci = 0.0;
      for (int k = 0; k < len / 2; ++k) {
        const double ur = re[i + k], ui = im[i + k];
        const double vr = re[i + k + len / 2] * cr - im[i + k + len / 2] * ci;
        const double vi = re[i + k + len / 2] * ci + im[i + k + len / 2] * cr;
        re[i + k] = ur + vr;
        im[i + k] = ui + vi;
        re[i + k + len / 2] = ur - vr;
        im[i + k + len / 2] = ui - vi;
        const double ncr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = ncr;
      }
    }
  }
}

// kaldi mel 尺度(非 HTK)
inline double mel_scale(double f) { return 1127.0 * std::log(1.0 + f / 700.0); }

struct MelBanks {
  // banks[bin][fft_bin], 80×256(第 257 列恒 0, 融进乘加时跳过)
  double b[kMelBins][kFftN / 2];

  MelBanks() {
    const double low_freq = 20.0, high_freq = 8000.0;  // high<=0 → Nyquist
    const double fft_bin_width = 16000.0 / static_cast<double>(kFftN);
    const double mel_low = mel_scale(low_freq);
    const double mel_high = mel_scale(high_freq);
    const double delta = (mel_high - mel_low) / (kMelBins + 1);
    for (int bin = 0; bin < kMelBins; ++bin) {
      const double left = mel_low + bin * delta;
      const double center = mel_low + (bin + 1.0) * delta;
      const double right = mel_low + (bin + 2.0) * delta;
      for (int k = 0; k < kFftN / 2; ++k) {
        const double m = mel_scale(fft_bin_width * k);
        const double up = (m - left) / (center - left);
        const double down = (right - m) / (right - center);
        double v = up < down ? up : down;
        if (v < 0.0) v = 0.0;
        b[bin][k] = v;
      }
    }
  }
};

}  // namespace

std::vector<float> kaldi_fbank_80(const float* waveform, size_t n, size_t* frames_out) {
  if (n < static_cast<size_t>(kWsz)) return {};
  const size_t m = 1 + (n - kWsz) / kWsh;

  // povey 窗(hann periodic=false)^0.85
  static double win[kWsz];
  static bool win_init = false;
  if (!win_init) {
    for (int i = 0; i < kWsz; ++i) {
      const double h = 0.5 - 0.5 * std::cos(2.0 * M_PI * i / (kWsz - 1));
      win[i] = std::pow(h, 0.85);
    }
    win_init = true;
  }
  static const MelBanks banks;

  std::vector<float> out(m * kMelBins);
  std::vector<double> frame(kFftN, 0.0), fre(kFftN), fim(kFftN, 0.0);

  for (size_t t = 0; t < m; ++t) {
    const float* src = waveform + t * kWsh;
    double mean = 0.0;
    for (int i = 0; i < kWsz; ++i) mean += src[i];
    mean /= kWsz;
    // preemphasis(j==0 replicate → x0·(1-a)) + dc offset 已减
    for (int i = 0; i < kWsz; ++i) {
      const double x = static_cast<double>(src[i]) - mean;
      const double prev = i == 0 ? x : static_cast<double>(src[i - 1]) - mean;
      frame[i] = (x - 0.97 * prev) * win[i];
    }
    // rfft
    std::memcpy(fre.data(), frame.data(), sizeof(double) * kFftN);
    std::fill(fim.begin(), fim.end(), 0.0);
    fft_radix2(fre.data(), fim.data(), kFftN);
    // power spectrum 前 257 点 → mel(第 257 列补零, 不参与)
    double mel[kMelBins] = {0};
    for (int k = 0; k < kFftN / 2; ++k) {
      const double p = fre[k] * fre[k] + fim[k] * fim[k];
      if (p == 0.0) continue;
      for (int b = 0; b < kMelBins; ++b) mel[b] += p * banks.b[b][k];
    }
    for (int b = 0; b < kMelBins; ++b)
      out[t * kMelBins + b] =
          static_cast<float>(std::log(mel[b] > kEps ? mel[b] : static_cast<double>(kEps)));
  }
  if (frames_out) *frames_out = m;
  return out;
}

}  // namespace gsv::encoder
