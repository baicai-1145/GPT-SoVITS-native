// pipeline_audio.hpp — C2: FFmpeg swresample 的精确复刻 (整段缓冲版)。
//
// CPUFast 经 PyAV/swresample 解码并重采样参考音频; HuBERT 码字对波形细节敏感,
// 近似重采样会在量化边界上翻转个别码 (实测 167 码错 4)。本实现按 swresample
// 源码逐式复刻: Kaiser 窗 sinc 滤波器组(beta=9, filter_size=32, cutoff=0.97,
// exact_rational 相位压缩) + 整数步进状态机(dst_incr_div/mod) + 首尾反射。
// 浮点求和顺序与 swr 的 SIMD 内核仍可能有末位差异, 但码字级应当稳定。
//
// 口径来源: libswresample/{resample.c,resample_template.c,options.c} (FFmpeg master)。
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <vector>

namespace gsv::rt::pipeline {

class Resampler {
 public:
  void init(int inRate, int outRate) {
    if (ready_ && inRate_ == inRate && outRate_ == outRate) return;
    inRate_ = inRate;
    outRate_ = outRate;
    ready_ = true;

    constexpr int kFilterSize = 32;
    constexpr double kCutoff = 0.97;
    constexpr double kKaiserBeta = 9.0;      // options.c 默认
    constexpr int kPhaseShift = 10;          // options.c 默认 (1024)
    const double factor =
        std::min(double(outRate_) * kCutoff / double(inRate_), 1.0);

    // exact_rational: 相位数压缩到输入/输出比率的精确分母
    int pe = outRate_, pq = inRate_;
    {
      const uint32_t g = std::gcd(uint32_t(pe), uint32_t(pq));
      pe /= int(g);
      pq /= int(g);
    }
    if (pe <= (1 << kPhaseShift)) {
      phaseCount_ = pe;
    } else {
      phaseCount_ = 1 << kPhaseShift;
    }

    // filter_length = align2(max(ceil(filter_size/factor),1))
    int fl = int(std::ceil(double(kFilterSize) / factor));
    if (fl < 1) fl = 1;
    if (fl % 2) ++fl;
    filterLen_ = fl;
    filterAlloc_ = (fl + 7) & ~7;  // FFALIGN(fl, 8)

    buildFilter(factor, kFilterSize, kKaiserBeta);

    // 步进: av_reduce(out_rate, in_rate*phase_count) 后倍增至任一达 2^20
    {
      int64_t sr, dr;
      reduceRatio(int64_t(outRate_), int64_t(inRate_) * phaseCount_, sr, dr);
      while (dr < (int64_t(1) << 20) && sr < (int64_t(1) << 20)) {
        dr *= 2;
        sr *= 2;
      }
      srcIncr_ = sr;
      dstIncr_ = dr;
    }
    dstIncrDiv_ = dstIncr_ / srcIncr_;
    dstIncrMod_ = dstIncr_ % srcIncr_;
    index_ = -int64_t(phaseCount_) * ((filterLen_ - 1) / 2);
    frac_ = 0;
  }

  // 整段处理 (含首尾反射, 对齐 swr 单流语义)。返回输出样本数。
  size_t process(const float* in, size_t nIn, std::vector<float>& out) {
    if (!ready_ || inRate_ == outRate_) {
      out.assign(in, in + nIn);
      return nIn;
    }
    // 缓冲 = [首反射 | 输入 | 尾反射]
    const size_t hist = size_t(filterLen_);
    // 尾反射长度 (resample_flush): reflection=(min(count,fl)+1)/2
    const size_t refl = (std::min(nIn, size_t(filterLen_)) + 1) / 2;
    buf_.assign(hist + nIn + refl, 0.f);
    // invert_initial_buffer: buffer[fl+n]=src[n]; 再镜像填历史 n=1..fl:
    //   buf[fl-n] = buf[fl+n]
    std::memcpy(buf_.data() + hist, in, nIn * sizeof(float));
    for (size_t nn = 1; nn <= hist && hist + nn <= nIn; ++nn)
      buf_[hist - nn] = buf_[hist + nn];
    // 尾反射 (resample_flush): 镜像末尾样本
    bufEnd_ = hist + nIn;
    for (size_t j = 0; j < refl; ++j)
      buf_[bufEnd_ + j] = buf_[bufEnd_ - 1 - j];
    bufEnd_ += refl;

    out.clear();
    out.reserve(size_t((uint64_t(nIn) * outRate_) / inRate_) + 16);
    int64_t idx = index_;  // 局部状态 (单次调用内推进; 多次调用不跨保持)
    int64_t fr = frac_;
    int64_t sampleIndex = 0;
    // invert_initial_buffer 收尾: 负索引折算为读起点前移
    int64_t pos0 = int64_t(hist);
    while (idx < 0) {
      --pos0;
      idx += phaseCount_;
    }
    const int64_t avail = int64_t(bufEnd_) - pos0;
    // multiple_resample: 先按公式一次性算出本批输出数
    const int64_t endIndex = (1 + avail - filterLen_) * phaseCount_;
    const int64_t deltaFrac = (endIndex - idx) * srcIncr_ - fr;
    int64_t deltaN = (deltaFrac + dstIncr_ - 1) / dstIncr_;  // ceil
    if (deltaN < 0) deltaN = 0;
    for (int64_t di = 0; di < deltaN; ++di) {
      const float* bank = bank_.data() + size_t(idx) * filterAlloc_;
      double val = 0.0;
      if (fr == 0 && dstIncrMod_ == 0) {  // common 路径
        for (int i = 0; i < filterLen_; ++i)
          val += double(buf_[size_t(pos0 + sampleIndex + i)] * bank[i]);
      } else {  // linear 路径 (相位插值)
        double v2 = 0.0;
        for (int i = 0; i < filterLen_; ++i) {
          val += double(buf_[size_t(pos0 + sampleIndex + i)] * bank[i]);
          v2 += double(buf_[size_t(pos0 + sampleIndex + i)] *
                       bank[i + filterAlloc_]);
        }
        val += (v2 - val) * (double(fr) / double(srcIncr_));
      }
      out.push_back(float(val));

      fr += dstIncrMod_;
      idx += dstIncrDiv_;
      if (fr >= srcIncr_) {
        fr -= srcIncr_;
        ++idx;
      }
      while (idx >= phaseCount_) {
        ++sampleIndex;
        idx -= phaseCount_;
      }
    }
    return out.size();
  }

 private:
  static void reduceRatio(int64_t a, int64_t b, int64_t& na, int64_t& nb) {
    // av_reduce 无精度损失版 (本场景 b>a 且均为整数比)
    const int64_t g = std::gcd(a, b);
    na = a / g;
    nb = b / g;
  }

  // build_filter 的 FLTP 分支复刻 (scale=1<<filter_shift=1, 归一化 ph=0 行)
  void buildFilter(double factor, int tapCountSrc, double beta) {
    const int pc = phaseCount_;
    const int phNb = (pc % 2) ? pc : pc / 2 + 1;
    const int center = (filterLen_ - 1) / 2;
    bank_.assign(size_t(filterAlloc_) * (pc + 2), 0.f);
    double norm = 0.0;
    std::vector<double> tab(filterLen_ + 1);
    for (int ph = 0; ph < phNb; ++ph) {
      double s = 0.0;
      if (factor == 1.0)
        s = std::sin(M_PI * double(ph) / pc) * ((center & 1) ? 1 : -1);
      for (int i = 0; i < filterLen_; ++i) {
        const double x =
            M_PI * (double(i - center) - double(ph) / double(pc)) * factor;
        double y;
        if (x == 0.0)
          y = 1.0;
        else if (factor == 1.0)
          y = s / x;
        else
          y = std::sin(x) / x;
        const double w = 2.0 * x / (factor * filterLen_ * M_PI);
        y *= besselI0(beta * std::sqrt(std::max(1.0 - w * w, 0.0)));
        tab[i] = y;
        s = -s;
        if (ph == 0) norm += y;
      }
      float* row = bank_.data() + size_t(ph) * filterAlloc_;
      for (int i = 0; i < filterLen_; ++i)
        row[i] = float(tab[i] / norm);
      if (pc % 2) break;  // 奇相位无镜像填充
      float* mir = bank_.data() + size_t(pc - ph) * filterAlloc_;
      for (int i = 0; i < filterLen_; ++i)
        mir[filterLen_ - 1 - i] = row[i];
    }
    // 尾部行 (linear 插值越界读): 复刻 resample_init 的两次 memcpy
    float* base = bank_.data();
    std::memcpy(base + size_t(pc + 1) * filterAlloc_, base,
                (filterAlloc_ - 1) * sizeof(float));
    std::memcpy(base + size_t(pc) * filterAlloc_,
                base + (filterAlloc_ - 1) * sizeof(float), sizeof(float));
    (void)tapCountSrc;
  }

  static double besselI0(double x) {  // av_bessel_i0 同款级数
    double s = 1.0, term = 1.0;
    const double hx = x * 0.5;
    for (int k = 1; k < 50; ++k) {
      term *= (hx / k) * (hx / k);
      s += term;
      if (term < 1e-18 * s) break;
    }
    return s;
  }

  bool ready_ = false;
  int inRate_ = 0, outRate_ = 0;
  int phaseCount_ = 0;
  int filterLen_ = 0, filterAlloc_ = 0;
  int64_t srcIncr_ = 1, dstIncr_ = 1, dstIncrDiv_ = 0, dstIncrMod_ = 0;
  int64_t index_ = 0, frac_ = 0;
  std::vector<float> bank_;
  std::vector<float> buf_;
  size_t bufEnd_ = 0;
};

}  // namespace gsv::rt::pipeline
