// test_kern.cpp — A3 内核原语单测: torch golden 对 + NEON/参考实现自洽 + 统计量纪律抽查
#include "test_util.h"

#include "kern/kern.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

#ifndef GSV_KERN_GOLDEN_DIR
#define GSV_KERN_GOLDEN_DIR "../tests/kern_golden"
#endif

struct Bin {
  std::vector<size_t> dims;
  std::vector<float> f32;
  std::vector<uint16_t> f16;
  size_t numel() const {
    size_t n = 1;
    for (size_t d : dims) n *= d;
    return n;
  }
};

Bin load_bin(const std::string& name, bool is_f16) {
  const std::string path = std::string(GSV_KERN_GOLDEN_DIR) + "/" + name +
                           (is_f16 ? ".f16" : ".f32");
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) {
    std::printf("    无法打开 %s\n", path.c_str());
    std::exit(1);
  }
  Bin b;
  uint32_t rank = 0;
  if (std::fread(&rank, 4, 1, f) != 1) std::exit(1);
  b.dims.resize(rank);
  if (rank) {
    std::vector<uint32_t> d32(rank);
    if (std::fread(d32.data(), 4, rank, f) != rank) std::exit(1);
    for (size_t i = 0; i < rank; ++i) b.dims[i] = d32[i];
  }
  const size_t n = b.numel();
  if (is_f16) {
    b.f16.resize(n);
    if (n && std::fread(b.f16.data(), 2, n, f) != n) std::exit(1);
  } else {
    b.f32.resize(n);
    if (n && std::fread(b.f32.data(), 4, n, f) != n) std::exit(1);
  }
  std::fclose(f);
  return b;
}

// 相对误差(以 golden 最大绝对值为归一化)与 cos 相似度 —— G1 同款度量
void check_close(const std::vector<float>& got, const Bin& want, double max_rel, double min_cos,
                 const char* tag) {
  CHECK_EQ(got.size(), want.numel());
  const size_t n = got.size();
  if (n != want.numel()) return;
  double dot = 0, ng = 0, nw = 0, peak = 0, worst = 0;
  for (size_t i = 0; i < n; ++i) {
    const double g = got[i], w = want.f32[i];
    dot += g * w;
    ng += g * g;
    nw += w * w;
    peak = peak < (w < 0 ? -w : w) ? (w < 0 ? -w : w) : peak;
    const double e = (g - w < 0 ? -(g - w) : g - w) / peak;
    if (e > worst) worst = e;
  }
  const double cos = dot / std::sqrt(ng * nw);
  char buf[160];
  std::snprintf(buf, sizeof buf, "%s: max_rel=%.3e cos=%.9f", tag, worst, cos);
  CHECK_MSG(worst <= max_rel, buf);
  CHECK_MSG(cos >= min_cos, buf);
}

// 确定性 xorshift 随机数(C++ 侧自洽测试用, 不依赖 golden)
struct Rng {
  uint64_t s = 0x9e3779b97f4a7c15ull;
  float next() {  // [-1,1)
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return static_cast<float>(static_cast<int64_t>(s >> 40) / static_cast<double>(1ull << 23)) -
           1.0f;
  }
};

}  // namespace

GSV_TEST(gemv_matches_torch_golden) {
  for (const char* tag : {"gemv_a", "gemv_b"}) {
    const Bin w = load_bin(std::string(tag) + "_w", true);
    const Bin x = load_bin(std::string(tag) + "_x", false);
    const Bin y = load_bin(std::string(tag) + "_y", false);
    std::vector<float> got(y.numel());
    // dims 在文件里: w=[out,in]
    gsv::kern::gemv_f16w_f32acc(w.f16.data(), x.f32.data(), got.data(), w.dims[0], w.dims[1]);
    check_close(got, y, 2e-5, 1.0 - 1e-12, tag);

    std::vector<float> got_ref(y.numel());
    gsv::kern::gemv_f16w_f32acc_ref(w.f16.data(), x.f32.data(), got_ref.data(), w.dims[0],
                                    w.dims[1]);
    check_close(got_ref, y, 2e-5, 1.0 - 1e-12, (std::string(tag) + "_ref").c_str());

    // NEON 版 vs 参考版仅归约顺序不同(双 4-lane 累加 vs 标量顺序):
    // 要求相对行峰值一致到 fp32 舍入噪声级, 不要求逐位
    float peak = 0;
    for (size_t i = 0; i < y.numel(); ++i) peak = std::max(peak, std::fabs(got_ref[i]));
    size_t bad = 0;
    for (size_t i = 0; i < y.numel(); ++i)
      if (std::fabs(got[i] - got_ref[i]) > 1e-5f * (peak < 1 ? 1 : peak)) ++bad;
    CHECK_MSG(bad == 0, "NEON 与参考实现在容差内不一致");
  }
}

GSV_TEST(gemv_neon_vs_ref_odd_shapes) {
  Rng rng;
  for (auto [out_, in_] : {std::pair{17u, 131u}, {3u, 8u}, {1u, 1u}, {64u, 7u}}) {
    std::vector<uint16_t> w(out_ * in_);
    std::vector<float> x(in_), y1(out_), y2(out_);
    for (auto& v : w) {
      _Float16 h(rng.next() * 0.3f);
      __builtin_memcpy(&v, &h, 2);
    }
    for (auto& v : x) v = rng.next();
    gsv::kern::gemv_f16w_f32acc(w.data(), x.data(), y1.data(), out_, in_);
    gsv::kern::gemv_f16w_f32acc_ref(w.data(), x.data(), y2.data(), out_, in_);
    size_t mismatch = 0;
    for (size_t i = 0; i < out_; ++i)
      if (std::fabs(y1[i] - y2[i]) > 1e-5f) ++mismatch;
    CHECK_EQ(mismatch, size_t{0});
  }
}

GSV_TEST(rmsnorm_matches_golden) {
  const Bin x = load_bin("rms_x", false), g = load_bin("rms_g", false), y = load_bin("rms_y", false);
  const size_t rows = x.dims[0], n = x.dims[1];
  std::vector<float> got(x.numel());
  for (size_t r = 0; r < rows; ++r)
    gsv::kern::rmsnorm(x.f32.data() + r * n, g.f32.data(), got.data() + r * n, n, 1e-5);
  check_close(got, y, 1e-6, 1.0 - 1e-12, "rmsnorm");
}

GSV_TEST(layernorm_matches_golden) {
  const Bin x = load_bin("ln_x", false), g = load_bin("ln_g", false), b = load_bin("ln_b", false),
            y = load_bin("ln_y", false);
  const size_t rows = x.dims[0], n = x.dims[1];
  std::vector<float> got(x.numel());
  for (size_t r = 0; r < rows; ++r)
    gsv::kern::layernorm(x.f32.data() + r * n, g.f32.data(), b.f32.data(), got.data() + r * n, n,
                         1e-5);
  check_close(got, y, 1e-5, 1.0 - 1e-12, "layernorm");
}

GSV_TEST(softmax_matches_golden_and_sums_to_one) {
  const Bin x = load_bin("sm_x", false), y = load_bin("sm_y", false);
  const size_t rows = x.dims[0], n = x.dims[1];
  std::vector<float> got(x.numel());
  gsv::kern::softmax_rows(x.f32.data(), got.data(), rows, n);
  check_close(got, y, 1e-6, 1.0 - 1e-12, "softmax");
  for (size_t r = 0; r < rows; ++r) {
    double s = 0;
    for (size_t i = 0; i < n; ++i) s += got[r * n + i];
    CHECK_NEAR(s, 1.0, 1e-6);
  }
}

GSV_TEST(activations_match_golden) {
  const Bin v = load_bin("act_v", false), si = load_bin("act_silu", false),
            re = load_bin("act_relu", false);
  std::vector<float> got(v.numel());
  gsv::kern::silu(v.f32.data(), got.data(), v.numel());
  check_close(got, si, 1e-6, 1.0 - 1e-12, "silu");
  gsv::kern::relu(v.f32.data(), got.data(), v.numel());
  check_close(got, re, 0.0, 1.0, "relu");  // relu 应逐位精确
}

GSV_TEST(rope_matches_both_styles) {
  const Bin q = load_bin("rope_in", false), gj = load_bin("rope_gptj_out", false),
            nx = load_bin("rope_neox_out", false);
  const size_t seq = q.dims[0], dim = q.dims[1];
  {
    std::vector<float> io = q.f32;
    gsv::kern::rope_prefill(io.data(), seq, 4, 32, 5, gsv::kern::RopeStyle::GptJ);
    check_close(io, gj, 1e-6, 1.0 - 1e-12, "rope_prefill gptj");
  }
  {
    std::vector<float> io = q.f32;
    gsv::kern::rope_prefill(io.data(), seq, 4, 32, 5, gsv::kern::RopeStyle::NeoX);
    check_close(io, nx, 1e-6, 1.0 - 1e-12, "rope_prefill neox");
  }
  {  // decode 单 token 与 C++ prefill 对应位置必须逐位一致(同一实现同一路径)
    std::vector<float> whole = q.f32;
    gsv::kern::rope_prefill(whole.data(), seq, 4, 32, 5, gsv::kern::RopeStyle::GptJ);
    std::vector<float> io(dim);
    const size_t t = 7;  // pos_base=5 → 位置 12
    __builtin_memcpy(io.data(), q.f32.data() + t * dim, dim * 4);
    gsv::kern::rope_decode(io.data(), 4, 32, 5 + t, gsv::kern::RopeStyle::GptJ);
    size_t mismatch = 0;
    for (size_t i = 0; i < dim; ++i)
      if (io[i] != whole[t * dim + i]) ++mismatch;
    CHECK_EQ(mismatch, size_t{0});
    Bin gj_row;
    gj_row.dims = {dim};
    gj_row.f32.assign(gj.f32.begin() + static_cast<long>(t * dim),
                      gj.f32.begin() + static_cast<long>((t + 1) * dim));
    check_close(io, gj_row, 1e-6, 1.0 - 1e-12, "rope_decode gptj(vs golden)");
  }
}

GSV_TEST_MAIN()
