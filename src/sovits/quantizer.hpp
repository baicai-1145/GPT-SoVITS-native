// quantizer.hpp — ResidualVectorQuantizer decode (仅推理: 码本查表)
// torch 侧: quantizer.decode(codes[n_q,B,T]) → 每层 F.embedding(embed, embed_book)
//           project_in/out 为 Identity (dim==codebook_dim==768); 本模型 n_q=1。
#pragma once

#include "sovits/sovits_types.hpp"

#include <cstdint>
#include <vector>

namespace gsv::sovits {

class Quantizer {
 public:
  size_t codebook_size = 0, dim = 0;
  std::vector<float> embed;  // [codebook_size, dim]

  void load(const rt::GsvFile& f, size_t cb, size_t d) {
    codebook_size = cb;
    dim = d;
    load_tensor_f32(f, "quantizer.vq.layers.0._codebook.embed", embed,
                    {cb, d});
  }

  // codes[T] (int64) → quantized [dim, T] (列 = 各码字向量)
  void decode(const int64_t* codes, size_t T, Tensor2D& y) const {
    y.reset(dim, T);
    for (size_t t = 0; t < T; ++t) {
      const int64_t id = codes[t];
      if (id < 0 || static_cast<size_t>(id) >= codebook_size)
        throw std::runtime_error("code index out of range");
      const float* e = embed.data() + static_cast<size_t>(id) * dim;
      float* yr = y.row(t);
      // y[:, t] = embed[id] — Tensor2D 行主是 [dim, T], 逐行散布
      for (size_t c = 0; c < dim; ++c) y.d[c * T + t] = e[c];
      (void)yr;
    }
  }
};

}  // namespace gsv::sovits
