// test_gsv_loader.cpp — GSV1 完整读取器单测: 真实 ar_s1v3.gsv 结构/数值锚点 + 合成坏输入
#include "test_util.h"

#include "runtime/gsv_loader.hpp"
#include "runtime/mini_json.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

#ifndef GSV_WEIGHTS_DIR
#define GSV_WEIGHTS_DIR "../weights"
#endif

std::string weights_file(const char* name) {
  return std::string(GSV_WEIGHTS_DIR) + "/" + name;
}

// fp32 段双精度求和(确定性), 用于与 python 导出的锚点对照
double sum_f64(const gsv::rt::TensorView* t) {
  const float* p = t->data_f32();
  double s = 0;
  for (size_t i = 0; i < t->numel(); ++i) s += static_cast<double>(p[i]);
  return s;
}

}  // namespace

GSV_TEST(loader_opens_real_ar) {
  const gsv::rt::GsvFile f(weights_file("ar_s1v3.gsv").c_str());
  CHECK_EQ(f.header().version, 1u);
  CHECK_EQ(f.header().name, std::string("ar_s1v3"));
  CHECK_EQ(f.header().n_tensors, 295u);
  CHECK_EQ(f.tensors().size(), size_t{295});
}

GSV_TEST(loader_config_json_values) {
  const gsv::rt::GsvFile f(weights_file("ar_s1v3.gsv").c_str());
  const auto* model_cfg = f.config().find("model_config");
  CHECK_MSG(model_cfg != nullptr, "缺 model_config");
  if (!model_cfg) return;
  const auto* m = model_cfg->find("model");
  CHECK_MSG(m != nullptr, "缺 model_config.model");
  if (!m) return;
  CHECK_EQ(m->find("hidden_dim")->as_int(), 512);
  CHECK_EQ(m->find("embedding_dim")->as_int(), 512);
  CHECK_EQ(m->find("head")->as_int(), 16);
  CHECK_EQ(m->find("n_layer")->as_int(), 24);
  CHECK_EQ(m->find("linear_units")->as_int(), 2048);
  CHECK_EQ(m->find("vocab_size")->as_int(), 1025);
  CHECK_EQ(m->find("phoneme_vocab_size")->as_int(), 732);
  CHECK_EQ(m->find("EOS")->as_int(), 1024);
}

GSV_TEST(loader_known_tensor_shapes_and_dtypes) {
  const gsv::rt::GsvFile f(weights_file("ar_s1v3.gsv").c_str());

  struct Expect {
    const char* name;
    std::vector<uint32_t> dims;
    bool want_f16_seg;
  };
  const Expect cases[] = {
      {"bert_proj.weight", {512, 1024}, true},
      {"bert_proj.bias", {512}, false},
      {"ar_text_embedding.word_embeddings.weight", {732, 512}, true},
      {"ar_audio_embedding.word_embeddings.weight", {1025, 512}, true},
      {"ar_text_position.alpha", {1}, false},
      {"h.layers.0.self_attn.in_proj_weight", {1536, 512}, true},
      {"h.layers.0.self_attn.in_proj_bias", {1536}, false},
      {"h.layers.0.self_attn.out_proj.weight", {512, 512}, true},
      {"h.layers.0.linear1.weight", {2048, 512}, true},
      {"h.layers.23.norm2.weight", {512}, false},
  };
  for (const auto& c : cases) {
    const auto* t = f.tensor(c.name);
    CHECK_MSG(t != nullptr, c.name);
    if (!t) continue;
    CHECK(t->dims == c.dims);
    CHECK(t->src_dtype == gsv::rt::DType::F16);  // s1v3 源权重全为 fp16 存储
    CHECK_EQ(t->has_f16(), c.want_f16_seg);
    if (c.want_f16_seg) CHECK(t->data_f16_raw() != nullptr);
    else CHECK(t->data_f16_raw() == nullptr);
    CHECK(t->data_f32() != nullptr);
  }
  CHECK(f.tensor("不存在的张量") == nullptr);
}

GSV_TEST(loader_numeric_anchors) {
  // 锚点由 numpy float64 求和导出; 相对容差 1e-6
  const gsv::rt::GsvFile f(weights_file("ar_s1v3.gsv").c_str());
  {
    const auto* t = f.tensor("bert_proj.bias");
    CHECK_NEAR(sum_f64(t), -712.2830870, 1e-6);
  }
  {
    const auto* t = f.tensor("ar_text_position.alpha");
    CHECK_NEAR(static_cast<double>(t->data_f32()[0]), 3.830078125, 0.0);
  }
  {
    const auto* t = f.tensor("h.layers.23.norm1.weight");
    CHECK_NEAR(sum_f64(t), 1457.921875, 1e-6);
  }
  {
    // fp16 段与 fp32 主拷贝必须逐元素相等(源权重本为 fp16, fp32 即精确升位)
    const auto* t = f.tensor("ar_audio_embedding.word_embeddings.weight");
    const float* a = t->data_f32();
    const uint16_t* hbits = t->data_f16_raw();
    size_t mismatches = 0;
    for (size_t i = 0; i < t->numel(); ++i) {
      _Float16 h;
      __builtin_memcpy(&h, &hbits[i], 2);
      float hf = static_cast<float>(h);
      if (a[i] != hf) ++mismatches;
    }
    CHECK_EQ(mismatches, size_t{0});
    CHECK_NEAR(sum_f64(t), -290416.5079, 0.5);
  }
}

GSV_TEST(loader_rejects_misaligned_directory) {
  // 构造一个最小合法头部 + 一个 f32_offset 未对齐的目录项
  std::vector<uint8_t> buf;
  auto put_u16 = [&](uint16_t v) {
    buf.push_back(uint8_t(v)); buf.push_back(uint8_t(v >> 8));
  };
  auto put_u32 = [&](uint32_t v) {
    for (int i = 0; i < 4; ++i) buf.push_back(uint8_t(v >> (8 * i)));
  };
  auto put_u64 = [&](uint64_t v) {
    for (int i = 0; i < 8; ++i) buf.push_back(uint8_t(v >> (8 * i)));
  };
  auto put_str16 = [&](const char* s) {
    size_t n = std::char_traits<char>::length(s);
    put_u16(uint16_t(n));
    buf.insert(buf.end(), s, s + n);
  };
  buf.insert(buf.end(), {'G', 'S', 'V', '1'});
  put_u32(1);       // version
  put_u32(0);       // flags
  put_str16("t");   // name
  put_u32(2);       // config len
  buf.push_back('{'); buf.push_back('}');
  put_u32(1);       // n_tensors
  put_str16("x");   // tensor name
  put_u16(1); put_u32(1);          // rank=1 dims={1}
  put_u16(1);                      // dtype f32
  put_u64(63);                     // f32_offset 未对齐!
  put_u64(4);                      // f32_nbytes
  put_u64(0);                      // f16_offset
  put_u64(0);                      // f16_nbytes
  buf.resize(buf.size() + 128, 0);

  bool threw = false;
  try {
    gsv::rt::GsvRawHeader h = gsv::rt::parse_gsv_raw_header(buf.data(), buf.size());
    size_t he = 12 + 2 + h.name.size() + 4 + h.config_json.size() + 4;
    std::vector<gsv::rt::TensorView> out;
    gsv::rt::parse_gsv_directory(buf.data(), buf.size(), h, he, out);
  } catch (const std::exception&) {
    threw = true;
  }
  CHECK(threw);
}

GSV_TEST(mini_json_basics) {
  using namespace gsv::rt::json;
  const std::string doc =
      R"({"a":1,"b":-2.5e2,"c":[true,false,null],"d":"he\"llo","e":{"f":3}})";
  const auto v = parse(doc.data(), doc.size());
  CHECK_EQ(v.find("a")->as_int(), 1);
  CHECK_NEAR(v.find("b")->as_double(), -250.0, 1e-12);
  const auto& arr = v.find("c")->arr;
  CHECK_EQ(arr.size(), size_t{3});
  CHECK(arr[0].is(JType::Bool) && arr[0].b);
  CHECK(arr[1].is(JType::Bool) && !arr[1].b);
  CHECK(arr[2].is(JType::Null));
  CHECK_EQ(v.find("d")->as_string(), std::string("he\"llo"));
  CHECK_EQ(v.find("e")->find("f")->as_int(), 3);
  bool threw = false;
  try {
    (void)parse("{\"a\":}", 6);
  } catch (const std::exception&) {
    threw = true;
  }
  CHECK(threw);
}

GSV_TEST_MAIN()
