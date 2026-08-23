// test_gsv_header.cpp — GSV1 头部解析单测（对真实 weights/ar_s1v3.gsv + 构造坏输入）
#include "test_util.h"

#include "runtime/gsv_header.hpp"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string weights_file(const char* name) {
#ifndef GSV_WEIGHTS_DIR
#define GSV_WEIGHTS_DIR "../weights"
#endif
  return std::string(GSV_WEIGHTS_DIR) + "/" + name;
}

}  // namespace

GSV_TEST(header_parses_real_ar_file) {
  const auto h = gsv::rt::load_gsv_raw_header(weights_file("ar_s1v3.gsv").c_str());
  CHECK_EQ(h.version, 1u);
  CHECK_EQ(h.flags, gsv::rt::GSV_FLAG_HAS_F16);
  CHECK_EQ(h.name, std::string("ar_s1v3"));
  CHECK_EQ(h.n_tensors, 295u);
  // config JSON 必含 convert.py 写出的两把钥匙
  CHECK(h.config_json.find("\"meta\"") != std::string::npos);
  CHECK(h.config_json.find("\"model_config\"") != std::string::npos);
}

GSV_TEST(header_rejects_bad_magic) {
  const char bad[] = {'X', 'S', 'V', '1', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  bool threw = false;
  try {
    (void)gsv::rt::parse_gsv_raw_header(bad, sizeof bad);
  } catch (const std::exception&) {
    threw = true;
  }
  CHECK(threw);
}

GSV_TEST(header_rejects_truncated) {
  // 合法 magic+version+flags 后截断
  const uint8_t buf[11] = {'G', 'S', 'V', '1', 1, 0, 0, 0, 1, 0, 0};
  bool threw = false;
  try {
    (void)gsv::rt::parse_gsv_raw_header(buf, sizeof buf);
  } catch (const std::exception&) {
    threw = true;
  }
  CHECK(threw);
}

GSV_TEST_MAIN()
