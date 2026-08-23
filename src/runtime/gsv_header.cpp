// gsv_header.cpp — GSV1 头部字段解析实现（小端、定长前缀）
#include "gsv_header.hpp"

#include <cstring>
#include <stdexcept>
#include <vector>

namespace gsv::rt {

namespace {

constexpr char kMagic[4] = {'G', 'S', 'V', '1'};

uint16_t rd_u16(const uint8_t* p) {
  uint16_t v;
  std::memcpy(&v, p, sizeof v);
  return v;
}

uint32_t rd_u32(const uint8_t* p) {
  uint32_t v;
  std::memcpy(&v, p, sizeof v);
  return v;
}

}  // namespace

GsvRawHeader parse_gsv_raw_header(const void* data, size_t size) {
  const uint8_t* p = static_cast<const uint8_t*>(data);
  if (size < 12 || std::memcmp(p, kMagic, 4) != 0) {
    throw std::runtime_error("gsv: magic 不是 GSV1");
  }
  GsvRawHeader h;
  h.version = rd_u32(p + 4);
  if (h.version != GSV_VERSION) {
    throw std::runtime_error("gsv: 不支持的版本 " + std::to_string(h.version));
  }
  h.flags = rd_u32(p + 8);

  // 之后所有字段经 take() 严格越界检查（相对头部起点 12 字节处）
  const uint8_t* q = p + 12;
  const size_t left = size - 12;
  auto take = [&](size_t n) -> const uint8_t* {
    const size_t consumed = static_cast<size_t>(q - p) - 12;
    if (left - consumed < n) throw std::runtime_error("gsv: 头部被截断");
    const uint8_t* r = q;
    q += n;
    return r;
  };

  const uint16_t name_len = rd_u16(take(2));
  h.name.assign(reinterpret_cast<const char*>(take(name_len)), name_len);

  const uint32_t cfg_len = rd_u32(take(4));
  h.config_json.assign(reinterpret_cast<const char*>(take(cfg_len)), cfg_len);

  h.n_tensors = rd_u32(take(4));
  return h;
}

GsvRawHeader load_gsv_raw_header(const char* path) {
  FILE* f = std::fopen(path, "rb");
  if (!f) throw std::runtime_error(std::string("gsv: 无法打开 ") + path);
  std::vector<uint8_t> buf(4096);  // 头部远小于此（名称+config JSON 均短）
  size_t got = std::fread(buf.data(), 1, buf.size(), f);
  bool err = std::ferror(f) != 0;
  std::fclose(f);
  if (err && got < buf.size()) throw std::runtime_error(std::string("gsv: 读取出错 ") + path);
  return parse_gsv_raw_header(buf.data(), got);
}

}  // namespace gsv::rt
