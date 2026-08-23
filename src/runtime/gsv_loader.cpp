// gsv_loader.cpp — mmap 只读加载 + GSV1 目录解析/校验
#include "gsv_loader.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <stdexcept>

namespace gsv::rt {

namespace {

constexpr uint64_t kAlign = 64;
constexpr uint64_t kMaxNumel = 1ull << 40;  // 单张量元素数上限(防目录项畸形)

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

uint64_t rd_u64(const uint8_t* p) {
  uint64_t v;
  std::memcpy(&v, p, sizeof v);
  return v;
}

}  // namespace

size_t TensorView::numel() const {
  size_t n = 1;
  for (uint32_t d : dims) n *= d;
  return n;
}

void parse_gsv_directory(const uint8_t* base, uint64_t file_size,
                         const GsvRawHeader& header, size_t header_end,
                         std::vector<TensorView>& out) {
  out.reserve(out.size() + header.n_tensors);
  const uint8_t* p = base + header_end;
  const uint8_t* const fend = base + file_size;
  bool any_f16 = false;
  for (uint32_t idx = 0; idx < header.n_tensors; ++idx) {
    auto bail = [&](const std::string& why) {
      throw std::runtime_error("gsv: 目录项 #" + std::to_string(idx) +
                               " (" + (idx < out.size() ? out[idx].name : "?") + ") " + why);
    };
    if (fend - p < 2) bail("截断(名称长度)");
    const uint16_t name_len = rd_u16(p);
    p += 2;
    if (static_cast<uint64_t>(fend - p) < name_len) bail("截断(名称)");
    TensorView t;
    t.name.assign(reinterpret_cast<const char*>(p), name_len);
    p += name_len;

    if (fend - p < 2) bail("截断(rank)");
    const uint16_t rank = rd_u16(p);
    p += 2;
    if (rank > 8) bail("rank 异常");
    if (static_cast<uint64_t>(fend - p) < static_cast<uint64_t>(rank) * 4) bail("截断(dims)");
    t.dims.resize(rank);
    for (uint16_t i = 0; i < rank; ++i) t.dims[i] = rd_u32(p + 4u * i);
    p += static_cast<size_t>(rank) * 4;

    if (fend - p < 2 + 32) bail("截断(dtype/offsets)");
    const uint16_t dt = rd_u16(p);
    if (dt != 1 && dt != 2) bail("未知 dtype 枚举");
    t.src_dtype = static_cast<DType>(dt);
    t.f32_offset = rd_u64(p + 2);
    t.f32_nbytes = rd_u64(p + 10);
    t.f16_offset = rd_u64(p + 18);
    t.f16_nbytes = rd_u64(p + 26);
    p += 2 + 32;

    const size_t numel = t.numel();
    if (numel > kMaxNumel) bail("numel 超限");
    if (t.f32_nbytes != numel * 4) bail("fp32 段字节数与形状不符");
    if (t.has_f16() && t.f16_nbytes != numel * 2) bail("fp16 段字节数与形状不符");
    if (t.f32_offset % kAlign != 0 || t.f32_offset + t.f32_nbytes > file_size)
      bail("fp32 段未 64B 对齐或越界");
    if (t.has_f16() && (t.f16_offset % kAlign != 0 || t.f16_offset + t.f16_nbytes > file_size))
      bail("fp16 段未 64B 对齐或越界");
    if (t.has_f16()) any_f16 = true;
    t.file_base = base;
    out.push_back(std::move(t));
  }
  // flags bit0 与实际 fp16 段存在性必须一致
  const bool flag16 = (header.flags & GSV_FLAG_HAS_F16) != 0;
  if (flag16 != any_f16)
    throw std::runtime_error("gsv: flags 的 fp16 位与目录实际内容不一致");
}

GsvFile::GsvFile(std::string path) : path_(std::move(path)) {
  open_and_parse();
}

void GsvFile::open_and_parse() {
  fd_ = ::open(path_.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd_ < 0)
    throw std::runtime_error("gsv: 无法打开 " + path_ + ": " + std::strerror(errno));
  struct stat st{};
  if (::fstat(fd_, &st) != 0 || st.st_size <= 0) {
    close();
    throw std::runtime_error("gsv: fstat 失败或空文件 " + path_);
  }
  size_ = static_cast<uint64_t>(st.st_size);
  void* m = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
  if (m == MAP_FAILED) {
    close();
    throw std::runtime_error("gsv: mmap 失败 " + path_ + ": " + std::strerror(errno));
  }
  map_ = m;
  // 不做 MADV_WILLNEED: 保持零拷贝懒加载启动, 页面由首次访问自然调入

  const uint8_t* base = static_cast<const uint8_t*>(map_);
  header_ = parse_gsv_raw_header(base, size_);

  // 目录起点 = 定长头部结束处（重走一遍头部布局求偏移）
  const uint8_t* q = base + 12;
  q += 2 + rd_u16(q);          // 名称
  q += 4 + rd_u32(q);          // config JSON
  q += 4;                      // n_tensors
  const size_t header_end = static_cast<size_t>(q - base);

  parse_gsv_directory(base, size_, header_, header_end, tensors_);

  try {
    config_ = json::parse(reinterpret_cast<const char*>(base) + 14 + header_.name.size() + 4,
                          header_.config_json.size());
  } catch (const std::exception& e) {
    throw std::runtime_error("gsv: config JSON 解析失败: " + std::string(e.what()));
  }
}

const TensorView* GsvFile::tensor(std::string_view name) const {
  for (const auto& t : tensors_)
    if (t.name == name) return &t;
  return nullptr;
}

void GsvFile::close() {
  if (map_) {
    ::munmap(map_, size_);
    map_ = nullptr;
  }
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
}

GsvFile::~GsvFile() { close(); }

GsvFile::GsvFile(GsvFile&& o) noexcept
    : path_(std::move(o.path_)), fd_(o.fd_), map_(o.map_), size_(o.size_),
      header_(std::move(o.header_)), config_(std::move(o.config_)),
      tensors_(std::move(o.tensors_)) {
  o.fd_ = -1;
  o.map_ = nullptr;
  o.size_ = 0;
}

GsvFile& GsvFile::operator=(GsvFile&& o) noexcept {
  if (this != &o) {
    close();
    path_ = std::move(o.path_);
    fd_ = o.fd_;
    map_ = o.map_;
    size_ = o.size_;
    header_ = std::move(o.header_);
    config_ = std::move(o.config_);
    tensors_ = std::move(o.tensors_);
    o.fd_ = -1;
    o.map_ = nullptr;
    o.size_ = 0;
  }
  return *this;
}

}  // namespace gsv::rt
