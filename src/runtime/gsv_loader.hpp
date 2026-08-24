// gsv_loader.hpp — GSV1 容器完整读取器（mmap 只读 + 目录解析 + 64B 对齐校验）
//
// 格式唯一权威: tools/convert.py 模块 docstring。
// 用法:
//   gsv::rt::GsvFile f("weights/ar_s1v3.gsv");
//   auto* t = f.tensor("h.layers.0.self_attn.in_proj_weight");
//   const float* w32 = t->data_f32();          // fp32 主拷贝
//   const uint16_t* w16 = t->data_f16_raw();   // fp16 拷贝(可能为空, 先 has_f16())
#pragma once

#include "gsv_header.hpp"
#include "mini_json.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace gsv::rt {

enum class DType : uint16_t { F32 = 1, F16 = 2 };

struct TensorView {
  std::string name;
  std::vector<uint32_t> dims;
  DType src_dtype = DType::F32;
  uint64_t f32_offset = 0;
  uint64_t f32_nbytes = 0;
  uint64_t f16_offset = 0;
  uint64_t f16_nbytes = 0;
  const uint8_t* file_base = nullptr;  // mmap 基址(GsvFile 持有生命周期)

  size_t numel() const;
  bool has_f16() const { return f16_nbytes != 0; }
  const float* data_f32() const {
    return reinterpret_cast<const float*>(file_base + f32_offset);
  }
  // 原始 fp16 位型(_Float16 视图见 kern 层转换); 空段返回 nullptr
  const uint16_t* data_f16_raw() const {
    return has_f16() ? reinterpret_cast<const uint16_t*>(file_base + f16_offset) : nullptr;
  }
};

// 内存上解析目录（GsvFile 内部使用; 单测直接对合成缓冲做负例验证）
// base+header 之后是目录区; 解析结果追加到 out。校验失败抛 std::runtime_error。
void parse_gsv_directory(const uint8_t* base, uint64_t file_size,
                         const GsvRawHeader& header, size_t header_end,
                         std::vector<TensorView>& out);

class GsvFile {
 public:
  explicit GsvFile(std::string path);
  ~GsvFile();
  GsvFile(GsvFile&& o) noexcept;
  GsvFile& operator=(GsvFile&& o) noexcept;
  GsvFile(const GsvFile&) = delete;
  GsvFile& operator=(const GsvFile&) = delete;

  const std::string& path() const { return path_; }
  const GsvRawHeader& header() const { return header_; }
  const json::JValue& config() const { return config_; }
  uint64_t file_size() const { return size_; }
  const std::vector<TensorView>& tensors() const { return tensors_; }
  // 按名查找; 不存在返回 nullptr
  const TensorView* tensor(std::string_view name) const;

  // E7: 异步预读全文件进页缓存 (madvise WILLNEED, 内核后台 IO, 不阻塞)。
  // 在 load 流程最早处对全部权重文件调用, 让磁盘读取与各引擎 CPU 转换重叠。
  void prefetch() const;

 private:
  void open_and_parse();
  void close();

  std::string path_;
  int fd_ = -1;
  void* map_ = nullptr;
  uint64_t size_ = 0;
  GsvRawHeader header_;
  json::JValue config_;
  std::vector<TensorView> tensors_;
};

// E7: 路径版预读 (临时 open+mmap+madvise+munmap, 不解析目录)。用于引擎外
// 的附属大文件 (如 g2pw_bert.gsv 由 textfront 内部才打开) 的提前预热。
void prefetch_file(const std::string& path);

}  // namespace gsv::rt
