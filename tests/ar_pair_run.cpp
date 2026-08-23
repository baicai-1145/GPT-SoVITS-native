// ar_pair_run.cpp — B12 验收 harness: 对 tools/dump_pairs_raw.py 导出的原始对逐个跑引擎
//
// 输入目录布局(每对一个子目录):
//   phones.i64 [T]  prompt.i64 [P]  bert.f32 [T*1024]  meta.txt("T P golden_steps")
// 输出(每对写入 --out 下同名子目录):
//   out_tokens.i32   每步未惩罚 argmax(golden `tokens` 口径)
//   out_sampled.i32  实际生成序列(prompt 后, 不含终止 EOS)
//   out_logits_first8.f32  min(8,steps)*vocab
//   out_logits_last.f32    vocab
//   out_meta.txt     "steps hit_eos prefill_ms decode_ms"
#include "ar/t2s_engine.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace {

template <typename T>
std::vector<T> read_vec(const std::string& path, size_t expect_n) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) {
    std::fprintf(stderr, "打不开 %s\n", path.c_str());
    std::exit(2);
  }
  std::vector<T> v(expect_n);
  if (std::fread(v.data(), sizeof(T), expect_n, f) != expect_n) {
    std::fprintf(stderr, "读短 %s\n", path.c_str());
    std::exit(2);
  }
  std::fclose(f);
  return v;
}

void write_file(const std::string& path, const void* p, size_t bytes) {
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f || std::fwrite(p, 1, bytes, f) != bytes) {
    std::fprintf(stderr, "写失败 %s\n", path.c_str());
    std::exit(2);
  }
  if (f) std::fclose(f);
}

}  // namespace

int main(int argc, char** argv) {
  const char* weights = nullptr;
  const char* pairs_dir = nullptr;
  const char* out_dir = nullptr;
  const char* only = nullptr;
  const char* dump_layers = nullptr;  // 可选: 导出 xy 与逐层 prefill 输出(调试对拍)
  int max_steps = static_cast<int>(gsv::ar::T2SEngine::kMaxDecodeSteps);
  for (int i = 1; i < argc; ++i) {
    auto next = [&]() { return argv[++i]; };
    if (!std::strcmp(argv[i], "--weights")) weights = next();
    else if (!std::strcmp(argv[i], "--pairs-dir")) pairs_dir = next();
    else if (!std::strcmp(argv[i], "--out")) out_dir = next();
    else if (!std::strcmp(argv[i], "--only")) only = next();
    else if (!std::strcmp(argv[i], "--dump-layers")) dump_layers = next();
    else if (!std::strcmp(argv[i], "--max-steps")) max_steps = std::atoi(next());
  }
  if (!weights || !pairs_dir || !out_dir) {
    std::fprintf(stderr,
                 "用法: %s --weights w.gsv --pairs-dir DIR --out DIR [--only stem] "
                 "[--max-steps N]\n",
                 argv[0]);
    return 2;
  }
  ::mkdir(out_dir, 0755);

  try {
    const gsv::rt::GsvFile wf(weights);
    gsv::ar::T2SEngine eng(wf);
    const size_t bert_dim = eng.dims().bert_dim;

    DIR* d = ::opendir(pairs_dir);
    if (!d) {
      std::fprintf(stderr, "打不开目录 %s\n", pairs_dir);
      return 2;
    }
    double sum_prefill = 0, sum_decode = 0;
    size_t sum_steps = 0, n_run = 0;
    while (const dirent* e = ::readdir(d)) {
      const std::string name = e->d_name;
      if (name.size() && name[0] == '.') continue;
      if (only && name != only) continue;
      const std::string pdir = std::string(pairs_dir) + "/" + name;
      struct stat st{};
      if (::stat(pdir.c_str(), &st) || !S_ISDIR(st.st_mode)) continue;

      // meta.txt: "T P golden_steps"
      FILE* mf = std::fopen((pdir + "/meta.txt").c_str(), "r");
      if (!mf) continue;
      long T = 0, P = 0, gs = 0;
      if (std::fscanf(mf, "%ld %ld %ld", &T, &P, &gs) != 3) {
        std::fclose(mf);
        continue;
      }
      std::fclose(mf);

      const auto phones = read_vec<int64_t>(pdir + "/phones.i64", static_cast<size_t>(T));
      const auto prompt = read_vec<int64_t>(pdir + "/prompt.i64", static_cast<size_t>(P));
      const auto bert = read_vec<float>(pdir + "/bert.f32", static_cast<size_t>(T) * bert_dim);

      const auto t0 = std::chrono::steady_clock::now();
      gsv::ar::GenDebug* dbg = nullptr;
      std::vector<float> xy_dump;
      std::vector<std::vector<float>> layer_dump;
      struct Dumper : gsv::ar::GenDebug {
        std::vector<float>* xy;
        std::vector<std::vector<float>>* layers;
        void on_input(const float* p, size_t S) override {
          xy->assign(p, p + S * 512);
        }
        void on_layer(size_t l, const float* p, size_t S) override {
          if (layers->size() <= l) layers->resize(l + 1);
          (*layers)[l].assign(p, p + S * 512);
        }
      } dumper;
      if (dump_layers) {
        dumper.xy = &xy_dump;
        dumper.layers = &layer_dump;
        dbg = &dumper;
      }
      gsv::ar::GenResult r =
          eng.generate(phones.data(), static_cast<size_t>(T), prompt.data(),
                       static_cast<size_t>(P), bert.data(),
                       static_cast<size_t>(max_steps), dbg);
      const auto t1 = std::chrono::steady_clock::now();
      const double wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

      const std::string odir = std::string(out_dir) + "/" + name;
      ::mkdir(odir.c_str(), 0755);
      write_file(odir + "/out_tokens.i32", r.raw_argmax.data(),
                 r.raw_argmax.size() * 4);
      write_file(odir + "/out_sampled.i32", r.sampled.data(), r.sampled.size() * 4);
      write_file(odir + "/out_logits_first8.f32", r.logits_first8.data(),
                 r.logits_first8.size() * 4);
      write_file(odir + "/out_logits_last.f32", r.logits_last.data(),
                 r.logits_last.size() * 4);
      char meta[256];
      std::snprintf(meta, sizeof meta, "%zu %d %.3f %.3f %.3f\n", r.steps,
                    r.hit_eos ? 1 : 0, eng.last_prefill_ms(), eng.last_decode_ms(),
                    wall_ms);
      write_file(odir + "/out_meta.txt", meta, std::strlen(meta));

      if (dump_layers && !xy_dump.empty()) {
        char path[512];
        std::snprintf(path, sizeof path, "%s/xy.f32", dump_layers);
        write_file(path, xy_dump.data(), xy_dump.size() * 4);
        for (size_t l = 0; l < layer_dump.size(); ++l) {
          std::snprintf(path, sizeof path, "%s/L%02zu.f32", dump_layers, l);
          write_file(path, layer_dump[l].data(), layer_dump[l].size() * 4);
        }
      }

      sum_prefill += eng.last_prefill_ms();
      sum_decode += eng.last_decode_ms();
      sum_steps += r.steps;
      ++n_run;
      std::printf("%-48s steps=%4zu (golden %4ld) eos=%d prefill=%7.2fms decode=%8.2fms\n",
                  name.c_str(), r.steps, gs, r.hit_eos ? 1 : 0,
                  eng.last_prefill_ms(), eng.last_decode_ms());
      std::fflush(stdout);
    }
    ::closedir(d);
    if (n_run)
      std::printf("[汇总] pairs=%zu 总步数=%zu 平均 %.3f ms/token, 平均 prefill %.2f ms\n",
                  n_run, sum_steps, sum_decode / static_cast<double>(sum_steps),
                  sum_prefill / static_cast<double>(n_run));
    return 0;
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "错误: %s\n", ex.what());
    return 1;
  }
}
