// fuzz_langsegment.cpp — FE-AUTO-1 扩样对拍: 随机混排语料 ×4 模式 vs python 权威输出
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "langsegmenter.hpp"
#include "runtime/mini_json.hpp"

namespace {
std::string readAll(const std::string& p, bool* ok) {
    *ok = false;
    FILE* f = std::fopen(p.c_str(), "rb");
    if (!f) return {};
    std::string out;
    char buf[65536];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) out.append(buf, n);
    std::fclose(f);
    *ok = true;
    return out;
}
}  // namespace

int main() {
    std::string lidPath =
        "/Volumes/2T/GPT-SoVITS-CPUFast/GPT_SoVITS/pretrained_models/"
        "fast_langdetect/lid.176.bin";
    if (const char* env = std::getenv("FE_LID_BIN")) lidPath = env;

    gsv::textfront::LangSegmenterCpp seg;
    std::string err;
    std::string budouxDir = FE_DATA_DIR + std::string("/src/textfront/data/budoux");
    if (!seg.load(lidPath, budouxDir, &err)) {
        std::printf("SKIP: %s\n", err.c_str());
        return 1;
    }

    bool ok;
    const std::string bytes = readAll(FE_FUZZ_JSON, &ok);
    if (!ok) { std::printf("FAIL: 无法读 fuzz json\n"); return 1; }
    gsv::rt::json::JValue root = gsv::rt::json::parse(bytes.data(), bytes.size());
    if (!root.is(gsv::rt::json::JType::Array)) {
        std::printf("FAIL: fuzz 根不是数组\n");
        return 1;
    }

    int total = 0, passRec = 0, failRec = 0;
    int totalPieces = 0, diffPieces = 0;
    for (const auto& rec : root.arr) {
        const auto* pm = rec.find("mode");
        const auto* pi = rec.find("input");
        const auto* pw = rec.find("pieces");
        if (!pm || !pi || !pw) continue;
        const std::string mode = pm->as_string();
        const std::string input = pi->as_string();
        // 卡口径: 仅移植了 auto(空参)与 zh(default_lang="zh")两口径
        if (mode != "auto(auto=空参)" && mode != "zh") continue;
        auto got = seg.getTexts(input, mode == "zh" ? "zh" : "");
        ++total;
        bool recOk = true;
        size_t idx = 0;
        for (const auto& piece : pw->arr) {
            if (piece.arr.size() < 2) continue;
            const std::string wl = piece.arr[0].as_string();
            const std::string wt = piece.arr[1].as_string();
            ++totalPieces;
            if (idx >= got.size()) { recOk = false; ++diffPieces; continue; }
            const auto& g = got[idx];
            if (g.lang != wl || g.text != wt) {
                ++diffPieces;
                recOk = false;
                if (failRec < 8)
                    std::printf("DIFF[%s] #%zu: py[%s]%s vs cpp[%s]%s\n",
                                mode.c_str(), idx, wl.c_str(), wt.c_str(),
                                g.lang.c_str(), g.text.c_str());
            }
            ++idx;
        }
        if (idx < got.size()) {
            recOk = false;
            if (failRec < 8)
                std::printf("DIFF[%s]: cpp 多出 %zu 片\n", mode.c_str(),
                            got.size() - idx);
        }
        recOk ? ++passRec : ++failRec;
    }
    std::printf("fuzz 对拍: 记录 %d/%d 全等, 片级差异 %d/%d\n", passRec, total,
                diffPieces, totalPieces);
    return failRec == 0 ? 0 : 1;
}
