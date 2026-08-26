// test_langsegment.cpp — FE-AUTO-1 验收: LangSegmenter auto 切分位级移植。
//
// 对照 tests/textfront/fixtures/langsegment_auto.json (16 语料 × 4 模式,
// python LangSegmenter.getTexts 空参口径权威定标):
//   * auto 模式: C++ getTexts(text) vs fixture pieces 逐字节全等
//   * 红线(all_zh): zh 片切分结果与 fixture["zh"] pieces 一致(空参短路链)
//   * 空 input 的 IndexError 口径: pieces 为空
//
// 需要: lid.176.bin(路径经环境变量 FE_LID_BIN 或默认 CPUFast 路径) +
// src/textfront/data/budoux/(由 tools/export_budoux_models.py 再生)。
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "langsegmenter.hpp"
#include "runtime/mini_json.hpp"

namespace {

int gFailed = 0;
int gTotal = 0;

std::string readAll(const std::string& path, bool* ok) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        if (ok) *ok = false;
        return {};
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    if (ok) *ok = true;
    return ss.str();
}

// json 字符串转义(python repr 风格展示)
std::string esc(const std::string& s) {
    std::string o;
    for (unsigned char c : s) {
        if (c == '\n') o += "\\n";
        else if (c == '\t') o += "\\t";
        else o += char(c);
    }
    return o;
}

}  // namespace

int main() {
    const char* fixtureDir = FE_FIXTURE_DIR;
    const char* dataDir = FE_DATA_DIR;

    // lid.176.bin: 本机 CPUFast 权威路径(不入库, 与 golden 同源)
    std::string lidPath = "/Volumes/2T/GPT-SoVITS-CPUFast/GPT_SoVITS/"
                          "pretrained_models/fast_langdetect/lid.176.bin";
    if (const char* env = std::getenv("FE_LID_BIN")) lidPath = env;

    gsv::textfront::LangSegmenterCpp seg;
    std::string err;
    // FE_DATA_DIR = 仓库根(tests/..); budoux 模型在 src/textfront/data/budoux
    std::string budouxDir = std::string(dataDir) + "/src/textfront/data/budoux";
    if (!seg.load(lidPath, budouxDir, &err)) {
        std::printf("SKIP: langsegmenter 加载失败: %s\n", err.c_str());
        return 1;  // 数据缺失必须显式失败, 不允许静默通过
    }

    bool ok;
    const std::string bytes = readAll(std::string(fixtureDir) +
                                          "/langsegment_auto.json",
                                      &ok);
    if (!ok) {
        std::printf("FAIL: 无法读 fixture\n");
        return 1;
    }
    gsv::rt::json::JValue root = gsv::rt::json::parse(bytes.data(), bytes.size());
    if (!root.is(gsv::rt::json::JType::Array)) {
        std::printf("FAIL: fixture 根不是数组\n");
        return 1;
    }

    for (const auto& entry : root.arr) {
        const std::string mode = entry.find("mode")->as_string();
        // 模式串: "zh"/"ja"/"ko"/"auto(auto=空参)" → getTexts 第二参
        std::string defaultLang;
        if (mode.find("auto") != 0) defaultLang = mode;
        const std::string input = entry.find("input")->as_string();
        const gsv::rt::json::JValue* wantPieces = entry.find("pieces");
        const std::string wantErr = entry.find("error")->as_string();
        (void)wantErr;  // IndexError 语料: python 空输入抛错; native 口径同空片

        ++gTotal;
        auto got = seg.getTexts(input, defaultLang);

        bool pass = got.size() == wantPieces->arr.size();
        // pieces 是 [[lang,text],...] 数组的数组
        std::vector<std::pair<std::string, std::string>> want;
        for (const auto& pr : wantPieces->arr) {
            if (!pr.is(gsv::rt::json::JType::Array) || pr.arr.size() < 2)
                continue;
            want.emplace_back(pr.arr[0].as_string(), pr.arr[1].as_string());
        }
        pass = got.size() == want.size();
        for (size_t i = 0; pass && i < got.size(); ++i)
            pass = got[i].lang == want[i].first &&
                   got[i].text == want[i].second;
        if (!pass) {
            ++gFailed;
            std::printf("FAIL [%s] input=<%s>\n", mode.c_str(), esc(input).c_str());
            for (size_t i = 0; i < got.size() || i < want.size(); ++i) {
                if (i < want.size())
                    std::printf("   want %s <%s>\n", want[i].first.c_str(),
                                esc(want[i].second).c_str());
                else
                    std::printf("   want <missing>\n");
                if (i < got.size())
                    std::printf("   got  %s <%s>%s\n", got[i].lang.c_str(),
                                esc(got[i].text).c_str(),
                                i >= want.size() ? "  (多余)" : "");
                else
                    std::printf("   got  <missing>\n");
            }
        }
    }

    std::printf("%s: %d/%d 语料位级全等\n", gFailed ? "FAIL" : "PASS",
                gTotal - gFailed, gTotal);
    return gFailed ? 1 : 0;
}
