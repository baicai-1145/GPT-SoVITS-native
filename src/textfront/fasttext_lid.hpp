// fasttext_lid.hpp — fasttext lid.176.bin 推理的原生移植(仅 predict 路径)。
//
// 位级复现 fasttext-predict 0.9.2.4 (pybind) 的 loadModel + predictLine:
//   text + '\n' -> Dictionary::getLine(tokenize + subword hashes)
//   hidden = mean(input rows[ids])                    (fp32 逐位同序累加)
//   HierarchicalSoftmaxLoss::predict(dfs Huffman 树, std_log=log(x+1e-5))
//   返回按 score 降序的 (label, exp(score)) 列表
//
// 仅支持 .bin 格式(version 12/11, 未量化), 不支持 .ftz; 与 CPUFast 运行时
// 实际使用的模型一致(pretrained_models/fast_langdetect/lid.176.bin)。
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace gsv::textfront {

class FastTextLid {
public:
    struct Pred {
        float score;      // exp(log(p+1e-5)), 即 python 侧概率
        std::string lang; // "__label__zh" -> "zh"
    };

    FastTextLid() = default;
    ~FastTextLid();
    FastTextLid(const FastTextLid&) = delete;
    FastTextLid& operator=(const FastTextLid&) = delete;

    // 加载失败返回 false 并写 err。predict 前必须成功调用一次。
    bool load(const std::string& path, std::string* err);

    // 复刻 m.predict(text, k, threshold): 返回按概率降序的前 k 个。
    void predict(const std::string& utf8Text, int k, float threshold,
                 std::vector<Pred>* out) const;

private:
    struct Entry {  // Dictionary::entry (type 字段文件中为 int8)
        std::string word;
        int64_t count = 0;
        int8_t type = 0;
        bool isLabel = false;
        std::vector<int32_t> subwords;  // initNgrams: word 自身 + ngram 桶 id
    };
    int32_t dim_ = 0, bucket_ = 0, minn_ = 0, maxn_ = 0;
    int32_t size_ = 0, nwords_ = 0, nlabels_ = 0;
    int64_t pruneidxSize_ = -1;
    bool lossIsHs_ = true;   // loss==1 (lid.176 实测)
    bool modelSup_ = true;   // model==sup
    std::vector<Entry> entries_;
    std::vector<int32_t> word2int_;

    std::vector<float> wi_;  // [nwords+bucket, dim] row-major
    std::vector<float> wo_;  // [nlabels, dim]

    // HierarchicalSoftmaxLoss 的 Huffman 树(load 后首个 predict 惰性构建,
    // 只依赖标签词频, 与输入无关 → 结果确定)
    struct HsNode {
        int32_t parent = -1, left = -1, right = -1;
        int64_t count = 0;
        bool binary = false;
    };
    std::vector<HsNode> hsTree_;
    int32_t hsOsz_ = 0;
    bool hsBuilt_ = false;

    void initNgrams();
    void computeSubwords(const std::string& word, std::vector<int32_t>* out) const;
    void buildHsTree();
    int32_t findSlot(const std::string& w, uint32_t h) const;
    void dfs(int k, float thresholdLog, int32_t node, float score,
             const float* hidden,
             std::vector<std::pair<float, int32_t>>* heap) const;

    static uint32_t fnvHash(const std::string& s);
};

}  // namespace gsv::textfront
