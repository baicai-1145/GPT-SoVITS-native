// fasttext_lid.cpp — 见 fasttext_lid.hpp。对 fasttext-predict 0.9.2.4 的
// loadModel/predictLine 做位级移植: FNV hash(signed char 版本)、开放寻址
// word2int、子词 ngram 哈希、HS Huffman 树 DFS 与 std_log=log(x+1e-5)。
// 本机 python 参考版(.tmp/lid_ref.py)已对官方 m.predict 校验(13 语料,
// 逐语种全同, 概率差 <2e-6)。
#include "fasttext_lid.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>

namespace gsv::textfront {
namespace {

constexpr int32_t kMagic = 793712314;
constexpr int32_t kMaxVersion = 12;
constexpr size_t kLabelPrefixLen = 9;  // "__label__"

inline bool isSplitWs(unsigned char c) {  // Dictionary::readWord 分词字符集
    return c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == '\v' ||
           c == '\f' || c == '\0';
}

bool heapGreater(const std::pair<float, int32_t>& l,
                 const std::pair<float, int32_t>& r) {
    return l.first > r.first;  // comparePairs: 构成以 first 为键的 min-heap
}

}  // namespace

FastTextLid::~FastTextLid() = default;

uint32_t FastTextLid::fnvHash(const std::string& s) {
    // 上游怪癖: h ^ uint32_t(int8_t(str[i])) — 符号扩展后按 uint32 截断
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < s.size(); ++i) {
        h = h ^ uint32_t(static_cast<int8_t>(s[i]));
        h = h * 16777619u;
    }
    return h;
}

int32_t FastTextLid::findSlot(const std::string& w, uint32_t h) const {
    int32_t size = static_cast<int32_t>(word2int_.size());
    int32_t id = static_cast<int32_t>(h % uint32_t(size));
    while (word2int_[size_t(id)] != -1 &&
           entries_[size_t(word2int_[size_t(id)])].word != w)
        id = (id + 1) % size;
    return id;
}

bool FastTextLid::load(const std::string& path, std::string* err) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        if (err) *err = "无法打开 lid 模型: " + path;
        return false;
    }
    auto fail = [&](const std::string& why) {
        if (err) *err = "fasttext_lid: " + why + " (" + path + ")";
        return false;
    };
    int32_t magic = 0, version = 0;
    in.read(reinterpret_cast<char*>(&magic), 4);
    in.read(reinterpret_cast<char*>(&version), 4);
    if (magic != kMagic || version > kMaxVersion)
        return fail("模型 magic/version 不符(需要未量化的 .bin)");

    struct ArgsBlock {  // Args::load 的二进制布局(每个 enum int32)
        int32_t dim, ws, epoch, minCount, neg, wordNgrams;
        int32_t loss, model, bucket, minn, maxn, lrUpdateRate;
    } a{};
    in.read(reinterpret_cast<char*>(&a), sizeof(a));
    double tDbl = 0;
    in.read(reinterpret_cast<char*>(&tDbl), 8);
    dim_ = a.dim;
    bucket_ = a.bucket;
    minn_ = a.minn;
    maxn_ = a.maxn;
    lossIsHs_ = (a.loss == 1);  // hs=1; 非 hs 走 SoftmaxLoss 不支持(见下)
    modelSup_ = (a.model == 3);  // sup=3
    (void)tDbl;

    in.read(reinterpret_cast<char*>(&size_), 4);
    in.read(reinterpret_cast<char*>(&nwords_), 4);
    in.read(reinterpret_cast<char*>(&nlabels_), 4);
    int64_t ntokens = 0;
    in.read(reinterpret_cast<char*>(&ntokens), 8);
    in.read(reinterpret_cast<char*>(&pruneidxSize_), 8);

    entries_.resize(size_t(size_));
    for (int32_t i = 0; i < size_; ++i) {
        auto& e = entries_[size_t(i)];
        char c;
        while (in.get(c) && c != 0) e.word.push_back(c);
        in.read(reinterpret_cast<char*>(&e.count), 8);
        in.read(reinterpret_cast<char*>(&e.type), 1);  // entry_type : int8_t!
        e.isLabel = (e.type == 1);
    }
    if (!in) return fail("字典区读取不完整");
    if (pruneidxSize_ > 0) {  // 完备性分支(lid.176 无 prune)
        for (int64_t i = 0; i < pruneidxSize_; ++i) in.seekg(8, std::ios::cur);
    }

    uint8_t quantInput = 0;
    in.read(reinterpret_cast<char*>(&quantInput), 1);
    if (quantInput) return fail("不支持量化矩阵(请用 lid.176.bin 而非 .ftz)");
    int64_t mIn = 0, nIn = 0;
    in.read(reinterpret_cast<char*>(&mIn), 8);
    in.read(reinterpret_cast<char*>(&nIn), 8);
    if (nIn != dim_) return fail("input 矩阵列数与 dim 不符");
    wi_.resize(size_t(mIn) * size_t(nIn));
    in.read(reinterpret_cast<char*>(wi_.data()),
            std::streamsize(size_t(mIn) * size_t(nIn) * 4));

    uint8_t qout = 0;
    in.read(reinterpret_cast<char*>(&qout), 1);
    if (qout) return fail("不支持 qout 量化输出矩阵");
    int64_t mOut = 0, nOut = 0;
    in.read(reinterpret_cast<char*>(&mOut), 8);
    in.read(reinterpret_cast<char*>(&nOut), 8);
    if (mOut != nlabels_ && !(modelSup_ && mOut == nlabels_))
        return fail("output 矩阵行数与 nlabels 不符");
    wo_.resize(size_t(mOut) * size_t(nOut));
    in.read(reinterpret_cast<char*>(wo_.data()),
            std::streamsize(size_t(mOut) * size_t(nOut) * 4));
    if (!in) return fail("矩阵区读取不完整");

    int32_t w2iSize = int32_t(std::ceil(double(size_) / 0.7));
    word2int_.assign(size_t(w2iSize), -1);
    for (int32_t i = 0; i < size_; ++i)
        word2int_[size_t(findSlot(entries_[size_t(i)].word,
                                  fnvHash(entries_[size_t(i)].word)))] = i;
    initNgrams();
    return true;
}

void FastTextLid::computeSubwords(const std::string& word,
                                  std::vector<int32_t>* out) const {
    for (size_t i = 0; i < word.size(); ++i) {
        if ((word[i] & 0xC0) == 0x80) continue;  // UTF-8 续字节不作起点
        size_t j = i;
        int n = 1;
        while (j < word.size() && n <= maxn_) {
            ++j;
            while (j < word.size() && (word[j] & 0xC0) == 0x80) ++j;
            if (n >= minn_ && !(n == 1 && (i == 0 || j == word.size()))) {
                // ngram 字节串 = word[i..j); 直接哈希
                std::string ngram = word.substr(i, j - i);
                out->push_back(nwords_ +
                               int32_t(fnvHash(ngram) % uint32_t(bucket_)));
            }
            ++n;
        }
    }
}

void FastTextLid::initNgrams() {
    const std::string kEos = "</s>";
    for (int32_t i = 0; i < size_; ++i) {
        auto& e = entries_[size_t(i)];
        e.subwords.clear();
        e.subwords.push_back(i);
        if (e.word != kEos) computeSubwords("<" + e.word + ">", &e.subwords);
    }
}

void FastTextLid::buildHsTree() {
    hsOsz_ = nlabels_;
    HsNode init{-1, -1, -1, int64_t(1e15), false};
    hsTree_.assign(size_t(2 * hsOsz_ - 1), init);
    for (int32_t i = 0; i < hsOsz_; ++i)
        hsTree_[size_t(i)].count = entries_[size_t(nwords_ + i)].count;
    int32_t leaf = hsOsz_ - 1;
    int32_t node = hsOsz_;
    for (int32_t i = hsOsz_; i < 2 * hsOsz_ - 1; ++i) {
        int32_t mini[2] = {0, 0};
        for (int32_t j = 0; j < 2; ++j) {
            if (leaf >= 0 &&
                hsTree_[size_t(leaf)].count < hsTree_[size_t(node)].count) {
                mini[j] = leaf--;
            } else {
                mini[j] = node++;
            }
        }
        hsTree_[size_t(i)].left = mini[0];
        hsTree_[size_t(i)].right = mini[1];
        hsTree_[size_t(i)].count =
            hsTree_[size_t(mini[0])].count + hsTree_[size_t(mini[1])].count;
        hsTree_[size_t(mini[0])].parent = i;
        hsTree_[size_t(mini[1])].parent = i;
        hsTree_[size_t(mini[1])].binary = true;
    }
    hsBuilt_ = true;
}

void FastTextLid::dfs(int k, float thresholdLog, int32_t node, float score,
                      const float* hidden,
                      std::vector<std::pair<float, int32_t>>* heap) const {
    if (score < thresholdLog) return;
    if (heap->size() == size_t(k) && score < heap->front().first) return;
    const HsNode& nd = hsTree_[size_t(node)];
    if (nd.left == -1 && nd.right == -1) {
        heap->emplace_back(score, node);
        std::push_heap(heap->begin(), heap->end(), heapGreater);
        if (heap->size() > size_t(k)) {
            std::pop_heap(heap->begin(), heap->end(), heapGreater);
            heap->pop_back();
        }
        return;
    }
    float f = 0.f;
    {  // dotRow(hidden, node-osz): fp32 同序累加
        const float* row = wo_.data() + size_t(node - hsOsz_) * size_t(dim_);
        float d = 0.f;
        for (int32_t j = 0; j < dim_; ++j) d += row[j] * hidden[j];
        f = 1.f / (1.f + std::exp(-d));
    }
    dfs(k, thresholdLog, nd.left, score + std::log((1.f - f) + 1e-5f),
        hidden, heap);
    dfs(k, thresholdLog, nd.right, score + std::log(f + 1e-5f), hidden, heap);
}

void FastTextLid::predict(const std::string& utf8Text, int k, float threshold,
                          std::vector<Pred>* out) const {
    out->clear();
    if (k <= 0) k = nlabels_;
    // ---- getLine: 分词 + 子词 (与 predictLine(text+'\n') 一致) ----
    std::vector<int32_t> words;
    {
        std::string data = utf8Text;
        data.push_back('\n');
        size_t pos = 0;
        while (pos < data.size()) {
            std::string token;
            if (data[pos] == '\n') {
                token = "</s>";
                ++pos;
            } else {
                while (pos < data.size() && !isSplitWs(uint8_t(data[pos]))) {
                    token.push_back(data[pos]);
                    ++pos;
                }
                // 非换行的空白被吞掉; 换行 ungetc 回来(下方循环会当 EOS)
                if (pos < data.size() && data[pos] != '\n') ++pos;
            }
            if (token.empty()) continue;
            bool isLabel = token.rfind("__label__", 0) == 0;
            if (!isLabel) {
                int32_t wid =
                    word2int_[size_t(findSlot(token, fnvHash(token)))];
                if (wid >= 0 && maxn_ <= 0) {
                    words.push_back(wid);
                } else if (wid >= 0) {
                    const auto& sw = entries_[size_t(wid)].subwords;
                    words.insert(words.end(), sw.begin(), sw.end());
                } else if (token != "</s>") {
                    computeSubwords("<" + token + ">", &words);
                }
            }
            if (token == "</s>") break;
        }
    }
    if (words.empty()) return;  // 上游: words 空 → predictions 为空列表

    // ---- hidden = mean(wi rows) (fp32 同序累加) ----
    std::vector<float> hidden(size_t(dim_), 0.f);
    for (auto id : words) {
        const float* row = wi_.data() + size_t(id) * size_t(dim_);
        for (int32_t j = 0; j < dim_; ++j) hidden[size_t(j)] += row[j];
    }
    {
        const float inv = 1.f / float(words.size());
        for (auto& v : hidden) v *= inv;
    }

    // ---- HS tree DFS ----
    if (!hsBuilt_) const_cast<FastTextLid*>(this)->buildHsTree();
    std::vector<std::pair<float, int32_t>> heap;
    dfs(k, std::log(threshold + 1e-5f), 2 * hsOsz_ - 2, 0.f, hidden.data(),
        &heap);
    std::sort_heap(heap.begin(), heap.end(), heapGreater);
    for (const auto& [score, node] : heap) {
        Pred p;
        p.score = std::exp(score);
        // 叶子 node 即 label id (getLabel: words_[lid + nwords_])
        const std::string& w = entries_[size_t(nwords_ + node)].word;
        p.lang = w.substr(kLabelPrefixLen);
        out->push_back(std::move(p));
    }
}

}  // namespace gsv::textfront
