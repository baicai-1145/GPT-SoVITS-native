#!/usr/bin/env python3
"""export_roberta_vocab.py — C2: 导出 roberta WordPiece 词表为逐行文本.

输入: <bert_dir>/tokenizer.json (HuggingFace tokenizers 格式)
输出: src/runtime/data/roberta_vocab.txt  每 id 一行一个 token (按 id 排序),
      行号即 id。C++ BertTokenizer 用它做 greedy longest-match-first WordPiece。

BertNormalizer/BertPreTokenizer 的 C++ 复刻口径见 pipeline_tokenizer.hpp:
clean_text + CJK 加空格 + lowercase + 标点切分 + "##" 续词, unk=[UNK]。
"""
import json
import os
import sys

BERT_DIR = "/Volumes/2T/GPT-SoVITS-native/pretrained_models/chinese-roberta-wwm-ext-large"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "src", "runtime", "data", "roberta_vocab.txt")


def main():
    tok = json.load(open(os.path.join(BERT_DIR, "tokenizer.json"), encoding="utf-8"))
    vocab = tok["model"]["vocab"]  # token -> id
    n = len(vocab)
    lines = [""] * n
    for t, i in vocab.items():
        assert lines[i] == "", f"duplicate id {i}"
        lines[i] = t
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    inv = {v: k for k, v in vocab.items()}
    print(f"vocab {n} tokens -> {OUT}")
    print("specials:", {k: vocab[k] for k in ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]") if k in vocab})
    # 自检: 若词表非连续 id 会破坏行号=id 约定
    assert set(range(n)) == set(inv.keys()), "vocab ids not contiguous"
    return 0


if __name__ == "__main__":
    sys.exit(main())
