#!/usr/bin/env python3
"""export_wordpiece.py — B6: tokenizer.json → 紧凑词表 (data/bert_vocab.txt)

格式: 每行一个 token, 行号=id; 共 21128 行。C++ 侧加载为 id↔token 双向表。
"""
import json
from pathlib import Path

SRC = Path("/Volumes/2T/GPT-SoVITS-native/pretrained_models/chinese-roberta-wwm-ext-large/tokenizer.json")
OUT = Path("/Volumes/2T/wt-gsv/SOV/src/textfront/data/bert_vocab.txt")


def main():
    t = json.load(open(SRC))
    vocab = t["model"]["vocab"]
    assert len(vocab) == 21128
    lines = [None] * len(vocab)
    for tok, i in vocab.items():
        lines[i] = tok
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(lines)} tokens)")


if __name__ == "__main__":
    main()
