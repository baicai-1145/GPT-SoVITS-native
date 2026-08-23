#!/usr/bin/env python3
"""Adversarial fuzz corpus for B5 jieba parity testing.

Generates thousands of random sentences over a mixed alphabet (common Han,
rare Han near U+9FD5, ASCII alnum, CJK/fullwidth punctuation, whitespace)
so that dictionary tie-breaks, HMM viterbi cold paths and block-splitting
edge cases all get exercised. Output format identical to gen_jieba_fixtures.py
("GSVFIX01").

    GPTSoVits-python tools/fuzz_jieba_fixtures.py --num 3000 --seed 42 \
        --out tests/textfront/fixtures/jieba_fuzz.txt
"""
import argparse
import os
import random
import sys

from gen_jieba_fixtures import esc


def build_alphabet(site_packages):
    sys.path.insert(0, site_packages)
    import jieba_fast
    jieba_fast.setLogLevel(50)
    import jieba_fast.posseg as psg
    # characters that appear in emissions or dict words: realistic Han
    han = sorted({c for em in psg.emit_P.values() for c in list(em)[:200]})
    # rare Han right at the re_han boundary
    boundary = [chr(c) for c in range(0x9FD0, 0x9FD6)]
    ascii_alnum = list("abcXYZ0123456789")
    punct = list("。，！？；：、""''《》（）…—～·%+#&._$/:@")
    ws = [" ", "\t", "\n", "\r\n", "\u3000", "\xa0"]
    return han + boundary + ascii_alnum + punct + ws


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--num", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--maxlen", type=int, default=60)
    ap.add_argument("--site-packages",
                    default="/Users/baicai1145/miniconda3/envs/GPTSoVits/"
                            "lib/python3.10/site-packages")
    ap.add_argument("--out",
                    default=os.path.join(here, "tests", "textfront",
                                         "fixtures", "jieba_fuzz.txt"))
    args = ap.parse_args()

    import jieba_fast
    jieba_fast.setLogLevel(50)
    import jieba_fast.posseg as psg

    rng = random.Random(args.seed)
    alpha = build_alphabet(args.site_packages)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    n_tok = 0
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("GSVFIX01\n")
        for idx in range(args.num):
            length = rng.randint(1, args.maxlen)
            text = "".join(rng.choice(alpha) for _ in range(length))
            toks = [(w.word, w.flag) for w in psg.lcut(text)]
            n_tok += len(toks)
            f.write(f"S\tfuzz{idx:05d}\t{esc(text)}\n")
            for w, flag in toks:
                f.write(f"W\t{esc(w)}\t{flag}\n")
            f.write("\n")
    print(f"wrote {args.out}: {args.num} cases, {n_tok} tokens "
          f"(seed={args.seed})")


if __name__ == "__main__":
    main()
