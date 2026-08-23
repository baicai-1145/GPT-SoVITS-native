#!/usr/bin/env python3
"""gen_textfront_goldens.py — python reference goldens for B7 textfront.

Usage:
  gen_textfront_goldens.py norm <input.txt> <out.golden>
      TextNormalizer chain + replace_punctuation +
      replace_consecutive_punctuation per line.
  gen_textfront_goldens.py g2p <input.txt> <out.golden>
      text_normalize -> g2p (is_g2pw monkeypatched False) -> symbols2 ids
      per line; rows: P<TAB>idx<TAB>phones<TAB>word2ph<TAB>ids, or
      E<TAB>idx<TAB>ExcName when the reference raises.

The G2PW torch converter is stubbed before importing chinese2 so the
reference runs CPU-only and takes the pypinyin path.
"""
import sys
import types


def _stub_g2pw():
    fake = types.ModuleType("text.g2pw.torch_api")

    class _FakeConverter:
        def __init__(self, *a, **k):
            pass

    fake.G2PWTorchConverter = _FakeConverter
    sys.modules["text.g2pw.torch_api"] = fake


def esc(s):
    return (s.replace("\\", "\\\\").replace("\n", "\\n")
            .replace("\t", "\\t").replace("\r", "\\r"))


def read_lines(path):
    lines = open(path, encoding="utf-8").read().split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def run_norm(inp, outp):
    sys.path.insert(0, "/Volumes/2T/GPT-SoVITS-CPUFast/GPT_SoVITS")
    import re
    from text.zh_normalization.text_normlization import TextNormalizer
    punctuation = ["!", "?", "…", ",", "."]
    punctuation.append("-")
    rep_map = {
        "：": ",", "；": ",", "，": ",", "。": ".", "！": "!", "？": "?",
        "\n": ".", "·": ",", "、": ",", "...": "…", "$": ".", "/": ",",
        "—": "-", "~": "…", "～": "…",
    }
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))
    keep = re.compile(r"[^\u4e00-\u9fa5" + "".join(punctuation) + r"]")

    def replace_punctuation(text):
        text = text.replace("嗯", "恩").replace("呣", "母")
        return keep.sub("", pattern.sub(lambda x: rep_map[x.group()], text))

    coll = re.compile(
        f"([{ ''.join(re.escape(p) for p in punctuation)}])"
        f"([{ ''.join(re.escape(p) for p in punctuation)}])+")

    tn = TextNormalizer()
    with open(outp, "w", encoding="utf-8") as f:
        for idx, text in enumerate(read_lines(inp)):
            try:
                sents = tn.normalize(text)
                dest = "".join(replace_punctuation(s) for s in sents)
                dest = coll.sub(r"\1", dest)
                f.write(f"N\t{idx}\t{esc(dest)}\n")
            except Exception as e:
                f.write(f"E\t{idx}\t{type(e).__name__}\n")


def run_g2p(inp, outp):
    sys.path.insert(0, "/Volumes/2T/GPT-SoVITS-CPUFast/GPT_SoVITS")
    _stub_g2pw()
    import logging
    logging.disable(logging.CRITICAL)
    from text import chinese2, cleaned_text_to_sequence
    chinese2.is_g2pw = False
    with open(outp, "w", encoding="utf-8") as f:
        for idx, text in enumerate(read_lines(inp)):
            try:
                norm = chinese2.text_normalize(text)
                phones, word2ph = chinese2.g2p(norm)
                ids = cleaned_text_to_sequence(phones, version="v2")
                f.write("P\t%d\t%s\t%s\t%s\n" % (
                    idx, esc(" ".join(phones)),
                    ",".join(map(str, word2ph)),
                    ",".join(map(str, ids))))
            except Exception as e:
                f.write(f"E\t{idx}\t{type(e).__name__}\n")


if __name__ == "__main__":
    mode, inp, outp = sys.argv[1], sys.argv[2], sys.argv[3]
    if mode == "norm":
        run_norm(inp, outp)
    elif mode == "g2p":
        run_g2p(inp, outp)
    else:
        raise SystemExit(f"unknown mode {mode}")
