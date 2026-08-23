#!/usr/bin/env python3
"""export_g2pw_corpus.py — B6 验收语料: B5 jieba_fixtures 句子 → tokenize/G2PW 双 golden

输出 tests/golden_local/g2pw_corpus/
  tokens.json : {sent_id: {"text":..., "tokens":[...], "ids":[...]}}   (tokenize_and_map 口径)
  pinyin.json : {sent_id: {"text":..., "pinyins":[...char-level...]}}  (G2PWConverter.__call__ 口径,
                 partial_results=None — phrase overrides 由上游 chinese2 编排注入, 不在本验收内)
"""
import json
import os
import sys
from pathlib import Path

CPUFAST = Path("/Volumes/2T/GPT-SoVITS-CPUFast")
sys.path.insert(0, str(CPUFAST))
sys.path.insert(0, str(CPUFAST / "GPT_SoVITS"))
os.chdir(CPUFAST)

from text.g2pw.dataset import tokenize_and_map  # noqa: E402
from text.g2pw.torch_api import G2PWTorchConverter  # noqa: E402

FIX = Path("/Volumes/2T/wt-gsv/B5/tests/textfront/fixtures/jieba_fixtures.txt")
FUZZ = [Path(f"/Volumes/2T/wt-gsv/B5/tests/textfront/fixtures/jieba_fuzz{i}.txt") for i in (2, 3)]
OUT = Path("/Volumes/2T/GPT-SoVITS-native/tests/golden_local/g2pw_corpus")
ROBERTA_DIR = "/Volumes/2T/GPT-SoVITS-native/pretrained_models/chinese-roberta-wwm-ext-large"
MAX_SENTS = 440


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sents = []
    seen = set()

    def add(line):
        if not line.startswith("S\t"):
            return
        text = line.split("\t", 2)[2]
        if text not in seen:
            seen.add(text)
            sents.append(text)

    for line in FIX.read_text(encoding="utf-8").splitlines():
        add(line)
    # fuzz 语料补足: 纯中文+标点优先(避开 pypinyin 分组不齐的 CPUFast 缺陷句)
    import re
    has_alnum = re.compile(r"[a-zA-Z0-9]")
    for fp in FUZZ:
        if len(sents) >= MAX_SENTS:
            break
        for line in fp.read_text(encoding="utf-8").splitlines():
            if len(sents) >= MAX_SENTS:
                break
            # 只补纯中文+标点句: 避开 pypinyin 连续非汉字段合并导致的 CPUFast 越界缺陷
            text = line.split("\t", 2)[2] if line.startswith("S\t") else None
            if text is None or has_alnum.search(text):
                continue
            add(line)
    print(f"{len(sents)} sentences", flush=True)

    # tokenizer golden
    from tokenizers import Tokenizer
    hf = Tokenizer.from_file(os.path.join(ROBERTA_DIR, "tokenizer.json"))
    unk_id = hf.token_to_id("[UNK]")
    tokens_doc = {}
    for i, s in enumerate(sents):
        toks, _, _ = tokenize_and_map(hf, s) if False else (None, None, None)
    # tokenize_and_map 需要 _TokenizerAdapter 形状 (tokenize/convert_tokens_to_ids)

    class _Adapt:
        def __init__(self, tk):
            self._t = tk
            self._unk = tk.token_to_id("[UNK]")

        def tokenize(self, text):
            return self._t.encode(text, add_special_tokens=False).tokens

        def convert_tokens_to_ids(self, tokens):
            out = []
            for t in tokens:
                v = self._t.token_to_id(t)
                out.append(self._unk if v is None else v)
            return out

    ad = _Adapt(hf)
    skipped = []
    kept = []
    for s in sents:
        try:
            toks, _, _ = tokenize_and_map(ad, s)
        except Exception:
            # utils.py 对 normalize 变长字符(重音/全角折叠)存在已知越界 bug,
            # 该类句子在 CPUFast 真实路径同样不可用, 剔除出验收集
            skipped.append(s)
            continue
        kept.append((s, toks))
    for i, (s, toks) in enumerate(kept):
        tokens_doc[str(i)] = {
            "text": s,
            "tokens": toks,
            "ids": ad.convert_tokens_to_ids(toks),
        }
    if skipped:
        print(f"skipped {len(skipped)} sentences with tokenize_and_map bug: "
              + " | ".join(skipped[:3]), flush=True)
    sents = [s for s, _ in kept]
    (OUT / "tokens.json").write_text(
        json.dumps(tokens_doc, ensure_ascii=False), encoding="utf-8")
    print("tokens.json done", flush=True)

    # G2PW golden
    conv = G2PWTorchConverter(model_dir=str(CPUFAST / "GPT_SoVITS/text/G2PWModel"),
                              style="pinyin", model_source=ROBERTA_DIR,
                              enable_non_tradional_chinese=True)
    results = {}
    skipped2 = []
    for s in sents:
        try:
            outs = conv([s])
        except Exception as e:
            # pypinyin 默认读音长度与句长不齐等 CPUFast 已知缺陷句, 同样剔除
            skipped2.append((s, str(e)[:60]))
            continue
        results[str(len(results))] = {"text": s, "pinyins": list(outs[0])}
    if skipped2:
        print(f"skipped {len(skipped2)} g2pw sentences: " +
              "; ".join(f"{s[:12]}..:{e}" for s, e in skipped2[:3]), flush=True)
    (OUT / "pinyin.json").write_text(
        json.dumps(results, ensure_ascii=False), encoding="utf-8")
    print("ALL DONE")


if __name__ == "__main__":
    main()
