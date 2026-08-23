#!/usr/bin/env python3
"""export_bert_fixtures.py — B8 验收 fixture 导出 (chinese_bert.py 口径)

用法: /Users/baicai1145/miniconda3/envs/GPTSoVits/bin/python tools/export_bert_fixtures.py
输出: tests/golden_local/bert_fixtures/<tag>/{inputs,hooks,meta.json}
口径: chinese_bert.get_bert_feature 同路径 — tokenizer 单句不 padding,
      BertForMaskedLM(output_hidden_states=True),
      hidden_states = (emb, layer0..layer23), [-3]=layer21 输出。
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

CPUFAST = Path("/Volumes/2T/GPT-SoVITS-CPUFast")
sys.path.insert(0, str(CPUFAST))
sys.path.insert(0, str(CPUFAST / "GPT_SoVITS"))
os.chdir(CPUFAST)

from text.chinese_bert import load_model, load_tokenizer  # noqa: E402

BERT_BASE = Path("/Volumes/2T/GPT-SoVITS-native/pretrained_models/chinese-roberta-wwm-ext-large")
OUTROOT = Path("/Volumes/2T/GPT-SoVITS-native/tests/golden_local/bert_fixtures")

# 3 句: 短句/常规/多音字(银行为 B6 铺垫)
TAG_SENTENCES = [
    ("hello_world", "你好，世界。"),
    ("hotpot", "重庆的火锅店终于开张了。"),
    ("bank_river", "银行旁边的河水平静地流着。"),
]


def save_f32(d: Path, name: str, t: torch.Tensor):
    t = t.detach().float().cpu().contiguous()
    t.numpy().tofile(d / f"{name}.bin")
    (d / f"{name}.shape").write_text(" ".join(map(str, t.shape)))


def save_i64(d: Path, name: str, t: torch.Tensor):
    t = t.detach().cpu().contiguous()
    t.numpy().tofile(d / f"{name}.bin")
    (d / f"{name}.shape").write_text(" ".join(map(str, t.shape)))


def main():
    OUTROOT.mkdir(parents=True, exist_ok=True)
    print("loading model/tokenizer ...", flush=True)
    model = load_model(str(BERT_BASE)).eval()
    tok = load_tokenizer(str(BERT_BASE))

    for tag, sent in TAG_SENTENCES:
        d = OUTROOT / tag
        (d / "inputs").mkdir(parents=True, exist_ok=True)
        (d / "hooks").mkdir(parents=True, exist_ok=True)
        enc = tok(sent)  # ChineseBertTokenizer.__call__ 返回 dict of tensors
        ids = enc["input_ids"]
        ttids = enc["token_type_ids"]
        amask = enc["attention_mask"]
        with torch.no_grad():
            outs = model(input_ids=ids, attention_mask=amask,
                         token_type_ids=ttids, output_hidden_states=True)
        hs = outs["hidden_states"]  # (emb, L0..L23) len 25
        assert len(hs) == 25 and hs[-3].shape == outs["last_hidden_state"].shape

        save_i64(d / "inputs", "input_ids", ids)
        save_i64(d / "inputs", "token_type_ids", ttids)
        save_i64(d / "inputs", "attention_mask", amask)
        save_f32(d / "hooks", "bert_emb_out", hs[0])
        save_f32(d / "hooks", "bert_layer0_out", hs[1])
        save_f32(d / "hooks", "bert_layer22_out", hs[-3])   # get_bert_feature 取点
        save_f32(d / "hooks", "bert_last_out", outs["last_hidden_state"])
        meta = {
            "tag": tag, "sentence": sent, "model": "roberta_wwm_ext_large",
            "hidden": 1024, "layers": 24, "heads": 16,
            "ln_eps": 1e-12, "note": "position_ids=arange(L) per chinese_bert.py",
        }
        (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        print(f"[{tag}] L={ids.shape[1]} saved", flush=True)
    print("ALL DONE")


if __name__ == "__main__":
    main()
