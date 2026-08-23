#!/usr/bin/env python3
"""export_g2pw_fixtures.py — B8 验收二: G2PW BERT-base (12L×768d) fixtures

用法: /Users/baicai1145/miniconda3/envs/GPTSoVits/bin/python tools/export_g2pw_fixtures.py
口径: G2PWTorchConverter._prepare_data + dataset.prepare_onnx_input 构造输入,
      取 batch 中前 N_QUERY 条样本, 存 input_ids/token_type/attention_mask +
      bert 末层 hidden + classifier logits + 最终 probs。
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

from text.g2pw.torch_api import G2PWTorchConverter  # noqa: E402
from text.g2pw.dataset import prepare_onnx_input  # noqa: E402

MODEL_DIR = CPUFAST / "GPT_SoVITS/text/G2PWModel"
OUT = Path("/Volumes/2T/GPT-SoVITS-native/tests/golden_local/g2pw_fixtures")
N_QUERY = 3  # 每句取前 3 个查询字符样本

# 多音字句子: 行(xing/hang) 长(zhang/chang) 重(zhong/chong)
SENTENCES = ["重庆的火锅店终于开张了。", "银行旁边的河水平静地流着。"]


def save_f32(d, name, arr):
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    arr.tofile(d / f"{name}.bin")
    (d / f"{name}.shape").write_text(" ".join(map(str, arr.shape)))


def save_i64(d, name, arr):
    arr = np.ascontiguousarray(arr, dtype=np.int64)
    arr.tofile(d / f"{name}.bin")
    (d / f"{name}.shape").write_text(" ".join(map(str, arr.shape)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("building G2PW converter ...", flush=True)
    # 本机无 'bert-base-chinese' 目录; hfl 中文 roberta 与 bert-base-chinese 同词表(21128),
    # 其 tokenizer.json 可直接替代 (G2PW 只用 convert_tokens_to_ids)
    ROBERTA_DIR = "/Volumes/2T/GPT-SoVITS-native/pretrained_models/chinese-roberta-wwm-ext-large"
    conv = G2PWTorchConverter(model_dir=str(MODEL_DIR), style="pinyin",
                              model_source=ROBERTA_DIR,
                              enable_non_tradional_chinese=True)
    model = conv.model
    assert isinstance(model, torch.nn.Module)
    texts, model_query_ids, result_query_ids, sent_ids, partial = (
        conv._prepare_data(sentences=list(SENTENCES), preset_partial_results=None))
    print("queries:", list(zip(texts, model_query_ids)), flush=True)

    mi = prepare_onnx_input(
        tokenizer=conv.tokenizer, labels=conv.labels,
        char2phonemes=conv.char2phonemes, chars=conv.chars,
        texts=texts, query_ids=model_query_ids,
        use_mask=conv.config.use_mask, window_size=None,
        char2id=conv.char2id, char_phoneme_masks=conv.char_phoneme_masks)

    ids_t = torch.from_numpy(mi["input_ids"])
    tt_t = torch.from_numpy(mi["token_type_ids"])
    am_t = torch.from_numpy(mi["attention_masks"])

    n_total = ids_t.shape[0]
    take = min(N_QUERY * len(SENTENCES), n_total)
    # bert hidden 全量算一次 (batch), 再逐条切片
    with torch.no_grad():
        hidden = model.bert(ids_t, tt_t, am_t)  # [B,T,768]
        h_gathered = hidden[torch.arange(n_total), torch.from_numpy(mi["position_ids"])]
        logits = model.classifier(h_gathered)
        pos_pred = model.pos_classifier(h_gathered).argmax(dim=1)
        mask_w = torch.sigmoid(
            model.descriptor_bias(torch.zeros_like(torch.from_numpy(mi["char_ids"])))
            + model.char_descriptor(torch.from_numpy(mi["char_ids"]))
            + model.second_order_descriptor(torch.from_numpy(mi["char_ids"]) * model.num_pos + pos_pred)
        ) * torch.from_numpy(mi["phoneme_masks"]).float()
        lm = logits.max(dim=1, keepdim=True).values
        probs = torch.exp(logits - lm) * mask_w
        probs = probs / probs.sum(dim=1, keepdim=True)

    for k in range(take):
        d = OUT / f"q{k}"
        d.mkdir(parents=True, exist_ok=True)
        T = int((am_t[k] == 1).sum())  # 有效长度 (无 padding 尾)
        save_i64(d, "input_ids", ids_t[k:k+1, :T].numpy())
        save_i64(d, "token_type_ids", tt_t[k:k+1, :T].numpy())
        save_i64(d, "attention_mask", am_t[k:k+1, :T].numpy())
        save_f32(d, "bert_hidden", hidden[k, :T].numpy())
        save_f32(d, "cls_logits", logits[k].numpy())
        save_f32(d, "final_probs", probs[k].numpy())
        meta = {
            "text": texts[sent_ids.index(k)] if False else None,
            "query_id": int(model_query_ids[k]) if k < len(model_query_ids) else None,
            "position_id": int(mi["position_ids"][k]),
            "char": None, "T": T,
            "note": "text/query 映射见导出日志; BERT-base eps=1e-5 mask=-10000",
        }
        # 记录该样本对应的 text 与 query 字符
        # sent_ids/model_query_ids 与 batch 顺序一一对应 (prepare_data 输出顺序)
        (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        print(f"[q{k}] T={T} pos_id={int(mi['position_ids'][k])}", flush=True)

    # 额外存全 batch 元数据便于追溯
    meta_all = {
        "sentences": SENTENCES,
        "texts": list(texts),
        "model_query_ids": [int(x) for x in model_query_ids],
        "result_query_ids": [int(x) for x in result_query_ids],
        "sent_ids": [int(x) for x in sent_ids],
        "labels_head": list(conv.labels[:8]),
        "n_labels": len(conv.labels),
    }
    (OUT / "meta.json").write_text(json.dumps(meta_all, ensure_ascii=False, indent=2))
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
