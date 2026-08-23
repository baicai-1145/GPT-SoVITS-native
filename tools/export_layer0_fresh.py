#!/usr/bin/env python3
"""export_layer0_fresh.py — 用 golden_export.augment 同款钩子重新抓取自洽快照

背景: 现有 pairs/*.pt 内的 layers_prefill 与其自带输入(phones_ids/prompt_tokens/
bert_feat_1024)数值上不自洽(官方 block0 复算 cos 仅 ~0.85, 且键名与任何已提交
脚本版本不符)。本脚本按文档规定的抓取点重新跑一次真实管线, 产出:
  {phones_ids, prompt_tokens, bert_feat_1024, layers_prefill[24], tokens, ...}
写入 tests/regen/<stem>.fresh.pt (不覆盖决策者数据)。
"""
import os
import sys
from pathlib import Path

import torch

REPO = Path("/Volumes/2T/GPT-SoVITS-native")
CPUFAST = Path("/Volumes/2T/GPT-SoVITS-CPUFast")
sys.path.insert(0, str(CPUFAST))
sys.path.insert(0, str(CPUFAST / "GPT_SoVITS"))
os.chdir(CPUFAST)
sys.path.insert(0, str(REPO))

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "gsv_golden_export", REPO / "tools" / "golden_export.py")
_mod = importlib.util.module_from_spec(_spec)
sys.modules["gsv_golden_export"] = _mod
_spec.loader.exec_module(_mod)
Golden = _mod.Golden

OUT_DIR = Path("/Volumes/2T/wt-gsv/A/tests/regen")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    g = Golden()
    tts = g.tts
    dec = tts.t2s_model.model

    state = {}
    hooks = []
    hooks.append(dec.ar_text_embedding.register_forward_pre_hook(
        lambda m, i: state.setdefault("phones_ids", i[0].detach().cpu())))

    def _aud(m, i):
        state["audio_tok_last"] = i[0].detach().cpu()
        state.setdefault("prompt_tokens", i[0].detach().cpu())
    hooks.append(dec.ar_audio_embedding.register_forward_pre_hook(_aud))
    hooks.append(dec.bert_proj.register_forward_pre_hook(
        lambda m, i: state.__setitem__("bert_in_1024", i[0].detach().float().cpu())))

    blocks = dec.t2s_transformer.blocks
    nl = len(blocks)
    origs = []
    for i, blk in enumerate(blocks):
        op, od = blk.process_prompt, blk.decode_next_token

        def wp(orig, bi=0):
            def f(*a, **k):
                out = orig(*a, **k)
                if f"pre_L{bi}" not in state:
                    o = out[0] if isinstance(out, (tuple, list)) else out
                    state[f"pre_L{bi}"] = o.detach().float().cpu()
                return out
            return f

        def wd(orig, bi=0):
            def f(*a, **k):
                out = orig(*a, **k)
                o = out[0] if isinstance(out, (tuple, list)) else out
                if f"dfirst_L{bi}" not in state:
                    state[f"dfirst_L{bi}"] = o.detach().float().cpu()
                state[f"last_L{bi}"] = o.detach().float().cpu()
                return out
            return f

        blk.process_prompt = wp(op, i)
        blk.decode_next_token = wd(od, i)
        origs.append((blk, op, od))

    # 与基线完全相同的 pair: refs[0] × sentences[0]
    wav = sorted((REPO / "test_wav").glob("*.wav"))[0]
    stem = f"{wav.stem}__s0"
    state.clear()
    rec = g.run_pair(wav, 0)

    base = torch.load(REPO / "tests/golden/pairs" / f"{stem}.pt",
                      map_location="cpu", weights_only=False)
    tok_ok = bool(torch.equal(rec["tokens"], base["tokens"]))
    print(f"[check] 重跑 tokens == 基线: {tok_ok}")

    hits = sum(1 for i in range(nl) if f"pre_L{i}" in state)
    print(f"[check] pre_L 命中 {hits}/{nl}; dfirst 命中 "
          f"{sum(1 for i in range(nl) if f'dfirst_L{i}' in state)}/{nl}")
    assert hits == nl, "process_prompt 快照不完整"

    fresh = {
        "ref_wav": wav.name, "sentence_idx": 0,
        "tokens": rec["tokens"],
        "phones_ids": state["phones_ids"],
        "prompt_tokens": state["prompt_tokens"],
        "gen_tokens_final": state["audio_tok_last"],
        "bert_feat_1024": state["bert_in_1024"],
        "layers_prefill": torch.stack([state[f"pre_L{i}"] for i in range(nl)], 0),
        "layers_laststep": torch.stack([state[f"last_L{i}"] for i in range(nl)], 0),
    }
    out_p = OUT_DIR / f"{stem}.fresh.pt"
    torch.save(fresh, out_p)
    print("written:", out_p)

    for blk, op_, od_ in origs:
        blk.process_prompt = op_
        blk.decode_next_token = od_
    for h in hooks:
        h.remove()


if __name__ == "__main__":
    main()
