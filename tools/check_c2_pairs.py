#!/usr/bin/env python3
"""check_c2_pairs.py — C2 验收 (b) 对照: CLI JSON vs tests/golden/pairs/*.pt.

对照项: phones_ids(逐位), tokens(逐位/差异位数), prompt_len。
用法: GPTSoVits env python tools/check_c2_pairs.py <cli.json> <ref_wav_stem>
"""
import json
import sys
from pathlib import Path

import torch

REPO = Path("/Volumes/2T/wt-gsv/AR")
PAIRS = REPO / "tests" / "golden" / "pairs"


PROMPT_TEXT = "原来你也玩原神。"


def prompt_phone_ids():
    """提示文本的 symbols2 ids (CPUFast clean_text 链, 与 golden 前缀对齐)。"""
    import os
    cpufast = "/Volumes/2T/GPT-SoVITS-CPUFast"
    sys.path.insert(0, cpufast)
    sys.path.insert(0, os.path.join(cpufast, "GPT_SoVITS"))
    os.chdir(cpufast)
    from text.cleaner import clean_text  # noqa: E402
    from text import cleaned_text_to_sequence  # noqa: E402
    p, _, _ = clean_text(PROMPT_TEXT, "zh", "v2ProPlus")
    return cleaned_text_to_sequence(p, "v2ProPlus")


def main():
    cli = json.load(open(sys.argv[1]))
    stem = sys.argv[2]
    pp = prompt_phone_ids()
    ok_all = True
    for run in cli["runs"]:
        si = run["sentence_idx"]
        f = PAIRS / f"{stem}__s{si}.pt"
        if not f.exists():
            print(f"[SKIP] s{si}: {f.name} 不存在")
            continue
        ref = torch.load(str(f), map_location="cpu", weights_only=False)
        if "phones_ids" not in ref or "tokens" not in ref:
            print(f"[SKIP] s{si}: pairs 缺 phones_ids/tokens (导出中断)")
            continue
        g_phones = ref["phones_ids"].reshape(-1).tolist()
        g_tokens = ref["tokens"].reshape(-1).tolist()  # 惩罚后 argmax 口径
        # AR 输入 phones = 提示前缀 + 段 phones
        c_phones = pp + list(run["phones"])
        c_tokens = run.get("raw_tokens", [])
        phones_ok = g_phones == c_phones
        tok_exact = g_tokens == c_tokens
        n_diff = sum(1 for a, b in zip(g_tokens, c_tokens) if a != b) \
            if len(g_tokens) == len(c_tokens) else -1
        # 提示语义码对照 (重采样差异直接量化)
        pc_ok, pc_note = None, ""
        if "prompt_tokens" in ref and "prompt_codes" in run:
            g_pc = ref["prompt_tokens"].reshape(-1).tolist()
            c_pc = run["prompt_codes"]
            same = g_pc == c_pc
            nd = sum(1 for a, b in zip(g_pc, c_pc) if a != b) \
                if len(g_pc) == len(c_pc) else -1
            pc_ok = same
            pc_note = f" | prompt_codes {'EXACT' if same else f'DIFF({nd}/{len(g_pc)})'}"
        print(f"s{si}: phones {'EXACT' if phones_ok else 'DIFF'} "
              f"(n={len(g_phones)} vs {len(c_phones)}) | "
              f"tokens {'EXACT' if tok_exact else f'DIFF({n_diff}/{len(g_tokens)})'} "
              f"(n={len(g_tokens)} vs {len(c_tokens)}) | eos={run['hit_eos']}"
              + pc_note)
        ok_all &= phones_ok and tok_exact and (pc_ok is not False)
    print("== ALL EXACT ==" if ok_all else "== 有差异(见上) ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())
