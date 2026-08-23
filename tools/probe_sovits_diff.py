#!/usr/bin/env python3
"""probe_sovits_diff.py — 定位重放与 golden wav 差异来源"""
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/Volumes/2T/GPT-SoVITS-native")
CPUFAST = Path("/Volumes/2T/GPT-SoVITS-CPUFast")
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(CPUFAST))
sys.path.insert(0, str(CPUFAST / "GPT_SoVITS"))
import os
os.chdir(CPUFAST)

from golden_export import Golden, PROMPT_TEXT  # noqa: E402

g = Golden()
tts = g.tts
vm = tts.vits_model
wav = REPO / "test_wav" / "vo_HTLQ001_3_hutao_16.wav"
pair = torch.load(REPO / "tests/golden/pairs/vo_HTLQ001_3_hutao_16__s0.pt",
                  weights_only=False)

cap = {}
orig_prep = vm.prepare_decode_latent
orig_noise = vm._sample_decode_noise_like


def prep_wrap(codes, text, refer, **kw):
    cap["codes"] = codes.detach().clone()
    cap["text"] = text.detach().clone()
    cap["ge"] = kw["ge"].detach().clone()
    cap["ge_text"] = kw["ge_text"].detach().clone()
    return orig_prep(codes, text, refer, **kw)


def noise_wrap(target, y_lengths, sequential=False):
    r = orig_noise(target, y_lengths, sequential=sequential)
    cap["noise"] = r.detach().clone()
    return r


vm.prepare_decode_latent = prep_wrap
vm._sample_decode_noise_like = noise_wrap


def run_once(tag):
    cap.clear()
    out = tts.run({
        "text": pair["sentence"], "text_lang": "zh",
        "ref_audio_path": str(wav), "aux_ref_audio_paths": [],
        "prompt_text": PROMPT_TEXT, "prompt_lang": "zh",
        "top_k": 1, "top_p": 1, "temperature": 1.0,
        "text_split_method": "cut0", "batch_size": 1, "split_bucket": False,
        "parallel_infer": False, "vits_parallel_infer": False,
        "speed_factor": 1.0, "seed": 42,
        "streaming_mode": False, "return_fragment": False,
    })
    chunk = None
    for chunk in out:
        pass
    sr, audio = chunk
    a = np.asarray(audio)
    print(f"[{tag}] wav len={a.shape} sum={a.astype(np.int64).sum()} "
          f"codes_sum={int(cap['codes'].sum())} noise[0,:4]={cap['noise'].flatten()[:4]} "
          f"ge_norm={float(cap['ge'].norm()):.6f} getext_norm={float(cap['ge_text'].norm()):.6f}")
    return a, dict(cap)


try:
    # 先按 golden_export 顺序: export_refs 全部(模拟原始导出时的缓存状态与RNG历史)
    print("=== export_refs (全部) ===")
    g.export_refs(REPO / "tests/golden/sovits_fixtures/_probe_refs")

    a1, c1 = run_once("run#1 after full export_refs")
    a2, c2 = run_once("run#2 repeat")
    same12 = a1.shape == a2.shape and np.array_equal(a1, a2)
    print("run1 == run2 ?", same12)

    gw = pair["wav"].numpy()
    n = min(len(a1), len(gw))
    d1 = np.abs(a1[:n].astype(np.float64) - gw[:n].astype(np.float64)).max()
    print(f"run1 vs golden maxdiff={d1}")

    # 对照 fixture (第一次导出的抓取)
    F = REPO / "tests/golden/sovits_fixtures/vo_HTLQ001_3_hutao_16__s0/inputs"
    fx_codes = torch.from_numpy(np.fromfile(F / "codes.bin", dtype=np.int64))
    fx_ge = torch.from_numpy(np.fromfile(F / "ge.bin", dtype=np.float32).copy())
    fx_noise = torch.from_numpy(np.fromfile(F / "noise.bin", dtype=np.float32).copy())
    print("fixture codes == run1 codes:", torch.equal(fx_codes.view(c1["codes"].shape), c1["codes"]))
    print("fixture ge == run1 ge:", bool(torch.equal(fx_ge.view_as(c1['ge']), c1['ge'])),
          float((fx_ge.view_as(c1['ge']) - c1['ge']).abs().max()))
    print("fixture noise == run1 noise:", bool(torch.equal(fx_noise.view_as(c1['noise']), c1['noise'])),
          float((fx_noise.view_as(c1['noise']) - c1['noise']).abs().max()))
finally:
    vm.prepare_decode_latent = orig_prep
    vm._sample_decode_noise_like = orig_noise
