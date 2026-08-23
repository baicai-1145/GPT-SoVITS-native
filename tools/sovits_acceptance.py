#!/opt/homebrew/bin/python3/bin/python3
"""sovits_acceptance.py — B3/B4 DoD 验收 (G1 中间张量 + G3 音频)

用法: python3 sovits_acceptance.py <fixtures_root> <cpp_dump_root> [--wav-only]
对每个 fixture pair:
  - 对照 hooks/*.bin vs cpp_dump/*.bin (cos>=0.9999, max_rel<=1e-3 → G1)
  - C++ wav(×32768) vs torch h_dec(×32768) mel_rel <= 0.05 → G3
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("fixroot")
ap.add_argument("dumproot")
ap.add_argument("--wav-only", action="store_true")
args = ap.parse_args()

FIXROOT, DPR = Path(args.fixroot), Path(args.dumproot)

# 中间量对照表: (torch hook 名, cpp dump 名, transpose 处理)
HOOK_PAIRS = [
    ("h_ssl_proj", "dbg_ssl_proj_out", None),
    ("h_encoder_ssl", "dbg_encoder_ssl_out", None),
    ("h_text_emb", "dbg_text_emb_t", "T"),
    ("h_encoder_text", "dbg_encoder_text_out", None),
    ("h_mrte", "dbg_mrte_out", None),
    ("h_encoder2", "dbg_encoder2_out", None),
    ("h_encp_proj", "h_encp_proj", None),
    ("h_flow_in", "h_flow_in", None),
    ("h_flow", "h_flow", None),
]
DEC_HOOKS = [(f"h_dec_up{i}", f"dbg_dec_up{i}") for i in range(5)]
FINAL = ("h_dec", "h_dec")


def load(p: Path) -> np.ndarray:
    return np.fromfile(str(p) + ".bin", dtype=np.float32)


def g1(a, b):
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    rel = float(np.abs(a - b).max()) / max(1e-9, float(np.abs(b).max()))
    return cos, rel


def torch_from(a):
    import torch
    return torch.from_numpy(a.copy())


def mel_rel(a16, b16, sr=32000):
    import torchaudio
    m = torchaudio.transforms.MelSpectrogram(
        sr, n_fft=2048, win_length=2048, hop_length=640,
        n_mels=128, f_min=0.0, f_max=None, power=2.0)
    ma, mb = m(a16 / 32768.0), m(b16 / 32768.0)
    t = min(ma.shape[-1], mb.shape[-1])
    l1 = float((ma[..., :t] - mb[..., :t]).abs().mean())
    return l1 / float(mb.mean().clamp_min(1e-9))


pairs = sorted(d for d in FIXROOT.iterdir() if d.is_dir())
total_fail = []
for pd in pairs:
    stem = pd.name
    dd = DPR / stem
    if not dd.exists():
        print(f"[{stem}] MISSING cpp dumps at {dd}")
        total_fail.append(stem)
        continue
    fails = []
    if not args.wav_only:
        for tn, cn, tr in [(a,b,c) for a,b,c in HOOK_PAIRS] + [(a,b,None) for a,b in DEC_HOOKS]:  # fix: DEC_HOOKS 是二元组
            try:
                t, c = load(pd / "hooks" / tn), load(dd / cn)
            except FileNotFoundError as e:
                print(f"[{stem}] {tn}: missing ({e})")
                fails.append(tn)
                continue
            if tr == "T":
                shape = tuple(int(x) for x in open(str(pd / "hooks" / tn) + ".shape").read().split())
                t = t.reshape(shape).transpose(0, 2, 1).reshape(-1)
            if t.size != c.size:
                print(f"[{stem}] {tn}: SIZE {t.size} vs {c.size}")
                fails.append(tn)
                continue
            cos, rel = g1(t, c)
            ok = cos >= 0.9999 and rel <= 1e-3
            if not ok:
                fails.append(tn)
                print(f"[{stem}] {tn:<16} cos={cos:.6f} rel={rel:.3e} FAIL")
    # G3 音频门
    try:
        c_wav = load(dd / FINAL[1])
        t_wav = load(pd / "hooks" / FINAL[0])
        n = min(c_wav.size, t_wav.size)
        mr = mel_rel(torch_from(c_wav[:n]), torch_from(t_wav[:n]))
        ok = mr <= 0.05
        print(f"[{stem}] G3 mel_rel={mr:.6f} {'PASS' if ok else 'FAIL'} "
              f"(n={n})")
        if not ok:
            fails.append("G3_mel")
    except Exception as e:
        print(f"[{stem}] G3 error: {e}")
        fails.append("G3_err")
    status = "PASS" if not fails else "FAIL " + ",".join(fails)
    print(f"== {stem}: {status}\n")
    if fails:
        total_fail.append(stem)


print("\n==== SoVITS B3/B4 acceptance:",
      "ALL PASS ✅" if not total_fail else f"{len(total_fail)} pairs FAIL: {total_fail}")
sys.exit(0 if not total_fail else 1)
