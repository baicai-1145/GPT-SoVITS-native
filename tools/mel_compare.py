#!/usr/bin/env python3
"""mel_compare.py — golden_export._mel_l1 同款 mel 相对误差 (需 torchaudio, 用 homebrew python)

用法: /opt/homebrew/bin/python3 tools/mel_compare.py a.pt[:key] b.pt[:key] ...
  传入单个 .pt 时对其中全部 key 两两比较; 传入两个时比较二者。
张量应为 int16 刻度 float (先 /32768), 与 CALIBRATION.md 口径一致。
"""
import sys
from pathlib import Path

import torch
import torchaudio


def mel_rel(a: torch.Tensor, b: torch.Tensor, sr=32000):
    a = a / 32768.0
    b = b / 32768.0
    n = min(a.numel(), b.numel())
    a, b = a[:n].float(), b[:n].float()
    m = torchaudio.transforms.MelSpectrogram(
        sr, n_fft=2048, win_length=2048, hop_length=640,
        n_mels=128, f_min=0.0, f_max=None, power=2.0)
    ma, mb = m(a), m(b)
    t = min(ma.shape[-1], mb.shape[-1])
    l1 = float((ma[..., :t] - mb[..., :t]).abs().mean())
    base = float(mb.mean().clamp_min(1e-9))
    return l1, l1 / base


def load(spec):
    if ":" in spec:
        p, k = spec.rsplit(":", 1)
        return torch.load(p, weights_only=False)[k].flatten()
    return torch.load(spec, weights_only=False).flatten()


args = sys.argv[1:]
if len(args) == 1:
    d = torch.load(args[0], weights_only=False)
    keys = list(d.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            l1, rel = mel_rel(d[keys[i]], d[keys[j]])
            print(f"{keys[i]} vs {keys[j]}: mel_l1={l1:.6g} mel_rel={rel:.6g}")
elif len(args) == 2:
    a, b = load(args[0]), load(args[1])
    l1, rel = mel_rel(a, b)
    print(f"mel_l1={l1:.6g} mel_rel={rel:.6g}")
else:
    print(__doc__)
    sys.exit(1)
