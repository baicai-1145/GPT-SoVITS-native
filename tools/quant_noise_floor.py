#!/usr/bin/env python3
"""quant_noise_floor.py — 量化噪声采样差异对 mel_rel 的贡献 (G3 可行性)

用 fixture 抓到的 m_p/logs_p/ge/ge_text/codes/phones,
分别以 (a) fixture 原始 noise, (b) 新随机 noise 走 flow+dec,
比较两段 wav 的 mel_rel — 即 native 复现时可期望的地板。
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/Volumes/2T/GPT-SoVITS-native")
CPUFAST = Path("/Volumes/2T/GPT-SoVITS-CPUFast")
sys.path.insert(0, str(CPUFAST))
sys.path.insert(0, str(CPUFAST / "GPT_SoVITS"))
import os
os.chdir(CPUFAST)

STEM = "vo_HTLQ001_3_hutao_16__s0"
F = REPO / "tests/golden/sovits_fixtures" / STEM


def loadb(p, shape):
    return torch.from_numpy(np.fromfile(p, dtype=np.float32).copy()).view(shape)


meta = json.load(open(F / "meta.json"))
pair = torch.load(REPO / "tests/golden/pairs" / f"{STEM}.pt", weights_only=False)

# 从 Golden 单例拿模型 (加载一次)
from golden_export import Golden  # noqa: E402

g = Golden()
vm = g.tts.vits_model

codes = torch.from_numpy(np.fromfile(F / "inputs/codes.bin", dtype=np.int64).copy()
                         ).view(1, 1, -1)
phones = torch.from_numpy(np.fromfile(F / "inputs/phones.bin", dtype=np.int64).copy()
                          ).view(1, -1)
ge = loadb(F / "inputs/ge.bin", (1, 1024, 1))
ge_text = loadb(F / "inputs/ge_text.bin", (1, 512, 1))
proj_out = loadb(F / "hooks/h_encp_proj.bin",
                 tuple(meta["hooks"]["h_encp_proj"]))
m_p, logs_p = proj_out[:, :192, :], proj_out[:, 192:, :]
noise_a = loadb(F / "inputs/noise.bin", tuple(m_p.shape))

print("m_p", tuple(m_p.shape), "ge", tuple(ge.shape))


def synth(noise):
    z_p = m_p + noise * torch.exp(logs_p) * 0.5
    y_lengths = torch.tensor([codes.shape[-1] * 2])
    y_mask = torch.ones(1, 1, z_p.shape[-1])
    z = vm.flow(z_p, y_mask, g=ge, reverse=True)
    o = vm.dec(z * y_mask, g=ge)
    return o.detach()[0, 0]


wav_a = synth(noise_a)
out = {"a": wav_a}
gw_full = pair["wav"]
n = min(wav_a.numel(), gw_full.numel())
torch.save({"a": wav_a, "golden": gw_full[:n]}, F / "noise_floor_wavs.pt")
print("saved", F / "noise_floor_wavs.pt")

for trial in range(3):
    noise_b = torch.randn_like(m_p)
    out[f"b{trial}"] = synth(noise_b)

# 参照: 同 noise 但 flow/dec 数值扰动 (模拟 fp32 归约差异量级)
out["c"] = synth(noise_a * (1 + 1e-6))
torch.save(out, F / "noise_floor_wavs.pt")
print("wavs:", {k: tuple(v.shape) for k, v in out.items()})
print("mel 比较用: /opt/homebrew/bin/python3 tools/mel_compare.py", F / "noise_floor_wavs.pt")
