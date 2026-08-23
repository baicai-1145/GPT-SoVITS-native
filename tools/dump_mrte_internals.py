#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np, torch, json
REPO = Path("/Volumes/2T/GPT-SoVITS-native"); CPUFAST = Path("/Volumes/2T/GPT-SoVITS-CPUFast")
sys.path.insert(0, str(CPUFAST)); sys.path.insert(0, str(CPUFAST/"GPT_SoVITS"))
import os; os.chdir(CPUFAST)
from golden_export import Golden  # noqa
FIX = REPO/"tests/golden/sovits_fixtures/vo_HTLQ001_3_hutao_16__s0"
OUT = FIX/"mha_dbg"
def loadb(p): return torch.from_numpy(np.fromfile(str(p),dtype=np.float32).copy())
g = Golden(); enc = g.tts.vits_model.enc_p
def save(name,t):
    t=t.detach().float().cpu().contiguous(); t.numpy().tofile(OUT/f"{name}.bin")
    (OUT/f"{name}.shape").write_text(" ".join(map(str,t.shape)))
meta=json.load(open(FIX/"meta.json"))
x_in=loadb(str(FIX/"hooks/h_encp_input.bin")).view(tuple(meta["hooks"]["h_encp_input"]))
gt=loadb(str(FIX/"hooks/h_encp_ge.bin")).view(tuple(meta["hooks"]["h_encp_ge"]))
with torch.no_grad():
    m=torch.ones(1,1,x_in.shape[-1])
    y=enc.ssl_proj(x_in*m)*m
    y=enc.encoder_ssl(y*m,m)
    tm=torch.ones(1,1,11)
    temb=enc.text_embedding(torch.tensor([[3,227,167,158,119,1,251,214,221,194,3]])).transpose(1,2)
    tenc=enc.encoder_text(temb*tm,tm)
    save("mte_ssl512", enc.mrte.c_pre(y*m))
    save("mte_text512", enc.mrte.text_pre(tenc*tm))
    cross=enc.mrte.cross_attention(enc.mrte.c_pre(y*m)*m, enc.mrte.text_pre(tenc*tm), tm.unsqueeze(2)*m.unsqueeze(-1))
    save("mte_cross", cross)
    x=enc.mrte.c_pre(y*m)+0  # alias
    tot=cross+enc.mrte.c_pre(y*m)+gt
    save("mte_sum", tot)
print("saved mrte internals")
