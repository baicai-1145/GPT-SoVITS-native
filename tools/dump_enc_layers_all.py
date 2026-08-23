#!/usr/bin/env python3
"""导出 encoder_ssl 全部 3 层输出"""
import sys
from pathlib import Path
import numpy as np, torch, json
REPO = Path("/Volumes/2T/GPT-SoVITS-native"); CPUFAST = Path("/Volumes/2T/GPT-SoVITS-CPUFast")
sys.path.insert(0, str(CPUFAST)); sys.path.insert(0, str(CPUFAST/"GPT_SoVITS"))
import os; os.chdir(CPUFAST)
from golden_export import Golden  # noqa
FIX = REPO/"tests/golden/sovits_fixtures/vo_HTLQ001_3_hutao_16__s0"
OUT = FIX/"mha_dbg"
def loadb(p, shape): return torch.from_numpy(np.fromfile(str(p),dtype=np.float32).copy()).view(shape)
g = Golden(); enc = g.tts.vits_model.enc_p
def save(name,t):
    t=t.detach().float().cpu().contiguous(); t.numpy().tofile(OUT/f"{name}.bin")
    (OUT/f"{name}.shape").write_text(" ".join(map(str,t.shape)))
meta=json.load(open(FIX/"meta.json"))
x_in=loadb(FIX/"hooks/h_encp_input.bin",tuple(meta["hooks"]["h_encp_input"]))
with torch.no_grad():
    m=torch.ones(1,1,x_in.shape[-1])
    y=enc.ssl_proj(x_in*m)*m
    E=enc.encoder_ssl
    for i in range(len(E.attn_layers)):
        L=E.attn_layers[i]; n1m=E.norm_layers_1[i]; ffn=E.ffn_layers[i]; n2m=E.norm_layers_2[i]
        am=m.unsqueeze(2)*m.unsqueeze(-1)
        xa=y*m
        ya=L(xa,xa,am)
        x=n1m(xa+ya)
        pad=(ffn.kernel_size-1)//2
        h=torch.relu(ffn.conv_1(torch.nn.functional.pad(x*m,(pad,ffn.kernel_size-1-pad))))
        xo=ffn.conv_2(torch.nn.functional.pad(h*m,(pad,ffn.kernel_size-1-pad)))*m
        y=n2m(x+xo)
        save(f"es_L{i}_out", y)
    print("saved es_L{0,1,2}_out")
