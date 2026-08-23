#!/usr/bin/env python3
"""dump_enc_layer.py — 导出 encoder_ssl 第 0 层内部 (LN1 后/FFN 后/层末)"""
import sys
from pathlib import Path
import numpy as np, torch, json
REPO = Path("/Volumes/2T/GPT-SoVITS-native"); CPUFAST = Path("/Volumes/2T/GPT-SoVITS-CPUFast")
sys.path.insert(0, str(CPUFAST)); sys.path.insert(0, str(CPUFAST/"GPT_SoVITS"))
import os; os.chdir(CPUFAST)
from golden_export import Golden  # noqa
FIX = REPO/"tests/golden/sovits_fixtures/vo_HTLQ001_3_hutao_16__s0"
OUT = FIX/"mha_dbg"; OUT.mkdir(exist_ok=True)
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
    layer = enc.encoder_sql if hasattr(enc,'encoder_sql') else None
    E = enc.encoder_ssl
    L0 = E.attn_layers[0]
    ymask = torch.ones(1,1,y.shape[-1])
    attn_mask = ymask.unsqueeze(2)*ymask.unsqueeze(-1)
    x0 = y*ymask
    y_attn = L0(x0,x0,attn_mask)
    save("l0_attn_full", y_attn)
    s1 = x0+y_attn
    n1 = E.norm_layers_1[0](s1)
    save("l0_after_ln1", n1)
    ffn = E.ffn_layers[0]
    pad=(ffn.kernel_size-1)//2
    xf = torch.nn.functional.pad(n1*ymask,(pad,ffn.kernel_size-1-pad))
    h = ffn.conv_1(xf); h=torch.relu(h)
    save("l0_ffn_h", h)
    xf2 = torch.nn.functional.pad(h*ymask,(pad,ffn.kernel_size-1-pad))
    o = ffn.conv_2(xf2)*ymask
    save("l0_ffn_out", o)
    s2 = n1+o
    n2 = E.norm_layers_2[0](s2)
    save("l0_after_ln2", n2)
print("saved l0 tensors")
