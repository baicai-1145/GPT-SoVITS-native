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
g = Golden(); vm = g.tts.vits_model
def save(name,t):
    t=t.detach().float().cpu().contiguous(); t.numpy().tofile(OUT/f"{name}.bin")
    (OUT/f"{name}.shape").write_text(" ".join(map(str,t.shape)))
meta=json.load(open(FIX/"meta.json"))
codes=loadb(str(FIX/"inputs/codes.bin")).view(-1).long()
phones=loadb(str(FIX/"inputs/phones.bin")).view(1,-1).long()
ge=loadb(str(FIX/"inputs/ge.bin")).view(1,1024,1)
gt=loadb(str(FIX/"inputs/ge_text.bin")).view(1,512,1)
noise=loadb(str(FIX/"inputs/noise.bin")).view(1,192,-1)
caps={}
def mk(name):
    def h(mod,inp,out):
        o = out[0] if isinstance(out,tuple) else out
        caps[name]=o.detach().float().cpu().clone()
    return h
hs=[]
# 手动复刻 prepare_decode_latent 数值路径 (无随机)
with torch.no_grad():
    quantized = vm.quantizer.decode(codes.view(1,1,-1))
    q2 = torch.nn.functional.interpolate(quantized, size=int(quantized.shape[-1]*2), mode="nearest")
    y_lengths=torch.tensor([q2.shape[-1]]); text_lengths=torch.tensor([11])
    # enc_p
    x, m_p, logs_p, y_mask, _, _ = vm.enc_p(q2, y_lengths, phones, text_lengths, gt, 1)
    save("f_m_p", m_p); save("f_logs_p", logs_p)
    z_p = m_p + noise*torch.exp(logs_p)*0.5
    save("f_z_p", z_p)
    for i,f in enumerate(vm.flow.flows):
        hs.append(f.register_forward_hook(mk(f"f_flow{i}")))
    z = vm.flow(z_p, y_mask, g=ge, reverse=True)
    for n,t in sorted(caps.items()):
        save(n,t)
    save("f_z", z)
print("saved flow internals:", sorted(caps.keys()))
