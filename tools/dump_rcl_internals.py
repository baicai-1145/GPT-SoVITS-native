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
def loadb(p): return torch.from_numpy(np.fromfile(str(p)+".bin" if not str(p).endswith(".bin") else str(p),dtype=np.float32).copy())
g = Golden(); vm = g.tts.vits_model
def save(name,t):
    t=t.detach().float().cpu().contiguous(); t.numpy().tofile(OUT/f"{name}.bin")
    (OUT/f"{name}.shape").write_text(" ".join(map(str,t.shape)))
rcl = vm.flow.flows[6]  # 最后一个 RCL (reverse 时第一个处理)
with torch.no_grad():
    z_p = loadb(str(OUT/"f_z_p.bin")).view(1,192,156)
    z_pf = torch.flip(z_p, [1])
    x0, x1 = torch.split(z_pf, [96]*2, 1)
    h = rcl.pre(x0) * 1.0
    save("r_flip", z_pf)
    save("r_pre_out", h)
    # WN 手动展开
    gcond = rcl.enc.cond_layer(torch.tensor([])) if False else None
    ge = loadb(str(FIX/"inputs/ge.bin")).view(1,1024,1)
    gg = rcl.enc.cond_layer(ge); save("r_cond", gg)
    x = h.clone(); output = torch.zeros_like(x)
    for i in range(len(rcl.enc.in_layers)):
        xin = rcl.enc.in_layers[i](x)
        gl = gg[:, i*384:(i+1)*384, :]
        in_act = xin + gl
        acts = torch.tanh(in_act[:, :192]) * torch.sigmoid(in_act[:, 192:])
        rs = rcl.enc.res_skip_layers[i](acts)
        if i < 3:
            x = (x + rs[:, :192]) * 1.0
            output = output + rs[:, 192:]
        else:
            output = output + rs
    save("r_enc_out", output*1.0)
    stats = rcl.post(output) * 1.0
    save("r_post_out", stats)
    x1n = x1 - stats
    save("r_x1_new", x1n)
print("saved rcl internals")

# 追加: torch WN 原生输出 vs 手动展开
import numpy as np
with torch.no_grad():
    z_pf = torch.flip(z_p,[1]); x0,_ = torch.split(z_pf,[96]*2,1)
    h = rcl.pre(x0)
    wn_native = rcl.enc(h, torch.ones(1,1,h.shape[-1]), g=ge)
    save("r_wn_native", wn_native)
    # 逐层中间
    x=h.clone(); output=torch.zeros_like(x)
    for i in range(4):
        xin=rcl.enc.in_layers[i](x); save(f"rw_{i}_xin",xin)
        gl=gg[:,i*384:(i+1)*384,:]
        in_act=xin+gl
        acts=torch.tanh(in_act[:,:192])*torch.sigmoid(in_act[:,192:])
        save(f"rw_{i}_acts",acts)
        rs=rcl.enc.res_skip_layers[i](acts); save(f"rw_{i}_rs",rs)
        if i<3:
            x=(x+rs[:,:192]); output=output+rs[:,192:]
        else:
            output=output+rs
        save(f"rw_{i}_out",output if i==3 else output)
print("appended wn internals")
