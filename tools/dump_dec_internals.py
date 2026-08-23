#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np, torch
REPO = Path("/Volumes/2T/GPT-SoVITS-native"); CPUFAST = Path("/Volumes/2T/GPT-SoVITS-CPUFast")
sys.path.insert(0, str(CPUFAST)); sys.path.insert(0, str(CPUFAST/"GPT_SoVITS"))
import os; os.chdir(CPUFAST)
from golden_export import Golden  # noqa
FIX=Path("/Volumes/2T/GPT-SoVITS-native/tests/golden/sovits_fixtures/vo_HTLQ001_3_hutao_16__s0")
OUT=FIX
def save(name,t):
    t=t.detach().float().cpu().contiguous(); t.numpy().tofile(OUT/f"{name}.bin")
    (OUT/f"{name}.shape").write_text(" ".join(map(str,t.shape)))
g=Golden(); vm=g.tts.vits_model; dec=vm.dec
z=torch.from_numpy(np.fromfile("/tmp/sov_dump/h_flow.bin",dtype=np.float32).copy()).reshape(1,192,156)
ge=torch.from_numpy(np.fromfile(str(FIX/"inputs/ge.bin"),dtype=np.float32).copy()).view(1,1024,1)
LRELU=0.1
with torch.no_grad():
    x=dec.conv_pre(z); save("d_convpre",x)
    x=x+dec.cond(ge); save("d_after_cond",x)
    for i in range(len(dec.ups)):
        x=torch.nn.functional.leaky_relu(x,LRELU)
        x=dec.ups[i](x); save(f"d_up{i}",x)
        off=i*3
        xs=None
        for j in range(3):
            blk=dec.resblocks[off+j]
            xx=x
            for c1,c2 in zip(blk.convs1,blk.convs2):
                xx=torch.nn.functional.leaky_relu(xx,LRELU)
                xx=c1(xx)
                xx=torch.nn.functional.leaky_relu(xx,LRELU)
                xx=c2(xx)
                xx=xx+xx  # placeholder to avoid alias confusion below
            # 正确: block 内串行残差
            xx=x
            for c1,c2 in zip(blk.convs1,blk.convs2):
                t_=c1(torch.nn.functional.leaky_relu(xx,LRELU))
                t_=c2(torch.nn.functional.leaky_relu(t_,LRELU))
                xx=t_+xx
            if xs is None: xs=xx.clone()
            else: xs.add_(xx)
        x=xs.mul_(1.0/3.0)
        save(f"d_res{i}",x)
    x=torch.nn.functional.leaky_relu(x)   # 默认 slope 0.01
    x=dec.conv_post(x); save("d_post",x)
    x=torch.tanh(x); save("d_tanh",x)
print("saved dec internals")

# 单块中间量 (stage0 block0)
with torch.no_grad():
    x0 = torch.from_numpy(np.fromfile("/tmp/sov_dump/dbg_dec_up0.bin",dtype=np.float32).copy()).reshape(1,384,-1)
    blk=dec.resblocks[0]
    xx=x0
    for jj,(c1,c2) in enumerate(zip(blk.convs1,blk.convs2)):
        t_=c1(torch.nn.functional.leaky_relu(xx,LRELU)); save(f"rb0_c1_{jj}",t_)
        t_=c2(torch.nn.functional.leaky_relu(t_,LRELU)); save(f"rb0_c2_{jj}",t_)
        xx=t_+xx; save(f"rb0_after_{jj}",xx)
    save("rb0_final",xx)
print("appended rb0 internals")
