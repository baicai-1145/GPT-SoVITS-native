#!/usr/bin/env python
"""export_c1_fixtures.py — C1 中间张量 fixtures 导出(CPUFast 真实管线 hook)。

每对 ref 输出到 <out>/<stem>/:
  wav16k.f32        未补零 16k 波形(HuBERT/SV 共用输入源)
  fbank80.f32       Kaldi fbank [m,80](SV 前端 golden)
  hub_cnn.f32       HuBERT CNN 特征提取输出 [T',512]
  hub_proj.f32      feature_projection 输出 [T,768]
  hub_L0.f32        第 0 个 encoder layer 输出 [T,768]
  hub_last.f32      last_hidden_state (= refs bundle 的 hubert_hidden)
  sv_conv1.f32      SV conv1+bn1+relu 后 [64,80,T]
  sv_layer{1..4}.f32 各 stage 输出
  sv_fuse34.f32     底部融合输出 [2048,10,T']
  sv_emb.f32        (= refs bundle 的 sv_emb)
  meta.txt          各张量形状与来源

另导出 selftest_fbank/: 合成扫频信号的 wav 采样与 fbank, 供单测对照。
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/Volumes/2T/wt-gsv/AR")
sys.path.insert(0, "/Volumes/2T/wt-gsv/AR/tools")
os.chdir("/Volumes/2T/GPT-SoVITS-CPUFast")

import golden_export as ge  # noqa: E402  (会 chdir 并装载全套模型)

OUT = sys.argv[1] if len(sys.argv) > 1 else "/Volumes/2T/wt-gsv/AR/tests/local/c1_fixtures"
N_REFS = int(sys.argv[2]) if len(sys.argv) > 2 else 6

g = ge.Golden()
tts = g.tts
hub = tts.cnhuhbert_model.model
sv = tts.sv_model.embedding_model
from tools.audio_utils import load_audio_mono  # noqa: E402
import kaldi as Kaldi  # noqa: E402

cap = {}
hooks = []


def mk(name):
    def hook(mod, inp, out):
        cap[name] = out.detach().float().cpu()
    return hook


# HuBERT 内部 hook
class CapProj(torch.nn.Module):
    pass


orig_proj_fwd = hub.feature_projection.forward
def proj_fwd(x):
    # 复刻 HubertFeatureProjection.forward 以分层捕获
    h = x
    if hub.feature_projection.feat_proj_layer_norm:
        h = hub.feature_projection.layer_norm(h)
    h = hub.feature_projection.projection(h)
    cap["hub_proj"] = h.detach().float().cpu()
    return h
hub.feature_projection.forward = proj_fwd

orig_conv_fwd = hub.feature_extractor.forward
def conv_fwd(iv):
    hs = iv[:, None]
    for cl in hub.feature_extractor.conv_layers:
        hs = cl(hs)
    cap["hub_cnn"] = hs.detach().float().cpu()
    return hs
hub.feature_extractor.forward = conv_fwd

enc_layers = hub.encoder.layers
orig_l0 = enc_layers[0].forward
def l0_fwd(x):
    out = orig_l0(x)
    cap["hub_L0"] = out.detach().float().cpu()
    return out
enc_layers[0].forward = l0_fwd

# 细粒度 hook(定位用): pos_conv / enc_ln / L0 内部四段
hub.encoder.pos_conv_embed.register_forward_hook(mk("hub_posconv"))
hub.encoder.layer_norm.register_forward_hook(mk("hub_encln"))
enc_layers[0].attention.register_forward_hook(mk("hub_l0_attn"))
enc_layers[0].layer_norm.register_forward_hook(mk("hub_l0_ln1"))
enc_layers[0].feed_forward.register_forward_hook(mk("hub_l0_ffn"))
enc_layers[0].final_layer_norm.register_forward_hook(mk("hub_l0_ln2"))

# SV 逐块 hook
sv.layer1[0].register_forward_hook(mk("sv_l1b0"))
b0 = sv.layer1[0]
b0.bn1.register_forward_hook(mk("sv_b0_bn1"))
b0.convs[0].register_forward_hook(mk("sv_b0_cv0"))
b0.bns[0].register_forward_hook(mk("sv_b0_bn0"))
b0.convs[1].register_forward_hook(mk("sv_b0_cv1"))
b0.conv3.register_forward_hook(mk("sv_b0_cv3"))
b0.bn3.register_forward_hook(mk("sv_b0_bn3"))
b0.shortcut.register_forward_hook(mk("sv_b0_sc"))
sv.layer1[1].register_forward_hook(mk("sv_l1b1"))
sv.layer2[0].register_forward_hook(mk("sv_l2b0"))
for bi in range(6):
    sv.layer3[bi].register_forward_hook(mk(f"sv_l3b{bi}"))
sv.layer3[0].fuse_models[0].register_forward_hook(mk("sv_f34_0"))
_la = sv.layer3[0].fuse_models[0].local_att
def mk_in(name):
    def hook(mod, inp, out):
        cap[name] = inp[0].detach().float().cpu()
    return hook
_la[0].register_forward_hook(mk("sv_att_cv1"))
_la[0].register_forward_hook(mk_in("sv_att_in"))
_la[1].register_forward_hook(mk("sv_att_bn1"))
_la[3].register_forward_hook(mk("sv_att_cv2"))
_la[4].register_forward_hook(mk("sv_att_bn2"))

# SV 内部 hook: 逐 stage
orig_f = sv.forward3
def fwd3(x):
    xx = x.permute(0, 2, 1).unsqueeze_(1)
    o = torch.nn.functional.relu(sv.bn1(sv.conv1(xx)))
    cap["sv_conv1"] = o.detach().float().cpu()
    l1 = sv.layer1(o); cap["sv_layer1"] = l1.detach().float().cpu()
    l2 = sv.layer2(l1); cap["sv_layer2"] = l2.detach().float().cpu()
    l3 = sv.layer3(l2); cap["sv_layer3"] = l3.detach().float().cpu()
    l4 = sv.layer4(l3); cap["sv_layer4"] = l4.detach().float().cpu()
    l3ds = sv.layer3_ds(l3)
    f34 = sv.fuse34(l4, l3ds); cap["sv_fuse34"] = f34.detach().float().cpu()
    return f34.flatten(start_dim=1, end_dim=2).mean(-1)
sv.forward3 = fwd3

man = json.load(open("/Volumes/2T/wt-gsv/AR/tests/golden/manifest.json"))
refmap = {r["wav"]: r for r in man["refs"]}
wavs = sorted(refmap.keys())
# 取时长分散的 6 个: 首、尾、中 + 最长最短
sizes = [(w, os.path.getsize("/Volumes/2T/GPT-SoVITS-native/test_wav/" + w)) for w in wavs]
sizes.sort(key=lambda t: t[1])
picks = sorted({sizes[0][0], sizes[-1][0], sizes[len(sizes)//2][0],
                sizes[len(sizes)//3][0], sizes[2*len(sizes)//3][0], sizes[1][0]})
print("picked:", picks)
if os.environ.get("C1_ONLY"):
    only = os.environ["C1_ONLY"]
    picks = [only if only.endswith(".wav") else only + ".wav"]

def dump(name, t):
    t.detach().numpy().tofile(os.path.join(d, name))

for wavname in picks:
    stem = os.path.splitext(wavname)[0]
    d = os.path.join(OUT, stem)
    os.makedirs(d, exist_ok=True)
    b = torch.load("/Volumes/2T/wt-gsv/AR/tests/golden/" + refmap[wavname]["file"], weights_only=False)

    A = np.asarray(load_audio_mono("/Volumes/2T/GPT-SoVITS-native/test_wav/" + wavname, 16000))
    At = torch.from_numpy(A)
    with torch.no_grad():
        fb = Kaldi.fbank(At.unsqueeze(0), num_mel_bins=80, sample_frequency=16000, dither=0)
        hub_in = torch.cat([At, torch.zeros(9600)]).unsqueeze(0)
        last = hub(hub_in.float())["last_hidden_state"]
        emb = orig_f(fb.unsqueeze(0)) if False else sv.forward3(fb.unsqueeze(0))
    assert torch.allclose(last, b["hubert_hidden"], atol=1e-4), "hubert 捕获不一致"
    cos = torch.nn.functional.cosine_similarity(
        emb.double().flatten(), b["sv_emb"].double().flatten(), dim=0).item()
    print(f"{stem}: fb={tuple(fb.shape)} hub={tuple(last.shape)} emb={tuple(emb.shape)} sv_cos={cos:.9f}")

    dump("wav16k.f32", At)
    dump("fbank80.f32", fb)
    dump("hub_cnn.f32", cap["hub_cnn"].squeeze(0).transpose(0, 1))   # [T',512]
    dump("hub_proj.f32", cap["hub_proj"].squeeze(0))                 # [T,768]
    dump("hub_L0.f32", cap["hub_L0"].squeeze(0))
    dump("hub_last.f32", last.squeeze(0))
    dump("sv_conv1.f32", cap["sv_conv1"].squeeze(0))                 # [64,80,T]
    dump("sv_layer1.f32", cap["sv_layer1"].squeeze(0))
    dump("sv_layer2.f32", cap["sv_layer2"].squeeze(0))
    dump("sv_layer3.f32", cap["sv_layer3"].squeeze(0))
    dump("sv_layer4.f32", cap["sv_layer4"].squeeze(0))
    dump("sv_fuse34.f32", cap["sv_fuse34"].squeeze(0))
    dump("hub_posconv.f32", cap["hub_posconv"].squeeze(0))
    dump("hub_encln.f32", cap["hub_encln"].squeeze(0))
    dump("hub_l0_attn.f32", cap["hub_l0_attn"][0].squeeze(0))
    dump("hub_l0_ln1.f32", cap["hub_l0_ln1"].squeeze(0))
    dump("hub_l0_ffn.f32", cap["hub_l0_ffn"][0].squeeze(0))
    dump("hub_l0_ln2.f32", cap["hub_l0_ln2"].squeeze(0))
    dump("sv_l1b0.f32", cap["sv_l1b0"].squeeze(0))
    dump("sv_l1b1.f32", cap["sv_l1b1"].squeeze(0))
    dump("sv_l2b0.f32", cap["sv_l2b0"].squeeze(0))
    dump("sv_b0_bn1.f32", cap["sv_b0_bn1"].squeeze(0))
    dump("sv_b0_cv0.f32", cap["sv_b0_cv0"].squeeze(0))
    dump("sv_b0_bn0.f32", cap["sv_b0_bn0"].squeeze(0))
    dump("sv_b0_bn3.f32", cap["sv_b0_bn3"].squeeze(0))
    dump("sv_b0_sc.f32", cap["sv_b0_sc"].squeeze(0))
    for bi in range(6):
        dump(f"sv_l3b{bi}.f32", cap[f"sv_l3b{bi}"].squeeze(0))
    dump("sv_f34_0.f32", cap["sv_f34_0"].squeeze(0))
    dump("sv_att_cv1.f32", cap["sv_att_cv1"].squeeze(0))
    dump("sv_att_in.f32", cap["sv_att_in"].squeeze(0))
    dump("sv_att_bn1.f32", cap["sv_att_bn1"].squeeze(0))
    dump("sv_att_cv2.f32", cap["sv_att_cv2"].squeeze(0))
    dump("sv_att_bn2.f32", cap["sv_att_bn2"].squeeze(0))
    dump("sv_emb.f32", emb.reshape(-1))
    with open(os.path.join(d, "meta.txt"), "w") as f:
        f.write(json.dumps({
            "wav": wavname,
            "shapes": {k: list(cap[k].squeeze(0).shape) for k in cap},
            "fbank": list(fb.shape), "hubert": list(last.squeeze(0).shape),
            "emb": list(emb.reshape(-1).shape),
        }))
print("fixtures done →", OUT)

# ---- selftest_fbank: 合成扫频 + 白噪声, 供 fbank 单测 ----
d = os.path.join(OUT, "selftest_fbank")
os.makedirs(d, exist_ok=True)
rng = np.random.default_rng(42)
t = np.arange(32000) / 16000.0
chirp = 0.6 * np.sin(2 * np.pi * (200 + 1800 * t / 2.0) * t).astype(np.float32)
noise = (0.05 * rng.standard_normal(32000)).astype(np.float32)
sig = chirp + noise
with torch.no_grad():
    fbsig = Kaldi.fbank(torch.from_numpy(sig).unsqueeze(0), num_mel_bins=80,
                        sample_frequency=16000, dither=0)
sig.tofile(os.path.join(d, "wav.f32"))
fbsig.numpy().tofile(os.path.join(d, "fbank.f32"))
with open(os.path.join(d, "meta.txt"), "w") as f:
    f.write(f"n={len(sig)} fbank={list(fbsig.shape)}\n")
print("selftest_fbank done")
