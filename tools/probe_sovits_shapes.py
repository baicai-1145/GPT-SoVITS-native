#!/usr/bin/env python3
"""probe_sovits_shapes.py — 探查 golden 导出路径上 SoVITS 各阶段张量形状"""
import sys
from pathlib import Path

import torch

REPO = Path("/Volumes/2T/GPT-SoVITS-native")
CPUFAST = Path("/Volumes/2T/GPT-SoVITS-CPUFast")
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(CPUFAST))
sys.path.insert(0, str(CPUFAST / "GPT_SoVITS"))
import os
os.chdir(CPUFAST)

from golden_export import Golden, PROMPT_TEXT, SENTENCES  # noqa: E402

g = Golden()
tts = g.tts
vm = tts.vits_model

# 完全复现 golden_export 流程: 先 export_refs(全部), 再 run pair
wav = REPO / "test_wav" / "vo_HTLQ001_3_hutao_16.wav"

# --- export_refs 部分 ---
tts.prompt_cache["ref_audio_path"] = str(wav)
tts._set_ref_spec(str(wav))
tts._set_prompt_semantic(str(wav))
tts._get_runtime_refer_audio_spec_and_sv_emb()
ge, ge_text = tts._get_runtime_decode_condition()
print("refs: spec", tts.prompt_cache["refer_spec"][0][0].shape,
      "prompt_sem", tts.prompt_cache["prompt_semantic"].shape)

# --- hooks ---
def mk(tag):
    def h(mod, inp, out):
        t = out if isinstance(out, torch.Tensor) else out[0]
        print(f"[hook] {tag}: out {tuple(t.shape)}")
    return h

hs = []
hs.append(vm.quantizer.decode and None)  # placeholder
for tag, mod in [("enc_p", vm.enc_p), ("flow", vm.flow), ("dec", vm.dec),
                 ("quantizer", vm.quantizer)]:
    hs.append(mod.register_forward_hook(mk(tag)))

orig_prep = vm.prepare_decode_latent
orig_noise = vm._sample_decode_noise_like
orig_nvsb = None

def prep_wrap(codes, text, refer, **kw):
    print(f"[prep] codes {tuple(codes.shape)} text {tuple(text.shape)} "
          f"code_lengths={kw.get('code_lengths')} text_lengths={kw.get('text_lengths')}")
    r = orig_prep(codes, text, refer, **kw)
    print(f"[prep] z {tuple(r[0].shape)} y_mask {tuple(r[1].shape)} y_lengths={r[4]}")
    return r

def noise_wrap(target, y_lengths, sequential=False):
    print(f"[noise] target {tuple(target.shape)} y_lengths={y_lengths} seq={sequential}")
    return orig_noise(target, y_lengths, sequential=sequential)

vm.prepare_decode_latent = prep_wrap
vm._sample_decode_noise_like = noise_wrap

try:
    rec = g.run_pair(wav, 0)
    print("final wav:", rec["wav"].numel(), "tokens:", rec["tokens"].numel())
finally:
    for h in hs:
        if h is not None:
            h.remove()
    vm.prepare_decode_latent = orig_prep
    vm._sample_decode_noise_like = orig_noise
