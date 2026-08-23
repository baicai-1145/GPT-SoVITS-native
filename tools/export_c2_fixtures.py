#!/usr/bin/env python3
"""export_c2_fixtures.py — C2 验收 fixture: 条件链 (spec/ref_enc/ge/ge_text).

口径 = CPUFast TTS.py._get_ref_spec + module/models.py.build_decode_condition。
为绕开重采样器差异(PyAV vs C++ sinc), 输入用确定性合成 32 kHz 信号:
    x = 0.9·sin(2π·440t) + 0.6·sin(2π·1320t) + 0.15·sin(2π·3000t), ~1.69s
经 _get_ref_spec 同款归一(maxx>1 时 /=min(2,maxx))后导出:
    audio32k.bin    归一后的 32k 音频   → C++ ConditionBuilder::spectrogram() 直接输入
    spec.bin        torch spectrogram_torch [1025,T]
    ref_enc_out.bin MelStyleEncoder 主干输出 w [1024] (sv 相加/PLU 前)
    ge.bin          prelu(ref_enc + sv_proj(sv_emb)) [1024]
    ge_text.bin     ge_to512(ge) [512]
sv_emb 取自 golden refs bundle (1,20480), 与验收链一致。

用法: GPTSoVits env python tools/export_c2_fixtures.py
输出: tests/local/c2_fixtures/cond/{*.bin,*.shape,meta.json} (本地, 不入库)
"""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

CPUFAST = Path("/Volumes/2T/GPT-SoVITS-CPUFast")
sys.path.insert(0, str(CPUFAST))
sys.path.insert(0, str(CPUFAST / "GPT_SoVITS"))
os.chdir(CPUFAST)

REPO = Path("/Volumes/2T/wt-gsv/AR")
OUTROOT = REPO / "tests" / "local" / "c2_fixtures"
REFS = REPO / "tests" / "golden" / "refs"


def save_f32(d: Path, name: str, t):
    t = t.detach().float().cpu().contiguous()
    t.numpy().tofile(d / f"{name}.bin")
    (d / f"{name}.shape").write_text(" ".join(map(str, t.shape)))


def main():
    from TTS_infer_pack.TTS import TTS, TTS_Config  # noqa: E402
    from module.mel_processing import spectrogram_torch  # noqa: E402

    cfg = TTS_Config({
        "custom": {
            "bert_base_path": str(REPO / "pretrained_models/chinese-roberta-wwm-ext-large"),
            "cnhuhbert_base_path": str(REPO / "pretrained_models/chinese-hubert-base"),
            "device": "cpu",
            "is_half": False,
            "version": "v2ProPlus",
            "t2s_weights_path": str(REPO / "pretrained_models/s1v3.ckpt"),
            "vits_weights_path": str(REPO / "pretrained_models/v2Pro/s2Gv2ProPlus.pth"),
        }
    })
    tts = TTS(cfg)
    sr = tts.configs.sampling_rate
    assert sr == 32000 and not tts.configs.is_half

    # ---- 合成 32k 音频并归一 (_get_ref_spec 前半段语义) ----
    n = 54000
    t = torch.arange(n, dtype=torch.float64) / sr
    audio = (0.9 * torch.sin(2 * math.pi * 440.0 * t)
             + 0.6 * torch.sin(2 * math.pi * 1320.0 * t)
             + 0.15 * torch.sin(2 * math.pi * 3000.0 * t)).float().unsqueeze(0)
    maxx = audio.abs().max()
    if maxx > 1:
        audio = audio / min(2, float(maxx))
    audio32k = audio[0]  # [N]

    # ---- spec ----
    spec = spectrogram_torch(audio32k.unsqueeze(0), 2048, sr, 640, 2048,
                             center=False)  # [1,1025,T]
    print("spec:", tuple(spec.shape))

    # ---- sv_emb: golden refs bundle ----
    ref_path = sorted(REFS.glob("*.pt"))[0]
    ref = torch.load(str(ref_path), map_location="cpu", weights_only=False)
    sv_emb = ref["sv_emb"].reshape(1, -1).float()
    print("sv_emb:", tuple(sv_emb.shape), "from", ref_path.name)

    # ---- build_decode_condition ----
    vits = tts.vits_model.eval()
    with torch.no_grad():
        ge, ge_text = vits.build_decode_condition(spec.float(), sv_emb)
        # ref_enc 主干输出 (sv 相加前): get_ge 前半段
        refer_lengths = torch.LongTensor([spec.size(2)])
        from module.commons import sequence_mask  # noqa: E402
        refer_mask = torch.unsqueeze(
            sequence_mask(refer_lengths, spec.size(2)), 1).to(spec.dtype)
        w = vits.ref_enc(spec[:, :704] * refer_mask)
    print("ge:", tuple(ge.shape), "ge_text:", tuple(ge_text.shape),
          "w:", tuple(w.shape))

    d = OUTROOT / "cond"
    d.mkdir(parents=True, exist_ok=True)
    save_f32(d, "audio32k", audio32k)
    save_f32(d, "spec", spec[0])
    save_f32(d, "ref_enc_out", w.reshape(-1))
    save_f32(d, "ge", ge.reshape(-1))
    save_f32(d, "ge_text", ge_text.reshape(-1))
    meta = {"sr": sr, "n_fft": 2048, "hop": 640, "win": 2048, "center": False,
            "sv_emb_src": ref_path.name}
    (d / "meta.json").write_text(json.dumps(meta, indent=2))
    print("ALL DONE ->", d)


if __name__ == "__main__":
    main()
