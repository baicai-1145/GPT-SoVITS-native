#!/usr/bin/env python3
"""export_sovits_fixtures.py — B3/B4 SoVITS golden 输入束 + 中间张量导出

对选定 pair 重放 CPUFast 推理, 抓:
  inputs/   : codes / phones_ids / ge / ge_text / noise (C++ 端到端直连的输入束)
  hooks/    : quantizer/enc_p/flow/dec 各代表张量的 torch hook golden (f32 raw bin)
  meta.json : 每个张量的形状与来源

用法:
  python3 tools/export_sovits_fixtures.py --pairs <name> [--pairs <name> ...]
  name 形如 vo_HTLQ001_3_hutao_16__s0 (tests/golden/pairs/<name>.pt)

输出目录: tests/golden/sovits_fixtures/<pair_name>/...
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path("/Volumes/2T/GPT-SoVITS-native")
CPUFAST = Path("/Volumes/2T/GPT-SoVITS-CPUFast")
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(CPUFAST))
sys.path.insert(0, str(CPUFAST / "GPT_SoVITS"))
import os
os.chdir(CPUFAST)

from golden_export import Golden, PROMPT_TEXT, SENTENCES  # noqa: E402

OUTROOT = REPO / "tests/golden/sovits_fixtures"


def save_bin(outdir: Path, name: str, t: torch.Tensor):
    """f32 raw bin + .shape 文本(空格分隔维度)"""
    t = t.detach().float().cpu().contiguous()
    t.numpy().tofile(outdir / f"{name}.bin")
    (outdir / f"{name}.shape").write_text(" ".join(str(d) for d in t.shape))


def export_pair(g: Golden, stem: str, outdir: Path, sentence: str, seed: int):
    # stem 形如 <ref_wav_stem>__s<idx>; 元数据来自 manifest 参数(不再依赖历史 pairs)
    ref_stem, _, sidx = stem.rpartition("__s")
    ref_wav = ref_stem + ".wav"
    wav_path = REPO / "test_wav" / ref_wav

    tts = g.tts
    vm = tts.vits_model
    cap = {}
    idir = outdir / "inputs"
    hdir = outdir / "hooks"
    idir.mkdir(parents=True, exist_ok=True)
    hdir.mkdir(parents=True, exist_ok=True)

    # ---- 输入束抓取: 包 prepare_decode_latent 与 _sample_decode_noise_like ----
    orig_prep = vm.prepare_decode_latent
    orig_noise = vm._sample_decode_noise_like

    def prep_wrap(codes, text, refer, **kw):
        cap["codes"] = codes.detach().clone()
        cap["text"] = text.detach().clone()
        cap["ge"] = kw["ge"].detach().clone() if torch.is_tensor(kw.get("ge")) else None
        cap["ge_text"] = kw["ge_text"].detach().clone() if torch.is_tensor(kw.get("ge_text")) else None
        return orig_prep(codes, text, refer, **kw)

    def noise_wrap(target, y_lengths, sequential=False):
        r = orig_noise(target, y_lengths, sequential=sequential)
        cap["noise"] = r.detach().clone()
        return r

    vm.prepare_decode_latent = prep_wrap
    vm._sample_decode_noise_like = noise_wrap

    # ---- 中间张量 hooks ----
    hooks = []

    def mk(name, dirname):
        def hook(mod, inp, out):
            t = out if isinstance(out, torch.Tensor) else out[0]
            cap[name] = t.detach().float().cpu()
        return hook

    for modname, module in [
        ("h_quantizer_dec", vm.quantizer),
        ("h_encp_in", vm.enc_p),          # hook 记录 out; in 另抓
        ("h_ssl_proj", vm.enc_p.ssl_proj),
        ("h_encoder_ssl", vm.enc_p.encoder_ssl),
        ("h_text_emb", vm.enc_p.text_embedding),
        ("h_encoder_text", vm.enc_p.encoder_text),
        ("h_mrte", vm.enc_p.mrte),
        ("h_encoder2", vm.enc_p.encoder2),
        ("h_encp_proj", vm.enc_p.proj),
        ("h_flow", vm.flow),
        ("h_dec", vm.dec),
        ("h_dec_conv_pre", vm.dec.conv_pre),
        ("h_dec_cond", vm.dec.cond),
    ]:
        hooks.append(module.register_forward_hook(mk(modname, hdir)))
    for i, up in enumerate(vm.dec.ups):
        hooks.append(up.register_forward_hook(mk(f"h_dec_up{i}", hdir)))

    # enc_p 的 m_p/logs_p 从 proj 输出 split; 另抓 enc_p 输入(上采样后的 quantized)
    orig_encp = vm.enc_p.forward

    def encp_wrap(y, y_lengths, text, text_lengths, ge=None, speed=1, **kw):
        cap["h_encp_input"] = y.detach().float().cpu()
        cap["h_encp_ge"] = ge.detach().float().cpu() if torch.is_tensor(ge) else None
        return orig_encp(y, y_lengths, text, text_lengths, ge=ge, speed=speed, **kw)

    vm.enc_p.forward = encp_wrap

    # flow 输入 z_p: flow.forward(x,...) 的 inp[0]
    def flow_hook(mod, inp, out):
        cap["h_flow_in"] = inp[0].detach().float().cpu()
    hooks.append(vm.flow.register_forward_hook(flow_hook))

    try:
        # 重放端到端推理 (与 golden_export.run_pair 同参)
        g.reset_captures()
        out = tts.run({
            "text": sentence, "text_lang": "zh",
            "ref_audio_path": str(wav_path), "aux_ref_audio_paths": [],
            "prompt_text": PROMPT_TEXT, "prompt_lang": "zh",
            "top_k": 1, "top_p": 1, "temperature": 1.0,
            "text_split_method": "cut0",
            "batch_size": 1, "split_bucket": False,
            "parallel_infer": False, "vits_parallel_infer": False,
            "speed_factor": 1.0, "seed": seed,
            "streaming_mode": False, "return_fragment": False,
        })
        chunk = None
        for chunk in out:
            pass
        sr, audio = chunk
        wav_tts = torch.from_numpy(np.asarray(audio)).float()

        # 一致性校验: 抓到的 wav 应与 golden pair 一致 (同 seed 同输入)
        print(f"[{stem}] replay wav len={wav_tts.numel()} (音频基线=h_dec hook)", flush=True)

        # ---- 输入束 ----
        ref_path = REPO / "tests/golden/refs" / f"{ref_stem}.pt"
        ref = (torch.load(ref_path, weights_only=False)
               if ref_path.exists() else None)
        save_bin(idir, "codes", cap["codes"])          # [n_q,B,T] int64
        (idir / "codes.dtype").write_text("int64")
        save_bin(idir, "phones", cap["text"])           # [B,T] int64
        (idir / "phones.dtype").write_text("int64")
        if torch.is_tensor(cap.get("ge")):
            save_bin(idir, "ge", cap["ge"])
            save_bin(idir, "ge_text", cap["ge_text"])
        elif ref is not None:  # 兜底用 refs 束
            save_bin(idir, "ge", ref["ge"])
            save_bin(idir, "ge_text", ref["ge_text"])
            print(f"[{stem}] WARN: ge 来自 refs bundle (prep kw 未带)", flush=True)
        else:
            raise RuntimeError("无可用 ge 来源")
        if "noise" in cap and cap["noise"] is not None:
            save_bin(idir, "noise", cap["noise"])
        else:
            print(f"[{stem}] WARN: noise 未抓到", flush=True)
        save_bin(outdir / "inputs", "tts_replay_wav", wav_tts.view(1, -1))

        # ---- hook goldens ----
        for k, v in sorted(cap.items()):
            if k.startswith("h_") and v is not None:
                save_bin(hdir, k, v)

        meta = {
            "stem": stem, "sentence": sentence, "ref_wav": ref_wav,
            "seed": seed, "sr": sr,
            "codes_shape": list(cap["codes"].shape), "phones_shape": list(cap["text"].shape),
            "wav_len": int(wav_tts.numel()),
            "hooks": {k: list(v.shape) for k, v in sorted(cap.items())
                      if k.startswith("h_") and v is not None},
        }
        (outdir / "meta.json").write_text(json.dumps(meta, indent=1))
        print(f"[{stem}] exported -> {outdir}", flush=True)
    finally:
        vm.prepare_decode_latent = orig_prep
        vm._sample_decode_noise_like = orig_noise
        vm.enc_p.forward = orig_encp
        for h in hooks:
            h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True,
                    help="形如 <ref_stem>__s<sent_idx>")
    args = ap.parse_args()
    g = Golden()
    for stem in args.pairs:
        outdir = OUTROOT / stem
        if (outdir / "meta.json").exists():
            print(f"[{stem}] skip (meta.json exists)")
            continue
        _, _, sidx = stem.rpartition("__s")
        t0 = time.time()
        export_pair(g, stem, outdir, SENTENCES[int(sidx)], 42)
        print(f"[{stem}] done in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
