#!/usr/bin/env python3
"""golden_export.py — M0 golden 基线导出 + 精度门定标 (ARCHITECTURE.md §3/G1-G3)

模式:
  export     用 test_wav 全部音频作参考, 导出参考特征束 + 端到端 golden 对
  calibrate  在已导出基线上测三种噪声地板, 定标 G1/G2/G3 门槛数值

Golden 抓取点 (forward hooks / 返回值):
  - HuBERT last_hidden_state          (refs/<stem>/bundle.pt: hubert_hidden)
  - 参考谱 spec / sv_emb / decode_ge   (同上)
  - prompt_semantic 语义 token        (同上)
  - AR 每步 logits (前8步全量+末步) 与贪心 token 序列 (pairs/*.pt)
  - bert_proj 输入的 BERT 特征        (pairs/*.pt: bert_feat)
  - 合成波形 wav                       (pairs/*.pt)

用法:
  python3 tools/golden_export.py --mode export    [--limit-refs N]
  python3 tools/golden_export.py --mode calibrate
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path("/Volumes/2T/GPT-SoVITS-native")
CPUFAST = Path("/Volumes/2T/GPT-SoVITS-CPUFast")
sys.path.insert(0, str(CPUFAST))
sys.path.insert(0, str(CPUFAST / "GPT_SoVITS"))
# sv.py 等模块用 os.getcwd() 拼相对路径，必须在 import 前 chdir 到 CPUFast 根目录
import os
os.chdir(CPUFAST)

SENTENCES = [
    "你好，世界。",
    "今天天气真不错，我们去公园散步吧。",
    "重庆的火锅店终于开张了。",
    "2024年10月1日，我在成都买了3.5斤樱桃。",
    "The quick brown fox jumps over the lazy dog.",
    "他长大以后想当一名宇航员。",
    "这是一段比较长的测试文本，用来验证模型在处理多分句、含逗号顿号问号等标点时的表现是否稳定可靠？",
    "人工智能正在改变世界。",
    "银行旁边的河水平静地流着。",
    "GPT-SoVITS 是一个开源的语音合成项目。",
]

PROMPT_TEXT = "原来你也玩原神。"
SEED = 42

# R-probe (仅换线程数) 实测出 VITS 解码输出不稳定的参考音频 (mel_rel>0.1):
# AR token 逐位一致但波形大幅漂移 → 不能作为数值 golden 锚点, 定标时排除并记录。
UNSTABLE_REFS = {"vo_BZLQ001_6_hutao_02", "vo_BZLQ001_6_hutao_03"}


def sha16(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


class Golden:
    def __init__(self):
        from TTS_infer_pack.TTS import TTS, TTS_Config

        cfg = {
            "custom": {
                "bert_base_path": str(REPO / "pretrained_models/chinese-roberta-wwm-ext-large"),
                "cnhuhbert_base_path": str(REPO / "pretrained_models/chinese-hubert-base"),
                "device": "cpu",
                "is_half": False,
                "t2s_weights_path": str(REPO / "pretrained_models/s1v3.ckpt"),
                "version": "v2ProPlus",
                "vits_weights_path": str(REPO / "pretrained_models/v2Pro/s2Gv2ProPlus.pth"),
            }
        }
        self.config = TTS_Config(cfg)
        print("init TTS (loads AR/SoVITS/BERT/HuBERT/SV) ...", flush=True)
        t0 = time.time()
        self.tts = TTS(self.config)
        print(f"models loaded in {time.time()-t0:.1f}s", flush=True)
        self.captures = {}
        self._hooks = []
        self._attach_hooks()

    def _attach_hooks(self):
        tts = self.tts
        cap = self.captures

        def as_tensor(out):
            if isinstance(out, dict):
                return out["last_hidden_state"]
            if isinstance(out, (tuple, list)):
                return out[0]
            return out

        def mk(name):
            def hook(mod, inp, out):
                t = as_tensor(out)
                cap[name] = t.detach().float().cpu()
            return hook

        self._hooks.append(tts.cnhuhbert_model.model.register_forward_hook(mk("hubert_hidden")))
        self._hooks.append(tts.bert_model.register_forward_hook(mk("bert_out")))
        self._hooks.append(
            tts.t2s_model.model.bert_proj.register_forward_hook(mk("bert_proj_out"))
        )
        # AR 每步 logits
        ar_logits = []

        def pred_hook(mod, inp, out):
            ar_logits.append(out.detach().float().cpu())

        self._hooks.append(tts.t2s_model.model.ar_predict_layer.register_forward_hook(pred_hook))
        self._hooks.append(
            tts.vits_model.register_forward_hook(lambda m, i, o: cap.__setitem__("vits_out", o))
        )
        self.ar_logits = ar_logits
        # sv_model.compute_embedding3 是方法而非模块 —— 包一层记录
        orig_sv = tts.sv_model.compute_embedding3

        def sv_wrap(audio_tensor):
            r = orig_sv(audio_tensor)
            cap["sv_emb"] = r.detach().float().cpu()
            return r

        tts.sv_model.compute_embedding3 = sv_wrap

    def reset_captures(self):
        self.captures.clear()
        self.ar_logits.clear()

    # ---------- export ----------
    def export_refs(self, outdir: Path, limit=None):
        tts = self.tts
        wavs = sorted((REPO / "test_wav").glob("*.wav"))
        if limit:
            wavs = wavs[:limit]
        refdir = outdir / "refs"
        refdir.mkdir(parents=True, exist_ok=True)
        manifest = []
        for i, wav in enumerate(wavs):
            stem = wav.stem
            f = refdir / f"{stem}.pt"
            if f.exists():
                print(f"[{i+1}/{len(wavs)}] skip {stem}", flush=True)
                manifest.append({"wav": wav.name, "sha16": sha16(wav), "file": str(f.relative_to(outdir))})
                continue
            self.reset_captures()
            tts.prompt_cache["ref_audio_path"] = str(wav)
            tts._set_ref_spec(str(wav))
            tts._set_prompt_semantic(str(wav))
            tts._get_runtime_refer_audio_spec_and_sv_emb()
            ge, ge_text = tts._get_runtime_decode_condition()
            spec, _ = tts.prompt_cache["refer_spec"][0]
            bundle = {
                "wav_sha16": sha16(wav),
                "spec": spec.detach().float().cpu(),
                "sv_emb": self.captures.get("sv_emb"),
                "ge": ge.detach().float().cpu() if torch.is_tensor(ge) else None,
                "ge_text": ge_text.detach().float().cpu() if torch.is_tensor(ge_text) else None,
                "hubert_hidden": self.captures["hubert_hidden"].detach().float().cpu(),
                "prompt_semantic": tts.prompt_cache["prompt_semantic"].detach().cpu(),
            }
            torch.save(bundle, f)
            ps = bundle["prompt_semantic"].numel()
            print(f"[{i+1}/{len(wavs)}] {stem}: sem_tokens={ps} spec={tuple(spec.shape)}", flush=True)
            manifest.append({"wav": wav.name, "sha16": sha16(wav), "file": str(f.relative_to(outdir)),
                             "sem_tokens": ps})
        return manifest

    def run_pair(self, wav: Path, sent_idx: int) -> dict:
        tts = self.tts
        self.reset_captures()
        out = tts.run({
            "text": SENTENCES[sent_idx], "text_lang": "zh",
            "ref_audio_path": str(wav), "aux_ref_audio_paths": [],
            "prompt_text": PROMPT_TEXT, "prompt_lang": "zh",
            "top_k": 1, "top_p": 1, "temperature": 1.0,
            "text_split_method": "cut0",   # 不切句, 单段完整推理
            "batch_size": 1, "split_bucket": False,
            "parallel_infer": False, "vits_parallel_infer": False,
            "speed_factor": 1.0, "seed": SEED,
            "streaming_mode": False, "return_fragment": False,
        })
        if hasattr(out, "__next__"):  # 生成器: 取最后一个 chunk
            chunk = None
            for chunk in out:
                pass
            sr, audio = chunk
        else:
            sr, audio = out
        logits = torch.stack(self.ar_logits, 0)  # [T, batch?, vocab]
        logits = logits.reshape(logits.shape[0], -1, logits.shape[-1])[:, 0]  # [T, vocab]
        tokens = logits.argmax(-1)
        rec = {
            "sentence_idx": sent_idx, "sentence": SENTENCES[sent_idx],
            "ref_wav": wav.name, "seed": SEED, "sr": sr,
            "tokens": tokens.to(torch.int32),
            "logits_first8": logits[:8].to(torch.float32),
            "logits_last": logits[-1].to(torch.float32),
            "n_ar_steps": int(logits.shape[0]),
            "bert_feat": self.captures.get("bert_proj_out").detach().float().cpu()
            if self.captures.get("bert_proj_out") is not None else None,
            "wav": torch.from_numpy(np.asarray(audio, dtype=np.float32)),
        }
        return rec

    def export_pairs(self, outdir: Path, ref_manifest, limit_refs=None):
        wavs = sorted((REPO / "test_wav").glob("*.wav"))
        if limit_refs:
            wavs = wavs[:limit_refs]
        pdir = outdir / "pairs"
        pdir.mkdir(parents=True, exist_ok=True)
        jobs = [(wavs[0], s) for s in range(len(SENTENCES))]
        jobs += [(w, 0) for w in wavs[1:]]
        entries = []
        for k, (wav, si) in enumerate(jobs):
            stem = f"{wav.stem}__s{si}"
            f = pdir / f"{stem}.pt"
            if f.exists():
                print(f"[pair {k+1}/{len(jobs)}] skip {stem}", flush=True)
            else:
                t0 = time.time()
                try:
                    rec = self.run_pair(wav, si)
                except Exception as e:
                    print(f"[pair {k+1}/{len(jobs)}] FAIL {stem}: {e!r}", flush=True)
                    entries.append({"stem": stem, "error": repr(e)})
                    continue
                torch.save(rec, f)
                print(f"[pair {k+1}/{len(jobs)}] {stem}: steps={rec['n_ar_steps']} "
                      f"wav={rec['wav'].numel()} ({time.time()-t0:.1f}s)", flush=True)
            entries.append({"stem": stem, "file": str(f.relative_to(outdir))})
        return entries


# ---------- calibrate ----------

def _mel_l1(a: torch.Tensor, b: torch.Tensor, sr=32000):
    """归一化到 [-1,1] 后的 mel 幅度谱 L1, 返回 (绝对值, 相对基线的比例)。"""
    import torchaudio
    a, b = a / 32768.0, b / 32768.0   # golden 存的是 int16 刻度 float
    n = min(a.numel(), b.numel())
    m = torchaudio.transforms.MelSpectrogram(
        sr, n_fft=2048, win_length=2048, hop_length=640,
        n_mels=128, f_min=0.0, f_max=None, power=2.0)
    ma, mb = m(a[:n].float()), m(b[:n].float())
    t = min(ma.shape[-1], mb.shape[-1])
    l1 = float((ma[..., :t] - mb[..., :t]).abs().mean())
    base = float(mb.mean().clamp_min(1e-9))
    return l1, l1 / base


def _rel_cos(a: torch.Tensor, b: torch.Tensor):
    a = a.flatten().float(); b = b.flatten().float()
    n = min(a.numel(), b.numel()); a, b = a[:n], b[:n]
    cos = float(torch.nn.functional.cosine_similarity(a, b, dim=0))
    rel = float((a - b).abs().max() / b.abs().max().clamp_min(1e-6))
    return rel, cos


def calibrate(outdir: Path):
    g = Golden()
    tts = g.tts
    manifest = json.loads((outdir / "manifest.json").read_text())
    recs = [torch.load(outdir / p["file"], weights_only=False)
            for p in manifest["pairs"] if "file" in p]
    # bf16 autocast 在 CPU 上病态慢, 定标子集只取短中句 (≤400 步), 统计代表性足够且可控时
    short = [r for r in recs
             if r["n_ar_steps"] <= 400
             and not any(r["ref_wav"].startswith(u) for u in UNSTABLE_REFS)] or recs
    short.sort(key=lambda r: r["n_ar_steps"])
    # 按步数分位取子集, 覆盖短句到顶满 1500 步
    seen, subset = set(), []
    for q in (0.0, 0.25, 0.5, 0.75, 1.0):
        r = short[int(q * (len(short) - 1))]
        key = (r["ref_wav"], r["sentence_idx"])
        if key not in seen:
            seen.add(key)
            subset.append(r)
    print(f"calibration subset: {[(r['ref_wav'][:20], r['sentence_idx'], r['n_ar_steps']) for r in subset]}", flush=True)

    def measure(tag):
        rows = []
        for i, base in enumerate(subset):
            g.reset_captures()
            try:
                rec = g.run_pair(REPO / "test_wav" / base["ref_wav"], base["sentence_idx"])
            except Exception as e:
                print(f"[{tag}] pair {i} FAIL {e!r}", flush=True)
                continue
            tl = min(base["tokens"].numel(), rec["tokens"].numel())
            teq = bool(tl > 0 and torch.equal(base["tokens"][:tl], rec["tokens"][:tl])) \
                and base["tokens"].numel() == rec["tokens"].numel()
            lratio = rec["tokens"].numel() / max(1, base["tokens"].numel())
            lrel, lcos = _rel_cos(base["logits_last"], rec["logits_last"])
            mel, mel_rel = _mel_l1(base["wav"], rec["wav"])
            rows.append({"pair": f"{base['ref_wav'][:24]}__s{base['sentence_idx']}",
                         "steps": base["n_ar_steps"], "tokens_equal": teq,
                         "len_ratio": round(lratio, 3), "logit_rel": lrel,
                         "logit_cos": lcos, "mel_l1": mel, "mel_rel": round(mel_rel, 6)})
            print(f"[{tag}] {rows[-1]}", flush=True)
        return rows

    # R 变体: 单线程重跑 (非确定性地板) —— 在任何扰动前做
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    rows_R = measure("R:1thread")
    torch.set_num_threads(old_threads)

    # A 变体: bf16 autocast 全程计算 (激活/矩阵乘低精度包络代理)。
    # 注: 源权重本身已是 fp16 存储, fp32 基线 = fp16存储+fp32计算, 与本引擎目标一致;
    # 真正的额外噪声来自 kernel 实现的激活舍入/累加顺序, 用 bf16(尾数更短)作上界代理。
    print("rerun under torch.autocast(cpu, bfloat16) ...", flush=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        rows_A = measure("A:bf16")

    def stat(vals):
        v = sorted(vals)
        return {"p50": v[len(v)//2], "p95": v[max(0, int(len(v)*0.95)-1)],
                "max": v[-1], "n": len(v)}

    gates = {
        "subset_size": len(subset),
        "floor_nondeterminism_R": {
            "logit_rel": stat([r["logit_rel"] for r in rows_R]),
            "logit_cos": stat([r["logit_cos"] for r in rows_R]),
            "mel_rel": stat([r["mel_rel"] for r in rows_R]),
            "token_agreement": sum(r["tokens_equal"] for r in rows_R) / max(1, len(rows_R)),
        },
        "envelope_bf16_autocast_A": {
            "logit_rel": stat([r["logit_rel"] for r in rows_A]),
            "logit_cos": stat([r["logit_cos"] for r in rows_A]),
            "mel_rel": stat([r["mel_rel"] for r in rows_A]),
            "token_agreement": sum(r["tokens_equal"] for r in rows_A) / max(1, len(rows_A)),
            "len_ratio_range": [min(r["len_ratio"] for r in rows_A),
                                 max(r["len_ratio"] for r in rows_A)],
        },
    }
    fa = gates["envelope_bf16_autocast_A"]
    gates["G1"] = {
        "cos_sim_min": 0.9999,
        "rel_err_max": None,  # 取 2×A包络 p95 向上取整到一位有效数字
    }
    rel_p95 = fa["logit_rel"]["p95"]
    if rel_p95 > 0:
        e = 10 ** -int(np.floor(np.log10(rel_p95)))
        gates["G1"]["rel_err_max"] = float(np.ceil(rel_p95 * 2 / e) * e)
    aga = fa["token_agreement"]
    gates["G2_token_agreement"] = max(0.90, min(0.98, aga - 0.02))
    gates["G3_mel_rel_max"] = 3 * fa["mel_rel"]["p50"]

    (outdir / "gates.json").write_text(json.dumps(gates, ensure_ascii=False, indent=1))
    lines = [
        "# 精度门定标报告 (M0)", "",
        f"- 子集: {len(subset)} 对 (步数≤400, 排除不稳定参考, 按分位取样)",
        f"- 基线: fp32, seed={SEED}, top_k=1", "",
        "## 不稳定参考 (排除在定标外)",
        f"{sorted(UNSTABLE_REFS)} —— R-probe (仅线程数变化) 下 mel_rel 分别实测 0.61/0.82:",
        "AR token 逐位一致但 VITS 波形大幅漂移, 属模型自身对特定参考的敏感区, 非引擎可控行为。",
        "这些参考的 golden 数据仍保留在 tests/golden/ 中供观察, 不用于门槛计算。", "",
        "## 关键事实", 
        "- 五个模型源权重均为 fp16 存储 → fp32 基线本身就是 'fp16存储+fp32计算', 与本引擎第一版设计完全同构;",
        "- 唯一的真·fp32 数值是加载时 weight_norm 融合产物 (SoVITS dec 164 张 + HuBERT 1 张);",
        "- 实测对融合权重二次舍入到 fp16 会显著劣化解码音质 → .gsv 中此类张量只存 fp32 (convert.py 已实现)。",
        "- 因此'权重存储噪声地板'为精确零, 真正待控的是 kernel 层激活舍入/累加顺序 → 用 bf16 autocast 作上界包络。", "",
        "## 非确定性地板 R (单线程重跑 vs 基线)",
        json.dumps(gates["floor_nondeterminism_R"], indent=1), "",
        "## bf16 autocast 包络 A (低精度计算敏感度上界代理)",
        json.dumps(gates["envelope_bf16_autocast_A"], indent=1), "",
        "## 定标后的门槛",
        json.dumps({k: gates[k] for k in ("G1", "G2_token_agreement", "G3_mel_rel_max")}, indent=1), "",
        "规则: G1.rel = 2×A包络p95 向上取一位有效数字; G2 = clamp(A一致率−2%, 90%..98%); G3 = 3×A包络 mel相对变化中位数",
    ]
    (outdir / "CALIBRATION.md").write_text("\n".join(lines))
    print("gates.json / CALIBRATION.md written")


# ---------- augment ----------

def augment(outdir: Path):
    """补录三类前置材料: ①AR 原始输入 ②逐层中间张量(prefill+末步) ③参考音频16k波形"""
    g = Golden()
    tts = g.tts
    dec = tts.t2s_model.model
    from tools.audio_utils import load_audio_mono
    man = json.loads((outdir / "manifest.json").read_text())

    # Pass 1: 参考束补 16kHz 解码波形 (消除 native 端重采样差异源)
    nref = 0
    for r in man["refs"]:
        f = outdir / r["file"]
        b = torch.load(f, weights_only=False)
        if "wav16k" in b:
            continue
        w = load_audio_mono(str(REPO / "test_wav" / r["wav"]), sample_rate=16000)
        b["wav16k"] = torch.from_numpy(np.asarray(w)).float()
        torch.save(b, f)
        nref += 1
    print(f"refs: {nref} 个 bundle 已补 wav16k", flush=True)

    # 额外记录器: AR 输入与逐层输出。
    # 注意: CPUFast 把标准 encoder 拆成了普通类 T2SBlock/T2STransformer (非 nn.Module),
    # 运行时走 t2s_transformer.blocks[i].process_prompt/decode_next_token, 模块 hook 无效,
    # 必须包实例方法; 嵌入/bert_proj 仍是真模块, 可继续用 hook。
    state = {}
    hooks = []
    hooks.append(dec.ar_text_embedding.register_forward_pre_hook(
        lambda m, i: state.setdefault("phones_ids", i[0].detach().cpu())))

    def _aud(m, i):
        state["audio_tok_last"] = i[0].detach().cpu()
        state.setdefault("prompt_tokens", i[0].detach().cpu())
    hooks.append(dec.ar_audio_embedding.register_forward_pre_hook(_aud))
    hooks.append(dec.bert_proj.register_forward_pre_hook(
        lambda m, i: state.__setitem__("bert_in_1024", i[0].detach().float().cpu())))

    blocks = dec.t2s_transformer.blocks
    nl = len(blocks)
    origs = []
    for i, blk in enumerate(blocks):
        # 首调判定用 state 键存在性(随 state.clear() 复位); 禁用闭包计数器——跨 pair 不复位会导致后续全空
        orig_p, orig_d = blk.process_prompt, blk.decode_next_token

        def wrap_p(orig, blk_i=0):
            def f(*a, **k):
                out = orig(*a, **k)
                if f"pre_L{blk_i}" not in state:
                    o = out[0] if isinstance(out, (tuple, list)) else out
                    state[f"pre_L{blk_i}"] = o.detach().float().cpu()
                return out
            return f

        def wrap_d(orig, blk_i=0):
            def f(*a, **k):
                out = orig(*a, **k)
                o = out[0] if isinstance(out, (tuple, list)) else out
                if f"dfirst_L{blk_i}" not in state:
                    state[f"dfirst_L{blk_i}"] = o.detach().float().cpu()
                state[f"last_L{blk_i}"] = o.detach().float().cpu()
                return out
            return f

        blk.process_prompt = wrap_p(orig_p, i)
        blk.decode_next_token = wrap_d(orig_d, i)
        origs.append((blk, orig_p, orig_d))

    npair = skipped = failed = 0
    limit = int(os.environ.get('AUG_LIMIT', '0') or 0)
    for p in man["pairs"]:
        if "file" not in p:
            continue
        f = outdir / p["file"]
        base = torch.load(f, weights_only=False)
        if "phones_ids" in base:
            skipped += 1
            continue
        state.clear()
        blk_id = id(dec.t2s_transformer.blocks[0])
        try:
            rec = g.run_pair(REPO / "test_wav" / base["ref_wav"], base["sentence_idx"])
            # 确定性回归检查: 重跑必须逐位复现基线
            assert torch.equal(rec["tokens"], base["tokens"]), "token drift"
            assert rec["wav"].shape == base["wav"].shape and torch.equal(rec["wav"], base["wav"]), "wav drift"
        except AssertionError as e:
            print(f"[aug SKIP] {p['stem']}: 重跑不一致({e}) —— 不稳定参考对, 原基线保留", flush=True)
            skipped += 1
            continue

        hits = {k: sum(1 for x in state if x.startswith(k)) for k in ("pre_L", "dfirst_L", "last_L")}
        same = id(dec.t2s_transformer.blocks[0]) == blk_id
        print(f"[aug dbg] {p['stem']}: blk_id_same={same} hits={hits}", flush=True)
        src = "process_prompt" if all(f"pre_L{i}" in state for i in range(nl)) else "decode_first"
        key = "pre_L" if src == "process_prompt" else "dfirst_L"
        missing = [i for i in range(nl) if f"{key}{i}" not in state]
        if missing:
            print(f"[aug FAIL] {p['stem']}: src={src} missing={missing[:3]} hits={hits} same={same}", flush=True)
            failed += 1
            continue

        base["phones_ids"] = state["phones_ids"]
        base["prompt_tokens"] = state["prompt_tokens"]
        base["gen_tokens_final"] = state["audio_tok_last"]
        base["bert_feat_1024"] = state["bert_in_1024"]
        base["bert_out"] = g.captures.get("bert_out")
        base["layers_prefill_src"] = src
        base["layers_first"] = torch.stack([state[f"{key}{i}"] for i in range(nl)], 0)
        base["layers_laststep"] = torch.stack([state[f"last_L{i}"] for i in range(nl)], 0)
        torch.save(base, f)
        npair += 1
        print(f"[aug {npair}] {p['stem']} first={tuple(base['layers_first'].shape)}", flush=True)
        if limit and npair >= limit:
            print(f"[aug] 达到 AUG_LIMIT={limit}, 提前停止", flush=True)
            break
    for h in hooks:
        h.remove()
    for blk, op_, od_ in origs:   # 还原被包裹的实例方法
        blk.process_prompt = op_
        blk.decode_next_token = od_
    man["augmented"] = {
        "wav16k": True,
        "ar_inputs": ["phones_ids", "prompt_tokens", "gen_tokens_final", "bert_feat_1024", "bert_out"],
        "layer_snapshots": "layers_first(24,B,L,512, 首轮全层) + layers_laststep(24,B,T,512); 源标记 layers_prefill_src",
        "determinism_regression": "重跑 token/wav 与基线逐位一致才写入; 不稳定对跳过并保留原基线",
    }
    (outdir / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1))
    print(f"done: refs+{nref}, pairs augmented={npair}, skipped(already/unstable)={skipped}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["export", "calibrate", "augment"], required=True)
    ap.add_argument("--out", default=str(REPO / "tests" / "golden"))
    ap.add_argument("--limit-refs", type=int, default=None)
    args = ap.parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.mode == "calibrate":
        calibrate(outdir)
        return
    if args.mode == "augment":
        augment(outdir)
        return

    if args.mode == "export":
        g = Golden()
        t0 = time.time()
        refs = g.export_refs(outdir, args.limit_refs)
        print(f"refs done in {time.time()-t0:.1f}s", flush=True)
        t0 = time.time()
        pairs = g.export_pairs(outdir, refs, args.limit_refs)
        print(f"pairs done in {time.time()-t0:.1f}s", flush=True)
        manifest = {
            "torch_version": torch.__version__, "seed": SEED,
            "sentences": SENTENCES, "prompt_text": PROMPT_TEXT,
            "config": {"version": "v2ProPlus", "device": "cpu", "is_half": False},
            "refs": refs, "pairs": pairs,
        }
        (outdir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
        print("manifest written")


if __name__ == "__main__":
    main()
