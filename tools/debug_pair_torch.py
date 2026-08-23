#!/usr/bin/env python
"""debug_pair_torch.py — 用真实 CPUFast 管线跑单 pair, 捕获:
  - 每次 ar_predict_layer 输出(step 对齐的原始 logits)
  - 每次 sample() 的实际采样 token
  - process_prompt / decode_next_token 的块输出
并与 tests/golden/pairs/*.pt 的字段逐一对齐验证。
"""
import sys, os
sys.path.insert(0, '/Volumes/2T/wt-gsv/AR')
sys.path.insert(0, '/Volumes/2T/wt-gsv/AR/tools')
sys.path.insert(0, '/Volumes/2T/GPT-SoVITS-CPUFast')

import torch
import golden_export as ge

g = ge.Golden()
tts = g.tts
dec = tts.t2s_model.model

cap = {"logits": [], "samples": [], "pp_out": [], "dn_out": []}

def pred_hook(mod, inp, out):
    cap["logits"].append(out.detach().float().cpu())
    # 记录该次采样的实际 token(在 sample 之后无从 hook, 改为记录输入 xy_dec 最后位置)
orig_pred = dec.ar_predict_layer.forward
def pred_forward(x):
    out = orig_pred(x)
    cap["logits"].append(out.detach().float().cpu())
    return out
dec.ar_predict_layer.forward = pred_forward

# sample() 捕获
import AR.models.utils as aru
orig_sample = aru.sample
def sample_probe(logits, previous_tokens=None, **kw):
    idx_next, probs = orig_sample(logits, previous_tokens=previous_tokens, **kw)
    cap["samples"].append((idx_next.detach().cpu().item(),
                           logits.detach().float().cpu().reshape(-1, logits.shape[-1])[-1],
                           kw.get("repetition_penalty", None),
                           logits.shape))
    return idx_next, probs
aru.sample = sample_probe
# infer_panel_naive 里 from ... import 的是模块内引用, 需确认调用点
import AR.models.t2s_model as atm
if hasattr(atm, 'sample'):
    atm.sample = sample_probe

blocks = dec.t2s_transformer.blocks
for i, blk in enumerate(blocks):
    op, od = blk.process_prompt, blk.decode_next_token
    def wp(orig, bi=i):
        def f(*a, **k):
            r = orig(*a, **k)
            if len(cap["pp_out"]) <= bi:
                cap["pp_out"].append(r[0].detach().float().cpu())
            return r
        return f
    def wd(orig, bi=i):
        def f(*a, **k):
            r = orig(*a, **k)
            if len(cap["dn_out"]) <= bi:
                pass
            if len(cap.setdefault("dn_last", [])) <= bi:
                cap.setdefault("dn_last", []).append(None)
            cap["dn_last"][bi] = r[0].detach().float().cpu()
            return r
        return f
    blk.process_prompt = wp(op)
    blk.decode_next_token = wd(od)

wav = sorted((ge.REPO / 'test_wav').glob('*.wav'))[0]
rec = g.run_pair(wav, 0)

print("n_logits_captures:", len(cap["logits"]))
print("n_samples:", len(cap["samples"]), "penalty:", cap['samples'][0][2] if cap['samples'] else None)
pt = torch.load('/Volumes/2T/wt-gsv/AR/tests/golden/pairs/vo_BZLQ001_4_hutao_02__s0.pt',
                map_location='cpu', weights_only=False)
L = torch.stack(cap["logits"], 0).reshape(len(cap["logits"]), -1, cap["logits"][0].shape[-1])[:, 0]
print("captured steps:", L.shape, "golden n_ar_steps:", pt["n_ar_steps"], "tokens:", tuple(pt["tokens"].shape))
toks_cap = L.argmax(-1)
print("captured argmax == golden tokens:", torch.equal(toks_cap.to(torch.int32), pt["tokens"]))
print("first mismatch:", next((i for i in range(min(len(toks_cap), len(pt['tokens']))) if toks_cap[i]!=pt['tokens'][i]), None))
# 逐步对比前几步
for k in range(4):
    c = torch.nn.functional.cosine_similarity(L[k].double(), pt["logits_first8"][k].double(), dim=0).item()
    print(f"step{k}: cos(captured, golden)={c:.6f} cap_argmax={toks_cap[k].item()} golden={pt['tokens'][k].item()}")
# pp_out vs layers_prefill
if cap["pp_out"]:
    l23 = cap["pp_out"][-1]
    print("cos(pp_out[L23], saved layers_prefill[23]):",
          torch.nn.functional.cosine_similarity(l23.flatten().double(), pt["layers_prefill"][23].flatten().double(), dim=0).item())

torch.save({"L": L[:32], "samples": [(s[0], s[2]) for s in cap["samples"][:32]],
            "pp_L23": cap["pp_out"][-1] if cap["pp_out"] else None},
           "/tmp/torch_debug_s0.pt")
print("saved /tmp/torch_debug_s0.pt")
