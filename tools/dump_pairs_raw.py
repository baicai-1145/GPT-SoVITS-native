#!/usr/bin/env python
"""dump_pairs_raw.py — 把 tests/golden/pairs/*.pt 摊平为 ar_pair_run 可读的原始二进制。

用法:
  python tools/dump_pairs_raw.py <pairs_dir> <out_dir> [--skip-incomplete]

每对一个子目录: phones.i64 / prompt.i64 / bert.f32 / meta.txt("T P golden_steps")。
bert_feat_1024 兼容两种存储朝向, 并用 bert_proj 复算对 post-hook 输出 bert_feat 校验。
"""
import argparse
import glob
import os
import sys

import torch
import torch.nn.functional as F

CKPT = "/Volumes/2T/GPT-SoVITS-CPUFast/GPT_SoVITS/pretrained_models/s1v3.ckpt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--skip-incomplete", action="store_true",
                    help="缺输入字段的 pair 直接跳过(默认报错)")
    args = ap.parse_args()

    sd = torch.load(CKPT, map_location="cpu", weights_only=False)["weight"]
    W = sd["model.bert_proj.weight"].float()
    bW = sd["model.bert_proj.bias"].float()

    os.makedirs(args.out_dir, exist_ok=True)
    n_ok = n_skip = 0
    for fp in sorted(glob.glob(os.path.join(args.pairs_dir, "*.pt"))):
        stem = os.path.splitext(os.path.basename(fp))[0]
        p = torch.load(fp, map_location="cpu", weights_only=False)
        need = ["tokens", "prompt_tokens", "bert_feat_1024", "logits_first8",
                "logits_last", "n_ar_steps", "phones_ids"]
        missing = [k for k in need if p.get(k) is None]
        if missing:
            if args.skip_incomplete:
                print(f"[skip] {stem}: 缺 {missing}")
                n_skip += 1
                continue
            sys.exit(f"{stem} 缺字段 {missing}")

        phones = p["phones_ids"].reshape(-1).to(torch.int64)
        prompt = p["prompt_tokens"].reshape(-1).to(torch.int64)
        bf = p["bert_feat_1024"].float()
        assert bf.dim() == 3 and bf.shape[0] == 1, bf.shape
        cand = bf.transpose(1, 2) if bf.shape[1] == 1024 else bf  # → [1,T,1024]
        T = phones.numel()
        assert cand.shape == (1, T, 1024), (cand.shape, T)

        # 朝向自证: bert_proj(cand)+bias 应等于 post-hook 输出 bert_feat [1,T,512]
        ref = p["bert_feat"].float().reshape(T, -1)
        rec = (cand.reshape(T, 1024) @ W.T + bW)
        cos = F.cosine_similarity(rec.flatten(), ref.flatten(), dim=0).item()
        assert abs(cos - 1.0) < 1e-4, f"{stem}: bert 朝向校验失败 cos={cos}"

        d = os.path.join(args.out_dir, stem)
        os.makedirs(d, exist_ok=True)
        phones.numpy().tofile(os.path.join(d, "phones.i64"))
        prompt.numpy().tofile(os.path.join(d, "prompt.i64"))
        cand.reshape(T, 1024).contiguous().numpy().tofile(os.path.join(d, "bert.f32"))
        with open(os.path.join(d, "meta.txt"), "w") as f:
            f.write(f"{T} {prompt.numel()} {int(p['n_ar_steps'])}\n")

        # golden 侧锚点也一并摊平, check 脚本免开 torch
        p["tokens"].to(torch.int32).numpy().tofile(os.path.join(d, "g_tokens.i32"))
        p["logits_first8"].float().numpy().tofile(os.path.join(d, "g_logits_first8.f32"))
        p["logits_last"].float().numpy().tofile(os.path.join(d, "g_logits_last.f32"))
        with open(os.path.join(d, "g_meta.txt"), "w") as f:
            f.write(f"{tuple(p['logits_first8'].shape)} {p['logits_last'].shape}\n".replace("(", "").replace(")", "").replace(", ", ","))
        n_ok += 1
    print(f"导出 {n_ok} 个 pair → {args.out_dir}" + (f"(跳过 {n_skip})" if n_skip else ""))


if __name__ == "__main__":
    main()
