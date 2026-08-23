#!/usr/bin/env python3
"""layer0_check.py — A3 层验证出口条件

用 A2(.gsv 读取器)+A3(kern 原语) 在 native 侧手工跑通 AR 第一个 transformer 层,
对照 golden 快照 pairs/*.pt 的 layers_prefill[0](第1层输出)。
门槛(G1): cos ≥ 0.9999 且 max-rel ≤ 1e-3。

流程:
  1. 从 pairs/*.pt 抽取输入(phones_ids/prompt_tokens/bert_feat_1024)存原始 bin
  2. 调 build/tests/layer0_check(读 weights/ar_s1v3.gsv + bin, 输出层0输出 bin)
  3. 对比 layers_prefill[0]

用法:
  /Users/baicai1145/miniconda3/envs/GPTSoVits/bin/python tools/layer0_check.py \\
      [--pair vo_BZLQ001_4_hutao_02__s0]
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
GOLDEN = REPO / "tests" / "golden"
WORK = Path("/tmp/gsv_layer0")
BIN = REPO / "build" / "tests" / "layer0_check"


def save_bin(name: str, arr: np.ndarray, dtype):
    WORK.mkdir(exist_ok=True)
    p = WORK / name
    p.write_bytes(np.ascontiguousarray(arr, dtype=dtype).tobytes())
    return str(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="vo_BZLQ001_4_hutao_02__s0")
    ap.add_argument("--pair-file", default=None, help="直接指定 .pt 路径(覆盖 --pair)")
    ap.add_argument("--weights", default=str(REPO / "weights" / "ar_s1v3.gsv"))
    args = ap.parse_args()

    ppath = Path(args.pair_file) if args.pair_file else \
        GOLDEN / "pairs" / f"{args.pair}.pt"
    rec = torch.load(ppath, map_location="cpu", weights_only=False)
    phones = rec["phones_ids"].numpy().astype(np.int64)          # (1,26)
    prompt = rec["prompt_tokens"].numpy().astype(np.int64)       # (1,L)
    bert_in = rec["bert_feat_1024"].numpy().astype(np.float32)   # (1,T,1024)
    ref = rec["layers_prefill"][0][0].numpy().astype(np.float32) # (193,512)

    p_phones = save_bin("phones.bin", phones.ravel(), np.int64)
    p_prompt = save_bin("prompt.bin", prompt.ravel(), np.int64)
    p_bert = save_bin("bert_in.bin", bert_in.ravel(), np.float32)

    out_bin = WORK / "layer0_out.bin"
    cmd = [str(BIN), "--weights", args.weights,
           "--phones", p_phones, "--prompt", p_prompt, "--bert-in", p_bert,
           "--text-len", str(phones.shape[1]), "--prompt-len", str(prompt.shape[1]),
           "--out", str(out_bin)]
    print("[run]", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        sys.exit(f"native 侧失败 rc={r.returncode}")

    got = np.frombuffer(out_bin.read_bytes(), dtype="<f4").reshape(ref.shape)

    # G1 度量(gates.json): cos 与 max-rel(以参考峰值为归一化)
    g, w = got.ravel().astype(np.float64), ref.ravel().astype(np.float64)
    cos = float((g @ w) / np.sqrt((g @ g) * (w @ w)))
    peak = np.abs(w).max()
    max_rel = float(np.abs(g - w).max() / peak)

    print(f"\n== 层0输出 vs layers_prefill[0] ({ppath.name}) ==")
    print(f"cos      = {cos:.9f}  (门 ≥ 0.9999)")
    print(f"max-rel  = {max_rel:.3e} (门 ≤ 1e-3)")
    ok = cos >= 0.9999 and max_rel <= 1e-3
    print("结果:", "PASS ✅" if ok else "FAIL ❌")

    # 附加诊断: bert_proj 复算一致性(native 已在 stdout 报告), 这里只报层级差异分布
    diff = np.abs(got - ref) / peak
    print(f"逐位置相对误差 p50={np.median(diff):.2e} p99={np.percentile(diff, 99):.2e}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
