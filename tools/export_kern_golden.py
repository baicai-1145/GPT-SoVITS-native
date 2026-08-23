#!/usr/bin/env python3
"""export_kern_golden.py — 导出 A3 内核原语的 torch golden 张量对

输出: tests/kern_golden/<name>.{f32,f16} 自描述二进制
  [u32 rank][u32 dims×rank][raw data]   (小端; f32=4B/elem, f16=2B/elem)

参考实现语义(与 src/kern C++ 严格同构):
  rmsnorm : x * rsqrt(mean(x²)+eps) * g          (统计量 fp64→fp32 模拟 fp32 统计)
  layernorm: F.layer_norm (有偏方差, 仿射)
  softmax : F.softmax(dim=-1) 稳定版
  silu    : F.silu ; relu: clamp_min(0)
  rope    : freq_i = base^(-2i/hd); GptJ 相邻配对 / NeoX 半分配对, 角度 float64 后取 float32
用法:
  /Users/baicai1145/miniconda3/envs/GPTSoVits/bin/python tools/export_kern_golden.py
"""
import struct
from pathlib import Path

import torch
import torch.nn.functional as F

OUT = Path(__file__).resolve().parent.parent / "tests" / "kern_golden"
torch.manual_seed(20250823)


def save(name: str, t: torch.Tensor):
    OUT.mkdir(parents=True, exist_ok=True)
    if t.dtype == torch.float16:
        ext, data = "f16", t.contiguous().view(torch.uint8).numpy().tobytes()
    else:
        assert t.dtype == torch.float32
        ext = "f32"
        data = t.contiguous().numpy().astype("<f4").tobytes()
    hdr = struct.pack(f"<I{len(t.shape)}I", len(t.shape), *t.shape)
    (OUT / f"{name}.{ext}").write_bytes(hdr + data)
    print(f"  {name}.{ext} {list(t.shape)}")


def main():
    # ---- gemv 两例(一偶一奇跨尾) ----
    for tag, out_, in_ in (("gemv_a", 128, 96), ("gemv_b", 33, 131)):
        w = (torch.randn(out_, in_) * 0.05).half()
        x = torch.randn(in_)
        y = w.float() @ x
        save(f"{tag}_w", w)
        save(f"{tag}_x", x)
        save(f"{tag}_y", y)

    # ---- rmsnorm ----
    n = 640
    x = torch.randn(3, n) * 2 + 1
    g = torch.randn(n) * 0.5 + 1
    eps = 1e-5
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * g
    save("rms_x", x)
    save("rms_g", g)
    save("rms_y", y)

    # ---- layernorm ----
    x = torch.randn(5, 512) * 3 - 0.5
    g = torch.randn(512).abs() + 0.5
    b = torch.randn(512) * 0.1
    y = F.layer_norm(x, (512,), g, b, 1e-5)
    save("ln_x", x)
    save("ln_g", g)
    save("ln_b", b)
    save("ln_y", y)

    # ---- softmax (奇数宽覆盖尾部) ----
    x = torch.randn(7, 257) * 3
    y = F.softmax(x, dim=-1)
    save("sm_x", x)
    save("sm_y", y)

    # ---- silu / relu ----
    v = torch.randn(1000) * 2
    save("act_v", v)
    save("act_silu", F.silu(v))
    save("act_relu", F.relu(v))

    # ---- rope 双风格 ----
    seq, heads, hd, pos_base, base = 13, 4, 32, 5, 10000.0
    q = torch.randn(seq, heads * hd)
    inv = torch.pow(base, -torch.arange(0, hd // 2, dtype=torch.float64) * 2 / hd)

    def rope(style):
        y = q.clone()
        for t in range(seq):
            row = y[t].view(heads, hd)
            ang = (pos_base + t) * inv  # [hd/2] float64
            c, s = ang.cos().float(), ang.sin().float()
            for h in range(heads):
                d = row[h]
                if style == "gptj":
                    x0, x1 = d[0::2].clone(), d[1::2].clone()
                    d[0::2] = x0 * c - x1 * s
                    d[1::2] = x0 * s + x1 * c
                else:  # neox
                    half = hd // 2
                    x0, x1 = d[:half].clone(), d[half:].clone()
                    d[:half] = x0 * c - x1 * s
                    d[half:] = x0 * s + x1 * c
        return y

    save("rope_in", q)
    save("rope_gptj_out", rope("gptj"))
    save("rope_neox_out", rope("neox"))
    print(f"OK -> {OUT}")


if __name__ == "__main__":
    main()
