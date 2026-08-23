#!/usr/bin/env python3
"""convert.py — GPT-SoVITS-native 权重转换器 (.pth/.ckpt/.bin → .gsv)

GSV1 容器格式（全部小端）:
  Header:
    '<4s'   magic = b'GSV1'
    '<I'    version = 1
    '<I'    flags          bit0: 存在 fp16 数据段
    '<H'    model_name_len + utf8 名称
    '<I'    config_json_len + JSON 字节（{"meta":..., "model_config":...}）
    '<I'    n_tensors
  目录项 × n_tensors:
    '<H' name_len + utf8 名称
    '<H' rank + rank×'<I' dims
    '<H' src_dtype         1=torch.float32, 2=torch.float16
    '<Q' f32_offset  '<Q' f32_nbytes     # fp32 段（必有）
    '<Q' f16_offset  '<Q' f16_nbytes     # fp16 段；nbytes=0 表示无（小张量回退 fp32）
  数据区: 先所有 fp32 块，后所有 fp16 块，每块起始 64B 对齐。

转换规则（ARCHITECTURE.md §3）:
  1. 剥离前缀 'model.' / 'module.'，其余键名原样保留
  2. weight_norm 融合: w = g * v / ||v||₂（除 dim0 外求范数），fp32 计算
  3. 非浮点张量跳过并记录（如 position_ids、num_batches_tracked）
  4. 全部张量写入 fp32 主拷贝；numel ≥ FP16_MIN_ELEMS 的另写 fp16 拷贝
  5. 写入后立即回读校验: 结构解析（纯 struct）+ 数值逐张量对照

用法:
  python3 tools/convert.py --all                 # 转换内置五个模型
  python3 tools/convert.py --src X --out Y.gsv --name Z [--config-json C.json] [--no-verify]
"""
import argparse
import collections
import json
import struct
import sys
from pathlib import Path

import torch
import numpy as np

MAGIC = b"GSV1"
VERSION = 1
ALIGN = 64
FP16_MIN_ELEMS = 4096  # 小于此元素数的张量不写 fp16 段（norm/bias 等保 fp32）
DTYPE_ENUM = {torch.float32: 1, torch.float16: 2}

PRESETS = {  # name -> (源路径, 提取键或 None, 外部 config.json 或 None)
    "ar_s1v3": ("pretrained_models/s1v3.ckpt", "weight", None),
    "sovits_v2ProPlus": ("pretrained_models/v2Pro/s2Gv2ProPlus.pth", "weight", None),
    "hubert_base": ("pretrained_models/chinese-hubert-base/pytorch_model.bin", None,
                    "pretrained_models/chinese-hubert-base/config.json"),
    "roberta_wwm_ext_large": ("pretrained_models/chinese-roberta-wwm-ext-large/pytorch_model.bin", None,
                              "pretrained_models/chinese-roberta-wwm-ext-large/config.json"),
    "eres2netv2_sv": ("pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt", None, None),
}


def strip_prefix(k: str) -> str:
    for p in ("module.", "model."):
        if k.startswith(p):
            return k[len(p):]
    return k


def fuse_weight_norm(sd: dict) -> tuple[dict, int]:
    """融合 weight_g/weight_v → weight。返回新字典、融合对数与融合产物键名集合。

    融合产物是真·fp32 值(源 g/v 是 fp16 但乘除结果需要 >16 位尾数)，
    实测二次舍入到 fp16 会显著劣化 SoVITS 解码音质 —— 这类张量不写 fp16 段。
    """
    out = dict(sd)
    fused_keys = set()
    fused = 0
    for gk in [k for k in sd if k.endswith(".weight_g")]:
        base = gk[: -len("weight_g")]
        vk, wk = base + "weight_v", base + "weight"
        g, v = sd[gk].float(), sd[vk].float()
        dims = list(range(1, v.dim()))
        norm = v.pow(2).sum(dim=dims, keepdim=True).sqrt().clamp_min(1e-12)
        out[wk] = g * v / norm
        del out[gk], out[vk]
        fused_keys.add(strip_prefix(wk))
        fused += 1
    return out, fused, fused_keys


def convert(src: str, out: str, name: str, config_json_path: str | None, verify: bool = True):
    src_p, out_p = Path(src), Path(out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{name}] loading {src_p} ...")
    obj = torch.load(src_p, map_location="cpu", weights_only=False)
    sd = obj[key] if (key := PRESETS.get(name, (None, None))[1]) else obj
    if not isinstance(sd, dict) or not all(isinstance(v, torch.Tensor) for v in sd.values()):
        raise SystemExit(f"[{name}] 意外的源结构: {type(sd)}")

    sd, n_fused, wn_keys = fuse_weight_norm(sd)

    # 过滤非浮点 + 规范键名
    tensors, skipped = {}, []
    for k, v in sd.items():
        nk = strip_prefix(k)
        if v.dtype not in DTYPE_ENUM:
            skipped.append((nk, str(v.dtype)))
            continue
        if nk in tensors:
            raise SystemExit(f"[{name}] 键名冲突: {nk}")
        tensors[nk] = v.contiguous()
    for nk, dt in skipped:
        print(f"[{name}]   skip(non-float): {nk} ({dt})")

    # config JSON
    model_config = None
    if config_json_path:
        model_config = json.loads(Path(config_json_path).read_text())
    elif isinstance(obj, dict) and isinstance(obj.get("config"), dict):
        model_config = obj["config"]
    cfg_obj = {
        "meta": {"model": name, "source": str(src_p), "converter": "convert.py GSV1",
                 "weight_norm_fused": n_fused},
        "model_config": model_config,
    }
    cfg_bytes = json.dumps(cfg_obj, ensure_ascii=False, sort_keys=True).encode()

    # ---- 计算布局 ----
    fixed = 4 + 4 + 4 + 2 + len(name.encode()) + 4 + len(cfg_bytes) + 4
    dir_size = sum(2 + len(k.encode()) + 2 + 4 * len(v.shape) + 2 + 32 for k, v in tensors.items())
    pos = fixed + dir_size
    pos = (pos + ALIGN - 1) // ALIGN * ALIGN

    layout = []  # (key, tensor, f32_off, f32_nb, f16_off, f16_nb)
    p32 = pos
    for k, v in tensors.items():
        nb = 4 * v.numel()
        layout.append((k, v, p32, nb, 0, 0))
        p32 += (nb + ALIGN - 1) // ALIGN * ALIGN
    p16 = p32
    has16 = False
    for i, (k, v, o32, nb32, _, _) in enumerate(layout):
        # WN 融合产物保 fp32 (见 fuse_weight_norm docstring); 其余按阈值写 fp16 段
        if v.numel() >= FP16_MIN_ELEMS and k not in wn_keys:
            nb16 = 2 * v.numel()
            layout[i] = (k, v, o32, nb32, p16, nb16)
            p16 += (nb16 + ALIGN - 1) // ALIGN * ALIGN
            has16 = True
    total = p16

    # ---- 序列化 ----
    buf = bytearray()
    buf += struct.pack("<4sIIH", MAGIC, VERSION, 1 if has16 else 0, len(name.encode()))
    buf += name.encode()
    buf += struct.pack("<I", len(cfg_bytes))
    buf += cfg_bytes
    buf += struct.pack("<I", len(layout))
    for k, v, o32, nb32, o16, nb16 in layout:
        nm = k.encode()
        buf += struct.pack("<H", len(nm)) + nm
        buf += struct.pack("<H", len(v.shape)) + struct.pack(f"<{len(v.shape)}I", *v.shape)
        buf += struct.pack("<HQQQQ", DTYPE_ENUM[v.dtype], o32, nb32, o16, nb16)
    assert len(buf) == fixed + dir_size, f"header size mismatch {len(buf)} != {fixed + dir_size}"
    pad = pos - len(buf)
    buf += b"\x00" * pad

    with open(out_p, "wb") as f:
        f.write(buf)
        for k, v, o32, nb32, o16, nb16 in layout:  # fp32 段
            a = v.float().numpy()
            f.seek(o32)
            f.write(a.tobytes())
        for k, v, _, _, o16, nb16 in layout:       # fp16 段
            if not nb16:
                continue
            f.seek(o16)
            f.write(v.half().numpy().tobytes())

    mb32 = sum(e[3] for e in layout) / 2**20
    mb16 = sum(e[5] for e in layout) / 2**20
    n16 = sum(1 for e in layout if e[5])
    print(f"[{name}] wrote {out_p}: {len(layout)} tensors "
          f"(fp16 copies: {n16}), wn_fused={n_fused}, "
          f"f32={mb32:.1f}MiB f16={mb16:.1f}MiB total={total/2**20:.1f}MiB")

    # ---- 回读校验 ----
    if verify:
        verify_file(out_p, tensors)


def verify_file(out_p: Path, ref: dict):
    raw = out_p.read_bytes()
    magic, ver, flags = struct.unpack_from("<4sII", raw, 0)
    assert magic == MAGIC and ver == VERSION, "magic/version 错误"
    nl = struct.unpack_from("<H", raw, 12)[0]
    off = 14 + nl
    cl = struct.unpack_from("<I", raw, off)[0]
    off += 4 + cl
    (n_ten,) = struct.unpack_from("<I", raw, off)
    off += 4
    checked = 0
    for _ in range(n_ten):
        (klen,) = struct.unpack_from("<H", raw, off); off += 2
        key = raw[off:off + klen].decode(); off += klen
        (rank,) = struct.unpack_from("<H", raw, off); off += 2
        dims = struct.unpack_from(f"<{rank}I", raw, off); off += 4 * rank
        src_dt, o32, nb32, o16, nb16 = struct.unpack_from("<HQQQQ", raw, off)
        off += 2 + 32
        assert o32 % ALIGN == 0 and (nb16 == 0 or o16 % ALIGN == 0), f"{key} 未对齐"
        t = torch.frombuffer(bytearray(raw[o32:o32 + nb32]), dtype=torch.float32).reshape(dims)
        r = ref[key]
        assert torch.equal(t, r.float()), f"{key} fp32 段不一致"
        if nb16:
            h = torch.frombuffer(bytearray(raw[o16:o16 + nb16]), dtype=torch.float16).reshape(dims)
            expect = r if r.dtype == torch.float16 else r.half()
            assert torch.equal(h, expect.contiguous()), f"{key} fp16 段不一致"
        checked += 1
    print(f"[verify] {out_p.name}: 结构 OK, {checked}/{n_ten} 张量数值一致")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--src"); ap.add_argument("--out"); ap.add_argument("--name")
    ap.add_argument("--config-json"); ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()
    if args.all:
        for nm, (s, _, c) in PRESETS.items():
            convert(s, f"weights/{nm}.gsv", nm, c, verify=not args.no_verify)
    elif args.src and args.out and args.name:
        convert(args.src, args.out, args.name, args.config_json, verify=not args.no_verify)
    else:
        ap.error("需要 --all 或 (--src --out --name)")


if __name__ == "__main__":
    main()
