#!/usr/bin/env python3
"""export_polyphone_overrides.py — B6/M3: correct_pronunciation 查表数据.

CPUFast 的 chinese2.py 在 G2PW 句级预测之后、initials/finals 转换之前,
用 text/g2pw/pronunciation.py 的 correct_pronunciation() 对每个分词单元的
读音做词典覆盖:
    1) phrase_override_dict[word]      (phrase_overrides*.pkl 合并)
    2) pp_dict[word]        (len(word)>1; polyphonic.rep + polyphonic-fix.rep)
    3) 逐字回退: pp_dict[ch][0] 替换单字位置 (其余槽位保留 G2PW 输出)

native 侧此前缺这一层 — 实测 pairs s9 "一个": G2PW 给 yi4, 而
pp_dict["一个"]=["yi2","ge4"] 整词覆盖后才是 golden 的 yi2。

输出: data/polyphone_overrides.bin (不入库, 与 cmudict.bin 同策略)
格式: "GSVPOLY1" | u32 nPhrase | phrases{u8 cpN, LE32 cps..., u8 nRd,
       rd{u8 len, bytes}}* | u32 nSingle | singles(同构)
phrases = phrase_override_dict ∪ pp_dict(len>1, override 优先),
singles = pp_dict 中单字条目 (逐字回退域)。
"""
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, "/Volumes/2T/GPT-SoVITS-CPUFast")
sys.path.insert(0, "/Volumes/2T/GPT-SoVITS-CPUFast/GPT_SoVITS")

OUT = Path(__file__).resolve().parent.parent / "src/textfront/data/polyphone_overrides.bin"


def main() -> int:
    import os

    os.chdir("/Volumes/2T/GPT-SoVITS-CPUFast")
    from text.g2pw.pronunciation import phrase_override_dict, pp_dict

    def enc_entry(word: str, readings) -> bytes:
        cps = [ord(c) for c in word]
        b = struct.pack("<B", len(cps))
        for c in cps:
            b += struct.pack("<I", c)
        b += struct.pack("<B", len(readings))
        for r in readings:
            rb = r.encode("utf-8")
            b += struct.pack("<B", len(rb)) + rb
        return b

    # phrase-level: override dict wins over pp_dict per python lookup order
    merged = {}
    for w, rds in pp_dict.items():
        if len(w) > 1:
            merged[w] = list(rds)
    for w, rds in phrase_override_dict.items():
        merged[w] = list(rds)
    singles = {w: list(rds) for w, rds in pp_dict.items() if len(w) == 1}

    out = b"GSVPOLY1"
    out += struct.pack("<I", len(merged))
    for w in sorted(merged):
        out += enc_entry(w, merged[w])
    out += struct.pack("<I", len(singles))
    for w in sorted(singles):
        out += enc_entry(w, singles[w])
    OUT.write_bytes(out)
    print(f"wrote {OUT} ({len(out)} bytes): "
          f"{len(merged)} phrase entries, {len(singles)} single chars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
