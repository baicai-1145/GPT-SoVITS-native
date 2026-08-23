#!/usr/bin/env python3
"""export_cmudict.py — pack the CPUFast English pronunciation dictionaries
into a single binary table for the native text frontend.

Sources (mirroring text/english.py read_dict_new + hot_reload_hot):
  1. cmudict.rep      from line 57: `WORD  P1 P2 ...` (two-space separator);
                      first occurrence wins
  2. cmudict-fast.rep every line: `WORD P1 P2 ...` (single-space); fills only
                      words not already present
  3. engdict-hot.rep  every line: overrides unconditionally

Post-processing identical to en_G2p.__init__:
  - delete entries AE AI AR IOS HUD OS (wrong acronym readings)
  - homograph table from g2p_en/homographs.en, with the hardcoded
    read/complex fixes applied on top

Output layout (little-endian):
  magic  "CMUBIN1"
  u32    nPhones
  nPhones x { u8 len, bytes }             -- arpa/punct symbol strings
  u32    nWords                            -- sorted by word bytes, deduped
  nWords x { u8 wlen, bytes, u8 plen, plen x u8 phoneId }
  u32    nHomographs
  nHomographs x { u8 wlen, bytes,
                  u8 p1len, p1len x u8 phoneId,
                  u8 p2len, p2len x u8 phoneId,
                  u8 posTag }               -- first byte of pos1 or 0

Usage:
  python3 tools/export_cmudict.py <cpufast_root> <out.bin>
"""
import os
import sys


def load_cmu(cpufast: str) -> dict[str, list[list[str]]]:
    text_dir = os.path.join(cpufast, "GPT_SoVITS", "text")
    g2p_dict: dict[str, list[list[str]]] = {}

    with open(os.path.join(text_dir, "cmudict.rep"), encoding="utf-8") as f:
        line_index = 1
        for line in f:
            if line_index >= 57:
                line = line.strip()
                parts = line.split("  ")
                word = parts[0].lower()
                if word not in g2p_dict and len(parts) > 1:
                    g2p_dict[word] = [parts[1].split(" ")]
            line_index += 1

    with open(os.path.join(text_dir, "cmudict-fast.rep"), encoding="utf-8") as f:
        for line_index, line in enumerate(f, start=1):
            if line_index >= 1:
                line = line.strip()
                parts = line.split(" ")
                word = parts[0].lower()
                if word not in g2p_dict:
                    g2p_dict[word] = [parts[1:]]

    with open(os.path.join(text_dir, "engdict-hot.rep"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ")
            word = parts[0].lower()
            g2p_dict[word] = [parts[1:]]

    # wrong acronym readings removed at construction time
    for word in ["AE", "AI", "AR", "IOS", "HUD", "OS"]:
        g2p_dict.pop(word.lower(), None)

    # flatten [[p]] -> [p]; every consumer uses dict[word][0]
    return {w: v[0] for w, v in g2p_dict.items()}


def load_homographs(g2p_en_dir: str) -> dict[str, tuple[list[str], list[str], str]]:
    homographs: dict[str, tuple[list[str], list[str], str]] = {}
    with open(os.path.join(g2p_en_dir, "homographs.en"), encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            headword, pron1, pron2, pos1 = line.split("|")
            homographs[headword.lower()] = (pron1.split(), pron2.split(), pos1)
    # hard-coded corrections from en_G2p.__init__
    homographs["read"] = (["R", "IY1", "D"], ["R", "EH1", "D"], "VBP")
    homographs["complex"] = (
        ["K", "AH0", "M", "P", "L", "EH1", "K", "S"],
        ["K", "AA1", "M", "P", "L", "EH0", "K", "S"],
        "JJ",
    )
    return homographs


def main() -> int:
    cpufast = os.path.abspath(sys.argv[1])
    out_path = sys.argv[2]
    cmu = load_cmu(cpufast)
    import importlib.util

    spec = importlib.util.find_spec("g2p_en")
    homo = load_homographs(spec.submodule_search_locations[0])

    # arpa set from text/english.py plus the extra symbols that can appear
    # in normalized pron streams (punct tokens verbatim, '->-, <unk>->UNK)
    arpa = set("""AH0 S AH1 EY2 AE2 EH0 OW2 UH0 NG B G AY0 M AA0 F AO0 ER2 UH1
        IY1 AH2 DH IY0 EY1 IH0 K N W IY2 T AA1 ER1 EH2 OY0 UH2 UW1 Z AW2 AW1 V
        UW2 AA2 ER AW0 UW0 R OW1 EH1 ZH AE0 IH2 IH Y JH P AY1 EY0 OY2 TH HH D
        ER0 CH AO1 AE1 AO2 OY1 AY2 IH1 OW0 L SH""".split())
    extra = {".", ",", "!", "?", "-", "'", "UNK"}
    phones = sorted(arpa | extra)
    pid = {p: i for i, p in enumerate(phones)}

    def phid(p: str) -> int:
        if p not in pid:
            pid[p] = len(phones)
            phones.append(p)
        return pid[p]

    import struct

    buf = bytearray()
    buf += b"CMUBIN1"
    buf += struct.pack("<I", len(phones))
    for p in phones:
        b = p.encode("ascii")
        buf.append(len(b))
        buf += b
    words = sorted(cmu.keys())
    buf += struct.pack("<I", len(words))
    for w in words:
        wb = w.encode("utf-8")
        ps = [phid(x) for x in cmu[w]]
        buf.append(len(wb))
        buf += wb
        buf.append(len(ps))
        buf += bytes(ps)
    hw = sorted(homo.keys())
    buf += struct.pack("<I", len(hw))
    for w in hw:
        wb = w.encode("utf-8")
        p1 = [phid(x) for x in homo[w][0]]
        p2 = [phid(x) for x in homo[w][1]]
        pos = homo[w][2][:1].encode("ascii") if homo[w][2] else b""
        buf.append(len(wb))
        buf += wb
        buf.append(len(p1))
        buf += bytes(p1)
        buf.append(len(p2))
        buf += bytes(p2)
        buf.extend(pos if pos else b"\x00")

    with open(out_path, "wb") as f:
        f.write(bytes(buf))
    print(f"words={len(words)} homographs={len(hw)} phones={len(phones)} -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
