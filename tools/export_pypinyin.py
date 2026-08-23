#!/usr/bin/env python3
"""Export pypinyin data + text-front lexicon constants for B7-part1.

Inputs (read directly from the GPTSoVits env so numbers/data are exact):
  * pypinyin/phrases_dict.json   word -> [[tone-marked syllables per char]]
  * pypinyin/pinyin_dict.json    "<codepoint>" -> "pī,yīn" tone-marked list
  * CPUFast zh_normalization/char_convert.py  traditional->simplified pairs
  * CPython str.isnumeric() over the BMP      (needed by ToneSandhi)

Outputs:
  src/textfront/data/pinyin.bin     binary "GSPPYX01" (regenerable, not committed)
  src/textfront/lexicon.hpp         generated constants (committed):
                                    - ToneSandhi must/must-not neutral words
                                    - chinese2 must_erhua / not_erhua
                                    - rep_map, punctuation set
                                    - COM_QUANTIFIERS expanded literal
                                      alternatives (regex-alternation order)
                                    - measure_dict ordered pairs

Binary layout "GSPPYX01" (little-endian), same section-directory scheme as
GSVJTB01:
  meta    : u32 format_version; u32 reserved
  cps     : u32 x N                 # shared codepoint pool for keys
  phrases : u32 n; n x {u32 cp_off; u16 cp_len; u32 syl_off; u16 syl_cnt;
                        u16 pad}
            # keys sorted lexicographically by codepoint sequence; syl_off
            # points into `sylpool`, syl_cnt syllables back to back
  sylpool : per char-run: u8 nsyl; nsyl x {u8 nbytes; utf-8 bytes}
            # tone-MARKED syllables ("chóng"); a phrase entry stores one run
            # per character, heteronym=False semantics = take run's first
  chars   : u32 n; n x {u32 cp; u32 syl_off; u16 syl_cnt; u16 pad}
            # sorted by cp; values are comma-separated candidates from
            # pypinyin pinyin_dict.json in original order (first = default)
  t2s     : u32 n; n x {u32 trad_cp; u32 simp_cp}   # sorted by trad_cp
  numrange: u32 n; n x {u32 lo; u32 hi} inclusive ranges where
            chr(cp).isnumeric() on the BMP
"""
import argparse
import json
import os
import re
import struct


def u16(v): return struct.pack("<H", v)
def u32(v): return struct.pack("<I", v)


def load_lexicon(cpufast_text):
    """Import ToneSandhi/chinese2 constant tables without heavy deps."""
    import types

    def load_mod(name, path):
        ns = {}
        with open(path, encoding="utf-8") as f:
            exec(compile(f.read(), path, "exec"), ns)
        return ns

    # char_convert defines two long strings only
    cc = load_mod("char_convert",
                  os.path.join(cpufast_text, "zh_normalization", "char_convert.py"))
    t2s_pairs = []
    simp = cc["simplified_charcters"]
    trad = cc["traditional_characters"]
    assert len(simp) == len(trad)
    for s, t in zip(simp, trad):
        if t != s:
            t2s_pairs.append((ord(t), ord(s)))
    # first occurrence wins? python dict comprehension: later assignment
    # OVERWRITES earlier -> last occurrence wins.
    seen = {}
    for tc, sc in t2s_pairs:
        seen[tc] = sc  # build dict like python (last wins)
    t2s_pairs = sorted(seen.items())

    # COM_QUANTIFIERS: parse the alternation into ordered literal alternatives,
    # expanding parenthesized sub-groups in place (equivalent first-match
    # semantics since all alternatives are literals).
    num_src = open(os.path.join(cpufast_text, "zh_normalization", "num.py"),
                   encoding="utf-8").read()
    m = re.search(r'COM_QUANTIFIERS\s*=\s*"(.*?)"\n', num_src, re.S)
    raw = m.group(1)
    assert raw.startswith("(") and raw.endswith(")")
    body = raw[1:-1]

    def split_top(s):
        out, depth, cur = [], 0, ""
        for ch in s:
            if ch == "(":
                depth += 1
                cur += ch
            elif ch == ")":
                depth -= 1
                cur += ch
            elif ch == "|" and depth == 0:
                out.append(cur)
                cur = ""
            else:
                cur += ch
        out.append(cur)
        return out

    alts = []
    for alt in split_top(body):
        if "(" in alt:
            mm = re.fullmatch(r"([^(]*)\(([^)]*)\)(.*)", alt)
            assert mm, alt
            pre, group, post = mm.group(1), mm.group(2), mm.group(3)
            for g in group.split("|"):
                alts.append(pre + g + post)
        else:
            alts.append(alt)
    return t2s_pairs, alts


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--site-packages",
                    default="/Users/baicai1145/miniconda3/envs/GPTSoVits/"
                            "lib/python3.10/site-packages")
    ap.add_argument("--cpufast-text",
                    default="/Volumes/2T/GPT-SoVITS-CPUFast/GPT_SoVITS/text")
    ap.add_argument("--bin-out",
                    default=os.path.join(here, "src", "textfront", "data",
                                         "pinyin.bin"))
    ap.add_argument("--hpp-out",
                    default=os.path.join(here, "src", "textfront",
                                         "lexicon.hpp"))
    args = ap.parse_args()

    sp = args.site_packages
    with open(os.path.join(sp, "pypinyin", "phrases_dict.json"),
              encoding="utf-8") as f:
        phrases = json.load(f)
    with open(os.path.join(sp, "pypinyin", "pinyin_dict.json"),
              encoding="utf-8") as f:
        char_dict_raw = json.load(f)

    t2s_pairs, quant_alts = load_lexicon(args.cpufast_text)

    # ---- syllable pool + phrase entries ----------------------------------
    cps_pool = []
    syl_pool = bytearray()

    def add_syl_runs(runs):
        """Store per-char candidate runs; heteronym=False always takes
        run[0], so each run is u8 nsyl + syllables."""
        off = len(syl_pool)
        for run in runs:
            assert 0 < len(run) < 256
            syl_pool.extend(bytes([len(run)]))
            for s in run:
                b = s.encode("utf-8")
                assert len(b) < 256
                syl_pool.extend(bytes([len(b)]))
                syl_pool.extend(b)
        return off, len(runs)

    ph_entries = []
    for key in sorted(phrases.keys(), key=lambda w: [ord(c) for c in w]):
        val = phrases[key]           # [[syl,...], [syl], ...] one run per char
        assert all(len(run) >= 1 for run in val), key
        off, cnt = add_syl_runs(val)
        ph_entries.append((len(cps_pool), len(key), off, cnt))
        cps_pool.extend(ord(c) for c in key)

    ch_entries = []
    items = sorted(((int(k), v) for k, v in char_dict_raw.items()))
    for cp, v in items:
        cands = v.split(",")
        off, cnt = add_syl_runs([cands])   # single run: candidates of one char
        ch_entries.append((cp, off, cnt))
        cps_pool.append(cp)

    print(f"phrases={len(ph_entries)} chars={len(ch_entries)} "
          f"sylpool={len(syl_pool)}B t2s={len(t2s_pairs)} quant_alts={len(quant_alts)}")

    # ---- assemble binary ---------------------------------------------------
    def section(name, blob):
        return name.encode("ascii").ljust(8, b"\0"), blob

    meta_sec = u32(1) + u32(0)
    tags_sec = b""  # unused placeholder keeps section ids stable
    cps_sec = b"".join(u32(c) for c in cps_pool)
    ph_sec = u32(len(ph_entries)) + b"".join(
        u32(o) + u16(l) + u32(so) + u16(sc) + u16(0)
        for o, l, so, sc in ph_entries)
    syl_sec = bytes(syl_pool)
    ch_sec = u32(len(ch_entries)) + b"".join(
        u32(cp) + u32(off) + u16(cnt) + u16(0) for cp, off, cnt in ch_entries)
    t2s_sec = u32(len(t2s_pairs)) + b"".join(u32(a) + u32(b) for a, b in t2s_pairs)

    # isnumeric ranges over BMP
    ranges = []
    start = None
    for cp in range(0x10000):
        if chr(cp).isnumeric():
            if start is None:
                start = cp
        elif start is not None:
            ranges.append((start, cp - 1))
            start = None
    nr_sec = u32(len(ranges)) + b"".join(u32(a) + u32(b) for a, b in ranges)
    print(f"isnumeric ranges={len(ranges)}")

    secs = [
        section("meta", meta_sec),
        section("tags", tags_sec),
        section("cps", cps_sec),
        section("phrases", ph_sec),
        section("sylpool", syl_sec),
        section("chars", ch_sec),
        section("t2s", t2s_sec),
        section("numrange", nr_sec),
    ]
    header_size = 16 + 32 * len(secs)
    directory = bytearray()
    body = bytearray()
    for name, blob in secs:
        directory += name + u64x(header_size + len(body)) + u64x(len(blob)) + b"\0" * 8
        body += blob
    out = bytearray(b"GSPPYX01" + u32(1) + u32(len(secs)))
    out += directory
    out += body
    os.makedirs(os.path.dirname(os.path.abspath(args.bin_out)), exist_ok=True)
    with open(args.bin_out, "wb") as f:
        f.write(out)
    print(f"wrote {args.bin_out}: {len(out)} bytes")

    # ---- lexicon.hpp --------------------------------------------------------
    ts = load_sandhi_words(args.cpufast_text)
    hpp = []
    hpp.append("// lexicon.hpp — GENERATED by tools/export_pypinyin.py, DO NOT EDIT.")
    hpp.append("// Constants mirrored verbatim from CPUFast text/tone_sandhi.py and")
    hpp.append('// text/chinese2.py (must_neural_tone_words, erhua sets, rep_map,')
    hpp.append("// punctuation, COM_QUANTIFIERS literal alternatives).")
    hpp.append("#pragma once")
    hpp.append("")
    hpp.append("#include <cstddef>")
    hpp.append("#include <cstdint>")
    hpp.append("#include <utility>")
    hpp.append("")
    hpp.append("namespace gsv::textfront {")
    hpp.append("")

    def esc_cpp(s):
        return (s.replace("\\", "\\\\").replace('"', '\\"')
                 .replace("\n", "\\n").replace("\t", "\\t")
                 .replace("\r", "\\r"))

    def emit_set(name, words, comment):
        hpp.append(f"// {comment}")
        hpp.append(f"inline const char* const {name}[] = {{")
        line = "   "
        for w in sorted(words):
            esc = esc_cpp(w)
            token = f' "{esc}",'
            if len(line) + len(token) > 96:
                hpp.append(line)
                line = "   "
            line += token
        if line.strip():
            hpp.append(line)
        hpp.append("};")
        hpp.append(f"inline constexpr size_t {name}_len = sizeof({name}) / sizeof({name}[0]);")
        hpp.append("")

    emit_set("kMustNeuralToneWords", ts["must_neural_tone_words"],
             "ToneSandhi.must_neural_tone_words")
    emit_set("kMustNotNeuralToneWords", ts["must_not_neural_tone_words"],
             "ToneSandhi.must_not_neural_tone_words")

    hpp.append("// chinese2.must_erhua (order irrelevant, membership only)")
    hpp.append("inline const char* const kMustErhua[] = {")
    for w in sorted(ts["must_erhua"]):
        hpp.append(f'    "{w}",')
    hpp.append("};")
    hpp.append("inline constexpr size_t kMustErhua_len = "
               "sizeof(kMustErhua) / sizeof(kMustErhua[0]);")
    hpp.append("")
    emit_set("kNotErhua", ts["not_erhua"], "chinese2.not_erhua")

    hpp.append("// ToneSandhi.punc")
    hpp.append(f'inline const char* const kSandhiPunc = '
               '"{0}";'.format(ts["punc"].replace("\\", "\\\\").replace('"', '\\"')))
    hpp.append("")

    hpp.append("// chinese2.rep_map (ordered! applied via one regex alternation)")
    hpp.append("inline const std::pair<const char*, const char*> kRepMap[] = {")
    for k, v in ts["rep_map"]:
        ek = esc_cpp(k)
        hpp.append(f'    {{"{ek}", "{v}"}},')
    hpp.append("};")
    hpp.append("inline constexpr size_t kRepMap_len = "
               "sizeof(kRepMap) / sizeof(kRepMap[0]);")
    hpp.append("")

    hpp.append("// symbols.punctuation (chinese2 imports this from text.symbols)")
    punct = ", ".join(f'"{p}"' for p in ts["punctuation"])
    hpp.append(f"inline const char* const kPunct[] = {{{punct}}};")
    hpp.append("inline constexpr size_t kPunct_len = "
               "sizeof(kPunct) / sizeof(kPunct[0]);")
    hpp.append("")

    hpp.append("// COM_QUANTIFIERS alternatives expanded to literals, preserving")
    hpp.append("// regex alternation order (first match at a position wins).")
    hpp.append("inline const char* const kQuantifierAlts[] = {")
    for a in quant_alts:
        ea = a.replace("\\", "\\\\").replace('"', '\\"')
        hpp.append(f'    "{ea}",')
    hpp.append("};")
    hpp.append("inline constexpr size_t kQuantifierAlts_len = "
               "sizeof(kQuantifierAlts) / sizeof(kQuantifierAlts[0]);")
    hpp.append("")
    hpp.append("}  // namespace gsv::textfront")
    hpp.append("")

    with open(args.hpp_out, "w", encoding="utf-8") as f:
        f.write("\n".join(hpp))
    print(f"wrote {args.hpp_out}")


def u64x(v):
    return struct.pack("<Q", v)


def load_sandhi_words(cpufast_text):
    """Extract word sets from tone_sandhi.py + chinese2.py via ast."""
    import ast as pyast

    def sets_from(path):
        src = open(path, encoding="utf-8").read()
        tree = pyast.parse(src)
        found = {}
        for node in pyast.walk(tree):
            if isinstance(node, pyast.Assign) and len(node.targets) == 1:
                t = node.targets[0]
                name = t.id if isinstance(t, pyast.Name) else (
                    t.attr if isinstance(t, pyast.Attribute) else None)
                if name is None:
                    continue
                if isinstance(node.value, (pyast.Set, pyast.List)) and \
                        name in {"must_neural_tone_words", "must_not_neural_tone_words",
                                 "must_erhua", "not_erhua"}:
                    words = [pyast.literal_eval(el) for el in node.value.elts]
                    found[name] = words
                elif isinstance(node.value, pyast.Constant) and \
                        name == "punc":
                    found[name] = node.value.value
                elif isinstance(node.value, pyast.Dict) and name == "rep_map":
                    found[name] = [(pyast.literal_eval(k), pyast.literal_eval(v))
                                   for k, v in zip(node.value.keys, node.value.values)]
        return found

    out = sets_from(os.path.join(cpufast_text, "tone_sandhi.py"))
    out.update(sets_from(os.path.join(cpufast_text, "chinese2.py")))
    out.setdefault("rep_map", [])
    out["punctuation"] = ["!", "?", "…", ",", ".", "-"]
    return out


if __name__ == "__main__":
    main()
