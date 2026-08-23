#!/usr/bin/env python3
"""Export jieba_fast dictionary + POS HMM models into a single binary file.

Usage:
    /path/to/GPTSoVits-env-python tools/export_jieba_trie.py \
        [--dict PATH_TO_dict.txt] [--out src/textfront/data/jieba_trie.bin]

The dict/HMM values are imported directly from the GPTSoVits env's
jieba_fast package so numbers are bit-exact (f64 via struct.pack).

Output format "GSVJTB01" (all integers little-endian):
    Header:
      0   magic "GSVJTB01"                     8B
      8   u32 format_version = 1               4B
      12  u32 n_sections                       4B
      16  n_sections x { name[8], u64 off, u64 nbytes, u64 reserved }  (32B each)
    Sections:
      dictmeta : i64 total            # sum of per-line freqs (duplicate words
                                      # double-counted, mirroring gen_pfdict)
      tags     : u32 n; n x {u8 len, utf8 bytes}   # unique POS tags, sorted
      cps      : u32 x N              # codepoint pool, concatenated entry keys
      entries  : u32 n; n x {u32 cp_off; u16 cp_len; u16 tag_id; i64 freq}
                 # one entry per key in jieba's FREQ dict = every real word +
                 # every proper prefix of every word (prefix -> freq==0,
                 # tag_id==0xFFFF). Sorted lexicographically by codepoint
                 # sequence so C++ can binary-search exact/prefix membership.
      states   : u32 n; n x {u8 bmes_ascii; u16 pos_tag_id; u8 pad}
                 # POS-HMM states as (B/M/E/S, tag) pairs, sorted by
                 # (bmes_char, tag_string) lexicographic order; the integer
                 # state id then orders exactly like python's tuple compare,
                 # which viterbi tie-breaking relies on.
      hmmstart : f64 x n                      # log P_start[state]
      hmmtrans : for each state s in id order:
                   u32 nnz; nnz x {u16 dst; u16 pad; f64 p}
                 # missing transition == -inf (posseg viterbi MIN_INF)
      hmmitmap : for each state: u32 n; n x {u32 cp; f64 p}   # sorted by cp
                 # missing emission == -3.14e100 (MIN_FLOAT), NOT -inf
      charstate: u32 nchars; nchars x {u32 cp; u32 first; u32 cnt};
                 u32 list_len; list_len x u16 state_id
                 # char -> ordered candidate state ids (source order kept;
                 # chars absent here fall back to ALL states in viterbi init)
"""
import argparse
import importlib
import os
import struct
import sys


def u16(v): return struct.pack("<H", v)
def u32(v): return struct.pack("<I", v)
def u64(v): return struct.pack("<Q", v)
def i64(v): return struct.pack("<q", v)
def f64(v): return struct.pack("<d", v)


def load_jieba(env_site_packages):
    sys.path.insert(0, env_site_packages)
    import jieba_fast  # noqa: F401  (initializes package globals)
    import jieba_fast.posseg as psg
    return psg


def build_freq_and_tags(dict_path):
    """Mirror Tokenizer.gen_pfdict + POSTokenizer.load_word_tag exactly."""
    freq = {}          # key: str(word or prefix) -> int
    tag_tab = {}       # word -> tag (last occurrence wins)
    total = 0
    with open(dict_path, "rb") as f:
        for raw in f:
            line = raw.strip().decode("utf-8")
            if not line:
                continue  # gen_pfdict would raise; dict.txt has none
            word, fstr = line.split(" ")[:2]
            freq_i = int(fstr)
            freq[word] = freq_i
            total += freq_i
            for ch in range(len(word)):
                wfrag = word[: ch + 1]
                if wfrag not in freq:
                    freq[wfrag] = 0
            _, _, tag = line.split(" ")
            tag_tab[word] = tag
    return freq, tag_tab, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dict", default=None,
                    help="defaults to <site-packages>/jieba_fast/dict.txt")
    ap.add_argument("--site-packages", default=None,
                    help="GPTSoVits env site-packages dir containing jieba_fast")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "src", "textfront", "data", "jieba_trie.bin"))
    args = ap.parse_args()

    if args.site_packages is None:
        cands = [
            "/Users/baicai1145/miniconda3/envs/GPTSoVits/lib/python3.10/site-packages",
        ]
        args.site_packages = next(c for c in cands if os.path.isdir(c))

    if args.dict is None:
        args.dict = os.path.join(args.site_packages, "jieba_fast", "dict.txt")

    psg = load_jieba(args.site_packages)

    freq, tag_tab, total = build_freq_and_tags(args.dict)
    print(f"dict: {sum(1 for v in freq.values() if v > 0)} words, "
          f"{len(freq)} keys incl. prefixes, total={total}")

    # ---- tags -----------------------------------------------------------
    # pool = dict tags ∪ tags referenced by the POS-HMM state tables
    # (e.g. 'bg' exists in char_state_tab but not in dict.txt)
    hmm_tags = {tg for bm, tg in psg.trans_P.keys()}
    tags = sorted(set(tag_tab.values()) | hmm_tags)
    tag_id = {t: i for i, t in enumerate(tags)}
    TAG_NONE = 0xFFFF
    print(f"tags: {len(tags)}")

    # ---- trie entries ----------------------------------------------------
    cps_pool = []
    entries = []  # (cp_off, cp_len, tag_id, freq)
    for key in sorted(freq.keys(), key=lambda w: [ord(c) for c in w]):
        cps = [ord(c) for c in key]
        entries.append((len(cps_pool), len(cps),
                        tag_id.get(tag_tab.get(key), TAG_NONE) if key in tag_tab else TAG_NONE,
                        freq[key]))
        cps_pool.extend(cps)
    print(f"trie: {len(entries)} entries, {len(cps_pool)} codepoints")

    # ---- POS HMM ---------------------------------------------------------
    start_P, trans_P, emit_P = psg.start_P, psg.trans_P, psg.emit_P
    char_state_tab = psg.char_state_tab_P

    all_states = list(trans_P.keys())                       # source order
    states_sorted = sorted(all_states, key=lambda st: (st[0], st[1]))
    sid = {st: i for i, st in enumerate(states_sorted)}
    S = len(states_sorted)
    assert len(set(states_sorted)) == S
    for st in states_sorted:
        assert st in start_P, f"start_P missing {st}"
    for char, sts in char_state_tab.items():
        for st in sts:
            assert st in sid, f"char_state_tab references unknown state {st}"
    print(f"hmm: {S} states, "
          f"{sum(len(v) for v in emit_P.values())} emissions, "
          f"{len(char_state_tab)} chars in char_state_tab")

    bmes = bytearray()
    stag = []
    for bm, tg in states_sorted:
        assert len(bm) == 1 and bm in "BMES"
        bmes.append(ord(bm))
        stag.append(u16(tag_id[tg]))

    start_blob = b"".join(f64(start_P[st]) for st in states_sorted)

    trans_blobs = []
    n_trans = 0
    for st in states_sorted:
        row = trans_P[st]
        items = sorted(((sid[d], p) for d, p in row.items()))
        trans_blobs.append(u32(len(items)) +
                           b"".join(u16(d) + u16(0) + f64(p) for d, p in items))
        n_trans += len(items)

    emit_blobs = []
    n_emit = 0
    for st in states_sorted:
        em = emit_P[st]
        items = sorted(((ord(c), p) for c, p in em.items()),
                       key=lambda t: t[0])
        for c, _ in items:
            assert c <= 0xFFFF, f"non-BMP emission U+{c:04X}"
        emit_blobs.append(u32(len(items)) +
                          b"".join(u32(c) + f64(p) for c, p in items))
        n_emit += len(items)

    cs_chars = sorted(char_state_tab.keys(), key=ord)
    cs_list = []
    cs_entries = []
    for c in cs_chars:
        ids = [sid[st] for st in char_state_tab[c]]
        cs_entries.append((ord(c), len(cs_list), len(ids)))
        cs_list.extend(ids)

    # ---- assemble --------------------------------------------------------
    def section(name, blob):
        return name.encode("ascii").ljust(8, b"\0"), blob

    secs = [
        section("dictmeta", i64(total)),
        section("tags", u32(len(tags)) +
                b"".join(bytes([len(t.encode())]) + t.encode() for t in tags)),
        section("cps", b"".join(u32(c) for c in cps_pool)),
        section("entries", u32(len(entries)) +
                b"".join(u32(o) + u16(l) + u16(t) + i64(fr)
                         for o, l, t, fr in entries)),
        section("states", u32(S) +
                b"".join(bytes([b]) + t + b"\0" for b, t in zip(bmes, stag))),
        section("hmmstart", start_blob),
        section("hmmtrans", b"".join(trans_blobs)),
        section("hmmitmap", b"".join(emit_blobs)),
        section("chars",
                u32(len(cs_entries)) +
                b"".join(u32(c) + u32(f) + u32(n) for c, f, n in cs_entries) +
                u32(len(cs_list)) + b"".join(u16(s) for s in cs_list)),
    ]

    header_size = 16 + 32 * len(secs)
    body = bytearray()
    directory = bytearray()
    for name, blob in secs:
        directory += name + u64(header_size + len(body)) + u64(len(blob)) + b"\0" * 8
        body += blob

    out = bytearray()
    out += b"GSVJTB01"
    out += u32(1)
    out += u32(len(secs))
    out += directory
    out += body

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(out)
    print(f"wrote {args.out}: {len(out)} bytes "
          f"(trans={n_trans}, emit={n_emit})")


if __name__ == "__main__":
    main()
