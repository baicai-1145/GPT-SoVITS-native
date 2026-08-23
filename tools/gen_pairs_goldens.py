#!/usr/bin/env python3
"""gen_pairs_goldens.py — export expected sentence-part phoneme ids from the
golden AR pairs for native TextFrontend parity checks.

pairs/*.pt store `phones_ids`, the exact input of the AR model's
ar_text_embedding:

    phones_ids = prompt_ids(PROMPT_LEN) ++ per_segment_text_ids

The text part comes from the CPUFast TTS runtime path:
    TTS.py: pre_seg_text(text, lang, cut0)   -> short-input "。" prepend when
            text[0] not in splits and len(get_first(text)) < 4; cut0 keeps a
            single segment
    per segment: LangSegmenter.getTexts(seg,"zh") -> clean_text_with_phone_units
            -> chinese2.text_normalize + g2p_with_phone_units (is_g2pw=True in
            the full runtime, i.e. G2PW polyphonic disambiguation IS active)

Reproduction status over the current 65-pair golden set:
    g2pw=True  reproduces 63/65 sentences (exceptions: pure/mixed English
               segments s4/s9 which need english.py cmudict g2p)
    g2pw=False (bare pypinyin, what the native frontend implements today)
               reproduces 61/65 — the extra gaps are the G2PW polyphonic
               decisions (长->chang2, 地->de5 class) that task B6 closes by
               injecting a G2PW PinyinResolver.

Usage:
  python3 tools/gen_pairs_goldens.py <repo>/tests/golden /tmp/pairs_expected.tsv

Rows: <marker><stem>\t<sentence>\t<ids-csv>  (sentence part only).
Markers prefix rows whose sentence part cannot be reproduced by the zh-only,
G2PW-free native frontend; C++ runners skip marked rows:
  #EN    non-zh segment present — needs english.py g2p (outside M3 scope)
  #G2PW  stored ids rely on G2PW polyphonic decisions — closes with B6
"""
import json
import os
import sys

PROMPT_LEN = 15  # prompt "原来你也玩原神。" -> 15 phones; constant across pairs


def tts_style_segments(sent: str, splits_set, get_first):
    """TTS.py prompt/text handling for cut0: strip newlines, prepend 。 for
    short unpunctuated inputs, single segment."""
    t = sent.strip("\n")
    if t and t[0] not in splits_set and len(get_first(t)) < 4:
        t = "。" + t
    return [t]


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    golden_dir = os.path.abspath(sys.argv[1])
    out_path = sys.argv[2]

    cpufast = os.environ.get("CPUFAST_ROOT", "/Volumes/2T/GPT-SoVITS-CPUFast")
    for p in (cpufast, os.path.join(cpufast, "GPT_SoVITS")):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.chdir(cpufast)
    os.environ.setdefault("version", "v2ProPlus")

    import torch  # noqa: E402
    from text.symbols2 import symbols  # noqa: E402
    import text.chinese2 as c2  # noqa: E402
    from TTS_infer_pack.TextPreprocessor import get_first, splits  # noqa: E402

    sym = {s: i for i, s in enumerate(symbols)}

    def seg_ids(sentence: str) -> list[int]:
        ph: list[str] = []
        for seg in tts_style_segments(sentence, splits, get_first):
            norm = c2.text_normalize(seg)
            p, _w2ph = c2.g2p(norm)
            ph.extend(p)
        return [sym[x] for x in ph]

    def has_non_zh(sentence: str) -> bool:
        # LangSegmenter routes ASCII digits into the zh segment (s3 passes);
        # only latin letters form an english segment needing cmudict g2p
        body = sentence.replace(" ", "")
        return any(ord(ch) < 128 and ch.isalpha() for ch in body)

    manifest = json.load(open(os.path.join(golden_dir, "manifest.json"), encoding="utf-8"))
    rows = []
    n_en = n_g2pw = n_plain = 0
    for entry in manifest["pairs"]:
        if "file" not in entry:
            continue
        bundle = torch.load(os.path.join(golden_dir, entry["file"]), map_location="cpu", weights_only=False)
        ids = bundle["phones_ids"][0].tolist()
        sent = bundle["sentence"]
        stored = ids[PROMPT_LEN:]

        marker = ""
        if has_non_zh(sent):
            marker, tag = "#EN", n_en
            n_en += 1
        else:
            c2.is_g2pw = False
            if seg_ids(sent) == stored:
                marker, tag = "", n_plain
                n_plain += 1
            else:
                marker, tag = "#G2PW", n_g2pw
                n_g2pw += 1
        rows.append(
            f"{marker}{entry['stem']}\t{sent}\t{','.join(map(str, stored))}"
        )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")
    print(f"exported {len(rows)} pair expectations -> {out_path}")
    print(f"  plain (native must match exactly): {n_plain}")
    print(f"  #G2PW (closes with B6 injection):  {n_g2pw}")
    print(f"  #EN   (english g2p, out of scope): {n_en}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
