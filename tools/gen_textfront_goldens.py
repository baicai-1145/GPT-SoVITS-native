#!/usr/bin/env python3
"""gen_textfront_goldens.py — python reference goldens for B7 textfront.

Usage:
  gen_textfront_goldens.py norm <input.txt> <out.golden>
      TextNormalizer chain + replace_punctuation +
      replace_consecutive_punctuation per line.
  gen_textfront_goldens.py g2p <input.txt> <out.golden>
      text_normalize -> g2p (is_g2pw monkeypatched False) -> symbols2 ids
      per line; rows: P<TAB>idx<TAB>phones<TAB>word2ph<TAB>ids, or
      E<TAB>idx<TAB>ExcName when the reference raises.

The G2PW torch converter is stubbed before importing chinese2 so the
reference runs CPU-only and takes the pypinyin path.
"""
import sys
import types


def _stub_g2pw():
    fake = types.ModuleType("text.g2pw.torch_api")

    class _FakeConverter:
        def __init__(self, *a, **k):
            pass

    fake.G2PWTorchConverter = _FakeConverter
    sys.modules["text.g2pw.torch_api"] = fake


def esc(s):
    return (s.replace("\\", "\\\\").replace("\n", "\\n")
            .replace("\t", "\\t").replace("\r", "\\r"))


def read_lines(path):
    lines = open(path, encoding="utf-8").read().split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def run_norm(inp, outp):
    sys.path.insert(0, "/Volumes/2T/GPT-SoVITS-CPUFast/GPT_SoVITS")
    import re
    from text.zh_normalization.text_normlization import TextNormalizer
    punctuation = ["!", "?", "…", ",", "."]
    punctuation.append("-")
    rep_map = {
        "：": ",", "；": ",", "，": ",", "。": ".", "！": "!", "？": "?",
        "\n": ".", "·": ",", "、": ",", "...": "…", "$": ".", "/": ",",
        "—": "-", "~": "…", "～": "…",
    }
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))
    keep = re.compile(r"[^\u4e00-\u9fa5" + "".join(punctuation) + r"]")

    def replace_punctuation(text):
        text = text.replace("嗯", "恩").replace("呣", "母")
        return keep.sub("", pattern.sub(lambda x: rep_map[x.group()], text))

    coll = re.compile(
        f"([{ ''.join(re.escape(p) for p in punctuation)}])"
        f"([{ ''.join(re.escape(p) for p in punctuation)}])+")

    tn = TextNormalizer()
    with open(outp, "w", encoding="utf-8") as f:
        for idx, text in enumerate(read_lines(inp)):
            try:
                sents = tn.normalize(text)
                dest = "".join(replace_punctuation(s) for s in sents)
                dest = coll.sub(r"\1", dest)
                f.write(f"N\t{idx}\t{esc(dest)}\n")
            except Exception as e:
                f.write(f"E\t{idx}\t{type(e).__name__}\n")


def run_g2p(inp, outp):
    sys.path.insert(0, "/Volumes/2T/GPT-SoVITS-CPUFast/GPT_SoVITS")
    _stub_g2pw()
    import logging
    logging.disable(logging.CRITICAL)
    from text import chinese2, cleaned_text_to_sequence
    chinese2.is_g2pw = False
    with open(outp, "w", encoding="utf-8") as f:
        for idx, text in enumerate(read_lines(inp)):
            try:
                norm = chinese2.text_normalize(text)
                phones, word2ph = chinese2.g2p(norm)
                ids = cleaned_text_to_sequence(phones, version="v2")
                f.write("P\t%d\t%s\t%s\t%s\n" % (
                    idx, esc(" ".join(phones)),
                    ",".join(map(str, word2ph)),
                    ",".join(map(str, ids))))
            except Exception as e:
                f.write(f"E\t{idx}\t{type(e).__name__}\n")


def _esc_join(parts):
    return esc("\n".join(esc(p) for p in parts))


def run_split(inp, outp):
    """pre_seg_text reference: S<TAB>idx<TAB>esc('\\n'.join(esc(seg))) rows.

    Mirrors TextPreprocessor.preprocess for lang="zh": global
    replace_consecutive_punctuation, then pre_seg_text with cutN.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "tsm", "/Volumes/2T/GPT-SoVITS-CPUFast/GPT_SoVITS/"
        "TTS_infer_pack/text_segmentation_method.py")
    tsm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tsm)
    get_seg_method = tsm.get_method
    split_big_text = tsm.split_big_text
    splits = tsm.splits
    import re

    punctuation = set(["!", "?", "…", ",", ".", "-", " "])
    punct_chinese = ["!", "?", "…", ",", ".", "-"]
    coll_pat = re.compile(
        f"([{ ''.join(re.escape(p) for p in punct_chinese)}])"
        f"([{ ''.join(re.escape(p) for p in punct_chinese)}])+")

    def replace_consecutive_punctuation(text):
        return coll_pat.sub(r"\1", text)

    def get_first(text):
        pat = "[" + "".join(re.escape(s) for s in splits) + "]"
        return re.split(pat, text)[0].strip()

    def filter_text(texts):
        return [t for t in texts if t not in [None, " ", ""]]

    def merge_short_text_in_array(texts, threshold):
        if len(texts) < 2:
            return texts
        result, acc = [], ""
        for ele in texts:
            acc += ele
            if len(acc) >= threshold:
                result.append(acc)
                acc = ""
        if acc:
            if result:
                result[-1] += acc
            else:
                result.append(acc)
        return result

    def pre_seg_text(text, method):
        text = text.strip("\n")
        if len(text) == 0:
            return []
        if text[0] not in splits and len(get_first(text)) < 4:
            text = "。" + text
        seg = get_seg_method(method)
        text = seg(text)
        while "\n\n" in text:
            text = text.replace("\n\n", "\n")
        texts = filter_text(text.split("\n"))
        texts = merge_short_text_in_array(texts, 5)
        out = []
        for item in texts:
            if len(item.strip()) == 0:
                continue
            if not re.sub(r"\W+", "", item):
                continue
            if item[-1] not in splits:
                item += "。"
            if len(item) > 510:
                out.extend(split_big_text(item))
            else:
                out.append(item)
        return out

    with open(outp, "w", encoding="utf-8") as f:
        for idx, line in enumerate(read_lines(inp)):
            # fixture lines carry "<method>|<text>"
            method, _, text = line.partition("|")
            try:
                segs = pre_seg_text(replace_consecutive_punctuation(text),
                                    method.strip())
                f.write(f"S\t{idx}\t{_esc_join(segs)}\n")
            except Exception as e:
                f.write(f"E\t{idx}\t{type(e).__name__}\n")


def run_process(inp, outp):
    """Full frontend golden: P rows carry per-line concatenated phones ids,
    word2ph, and the segment list; E rows mark reference exceptions."""
    sys.path.insert(0, "/Volumes/2T/GPT-SoVITS-CPUFast/GPT_SoVITS")
    _stub_g2pw()
    import logging
    logging.disable(logging.CRITICAL)
    from text import chinese2, cleaned_text_to_sequence
    chinese2.is_g2pw = False
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "tsm2", "/Volumes/2T/GPT-SoVITS-CPUFast/GPT_SoVITS/"
        "TTS_infer_pack/text_segmentation_method.py")
    tsm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tsm)
    splits = tsm.splits
    get_seg_method = tsm.get_method
    split_big_text = tsm.split_big_text
    import re
    punct_chinese = ["!", "?", "…", ",", ".", "-"]
    coll_pat = re.compile(
        f"([{ ''.join(re.escape(p) for p in punct_chinese)}])"
        f"([{ ''.join(re.escape(p) for p in punct_chinese)}])+")

    def get_first(text):
        pat = "[" + "".join(re.escape(s) for s in splits) + "]"
        return re.split(pat, text)[0].strip()

    def filter_text(texts):
        return [t for t in texts if t not in [None, " ", ""]]

    def merge_short_text_in_array(texts, threshold):
        if len(texts) < 2:
            return texts
        result, acc = [], ""
        for ele in texts:
            acc += ele
            if len(acc) >= threshold:
                result.append(acc)
                acc = ""
        if acc:
            if result:
                result[-1] += acc
            else:
                result.append(acc)
        return result

    def pre_seg_text(text, method):
        text = text.strip("\n")
        if len(text) == 0:
            return []
        if text[0] not in splits and len(get_first(text)) < 4:
            text = "。" + text
        seg = get_seg_method(method)
        text = seg(text)
        while "\n\n" in text:
            text = text.replace("\n\n", "\n")
        texts = filter_text(text.split("\n"))
        texts = merge_short_text_in_array(texts, 5)
        out = []
        for item in texts:
            if len(item.strip()) == 0:
                continue
            if not re.sub(r"\W+", "", item):
                continue
            if item[-1] not in splits:
                item += "。"
            if len(item) > 510:
                out.extend(split_big_text(item))
            else:
                out.append(item)
        return out

    with open(outp, "w", encoding="utf-8") as f:
        for idx, line in enumerate(read_lines(inp)):
            method, _, text = line.partition("|")
            try:
                collapsed = coll_pat.sub(r"\1", text)
                segs = pre_seg_text(collapsed, method.strip())
                all_ids, all_w2ph = [], []
                for seg in segs:
                    norm = chinese2.text_normalize(seg)
                    phones, w2ph = chinese2.g2p(norm)
                    all_ids.extend(cleaned_text_to_sequence(phones, version="v2"))
                    all_w2ph.extend(w2ph)
                f.write("P\t%d\t%s\t%s\t%s\n" % (
                    idx, esc(",".join(map(str, all_ids))),
                    ",".join(map(str, all_w2ph)),
                    _esc_join(segs)))
            except Exception as e:
                f.write(f"E\t{idx}\t{type(e).__name__}\n")


def run_process_en(inp, outp):
    """Mixed zh/en frontend golden mirroring the CPUFast TTS runtime path:
    preprocess(cut) -> per segment LangSegmenter.getTexts(text,"zh") ->
    zh segments via chinese2 (is_g2pw=False), en segments via english.g2p.
    P rows: idx, ids, w2ph, segs. En-segment word2ph entries are per-token
    pronunciation lengths (phone_units granularity) so that
    sum(word2ph)==len(phones) holds globally."""
    sys.path.insert(0, "/Volumes/2T/GPT-SoVITS-CPUFast/GPT_SoVITS")
    _stub_g2pw()
    import logging
    logging.disable(logging.CRITICAL)
    from text import chinese2, cleaned_text_to_sequence
    chinese2.is_g2pw = False
    from text import english as eng
    from text.LangSegmenter import LangSegmenter
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "tsm3", "/Volumes/2T/GPT-SoVITS-CPUFast/GPT_SoVITS/"
        "TTS_infer_pack/text_segmentation_method.py")
    tsm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tsm)
    splits = tsm.splits
    get_seg_method = tsm.get_method
    split_big_text = tsm.split_big_text
    import re
    punct_chinese = ["!", "?", "\u2026", ",", ".", "-"]
    coll_pat = re.compile(
        f"([{''.join(re.escape(p) for p in punct_chinese)}])"
        f"([{''.join(re.escape(p) for p in punct_chinese)}])+")

    def get_first(text):
        pat = "[" + "".join(re.escape(x) for x in splits) + "]"
        return re.split(pat, text)[0].strip()

    def filter_text(texts):
        return [t for t in texts if t not in [None, " ", ""]]

    def merge_short_text_in_array(texts, threshold):
        if len(texts) < 2:
            return texts
        result, acc = [], ""
        for ele in texts:
            acc += ele
            if len(acc) >= threshold:
                result.append(acc)
                acc = ""
        if acc:
            if result:
                result[-1] += acc
            else:
                result.append(acc)
        return result

    def pre_seg_text(text, method):
        text = text.strip("\n")
        if len(text) == 0:
            return []
        if text[0] not in splits and len(get_first(text)) < 4:
            text = "\u3002" + text
        seg = get_seg_method(method)
        text = seg(text)
        while "\n\n" in text:
            text = text.replace("\n\n", "\n")
        texts = filter_text(text.split("\n"))
        texts = merge_short_text_in_array(texts, 5)
        out = []
        for item in texts:
            if len(item.strip()) == 0:
                continue
            if not re.sub(r"\W+", "", item):
                continue
            if item[-1] not in splits:
                item += "\u3002"
            if len(item) > 510:
                out.extend(split_big_text(item))
            else:
                out.append(item)
        return out

    with open(outp, "w", encoding="utf-8") as f:
        for idx, line in enumerate(read_lines(inp)):
            method, _, text = line.partition("|")
            try:
                collapsed = coll_pat.sub(r"\1", text)
                segs = pre_seg_text(collapsed, method.strip())
                all_ids, all_w2ph = [], []
                for seg in segs:
                    for lseg in LangSegmenter.getTexts(seg, "zh"):
                        lang, stext = lseg["lang"], lseg["text"]
                        if lang == "en":
                            ph = eng.g2p(eng.text_normalize(stext))
                            # en tokens: word2ph at token granularity — count
                            # phones contributed per simple_word_tokenize
                            # token (gap tokens like spaces contribute none;
                            # punctuation tokens contribute exactly 1)
                            import re as _re
                            toks = eng.simple_word_tokenize(
                                eng.text_normalize(stext))
                            counts = []
                            for tk in toks:
                                n = len(tk) and 1 or 1
                                counts.append(max(n, 1))
                            all_ids.extend(
                                cleaned_text_to_sequence(ph, version="v2"))
                            # exact per-token lengths recomputed below
                            counts = []
                            cursor = 0
                            from text.english import normalize_pronunciation
                            for tk in toks:
                                pron = eng._g2p.pronounce_token(
                                    tk, "")
                                np_ = normalize_pronunciation(pron)
                                counts.append(len(np_))
                                cursor += len(np_)
                            all_w2ph.extend(counts)
                        else:
                            phones, w2ph = chinese2.g2p(
                                chinese2.text_normalize(stext))
                            all_ids.extend(
                                cleaned_text_to_sequence(phones,
                                                         version="v2"))
                            all_w2ph.extend(w2ph)
                f.write("P\t%d\t%s\t%s\t%s\n" % (
                    idx, esc(",".join(map(str, all_ids))),
                    ",".join(map(str, all_w2ph)),
                    _esc_join(segs)))
            except Exception as e:
                f.write(f"E\t{idx}\t{type(e).__name__}\n")


if __name__ == "__main__":
    mode, inp, outp = sys.argv[1], sys.argv[2], sys.argv[3]
    if mode == "norm":
        run_norm(inp, outp)
    elif mode == "g2p":
        run_g2p(inp, outp)
    elif mode == "split":
        run_split(inp, outp)
    elif mode == "process":
        run_process(inp, outp)
    elif mode == "process_en":
        run_process_en(inp, outp)
    else:
        raise SystemExit(f"unknown mode {mode}")
