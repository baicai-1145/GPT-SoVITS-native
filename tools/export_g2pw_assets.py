#!/usr/bin/env python3
"""export_g2pw_assets.py — B6: G2PW 静态资产 + compact_pypinyin 默认读音 → data/g2pw_assets.bin

口径: G2PWConverter._prepare_data 所需全部查表数据。
  - labels(1305 bopomofo+tone), polyphonic_chars_new, monophonic_chars_dict,
    char2id/char_phoneme_masks(use_mask=True), bopomofo_convert_dict
  - default_pinyin: compact_pypinyin(chinese2.py install 后) 对 U+4E00..U+9FA5 全表
    pinyin(ch, neutral_tone_with_five=True, style=Style.TONE3) 单字读音
  - t2s: tranditional_to_simplified 单字映射 (pypinyin 参考读音用)

bin 格式 "GSVG2PW2" + 小端 u32 计数段, 见 write_ 函数注释。
"""
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

CPUFAST = Path("/Volumes/2T/GPT-SoVITS-CPUFast")
sys.path.insert(0, str(CPUFAST))
sys.path.insert(0, str(CPUFAST / "GPT_SoVITS"))
os.chdir(CPUFAST)

from text.g2pw.base_api import _load_or_build_static_assets, _find_first_existing_file  # noqa: E402
from text.g2pw.torch_api import G2PWTorchConverter  # noqa: E402

OUT = Path("/Volumes/2T/wt-gsv/SOV/src/textfront/data/g2pw_assets.bin")
ROBERTA_DIR = "/Volumes/2T/GPT-SoVITS-native/pretrained_models/chinese-roberta-wwm-ext-large"

MAGIC = b"GSVG2PW2"


def w_strs(f, items):
    f.write(struct.pack("<I", len(items)))
    for s in items:
        b = s.encode("utf-8")
        f.write(struct.pack("<H", len(b)))
        f.write(b)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print("loading static assets ...", flush=True)
    a = _load_or_build_static_assets(
        str(CPUFAST / "GPT_SoVITS/text/G2PWModel"), use_char_phoneme=False, use_mask=True)

    conv = G2PWTorchConverter(model_dir=str(CPUFAST / "GPT_SoVITS/text/G2PWModel"),
                              style="pinyin", model_source=ROBERTA_DIR,
                              enable_non_tradional_chinese=True)
    # 一致性校验: conv 的资产与直接构建的相同
    assert list(a["labels"]) == list(conv.labels)
    assert a["chars"] == conv.chars

    # 默认读音表 (compact_pypinyin 已由 chinese2 install; 此处独立 import 前先装)
    from text.g2pw.compact_pypinyin import install as _install_compact
    _install_compact()
    from pypinyin import pinyin as _pinyin, Style as _Style
    from text.zh_normalization.char_convert import tranditional_to_simplified as _t2s_fn

    chars = a["chars"]                       # 3582 (模型 descriptor 索引域)
    n = 0x9FA5 - 0x4E00 + 1
    print("exporting default pinyin table (U+4E00..U+9FA5) ...", flush=True)
    defaults = []
    bad = []
    for cp in range(0x4E00, 0x9FA5 + 1):
        ch = chr(cp)
        r = _pinyin(ch, neutral_tone_with_five=True, style=_Style.TONE3,
                    errors=lambda x: [x])
        py = r[0][0]
        if py == ch or not py:
            bad.append(ch)
        defaults.append(py)
    print(f"  defaults: {len(defaults)} entries, passthrough={len(bad)}", flush=True)

    # t2s 全 BMP CJK 兼容区常用段: 只导有变化的映射
    t2s_pairs = []
    for cp in range(0x3400, 0xA000):
        ch = chr(cp)
        s = _t2s_fn(ch)
        if s != ch:
            t2s_pairs.append((ch, s))

    # BertNormalizer 单码点归一化映射表(仅存变化项):
    # 直接调 HF tokenizers 的 normalize 保证口径一致(clean_text/handle_chinese_chars/
    # lowercase+strip_accents=None 即 NFD→lower→去Mn→NFC)
    norm = conv.tokenizer._tokenizer.normalizer
    norm_pairs = []
    for cp in range(0x20, 0x10000):
        ch = chr(cp)
        try:
            ns = norm.normalize_str(ch)
        except Exception:
            continue
        if ns != ch:
            norm_pairs.append((ch, ns))
    import json as _json
    pd_path = "/Users/baicai1145/miniconda3/envs/GPTSoVits/lib/python3.10/site-packages/pypinyin/phrases_dict.json"
    with open(pd_path, encoding="utf-8") as _f:
        _pd = _json.load(_f)

    # OpenCC s2tw 三表 (opencc_s2tw.py 口径: STPhrases 词组优先 → STCharacters 单字 → TWVariants)
    from text.opencc_s2tw import _load_assets as _load_s2t
    _s2t_phrases, _s2t_chars, _tw_variants, _, _ = _load_s2t()
    print(f"  t2s pairs={len(t2s_pairs)} norm pairs={len(norm_pairs)} "
          f"phrases={len(_pd)} s2t(phr={len(_s2t_phrases)},chr={len(_s2t_chars)},var={len(_tw_variants)})", flush=True)

    with open(OUT, "wb") as f:
        f.write(MAGIC)
        w_strs(f, list(a["labels"]))
        w_strs(f, list(a["polyphonic_chars_new"]))
        mono_items = sorted(a["monophonic_chars_dict"].items())
        f.write(struct.pack("<I", len(mono_items)))
        for ch, py in mono_items:
            f.write(struct.pack("<I", ord(ch[0])))
            b = py.encode("utf-8")
            f.write(struct.pack("<H", len(b)))
            f.write(b)
        # char2id + masks
        f.write(struct.pack("<I", len(chars)))
        for i, ch in enumerate(chars):
            f.write(struct.pack("<II", ord(ch[0]), i))
        mask = np.array(a["char_phoneme_masks"], dtype=np.uint8)  # [3582,1305]
        assert mask.shape == (len(chars), len(a["labels"]))
        f.write(mask.tobytes())
        bc = sorted(a["bopomofo_convert_dict"].items())
        f.write(struct.pack("<I", len(bc)))
        for k, v in bc:
            kb, vb = k.encode(), v.encode()
            f.write(struct.pack("<H", len(kb)))
            f.write(kb)
            f.write(struct.pack("<H", len(vb)))
            f.write(vb)
        # pinyin_full: PINYIN_DICT 全量单字首音 (cp → TONE3), 覆盖所有有音汉字
        from pypinyin.core import PINYIN_DICT as _PD
        from pypinyin.contrib.tone_convert import to_tone3 as _tt3
        def _kcp(k):
            return k if isinstance(k, int) else ord(k)
        items_pf = sorted(
            (_kcp(k), _tt3((v.split(",")[0] if isinstance(v, str) else v[0]),
                           neutral_tone_with_five=True))
            for k, v in _PD.items())
        f.write(struct.pack("<I", len(items_pf)))
        for cp, sy in items_pf:
            b = sy.encode("utf-8")
            f.write(struct.pack("<IH", cp, len(b)))
            f.write(b)
        f.write(struct.pack("<I", len(t2s_pairs)))
        for k, v in t2s_pairs:
            kb, vb = k.encode(), v.encode()
            f.write(struct.pack("<H", len(kb)))
            f.write(kb)
            f.write(struct.pack("<H", len(vb)))
            f.write(vb)
        # phrases: word -> 逐字音节列表 (TONE3 数字声调)
        items = sorted(_pd.items())
        f.write(struct.pack("<I", len(items)))
        for w, syls in items:
            wb = w.encode("utf-8")
            from pypinyin.contrib.tone_convert import to_tone3 as _to_tone3
            flat = [_to_tone3(per_char[0], neutral_tone_with_five=True)
                    for per_char in syls]
            f.write(struct.pack("<H", len(wb)))
            f.write(wb)
            f.write(struct.pack("<H", len(flat)))
            for sy in flat:
                b = sy.encode("utf-8")
                f.write(struct.pack("<H", len(b)))
                f.write(b)
        # s2t: phrases(word→word) / chars(cp→utf8) / variants(cp→utf8)
        items_sp = sorted(_s2t_phrases.items())
        f.write(struct.pack("<I", len(items_sp)))
        for k, v in items_sp:
            kb, vb = k.encode("utf-8"), v.encode("utf-8")
            f.write(struct.pack("<H", len(kb))); f.write(kb)
            f.write(struct.pack("<H", len(vb))); f.write(vb)
        def _w_cp_map(m):
            f.write(struct.pack("<I", len(m)))
            for k, v in sorted(m.items()):
                f.write(struct.pack("<I", ord(k)))
                b = v.encode("utf-8")
                f.write(struct.pack("<H", len(b))); f.write(b)
        _w_cp_map(_s2t_chars)
        _w_cp_map(_tw_variants)
        # normalize 映射 (BertNormalizer 口径)
        f.write(struct.pack("<I", len(norm_pairs)))
        for k, v in norm_pairs:
            kb, vb = k.encode(), v.encode()
            f.write(struct.pack("<I", ord(k)))
            f.write(struct.pack("<H", len(vb)))
            f.write(vb)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
