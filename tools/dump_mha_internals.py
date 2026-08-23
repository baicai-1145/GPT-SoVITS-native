#!/opt/homebrew/bin/python3/bin/python3
"""dump_mha_internals.py — 导出 encoder_ssl 第 0 层 MHA 内部中间张量"""
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/Volumes/2T/GPT-SoVITS-native")
CPUFAST = Path("/Volumes/2T/GPT-SoVITS-CPUFast")
sys.path.insert(0, str(CPUFAST))
sys.path.insert(0, str(CPUFAST / "GPT_SoVITS"))
import os
os.chdir(CPUFAST)

from golden_export import Golden  # noqa: E402

OUT = REPO / "tests/golden/sovits_fixtures/vo_HTLQ001_3_hutao_16__s0/mha_dbg"
OUT.mkdir(parents=True, exist_ok=True)

FIX = REPO / "tests/golden/sovits_fixtures/vo_HTLQ001_3_hutao_16__s0"


def loadb(p, shape):
    return torch.from_numpy(np.fromfile(str(p), dtype=np.float32).copy()).view(shape)


g = Golden()
vm = g.tts.vits_model

meta = __import__("json").load(open(FIX / "meta.json"))
x_in = loadb(FIX / "hooks/h_encp_input.bin", tuple(meta["hooks"]["h_encp_input"]))
ge_text = loadb(FIX / "hooks/h_encp_ge.bin", tuple(meta["hooks"]["h_encp_ge"]))

enc = vm.enc_p


def save(name, t):
    t = t.detach().float().cpu().contiguous()
    t.numpy().tofile(OUT / f"{name}.bin")
    (OUT / f"{name}.shape").write_text(" ".join(map(str, t.shape)))


# 复刻 TextEncoder.forward 到 encoder_ssl
with torch.no_grad():
    y_mask = torch.ones(1, 1, x_in.shape[-1])
    y = enc.ssl_proj(x_in * y_mask) * y_mask
    save("y_after_sslproj", y)
    mod = enc.encoder_ssl.attn_layers[0]
    orig_attention = type(mod).attention

    captured = {}

    def patched(self, query, key, value, mask=None):
        import math
        b, d, t_s, t_t = (*key.size(), query.size(2))
        q = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
        k = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        v = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        captured["q"] = q.clone()
        captured["k"] = k.clone()
        captured["v"] = v.clone()
        scores = torch.matmul(q / math.sqrt(self.k_channels),
                              k.transpose(-2, -1))
        captured["scores_core"] = scores.clone()
        kr = self._get_relative_embeddings(self.emb_rel_k, t_s)
        rel_logits = self._matmul_with_relative_keys(
            q / math.sqrt(self.k_channels), kr)
        captured["rel_logits"] = rel_logits.clone()
        scores_local = self._relative_position_to_absolute_position(rel_logits)
        captured["rel_abs"] = scores_local.clone()
        scores = scores + scores_local
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
        p = torch.nn.functional.softmax(scores, dim=-1)
        captured["p_attn"] = p.clone()
        output = torch.matmul(p, v)
        rw = self._absolute_position_to_relative_position(p)
        vr = self._get_relative_embeddings(self.emb_rel_v, t_s)
        captured["prel"] = rw.clone()
        output = output + self._matmul_with_relative_values(rw, vr)
        captured["attn_out"] = output.clone()
        return output.transpose(2, 3).contiguous().view(b, d, t_t), p

    mod.attention = patched.__get__(mod, type(mod))
    try:
        y2 = enc.encoder_ssl(y * y_mask, y_mask)
        save("y_after_encoder_ssl", y2)
    finally:
        del type(mod).attention
    for name, t in captured.items():
        save(name, t)
print("saved to", OUT)
