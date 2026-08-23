#!/usr/bin/env python
"""check_b12.py — B12 验收判定: G1 张量级 + G2 解码级, 门槛见 tests/golden/gates.json。

G1: logits_first8 与 logits_last 对 golden cos>=0.9999 且 rel=max|a-b|/max|b| <= 1e-3
    (口径同 tools/golden_export.py::_rel_cos)
G2: 每步未惩罚 argmax 序列 vs golden `tokens` 一致率 >= 98%, 长度比 ∈ [0.8,1.25]
    稳定对(golden 步数 >= 79, CALIBRATION.md 口径)作硬门; 短对单独记录不作硬门。

用法: python tools/check_b12.py <raw_dir> <native_out_dir>
"""
import json
import math
import os
import sys


def f32(path):
    import struct
    with open(path, "rb") as f:
        b = f.read()
    n = len(b) // 4
    return struct.unpack(f"<{n}f", b[: n * 4])


def i32(path):
    import struct
    with open(path, "rb") as f:
        b = f.read()
    n = len(b) // 4
    return struct.unpack(f"<{n}i", b[: n * 4])


def cos_rel(a, b):
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    cos = dot / (na * nb) if na > 0 and nb > 0 else 0.0
    mb = max(abs(y) for y in b) or 1e-6
    rel = max(abs(x - y) for x, y in zip(a, b)) / mb
    return cos, rel


def main():
    raw_dir, out_dir = sys.argv[1], sys.argv[2]
    stems = sorted(d for d in os.listdir(raw_dir)
                   if os.path.isdir(os.path.join(raw_dir, d)))
    rows = []
    n_g1_fail = n_g2_fail_stable = n_run = 0
    tot_decode_ms = tot_prefill_ms = tot_steps = 0.0
    for stem in stems:
        rd = os.path.join(raw_dir, stem)
        nd = os.path.join(out_dir, stem)
        if not os.path.isdir(nd):
            print(f"[缺] {stem}: 无 native 输出")
            continue
        gt = i32(os.path.join(rd, "g_tokens.i32"))
        nt = i32(os.path.join(nd, "out_tokens.i32"))
        gl8 = f32(os.path.join(rd, "g_logits_first8.f32"))
        nl8 = f32(os.path.join(nd, "out_logits_first8.f32"))
        gll = f32(os.path.join(rd, "g_logits_last.f32"))
        nll = f32(os.path.join(nd, "out_logits_last.f32"))
        with open(os.path.join(rd, "meta.txt")) as f:
            T, P, gs = map(int, f.read().split())
        with open(os.path.join(nd, "out_meta.txt")) as f:
            steps_native, hit_eos, prefill_ms, decode_ms, wall_ms = f.read().split()
        steps_native, hit_eos = int(steps_native), int(hit_eos)
        prefill_ms, decode_ms, wall_ms = map(float, (prefill_ms, decode_ms, wall_ms))

        c8, r8 = cos_rel(nl8, gl8)
        cl, rl = cos_rel(nll, gll)
        g1_ok = c8 >= 0.9999 and r8 <= 1e-3 and cl >= 0.9999 and rl <= 1e-3

        m = min(len(gt), len(nt))
        agree = (sum(1 for i in range(m) if gt[i] == nt[i]) / max(len(gt), 1))
        len_ratio = steps_native / max(gs, 1)
        g2_ok = agree >= 0.98 and 0.8 <= len_ratio <= 1.25
        stable = gs >= 79

        ok = g1_ok and (g2_ok or not stable)
        n_g1_fail += (not g1_ok)
        n_g2_fail_stable += (stable and not g2_ok)
        n_run += 1
        tot_prefill_ms += prefill_ms
        tot_decode_ms += decode_ms
        tot_steps += steps_native
        rows.append(dict(stem=stem, gs=gs, ns=steps_native, T=T, P=P,
                         cos8=round(c8, 7), rel8=f"{r8:.2e}",
                         cosl=round(cl, 7), rell=f"{rl:.2e}",
                         agree=round(agree, 4), lratio=round(len_ratio, 3),
                         eos=hit_eos, stable=int(stable), ok=int(ok),
                         g2_ok=int(g2_ok)))

    # ---- 报表 ----
    hdr = (f"{'pair':<46} {'gs':>4} {'ns':>4} {'cos8':>9} {'rel8':>8} "
           f"{'cosL':>9} {'relL':>8} {'agre':>6} {'lr':>5} eos ok")
    print(hdr)
    for r in rows:
        flag = "" if r["ok"] else ("  ←FAIL")
        print(f"{r['stem']:<46} {r['gs']:>4} {r['ns']:>4} {r['cos8']:>9} "
              f"{r['rel8']:>8} {r['cosl']:>9} {r['rell']:>8} "
              f"{r['agree']:>6} {r['lratio']:>5} {r['eos']}   {r['stable']}{flag}")

    stable_rows = [r for r in rows if r["stable"]]
    short_rows = [r for r in rows if not r["stable"]]
    summary = dict(
        pairs=n_run,
        G1=dict(pass_=n_run - n_g1_fail, fail=n_g1_fail),
        G2_stable_hard=dict(total=len(stable_rows),
                            pass_=sum(1 for r in stable_rows if r["g2_ok"]),
                            fail=sum(1 for r in stable_rows if not r["g2_ok"])),
        G2_short_record_only=dict(total=len(short_rows),
                                  g2_pass=sum(1 for r in short_rows if r["g2_ok"]),
                                  agree_min=min((r["agree"] for r in short_rows), default=None),
                                  lratio_max=max((r["lratio"] for r in short_rows), default=None)),
        timing=dict(avg_prefill_ms=round(tot_prefill_ms / max(n_run, 1), 2),
                    avg_ms_per_token=round(tot_decode_ms / max(tot_steps, 1), 4),
                    total_tokens=int(tot_steps)),
    )
    print("\n=== 判定 ===")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    verdict = (n_g1_fail == 0 and n_g2_fail_stable == 0)
    print("B12 验收:", "PASS" if verdict else "FAIL")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
