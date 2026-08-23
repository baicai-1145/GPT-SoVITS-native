#!/opt/homebrew/bin/python3/bin/python3
"""check_sovits_golden.py — C++ dump vs torch hook golden 对照 (G1)

用法: python3 check_sovits_golden.py <fixture_dir> <cpp_dump_dir>
对 hooks/*.bin (torch) 与 cpp_dump/*.bin 逐名计算 cos / max-rel, 报 G1 门。
"""
import sys
from pathlib import Path

import numpy as np

FIX = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
DMP = Path(sys.argv[2] if len(sys.argv) > 2 else ".")

# torch hook 名 → C++ dump 名
PAIRS = [
    ("h_quantizer_dec", "h_quantizer_dec"),
    ("h_encp_input", "h_encp_input"),
    ("h_encp_proj", "h_encp_proj"),
    ("h_flow_in", "h_flow_in"),
    ("h_flow", "h_flow"),
    ("h_dec", "h_dec"),
]


def load(p: Path):
    a = np.fromfile(p, dtype=np.float32)
    shape = tuple(int(x) for x in p.with_suffix(".shape").read_text().split())
    return a.reshape(shape)


def g1(a, b):
    n = min(a.size, b.size)
    a, b = a.flatten()[:n], b.flatten()[:n]
    cos = float(np.dot(a, b) /
                (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    denom = max(1e-9, float(np.abs(b).max()))
    rel = float(np.abs(a - b).max()) / denom
    mad = float(np.abs(a - b).mean())
    return cos, rel, mad


print(f"{'tensor':<18}{'cos':>12}{'max_rel':>12}{'mean_absdiff':>14}  G1")
fails = []
for tn, cn in PAIRS:
    fp = FIX / "hooks" / f"{tn}.bin"
    cp = DMP / f"{cn}.bin"
    if not cp.exists():
        print(f"{tn:<18}  MISSING cpp dump")
        fails.append(tn)
        continue
    t, c = load(fp), load(cp)
    if t.size != c.size:
        print(f"{tn:<18}  SIZE MISMATCH {t.size} vs {c.size}")
        fails.append(tn)
        continue
    cos, rel, mad = g1(t, c)
    ok = cos >= 0.9999 and rel <= 1e-3
    if not ok:
        fails.append(tn)
    print(f"{tn:<18}{cos:>12.6f}{rel:>12.3e}{mad:>14.4e}  {'PASS' if ok else 'FAIL'}")

print("\nG1:", "ALL PASS" if not fails else f"FAIL {fails}")
sys.exit(0 if not fails else 1)
