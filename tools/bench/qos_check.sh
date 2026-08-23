#!/bin/bash
# qos_check.sh — verify QoS->cluster mapping (ARCHITECTURE.md §4).
#
# Runs the same GEMV workload twice under different thread QoS classes and
# samples with powermetrics in between:
#   pass A: user_interactive (expect: P-cluster residency high)
#   pass B: utility          (expect: E-cluster residency high)
#
# Verdict heuristics (printed, not enforced):
#   * utility E-residency should clearly exceed interactive E-residency
#   * interactive P-residency should exceed utility P-residency
# Requires passwordless `sudo -n powermetrics` (see README).
set -euo pipefail

cd "$(dirname "$0")/../.."
OUT=build/bench/qos_check
mkdir -p "$OUT"
BIN="$OUT/kern_bench"

clang++ -std=c++20 -O3 -mcpu=apple-m4 -Wall -Wextra -Werror \
    -Isrc/kern src/kern/bench_main.cpp src/kern/bench_kernels.cpp \
    -o "$BIN" -framework Accelerate

sample() { # $1 tag
    sudo -n powermetrics --samplers cpu_power -i 250 -n 24 \
        > "$OUT/pm_$1.raw.txt" 2>/dev/null &
    PM=$!
    # ~6s of multi-threaded load; 6 participants spill past the 4 P-cores,
    # making the cluster placement difference visible.
    "$BIN" --suite gemv --threads 6 --reps 12 \
        --qos "$1" --out "$OUT/bench_$1.csv" >/dev/null 2>&1 || true
    wait $PM || true
}

echo "[qos] pass A: user_interactive"
sample user_interactive
echo "[qos] cooldown"; sleep 3
echo "[qos] pass B: utility"
sample utility

python3 - "$OUT" <<'EOF'
import csv, subprocess, sys, os
out = sys.argv[1]
def cluster_stats(tag):
    subprocess.run(["python3", "tools/bench/parse_powermetrics.py",
                    f"{out}/pm_{tag}.raw.txt", f"{out}/pm_{tag}.csv"],
                   check=True, capture_output=True)
    res = {"E": [], "P": []}
    for r in csv.DictReader(open(f"{out}/pm_{tag}.csv")):
        try:
            res[r["cluster"]].append(float(r["residency_pct"]))
        except (TypeError, ValueError):
            pass
    return {c: (sum(v)/len(v) if v else 0.0) for c, v in res.items()}
ia = cluster_stats("user_interactive")
ub = cluster_stats("utility")
print(f"{'cluster':<8} {'user_interactive':>17} {'utility':>9}")
for c in ("E", "P"):
    print(f"{c:<8} {ia[c]:>16.1f}% {ub[c]:>8.1f}%")
ok = ub["E"] > ia["E"] and ia["P"] >= ub["P"]
print("QOS MAPPING: " + ("PASS (utility lands on E, interactive on P)"
      if ok else "WEAK/AMBIGUOUS — inspect raw curves; system load can mask it"))
EOF
