#!/usr/bin/env python3
"""report.py — kern microbenchmark CSV -> markdown report (D1-part1).

Usage:
    report.py <kern_bench.csv> [powermetrics.csv] [-o report.md]

Emits the §5 acceptance-metric template with measured rows filled in and
the end-to-end slots (RTF / first-packet / per-segment token rate) left as
placeholders for D1-part2/C2 wiring.

Peak-accounting model (also in README):
  * P-core fp32 NEON peak  = p_ghz * 32 flop/cycle/core   (4x128b FMA pipes)
  * E-core counted at half issue rate.
  * Accelerate sgemm uses the AMX/MMA units, whose throughput EXCEEDS the
    NEON model — so "sgemm vs NEON-theory" >100% is expected and is not an
    error; the meaningful utilization for Accelerate calls is its own
    stability, recorded via cv_pct.
"""
import csv
import sys
from collections import defaultdict

UNSTABLE_CV = 25.0  # % — above this a row is flagged


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f(x, nd=2):
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    out = None
    if "-o" in sys.argv:
        out = sys.argv[sys.argv.index("-o") + 1]
    if not args:
        print(__doc__)
        return 2
    rows = load(args[0])
    pm = load(args[1]) if len(args) > 1 else []

    meta = next((r for r in rows if r["suite"] == "meta"), None)
    p_ghz = float(meta["gflops"]) if False else 4.41
    if meta and "p_ghz=" in meta["params"]:
        for kv in meta["params"].replace(";", ",").split(","):
            k, _, v = kv.partition("=")
            if k == "p_ghz":
                p_ghz = float(v)

    lines = []
    ap = lines.append
    ap("# kern micro-benchmark report (D1-part1)")
    ap("")
    ap(f"- Machine profile: `{meta['params']}`" if meta else "")
    ap(f"- Peak model: P-core fp32 NEON = freq x 32 flop/cycle "
       f"(= {f(p_ghz * 32, 1)} GFLOPS/core @ {f(p_ghz)} GHz); "
       "E-core half-rate. Accelerate sgemm rides AMX/MMA and exceeds this "
       "model by design.")
    ap("")

    # ---- per-suite tables -------------------------------------------------
    suites = defaultdict(list)
    for r in rows:
        if r["suite"] == "meta":
            continue
        suites[r["suite"]].append(r)

    for suite, srows in suites.items():
        ap(f"## {suite}")
        ap("")
        ap("GFLOPS/GB/s are computed from **ms_min** (best batch): timing "
           "noise is one-sided, so the minimum is the stable cross-run "
           "statistic; median shown for reference.")
        ap("")
        ap("| kernel | params | thr | ms(min) | ms(med) | cv% | GFLOPS | "
           "GB/s | %NEON-peak(1P) | notes |")
        ap("|---|---|---|---|---|---|---|---|---|---|")
        for r in srows:
            gf = float(r["gflops"]) if r["gflops"] else 0.0
            pct = 100.0 * gf / (p_ghz * 32.0) if gf else ""
            notes = []
            if float(r["cv_pct"]) > UNSTABLE_CV:
                notes.append(f"**UNSTABLE cv={r['cv_pct']}%**")
            ms_min = r.get("ms_min", "")
            ap(f"| {r['kernel']} | {r['params']} | {r['threads']} | "
               f"{f(ms_min, 4) if ms_min else ''} | "
               f"{f(r['ms_median'], 4)} | {f(r['cv_pct'], 1)} | "
               f"{f(gf, 2)} | {f(r['gbps'], 2)} | "
               f"{f(pct, 1) if pct != '' else ''} | "
               f"{'; '.join(notes)} |")
        ap("")

    # ---- powermetrics summary --------------------------------------------
    if pm:
        ap("## powermetrics sampling summary")
        ap("")
        by_cluster = defaultdict(list)
        freqs = defaultdict(list)
        samples = set()
        for r in pm:
            try:
                res = float(r["residency_pct"])
                frq = float(r["freq_mhz"])
            except (TypeError, ValueError):
                continue
            by_cluster[r["cluster"]].append(res)
            freqs[r["cluster"]].append(frq)
            samples.add(r["sample"])
        ap("| cluster | mean active residency % | mean MHz | max MHz |")
        ap("|---|---|---|---|")
        for cl in sorted(by_cluster):
            xs = by_cluster[cl]
            fs = freqs[cl]
            ap(f"| {cl} | {f(sum(xs)/len(xs),1)} | "
               f"{f(sum(fs)/len(fs),0)} | {f(max(fs),0)} |")
        ap("")
        ap(f"samples: {len(samples)}")

    # ---- §5 acceptance template ------------------------------------------
    ap("")
    ap("## §5 acceptance metrics (template)")
    ap("")
    ap("| # | metric | threshold | current | status |")
    ap("|---|---|---|---|---|")
    ap("| 1 | P-core util during synth (pm) | >=90% | "
       "(sample with workload; see run_kern_bench.sh --with-pm) | TBD |")
    ap("| 1 | E-core util during synth (pm) | >=50% | same | TBD |")
    ap("| 2 | AR decode DRAM BW vs theory | >=60% (int8: 70%) | needs C2 "
       "full-link counters | TBD (powermetrics has no DRAM counter on AS; "
       "use IOReport/asitop proxy) |")
    best_gemm = max((float(r["gflops"]) for r in rows
                     if r["suite"] == "sgemm"), default=0)
    ap("| 3 | GEMM effective FLOPS vs Accelerate scale | >=80% of "
       "same-scale cblas/AMX | see sgemm table | TBD after A4 integration |")
    ap("| 4 | RTF / first packet / per-segment token rate | record only | "
       "wired at C2/D1-part2 | PLACEHOLDER |")
    ap("| 5 | numerical correctness | always wins | golden gates G1/G2/G3 "
       "unaffected by bench | OK-by-construction |")
    ap("")

    md = "\n".join(l for l in lines if l is not None)
    if out:
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(md + "\n")
        print(f"wrote {out}")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
