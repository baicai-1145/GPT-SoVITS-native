#!/usr/bin/env python3
"""parse_powermetrics.py — powermetrics text -> tidy CSV.

Columns: timestamp,sample,cpu,cluster,freq_mhz,residency_pct,
         cpu_power_mw,gpu_power_mw

Topology is taken from the report's own section structure: CPUs listed
under an E-Cluster heading belong to E, likewise P (on Mac16,12/M4 that is
cpus 0-5 -> E and 6-9 -> P; do NOT hardcode).

Power lines on current macOS are package-level only ("CPU Power",
"GPU Power"); they are forward-filled into every row of the same sample.
Per-cluster mW is not emitted by this sampler — see README limitations.
"""
import csv
import re
import sys

RE_SAMPLE = re.compile(r"^\*\*\*\s*Sampled system activity")
RE_TIME = re.compile(r"Sampled system activity\s*\(([^)]+)\)")
RE_SECTION_E = re.compile(r"^\s*E-Cluster\b")
RE_SECTION_P = re.compile(r"^\s*P-Cluster\b")
RE_CPU_FREQ = re.compile(r"^\s*CPU (\d+) frequency:\s*(\d+)\s*MHz")
RE_CPU_RES = re.compile(r"^\s*CPU (\d+) active residency:\s*([\d.]+)%")
RE_CPU_PWR = re.compile(r"^CPU Power:\s*(\d+)\s*mW", re.I)
RE_GPU_PWR = re.compile(r"^GPU Power:\s*(\d+)\s*mW", re.I)


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: parse_powermetrics.py <raw.txt> <out.csv>",
              file=sys.stderr)
        return 2
    src, dst = sys.argv[1], sys.argv[2]
    rows = []
    unrecognized = 0
    cur = None
    sample_idx = -1
    section = ""  # "" | "E" | "P"

    def flush():
        if cur is None:
            return
        for cpu in sorted(cur["cores"]):
            freq, res, cluster = cur["cores"][cpu]
            rows.append([cur["ts"], cur["idx"], cpu, cluster, freq, res,
                         cur.get("cpu_mw", ""), cur.get("gpu_mw", "")])

    with open(src, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if RE_SAMPLE.search(line):
                flush()
                sample_idx += 1
                m = RE_TIME.search(line)
                cur = {"ts": m.group(1) if m else "", "idx": sample_idx,
                       "cores": {}}
                section = ""
                continue
            if cur is None:
                continue
            if RE_SECTION_E.match(line):
                section = "E"
                continue
            if RE_SECTION_P.match(line):
                section = "P"
                continue
            m = RE_CPU_FREQ.match(line)
            if m:
                cpu = int(m.group(1))
                slot = cur["cores"].setdefault(cpu, ["", "", section])
                slot[0] = int(m.group(2))
                slot[2] = section or slot[2]
                continue
            m = RE_CPU_RES.match(line)
            if m:
                cpu = int(m.group(1))
                slot = cur["cores"].setdefault(cpu, ["", "", section])
                slot[1] = float(m.group(2))
                slot[2] = section or slot[2]
                continue
            m = RE_CPU_PWR.match(line)
            if m:
                cur["cpu_mw"] = int(m.group(1))
                continue
            m = RE_GPU_PWR.match(line)
            if m:
                cur["gpu_mw"] = int(m.group(1))
                continue
            if line.strip():
                unrecognized += 1
    flush()

    with open(dst, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "sample", "cpu", "cluster", "freq_mhz",
                    "residency_pct", "cpu_power_mw", "gpu_power_mw"])
        w.writerows(rows)
    print(f"parsed {len(rows)} core-samples ({sample_idx + 1} samples), "
          f"{unrecognized} unrecognized lines", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
