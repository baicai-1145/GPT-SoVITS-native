#!/bin/bash
# sample_powermetrics.sh — D1-part1 utilization/frequency sampler.
#
# Samples per-core frequency + active residency and cluster/package power
# during a synthetic workload, writing raw text plus a parsed CSV.
#
# Usage:
#   sample_powermetrics.sh [duration_s] [interval_ms] [out_prefix]
#     defaults: 30 250 /tmp/pm_sample
#
# Outputs:
#   <out>.raw.txt   verbatim powermetrics text (kept for audit)
#   <out>.csv       timestamp,sample,cpu,cluster,freq_mhz,residency_pct,
#                   ecluster_mw,pcluster_mw,gpu_mw
#
# Permissions — pick ONE (details in tools/bench/README.md):
#   1. Run this script under `sudo`.
#   2. Passwordless sudoers fragment (recommended for CI-ish loops):
#        sudo visudo -f /etc/sudoers.d/gsv-bench
#      content:
#        <youruser> ALL=(root) NOPASSWD: /usr/bin/powermetrics
#
# DRAM bandwidth caveat: powermetrics on Apple Silicon does not expose a
# first-class DRAM BW counter. As a proxy we record GPU BW lines when the
# gpu_power sampler prints them and note that real DRAM accounting needs
# IOReport (see README "asitop" section). Treat gbps columns from the
# microbenchmarks as *effective kernel traffic*, not measured DRAM.
set -euo pipefail

DUR=${1:-30}
IV=${2:-250}
OUT=${3:-/tmp/pm_sample}

NSAMPLES=$(( DUR * 1000 / IV ))
if [ "$NSAMPLES" -lt 1 ]; then NSAMPLES=1; fi

RAW="$OUT.raw.txt"
CSV="$OUT.csv"

echo "[pm] sampling ${NSAMPLES} samples @ ${IV}ms -> $RAW"
# shellcheck disable=SC2086
if [ -n "${PM_PRESET_WORKLOAD:-}" ]; then
    echo "[pm] PM_PRESET_WORKLOAD is set; launch your workload now."
fi

SUDO=""
if [ "$(id -u)" -ne 0 ]; then SUDO="sudo"; fi

"$SUDO" powermetrics \
    --samplers cpu_power,gpu_power \
    -i "$IV" -n "$NSAMPLES" \
    --show-process-gpu false \
    > "$RAW" 2>/dev/null || "$SUDO" powermetrics \
    --samplers cpu_power,gpu_power -i "$IV" -n "$NSAMPLES" > "$RAW"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$SCRIPT_DIR/parse_powermetrics.py" "$RAW" "$CSV"
echo "[pm] wrote $CSV ($(($(wc -l < "$CSV") - 1)) sample-core rows)"
