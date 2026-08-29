#!/bin/bash
# f2_pair.sh — E13 终局验收 F2: 单轮交错配对(基线04d4b26 ↔ 合流态e39251e)
# 用法: f2_pair.sh <round> [start_side]   start_side: base|merged (默认按轮次奇偶交替)
set -u
MAIN=/Volumes/2T/wt-gsv/E13
BASE=/Volumes/2T/wt-gsv/E13-F2BASE
OUTD="$MAIN/.tmp"
PT="就是客人的重要度划分，分为胡桃竹木四级，往往级别越高，往来就越密。"
TXT="重庆的火锅店终于开张了。"
ROUND="${1:-r1}"
if [ $# -ge 2 ]; then FIRST="$2"; elif [ $((RANDOM % 2)) -eq 0 ]; then FIRST="base"; else FIRST="merged"; fi
[ "$FIRST" = "base" ] && ORDER=(base merged) || ORDER=(merged base)

run_one() {  # <side>
  local D LOAD
  if [ "$1" = "base" ]; then D="$BASE"; else D="$MAIN"; fi
  LOAD=$(uptime | sed 's/.*load average[s]*: //')
  echo "== [$ROUND/$1] $(date +%H:%M:%S) load=$LOAD" >> "$OUTD/f2_${1}_${ROUND}.log"
  (cd "$D" && GSV_REF_TIMING=1 GSV_HUBERT_SDPA_TIMING=1 ./build/gsv_native \
    --amx --amx-bert --amx-enc --no-cache \
    --prompt-text "$PT" --sample --sample-seed 7 --text "$TXT" \
    --ref-wav test_wav/vo_HTLQ001_3_hutao_16.wav \
    --out ".tmp/f2_${1}_${ROUND}.wav") >> "$OUTD/f2_${1}_${ROUND}.log" 2>&1
}

run_one "${ORDER[0]}"
run_one "${ORDER[1]}"
echo "-- pair $ROUND done ($FIRST first)"
