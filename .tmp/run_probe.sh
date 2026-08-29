#!/bin/bash
# E13 probe runner: $1=tag  env via EXTRA_ENV
cd /Volumes/2T/wt-gsv/E13
TAG="$1"
OUT=".tmp/probe_${TAG}.log"
env ${EXTRA_ENV} GSV_REF_TIMING=1 ./build/gsv_native \
  --amx --amx-bert --amx-enc --no-cache \
  --prompt-text "就是客人的重要度划分，分为胡桃竹木四级，往往级别越高，往来就越密。" \
  --sample --text "重庆的火锅店终于开张了。" \
  --ref-wav test_wav/vo_HTLQ001_3_hutao_16.wav \
  --out .tmp/e13_out.wav > "$OUT" 2>&1
EXIT=$?
echo "== [$TAG] exit=$EXIT load=$(uptime | sed 's/.*load averages//' | tr -d ':') =="
grep -iE "timing|ms|ref|sv|hubert|cond|fbank|resample" "$OUT" | grep -viE "^(info|warn)" | tail -30
