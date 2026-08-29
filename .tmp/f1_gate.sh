#!/bin/bash
# f1_gate.sh — E13 终局验收 F1: 合流态(e39251e) N=56 codes/wav 差分门禁
# 单变量: --amx-enc (两侧均挂 --amx --amx-bert --no-cache --sample --sample-seed 7)
# 复用 T14 口径(.tmp/t14_mel.sh / t14_gate_allon.md / e13_accept.sh 门2)
set -u
cd /Volumes/2T/wt-gsv/E13 || exit 2
BIN=./build/gsv_native
T=.tmp/f1work
mkdir -p "$T"
PT="就是客人的重要度划分，分为胡桃竹木四级，往往级别越高，往来就越密。"
TXT="重庆的火锅店终于开张了。"

: > "$T/codes_diff.txt"
: > "$T/wav_diff.txt"
n=0
for W in test_wav/*.wav; do
  n=$((n+1))
  TAG=$(basename "$W" .wav)
  # ---- OFF 臂 ----
  GSV_RVQ_DUMP="$T/d_off_${TAG}.bin" "$BIN" --amx --amx-bert --no-cache \
    --prompt-text "$PT" --sample --sample-seed 7 --text "$TXT" \
    --ref-wav "test_wav/${TAG}.wav" --out "$T/off_${TAG}.wav" \
    > "$T/log_off_${TAG}.txt" 2>&1
  # ---- ON 臂 ----
  GSV_RVQ_DUMP="$T/d_on_${TAG}.bin" "$BIN" --amx --amx-bert --amx-enc --no-cache \
    --prompt-text "$PT" --sample --sample-seed 7 --text "$TXT" \
    --ref-wav "test_wav/${TAG}.wav" --out "$T/on_${TAG}.wav" \
    > "$T/log_on_${TAG}.txt" 2>&1
  # codes 段 md5(dump=前 Tq*8B codes + Tq*768*4B proj; Tq=filesize/3080)
  TB=$(stat -f%z "$T/d_off_${TAG}.bin" 2>/dev/null || echo 0)
  TA=$(stat -f%z "$T/d_on_${TAG}.bin" 2>/dev/null || echo 0)
  if [ "$TB" -eq 0 ] || [ "$TA" -eq 0 ]; then
    echo "ERR  $TAG dump-missing(off=$TB on=$TA)" >> "$T/codes_diff.txt"
    continue
  fi
  TQ=$((TB / 3080)); CB=$((TQ * 8))
  MB=$(head -c $CB "$T/d_off_${TAG}.bin" | md5)
  MA=$(head -c $CB "$T/d_on_${TAG}.bin" | md5)
  MW_O=$(md5 -q "$T/off_${TAG}.wav")
  MW_A=$(md5 -q "$T/on_${TAG}.wav")
  if [ "$MB" != "$MA" ]; then echo "CODES-DIFF $TAG Tq=$TQ off=${MB##*= } on=${MA##*= }" >> "$T/codes_diff.txt"; fi
  if [ "$MW_O" != "$MW_A" ]; then echo "WAV-DIFF $TAG off=$MW_O on=$MW_A" >> "$T/wav_diff.txt"; fi
  echo "#[$n] $TAG done"
done
echo "== F1 sweep complete =="
echo "--- codes diff ($(wc -l < "$T/codes_diff.txt" | tr -d ' ') 条) ---"
cat "$T/codes_diff.txt"
echo "--- wav diff ($(wc -l < "$T/wav_diff.txt" | tr -d ' ') 条) ---"
cat "$T/wav_diff.txt"
