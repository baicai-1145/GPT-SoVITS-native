#!/bin/bash
# e13_accept.sh — E13-MIX 三路合流后一键最终验收
# 六门: ①默认位级 ②codes 位级(dump) ③mel 门(ON vs OFF) ④段耗时×3 ⑤ctest ⑥c2_pairs 无污染证明
# 用法: ./.tmp/e13_accept.sh [repoRoot]   (默认 /Volumes/2T/wt-gsv/E13-MIX)
# 输出: stdout PASS/FAIL 汇总表; 每门日志 .tmp/accept_gate*.log; 摘要 .tmp/accept-selftest.md
set -u
REPO="${1:-/Volumes/2T/wt-gsv/E13-MIX}"
cd "$REPO" || exit 2
BIN=./build/gsv_native
T=".tmp"
mkdir -p $T
PT_MAIN="就是客人的重要度划分，分为胡桃竹木四级，往往级别越高，往来就越密。"
TXT_MAIN="重庆的火锅店终于开张了。"
TXT2="我看看是什么样子的。"
TXT3="我爱吃火锅，也爱喝可乐。"
BASELINE_MD5="0654e52a6051ed7f4d8f28f2e46b436f"   # main 基线位型(T7 判定; fa78 已过期)
PASS=0; FAIL=0
declare -a SUMMARY

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$T/accept_run.log"; }
gate_result() {  # <名> <PASS|FAIL> <摘要>
  if [ "$2" = PASS ]; then PASS=$((PASS+1)); else FAIL=$((FAIL+1)); fi
  SUMMARY+=("$1: $2 — $3")
}
md5q() { md5 -q "$1" 2>/dev/null; }

rm -f "$T/accept_run.log"

# ---------- 门1 默认路径位级 ----------
log "门1: 默认路径位级 (贪心 ×2 稳定 == $BASELINE_MD5)"
{
  echo "== 门1 默认路径位级 =="
  for i in 1 2; do
    "$BIN" --no-cache --prompt-text "$PT_MAIN" --text "$TXT_MAIN" \
      --ref-wav test_wav/vo_HTLQ001_3_hutao_16.wav --out "$T/g1_$i.wav" > /dev/null 2>&1
    echo "run$i: $(md5q "$T/g1_$i.wav")"
  done
} > "$T/accept_gate1.log" 2>&1
M1=$(sed -n 's/^run1: //p' "$T/accept_gate1.log")
M2=$(sed -n 's/^run2: //p' "$T/accept_gate1.log")
if [ "$M1" = "$BASELINE_MD5" ] && [ "$M1" = "$M2" ]; then
  gate_result "门1默认位级" PASS "$M1 (×2 稳定)"
else
  gate_result "门1默认位级" FAIL "got=$M1/$M2 expect=$BASELINE_MD5"
fi

# ---------- 门2 codes 位级 (dump 前 Tq*8 字节 i64 codes 段) ----------
log "门2: codes 位级 (--amx-enc ON vs OFF 同环境; codes 段=前 Tq*8B)"
{
  echo "== 门2 GSV_RVQ_DUMP codes 段对比 =="
  echo "# 注: dump=codes(Tq*8B)+proj; proj 随 --amx-enc 的 HuBERT tile 累加序呈 1e-3 级允许差,"
  echo "#      故位级门只对 codes 段。"
  GSV_RVQ_DUMP="$T/g2_base.bin" "$BIN" --amx --amx-bert --no-cache \
    --prompt-text "$PT_MAIN" --sample-seed 7 --sample --text "$TXT_MAIN" \
    --ref-wav test_wav/vo_HTLQ001_3_hutao_16.wav --out "$T/g2_b.wav" >/dev/null 2>&1
  GSV_RVQ_DUMP="$T/g2_amx.bin" "$BIN" --amx --amx-bert --amx-enc --no-cache \
    --prompt-text "$PT_MAIN" --sample-seed 7 --sample --text "$TXT_MAIN" \
    --ref-wav test_wav/vo_HTLQ001_3_hutao_16.wav --out "$T/g2_a.wav" >/dev/null 2>&1
  TB=$(stat -f%z "$T/g2_base.bin"); TQ=$((TB / 3080)); CB=$((TQ * 8))
  echo "sizes: base=$TB amx=$(stat -f%z "$T/g2_amx.bin") Tq=$TQ codes_bytes=$CB"
  MB=$(head -c $CB "$T/g2_base.bin" | md5); MA=$(head -c $CB "$T/g2_amx.bin" | md5)
  echo "base codes md5: $MB"
  echo "amx  codes md5: $MA"
  if [ "$MB" = "$MA" ]; then echo "CMP: BITWISE EQUAL"; else echo "CMP: DIFFER"; fi
} > "$T/accept_gate2.log" 2>&1
if grep -q "BITWISE EQUAL" "$T/accept_gate2.log"; then
  gate_result "门2codes位级" PASS "$(grep 'codes md5' "$T/accept_gate2.log" | head -1 | awk '{print $4}')"
else
  gate_result "门2codes位级" FAIL "$(tail -1 "$T/accept_gate2.log") — 见 accept_gate2.log"
fi

# ---------- 门3 mel 门 ----------
log "门3: mel 包络 ON vs OFF ×3 场景 ≤0.005"
{
  echo "== 门3 mel ON/OFF 对照 =="
  COMMON_ON=(--amx --amx-bert --amx-enc --no-cache --sample --sample-seed 7)
  COMMON_OFF=(--amx --amx-bert --no-cache --sample --sample-seed 7)
  run_mel() {
    "$BIN" "${COMMON_OFF[@]}" --prompt-text "$3" --text "$TXT_MAIN" \
      --ref-wav "test_wav/$4.wav" --out "$1" >/dev/null 2>&1
    "$BIN" "${COMMON_ON[@]}" --prompt-text "$3" --text "$TXT_MAIN" \
      --ref-wav "test_wav/$4.wav" --out "$2" >/dev/null 2>&1
    /opt/homebrew/bin/python3 - "$1" "$2" << 'PY'
import sys, torch, torchaudio
a,_ = torchaudio.load(sys.argv[1]); b,_ = torchaudio.load(sys.argv[2])
a=a.flatten(); b=b.flatten(); n=min(a.numel(),b.numel()); a,b=a[:n].float(),b[:n].float()
m=torchaudio.transforms.MelSpectrogram(32000,n_fft=2048,win_length=2048,hop_length=640,n_mels=128)
ma,mb=m(a),m(b); t=min(ma.shape[-1],mb.shape[-1])
l1=float((ma[...,:t]-mb[...,:t]).abs().mean()); base=float(mb.mean().clamp_min(1e-9))
rel=l1/base; c=float(torch.corrcoef(torch.stack([a,b]))[0,1])
print(f"mel_rel={rel:.6g} corr={c:.7f} {'PASS' if rel<=0.005 else 'FAIL'}")
PY
  }
  run_mel "$T/g3a_1.wav" "$T/g3b_1.wav" "$PT_MAIN" vo_HTLQ001_3_hutao_16
  run_mel "$T/g3a_2.wav" "$T/g3b_2.wav" "$TXT2" vo_HTLQ001_4_hutao_07
  run_mel "$T/g3a_3.wav" "$T/g3b_3.wav" "$TXT3" vo_card_hutao_endOfGame_win_01
} > "$T/accept_gate3.log" 2>&1
G3N=$(grep -c PASS "$T/accept_gate3.log"); G3F=$(grep -c FAIL "$T/accept_gate3.log")
if [ "$G3F" -eq 0 ] && [ "$G3N" -eq 3 ]; then
  gate_result "门3mel门" PASS "3/3 场景 $(grep -o 'mel_rel=[^ ]*' "$T/accept_gate3.log" | tr '\n' ' ')"
else
  gate_result "门3mel门" FAIL "$G3N过/$G3F败 — 见 accept_gate3.log"
fi

# ---------- 门4 段耗时 ×3 ----------
log "门4: 分段耗时 ×3 (--amx 三旗, 配对窗口径, 诊断性)"
{
  echo "== 门4 分段耗时 ×3 + 负载列 =="
  for i in 1 2 3; do
    echo "-- run$i [$(uptime | grep -o 'load averages.*')] --"
    GSV_REF_TIMING=1 GSV_COND_TIMING=1 "$BIN" --amx --amx-bert --amx-enc --no-cache \
      --prompt-text "$PT_MAIN" --sample-seed 7 --sample --text "$TXT_MAIN" \
      --ref-wav test_wav/vo_HTLQ001_3_hutao_16.wav --out "$T/g4.wav" >/dev/null 2>> "$T/accept_gate4.log"
  done
} > "$T/accept_gate4.log" 2>&1
gate_result "门4段耗时" PASS "诊断输出见 accept_gate4.log (无硬门限)"

# ---------- 门5 ctest ----------
log "门5: ctest 全绿"
ctest --test-dir build > "$T/accept_gate5.log" 2>&1
CT_N=$(grep -oE '[0-9]+% tests passed' "$T/accept_gate5.log" | tail -1)
if grep -q "100% tests passed" "$T/accept_gate5.log"; then
  gate_result "门5ctest" PASS "$CT_N"
else
  gate_result "门5ctest" FAIL "$CT_N — 见 accept_gate5.log"
fi

# ---------- 门6 c2_pairs_run + 无交集污染证明 ----------
if [ -x build/tests/c2_pairs_run ]; then
  log "门6: c2_pairs_run(all_zh) + pre/post-T6 下游快照对比"
  {
    echo "== 门6 c2_pairs_run + 快照对拍 =="
    ./build/tests/c2_pairs_run weights src/runtime/data \
      test_wav/vo_HTLQ001_3_hutao_16.wav "$T/g6.json" >/dev/null 2>&1 \
      && echo "c2_pairs_run rc=0" || echo "c2_pairs_run rc≠0"
    /opt/homebrew/bin/python3 - "$T/g6.json" << 'PY'
import json, sys, torch
from pathlib import Path
j = json.load(open(sys.argv[1])); run = j['runs'][0]
f = Path('/Volumes/2T/wt-gsv/AR/tests/golden/pairs')/'vo_HTLQ001_3_hutao_16__s0.pt'
r = torch.load(str(f), map_location='cpu', weights_only=False)
pc_ok = r['prompt_tokens'].reshape(-1).tolist() == run['prompt_codes']
print(f"prompt_codes vs golden: {'EXACT' if pc_ok else 'DIFF'} ({len(run['prompt_codes'])} 码)")
PY
    if [ -x "$T/gsv_A_old" ]; then
      "$T/gsv_A_old" --no-cache --dump-sovits-in "$T/g6_pre.bin" --prompt-text "$PT_MAIN" --text "$TXT_MAIN" \
        --ref-wav test_wav/vo_HTLQ001_3_hutao_16.wav --out /dev/null >/dev/null 2>&1
      "$BIN" --no-cache --dump-sovits-in "$T/g6_post.bin" --prompt-text "$PT_MAIN" --text "$TXT_MAIN" \
        --ref-wav test_wav/vo_HTLQ001_3_hutao_16.wav --out /dev/null >/dev/null 2>&1
      cmp -s "$T/g6_pre.bin" "$T/g6_post.bin" && echo "pre/post-T6 textfront+AR 快照: BITWISE EQUAL" || echo "pre/post-T6 快照: DIFFER"
    else
      echo "(无 $T/gsv_A_old, 跳过快照对拍)"
    fi
  } > "$T/accept_gate6.log" 2>&1
  G6P=true
  grep -q "EXACT" "$T/accept_gate6.log" || G6P=false
  grep -q "BITWISE EQUAL" "$T/accept_gate6.log" || G6P=false
  grep -q "rc=0" "$T/accept_gate6.log" || G6P=false
  if $G6P; then
    gate_result "门6无污染" PASS "c2_pairs rc=0 + prompt_codes EXACT + 快照 BITWISE EQUAL"
  else
    gate_result "门6无污染" FAIL "见 accept_gate6.log"
  fi
fi

# ---------- 汇总 ----------
echo
echo "================ E13-MIX 验收汇总 ($REPO @ $(git rev-parse --short HEAD)) ================"
for s in "${SUMMARY[@]}"; do echo "  $s"; done
echo "------------------------------------------------------------"
echo "  TOTAL: PASS=$PASS FAIL=$FAIL"
if [ $FAIL -eq 0 ]; then echo "  RESULT: ✅ ALL PASS"; else echo "  RESULT: ❌ HAS FAILURE"; fi
echo "============================================================"
{
  echo "# e13_accept 自测记录"
  echo "- 时间: $(date '+%F %T')  HEAD: $(git rev-parse --short HEAD)  load: $(uptime | grep -o 'load averages.*')"
  echo ""
  for s in "${SUMMARY[@]}"; do echo "- $s"; done
  echo ""
  echo "详细日志: accept_gate{1..6}.log；逐行耗时表在 accept_gate4.log。"
} > "$T/accept-selftest.md"
exit $FAIL
