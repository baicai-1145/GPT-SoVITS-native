#!/bin/bash
# f2_waiter.sh — 安静窗守望(load<2.5 触发, 40min 上限; 超时取当前窗并在报告标注)
# 触发后连跑 3 轮交错配对(轮间隔 20s)
set -u
cd /Volumes/2T/wt-gsv/E13
DEADLINE=$((SECONDS + 2400))
while :; do
  L=$(uptime | sed 's/.*load average[s]*: //' | awk -F', ' '{print $1}' | tr -d ' ')
  OK=$(awk "BEGIN{print ($L<2.5)?1:0}")
  echo "$(date +%H:%M:%S) load=$L ok=$OK" >> .tmp/f2_waiter.log
  [ "$OK" = "1" ] && break
  [ $SECONDS -ge $DEADLINE ] && { echo "$(date +%H:%M:%S) TIMEOUT.pick-lowest" >> .tmp/f2_waiter.log; break; }
  sleep 60
done
sleep 5
for R in f2a f2b f2c; do
  .tmp/f2_pair.sh "$R"
  sleep 20
done
echo "=== f2 rounds complete ==="
grep -h "ref-timing\|hubert-sdpa\|^== \[" .tmp/f2_base_f2?.log .tmp/f2_merged_f2?.log
