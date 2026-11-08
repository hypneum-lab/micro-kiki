#!/usr/bin/env bash
# NH-81 KXKM (amendment r4): start kxkm_launcher.sh once no E1-small or
# N10-C job remains, from the process table and the GPU compute-process
# list (not a fixed time). Read-only on other jobs; never signals any
# process. Deadline 2026-11-04 12:00 CET, then stop and report.
set -u
LOG=/ai/saisail/nh81-kxkm/results/wait.log
DEADLINE=$(date -d "2026-11-04 12:00" +%s)
PATS='run_e1s_phase|/ai/saisail/e1-small/venv/|procgen_multiagent_gtm_comm\.(harness|campaign)\.|pretrain_lane.sh|gate_bis_lane.sh|run_campaign'
gpu_foreign() {  # compute processes other than ComfyUI (resident, idle)
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
    | while read -r p; do ps -o args= -p "$p" 2>/dev/null; done | grep -v ComfyUI2 | head -1
}
busy() {
  local r
  r=$(pgrep -af "$PATS" | head -1) && { echo "proc: ${r:0:120}"; return 0; }
  r=$(gpu_foreign) && [ -n "$r" ] && { echo "gpu: ${r:0:120}"; return 0; }
  return 1
}
echo "$(date '+%F %T %Z') waiting (pid $$)" >> $LOG
last=""
while r=$(busy); do
  [ "$r" != "$last" ] && { echo "$(date '+%F %T %Z') busy: $r" >> $LOG; last=$r; }
  [ "$(date +%s)" -gt "$DEADLINE" ] && { echo "$(date '+%F %T %Z') deadline reached, not launched" >> $LOG; exit 1; }
  sleep 300
done
echo "$(date '+%F %T %Z') other jobs gone; settling 120 s" >> $LOG
sleep 120
echo "$(date '+%F %T %Z') launching kxkm_launcher (r4)" >> $LOG
exec /ai/saisail/gpu-swap/launchers/nh81-measure.sh
