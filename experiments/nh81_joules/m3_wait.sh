#!/usr/bin/env bash
# NH-81 macM3 waiter: every 10 min, check that no heavy process (python,
# ffmpeg, bouba) is above 20 % CPU, the M5 grid is gone and the load
# average is < 6; then start ONE complete measurement (m3_launcher.sh,
# whose start guard decides). Read-only on other jobs; never signals any
# process. Deadline 2026-11-04 12:00 CET.
set -u
ROOT=$HOME/scratch/nh81-m3
LOG=$ROOT/results/wait.log
DEADLINE=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-04 12:00" +%s)
busy() {
  local r
  r=$(ps -Ao pcpu=,pid=,comm= | awk -v me=$$ 'tolower($3) ~ /python|ffmpeg|bouba/ && $1+0 > 20 && $2 != me {print $1"% "$3; exit}')
  [ -n "$r" ] && { echo "heavy: $r"; return 0; }
  pgrep -f 'scratch/m5-macm3' >/dev/null && { echo "M5 grid running"; return 0; }
  r=$(sysctl -n vm.loadavg | awk '{print $2}')
  awk -v l="$r" 'BEGIN{exit !(l >= 6.0)}' && { echo "load $r"; return 0; }
  return 1
}
mkdir -p $ROOT/results
echo "$(date '+%F %T %Z') waiting (pid $$)" >> $LOG
last=""
while r=$(busy); do
  [ "$r" != "$last" ] && { echo "$(date '+%F %T %Z') busy: $r" >> $LOG; last=$r; }
  [ "$(date +%s)" -gt "$DEADLINE" ] && { echo "$(date '+%F %T %Z') deadline reached, not launched" >> $LOG; exit 1; }
  sleep 600
done
echo "$(date '+%F %T %Z') host looks free; launching m3_launcher.sh" >> $LOG
exec "$ROOT/code/experiments/nh81_joules/m3_launcher.sh"
