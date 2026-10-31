#!/usr/bin/env bash
# GB300 35B LoRA demo (docs/prereg/demo-35b-lora-gb300/, PI decisions 2026-10-31).
# DEMO, NOT A SCIENTIFIC CLAIM. Writes only under $ROOT.
# Prerequisites (operator): venv from requirements-gb300-demo.txt with
# --require-hashes; NH81_GPU_PCI_BUS_ID set; the NH-81 GB300 run is over.
set -euo pipefail
: "${NH81_GPU_PCI_BUS_ID:?set the GB300 PCI bus id}"
ROOT=${DEMO_GB300_ROOT:-$HOME/demo-35b-gb300}
REPO=${NH81_REPO:-$HOME/nh81-gb300/micro-kiki}
PY=$ROOT/venv/bin/python
HERE=$(cd "$(dirname "$0")" && pwd)
PRE=$REPO/docs/prereg/demo-35b-lora-gb300
OUT=$ROOT/results
BASE=Qwen/Qwen3.6-35B-A3B; BREV=995ad96eacd98c81ed38be0c5b274b04031597b0
AD=hypneum-lab/micro-kiki-v4-sota; AREV=9add98c22d2341e454cf1c204361f973a72b86c2
DOMAINS="chat-fr python cpp rust kicad-dsl spice embedded power"
[ -e "$OUT/runs.jsonl" ] && { echo "refusing: $OUT/runs.jsonl exists"; exit 1; }
mkdir -p "$OUT" "$ROOT/base" "$ROOT/adapters" "$ROOT/code"
START=$(date +%s); DEADLINE=$(( START + 3 * 3600 ))   # PI: 3 h from the download

# 1. Download by URL and exact revision (curl), then SHA-256 against the manifest.
fetch() { local repo=$1 rev=$2 path=$3 dest=$4
  mkdir -p "$(dirname "$dest")"
  local left=$(( DEADLINE - $(date +%s) ))
  [ "$left" -gt 0 ] || { echo "deadline reached during download"; exit 3; }
  [ -s "$dest" ] || curl -fL --retry 3 --max-time "$left" -o "$dest" "https://huggingface.co/$repo/resolve/$rev/$path"; }
grep -v '^#' "$PRE/sources-sha256.txt" > "$ROOT/sources.list"
while read -r sha ref; do
  path=${ref#*:}
  case $ref in
    Qwen/*) fetch "$BASE" "$BREV" "$path" "$ROOT/base/$path"; echo "$sha  $ROOT/base/$path" ;;
    hypneum-lab/*) fetch "$AD" "$AREV" "$path" "$ROOT/adapters/$path"; echo "$sha  $ROOT/adapters/$path" ;;
  esac
done < "$ROOT/sources.list" > "$ROOT/expected.sha256"
sha256sum -c "$ROOT/expected.sha256"

# 2. Environment (same instrument as NH-81 GB300).
TAG=prereg-nh81-m3pro-20261030; TREE=d1b67e6ea1f1d6c7064ac8a9bea59152de00101c
[ "$(git -C "$REPO" rev-parse "$TAG:experiments/nh81_joules")" = "$TREE" ] || { echo "tree mismatch"; exit 1; }
rm -rf "$ROOT/code/experiments"
git -C "$REPO" archive "$TAG" experiments/nh81_joules | tar -x -C "$ROOT/code"
export NH81_CODE_DIR=$ROOT/code/experiments/nh81_joules
export PYTHONPATH=$HERE:$HERE/../nh81_gb300:$NH81_CODE_DIR
export CUDA_VISIBLE_DEVICES=$($PY -c "import power_gb300 as p; print(p.uuid())")
SP=$($PY -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
mkdir -p "$ROOT/cuda_home/include"
for d in "$SP"/nvidia/cuda_runtime/include "$SP"/nvidia/cuda_cccl/include "$SP"/nvidia/cuda_nvrtc/include; do
  [ -d "$d" ] && cp -rs "$d"/. "$ROOT/cuda_home/include/" 2>/dev/null || true
done
export CUDA_HOME=$ROOT/cuda_home
{ nvidia-smi -q; uname -a; getconf PAGESIZE; cat /proc/loadavg; "$PY" -m pip freeze 2>/dev/null || true
  ps -eo pid,user,pcpu,comm,args; } > "$OUT/host.txt" 2>&1

# 3. Smoke test: if MLX-CUDA fails, the demo is abandoned (PI decision 9).
$PY -c "import mlx.core as mx; mx.eval(mx.ones(4) @ mx.ones(4)); print('smoke ok', mx.default_device())" \
  | tee "$OUT/smoke.txt" || { echo "MLX-CUDA smoke failed: demo abandoned" | tee "$OUT/abandoned.txt"; exit 4; }

# 4. Demo under the deadline.
LEFT=$(( DEADLINE - $(date +%s) ))
[ "$LEFT" -gt 0 ] || { echo "deadline reached before the demo"; exit 3; }
set +e
timeout --signal=INT --kill-after=60 "$LEFT" $PY "$HERE/run_demo.py" --base "$ROOT/base" --adapters "$ROOT/adapters" \
  --prompts "$PRE/prompts.json" --out "$OUT"
RC=$?
echo "demo exit $RC" | tee "$OUT/demo_exit.txt"
set -e

# 5. Weights deleted after the slot (PI decision 8), results kept.
[ "${KEEP_WEIGHTS:-0}" = 1 ] || rm -rf "$ROOT/base" "$ROOT/adapters"
exit "$RC"
