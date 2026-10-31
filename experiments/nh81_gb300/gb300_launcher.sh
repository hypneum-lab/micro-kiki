#!/usr/bin/env bash
# NH-81 GB300 descriptive run (docs/prereg/nh81-gb300/, PI decisions 2026-10-31).
# Never stops or reconfigures anything; writes only under $ROOT.
# Prerequisites (operator): venv installed from requirements-gb300-nh81.txt
# with --require-hashes; checkpoints, splits and selection.json copied to
# $ROOT/ckpt and $ROOT/data; NH81_GPU_PCI_BUS_ID set from `nvidia-smi -q`.
set -euo pipefail
: "${NH81_GPU_PCI_BUS_ID:?set the GB300 PCI bus id}"
ROOT=${NH81_GB300_ROOT:-$HOME/nh81-gb300}
REPO=${NH81_REPO:-$ROOT/micro-kiki}
PY=$ROOT/venv/bin/python
HERE=$(cd "$(dirname "$0")" && pwd)
TAG=prereg-nh81-m3pro-20261030
TREE=d1b67e6ea1f1d6c7064ac8a9bea59152de00101c
OUT=$ROOT/results
[ -e "$OUT/measure.jsonl" ] && { echo "refusing: $OUT/measure.jsonl exists"; exit 1; }
mkdir -p "$OUT" "$ROOT/code"

# 1. Frozen code tree, checked by git tree hash.
[ "$(git -C "$REPO" rev-parse "$TAG:experiments/nh81_joules")" = "$TREE" ] || { echo "tree mismatch"; exit 1; }
rm -rf "$ROOT/code/experiments"
git -C "$REPO" archive "$TAG" experiments/nh81_joules | tar -x -C "$ROOT/code"
export NH81_CODE_DIR=$ROOT/code/experiments/nh81_joules

# 2. Data, by SHA-256 (prereg section 1).
sha256sum -c - <<SHA
3c0891fbb528f122b5ac8179b180fab879d38415e385c29b7acb7be7ef12afd7  $ROOT/ckpt/ann1_ckpt_12000.npz
e9a3c9294c4737eddf019fa2ace51f2fe967c942d89039fcbd2870aa26eaa074  $ROOT/ckpt/spkt_ckpt_12000.npz
ad59a2f13afab4f47983561f6f12dd269e31076e4a411a8e0dd071c1021cfd2d  $ROOT/data/test.i32
bc7038948319eaed68c7ee45e61b950bda706cd6bd4d2e3109fd16687d94257a  $ROOT/data/val.i32
d51c35fda2f75330b2d200a51e94f55b631b4592de1fcd2ce01bfaf170d8f52d  $ROOT/data/selection.json
SHA

# 3. Environment: GPU by PCI bus id -> UUID, CUDA headers from the wheels.
export PYTHONPATH=$HERE:$NH81_CODE_DIR
UUID=$($PY -c "import power_gb300 as p; print(p.uuid())")
export CUDA_VISIBLE_DEVICES=$UUID MLX_ENABLE_TF32=0
SP=$($PY -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
mkdir -p "$ROOT/cuda_home/include"
for d in "$SP"/nvidia/cuda_runtime/include "$SP"/nvidia/cuda_cccl/include "$SP"/nvidia/cuda_nvrtc/include; do
  [ -d "$d" ] && cp -rs "$d"/. "$ROOT/cuda_home/include/" 2>/dev/null || true
done
export CUDA_HOME=$ROOT/cuda_home
export NH81_PROMPT_IDS=$NH81_CODE_DIR/prompts_ids.json

# 4. Host record (no assumption).
{ echo "== nvidia-smi -q"; nvidia-smi -q; echo "== uname"; uname -a; echo "== pagesize"; getconf PAGESIZE
  echo "== loadavg"; cat /proc/loadavg; echo "== pip freeze"; "$PY" -m pip freeze 2>/dev/null || true
  echo "== ps"; ps -eo pid,user,pcpu,comm,args; } > "$OUT/host.txt" 2>&1

# 5. Smoke test, TF32 check, iso-quality (reproduction check vs macM1).
$PY -c "import mlx.core as mx; mx.eval(mx.ones(4) @ mx.ones(4)); print('smoke ok', mx.default_device())" | tee "$OUT/smoke.txt"
for f in 0 1; do MLX_ENABLE_TF32=$f $PY "$HERE/tf32_check.py"; done | tee "$OUT/tf32_check.jsonl"
(cd "$NH81_CODE_DIR" && $PY isoquality.py --dense "$ROOT/ckpt/ann1_ckpt_12000.npz" \
  --spiking "$ROOT/ckpt/spkt_ckpt_12000.npz" --data "$ROOT/data" \
  --macm1 "$ROOT/data/selection.json" --out "$OUT/isoquality.json")
$PY -c "import json,sys; m=json.load(open('$OUT/isoquality.json'))['models']
bad=[(k,s) for k,v in m.items() for s in ('val','test') if abs(v[s+'_dev'])>0.005]
sys.exit('not replicable as specified: %s' % bad if bad else 0)"

# 6. Rest threshold (deadline: 2 h from here), measurement, final probe.
DEADLINE=$(( $(date +%s) + 7200 ))
ARGS=(--dense "$ROOT/ckpt/ann1_ckpt_12000.npz" --spiking "$ROOT/ckpt/spkt_ckpt_12000.npz" --test "$ROOT/data/test.i32")
timeout --signal=INT --kill-after=60 "$(( DEADLINE - $(date +%s) ))" $PY "$HERE/rest_probe_gb300.py" --out "$OUT/rest_threshold.json" "${ARGS[@]}"
export NH81_REST_THRESHOLD=$OUT/rest_threshold.json NH81_REST_THRESHOLD_IN=$OUT/rest_threshold.json
LEFT=$(( DEADLINE - $(date +%s) ))
[ "$LEFT" -gt 0 ] || { echo "deadline reached before measurement"; exit 3; }
set +e
timeout --signal=INT --kill-after=60 "$LEFT" $PY "$HERE/measure_gb300.py" "${ARGS[@]}" --out "$OUT/measure.jsonl"
RC=$?
echo "measure exit $RC" | tee "$OUT/measure_exit.txt"
set -e
timeout --signal=INT --kill-after=60 120 $PY "$HERE/rest_probe_gb300.py" --final --out "$OUT/final_probe.json"
$PY "$HERE/describe.py" --measure "$OUT/measure.jsonl" --out "$OUT/summary.json"
exit "$RC"
