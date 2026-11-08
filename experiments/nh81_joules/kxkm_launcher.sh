#!/usr/bin/env bash
# NH-81 KXKM NVML replication launcher (prereg r2, r3, amendment r4: tag prereg-nh81-kxkm-20261030-r4).
# Usage: nh81-measure.sh [quick]   -- quick = code dry run (NH81_QUICK=1)
# Never stops or touches any service; read-only on everything but NH81 dirs.
set -euo pipefail
ROOT=/ai/saisail/nh81-kxkm
CODE=$ROOT/code/experiments/nh81_joules
PY=/ai/saisail/n10c-run/.venv/bin/python
export CUDA_HOME=/ai/saisail/n10c-bench/cuda_home MLX_ENABLE_TF32=0 NH81_HOST=kxkm
export NH81_PROMPT_IDS=$CODE/prompts_ids.json
ARGS=(--dense $ROOT/ckpt/ann1_ckpt_12000.npz --spiking $ROOT/ckpt/spkt_ckpt_12000.npz
      --test $ROOT/data/test.i32)
cd "$CODE"
sha256sum -c - <<SHA
3c0891fbb528f122b5ac8179b180fab879d38415e385c29b7acb7be7ef12afd7  $ROOT/ckpt/ann1_ckpt_12000.npz
e9a3c9294c4737eddf019fa2ace51f2fe967c942d89039fcbd2870aa26eaa074  $ROOT/ckpt/spkt_ckpt_12000.npz
ad59a2f13afab4f47983561f6f12dd269e31076e4a411a8e0dd071c1021cfd2d  $ROOT/data/test.i32
SHA
if [ "${1:-}" = quick ]; then
  mkdir -p $ROOT/dryrun; rm -f $ROOT/dryrun/quick.jsonl
  NH81_QUICK=1 exec $PY measure.py "${ARGS[@]}" --pairs 2 --skip-guard --out $ROOT/dryrun/quick.jsonl
fi
OUT=$ROOT/results
mkdir -p $OUT
[ -e $OUT/measure.jsonl ] && { echo "refusing: $OUT/measure.jsonl exists"; exit 1; }
$PY idle_probe.py --out $OUT/idle_probe.json
exec $PY measure.py "${ARGS[@]}" --out $OUT/measure.jsonl
