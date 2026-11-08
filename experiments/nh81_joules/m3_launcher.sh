#!/usr/bin/env bash
# NH-81 macM3 (M3 Pro) replication launcher (tag prereg-nh81-m3pro-20261030).
# Usage: m3_launcher.sh [quick]  -- quick = code dry run (NH81_QUICK=1)
set -euo pipefail
ROOT=$HOME/scratch/nh81-m3
CODE=$ROOT/code/experiments/nh81_joules
PY=$ROOT/code/.venv/bin/python
export MLX_ENABLE_TF32=0 NH81_HOST=m3pro NH81_BPE=$ROOT/bsk_bpe.json
ARGS=(--dense $ROOT/ckpt/ann1_ckpt_12000.npz --spiking $ROOT/ckpt/spkt_ckpt_12000.npz
      --test $ROOT/data/test.i32)
cd "$CODE"
shasum -a 256 -c - <<SHA
3c0891fbb528f122b5ac8179b180fab879d38415e385c29b7acb7be7ef12afd7  $ROOT/ckpt/ann1_ckpt_12000.npz
e9a3c9294c4737eddf019fa2ace51f2fe967c942d89039fcbd2870aa26eaa074  $ROOT/ckpt/spkt_ckpt_12000.npz
ad59a2f13afab4f47983561f6f12dd269e31076e4a411a8e0dd071c1021cfd2d  $ROOT/data/test.i32
78bce0d30395fcab80c370e6e6096a84d96f5ce2d908b1a2f94fbfd9dad60f4c  $ROOT/bsk_bpe.json
SHA
if [ "${1:-}" = quick ]; then
  mkdir -p $ROOT/dryrun; rm -f $ROOT/dryrun/quick.jsonl
  NH81_QUICK=1 exec caffeinate -i $PY measure.py "${ARGS[@]}" --pairs 2 --skip-guard --out $ROOT/dryrun/quick.jsonl
fi
OUT=$ROOT/results
mkdir -p $OUT
[ -e $OUT/measure.jsonl ] && { echo "refusing: $OUT/measure.jsonl exists"; exit 1; }
caffeinate -i $PY idle_probe_m3.py --seconds 60 --out $OUT/idle_probe.json
exec caffeinate -i $PY measure.py "${ARGS[@]}" --out $OUT/measure.jsonl
