#!/usr/bin/env bash
# V-35B v2 — Metal optimized, 32/40 layers, 1000 iters foundations
set -euo pipefail

MODEL="models/Qwen3.6-35B-A3B"
DATA="data/micro-kiki"
OUTPUT="output/micro-kiki/lora-qwen36-35b-v2"
PYTHON="/opt/homebrew/bin/python3.12"
NUM_LAYERS=32
SEQ_LEN=1024
LR="2e-5"

CURRICULUM=(
  chat-fr reasoning python typescript cpp rust
  html-css shell sql yaml-json docker kicad-dsl spice lua-upy
  embedded stm32 iot freecad platformio power emc dsp
  spice-sim electronics kicad-pcb
  web-frontend web-backend music-audio devops llm-orch
  math security
  components llm-ops ml-training
)

mkdir -p "$OUTPUT"

echo "================================================================"
echo "V-35B v2 — 32 layers, Metal optimized"
echo "================================================================"

for i in "${!CURRICULUM[@]}"; do
  domain="${CURRICULUM[$i]}"
  idx=$((i + 1))
  adapter="$OUTPUT/$domain"

  [ ! -f "$DATA/$domain/train.jsonl" ] && echo "[$idx] SKIP $domain (no data)" && continue
  [ -f "$adapter/adapters.safetensors" ] && echo "[$idx] SKIP $domain (done)" && continue

  n=$(wc -l < "$DATA/$domain/train.jsonl")

  # Foundations: 1000 iters, others: adaptive
  case "$domain" in
    chat-fr|reasoning|python) iters=1000 ;;
    *) iters=$(python3 -c "print(min(500, max(100, $n // 20)))")  ;;
  esac

  echo ""
  echo "[$idx/${#CURRICULUM[@]}] $domain ($n ex, $iters iters, $NUM_LAYERS layers)"

  $PYTHON -c "
import mlx.core as mx
mx.set_memory_limit(460 * 1024**3)
mx.set_cache_limit(32 * 1024**3)

import sys
sys.argv = [
    'mlx_lm', 'lora',
    '--model', '$MODEL',
    '--data', '$DATA/$domain',
    '--train',
    '--iters', '$iters',
    '--batch-size', '1',
    '--learning-rate', '$LR',
    '--adapter-path', '$adapter',
    '--max-seq-length', '$SEQ_LEN',
    '--num-layers', '$NUM_LAYERS',
    '--steps-per-report', '50',
    '--steps-per-eval', '100',
    '--grad-checkpoint',
]
from mlx_lm.cli import main
main()
" 2>&1 | tee "$OUTPUT/log-$domain.txt"

  echo "[$idx] $domain DONE"
  sleep 5
done

echo "================================================================"
echo "ALL COMPLETE"
echo "================================================================"
