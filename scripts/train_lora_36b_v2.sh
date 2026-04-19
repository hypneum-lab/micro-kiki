#!/usr/bin/env bash
# Train LoRA on Qwen3.6-35B-A3B — GPU hang fix version
# Lower seq_len (1024), aggressive cache clear (0.3), sleep between stacks
set -euo pipefail

MODEL="models/Qwen3.6-35B-A3B"
DATA="data/micro-kiki"
OUTPUT="output/micro-kiki/lora-qwen36-35b"
PYTHON="/opt/homebrew/bin/python3.12"

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

for i in "${!CURRICULUM[@]}"; do
  domain="${CURRICULUM[$i]}"
  idx=$((i + 1))
  adapter="$OUTPUT/$domain"

  [ ! -f "$DATA/$domain/train.jsonl" ] && echo "[$idx] SKIP $domain" && continue
  [ -f "$adapter/adapters.safetensors" ] && echo "[$idx] SKIP $domain (done)" && continue

  n=$(wc -l < "$DATA/$domain/train.jsonl")
  iters=$(python3 -c "print(min(300, max(50, $n // 30)))")

  echo "[$idx/${#CURRICULUM[@]}] $domain ($n ex, $iters iters)"

  $PYTHON -m mlx_lm lora \
    --model "$MODEL" \
    --data "$DATA/$domain" \
    --train \
    --iters "$iters" \
    --batch-size 1 \
    --learning-rate 1e-5 \
    --adapter-path "$adapter" \
    --max-seq-length 1024 \
    --steps-per-report 25 \
    --steps-per-eval 50 \
    --grad-checkpoint \
    --clear-cache-threshold 0.3 \
    2>&1 | tail -8

  echo "[$idx] $domain DONE"
  sleep 5  # Let GPU cool down between stacks
done

echo "ALL COMPLETE"
