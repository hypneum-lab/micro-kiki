#!/usr/bin/env bash
# Train LoRA adapters on Qwen3.6-35B-A3B for all 35 domains
set -euo pipefail

MODEL="models/Qwen3.6-35B-A3B"
DATA_BASE="data/micro-kiki"
OUTPUT_BASE="output/micro-kiki/lora-qwen36-35b"
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

mkdir -p "$OUTPUT_BASE"

echo "================================================================"
echo "LoRA on Qwen3.6-35B-A3B — ${#CURRICULUM[@]} domains"
echo "Model: $MODEL | BF16 | grad-checkpoint"
echo "================================================================"

for i in "${!CURRICULUM[@]}"; do
  domain="${CURRICULUM[$i]}"
  idx=$((i + 1))
  data_dir="$DATA_BASE/$domain"
  adapter_dir="$OUTPUT_BASE/$domain"

  [ ! -f "$data_dir/train.jsonl" ] && echo "[$idx] SKIP $domain (no data)" && continue
  [ -f "$adapter_dir/adapters.safetensors" ] && echo "[$idx] SKIP $domain (done)" && continue

  n_examples=$(wc -l < "$data_dir/train.jsonl")
  iters=$(python3 -c "print(min(500, max(100, $n_examples // 20)))")

  echo ""
  echo "================================================================"
  echo "[$idx/${#CURRICULUM[@]}] $domain ($n_examples ex, $iters iters)"
  echo "================================================================"

  $PYTHON -m mlx_lm lora \
    --model "$MODEL" \
    --data "$data_dir" \
    --train \
    --iters "$iters" \
    --batch-size 1 \
    --learning-rate 1e-5 \
    --adapter-path "$adapter_dir" \
    --max-seq-length 2048 \
    --steps-per-report 50 \
    --steps-per-eval 100 \
    --grad-checkpoint \
    --clear-cache-threshold 0.5 \
    2>&1 | tee "$OUTPUT_BASE/log-$domain.txt"

  echo "[$idx] $domain DONE"
done

echo "================================================================"
echo "ALL 35 DOMAINS COMPLETE"
echo "================================================================"
ls "$OUTPUT_BASE"/*/adapters.safetensors 2>/dev/null | wc -l
echo "adapters trained on Qwen3.6-35B-A3B"
