#!/usr/bin/env bash
# V-35B v4 SOTA — LoRA r16 alpha16, 32 layers, LR 1e-5
# Based on arXiv 2602.04998 + 2602.06204 + 2601.22708
set -euo pipefail

MODEL="models/Qwen3.6-35B-A3B"
DATA="data/micro-kiki"
OUTPUT="output/micro-kiki/lora-qwen36-35b-v4-sota"
PYTHON="/opt/homebrew/bin/python3.12"

CURRICULUM=(
  "chat-fr:1000"
  "reasoning:1000"
  "python:1000"
  "typescript:500"
  "cpp:500"
  "rust:500"
  "html-css:500"
  "shell:500"
  "sql:500"
  "yaml-json:200"
  "docker:500"
  "kicad-dsl:500"
  "spice:200"
  "lua-upy:200"
  "embedded:500"
  "stm32:300"
  "iot:200"
  "freecad:500"
  "platformio:100"
  "power:500"
  "emc:200"
  "dsp:200"
  "spice-sim:100"
  "electronics:500"
  "kicad-pcb:500"
  "web-frontend:200"
  "web-backend:200"
  "music-audio:100"
  "devops:200"
  "llm-orch:200"
  "math:200"
  "security:300"
  "components:500"
  "llm-ops:200"
  "ml-training:200"
)

mkdir -p "$OUTPUT"

echo "================================================================"
echo "V-35B v4 SOTA — LoRA r16 a16, 32 layers, LR 1e-5"
echo "arxiv: 2602.04998, 2602.06204, 2601.22708"
echo "================================================================"

for entry in "${CURRICULUM[@]}"; do
  domain="${entry%%:*}"
  iters="${entry##*:}"
  adapter="$OUTPUT/$domain"

  [ ! -f "$DATA/$domain/train.jsonl" ] && echo "SKIP $domain (no data)" && continue
  [ -f "$adapter/adapters.safetensors" ] && echo "SKIP $domain (done)" && continue

  n=$(wc -l < "$DATA/$domain/train.jsonl")
  echo ""
  echo "=== $domain (n=$n, iters=$iters, LoRA r16 a16, 32L) ==="

  cat > "$OUTPUT/config-$domain.yaml" << YAMLEOF
model: "$MODEL"
data: "$DATA/$domain"
train: true
fine_tune_type: lora
iters: $iters
batch_size: 1
learning_rate: 1e-5
adapter_path: "$adapter"
max_seq_length: 1024
num_layers: 32
steps_per_report: 50
steps_per_eval: 100
grad_checkpoint: true
lora_parameters:
  rank: 16
  alpha: 16.0
  dropout: 0.0
  scale: 20.0
YAMLEOF

  $PYTHON -c "
import mlx.core as mx
mx.set_memory_limit(460 * 1024**3)
mx.set_cache_limit(96 * 1024**3)  # x3 per fix(train): cache_limit x3 + rank 8 for MoE (5fb1b87) — avoids Metal OOM at rank 16 + num_layers 32
import sys
sys.argv = ['mlx_lm', 'lora', '-c', '$OUTPUT/config-$domain.yaml']
from mlx_lm.cli import main
main()
" 2>&1 | tee "$OUTPUT/log-$domain.txt"

  echo "$domain DONE"
  sleep 5
done

echo "================================================================"
echo "ALL COMPLETE"
echo "================================================================"
