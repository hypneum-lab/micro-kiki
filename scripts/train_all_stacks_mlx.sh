#!/usr/bin/env bash
# train_all_stacks_mlx.sh — Sequential MLX LoRA training on Studio
#
# Usage: ./scripts/train_all_stacks_mlx.sh [start_domain] [--iters N]
#
# Trains all 32 domain stacks sequentially using mlx_lm LoRA on Studio.
# Each stack trains for --iters iterations (default 500), saves adapter
# to outputs/stacks/stack-<domain>/, then moves to the next.
#
# Requires: KIKI-Mac_tunner venv with mlx + mlx_lm installed.

set -euo pipefail

VENV="${HOME}/KIKI-Mac_tunner/.venv/bin/python3"
MODEL="models/qwen3.5-35b-a3b"
DATA_ROOT="${HOME}/KIKI-Mac_tunner/data/micro-kiki"
OUTPUT_ROOT="outputs/stacks"
LOG_DIR="outputs"
ITERS="${2:-500}"
BATCH_SIZE=2
LR="2e-4"

# Curriculum order (matching configs/stack-*.yaml)
DOMAINS=(
    chat-fr reasoning python typescript cpp rust
    html-css shell sql yaml-json docker kicad-dsl spice lua-upy
    embedded stm32 iot freecad platformio power emc dsp spice-sim electronics kicad-pcb
    web-frontend web-backend music-audio devops llm-orch math security
)

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${CYAN}[train]${NC} $(date +%H:%M:%S) $*"; }
ok()   { echo -e "${GREEN}[  OK ]${NC} $(date +%H:%M:%S) $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $(date +%H:%M:%S) $*"; }
err()  { echo -e "${RED}[ ERR]${NC} $(date +%H:%M:%S) $*" >&2; }

# Find start position
START="${1:-chat-fr}"
STARTED=false

log "Sequential MLX LoRA training: ${#DOMAINS[@]} domains"
log "Model: $MODEL | Iters: $ITERS | Batch: $BATCH_SIZE | LR: $LR"
log "Starting from: $START"
echo ""

TRAINED=0
FAILED=0

for domain in "${DOMAINS[@]}"; do
    # Skip until we reach the start domain
    if [ "$STARTED" = false ]; then
        if [ "$domain" = "$START" ]; then
            STARTED=true
        else
            continue
        fi
    fi

    ADAPTER_DIR="${OUTPUT_ROOT}/stack-${domain}"
    DATA_DIR="${DATA_ROOT}/${domain}"
    LOG_FILE="${LOG_DIR}/train-${domain}-mlx.log"

    # Check data exists
    if [ ! -f "${DATA_DIR}/train.jsonl" ]; then
        warn "No data for ${domain} at ${DATA_DIR}/train.jsonl — skipping"
        continue
    fi

    EXAMPLES=$(wc -l < "${DATA_DIR}/train.jsonl" | tr -d ' ')

    # Skip if adapter already exists
    if [ -f "${ADAPTER_DIR}/adapters.safetensors" ]; then
        ok "${domain}: adapter exists, skipping (${ADAPTER_DIR})"
        continue
    fi

    log "Training ${domain} (${EXAMPLES} examples) → ${ADAPTER_DIR}"

    mkdir -p "${ADAPTER_DIR}"

    "$VENV" -m mlx_lm lora \
        --model "$MODEL" \
        --train \
        --data "$DATA_DIR" \
        --adapter-path "$ADAPTER_DIR" \
        --batch-size "$BATCH_SIZE" \
        --iters "$ITERS" \
        --learning-rate "$LR" \
        --grad-checkpoint \
        --save-every 100 \
        --steps-per-report 10 \
        --max-seq-length 4096 \
        > "$LOG_FILE" 2>&1

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ] && [ -f "${ADAPTER_DIR}/adapters.safetensors" ]; then
        LOSS=$(grep "^Iter ${ITERS}:" "$LOG_FILE" 2>/dev/null | grep -oP 'Train loss \K[0-9.]+' || echo "?")
        ok "${domain}: done (loss ${LOSS})"
        TRAINED=$((TRAINED + 1))

        # Append to progress
        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Trained: stack-${domain} (${EXAMPLES} examples, loss ${LOSS})" >> .ralph/progress.txt
    else
        err "${domain}: FAILED (exit code ${EXIT_CODE})"
        FAILED=$((FAILED + 1))
        tail -5 "$LOG_FILE" 2>/dev/null
        echo ""
        # Continue to next domain (don't abort pipeline)
    fi
done

echo ""
log "═══════════════════════════════════════"
log "Training complete: ${TRAINED} trained, ${FAILED} failed"
log "═══════════════════════════════════════"

exit $((FAILED > 0 ? 1 : 0))
