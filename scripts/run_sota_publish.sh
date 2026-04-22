#!/usr/bin/env bash
# run_sota_publish.sh — sequential V4 SOTA eval matrix on Mac Studio.
#
# Matrix: 4 benchmarks (humaneval / mbpp / gsm8k / multipl-e-py)
#       x (base + 5 adapters) = 24 runs, ~6 hours total.
#
# Runs one model at a time (no parallel loads). Absolute paths only — no
# `cd`, so the electron user can invoke this directly.
#
#   bash /Users/clems/tmp/sota-eval/run_sota_publish.sh
#
# Outputs: /tmp/sota-publish/<bench>-<adapter>.json
# Master log: /tmp/sota-publish.log
set -euo pipefail

VENV="/Users/clems/KIKI-Mac_tunner/.venv/bin/python"
SCRIPTS="/Users/clems/tmp/sota-eval"
MODEL="/Users/clems/KIKI-Mac_tunner/models/Qwen3.6-35B-A3B"
ADAPTERS="/Users/clems/KIKI-Mac_tunner/output/micro-kiki/lora-qwen36-35b-v4-sota"
FIXTURES="/tmp/sota-eval-fixtures"
OUTDIR="/tmp/sota-publish"
LOGFILE="/tmp/sota-publish.log"

mkdir -p "$OUTDIR" "$FIXTURES"
: > "$LOGFILE"

log() {
    local msg="$1"
    local ts
    ts="$(date '+%Y-%m-%d %H:%M:%S')"
    printf '[%s] %s\n' "$ts" "$msg" | tee -a "$LOGFILE"
}

# Per-bench defaults
HE_N=20            # HumanEval
MBPP_N=50          # MBPP
GSM_N=50           # GSM8K
MPE_N=20           # MultiPL-E (Python slice)

# Tune max-tokens per bench
HE_TOK=384
MBPP_TOK=512
GSM_TOK=384
MPE_TOK=512

# Adapter list: "<label> <adapter-subdir>"; empty subdir => base-only run.
ADAPTER_LIST=(
    "base:"
    "python:python"
    "cpp:cpp"
    "typescript:typescript"
    "rust:rust"
    "shell:shell"
    "math:math"
    "reasoning:reasoning"
)

# Bench list: "<bench> <script>"
BENCH_LIST=(
    "humaneval:eval_humaneval_v4.py"
    "mbpp:eval_mbpp_v4.py"
    "gsm8k:eval_gsm8k_v4.py"
    "multipl-e-py:eval_multipl_e_v4.py"
)

run_one() {
    local bench="$1"
    local script="$2"
    local adapter_label="$3"
    local adapter_subdir="$4"

    local out="$OUTDIR/${bench}-${adapter_label}.json"
    local cmd=( "$VENV" "$SCRIPTS/$script"
                --base-model "$MODEL"
                --output "$out"
                --label "$adapter_label" )

    if [[ -n "$adapter_subdir" ]]; then
        cmd+=( --adapter "$ADAPTERS/$adapter_subdir" )
    fi

    case "$bench" in
        humaneval)
            cmd+=( --fixture "$FIXTURES/humaneval_${HE_N}.jsonl"
                   --n "$HE_N" --max-tokens "$HE_TOK" )
            ;;
        mbpp)
            cmd+=( --fixture "$FIXTURES/mbpp_test.jsonl"
                   --n "$MBPP_N" --max-tokens "$MBPP_TOK" )
            ;;
        gsm8k)
            cmd+=( --fixture "$FIXTURES/gsm8k_test.jsonl"
                   --n "$GSM_N" --max-tokens "$GSM_TOK" )
            ;;
        multipl-e-py)
            cmd+=( --lang py
                   --fixture "$FIXTURES/multipl_e_humaneval_py.jsonl"
                   --n "$MPE_N" --max-tokens "$MPE_TOK" )
            ;;
        *)
            log "ERROR: unknown bench $bench"
            return 2
            ;;
    esac

    log "START bench=$bench adapter=$adapter_label -> $out"
    if "${cmd[@]}" >> "$LOGFILE" 2>&1; then
        log "OK    bench=$bench adapter=$adapter_label"
    else
        local rc=$?
        log "FAIL  bench=$bench adapter=$adapter_label rc=$rc"
        return $rc
    fi
}

log "sota-publish: starting matrix on $(hostname)"
log "scripts=$SCRIPTS model=$MODEL adapters=$ADAPTERS"

total=0
for b in "${BENCH_LIST[@]}"; do
    for a in "${ADAPTER_LIST[@]}"; do
        total=$((total + 1))
    done
done
log "matrix: $total runs"

idx=0
for bench_entry in "${BENCH_LIST[@]}"; do
    bench="${bench_entry%%:*}"
    script="${bench_entry#*:}"
    for adapter_entry in "${ADAPTER_LIST[@]}"; do
        adapter_label="${adapter_entry%%:*}"
        adapter_subdir="${adapter_entry#*:}"
        idx=$((idx + 1))
        log "[$idx/$total] bench=$bench adapter=$adapter_label"
        run_one "$bench" "$script" "$adapter_label" "$adapter_subdir"
    done
done

log "sota-publish: ALL $total RUNS COMPLETED"
exit 0
