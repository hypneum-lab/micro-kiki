#!/bin/bash
# eval_mbpp_full.sh — MBPP pass@1 N=500 across all coding adapters (original + v2)
#
# Usage:
#   ./scripts/eval_mbpp_full.sh
#
# Kills the pipeline server to free Metal, runs 14 configs sequentially, prints
# a summary table, then restarts the server. Expected wall-time: ~4h.
#
# Do NOT run during the day — this is an overnight job.

set -euo pipefail

ADAPTERS_DIR="/Users/clems/KIKI-Mac_tunner/output/micro-kiki/lora-qwen36-35b-v4-sota"
BASE_MODEL="/Users/clems/KIKI-Mac_tunner/models/Qwen3.6-35B-A3B"
OUTPUT_DIR="results/mbpp-full-n500"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
N=500

cd "$REPO_DIR"

mkdir -p "$OUTPUT_DIR"

echo "================================================================"
echo "  MBPP full eval — N=$N — $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Output dir: $OUTPUT_DIR"
echo "================================================================"

# ---------------------------------------------------------------------------
# Kill pipeline server to free Metal before loading the 35B model
# ---------------------------------------------------------------------------
echo ""
echo "--- Stopping pipeline server ---"
# Try full_pipeline_server first, then any uvicorn on port 9200
pkill -f "full_pipeline_server" 2>/dev/null && echo "  killed full_pipeline_server" || true
pkill -f "port 9200"            2>/dev/null && echo "  killed port-9200 process"    || true
sleep 5
echo "  Metal resources freed."

# ---------------------------------------------------------------------------
# Run each configuration
# ---------------------------------------------------------------------------
ADAPTERS=(
    "base"
    "python"
    "python-v2"
    "cpp"
    "cpp-v2"
    "rust"
    "rust-v2"
    "shell"
    "shell-v2"
    "typescript"
    "typescript-v2"
    "html-css"
    "html-css-v2"
    "math"
)

for adapter in "${ADAPTERS[@]}"; do
    echo ""
    echo "================================================================"
    echo "  Config: $adapter  — $(date '+%H:%M:%S')"
    echo "================================================================"

    OUTPUT_FILE="$OUTPUT_DIR/mbpp-${adapter}.json"
    LOG_FILE="$OUTPUT_DIR/mbpp-${adapter}.log"

    if [ -f "$OUTPUT_FILE" ]; then
        echo "  [SKIP] $OUTPUT_FILE already exists — delete to re-run."
        continue
    fi

    if [ "$adapter" = "base" ]; then
        ADAPTER_ARGS=()
    else
        ADAPTER_PATH="$ADAPTERS_DIR/$adapter"
        if [ ! -d "$ADAPTER_PATH" ]; then
            echo "  [WARN] Adapter directory not found: $ADAPTER_PATH — skipping."
            continue
        fi
        ADAPTER_ARGS=("--adapter" "$ADAPTER_PATH")
    fi

    .venv/bin/python scripts/eval_mbpp_v4.py \
        --base-model  "$BASE_MODEL" \
        "${ADAPTER_ARGS[@]}" \
        --label       "$adapter" \
        --n           "$N" \
        --output      "$OUTPUT_FILE" \
        2>&1 | tee -a "$LOG_FILE"

    echo "  Done: $OUTPUT_FILE"
done

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  SUMMARY — MBPP pass@1 at N=$N"
echo "================================================================"
printf "  %-20s  %s\n" "Config" "pass@1"
printf "  %-20s  %s\n" "------" "------"

for adapter in "${ADAPTERS[@]}"; do
    OUTPUT_FILE="$OUTPUT_DIR/mbpp-${adapter}.json"
    if [ ! -f "$OUTPUT_FILE" ]; then
        printf "  %-20s  %s\n" "$adapter" "MISSING"
        continue
    fi

    # Extract pass@1 from the JSON result (field written by eval_mbpp_v4.py)
    PASS1=$(.venv/bin/python3 - "$OUTPUT_FILE" <<'PYEOF'
import json, sys
data = json.load(open(sys.argv[1]))
v = data.get("pass@1")
if v is None:
    # Fallback: count per_problem results
    results = data.get("per_problem", data.get("results", []))
    passed = sum(1 for p in results if p.get("passed", False))
    total = len(results)
    print(f"{passed}/{total}  ({passed/total*100:.1f}%)" if total else "0/0")
else:
    n = data.get("n_problems", data.get("n", "?"))
    passed = round(v * n) if isinstance(n, int) else "?"
    print(f"{passed}/{n}  ({v*100:.1f}%)")
PYEOF
)
    printf "  %-20s  %s\n" "$adapter" "$PASS1"
done

echo ""
echo "  Full results in: $OUTPUT_DIR"
echo "================================================================"

# ---------------------------------------------------------------------------
# Restart pipeline server
# ---------------------------------------------------------------------------
echo ""
echo "--- Restarting pipeline server ---"
nohup .venv/bin/python -m uvicorn \
    src.serving.full_pipeline_server:make_default_app \
    --factory \
    --host 127.0.0.1 \
    --port 9200 \
    > /tmp/kiki-server.log 2>&1 &

echo "  Server PID $! — logs at /tmp/kiki-server.log"
echo ""
echo "Done — $(date '+%Y-%m-%d %H:%M:%S')"
