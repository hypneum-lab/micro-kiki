#!/bin/bash
# Deploy + (re)start the Studio LoRA swap server.
# Assumes: studio ssh alias, ~/micro-kiki checkout present, .venv with mlx_lm + fastapi + uvicorn.
set -euo pipefail

STUDIO_REPO="${STUDIO_REPO:-/Users/clems/micro-kiki}"
BASE_MODEL="${BASE_MODEL:-/Users/clems/models/Qwen3.6-35B-A3B}"
STACKS_ROOT="${STACKS_ROOT:-/Users/clems/micro-kiki/outputs/stacks}"
PORT="${PORT:-19000}"

echo "=== rsync server module ==="
rsync -az src/serving/ studio:"${STUDIO_REPO}/src/serving/"

echo "=== ensure deps on Studio ==="
ssh studio "cd ${STUDIO_REPO} && .venv/bin/pip install -q fastapi uvicorn || true"

echo "=== kill old server + launch ==="
ssh studio "pkill -f studio_lora_server 2>/dev/null; sleep 2; \
  nohup ${STUDIO_REPO}/.venv/bin/python -m src.serving.studio_lora_server \
    --base ${BASE_MODEL} \
    --stacks-root ${STACKS_ROOT} \
    --host 0.0.0.0 --port ${PORT} \
    > /tmp/studio-lora-server.log 2>&1 < /dev/null &"

echo "=== waiting for health on port ${PORT} ==="
for i in $(seq 1 60); do
    if ssh studio "curl -sf -o /dev/null http://localhost:${PORT}/v1/models"; then
        echo "READY (after ${i} × 10s)"
        exit 0
    fi
    sleep 10
done
echo "TIMEOUT — check /tmp/studio-lora-server.log on Studio"
exit 1
