<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-17 | Updated: 2026-04-17 -->

# docker

## Purpose
Container image definitions. Currently only `vllm.Dockerfile`, a thin wrapper over `vllm/vllm-openai:latest` that enables runtime LoRA hot-swap (`VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`) so the router can activate / deactivate per-domain adapters without restarting the server. The Dockerfile has no CMD / ENTRYPOINT by design — entrypoint is provided by the caller (docker-compose or CLI) so the exact vLLM flags can live alongside deployment configs.

## Key Files
| File | Description |
|------|-------------|
| `vllm.Dockerfile` | 2 lines: `FROM vllm/vllm-openai:latest` + `ENV VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`. Entrypoint deferred to docker-compose |

## Subdirectories
None.

## For AI Agents

### Working In This Directory
- Keep the image minimal — `vllm/vllm-openai:latest` already ships CUDA, torch, and vLLM. Do NOT reinstall torch from pip; version drift breaks kernel ABI.
- `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` is REQUIRED: the router activates / deactivates stacks at runtime.
- This image is for kxkm-ai (RTX 4090) Q4 inference. Do NOT try to serve BF16 35B — 24 GB VRAM is not enough.
- Build with GPU runtime:
  ```bash
  docker build -f docker/vllm.Dockerfile -t micro-kiki-vllm .
  docker run --gpus all -p 8100:8100 micro-kiki-vllm
  ```
- If you pin `vllm` to a specific tag, coordinate with `deploy/systemd/micro-kiki.service` — they must use matching flags.
- No secrets here. HF tokens go via `--env` at run time, not `ENV` in the image.

### Testing Requirements
No pytest coverage. Before committing a Dockerfile change:
```bash
docker build -f docker/vllm.Dockerfile -t micro-kiki-vllm:test .
docker run --rm --gpus all micro-kiki-vllm:test vllm --help   # smoke
```
`tests/test_vllm_server.py` covers the serving module but not the image itself.

### Common Patterns
- Base image pinning: `latest` is fine here because upstream vLLM tags regularly. Bump to a SHA if reproducibility becomes critical.
- All runtime flags (model path, LoRA modules, max num seqs) are supplied by the orchestrator / compose file — keep the image generic.

## Dependencies

### Internal
Runs `src.serving.vllm_server` indirectly (via the vLLM openai-compatible server).

### External
- `vllm/vllm-openai` Docker Hub image (upstream).
- NVIDIA Container Toolkit on the host (`--gpus all`).

<!-- MANUAL: -->
