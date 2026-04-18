<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-17 | Updated: 2026-04-17 -->

# deploy/systemd

## Purpose
Linux systemd unit files for micro_kiki services. Two units ship here:
- `micro-kiki.service` runs the vLLM inference server on kxkm-ai (RTX 4090, Q4 quant serving).
- `aeon-compress.service` runs the Aeon memory compression daemon on an hourly interval (scoped to Mac Studio in practice — `User=clems` + `/Users/clems/...` paths — despite living under `systemd/`).

Parent `deploy/README.md` documents the staging + `systemctl` workflow.

## Key Files

| File | Description |
|------|-------------|
| `micro-kiki.service` | vLLM serving on kxkm-ai. `User=kxkm`, `WorkingDirectory=/home/kxkm/micro-kiki`, `ExecStart=uv run python -m src.serving.vllm_server`. `Restart=on-failure`, `RestartSec=30`. Envs: `PYTHONUNBUFFERED=1`, `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`. |
| `aeon-compress.service` | Aeon memory compression daemon. `User=clems`, `WorkingDirectory=/Users/clems/micro-kiki`, `ExecStart=uv run python scripts/aeon_compress_daemon.py --interval 3600 --max-age-days 30`. `Restart=on-failure`, `RestartSec=60`. |

## For AI Agents

### Working In This Directory

- **Staging only.** Per parent `README.md`: "do NOT enable without review". Copy to `/etc/systemd/system/`, `systemctl daemon-reload`, then `start`/`enable` intentionally.
- `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` is load-bearing — the router swaps LoRA adapters at runtime. Do not drop this env.
- `micro-kiki.service` paths are Linux (`/home/kxkm/...`); `aeon-compress.service` paths are Mac (`/Users/clems/...`). The `aeon-compress.service` lives in `systemd/` but runs on Mac via **systemd-via-homebrew is not standard** — on Mac Studio this should actually be run as a launchd agent. Treat the file as a portable template; if you deploy Aeon on Linux, change `User` and `WorkingDirectory`.
- Keep `Restart=on-failure` (not `always`) — a hard config error should not crash-loop at 1 Hz.
- `After=network.target` is correct for both — vLLM needs the network for the HTTP API, Aeon may reach the memory backends (Qdrant / Neo4j on Tower).
- When adding a new unit: one service per file, `[Unit] / [Service] / [Install]` in that order, `WantedBy=multi-user.target` for always-on system services.

### Testing Requirements

- `systemd-analyze verify <unit>` MUST pass.
- Dry-run: `systemctl daemon-reload && systemctl start <service> && systemctl status <service>` after staging to `/etc/systemd/system/`.
- Logs: `journalctl -u <service> -f` — expect "Uvicorn running on ..." (vLLM) or "Aeon compression cycle complete" (aeon-compress).

### Common Patterns

- `uv run python -m <module>` entry style (matches the launchd plist convention).
- `PYTHONUNBUFFERED=1` env for live log tailing.
- Absolute paths for `ExecStart` binary (`/home/kxkm/.local/bin/uv`, `/Users/clems/.local/bin/uv`).
- `RestartSec` 30 for serving (fast recovery), 60 for batch (avoid thrash).

## Dependencies

### Internal
- `micro-kiki.service` launches `src.serving.vllm_server` — must be importable.
- `aeon-compress.service` launches `scripts/aeon_compress_daemon.py`.
- Paired with `../launchd/cc.saillant.micro-kiki.plist` (macOS MLX serving counterpart).
- Documented in `../README.md`.

### External
- systemd (Linux). `uv` at pinned paths. vLLM + CUDA drivers on kxkm-ai.
- Aeon backends: Qdrant (:6333-6334), Neo4j (on Tower) — see `/home/kxkm/CLAUDE.md` infrastructure map.

<!-- MANUAL: -->
