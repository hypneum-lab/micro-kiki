<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-17 | Updated: 2026-04-17 -->

# deploy

## Purpose
Service definitions that stage micro-kiki as a long-running service on either host: launchd plists for the Mac Studio (MLX serving) and systemd units for Linux / kxkm-ai (vLLM serving + Aeon compression daemon). Nothing here is loaded automatically — the `README.md` gives explicit install and enable commands, and the repo convention is "stage, do NOT load in production without review".

## Key Files
| File | Description |
|------|-------------|
| `README.md` | Install & load/unload instructions for launchd (Mac) and systemd (Linux), plus Aeon compression daemon and Docker run |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `launchd/` | `cc.saillant.micro-kiki.plist` — Mac Studio MLX server launch agent |
| `systemd/` | `micro-kiki.service` (vLLM server, `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`, runs `uv run python -m src.serving.vllm_server`), `aeon-compress.service` (periodic compression via `scripts/aeon_compress_daemon.py --interval 3600 --max-age-days 30`) |

## For AI Agents

### Working In This Directory
- NEVER commit secrets (API tokens, HF tokens) into plists or unit files — reference them via `EnvironmentFile=` or launchd `EnvironmentVariables` sourced from an out-of-tree secret file.
- `micro-kiki.service` runs on kxkm-ai as user `kxkm`, CWD `/home/kxkm/micro-kiki`. Update both `User=` and `WorkingDirectory=` if those change.
- Do NOT enable the vLLM service on a non-GPU box; it assumes the RTX 4090 and will OOM / fail CUDA init elsewhere.
- Do NOT load the Mac plist into `~/Library/LaunchAgents/` in shared sessions — MLX holds VRAM permanently and blocks training. The repo pattern is stage-only.
- Keep `Restart=on-failure` + `RestartSec=30` — vLLM has crash patterns that benefit from cool-down.
- If you add a new service, also extend `README.md` with the install/load/unload commands.

### Testing Requirements
No pytest coverage here. Before committing a unit change:
```bash
systemd-analyze verify deploy/systemd/micro-kiki.service
plutil -lint deploy/launchd/cc.saillant.micro-kiki.plist
```
Integration testing is manual: stage on target host, `systemctl start --user` or `launchctl bootstrap`, inspect logs.

### Common Patterns
- systemd units: `Type=simple`, `PYTHONUNBUFFERED=1`, explicit `ExecStart` with full path to `uv`.
- Paths hardcoded to `/home/kxkm/micro-kiki/...` and `/home/kxkm/.local/bin/uv` — kxkm-ai layout.
- Aeon compression runs hourly with 30-day retention.

## Dependencies

### Internal
`src.serving.vllm_server` (systemd), `src.serving.mlx_server` (launchd), `scripts/aeon_compress_daemon.py`.

### External
- `uv` on PATH for both hosts.
- vLLM install with `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` support.
- launchd on macOS; systemd on Linux.

<!-- MANUAL: -->
