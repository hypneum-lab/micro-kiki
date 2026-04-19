<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-17 | Updated: 2026-04-17 -->

# deploy/launchd

## Purpose
macOS `launchd` service definitions for running micro_kiki in production on the Mac Studio M3 Ultra. The single plist here (`cc.saillant.micro-kiki.plist`) launches the MLX inference server (`src.serving.mlx_server`) under the `clems` user, keeps it alive across crashes, and writes stdout/stderr logs to `outputs/mlx-server.{log,err}`. The parent `deploy/README.md` documents the staging + `launchctl load` / `unload` workflow.

## Key Files

| File | Description |
|------|-------------|
| `cc.saillant.micro-kiki.plist` | launchd agent plist. `Label: cc.saillant.micro-kiki`. `ProgramArguments`: `/Users/clems/.local/bin/uv run python -m src.serving.mlx_server`. `WorkingDirectory: /Users/clems/micro-kiki`. `RunAtLoad: true`, `KeepAlive: true`, `PYTHONUNBUFFERED=1`. Logs to `outputs/mlx-server.log` and `outputs/mlx-server.err`. |

## For AI Agents

### Working In This Directory

- **Staging only** — the parent `deploy/README.md` is explicit: "do NOT load in production without review". Copy to `~/Library/LaunchAgents/` and `launchctl load/unload` intentionally.
- The plist is Mac-specific. Hardcoded paths (`/Users/clems/...`) are intentional; this service only runs on Mac Studio. Do not parametrize — a different operator gets a different plist file.
- If you rename the launch target (e.g. swap `src.serving.mlx_server` for a different entrypoint), update both `ProgramArguments` and any documentation in `deploy/README.md`.
- Keep `KeepAlive: true` — the MLX server is the serving plane; a crash-loop restart is correct behavior here.
- Log paths are relative to `WorkingDirectory`; if you change `WorkingDirectory`, fix log paths in the same edit.
- Reverse-DNS label (`cc.saillant.micro-kiki`) is the domain owner's convention — don't rename it lightly; it shows up in `launchctl list` and log rotation configs.

### Testing Requirements

- `plutil -lint cc.saillant.micro-kiki.plist` MUST pass (valid property list).
- Dry-run: `launchctl load -w <staged-path>` then `launchctl list | grep micro-kiki`, expect PID. `launchctl unload -w` to stop.
- Check log outputs land at the declared paths after first launch.

### Common Patterns

- `/Users/clems/.local/bin/uv` — absolute path to `uv` (launchd doesn't inherit a shell PATH).
- `PYTHONUNBUFFERED=1` — required for log tail during debugging.
- XML plist (not JSON, not binary) — readable in diffs, diff-friendly.

## Dependencies

### Internal
- Launches `src.serving.mlx_server` — that module must be importable from `WorkingDirectory`.
- Paired with `../systemd/micro-kiki.service` (Linux counterpart).
- Documented in `../README.md`.

### External
- macOS `launchd` (loaded via `launchctl`). `uv` must be installed at the pinned path.

<!-- MANUAL: -->
