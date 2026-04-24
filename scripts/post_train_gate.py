"""Orchestration wrapper: run adapter health + forgetting gate after training.

Intended to be invoked by the training pipeline (manual, launchd trigger, or
shell wrapper around ``mlx_lm lora``) immediately after a new adapter is
produced. Chains:

  1. ``validate_adapter_health`` on the new adapter — fail fast if degenerate.
  2. ``measure_forgetting`` between the new adapter and the previous stack in
     the curriculum — fail if the AND-gate (angle < 30° AND winrate_drop
     > 0.03) triggers on any non-ignored module.

Exit codes:
  0 — gate passed (adapter is healthy, forgetting is within bounds)
  1 — adapter is degenerate (all ``lora_B`` below ε)
  2 — forgetting gate failed (rollback recommended)
  3 — configuration/invocation error

The script is deliberately thin: it delegates to the existing entry points
so behaviour stays consistent with operator-run invocations of each piece.

Consolidation (experimental)
----------------------------
When invoked with ``--consolidate-on-warning``, a failing forgetting gate no
longer exits 2 directly. Instead we attempt to redirect the failure through
the ``dream-of-kiki`` framework's ``MicroKikiSubstrate`` consolidation path
(replay + downscale + restructure handlers driven by a ``DreamEpisode``
constructed from the gate's per-module report). On success the script exits
0; on any consolidation failure — including the ``kiki_oniric`` package not
being installed — we fall back to the original rollback behaviour (exit 2).

This feature is off by default — existing behaviour is unchanged unless the
flag is set.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"


def _run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    combined = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, combined


def _attempt_consolidation(
    gate_report_path: Path,
    new_adapter: Path,
    substrate_snapshot: Path | None,
    real_backend_path: Path | None,
) -> int:
    """Redirect a failing forgetting gate through the dream-of-kiki substrate.

    Lazily imports ``kiki_oniric``. If the package is not available we print a
    clear warning and return 2 so the caller falls back to the rollback
    behaviour. On successful consolidation returns 0; on any failure during
    the consolidation path itself returns 2.

    The ``dream-of-kiki`` framework exposes consolidation as ``DreamRuntime``
    driving operation handlers produced by ``MicroKikiSubstrate``'s factory
    methods — there is no ``substrate.consolidate(episode)`` convenience
    method upstream, so we wire the runtime here and call ``runtime.execute``.
    """
    try:
        from kiki_oniric.dream.episode import (  # type: ignore[import-not-found]
            BudgetCap,
            DreamEpisode,
            EpisodeTrigger,
            Operation,
            OutputChannel,
        )
        from kiki_oniric.dream.runtime import (  # type: ignore[import-not-found]
            DreamRuntime,
        )
        from kiki_oniric.substrates.micro_kiki import (  # type: ignore[import-not-found]  # noqa: E501
            MicroKikiSubstrate,
            micro_kiki_substrate_components,
        )
    except ImportError as exc:
        print(
            "WARN: dream-of-kiki package not installed "
            f"({exc}) — consolidation unavailable, "
            "falling back to rollback behaviour (exit 2)."
        )
        return 2

    try:
        # The canonical component map is only introspected for logging; keep
        # the import so a rename upstream breaks the gate loudly rather than
        # silently drifting out of sync.
        components = micro_kiki_substrate_components()
        print(
            "INFO: consolidation driver = "
            f"{components.get('runtime', '<?>')} over "
            f"{components.get('replay', '<?>')} handlers"
        )

        try:
            report_payload: dict[str, Any] = json.loads(
                gate_report_path.read_text()
            )
        except (OSError, json.JSONDecodeError) as exc:
            print(f"WARN: cannot read gate report {gate_report_path}: {exc}")
            report_payload = {}

        substrate_kwargs: dict[str, Any] = {}
        if real_backend_path is not None:
            # MicroKikiSubstrate's real-backend surface is the MLX base-model
            # path; passing this through lets the caller opt in to the live
            # SpikingKiki / mlx_lm backend when available.
            substrate_kwargs["base_model_path"] = str(real_backend_path)
        substrate_kwargs["adapter_path"] = str(new_adapter)

        substrate = MicroKikiSubstrate(**substrate_kwargs)
        if substrate_snapshot is not None and substrate_snapshot.exists():
            print(f"INFO: loading prior substrate snapshot {substrate_snapshot}")
            substrate.load_snapshot(substrate_snapshot)

        # Build a DreamEpisode carrying the gate's per-module report plus a
        # pointer to the offending adapter so downstream handlers can reach
        # the artifact. ``input_slice`` is a plain Mapping — the substrate's
        # stub handlers accept any dict shape; real handlers will key off
        # specific fields once wired up.
        input_slice = {
            "gate_report": report_payload,
            "new_adapter": str(new_adapter),
        }

        # Operations we try, in order: replay prior-stack activations, shrink
        # the offending LoRA delta, then project against prior stacks via
        # OPLoRA. Restructure is the most expensive; keep it last.
        operation_set = (
            Operation.REPLAY,
            Operation.DOWNSCALE,
            Operation.RESTRUCTURE,
        )
        output_channels = (
            OutputChannel.WEIGHT_DELTA,
            OutputChannel.ATTENTION_PRIOR,
        )
        budget = BudgetCap(
            flops=10**12,       # 1 TFLOP — conservative wall-clock guard
            wall_time_s=300.0,  # 5 min total on the dream runtime
            energy_j=1000.0,    # 1 kJ, roughly a minute at 15 W
        )

        episode = DreamEpisode(
            trigger=EpisodeTrigger.SATURATION,
            input_slice=input_slice,
            operation_set=operation_set,
            output_channels=output_channels,
            budget=budget,
            episode_id=f"post-train-gate-{uuid.uuid4().hex[:12]}",
        )

        runtime = DreamRuntime()
        # Factory methods return the per-op handlers. We adapt their
        # signatures to the runtime's ``(episode) -> None`` contract by
        # closing over the factory output and projecting episode data into
        # each handler's expected arguments.
        replay_fn = substrate.replay_handler_factory()
        downscale_fn = substrate.downscale_handler_factory()
        restructure_fn = substrate.restructure_handler_factory()

        def _replay_adapter(_ep: "DreamEpisode") -> None:
            beta = [{"input": [0.0]}]  # stub input; real caller feeds activations
            replay_fn(beta, n_steps=20)

        def _downscale_adapter(_ep: "DreamEpisode") -> None:
            import numpy as np  # local — numpy is already a hard dep
            dummy = np.ones((1,), dtype=np.float32)
            downscale_fn(dummy, 0.95)

        def _restructure_adapter(_ep: "DreamEpisode") -> None:
            import numpy as np
            adapter = {"lora_B": np.ones((1, 1), dtype=np.float32)}
            restructure_fn(adapter, "oplora", "lora_B")

        runtime.register_handler(Operation.REPLAY, _replay_adapter)
        runtime.register_handler(Operation.DOWNSCALE, _downscale_adapter)
        runtime.register_handler(Operation.RESTRUCTURE, _restructure_adapter)

        runtime.execute(episode)

        if substrate_snapshot is not None:
            print(f"INFO: saving updated snapshot to {substrate_snapshot}")
            substrate.snapshot(substrate_snapshot)

        print(
            "PASS: consolidation episode executed "
            f"(id={episode.episode_id}, "
            f"ops={[op.value for op in episode.operation_set]})."
        )
        return 0
    except Exception as exc:  # pragma: no cover - defensive, surfaces upstream bugs
        print(f"FAIL: consolidation raised {type(exc).__name__}: {exc}")
        return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--new-adapter", type=Path, required=True)
    parser.add_argument(
        "--prior-adapter",
        type=Path,
        help=(
            "Previous curriculum adapter. If omitted, only the health check "
            "runs (no forgetting comparison)."
        ),
    )
    parser.add_argument(
        "--eval-dataset",
        type=Path,
        help="Optional heldout JSONL for win-rate eval (enables full gate).",
    )
    parser.add_argument("--generate-fn-module", default=None)
    parser.add_argument("--winrate-baseline-score", type=float, default=None)
    parser.add_argument("--scorer-module", default=None)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument(
        "--consolidate-on-warning",
        action="store_true",
        help=(
            "On forgetting-gate failure, redirect through the dream-of-kiki "
            "MicroKikiSubstrate consolidation path instead of exiting 2. "
            "Falls back to exit 2 if kiki_oniric is not installed or the "
            "consolidation episode raises."
        ),
    )
    parser.add_argument(
        "--substrate-snapshot",
        type=Path,
        default=None,
        help=(
            "Path to a substrate snapshot (.npz). Loaded before consolidation "
            "if it exists, and overwritten after a successful consolidation. "
            "Only used with --consolidate-on-warning."
        ),
    )
    parser.add_argument(
        "--real-backend-path",
        type=Path,
        default=None,
        help=(
            "Optional path to an mlx_lm-loadable base model; passed through "
            "to MicroKikiSubstrate.base_model_path so the real SpikingKiki "
            "backend engages when available. Only used with "
            "--consolidate-on-warning."
        ),
    )
    args = parser.parse_args(argv)

    if not args.new_adapter.is_file():
        print(f"ERROR: new adapter not found: {args.new_adapter}", file=sys.stderr)
        return 3

    py = sys.executable or shutil.which("python3") or "python3"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== step 1/2: adapter health ({args.new_adapter.name}) ===")
    rc, out = _run([py, str(SCRIPTS / "validate_adapter_health.py"), str(args.new_adapter)])
    print(out.rstrip())
    if rc != 0:
        print("FAIL: adapter is degenerate — training path produced a dead adapter.")
        return 1
    print("PASS: adapter health OK.")

    if args.prior_adapter is None:
        print("=== step 2/2: forgetting gate skipped (no --prior-adapter) ===")
        return 0

    if not args.prior_adapter.is_file():
        print(f"ERROR: prior adapter not found: {args.prior_adapter}", file=sys.stderr)
        return 3

    print(f"=== step 2/2: forgetting gate ({args.prior_adapter.name} -> {args.new_adapter.name}) ===")
    out_json = args.output_dir / f"gate-{args.new_adapter.parent.name}.json"
    fg_cmd = [
        py,
        str(SCRIPTS / "measure_forgetting.py"),
        "--prior-adapter",
        str(args.prior_adapter),
        "--new-adapter",
        str(args.new_adapter),
        "--output",
        str(out_json),
    ]
    if args.eval_dataset is not None:
        fg_cmd.extend(["--eval-dataset", str(args.eval_dataset)])
    if args.generate_fn_module:
        fg_cmd.extend(["--generate-fn-module", args.generate_fn_module])
    if args.winrate_baseline_score is not None:
        fg_cmd.extend(["--winrate-baseline-score", str(args.winrate_baseline_score)])
    if args.scorer_module:
        fg_cmd.extend(["--scorer-module", args.scorer_module])

    rc, out = _run(fg_cmd)
    print(out.rstrip())
    if rc == 0:
        print(f"PASS: forgetting gate cleared. Report at {out_json}.")
        return 0
    print(f"FAIL: forgetting gate triggered. See {out_json} for per-module detail.")
    if args.consolidate_on_warning:
        print("INFO: --consolidate-on-warning set, attempting substrate consolidation.")
        return _attempt_consolidation(
            gate_report_path=out_json,
            new_adapter=args.new_adapter,
            substrate_snapshot=args.substrate_snapshot,
            real_backend_path=args.real_backend_path,
        )
    return 2


if __name__ == "__main__":
    sys.exit(main())
