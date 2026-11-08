# Changelog

## [0.4.0] — 2026-11-08

Covers `v0.3.0..main` (223 commits, 29 merged pull requests; most commits
landed directly on `main`, so entries without a PR number cite commit
subjects). Scientific statements below are quoted from files in this
repository; follow the links for scope, limits and caveats. Nothing here
is a new result unless it links to a committed artifact.

### Added

- Full pipeline server: 7-stage `full_pipeline_server` (Aeon recall,
  adapter select/apply, K-candidate inference, Negotiator, AntiBias,
  Aeon write), `kiki-*` OpenAI alias table, chat endpoint, 429 queue
  guard, per-stage Prometheus metrics, launchd plist (#4).
- OpenAI-compatible serving layer: schemas, MLX runtime abstraction,
  FastAPI app, tool calling and JSON mode, SSE streaming and cancel,
  KV-status and context admission, `/v1/route` endpoint.
- Serving performance: LRU prompt cache with 8-bit KV quantisation,
  zero-swap adapter via `unpatch_all_lora`, adapter preload, eager MoE
  expert materialisation, batching of K candidates in one pass, MLX
  inference offloaded to a thread, skip adapter for base-only domains,
  domain-conditional cognitive gating.
- Observability: Prometheus histogram for adapter swap; first adapter
  load observed as `method=initial`.
- Routing: `spice-sim` added as 35th niche domain; router V4 retrains
  (35 outputs, clean data, 30 epochs); RLM recursive domain router;
  few-shot prompts for the RLM router; router V5 retrain chain script.
- Brainstacks V4 null-space projector, gradient-projection loop and V4
  training CLI.
- Cognitive layer: real `AeonPalace` (MiniLM `embed_fn`), real Negotiator
  and AntiBias, real MetaRouter v4 adapter; RLM recursive negotiator PoC.
- dream-of-kiki substrate prototype and `run_dream_cycle.py`; phase-4
  conformance run; `--consolidate-on-warning` gate flag.
- SpikingKiki: LoRA-to-SNN adapter prototype, `linear_attn` and fused
  MoE support for Qwen3.6, parallel block range CLI, phase E validation
  sanity checks.
- Evaluation tooling: V4 PPL eval for all 35 domains, SOTA bench
  runners, MBPP N=500 comparison script, router ablation scripts,
  domain-coherence validation script, HF model-card sync script.
- NH-81 measurement protocol and analysis code (`experiments/nh81_joules`)
  (#18); GB300 host delta and launcher (`experiments/nh81_gb300`) and
  GB300 35B LoRA demo harness (`experiments/demo_35b_gb300`).
- Deployment: consolidated launchd with templated install; YAML and
  actionlint pre-commit hooks; nested `CLAUDE.md` guidance files.
- Paper drafts (`paper/`, `docs/paper-a/`): arXiv draft, 2026 related
  work and limits, literature actions of 2026-10-23, Makefile for the
  LaTeX build (#7, #9).
- `docs/ERRATA.md`: dated errata log (see Docs).
- `CONTRIBUTING.md` and GitHub issue / pull request templates (#36).
- Declarations: AI-assistance note and CONTRIBUTORS pointer (#21, #22),
  author ORCID in `CITATION.cff` (#20), CC BY 4.0 for docs (#29).

### Changed

- Router: documented top-1 / top-3 now the artifact-backed 46.4 % /
  69.6 % (`results/router-v4-eval.json`); see E-2026-10-25-03 (#16).
- README hosting note and repaired links (#15, #5).
- `CITATION.cff` licence aligned with `LICENSE` (#23); base model named
  as Qwen3.6-35B-A3B (#27); docs licence and affiliation aligned (#29).
- Docs: "Gated DeltaNet, not GLA" in scripts; LoRA scale confirmed from
  adapter configs; CLAUDE.md aligned to the real LoRA convention.
- Claims scope corrected (`f9453c3`, "fix(claims): correct
  forgetting/alpha/PPL scope"): alpha = 2 x rank; forgetting sweep covers
  5 adapters / 20 pairs; PPL -3.44 is the 6-domain mean and -7.30 the
  35-domain mean; adapter count 35 v4 stacks + 6 v2 retrains = 41
  artefacts.
- `cache_limit` 32 to 96 GB in the V4 training script.

### Fixed

- Serving: correct MLX prompt-cache reuse (#6). On a cache hit the
  prompt was fed twice; reused candidates attended to a duplicated
  context.
- Routing: joint MI estimate capped at H(Y) (`d25e701`, E-2026-10-25-02).
  The previous `estimate_mi_bits` summed per-feature marginal MIs.
- CI: install `httpx` in the forgetting-tests job (#8).
- Serving: map merged domains to adapters; `strip_thinking` default true;
  lock threshold mutations in `/v1/route`; force CPU tensor in MetaRouter
  v4 adapter.
- Training: use `linear_to_lora_layers` (mlx_lm 0.31), `tree_flatten` for
  parameter save, remove `LoRASwitchLinear` before re-apply; PPL eval
  uses manual tokenize + loss; filter python training stubs.
- SpikingKiki: `Parameter` vs `Linear` and fused MoE experts for
  Qwen3.6.
- NH-81 code: fail closed on failed GPU reads (`fdd0e37`); Codex review
  findings on the GB300 code and demo harness (`2555d89`, `91ee680`,
  `849e298`).
- Deploy: bridge consumes kiki-flow `MANIFEST.json`; safetensors
  untracked from LFS; `loguru` and `hdbscan` declared; launchd plist
  `--extra embeddings`.

### Docs

Errata (full text in [`docs/ERRATA.md`](docs/ERRATA.md); entries quoted,
not summarised):

- E-2026-10-25-01, SNN energy "0.032 mJ/token" and "91.6x" (issue #10):
  "Both figures removed everywhere listed above and replaced by: 'not
  measured on the target system; external 2026 measurements unfavourable;
  J/token to be measured (NH-81)'."
  Update 2026-10-28: "The energy argument is withdrawn for GPU/M1-class
  hardware (`docs/results/2026-10-25-nh81-joules.md`); NVML replication
  on the RTX 4090 pending." Addendum 2026-10-31 (#24, #25): "The sentence
  'No ANN->SNN conversion of the 35B model was ever run' is wrong. A
  partial conversion of Qwen3.6-35B-A3B was run on 2026-04-21 [...] It
  was never evaluated".
- E-2026-10-25-02, C5 information bound "0.911" (issue #11, MK-23):
  "The 0.911 figure is withdrawn in every document listed above. MK-23
  is requalified from 'confirmed' to **refuted**: the derivation is
  invalid and no valid bound has been established." Update 2026-11-08
  (#39, #40): `results/c5-info-bound.json` and
  `docs/paper-a/c5-bound-figure.pdf` regenerated with the joint k-NN
  estimator; `bound_vqc_4qubit` 0.9110 -> 0.6356 (joint MI 1.167 bits on
  PCA-4, all values below H(Y) = 3.32 bits). PI decision: MK-23 stays
  **refuted**; the corrected value is a different statement (plain
  Fano/DPI bound, the Holevo cap never binds), not a confirmation of the
  withdrawn 0.911.
- E-2026-10-25-03, router accuracy "70.7 % top-1 / 89.5 % top-3"
  (issue #12, MK-01): "The 70.7 % / 89.5 % figure is withdrawn: it is not
  reproducible from the repository."
- Paper-a citation corrections: Caro et al. 2022 bound stated as in
  Theorem 1, relevance paragraph corrected (#35), "negligible" softened,
  Du et al. theorem citations corrected, MoLA reference [12].
- Bibliography fixes: identifiers and entry types, misattributed
  citations, TD-MPC2 id and Aeon title, OpenReview forum id for LeCun
  2022, Rome group cited where inventoried (#7, #34).
- NH-81 result note (`docs/results/2026-10-25-nh81-joules.md`, #18).
  Verdict as written there: "Withdraw the energy argument definitively
  for GPU/M1-class hardware. Primary block (W1 generation, fused spiking
  path, GPU rail): R-hat = 0.91, 95 % CI [0.83, 1.01], 12/12 valid pairs,
  iso-quality met." Limits as written: "One pair at 26.5 M parameters,
  one seed, fp32, one GPU (M1 Max), dense kernels only. The result is
  about GPU/M1-class execution, not about neuromorphic silicon."
- Related-work additions: SuffixReplay cited for the prefix cache
  (#26, #28); DPO alignment strategy note; Brainstacks V4 plan;
  publication readiness plan.
- Results committed as artifacts (titles verbatim, see `results/`):
  "v3 <domain> gate" for 14 domains (2026-04-20/21); "eval: null-space
  ablation results (negative)"; "eval: cognitive A/B results (14-16
  split)"; "eval: router ablation results"; PPL evaluation over 35 domains
  (figure corrected in `f9453c3`: mean -7.30, `avg_delta` -7.303 in
  `results/v4-ppl-35domains/ppl-all-domains.json`, single-seed, no CI;
  see Changed); "chore(results): GSM8K sweep finds reasoning
  toxic"; "chore(results): SOTA sweep small-N directional". Read the
  committed JSON and `docs/ERRATA.md` (claims-scope correction above)
  before citing any of them.

### Preregistration

No tag listed here is created or modified by this release.

- NH-81 joules per token, spiking vs dense: frozen under tag
  `prereg-nh81-20261025` (`docs/prereg/2026-10-25-nh81-joules-per-token.md`,
  #18). Post-measurement amendment 1, dated 2026-10-28:
  `docs/prereg/2026-10-28-nh81-amendment-1.md`.
- NH-81 on GB300, descriptive (L0): `docs/prereg/nh81-gb300/` (#30);
  pre-data amendment (#32); freeze record (#33). Tags
  `nh81-gb300-freeze-20261031` (superseded, kept) and
  `nh81-gb300-freeze-20261031b` (current). Per the freeze record, "Any
  GB300 datum must be produced after 2026-10-31T23:09:53+01:00".
- GB300 35B LoRA demo (L0): `docs/prereg/demo-35b-lora-gb300/` (#31);
  tags `demo-35b-gb300-freeze-20261031` (superseded, kept) and
  `demo-35b-gb300-freeze-20261031b` (current). No GB300 result is
  included in this release.

## [Unreleased] — bridge-surrogate-v0.3 (pending)

- Updates bridge surrogate from v0.2-d128 (synthetic JKO pairs only) to
  v0.3 (text-conditioned via QueryConditionedF + JEPA pre-training).
- DEPENDS ON: kiki-flow-research artifacts/v0.3.safetensors (produced by
  T22/T23 of the workshop sprint, not yet merged).
