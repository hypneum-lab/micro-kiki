# Ralph Loop Guardrails — v0.2 quantum-inspired

Constraints and boundaries for the v0.2 ralph loop execution.

## Scope Boundaries

This loop implements the plan at: `.claude/plans/micro-kiki-v0.2-quantum-inspired.md`

### In Scope
- Stories defined in `.ralph-v0.2/prd.json`
- Files mentioned in the v0.2 plan
- Classical tensor-network libraries (quimb, opt_einsum, tensornetwork)
- Quality gates listed in `.ralph-v0.2/CLAUDE.md`

### Out of Scope
- Features not in the v0.2 plan
- Any modification to `.ralph/` or the v0.1 plan
- Changes to main branch (v0.2 lives on `quantum-inspired` only)
- Refactoring unrelated code

## Quantum / classical boundary

**This loop is 100% classical execution.**

- NO QPU calls (no IonQ, no IBM Quantum, no Rigetti, no AWS Braket)
- NO `qiskit_ibm_runtime`, `braket.aws`, `cirq.google`, `pennylane.plugins.qulacs` with a QPU backend
- Local simulators (`qiskit-aer`, `pennylane-lightning`) are technically allowed but outside v0.2 scope — if a story seems to need them, stop and flag for QPU-only branch in `micro-kiki-quantum`
- Credentials for QPU providers must NEVER appear in this repo

## Bond-dimension limits

- CompactifAI: chi range {8, 16, 32, 64}. Never above 64 (defeats compression) or below 8 (accuracy cliff)
- QTHA adapter: bond dim <= 16
- TN router: MPS bond dim <= 8

## Quality Requirements

All changes must:
1. Pass defined quality gates
2. Include appropriate tests
3. Follow existing code patterns from v0.1
4. Not break existing v0.1 functionality on main (verify via isolated import)

## Commit Standards

- One commit per story
- Conventional commit format (see parent `CLAUDE.md` at repo root)
- Subject <= 50 chars (pre-commit hook enforces)
- No `Co-Authored-By` trailer (hook rejects)
- Branch: `quantum-inspired` only

## Blocking Conditions

Stop and document in `.ralph-v0.2/progress.txt` if:
- Quality gates fail after 3 attempts
- Story requires clarification not in plan
- Tensor-network library install fails
- Compact base accuracy drop > 10 pp at chi=32 (regression)

## Recovery

If the loop fails:
1. Check `.ralph-v0.2/progress.txt` for last successful story
2. Check git log for committed work
3. Review `.ralph-v0.2/prd.json` for story states
4. Resume with: `uv run .ralph-v0.2/loop.py`
