# Ralph Loop Task — v0.2 quantum-inspired

You are executing ONE iteration of the v0.2 ralph loop. Complete ONE story, then exit.

## Your Task

1. Read `.ralph-v0.2/prd.json` to find the next incomplete story (passes: false, lowest priority)
2. Implement ONLY that story per `.claude/plans/micro-kiki-v0.2-quantum-inspired.md`
3. Run quality gates
4. Commit your changes
5. Update progress
6. Exit

## Quality Gates

Run these commands before committing. ALL must pass:

- `uv run pytest` (once tests exist for the story)

Light checks (best-effort, not blocking if tool not installed):

- `uv run python -c "import src"` (package imports cleanly)

If any gate fails, fix the issue before committing.

## Story Completion Protocol

When you complete a story:

1. **Check for changes and commit if needed:**
   ```bash
   git status --porcelain
   ```

   If there ARE changes:
   - Stage changes: `git add -A`
   - Commit with a conventional commit message describing your implementation
   - Subject <= 50 chars, no `Co-Authored-By` trailer (hook rejects it)
   - If commit is rejected, read the error, fix the message, retry (max 3 attempts)
   - After 3 failures, log the error to `.ralph-v0.2/progress.txt` and exit

   If there are NO changes, proceed directly to step 2.

2. **Update prd.json:**
   - Set `passes: true` for the completed story in `.ralph-v0.2/prd.json`

3. **Append to `.ralph-v0.2/progress.txt`:**
   ```
   [{timestamp}] Completed: {story-id} - {story-title}
   ```

4. **Exit immediately** - Do not start another story

## Guardrails

See `.ralph-v0.2/guardrails.md` for v0.2-specific constraints (no QPU calls, classical only, bond-dim limits).

## Important

- Complete exactly ONE story per iteration
- Do not skip quality gates
- Do not modify stories you are not implementing
- Do not touch `.ralph/` (that's the v0.1 loop, running on `main`)
- Only ever work on branch `quantum-inspired`
- If blocked, document in `.ralph-v0.2/progress.txt` and exit
- Trust the loop script to handle the next iteration
