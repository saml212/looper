Implement a single, scoped task.

Usage: /implement <description of what to implement>

Task: $ARGUMENTS

Steps:

**1. Context Loading**
- Read `CLAUDE.md` for rules and constraints.
- Read relevant docs and existing code.
- If code exists on the `v1` branch that's relevant, check it: `git show v1:<path>`.

**2. Plan**
- State what you will implement (one specific thing).
- State what you will NOT touch.
- Identify the tests you will write.
- If the task is ambiguous, STOP and ask.

**3. Implement (Delegate to Subagents with Review Loop)**
When working on a multi-step task, delegate to subagents. For EVERY subagent:

  a) **Start** the subagent — give it a scoped task with clear boundaries
  b) **Review** the subagent's output — check scope, quality, minimal code
  c) **Resume** the subagent with refinement instructions:
     - Reduce code as much as possible
     - Simplify tests (no redundant assertions, use fixtures)
     - Remove dead code and unused imports
     - Rerun all tests to verify
     - Do an end-to-end test: mock or real data in → verify expected output
  d) **Final review** — only accept when code is minimal and clean

This start→review→resume loop is how we maintain quality without bloat.

**4. Verify**
- Run `ruff format looper/ tests/` and `ruff check looper/ tests/`
- Run `pytest tests/ -v`
- If there's a CLI or script, do an end-to-end smoke test with real or mock data

**5. Codify**
Before finishing, ask:
- Did I discover a pattern, gotcha, or unexpected behavior?
- Did I make a design decision that should be recorded?
- Did I learn something about MLX, Ollama, SWE-Bench, or a library?

If YES: update the relevant doc (LEARNINGS.md, docs/architecture.md, CLAUDE.md, etc.)

**6. Summary**
- Present what was implemented, tests written, learnings codified
- Show diff summary
- Wait for human review
